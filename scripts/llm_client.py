"""统一的大模型调用客户端（OpenAI 兼容接口）。

两个硬性设计约束，都是为了结果可复现：

1. **锁版本快照**。不要用 `qwen-max`、`deepseek-chat` 这类浮动别名——厂商会静默升级，
   你 9 月跑的结果和 12 月复现的可能不是同一个模型。用带日期的快照名，并把 API 返回的
   `model` 字段记进结果表。

2. **temperature=0 并记录全部元数据**（token 数、耗时、返回的模型串）。报告里每个数字
   都要能追溯到某次具体调用。
"""

import os
import sys
import time

from openai import OpenAI

# ---- 供应商与固定版本 ----------------------------------------------------
# ⚠️ 下面的快照名请到对应控制台核对后再跑。写错会直接报错，这是好事——
#    总比用浮动别名跑完才发现版本漂移了强。

DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ⚠️ 关于「为什么都用非推理模型」——这是实验设计决定的，不是省钱：
#
#   E/F 组的微调基座是 Qwen2.5-1.5B，一个不会推理的小模型。
#   如果 A–D 用会思考上千 token 的推理模型（Qwen3.7/3.8-max 默认、DeepSeek v4 全部），
#   那 RQ2「微调能否超过大模型提示词工程」比的就成了「推理能力强弱」，
#   而不是「提示词方法优劣」——对照组直接失效。
#
#   实测（同一帖子、同一提示词、temperature=0）：
#     qwen3.7-max 开思考      36.3s / 1798 tok
#     qwen3.8-max 关思考       2.5s /   72 tok
#     deepseek-v4-flash-0731  11.1s /  891 tok（且 max_tokens 给小了会吐空字符串）
#     deepseek-v3.2            1.8s /   75 tok（天然非推理）
#
#   所以：Qwen 显式关思考，DeepSeek 选唯一非推理的 v3.2。

PROVIDERS = {
    # ---- 百炼托管：共用一个 key，消除「平台差异」这个混淆变量 ----
    "qwen": {
        "base_url": DASHSCOPE_URL,
        "api_key_env": "DASHSCOPE_API_KEY",
        "model": "qwen3.8-max-0902",
        "extra_body": {"enable_thinking": False},
        "note": "最新代 Qwen-Max，显式关闭思考模式以对齐非推理的微调基座。",
    },
    "deepseek": {
        "base_url": DASHSCOPE_URL,
        "api_key_env": "DASHSCOPE_API_KEY",
        "model": "deepseek-v3.2",
        "extra_body": None,
        "note": "DeepSeek 家族里唯一不支持/不需要推理模式的型号，与 qwen 共用 key。",
    },
    # ---- 第三方家族：仅用于「预标注」，不参与实验组 ----
    #
    # 预标注模型绝不能是 A–D 里要评测的模型。如果人工标注时看着 Qwen 的建议去定 gold，
    # gold 就会被 Qwen 锚定，再拿 Qwen 去考这套 gold 就是循环论证、准确率虚高。
    # 用第三方家族，锚定偏差就成了所有实验组共同承受的常数。
    "glm": {
        "base_url": DASHSCOPE_URL,
        "api_key_env": "DASHSCOPE_API_KEY",
        "model": "glm-5.3",
        "extra_body": None,
        "note": "仅用于生成标注辅助建议，不参与 A–F 任何实验组。",
    },
    # ---- DeepSeek 官方 API：备选 ----
    "deepseek-official": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model": "deepseek-flash",
        "extra_body": None,
        "note": "官方接口只提供浮动别名，无法锁版本。"
                "⚠️ 实测 deepseek-chat 会被静默重定向到 deepseek-flash；"
                "deepseek-v4-pro 是推理模型，max_tokens 给小了会返回空字符串。"
                "要可复现请用百炼那条。",
    },
}


def resolve_api_key(env_name: str) -> tuple[str | None, str]:
    """取 API key：先看进程环境变量，再查 Windows 用户环境变量（注册表）。

    为什么要查注册表：Windows 上设置「永久」环境变量后，**已经在运行的程序**
    （VS Code、Claude Code 等）读不到新值——进程的环境块在启动那一刻就固定了。
    查注册表可以免去为这一件事重启整个编辑器。
    """
    v = os.environ.get(env_name)
    if v:
        return v, "进程环境变量"

    if sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                v, _ = winreg.QueryValueEx(k, env_name)
                if v:
                    return str(v), "Windows 用户环境变量"
        except (FileNotFoundError, OSError):
            pass

    return None, "未设置"


class LLMClient:
    def __init__(self, provider: str, model: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 1024,
                 max_retries: int = 4, timeout: float = 120.0):
        if provider not in PROVIDERS:
            raise ValueError(f"未知供应商 {provider}，可选：{list(PROVIDERS)}")
        cfg = PROVIDERS[provider]

        key, source = resolve_api_key(cfg["api_key_env"])
        if not key:
            raise RuntimeError(
                f"缺少 API key。请设置环境变量 {cfg['api_key_env']}，例如：\n"
                f'  $env:{cfg["api_key_env"]} = "sk-..."'
            )
        self.key_source = source

        self.provider = provider
        self.model = model or cfg["model"]
        self.extra_body = cfg.get("extra_body")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.client = OpenAI(api_key=key, base_url=cfg["base_url"], timeout=timeout)

    def chat(self, prompt: str, max_tokens: int | None = None) -> dict:
        """单轮调用。返回文本与全部计量元数据，失败会指数退避重试。

        推理模型常把 token 预算全花在隐藏的思考上，正文被截断甚至为空。
        调用方可以传 max_tokens 覆盖本次预算（例如解析失败后用双倍预算重试）。
        """
        budget = max_tokens or self.max_tokens
        last_err = None
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                kwargs = {}
                if self.extra_body:
                    kwargs["extra_body"] = self.extra_body
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=budget,
                    **kwargs,
                )
                latency = time.perf_counter() - t0
                u = resp.usage
                return {
                    "text": resp.choices[0].message.content or "",
                    # length = 预算被思考耗尽、正文被截断；这是推理模型最常见的静默失败
                    "finish_reason": resp.choices[0].finish_reason,
                    "model_returned": getattr(resp, "model", self.model),
                    "prompt_tokens": getattr(u, "prompt_tokens", None),
                    "completion_tokens": getattr(u, "completion_tokens", None),
                    "max_tokens_used": budget,
                    "latency_s": round(latency, 3),
                    "error": None,
                }
            except Exception as e:
                last_err = e
                wait = 2 ** attempt
                print(f"    ! 调用失败（{type(e).__name__}: {str(e)[:120]}），{wait}s 后重试",
                      flush=True)
                time.sleep(wait)

        return {
            "text": "", "finish_reason": None, "model_returned": self.model,
            "prompt_tokens": None, "completion_tokens": None,
            "max_tokens_used": budget, "latency_s": None,
            "error": f"{type(last_err).__name__}: {last_err}",
        }


def extract_json(text: str) -> dict | None:
    """从模型回复里抠出 JSON 对象。

    模型常会带 markdown 代码块、或在 JSON 前后加解释，所以不能直接 json.loads。
    从第一个 { 扫到与之配对的 }，跳过字符串内的花括号。
    """
    import json

    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    start = text.find("{")
    if start < 0:
        return None

    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None
