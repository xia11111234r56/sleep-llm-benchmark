"""检查 API key 是否配置正确，并列出账号下真实可用的模型名。

主要目的：解决「不知道版本快照名该填什么」的问题——直接问 API，不要猜。

用法：
    python check_api.py              # 只检查配置 + 列出模型
    python check_api.py --test-call  # 额外发一次真实调用，验证 key 能跑通
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_client import PROVIDERS, resolve_api_key  # noqa: E402

from openai import OpenAI  # noqa: E402


def mask(key: str) -> str:
    """只显示首尾，避免 key 被完整打印到屏幕/日志里。"""
    if not key:
        return "(空)"
    return key[:7] + "…" + key[-4:] if len(key) > 14 else "***"


def snapshot_tag(name: str) -> str:
    """粗判模型名里有没有可锁定的版本标识。

    注意这是**启发式**，会误判。真正的判据是厂商文档，别只信这个标记。
    能识别的形态：
      xxx-2026-04-02   标准日期快照（最可靠）
      xxx-0731         月日后缀快照（次可靠，需去控制台确认）
    """
    import re
    if re.search(r"-\d{4}-\d{2}-\d{2}$", name):
        return " [日期快照]"
    if re.search(r"-\d{4}$", name):
        return " [疑似日期后缀]"
    if re.search(r"v\d+\.\d+", name):
        return " [带小版本号]"
    return " [无版本标识]"


def main():
    test_call = "--test-call" in sys.argv
    no_list = "--no-list" in sys.argv  # 百炼有 260+ 个模型，复检时用这个跳过清单

    for provider, cfg in PROVIDERS.items():
        env = cfg["api_key_env"]
        print("=" * 68)
        print(f"【{provider}】环境变量 {env}")
        print(f"  默认模型名：{cfg['model']}")
        print(f"  base_url  ：{cfg['base_url']}")

        key, source = resolve_api_key(env)
        if not key:
            print(f"  ✗ 未设置。在 PowerShell 里执行：")
            print(f'      [Environment]::SetEnvironmentVariable("{env}", "sk-你的key", "User")')
            print(f"    然后重开一个窗口（或重启编辑器）让它生效")
            continue

        print(f"  ✓ 已设置：{mask(key)}（长度 {len(key)}）来源：{source}")
        if source == "Windows 用户环境变量":
            print(f"    ⚠️ 本进程的环境块是旧的，已从注册表兜底读取。")
            print(f"       长期使用仍建议重启 VS Code，让所有脚本都能直接读到。")

        client = OpenAI(api_key=key, base_url=cfg["base_url"], timeout=30)

        # 1) 列出可用模型
        if no_list:
            print("  （已跳过模型清单，--no-list）")
        else:
            print("  → 查询可用模型 …")
            try:
                models = sorted(m.id for m in client.models.list())
                print(f"    查到 {len(models)} 个模型：")
                for m in models:
                    mark = "  ← 你配置的" if m == cfg["model"] else ""
                    print(f"      {m}{snapshot_tag(m)}{mark}")
                if cfg["model"] not in models:
                    print(f"    ⚠️ 你配置的 {cfg['model']} 不在列表里，跑之前要改")
            except Exception as e:
                print(f"    ! 列模型失败（{type(e).__name__}: {str(e)[:150]}）")
                print("      有些兼容接口不支持 models.list，可以跳过这步直接 --test-call")

        # 2) 真实调用
        if test_call:
            print("  → 发一次测试调用 …")
            try:
                r = client.chat.completions.create(
                    model=cfg["model"],
                    messages=[{"role": "user", "content": "回复两个字：收到"}],
                    temperature=0, max_tokens=16,
                )
                print(f"    ✓ 调用成功")
                print(f"      请求的模型：{cfg['model']}")
                print(f"      返回的模型：{getattr(r, 'model', '(未返回)')}")
                print(f"      回复内容  ：{r.choices[0].message.content!r}")
                print(f"      token 用量：{r.usage.prompt_tokens} in / "
                      f"{r.usage.completion_tokens} out")
                print(f"      ↑ 这个『返回的模型』要抄进报告，作为版本证据")
            except Exception as e:
                print(f"    ✗ 调用失败（{type(e).__name__}: {str(e)[:200]}）")

        print()

    print("=" * 68)
    print("提示：模型名要选带日期的快照（如 xxx-2026-04-02），不要用浮动别名。")
    print("     浮动别名会被厂商静默升级，你的结果就无法复现了。")


if __name__ == "__main__":
    main()
