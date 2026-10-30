"""A–D 四组提示词实验：同一批测试集，跑同一套指标。

四组唯一的差异是提示词策略，任务定义、标签口径、输出 schema 全部一致，
否则比出来的差异分不清是策略造成的还是格式造成的。

    A = v1_zeroshot    基线
    B = v2_fewshot     给示例
    C = v3_structured  拆维度分步问
    D = v4_grounded    强制引用原文 + 拒答

用法：
    $env:DASHSCOPE_API_KEY = "sk-..."
    python run_prompt_exp.py --provider qwen --limit 5      # 冒烟测试
    python run_prompt_exp.py --provider qwen                # 全量跑

支持断点续跑：结果逐条 append 到 jsonl，重跑会跳过已完成的 (group, id)。
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_client import PROVIDERS, LLMClient, extract_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

GROUPS = {
    "A": "v1_zeroshot.txt",
    "B": "v2_fewshot.txt",
    "C": "v3_structured.txt",
    "D": "v4_grounded.txt",
}

ATTR_LABELS = {"焦虑型", "压力型", "生理型", "环境型", "其他"}
TEND_LABELS = {"有抑郁倾向", "无抑郁倾向", "无法判断"}

N_SHOTS = 5  # B 组给几个示例


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def format_examples(shots: list[dict]) -> str:
    """把已标注的少样本示例格式化成提示词片段。

    ⚠️ 示例必须来自训练集，绝不能来自测试集，否则就是数据泄漏。
    ⚠️ 示例分布会影响模型倾向：示例里没有哪一类，模型就很少判哪一类。
       当前 7 条覆盖 环境/生理/压力/其他，无焦虑型 —— 报告里要说明这个局限。
    """
    blocks = []
    for n, ex in enumerate(shots, 1):
        ans = {
            "attribution": ex.get("attribution", ""),
            "tendency": ex.get("tendency", ""),
            "evidence": ex.get("evidence", "") or "N/A",
        }
        blocks.append(
            f"示例 {n}\n帖子：{ex['text']}\n标注：{json.dumps(ans, ensure_ascii=False)}"
        )
    return "\n\n".join(blocks)


def render(template: str, post: str, examples: str) -> str:
    # 用 replace 而不是 .format：提示词里含 JSON 花括号，.format 会炸
    return template.replace("{{POST}}", post).replace("{{EXAMPLES}}", examples)


def validate(obj: dict) -> tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "不是 JSON 对象"
    a, t = obj.get("attribution"), obj.get("tendency")
    if a not in ATTR_LABELS:
        return False, f"归因标签非法：{a!r}"
    if t not in TEND_LABELS:
        return False, f"倾向标签非法：{t!r}"
    if not isinstance(obj.get("evidence"), str):
        return False, "evidence 缺失或非字符串"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="qwen", choices=sorted(PROVIDERS),
                    help="qwen / deepseek 走百炼（共用 DASHSCOPE_API_KEY）；"
                         "deepseek-official 走 DeepSeek 官方")
    ap.add_argument("--groups", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--model", default=None, help="覆盖默认模型（锁版本快照用）")
    ap.add_argument("--testset", default="data/splits/test_gold.jsonl")
    ap.add_argument("--shots", default="data/splits/shots.jsonl",
                    help="B 组的少样本示例（必须有标签，且来自训练集）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条，冒烟测试用")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out-responses", default="outputs/raw_prompt_responses.jsonl",
                    help="冒烟测试时改到别处，别污染正式结果")
    args = ap.parse_args()

    test_path = ROOT / args.testset
    test = load_jsonl(test_path)
    if not test:
        raise SystemExit(
            f"测试集为空或不存在：{test_path}\n"
            f"提示：标注（阶段 2）完成后才会有 test_gold.jsonl。\n"
            f"冒烟测试可以先用 data/clean/tieba_clean.jsonl 临时顶替。"
        )
    if args.limit:
        test = test[:args.limit]

    shots = load_jsonl(ROOT / args.shots)
    examples = format_examples(shots) if shots else ""
    if "B" in args.groups and not examples:
        print(f"⚠️  B 组需要少样本示例，{ROOT / args.shots} 不存在或为空，B 组将跳过")
        args.groups = [g for g in args.groups if g != "B"]

    out_path = ROOT / args.out_responses
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 断点续跑键必须含 provider：否则换家族重跑时会把已有的 (group,id) 误判为已完成，
    # 静默跳过全部条目，产出空的第二组结果。
    done = {(r["provider"], r["group"], r["id"]) for r in load_jsonl(out_path)}
    print(f"测试集 {len(test)} 条 × {len(args.groups)} 组，已有 {len(done)} 条结果")

    client = LLMClient(args.provider, model=args.model, max_tokens=args.max_tokens)
    print(f"模型：{client.model}（provider={args.provider}，temperature=0）\n")

    total = len(test) * len(args.groups)
    n = 0
    for group in args.groups:
        template = (ROOT / "prompts" / GROUPS[group]).read_text(encoding="utf-8")
        for item in test:
            n += 1
            if (args.provider, group, item["id"]) in done:
                continue

            prompt = render(template, item["text"], examples)
            res = client.chat(prompt)
            obj = extract_json(res["text"]) if res["text"] else None
            ok, why = validate(obj) if obj else (False, res["error"] or "无法解析出 JSON")

            rec = {
                "group": group,
                "id": item["id"],
                "provider": args.provider,
                "model_requested": client.model,
                "model_returned": res["model_returned"],
                "attribution": (obj or {}).get("attribution"),
                "tendency": (obj or {}).get("tendency"),
                "evidence": (obj or {}).get("evidence"),
                "valid": ok,
                "invalid_reason": why,
                "raw_response": res["text"],
                "prompt_tokens": res["prompt_tokens"],
                "completion_tokens": res["completion_tokens"],
                "latency_s": res["latency_s"],
                "ran_at": datetime.now().isoformat(timespec="seconds"),
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            flag = "OK " if ok else "BAD"
            print(f"[{n}/{total}] {group} {item['id']} {flag} "
                  f"{(obj or {}).get('attribution', '-')}/{(obj or {}).get('tendency', '-')}"
                  + ("" if ok else f"  ({why})"), flush=True)

    rows = load_jsonl(out_path)
    print(f"\n完成，共 {len(rows)} 条结果")
    print(" 分组有效数:", dict(Counter(r["group"] for r in rows if r["valid"])))
    print(" 分组无效数:", dict(Counter(r["group"] for r in rows if not r["valid"])))
    print(f"\n输出：{out_path}")
    print("下一步：python eval.py  生成指标表")


if __name__ == "__main__":
    main()
