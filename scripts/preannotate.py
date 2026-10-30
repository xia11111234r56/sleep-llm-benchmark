"""生成人工标注的辅助建议（预标注）。

⚠️ 为什么默认用 GLM 而不是 Qwen / DeepSeek：

   预标注会「锚定」人工标注。如果标注者看着 Qwen 的建议定 gold label，
   再拿这套 gold 去评测 Qwen（A–D 组），就是循环论证——模型在考自己影响出来的答案。

   所以预标注模型必须是**不参与实验的第三方家族**。这样锚定偏差变成
   所有实验组共同承受的常数，而不是偏向某一组。

   这条要写进报告的「局限性」一节。

产出的文件只是「建议」，不是金标。人类可以随意推翻，且必须逐条确认——
标注手册里的判定规则才是唯一依据。

用法：
    python preannotate.py
    python preannotate.py --provider glm --limit 10    # 试跑
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_client import PROVIDERS, LLMClient, extract_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

ATTR_LABELS = {"焦虑型", "压力型", "生理型", "环境型", "其他"}
TEND_LABELS = {"有抑郁倾向", "无抑郁倾向", "无法判断"}


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="glm", choices=sorted(PROVIDERS),
                    help="必须是第三方家族，不要用 qwen/deepseek")
    ap.add_argument("--model", default=None)
    ap.add_argument("--testset", default="data/splits/test_pool.jsonl")
    ap.add_argument("--prompt", default="prompts/v1_zeroshot.txt")
    ap.add_argument("--out", default="data/splits/test_preannot.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    # GLM-5.3 是推理模型，实测单条最多花了 3325 个思考 token。
    # 预算给小了会静默返回被截断的 JSON，默认必须给足。
    ap.add_argument("--max-tokens", type=int, default=8192)
    args = ap.parse_args()

    if args.provider in ("qwen", "deepseek", "deepseek-official"):
        raise SystemExit(
            f"❌ 不能用 {args.provider} 做预标注——它是实验组要评测的模型，\n"
            f"   会让 gold label 被它锚定，评测结果失去意义。\n"
            f"   请用第三方家族：glm 等。"
        )

    test = load_jsonl(ROOT / args.testset)
    if not test:
        raise SystemExit(f"测试集为空：{ROOT / args.testset}\n先跑 make_splits.py")
    if args.limit:
        test = test[:args.limit]

    out_path = ROOT / args.out
    done = {r["id"] for r in load_jsonl(out_path)}

    template = (ROOT / args.prompt).read_text(encoding="utf-8")
    client = LLMClient(args.provider, model=args.model, max_tokens=args.max_tokens)
    print(f"预标注模型：{client.provider} / {client.model}")
    print(f"待标注 {len(test)} 条，已有 {len(done)} 条\n")

    ok = bad = 0
    for i, item in enumerate(test, 1):
        if item["id"] in done:
            continue
        prompt = template.replace("{{POST}}", item["text"])
        res = client.chat(prompt)
        obj = extract_json(res["text"]) if res["text"] else None

        # GLM-5.3 是推理模型：预算不够时 finish_reason='length'，JSON 被拦腰截断
        # （实测 2048 时只返回 "{"，而这一条实际需要 3325 个思考 token）。
        # 只在确认是「预算耗尽」时才加倍重试，不做盲目重试。
        if obj is None and res.get("finish_reason") == "length":
            print(f"    预算耗尽({res['completion_tokens']} tok)，加倍重试 …", flush=True)
            res2 = client.chat(prompt, max_tokens=args.max_tokens * 2)
            obj2 = extract_json(res2["text"]) if res2["text"] else None
            if obj2 is not None:
                res, obj = res2, obj2

        rec = {
            "id": item["id"],
            "suggest_attribution": (obj or {}).get("attribution"),
            "suggest_tendency": (obj or {}).get("tendency"),
            "suggest_evidence": (obj or {}).get("evidence"),
            "model": res["model_returned"],
            "ok": bool(obj),
            "raw": res["text"][:500],
            "at": datetime.now().isoformat(timespec="seconds"),
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if obj:
            ok += 1
            print(f"[{i}/{len(test)}] {item['id']} "
                  f"{rec['suggest_attribution']}/{rec['suggest_tendency']}", flush=True)
        else:
            bad += 1
            print(f"[{i}/{len(test)}] {item['id']} 解析失败：{res['error'] or res['text'][:80]}",
                  flush=True)

    print(f"\n完成：成功 {ok}，失败 {bad}")
    print(f"输出：{out_path}")
    print("\n提醒：这只是「建议」。标注时以标注手册的判定规则为准，不同意的直接改。")


if __name__ == "__main__":
    main()
