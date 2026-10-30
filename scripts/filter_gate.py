"""用模型判定帖子是否为「发帖人本人的失眠叙事/求助」，滤掉科普、广告、文献、他人故事。

为什么需要这一步：
    正则规则对中文科普文几乎无效。实测一条 401 字的《景岳全书》文献帖完整通过了所有
    关键词规则——它没有"广告"字样，也没有明显的结构标记。靠继续加正则修不动。

为什么用模型不违反「不得编造」：
    这不是给数据打标签，只是判断「这条是不是本人求助帖」。它和归因分类是正交的
    （判完照样要人工标归因），不会往标签里注入任何偏好。判定结果全部落盘可审计。

用法：
    python filter_gate.py              # 全量
    python filter_gate.py --limit 20   # 试跑
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_client import LLMClient, extract_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

VERDICTS = ["本人求助", "科普文献", "广告推广", "他人故事"]

PROMPT = """判断下面这条贴吧帖子的类型，只输出一个 JSON 对象，不要任何解释文字：

{{"verdict": "本人求助 | 科普文献 | 广告推广 | 他人故事", "reason": "一句话理由"}}

判定标准：

- **本人求助**：发帖人叙述**自己**的失眠经历、困扰或求助。
- **科普文献**：医学或养生知识、文献摘录、疗法介绍、书目转发。没有发帖人自己的失眠叙事。
- **广告推广**：推销产品或服务，含联系方式、购买链接、机构宣传。
- **他人故事**：讲述朋友、家人、名人的失眠；发帖人自己不受失眠困扰。

三条容易判错的提醒：

1. 中文常省略主语。「每天凌晨三点睡不着，怎么办」这类**属于本人求助**，不要因为没写「我」就误判。
2. 「药」「中医」「医院」出现在真实求助里很正常，**不要因此判成广告**。广告要看有没有推销意图。
3. 帖子里大段引用古文、药方、医案，但前后有自己的经历描述 —— 属于**本人求助**；
   如果通篇只有医理和文献、没有个人叙事 —— 属于**科普文献**。

帖子正文：
{{POST}}"""


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="data/clean/tieba_clean.jsonl")
    ap.add_argument("--out", default="data/clean/gate.jsonl")
    ap.add_argument("--provider", default="qwen", help="判定模型（要快，GLM 太慢）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    rows = load_jsonl(ROOT / args.clean)
    if not rows:
        raise SystemExit(f"没有数据：{ROOT / args.clean}")
    if args.limit:
        rows = rows[:args.limit]

    out_path = ROOT / args.out
    done = {r["id"] for r in load_jsonl(out_path)}
    client = LLMClient(args.provider, max_tokens=args.max_tokens)
    print(f"判定模型：{client.provider} / {client.model}")
    print(f"待判定 {len(rows)} 条，已有 {len(done)} 条\n")

    for i, rec in enumerate(rows, 1):
        if rec["id"] in done:
            continue
        res = client.chat(PROMPT.replace("{{POST}}", rec["text"]))
        obj = extract_json(res["text"]) if res["text"] else None
        verdict = (obj or {}).get("verdict")
        if verdict not in VERDICTS:
            verdict, reason = "判定失败", str(obj or res.get("error") or res["text"][:100])
        else:
            reason = (obj or {}).get("reason", "")

        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "id": rec["id"], "verdict": verdict, "reason": reason,
                "model": res["model_returned"],
                "at": datetime.now().isoformat(timespec="seconds"),
            }, ensure_ascii=False) + "\n")

        mark = "" if verdict == "本人求助" else f"  ← {verdict}"
        print(f"[{i}/{len(rows)}] {rec['id']} {verdict}{mark}", flush=True)

    all_rows = load_jsonl(out_path)
    print(f"\n完成 {len(all_rows)} 条")
    for v, n in Counter(r["verdict"] for r in all_rows).most_common():
        print(f"  {v:<8} {n:>4} ({n/len(all_rows):.1%})")
    print(f"\n输出：{out_path}")


if __name__ == "__main__":
    main()
