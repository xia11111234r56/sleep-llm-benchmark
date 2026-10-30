"""应用相关性判定结果，只保留「本人求助」帖，产出最终数据集。

流水线位置：
    crawl.py → clean.py → filter_gate.py → apply_gate.py → make_splits.py

为什么单独一步：相关性判定要调用模型，和纯规则清洗是两个层次。
分开之后，判定结果（gate.jsonl）本身可以审计、可以回退。
"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEEP = "本人求助"


def load(p):
    p = ROOT / p
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


def main():
    clean = load("data/clean/tieba_clean.jsonl")
    gate = {r["id"]: r for r in load("data/clean/gate.jsonl")}

    missing = [r["id"] for r in clean if r["id"] not in gate]
    if missing:
        print(f"⚠️ {len(missing)} 条没有判定结果，已剔除。先跑 filter_gate.py 补上。")

    kept = [r for r in clean if gate.get(r["id"], {}).get("verdict") == KEEP]
    dropped = [r for r in clean if r["id"] not in {k["id"] for k in kept}]

    out = ROOT / "data/clean/tieba_final.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"清洗后 {len(clean)} 条 → 判定为本人求助 {len(kept)} 条，剔除 {len(dropped)} 条")
    print("剔除原因：", dict(Counter(gate[r["id"]]["verdict"]
                                    for r in dropped if r["id"] in gate)))
    lens = sorted(r["n_chars"] for r in kept)
    print(f"字数：最小 {lens[0]} / 中位 {lens[len(lens)//2]} / 最大 {lens[-1]}")
    print(f"\n输出：{out}")


if __name__ == "__main__":
    main()
