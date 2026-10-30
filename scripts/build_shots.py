"""把人工标注的少样本示例写入 shots.jsonl，并校验 evidence 是不是原文子串。

B 组（few-shot）的示例必须来自训练集。写入前校验，避免示例本身带错——
示例错了，B 组的分数就是错的，而且很难事后发现。
"""

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 前 5 条由标注者人工判定；后 2 条按 v0.2 规则补判（零归因线索 → 其他），
# 目的是补上「其他」这个占 gold 50% 却完全没有示例的类别。
# 来源在 note 里区分，便于报告里说明。
LABELS = [
    ("tb_10969554365", "环境型", "无抑郁倾向", "因为被酒店空调外机的声音吵到", "人工标注"),
    ("tb_10983480298", "生理型", "有抑郁倾向", "从腰部以下到小腿持续性游走酸痛", "人工标注"),
    ("tb_10928167996", "压力型", "有抑郁倾向",
     "有时候希望睡一觉就再也醒不来了 | 最近离职了也一直没找到工作", "人工标注"),
    ("tb_10880531807", "生理型", "无抑郁倾向", "吃完不仅失眠严重了", "人工标注"),
    ("tb_10924255598", "生理型", "无抑郁倾向",
     "出现这种睡眠问题之前会经常性的熬夜并且有过几次通宵", "人工标注"),
    ("tb_10878198908", "其他", "无抑郁倾向", "N/A",
     "按v0.2补判：通篇设问无任何归因线索；无五类标记"),
    ("tb_10946817937", "其他", "无抑郁倾向", "N/A",
     "按v0.2补判：只描述早醒表现，无归因线索；无五类标记"),
]

ATTR = {"焦虑型", "压力型", "生理型", "环境型", "其他"}
TEND = {"有抑郁倾向", "无抑郁倾向", "无法判断"}


def norm(s):
    s = "".join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in s)
    return re.sub(r"[\s\W_]+", "", s).lower()


def main():
    train = {r["id"]: r for r in
             (json.loads(l) for l in
              (ROOT / "data/splits/train.jsonl").open(encoding="utf-8") if l.strip())}

    out, problems = [], []
    for rid, attr, tend, ev, source in LABELS:
        if rid not in train:
            problems.append(f"{rid} 不在训练集里")
            continue
        if attr not in ATTR:
            problems.append(f"{rid} 归因非法：{attr}")
        if tend not in TEND:
            problems.append(f"{rid} 倾向非法：{tend}")
        text = train[rid]["text"]
        for piece in ev.split("|"):
            p = piece.strip().strip("「」\"'")
            if p and p.upper() not in ("N/A", "NA") and norm(p) not in norm(text):
                problems.append(f"{rid} evidence 不是原文子串：{p!r}")
        out.append({
            "id": rid, "attribution": attr, "tendency": tend,
            "evidence": ev, "text": text, "n_chars": train[rid]["n_chars"],
            "source": source,
        })

    p = ROOT / "data/splits/shots.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"写入 {len(out)} 条 → {p}")
    if problems:
        print("\n⚠️ 问题：")
        for x in problems:
            print("  ✗", x)
    else:
        print("✓ evidence 全部通过原文子串校验")

    print("\n归因分布：")
    for k, v in Counter(r["attribution"] for r in out).most_common():
        print(f"  {k:<6} {v}")
    print("倾向分布：")
    for k, v in Counter(r["tendency"] for r in out).most_common():
        print(f"  {k:<8} {v}")


if __name__ == "__main__":
    main()
