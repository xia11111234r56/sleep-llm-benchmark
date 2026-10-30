"""把 Word/表格里手工标的结果录回 jsonl。

为什么要校验：手工录入 + 剪贴板，最容易出的错是 id 对不上、标签写错、
evidence 不是原文子串。这些错在算指标时才暴露，代价很高，所以录入时就拦。

用法：
    python import_annotations.py --raw data/labeled/_A_raw.txt --annotator A
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ATTR_LABELS = {"焦虑型", "压力型", "生理型", "环境型", "其他"}
TEND_LABELS = {"有抑郁倾向", "无抑郁倾向", "无法判断"}


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def normalize(s: str) -> str:
    s = "".join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in s)
    return re.sub(r"[\s\W_]+", "", s).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="横线分隔的原始表格（TSV）")
    ap.add_argument("--annotator", required=True, choices=["A", "B"])
    ap.add_argument("--dry-run", action="store_true", help="只校验不写入")
    args = ap.parse_args()

    rows = list(csv.DictReader((ROOT / args.raw).open(encoding="utf-8"),
                               delimiter="\t"))
    print(f"读到 {len(rows)} 行\n")

    test = {r["id"]: r for r in load_jsonl(ROOT / "data/splits/test_pool.jsonl")}
    out_path = ROOT / f"data/labeled/annot_{args.annotator}.jsonl"
    existing = {r["id"]: r for r in load_jsonl(out_path)}
    print(f"已有标注 {len(existing)} 条（本次只追加，不覆盖）\n")

    import datetime
    today = datetime.date.today().isoformat()

    problems = []
    accepted = []
    seen = set()

    for i, r in enumerate(rows, 1):
        rid = (r.get("id") or "").strip()
        emo = (r.get("抑郁倾向") or "").strip()
        attr = (r.get("失眠归因") or "").strip()
        ev = (r.get("evidence") or "").strip()
        note = (r.get("备注") or "").strip()

        if not rid:
            problems.append(f"第 {i} 行没有 id")
            continue
        if rid in seen:
            problems.append(f"第 {i} 行 id 重复：{rid}")
            continue
        seen.add(rid)

        if rid not in test:
            problems.append(f"{rid} 不在测试集里")
            continue
        if rid in existing:
            problems.append(f"{rid} 已有标注，跳过")
            continue
        if emo not in TEND_LABELS:
            problems.append(f"{rid} 倾向标签非法：{emo!r}")
        if attr not in ATTR_LABELS:
            problems.append(f"{rid} 归因标签非法：{attr!r}")

        text = test[rid]["text"]
        if ev and ev.upper() not in ("N/A", "NA"):
            for piece in ev.replace("｜", "|").split("|"):
                p = piece.strip().strip("「」\"'")
                if p and normalize(p) not in normalize(text):
                    problems.append(f"{rid} evidence 不是原文子串：{p[:30]!r}")
        elif emo == "有抑郁倾向":
            problems.append(f"{rid} 判了「有抑郁倾向」但 evidence 是 N/A"
                            f"（手册要求必须指出具体标记原句）")

        accepted.append({
            "id": rid,
            "attribution": attr,
            "tendency": emo,
            "evidence": ev,
            "annotator": args.annotator,
            "annotated_at": today,
            "note": note,
            "text": text,
        })

    print(f"可录入 {len(accepted)} 条，问题 {len(problems)} 处\n")
    if problems:
        print("=== 问题清单 ===")
        for p in problems:
            print("  ✗", p)
        print()

    if args.dry_run:
        print("（--dry-run，未写入）")
        return

    if not accepted:
        return
    with out_path.open("a", encoding="utf-8") as f:
        for rec in accepted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"已写入 {out_path}（现有 {len(existing) + len(accepted)} 条）")


if __name__ == "__main__":
    main()
