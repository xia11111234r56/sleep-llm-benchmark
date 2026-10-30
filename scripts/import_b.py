"""解析 B 的标注格式并录入。

B 的表格格式和 A 不同：数字后用英文逗号，字段内用中文逗号，且 evidence 和备注
混在一串里没有分隔。处理办法：
  1. 前两个中文逗号字段一定是「倾向」「归因」——这两个决定 Kappa，解析必须可靠
  2. 后面的部分用「是否仍是原文连续子串」贪心切出 evidence，剩下的都归备注
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ATTR_LABELS = {"焦虑型", "压力型", "生理型", "环境型", "其他"}
TEND_LABELS = {"有抑郁倾向", "无抑郁倾向", "无法判断"}


def normalize(s: str) -> str:
    s = "".join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in s)
    return re.sub(r"[\s\W_]+", "", s).lower()


def main():
    import datetime
    today = datetime.date.today().isoformat()

    test = {r["id"]: r for r in
            (json.loads(l) for l in
             (ROOT / "data/splits/test_pool.jsonl").open(encoding="utf-8") if l.strip())}
    out_path = ROOT / "data/labeled/annot_B.jsonl"

    lines = [l for l in (ROOT / "data/labeled/_B_raw.txt").open(encoding="utf-8") if l.strip()]
    print(f"读到 {len(lines)} 行\n")

    records, problems = [], []
    for line in lines:
        parts = line.split(",")
        if len(parts) < 3:
            problems.append(f"行格式异常：{line[:40]}")
            continue
        rid = parts[1].strip()
        rest = ",".join(parts[2:])
        fields = [f.strip() for f in rest.split("，") if f.strip()]

        if len(fields) < 2:
            problems.append(f"{rid} 字段不足")
            continue
        tend, attr = fields[0], fields[1]
        tail = fields[2:]

        if rid not in test:
            problems.append(f"{rid} 不在测试集")
            continue
        if tend not in TEND_LABELS:
            problems.append(f"{rid} 倾向非法：{tend!r}")
        if attr not in ATTR_LABELS:
            problems.append(f"{rid} 归因非法：{attr!r}")

        text_norm = normalize(test[rid]["text"])

        if tail and tail[0].upper() in ("N/A", "NA"):
            evidence = "N/A"
            note = "，".join(tail[1:])
        else:
            ev_parts = []
            for f in tail:
                cand = "，".join(ev_parts + [f])
                if normalize(cand) in text_norm:
                    ev_parts.append(f)
                else:
                    break
            if ev_parts:
                evidence = "，".join(ev_parts)
                note = "，".join(tail[len(ev_parts):])
            else:
                evidence = "N/A"
                note = "，".join(tail)

        if evidence != "N/A" and normalize(evidence) not in text_norm:
            problems.append(f"{rid} evidence 不是原文子串")

        records.append({
            "id": rid, "attribution": attr, "tendency": tend,
            "evidence": evidence, "annotator": "B",
            "annotated_at": today, "note": note, "text": test[rid]["text"],
        })

    print(f"可录入 {len(records)} 条，问题 {len(problems)} 处")
    for p in problems[:20]:
        print("  ✗", p)

    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n已写入 {out_path}")

    # evidence 抽取质量抽查
    got_ev = sum(1 for r in records if r["evidence"] != "N/A")
    print(f"抽到原文 evidence 的: {got_ev} 条，其余为 N/A 或未抽出")


if __name__ == "__main__":
    main()
