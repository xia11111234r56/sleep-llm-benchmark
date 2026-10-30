"""根据协商裁定生成最终 gold label。

来源优先级：
  1. 协商裁定的 42 条（gold_adjudicated.csv）—— 覆盖 A/B 原判断
  2. 其余条目用 A 的标注（30 条协商稿 + 78 条 A/B 一致的）

Kappa 不受影响：它已经按独立标注算完并写进报告，这里只是产出 gold。
"""

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCLUDE = "剔除"


def load_jsonl(p):
    p = ROOT / p
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


def norm(s):
    import re
    s = "".join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in s)
    return re.sub(r"[\s\W_]+", "", s).lower()


def main():
    pool = {r["id"]: r for r in load_jsonl("data/splits/test_pool.jsonl")}
    A = {r["id"]: r for r in load_jsonl("data/labeled/annot_A.jsonl")}
    B = {r["id"]: r for r in load_jsonl("data/labeled/annot_B.jsonl")}

    with (ROOT / "data/labeled/gold_adjudicated.csv").open(encoding="utf-8-sig", newline="") as f:
        adj = list(csv.DictReader(f))

    adj_ids = {r["id"] for r in adj}
    print(f"协商裁定 {len(adj)} 条")

    # 核对：裁定表覆盖的是不是正好那 42 条分歧
    A_ids, B_ids = set(A), set(B)
    independent = A_ids & B_ids
    disagreements = {i for i in independent
                     if A[i]["attribution"] != B[i]["attribution"]
                     or A[i]["tendency"] != B[i]["tendency"]}
    print(f"实际分歧 {len(disagreements)} 条")
    extra = adj_ids - disagreements
    missing_adj = disagreements - adj_ids
    if missing_adj:
        print(f"  ⚠️ 裁定表缺少 {len(missing_adj)} 条分歧：{sorted(missing_adj)[:5]}")
    else:
        print(f"  ✓ 覆盖全部 {len(disagreements)} 条分歧"
              + (f"（另含 {len(extra)} 条补判）" if extra else ""))
    for i in sorted(extra):
        print(f"     补判：{i}")

    gold, excluded = [], []
    for rid, item in pool.items():
        if rid in adj_ids:
            row = next(r for r in adj if r["id"] == rid)
            attr, tend = row["最终归因"].strip(), row["最终倾向"].strip()
            note = row.get("备注", "").strip()
            source = "协商裁定"
        elif rid in A:
            attr, tend = A[rid]["attribution"], A[rid]["tendency"]
            note = A[rid].get("note", "")
            source = "A标注" if rid not in independent else "A/B一致"
        else:
            print(f"  ⚠️ {rid} 没有任何标注，跳过")
            continue

        if EXCLUDE in (attr, tend):
            excluded.append({"id": rid, "reason": note})
            continue

        # evidence：优先取 A，A 为空则取 B
        ev = (A.get(rid, {}).get("evidence") or "").strip()
        if not ev or ev.upper() in ("N/A", "NA"):
            evb = (B.get(rid, {}).get("evidence") or "").strip()
            if evb and evb.upper() not in ("N/A", "NA"):
                ev = evb

        gold.append({
            "id": rid,
            "attribution": attr,
            "tendency": tend,
            "evidence": ev or "N/A",
            "text": item["text"],
            "n_chars": item["n_chars"],
            "source": source,
            "note": note,
        })

    out = ROOT / "data/splits/test_gold.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in gold:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n测试集 {len(pool)} 条 → gold {len(gold)} 条，剔除 {len(excluded)} 条")
    for e in excluded:
        print(f"  剔除 {e['id']}：{e['reason'][:40]}")

    print("\n归因分布：")
    for k, v in Counter(r["attribution"] or "(未标)" for r in gold).most_common():
        print(f"  {k:<8} {v:>3} ({v/len(gold):.1%})")
    print("倾向分布：")
    for k, v in Counter(r["tendency"] or "(未标)" for r in gold).most_common():
        print(f"  {k:<10} {v:>3} ({v/len(gold):.1%})")

    missing = [r for r in gold if not r["attribution"] or not r["tendency"]]
    if missing:
        print(f"\n⚠️ {len(missing)} 条标签缺失，需要补标：")
        for r in missing:
            print(f"  {r['id']}  归因={r['attribution']}  倾向={r['tendency']}")
            print(f"    正文：{r['text'][:80]}")

    n50 = [r for r in gold if r["n_chars"] >= 50]
    if n50:
        o = sum(1 for r in n50 if r["attribution"] == "其他")
        print(f"\n≥50 字子集 {len(n50)} 条，其他 {o} 条 ({o/len(n50):.1%})")
    print(f"\n输出：{out}")


if __name__ == "__main__":
    main()
