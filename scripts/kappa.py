"""计算双人标注的一致性（Cohen's Kappa），并导出争议条目。

判定标准（项目方案）：Kappa ≥ 0.6 可用；低于 0.6 要回去改标注手册、重新协商。

⚠️ Kappa 是**测量标注质量的工具，不是要交付的成果**。
   为了让数字好看而商量着把测试集标一致，是在伪造证据。
   真正要看的是「不一致的条目长什么样」——那才暴露手册哪里没说清。

用法：
    python kappa.py
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent

ATTR_LABELS = ["焦虑型", "压力型", "生理型", "环境型", "其他"]
TEND_LABELS = ["有抑郁倾向", "无抑郁倾向", "无法判断"]

# Landis & Koch (1977) 的解读口径，报告里引用时注明出处
LANDIS_KOCH = [
    (0.00, "几乎不一致 (poor)"),
    (0.20, "轻微一致 (slight)"),
    (0.40, "中等一致 (fair)"),
    (0.60, "较强一致 (moderate)"),
    (0.80, "高度一致 (substantial)"),
    (1.01, "几乎完全一致 (almost perfect)"),
]


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def interpret(k: float) -> str:
    for th, label in LANDIS_KOCH:
        if k < th:
            return label
    return LANDIS_KOCH[-1][1]


def agree_rate(pairs) -> float:
    if not pairs:
        return 0.0
    return sum(1 for a, b in pairs if a == b) / len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="data/labeled/annot_A.jsonl")
    ap.add_argument("--b", default="data/labeled/annot_B.jsonl")
    args = ap.parse_args()

    A = {r["id"]: r for r in load_jsonl(ROOT / args.a)}
    B = {r["id"]: r for r in load_jsonl(ROOT / args.b)}

    if not A or not B:
        raise SystemExit(
            f"缺少标注文件。\n  A: {ROOT / args.a}（{len(A)} 条）\n"
            f"  B: {ROOT / args.b}（{len(B)} 条）\n"
            f"两个标注者标完后才能算 Kappa。"
        )

    ids = sorted(set(A) & set(B))
    only_a = sorted(set(A) - set(B))
    only_b = sorted(set(B) - set(A))
    if not ids:
        raise SystemExit("两位标注者没有共同标注的条目。")

    attr_pairs = [(A[i]["attribution"], B[i]["attribution"]) for i in ids]
    tend_pairs = [(A[i]["tendency"], B[i]["tendency"]) for i in ids]

    k_attr = cohen_kappa_score([x for x, _ in attr_pairs], [y for _, y in attr_pairs],
                               labels=ATTR_LABELS)
    k_tend = cohen_kappa_score([x for x, _ in tend_pairs], [y for _, y in tend_pairs],
                               labels=TEND_LABELS)

    # 不一致条目
    disagreements = []
    for i in ids:
        a, b = A[i], B[i]
        diffs = []
        if a["attribution"] != b["attribution"]:
            diffs.append(f"归因 A={a['attribution']} / B={b['attribution']}")
        if a["tendency"] != b["tendency"]:
            diffs.append(f"倾向 A={a['tendency']} / B={b['tendency']}")
        if diffs:
            disagreements.append({
                "id": i, "分歧": "；".join(diffs),
                "note_A": a.get("note", ""), "note_B": b.get("note", ""),
                "正文": a.get("text", "")[:300],
            })

    # 混淆：A 的标签 → B 的标签分布
    def conf(pairs):
        c = defaultdict(int)
        for x, y in pairs:
            c[(x, y)] += 1
        return c

    c_attr = conf(attr_pairs)

    # ---- 报告 ----
    lines = [
        "# 标注一致性报告",
        "",
        f"- 标注者 A：{len(A)} 条（`{args.a}`）",
        f"- 标注者 B：{len(B)} 条（`{args.b}`）",
        f"- **共同标注：{len(ids)} 条**（Kappa 在此子集上计算）",
    ]
    if only_a:
        lines.append(f"- ⚠️ 仅 A 标注：{len(only_a)} 条（未参与计算）")
    if only_b:
        lines.append(f"- ⚠️ 仅 B 标注：{len(only_b)} 条（未参与计算）")

    lines += [
        "",
        "## 结果",
        "",
        "| 任务 | 一致率 | Cohen's Kappa | 解读 |",
        "|---|---|---|---|",
        f"| 归因五分类 | {agree_rate(attr_pairs):.3f} | **{k_attr:.3f}** | {interpret(k_attr)} |",
        f"| 抑郁倾向 | {agree_rate(tend_pairs):.3f} | **{k_tend:.3f}** | {interpret(k_tend)} |",
        "",
        "> 解读口径：Landis & Koch (1977)。项目方案要求 Kappa ≥ 0.6 方可用于后续评测。",
        "",
        "## 判定",
        "",
    ]

    for name, k in (("归因五分类", k_attr), ("抑郁倾向", k_tend)):
        if k >= 0.6:
            lines.append(f"- **{name}**：Kappa = {k:.3f} ≥ 0.6，✅ 达标。")
        else:
            lines.append(f"- **{name}**：Kappa = {k:.3f} < 0.6，❌ **未达标**。"
                         f"需回改标注手册、明确边界案例，两人重新协商定义后抽 50 条重标验证。")

    lines += [
        "",
        "## 归因标签的混淆（行=A，列=B）",
        "",
        "| A \\ B | " + " | ".join(ATTR_LABELS) + " |",
        "|---" * (len(ATTR_LABELS) + 1) + "|",
    ]
    for x in ATTR_LABELS:
        row = [str(c_attr.get((x, y), 0)) for y in ATTR_LABELS]
        lines.append(f"| {x} | " + " | ".join(row) + " |")

    lines += [
        "",
        f"## 分歧条目（{len(disagreements)} 条）",
        "",
        "这些是协商会要逐条过的。**不要为了让 Kappa 好看而商量着标成一致**——",
        "要判断的是「哪一类分歧说明手册没写清」，据此改手册，而不是改数据。",
        "",
    ]
    if disagreements:
        lines.append("| id | 分歧 | A 的备注 | B 的备注 |")
        lines.append("|---|---|---|---|")
        for d in disagreements[:50]:
            lines.append(f"| {d['id']} | {d['分歧']} | {d['note_A']} | {d['note_B']} |")
        if len(disagreements) > 50:
            lines.append(f"\n（仅列出前 50 条，全部 {len(disagreements)} 条见 disagreements.csv）")

    out_md = ROOT / "data/labeled/一致性报告.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")

    with (ROOT / "data/labeled/disagreements.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "分歧", "note_A", "note_B", "正文"])
        w.writeheader()
        w.writerows(disagreements)

    print(f"共同标注 {len(ids)} 条")
    print(f"  归因五分类  Kappa = {k_attr:.3f}  ({interpret(k_attr)})  一致率 {agree_rate(attr_pairs):.3f}")
    print(f"  抑郁倾向    Kappa = {k_tend:.3f}  ({interpret(k_tend)})  一致率 {agree_rate(tend_pairs):.3f}")
    print(f"  分歧条目 {len(disagreements)} 条")
    print(f"\n报告：{out_md}")
    print(f"分歧清单：data/labeled/disagreements.csv")


if __name__ == "__main__":
    main()
