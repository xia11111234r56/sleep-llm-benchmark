"""对 A–D 做配对显著性检验（McNemar）+ 准确率置信区间。

为什么必须做：147 条的样本量下，准确率的标准误约 3.5 个百分点。
差 5 个点很可能是噪声。不做检验就宣称「X 组更好」，是最容易被追问的地方。

McNemar 检验适合这里：同样的 147 条，比较两组「谁对谁错」的配对差异。
"""

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load():
    gold = {r["id"]: r for r in
            (json.loads(l) for l in
             (ROOT / "data/splits/test_gold.jsonl").open(encoding="utf-8") if l.strip())}
    resp = [json.loads(l) for l in
            (ROOT / "outputs/raw_prompt_responses.jsonl").open(encoding="utf-8") if l.strip()]
    return gold, resp


def mcnemar(b, c):
    """精确二项检验（双尾），b/c 是两种「一组对一组错」的计数。"""
    n = b + c
    if n == 0:
        return 1.0
    # 双尾精确 p 值
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return min(1.0, p)


def wilson(p, n, z=1.96):
    if n == 0:
        return (0, 0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0, center - half), min(1, center + half))


def main():
    gold, resp = load()

    # 每条 (组, id) 是否正确
    correct = defaultdict(dict)
    for r in resp:
        if not r.get("valid") or r["id"] not in gold:
            correct[r["group"]][r["id"]] = (False, False)  # 无效记为错
            continue
        g = gold[r["id"]]
        correct[r["group"]][r["id"]] = (
            r["attribution"] == g["attribution"],
            r["tendency"] == g["tendency"],
        )

    groups = sorted(correct)
    ids = sorted(set.intersection(*[set(correct[g]) for g in groups]))

    print(f"共同条目 {len(ids)} 条\n")

    for task, idx in (("归因", 0), ("倾向", 1)):
        print("=" * 66)
        print(f"【{task}】准确率与 95% 置信区间")
        acc = {}
        for g in groups:
            n = len(ids)
            k = sum(1 for i in ids if correct[g][i][idx])
            p = k / n
            acc[g] = p
            lo, hi = wilson(p, n)
            print(f"  {g}: {p:.3f}  [{lo:.3f}, {hi:.3f}]  ({k}/{n})")

        print(f"\n  两两比较（McNemar 精确检验，p<0.05 才算显著差异）")
        print(f"  {'对比':<10}{'差值':>8}{'仅左对':>8}{'仅右对':>8}{'p 值':>9}  结论")
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = groups[i], groups[j]
                only_a = sum(1 for x in ids if correct[a][x][idx] and not correct[b][x][idx])
                only_b = sum(1 for x in ids if correct[b][x][idx] and not correct[a][x][idx])
                p = mcnemar(only_a, only_b)
                diff = acc[a] - acc[b]
                verdict = "显著" if p < 0.05 else "不显著"
                print(f"  {a} vs {b:<6}{diff:>+8.3f}{only_a:>8}{only_b:>8}{p:>9.3f}  {verdict}")
        print()


if __name__ == "__main__":
    main()
