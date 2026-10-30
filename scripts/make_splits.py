"""把清洗后的数据切成测试集与训练集。

设计要点：

1. **固定随机种子**。别人重跑必须拿到完全一样的两份数据，否则结果无法复现。

2. **按文本长度分层抽样**。纯随机可能让测试集恰好全是长帖或全是短帖，
   而短帖更可能被判「无法判断」，会系统性影响指标。分层能保证长度分布对齐。

3. **训练集里剔除测试集**，严防泄漏。少样本示例、微调数据都只能从训练集取。

用法：
    python make_splits.py                    # 默认测试集 180 条
    python make_splits.py --test-size 150
"""

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 长度分层的边界（按清洗后的正文字数）
LEN_BUCKETS = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 10**9)]


def bucket_of(n: int) -> int:
    for i, (lo, hi) in enumerate(LEN_BUCKETS):
        if lo <= n < hi:
            return i
    return len(LEN_BUCKETS) - 1


def main():
    ap = argparse.ArgumentParser()
    # 用 apply_gate.py 产出的最终数据集（已剔除科普/广告/他人故事）
    ap.add_argument("--clean", default="data/clean/tieba_final.jsonl")
    ap.add_argument("--test-size", type=int, default=180)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = ROOT / args.clean
    rows = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
    if not rows:
        raise SystemExit(f"没有数据：{src}")

    rng = random.Random(args.seed)

    # 按长度分层，每层按比例抽取
    by_bucket = defaultdict(list)
    for r in rows:
        by_bucket[bucket_of(r["n_chars"])].append(r)
    for v in by_bucket.values():
        rng.shuffle(v)

    total = len(rows)
    target = min(args.test_size, total // 2)  # 测试集不超过一半

    test, train = [], []
    for b in sorted(by_bucket):
        pool = by_bucket[b]
        n_take = round(len(pool) * target / total)
        test.extend(pool[:n_take])
        train.extend(pool[n_take:])

    # 补齐/回退，保证精确条数
    shortfall = target - len(test)
    if shortfall > 0:
        extra = [r for r in train if r not in test][:shortfall]
        test.extend(extra)
    elif shortfall < 0:
        train.extend(test[shortfall:])
        test = test[:target]

    train = [r for r in train if r["id"] not in {t["id"] for t in test}]
    rng.shuffle(test)
    rng.shuffle(train)

    out_dir = ROOT / "data/splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    def write(name, data, note):
        p = out_dir / name
        with p.open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {name:<24} {len(data):>4} 条   {note}")
        return p

    print(f"源数据：{src}（{total} 条），seed={args.seed}\n")
    write("test_pool.jsonl", test, "待标注的测试集（双人独立标注）")
    write("train.jsonl", train, "训练集（微调用，也是少样本示例的来源）")

    # 分层构成，便于报告里说明样本构成
    print("\n长度分层构成：")
    print(f"  {'区间':<14}{'测试集':>8}{'训练集':>8}{'合计':>8}")
    for i, (lo, hi) in enumerate(LEN_BUCKETS):
        label = f"{lo}-{'∞' if hi > 10**8 else hi} 字"
        nt = sum(1 for r in test if bucket_of(r["n_chars"]) == i)
        nr = sum(1 for r in train if bucket_of(r["n_chars"]) == i)
        print(f"  {label:<14}{nt:>8}{nr:>8}{nt + nr:>8}")

    print(f"\n下一步：")
    print(f"  1. python preannotate.py          # 用第三方模型生成预标注")
    print(f"  2. streamlit run label_tool.py    # 双人独立标注")


if __name__ == "__main__":
    main()
