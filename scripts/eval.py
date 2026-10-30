"""评测：把模型输出和人工金标对比，算出各组的指标。

指标口径（对应方案第二部分「指标口径」，每个数字报告时必须带齐三样：
在哪个测试集上算的、什么协议算的、样本量多少）：

  归因五分类：准确率、宏平均 F1、加权 F1、混淆矩阵
      宏平均 F1 会被长尾类别拉低，所以和加权 F1 两个口径都要报。

  抑郁倾向：准确率、查准率、查全率、F1
      ⚠️「无法判断」单独统计，**不计入分母**。模型拒答和模型判错是两回事，
      混在一起算会掩盖真实表现。

  依据可追溯性：引用命中率、幻觉率
      命中率 = 模型引用的片段真实存在于原文的比例。
      幻觉率 = 1 - 命中率（越低越好）。这是本项目区别于同类工作的关键指标。

用法：
    python eval.py
    python eval.py --testset data/splits/test_gold.jsonl
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ATTR_LABELS = ["焦虑型", "压力型", "生理型", "环境型", "其他"]
TEND_LABELS = ["有抑郁倾向", "无抑郁倾向", "无法判断"]
POS, NEG, UNK = "有抑郁倾向", "无抑郁倾向", "无法判断"


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def normalize(s: str) -> str:
    """归一化用于字符串匹配：全角转半角、去掉所有空白与常见标点差异。

    标注端记录的是原文连续子串，但换行/空格/全半角会在比对时造成假不命中，
    所以两边都先归一化。这一步的规则必须写进报告。
    """
    if not s:
        return ""
    s = "".join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in s)
    return re.sub(r"[\s\W_]+", "", s).lower()


# ---- 依据可追溯性 ------------------------------------------------------------
#
# 为什么要拆成两个指标：
#
#   不强制引用时（A/C 组），模型输出的 evidence 往往是一整段解释性散文
#   （「……——帖主明确指出工作事件，属于压力型」），它不是原文的连续子串。
#   按「幻觉率 = 1 − 引用命中率」的老口径，A 组会得到 100% 幻觉率——
#   可 A 组压根没有引用，也就谈不上编造。这是两个性质完全不同的问题：
#
#     编造引用：声称引用了原文，但那句话不存在  → 幻觉
#     不产出引用：写了一段无法核查的解释        → 不可核验
#
#   混在一起会得出「A 组幻觉率 100%」这种答辩时解释不通的数字。

# 片段分隔符。模型可能用竖线、分号或换行连接多个引用，
# A/C 组没有格式约束，实际用分号的不少。这个口径要写进报告。
PIECE_SEP = re.compile(r"[|｜;；\n]+")
MIN_QUOTE = 5    # 与 v4 提示词里的引用长度要求保持一致
MAX_QUOTE = 50


def classify_pieces(evidence: str, text: str) -> tuple[int, int, int]:
    """把 evidence 切成片段并分三类，返回 (可核验, 疑似编造, 解释性)。

    - 可核验  ：长度在 [5,50] 且能在原文中找到 → 合格的引用
    - 疑似编造：长度在 [5,50] 但原文中找不到   → 声称引用却不存在的，这才是幻觉
    - 解释性  ：长度 >50 或 <5                → 没按引用格式输出，无法核查

    标 N/A 表示拒答，整体不计入分母（那是拒答，不是引用）。
    """
    if not evidence:
        return 0, 0, 0
    if evidence.strip().upper() in ("N/A", "NA", "无", "无法判断"):
        return 0, 0, 0

    norm_text = normalize(text)
    ver = fab = exp = 0
    for raw_piece in PIECE_SEP.split(evidence):
        p = raw_piece.strip().strip("「」“”\"'（）()")
        if not p:
            continue
        if not (MIN_QUOTE <= len(p) <= MAX_QUOTE):
            exp += 1
        elif normalize(p) in norm_text:
            ver += 1
        else:
            fab += 1
    return ver, fab, exp


def attr_metrics(pairs: list[tuple[str, str]]) -> dict:
    """pairs = [(gold, pred)]"""
    n = len(pairs)
    if not n:
        return {}
    correct = sum(1 for g, p in pairs if g == p)

    per = {}
    for lab in ATTR_LABELS:
        tp = sum(1 for g, p in pairs if g == lab and p == lab)
        fp = sum(1 for g, p in pairs if g != lab and p == lab)
        fn = sum(1 for g, p in pairs if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[lab] = {"n": tp + fn, "precision": prec, "recall": rec, "f1": f1}

    macro_f1 = sum(v["f1"] for v in per.values()) / len(ATTR_LABELS)
    weighted_f1 = sum(v["f1"] * v["n"] for v in per.values()) / n if n else 0.0

    conf = defaultdict(int)
    for g, p in pairs:
        conf[(g, p)] += 1

    return {
        "n": n,
        "accuracy": correct / n,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per,
        "confusion": dict(conf),
    }


def tend_metrics(pairs: list[tuple[str, str]]) -> dict:
    """二分类指标。「无法判断」不计入分母，单独统计。"""
    decidable = [(g, p) for g, p in pairs if g != UNK and p != UNK]
    gold_unk = sum(1 for g, p in pairs if g == UNK)
    pred_unk = sum(1 for g, p in pairs if p == UNK)

    out = {
        "n_total": len(pairs),
        "n_decidable": len(decidable),
        "gold_unknown": gold_unk,
        "pred_unknown": pred_unk,
        "abstain_rate": pred_unk / len(pairs) if pairs else 0.0,
    }
    if not decidable:
        return out

    tp = sum(1 for g, p in decidable if g == POS and p == POS)
    fp = sum(1 for g, p in decidable if g == NEG and p == POS)
    fn = sum(1 for g, p in decidable if g == POS and p == NEG)
    tn = sum(1 for g, p in decidable if g == NEG and p == NEG)

    def prf(tp_, fp_, fn_):
        p = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        r = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    prec, rec, f1 = prf(tp, fp, fn)              # 正类「有抑郁倾向」
    neg_prec, neg_rec, neg_f1 = prf(tn, fn, fp)  # 反类「无抑郁倾向」

    out.update({
        "accuracy": (tp + tn) / len(decidable),
        "precision": prec, "recall": rec, "f1": f1,
        "negative_precision": neg_prec,
        "negative_recall": neg_rec,
        "negative_f1": neg_f1,
        "macro_f1": (f1 + neg_f1) / 2,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", default="outputs/raw_prompt_responses.jsonl")
    ap.add_argument("--testset", default="data/splits/test_gold.jsonl")
    ap.add_argument("--out", default="outputs/results_prompt.csv")
    args = ap.parse_args()

    resp = load_jsonl(ROOT / args.responses)
    gold_rows = load_jsonl(ROOT / args.testset)
    if not resp:
        raise SystemExit(f"没有模型输出：{ROOT / args.responses}\n先跑 run_prompt_exp.py")
    if not gold_rows:
        raise SystemExit(
            f"没有金标测试集：{ROOT / args.testset}\n"
            f"标注（阶段 2）完成后才有。评测必须对着人工标注，不能对着模型自己的输出。"
        )

    gold = {r["id"]: r for r in gold_rows}
    # 原文用于算引用命中率
    raw = {r["id"]: r for r in load_jsonl(ROOT / "data/clean/tieba_clean.jsonl")}

    # 按 (provider, group) 分组，不能只按 group ——
    # 否则跨家族复跑时两个家族的输出会被混在一起算成一组。
    by_key = defaultdict(list)
    for r in resp:
        if r["id"] in gold:
            by_key[(r.get("provider", "?"), r["group"])].append(r)

    summary = []
    for provider, group in sorted(by_key):
        rows = by_key[(provider, group)]
        valid = [r for r in rows if r.get("valid")]

        attr_pairs = [(gold[r["id"]]["attribution"], r["attribution"]) for r in valid]
        tend_pairs = [(gold[r["id"]]["tendency"], r["tendency"]) for r in valid]

        ver = fab = exp = 0
        for r in valid:
            text = raw.get(r["id"], {}).get("text", "")
            v, f, e = classify_pieces(r.get("evidence") or "", text)
            ver += v
            fab += f
            exp += e
        n_pieces = ver + fab + exp

        am = attr_metrics(attr_pairs)
        tm = tend_metrics(tend_pairs)

        summary.append({
            "provider": provider,
            "group": group,
            "n_total": len(rows),
            "n_valid": len(valid),
            "invalid_rate": 1 - len(valid) / len(rows) if rows else 0,
            "attr_accuracy": am.get("accuracy"),
            "attr_macro_f1": am.get("macro_f1"),
            "attr_weighted_f1": am.get("weighted_f1"),
            "tend_n_decidable": tm.get("n_decidable"),
            # 注意区分两个口径：
            #   tend_accuracy     只在「金标和预测都不是无法判断」的样本上算 → 敢答时答得准不准
            #   tend_end2end_acc  全部样本都算，无法判断当作一个类别 → 端到端整体对不对
            # 强制引用组（D）拒答率接近一半，只看前者会严重高估它。
            "tend_accuracy": tm.get("accuracy"),
            "tend_end2end_acc": (
                sum(1 for g, p in tend_pairs if g == p) / len(tend_pairs)
                if tend_pairs else None),
            "tend_precision": tm.get("precision"),
            "tend_recall": tm.get("recall"),
            "tend_f1": tm.get("f1"),
            "tend_macro_f1": tm.get("macro_f1"),
            "abstain_rate": tm.get("abstain_rate"),
            # 可核验引用产出率：判断依据能不能被核查（分母含解释性片段）
            "verifiable_rate": ver / n_pieces if n_pieces else None,
            # 引用命中率与幻觉率：只在「像引用」的片段里算（分母不含解释性片段）
            "citation_hit_rate": ver / (ver + fab) if (ver + fab) else None,
            "hallucination_rate": fab / (ver + fab) if (ver + fab) else None,
            "n_pieces": n_pieces,
            "n_verifiable": ver,
            "n_fabricated": fab,
            "n_explanatory": exp,
            "mean_latency_s": (
                sum(r["latency_s"] for r in valid if r["latency_s"]) /
                max(1, sum(1 for r in valid if r["latency_s"]))
            ),
            "total_completion_tokens": sum(
                r["completion_tokens"] or 0 for r in valid),
        })

        # 混淆矩阵单独存，便于画图
        conf_path = ROOT / "outputs" / f"confusion_{provider}_{group}.csv"
        labels = ATTR_LABELS
        with conf_path.open("w", encoding="utf-8") as f:
            f.write("gold\\pred," + ",".join(labels) + "\n")
            for g in labels:
                f.write(g + "," + ",".join(
                    str(am.get("confusion", {}).get((g, p), 0)) for p in labels) + "\n")

    import csv
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # 打印
    print(f"测试集：{args.testset}（{len(gold_rows)} 条）")
    print(f"模型输出：{args.responses}\n")
    hdr = (f"{'模型:组':<20}{'有效':>6}{'归因Acc':>9}{'宏F1':>8}"
           f"{'倾向Acc':>9}{'倾向F1':>8}{'拒答率':>8}"
           f"{'可核验率':>9}{'命中率':>8}{'幻觉率':>8}")
    print(hdr)
    print("-" * len(hdr))

    def fmt(v, p=".3f"):
        return f"{v:{p}}" if isinstance(v, float) else "-"

    for s in summary:
        print(f"{s['provider'] + ':' + s['group']:<20}{s['n_valid']:>6}"
              f"{fmt(s['attr_accuracy']):>9}{fmt(s['attr_macro_f1']):>8}"
              f"{fmt(s['tend_accuracy']):>9}{fmt(s['tend_f1']):>8}"
              f"{fmt(s['abstain_rate']):>8}"
              f"{fmt(s['verifiable_rate']):>9}{fmt(s['citation_hit_rate']):>8}"
              f"{fmt(s['hallucination_rate']):>8}")

    print(f"\n结果表：{out_path}")
    print("混淆矩阵：outputs/confusion_<provider>_<组>.csv")
    print("\n指标口径说明：")
    print("  可核验率 = 合格引用片段 / 全部片段（含解释性片段）")
    print("           → 回答「判断依据能不能被核查」。低不代表编造，代表给不出可核查的依据。")
    print("  命中率   = 合格引用 / (合格引用 + 疑似编造)")
    print("           → 只在「声称是引用」的片段里算。分母为 0 时显示 '-'，不是 0%。")
    print("  幻觉率   = 疑似编造 / (合格引用 + 疑似编造)")
    print("  '依据支持率'需人工判读，不在本脚本内；抽样核对流程见标注手册。")


if __name__ == "__main__":
    main()
