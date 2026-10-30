"""汇总 A–D 结果并出图。

生成：
  outputs/figures/fig1_accuracy.png    三组任务准确率对比
  outputs/figures/fig2_citation.png    引用可核验性对比
  outputs/figures/fig3_confusion.png   四组归因混淆矩阵
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "outputs/figures"
FIG.mkdir(parents=True, exist_ok=True)

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

LABELS = ["焦虑型", "压力型", "生理型", "环境型", "其他"]
GROUP_NAME = {
    "A": "A 零样本", "B": "B 少样本",
    "C": "C 结构化", "D": "D 强制引用",
}
COLORS = ["#9aa6b2", "#5b8def", "#f0a23c", "#3fa66a"]


def load_results():
    with (ROOT / "outputs/results_prompt.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("provider", "group"):
                continue
            try:
                r[k] = float(v) if v not in ("", None) else None
            except ValueError:
                r[k] = None
    return rows


def fig_accuracy(rows):
    groups = [r["group"] for r in rows]
    # 倾向放两个口径：D 组拒答率近一半，只看「敢答时」会严重高估它
    metrics = [("attr_accuracy", "归因准确率"), ("attr_macro_f1", "归因宏F1"),
               ("tend_accuracy", "倾向(敢答时)"), ("tend_end2end_acc", "倾向(端到端)")]

    x = np.arange(len(metrics))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, r in enumerate(rows):
        vals = [r[k] or 0 for k, _ in metrics]
        bars = ax.bar(x + (i - 1.5) * w, vals, w,
                      label=GROUP_NAME.get(r["group"], r["group"]), color=COLORS[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in metrics])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("分数")
    ax.set_title("A–D 四组在三个任务上的表现", fontsize=13)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_accuracy.png", dpi=150)
    plt.close(fig)


def fig_citation(rows):
    groups = [r["group"] for r in rows]
    x = np.arange(len(groups))
    w = 0.26

    def series(key):
        return [r[key] if r[key] is not None else 0 for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, series("verifiable_rate"), w, label="可核验引用产出率", color="#5b8def")
    ax.bar(x, series("hallucination_rate"), w, label="幻觉率", color="#d9534f")
    ax.bar(x + w, series("abstain_rate"), w, label="拒答率", color="#f0a23c")

    for i, r in enumerate(rows):
        for j, key in enumerate(("verifiable_rate", "hallucination_rate", "abstain_rate")):
            v = r[key]
            if v is None:
                ax.text(i + (j - 1) * w, 0.02, "无", ha="center", fontsize=8, color="gray")
            else:
                ax.text(i + (j - 1) * w, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_NAME.get(g, g) for g in groups])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("比例")
    ax.set_title("依据可追溯性：能核查多少、编造多少、拒答多少", fontsize=13)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_citation.png", dpi=150)
    plt.close(fig)


def fig_confusion(rows):
    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 4))
    if len(rows) == 1:
        axes = [axes]
    for ax, r in zip(axes, rows):
        p = ROOT / "outputs" / f"confusion_{r['provider']}_{r['group']}.csv"
        if not p.exists():
            ax.set_title(f"{r['group']}：无数据")
            continue
        with p.open(encoding="utf-8-sig", newline="") as f:
            rd = list(csv.reader(f))
        mat = np.array([[int(x) for x in row[1:]] for row in rd[1:]])
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(len(LABELS)))
        ax.set_yticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(LABELS, fontsize=8)
        ax.set_xlabel("模型预测", fontsize=8)
        ax.set_ylabel("人工 gold", fontsize=8)
        ax.set_title(GROUP_NAME.get(r["group"], r["group"]), fontsize=11)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i, j]:
                    ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=8,
                            color="white" if mat[i, j] > mat.max() * 0.55 else "black")
    fig.suptitle("归因五分类混淆矩阵", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_confusion.png", dpi=150)
    plt.close(fig)


def main():
    rows = load_results()
    fig_accuracy(rows)
    fig_citation(rows)
    fig_confusion(rows)
    print("已生成：")
    for p in sorted(FIG.glob("*.png")):
        print(f"  {p.name}  {p.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
