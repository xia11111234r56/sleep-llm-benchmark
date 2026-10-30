"""清洗原始帖子：去重、去掉非求助内容、统计样本构成。

设计原则：**过滤要可审计**。每一条被丢弃的记录都会写进 removed.jsonl 并附上原因，
方便事后回查是不是误杀。清洗规则宁可保守——漏掉一条广告的代价，
远小于把真实求助帖当成广告删掉。

用法：
    python clean.py
    python clean.py --min-chars 30
"""

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- 过滤规则 ----------------------------------------------------------

# 吧务公告 / 置顶帖：不是用户求助
ANNOUNCE_PATTERNS = [
    r"本吧禁止", r"吧规", r"吧务", r"置顶", r"小吧主", r"招募",
    r"广告专用帖", r"请勿.{0,6}(发|发布|回复)", r"删帖", r"封禁",
]

# 软广 / 引流 / 卖货：用较长词组，避免误伤正常提到"药""中医"的求助帖
AD_PATTERNS = [
    r"加\s*(我|一下)?\s*微信", r"加\s*V\s*[:：xX]", r"微信号", r"私聊我", r"私信我",
    r"联系我.{0,6}(购买|下单|咨询)", r"需要的?私", r"点击链接",
    r"包治|根治|药到病除|彻底治愈|永不复发",
    r"亲测有效|我就是用这个好的|用了.{0,4}就好了|推荐给大家.{0,6}(产品|贴|药)",
    r"代购|进口特效", r"独家配方|祖传秘方",
]

# 非用户内容：法律条文集、网盘转载、明确的产品推广。
#
# 为什么需要这一组：实测 488 条里有大量「标题是失眠，正文粘贴民法典/合同法」的垃圾帖，
# 以及产品软广（"咨询渡康官方微信"）。它们不是求助帖，留着会大量落进「其他」类，
# 把归因五分类做成一个「辨认垃圾帖」的任务。
#
# 规则刻意收紧：裸的 `购买|淘宝|https://` 会误伤真实帖子
#（实测有用户在帖子里放自己爱听的电台网址、提到淘宝买助眠产品），所以只保留
# 带明确推销意图的组合。
NONUSER_PATTERNS = [
    r"中华人民共和国|依照本法|本法第|第[一二三四五六七八九十百零]{1,3}条",
    r"pan\.baidu\.com|ISBN|出版社\s*[/／]",
    # 只认带明确推销意图的组合。不能写「加…微信」——实测会误杀
    # 「我发过誓再也不加我前前男友微信了」这类真实叙事。
    r"官方微信|微信号\s*[:：]|(咨询|了解|购买|详情|预约).{0,6}微信"
    r"|微信.{0,6}(咨询|购买|下单|预约)|扫码|先治疗后收",
]


def looks_like_article(text: str) -> bool:
    """判别科普文/文献转载。需要 3 个以上软信号才判，避免误杀真实长帖。"""
    signals = 0
    if len(text) >= 500:
        signals += 1
    if re.search(r"^[一二三四五六七八九十]+、", text, re.M):
        signals += 1
    if re.search(r"^\s*\d+\s*[、．.]", text, re.M):
        signals += 1
    if re.search(r"疗效|痊愈|显效|总有效率|适用于|方剂|穴位|经络|药对", text):
        signals += 1
    if re.search(r"方法\s*[:：]", text):
        signals += 1
    if not re.search(r"我|本人|自己", text):
        signals += 1
    return signals >= 3


def normalize(text: str) -> str:
    """归一化用于去重：去掉所有空白与标点，全角转半角。"""
    t = text.lower()
    t = "".join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in t)
    return re.sub(r"[\s\W_]+", "", t)


def match_any(patterns, text: str) -> str | None:
    for p in patterns:
        if re.search(p, text):
            return p
    return None


def clean_record(rec: dict, min_chars: int):
    """返回 (保留的记录 | None, 丢弃原因 | None)。"""
    text = (rec.get("text") or "").strip()

    if not text:
        return None, "无正文"

    # 公告类：标题或正文命中
    both = f"{rec.get('title', '')}\n{text}"
    p = match_any(ANNOUNCE_PATTERNS, both)
    if p:
        return None, f"吧务公告(/{p}/)"

    if len(text) < min_chars:
        return None, f"正文过短(<{min_chars}字)"

    p = match_any(AD_PATTERNS, text)
    if p:
        return None, f"疑似广告(/{p}/)"

    p = match_any(NONUSER_PATTERNS, text)
    if p:
        return None, f"非用户内容(/{p}/)"

    if looks_like_article(text):
        return None, "非用户内容(科普/文献转载)"

    rec = dict(rec)
    rec["text"] = re.sub(r"\n{2,}", "\n", text).strip()
    rec["title"] = (rec.get("title") or "").strip()
    rec["n_chars"] = len(rec["text"])
    return rec, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/tieba_raw.jsonl")
    ap.add_argument("--out", default="data/clean/tieba_clean.jsonl")
    # 与标注手册 §0 保持一致：手册规定「正文不足 15 字」才剔除。
    # 手册是规格，代码要跟着它走，不能各写各的。
    ap.add_argument("--min-chars", type=int, default=15)
    args = ap.parse_args()

    raw_path = ROOT / args.raw
    out_path = ROOT / args.out
    removed_path = out_path.parent / "removed.jsonl"
    stats_path = out_path.parent / "stats.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise SystemExit(f"找不到原始数据：{raw_path}")

    rows = [json.loads(l) for l in raw_path.open(encoding="utf-8") if l.strip()]
    total = len(rows)

    kept, removed = [], []
    reasons = Counter()
    seen_norm = {}

    for rec in rows:
        cleaned, reason = clean_record(rec, args.min_chars)
        if cleaned is None:
            reasons[reason] += 1
            removed.append({"id": rec.get("id"), "reason": reason,
                            "title": rec.get("title"), "text": rec.get("text", "")[:200]})
            continue

        # 去重：归一化后完全相同的视为重复；保留先出现的
        key = normalize(cleaned["text"])
        if key in seen_norm:
            reasons["重复帖"] += 1
            removed.append({"id": cleaned["id"], "reason": f"与 {seen_norm[key]} 重复",
                            "title": cleaned["title"], "text": cleaned["text"][:200]})
            continue
        seen_norm[key] = cleaned["id"]
        kept.append(cleaned)

    with out_path.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with removed_path.open("w", encoding="utf-8") as f:
        for r in removed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- 统计 ----
    lens = sorted(r["n_chars"] for r in kept)

    def pct(p):
        if not lens:
            return 0
        return lens[min(len(lens) - 1, int(len(lens) * p))]

    src = Counter(r.get("source", "?") for r in kept)
    lines = [
        "# 清洗统计",
        "",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## 总量",
        "",
        f"- 原始记录：{total}",
        f"- 保留：**{len(kept)}**",
        f"- 丢弃：{total - len(kept)}",
        "",
        "## 丢弃原因",
        "",
        "| 原因 | 条数 |",
        "|---|---|",
    ]
    for reason, n in reasons.most_common():
        lines.append(f"| {reason} | {n} |")

    lines += [
        "",
        "## 文本长度（保留部分）",
        "",
        "| 统计量 | 字数 |",
        "|---|---|",
        f"| 最小 | {lens[0] if lens else 0} |",
        f"| P25 | {pct(0.25)} |",
        f"| 中位数 | {pct(0.5)} |",
        f"| P75 | {pct(0.75)} |",
        f"| 最大 | {lens[-1] if lens else 0} |",
        f"| 平均 | {sum(lens) // len(lens) if lens else 0} |",
        "",
        f"短于 30 字的记录：{sum(1 for x in lens if x < 30)} 条"
        f"（{100 * sum(1 for x in lens if x < 30) / len(lens):.1f}%）"
        if lens else "",
        "",
        "## 来源分布",
        "",
        "| 来源 | 条数 |",
        "|---|---|",
    ]
    for s, n in src.most_common():
        lines.append(f"| {s} | {n} |")

    lines += [
        "",
        "## 说明",
        "",
        f"被丢弃的 {len(removed)} 条完整记录见 `removed.jsonl`，含丢弃原因，可逐条回查是否误杀。",
        "过滤规则偏保守：宁可漏掉广告，也不误删真实求助帖。",
    ]

    stats_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"原始 {total} → 保留 {len(kept)}，丢弃 {total - len(kept)}")
    for reason, n in reasons.most_common():
        print(f"  - {reason}: {n}")
    print(f"\n输出：{out_path}")
    print(f"       {removed_path}（审计用）")
    print(f"       {stats_path}")


if __name__ == "__main__":
    main()
