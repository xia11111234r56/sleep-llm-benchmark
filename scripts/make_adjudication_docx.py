"""生成协商会用的 Word：4 条规则拟订 + 42 条待裁定条目。

重点是把「逐条争 42 条」变成「先定 4 条规则，再一次性裁定」——
分歧不是 42 个独立问题，而是 4 处手册空白的不同解读。
"""

import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(p):
    p = ROOT / p
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


def set_font(run, size=10.5, bold=False, name="宋体"):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)


def para(doc, text, size=10.5, bold=False, name="宋体", indent=0, after=2, color=None):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Pt(indent)
    p.paragraph_format.space_after = Pt(after)
    r = p.add_run(text)
    set_font(r, size, bold, name)
    if color:
        r.font.color.rgb = color
    return p


RULES = [
    {
        "title": "缺口 1：服药算不算「生理型」成因？",
        "n": "约 6 条",
        "conflict": [
            "A 的做法：药物必须是被帖主指为失眠成因（副作用 / 停药反弹 / 耐药）才算；"
            "正在吃安眠药属于治疗行为，不构成归因线索。",
            "B 的做法：帖子里提到服药，就算生理型线索。",
        ],
        "risk": "失眠吧里人人都在吃药。按 B 的读法，几乎全部帖子都会变成「生理型」，"
                "这个类别就失去区分度了。",
        "proposal": [
            "服药只有在被帖主**指为失眠成因**时才构成「生理型」线索，具体包括三种：",
            "　① 药物副作用导致失眠；② 停药后反弹；③ 耐药/失效导致失眠加重。",
            "单纯「我在吃安眠药 / 我去开了药 / 这药没什么用」属于**治疗行为**，不作归因线索。",
            "若全帖除用药外没有任何其他线索 → 判「其他」（而不是「生理型」）。",
        ],
        "example": "tb_10717424459、tb_10896540867、tb_9993579707",
    },
    {
        "title": "缺口 2：「无法判断」的判定依据是字数还是内容？",
        "n": "约 10 条",
        "conflict": [
            "A 的做法：30 字以上，五类标记都没出现 → 「无抑郁倾向」。",
            "B 的做法：三四十字、通篇提问 → 「无法判断」（信息不足）。",
        ],
        "risk": "手册原文写的是「正文不足 30 字」，两人都合规，但对 30–56 字的帖子读法不同。",
        "proposal": [
            "**把判定依据从「字数」改为「有无自我状态描述」。**",
            "只要帖子里能读到帖主**本人**的睡眠表现、感受或身体反应 → 按五类标记判「有 / 无」。",
            "只有完全没有自我状态描述（纯问「怎么办」「吃什么药」「有人和我一样吗」）→ 判「无法判断」。",
            "字数只作为辅助提示：不足 30 字通常没有自我描述，但不是硬规则。",
        ],
        "example": "tb_10710793619（40字）、tb_10994686566（31字）、tb_9416478444（56字）",
    },
    {
        "title": "缺口 3：躯体化的边界到哪里？",
        "n": "约 8 条",
        "conflict": [
            "A 的做法：脱发、内分泌失调、虫爬感是否算躯体化，逐条存疑、倾向从严。",
            "B 的做法：只要不是睡眠问题本身的身体症状，一律算躯体化。",
        ],
        "risk": "这是倾向任务里分歧最大的一处，直接影响「有抑郁倾向」的判定。",
        "proposal": [
            "躯体化 = 帖主**本人**描述的、**除睡眠问题以外**的身体不适。",
            "明确**算**：疼痛、心悸、头晕、胸闷、乏力、食欲体重变化、脱发、内分泌失调、"
            "皮肤蚁行感（虫爬感）、手脚冰凉。",
            "明确**不算**：失眠、早醒、多梦、白天困倦、注意力差、记性差（认知症状归入另一类，不算躯体化）。",
            "**不追问因果**：只要帖主描述了这个不适，就算躯体化，不必是失眠导致的。",
        ],
        "example": "tb_9994791633（脱发）、tb_10819421622（内分泌失调）、tb_10836518167（虫爬感）",
    },
    {
        "title": "缺口 4：「焦虑型」和「其他」怎么分？",
        "n": "约 4 条",
        "conflict": [
            "A 的做法：帖子只写「睡前想很多 / 玩手机停不下来」，按排除法兜底判「焦虑型」。",
            "B 的做法：没有明确的成因线索，判「其他」。",
        ],
        "risk": "焦虑型本来就是「排除掉外部因素后剩下的」，两人对「剩下什么算焦虑」理解不同。",
        "proposal": [
            "「焦虑型」要求帖主描述了**情绪或认知层面**的困扰：担心、紧张、害怕、"
            "胡思乱想、脑子停不下来、反刍。",
            "如果帖子只描述**行为**（睡前玩手机、看视频、主动熬夜）→ 判「生理型」（作息紊乱）。",
            "如果**完全没有成因描述**，只有睡眠表现或情绪宣泄 → 判「其他」。",
            "判「焦虑型」时，evidence 必须包含那句情绪/认知描述的原文。",
        ],
        "example": "tb_10839990109、tb_10922799011、tb_10870751154",
    },
]


def build():
    A = {r["id"]: r for r in load_jsonl("data/labeled/annot_A.jsonl")}
    B = {r["id"]: r for r in load_jsonl("data/labeled/annot_B.jsonl")}
    pool = {r["id"]: r for r in load_jsonl("data/splits/test_pool.jsonl")}
    import csv
    with (ROOT / "data/labeled/disagreements.csv").open(encoding="utf-8-sig", newline="") as f:
        diff_rows = list(csv.DictReader(f))

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "宋体"
    st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    h = doc.add_paragraph()
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(h.add_run("标注协商会材料"), 18, True, "黑体")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run("先定 4 条规则 → 再一次性裁定 42 条分歧"), 10.5)
    doc.add_paragraph()

    para(doc, "怎么用这份材料", 13, True, "黑体")
    for t in [
        "1. 这份材料的核心不是逐条吵 42 次，而是先定下后面的 4 条规则。",
        "2. 4 条规则定完，42 条绝大多数会自动有答案，剩下的再单独看。",
        "3. 定完的规则要写进标注手册 v0.2，作为以后所有标注的唯一依据。",
        "4. ⚠️ 注意：协商不会改变已经算出的 Kappa（归因 0.732 / 倾向 0.626）。",
        "   Kappa 反映的是独立标注时的一致性，是如实上报的数字，不是用来优化的指标。",
        "   协商会产出的是 gold label（每条一个最终答案），两者是两回事。",
    ]:
        para(doc, t, 10.5)

    doc.add_page_break()

    # ---------- 第一部分：规则 ----------
    para(doc, "第一部分　4 条规则的拟订方案", 16, True, "黑体", after=10)

    for i, rule in enumerate(RULES, 1):
        para(doc, f"{i}. {rule['title']}　（影响 {rule['n']}）", 12, True, "黑体", after=4)
        para(doc, "分歧在哪：", 10.5, True)
        for c in rule["conflict"]:
            para(doc, "　" + c, 10.5, indent=10)
        para(doc, "为什么要定：", 10.5, True)
        para(doc, "　" + rule["risk"], 10.5, indent=10)
        para(doc, "拟订新规则：", 10.5, True)
        for t in rule["proposal"]:
            # Word 不认 markdown 的 ** 强调，直接去掉，否则会显示成字面星号
            para(doc, "　" + t.replace("**", ""), 10.5, indent=10)
        para(doc, f"涉及条目：{rule['example']}", 9.5, color=RGBColor(0x66, 0x66, 0x66))

        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(4)
        set_font(p.add_run("　　裁定：□ 采纳拟订方案　　□ 修改（写明）："), 10.5, True)
        set_font(p.add_run("＿" * 18), 10.5)
        para(doc, "─" * 52, 8, color=RGBColor(0xAA, 0xAA, 0xAA), after=10)

    doc.add_page_break()

    # ---------- 第二部分：分歧条目 ----------
    para(doc, "第二部分　42 条分歧条目", 16, True, "黑体", after=4)
    para(doc, "按上面的规则裁定后，仍然拿不准的再逐条看。每条只需填最终结论。", 10.5, after=10)

    for n, d in enumerate(diff_rows, 1):
        rid = d["id"]
        a, b = A.get(rid, {}), B.get(rid, {})
        text = d.get("正文", "")
        full_len = pool.get(rid, {}).get("n_chars", len(text))

        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
        set_font(p.add_run(f"【{n:02d}】{rid}　·　全文 {full_len} 字"), 10.5, True, "黑体")

        para(doc, f"正文：{text[:150]}", 9.5, indent=10)

        pa = doc.add_paragraph()
        pa.paragraph_format.left_indent = Pt(10)
        pa.paragraph_format.space_after = Pt(1)
        set_font(pa.add_run("A："), 10, True)
        set_font(pa.add_run(f"{a.get('attribution','?')} / {a.get('tendency','?')}"), 10)
        if a.get("note"):
            rn = pa.add_run(f"　（{a['note'][:40]}）")
            set_font(rn, 9)
            rn.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

        pb = doc.add_paragraph()
        pb.paragraph_format.left_indent = Pt(10)
        pb.paragraph_format.space_after = Pt(2)
        set_font(pb.add_run("B："), 10, True)
        set_font(pb.add_run(f"{b.get('attribution','?')} / {b.get('tendency','?')}"), 10)
        if b.get("note"):
            rn = pb.add_run(f"　（{b['note'][:40]}）")
            set_font(rn, 9)
            rn.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        set_font(p.add_run("最终裁定：归因 "), 10, True)
        set_font(p.add_run("＿" * 6), 10)
        set_font(p.add_run("　倾向 "), 10, True)
        set_font(p.add_run("＿" * 8), 10)

        if n % 3 == 0 and n != len(diff_rows):
            doc.add_page_break()

    out = ROOT / "data/labeled/协商会材料.docx"
    doc.save(out)
    print(f"已生成：{out}")
    print(f"  4 条规则 + {len(diff_rows)} 条分歧")


if __name__ == "__main__":
    build()
