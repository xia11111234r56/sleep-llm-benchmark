"""生成训练集的离线标注文档（紧凑版）。

训练集是单人标注、只供微调用，不需要双人独立，所以文档比测试集那份精简：
说明压到半页，每条只留三行填写位，4 条一页。

用法：
    python make_train_docx.py
"""

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
PER_PAGE = 4


def load_jsonl(p):
    p = ROOT / p
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


def set_font(run, size=10, bold=False, name="宋体"):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)


def build(items, out_path):
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "宋体"
    st.font.size = Pt(10)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    # ---- 说明（压缩成半页）----
    h = doc.add_paragraph()
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(h.add_run(f"训练集标注　共 {len(items)} 条"), 16, True, "黑体")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run("按标注手册 v0.2 判定　·　单人标注"), 9.5)
    doc.add_paragraph()

    for text, bold in [
        ("① 抑郁倾向（三选一）", True),
        ("在原文找五类标记：无望感 / 自责 / 兴趣丧失 / 自杀意念 / 躯体化。", False),
        ("命中任一 → 有抑郁倾向；全无 → 无抑郁倾向。", False),
        ("· 躯体化 = 本人描述的、失眠以外的身体不适（疼痛/心悸/头晕/脱发/内分泌/蚁行感…）。"
         "命中即判「有」，不必再有无望自责。认知症状（健忘/迟钝/口吃）不算。", False),
        ("· 完全没有任何自我状态描述（纯问怎么办、吃什么药）→ 无法判断。不看字数，看内容。", False),
        ("· 「痛苦」「难受」「焦虑」不是标记；失眠本身不算躯体化。", False),
        ("", False),
        ("② 失眠归因（五选一）", True),
        ("1. 帖主明确说了因果（因为…所以睡不着）→ 直接采信。", False),
        ("2. 没说 → 压力 > 生理 > 环境 > 焦虑（焦虑是排除后的兜底）。", False),
        ("3. 多条并列无主次 → 其他；零线索 → 其他。", False),
        ("· 服药只在被指为成因时才算生理型（副作用/停药反弹/耐药）；单纯吃药属治疗行为，不算。", False),
        ("· 行为性晚睡（玩手机/主动熬夜）→ 生理型（作息紊乱）；想太多/担心 → 焦虑型。", False),
        ("", False),
        ("③ 依据 evidence", True),
        ("从原文复制一句，一字不改，5–50 字。不适用写 N/A。", False),
        ("", False),
        ("填写方式：在横线上直接写标签名，例如「焦虑型」「有抑郁倾向」。", False),
    ]:
        para = doc.add_paragraph()
        para.paragraph_format.space_after = Pt(1)
        set_font(para.add_run(text), 10 if bold else 9.5, bold, "黑体" if bold else "宋体")

    doc.add_page_break()

    # ---- 条目 ----
    for n, it in enumerate(items, 1):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(1)
        set_font(p.add_run(f"【{n}】{it['id']}　·　{it['n_chars']} 字"), 10, True, "黑体")

        body = doc.add_paragraph()
        body.paragraph_format.left_indent = Pt(10)
        body.paragraph_format.space_after = Pt(2)
        set_font(body.add_run(it["text"]), 9.5)

        for label in ("倾向", "归因"):
            para = doc.add_paragraph()
            para.paragraph_format.left_indent = Pt(10)
            para.paragraph_format.space_after = Pt(1)
            set_font(para.add_run(f"{label}："), 9.5, True)
            set_font(para.add_run("＿" * 18), 9.5)

        para = doc.add_paragraph()
        para.paragraph_format.left_indent = Pt(10)
        para.paragraph_format.space_after = Pt(1)
        set_font(para.add_run("依据："), 9.5, True)
        set_font(para.add_run("＿" * 32), 9.5)

        sep = doc.add_paragraph()
        sep.paragraph_format.space_after = Pt(2)
        r = sep.add_run("·" * 40)
        set_font(r, 7)
        r.font.color.rgb = RGBColor(0xBB, 0xBB, 0xBB)

        if n % PER_PAGE == 0 and n != len(items):
            doc.add_page_break()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    print(f"已生成：{out_path}　（{len(items)} 条，约 {(len(items)//PER_PAGE)+2} 页）")


def main():
    pool = load_jsonl("data/splits/train.jsonl")
    done = {r["id"] for r in load_jsonl("data/labeled/train_annot_A.jsonl")}
    todo = [r for r in pool if r["id"] not in done]
    if not todo:
        raise SystemExit("训练集已全部标注")
    print(f"训练集 {len(pool)} 条，已标 {len(done)} 条，待标 {len(todo)} 条")
    build(todo, ROOT / "data/labeled/训练集标注表.docx")


if __name__ == "__main__":
    main()
