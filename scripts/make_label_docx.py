"""生成离线标注用的 Word 表格。

为什么要有这个：Streamlit 界面要求两人在同一台机器上轮流用，很难保证「独立、互不可见」。
打印或在两台电脑上各填一份，反而更容易做到真正独立——而独立正是 Kappa 的前提。

产出两份内容相同、文件名不同的文档，避免两人拿混。

用法：
    python make_label_docx.py
"""

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent

ATTR = "焦虑型 / 压力型 / 生理型 / 环境型 / 其他"
TEND = "有抑郁倾向 / 无抑郁倾向 / 无法判断"


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def set_font(run, size=10.5, bold=False, name="宋体"):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)


def build(annotator: str, items: list[dict], out_path: Path):
    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "宋体"
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    # ---------- 封面说明 ----------
    h = doc.add_paragraph()
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(h.add_run(f"失眠文本标注表　—　标注者 {annotator}"), 18, True, "黑体")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run(f"共 {len(items)} 条　·　请务必独立完成，不要与另一位标注者交流"),
             10.5, False)

    doc.add_paragraph()

    rules = [
        ("怎么判（每条的三个字段）", True),
        ("① 抑郁倾向：在原文里找五类语言标记，命中任意一类 → 有抑郁倾向；五类全无 → 无抑郁倾向；"
         "正文不足 30 字、只有标题、或纯提问无自我描述 → 无法判断。", False),
        ("　　五类标记：无望感 / 自责 / 兴趣丧失 / 自杀意念 / 躯体化（失眠以外的身体症状）。", False),
        ("　　⚠️ 「痛苦」「难受」「焦虑」不是抑郁标记；失眠本身也不算躯体化。", False),
        ("　　⚠️ 指不出具体是哪一类标记的哪一句，就不能判「有抑郁倾向」。", False),
        ("② 失眠归因：", False),
        ("　　1. 帖主明确说了因果（「因为……所以我睡不着」）→ 直接采信。", False),
        ("　　2. 没说 → 按线索优先级 压力 > 生理 > 环境 > 焦虑。"
         "焦虑型是排除掉外部因素后剩下的兜底类，不是「看起来最像」的类。", False),
        ("　　3. 多条线索并列且帖主未分主次 → 其他。　4. 零线索 → 其他。", False),
        ("③ evidence：从原文里复制一段支持你判断的话，一字不改，5–50 字。"
         "多条用 | 分隔。标「其他」零线索或「无法判断」时填 N/A。", False),
        ("", False),
        ("三条纪律", True),
        ("1. 独立标注。不要和另一位标注者讨论任何具体条目——商量的结果算不出有意义的一致性。", False),
        ("2. 拿不准的不要停，按最接近的规则硬标，然后在「备注」写一句为什么拿不准。", False),
        ("3. 只依据原文。不脑补、不推测帖主没写的东西。", False),
        ("", False),
        ("填法：在每条下方的横线上写标签名即可，例如「焦虑型」「有抑郁倾向」。", False),
    ]
    for text, bold in rules:
        if bold:
            para = doc.add_paragraph()
            set_font(para.add_run(text), 12, True, "黑体")
        else:
            para = doc.add_paragraph()
            set_font(para.add_run(text), 10.5)

    doc.add_page_break()

    # ---------- 逐条 ----------
    for n, it in enumerate(items, 1):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
        set_font(p.add_run(f"第 {n} 条　　{it['id']}　　{it['n_chars']} 字"), 11, True, "黑体")

        body = doc.add_paragraph()
        body.paragraph_format.left_indent = Pt(12)
        body.paragraph_format.space_after = Pt(4)
        set_font(body.add_run(it["text"]), 10.5)

        for label, hint in (
            ("① 抑郁倾向", TEND),
            ("② 失眠归因", ATTR),
        ):
            para = doc.add_paragraph()
            para.paragraph_format.space_after = Pt(2)
            set_font(para.add_run(f"{label}（{hint}）："), 10.5, True)
            set_font(para.add_run("＿" * 22), 10.5)

        for label in ("③ 依据 evidence", "④ 备注"):
            para = doc.add_paragraph()
            para.paragraph_format.space_after = Pt(2)
            set_font(para.add_run(f"{label}："), 10.5, True)
            set_font(para.add_run("＿" * 34), 10.5)

        # 分隔线
        sep = doc.add_paragraph()
        sep.paragraph_format.space_after = Pt(6)
        run = sep.add_run("─" * 46)
        set_font(run, 8)
        run.font.color.rgb = RGBColor(0xAA, 0xAA, 0xAA)

        # 大约每 3 条分页
        if n % 3 == 0 and n != len(items):
            doc.add_page_break()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    print(f"  {out_path.name:<28} {len(items)} 条")


def main():
    test = load_jsonl(ROOT / "data/splits/test_pool.jsonl")
    done = {r["id"] for r in load_jsonl(ROOT / "data/labeled/annot_A.jsonl")}
    todo = [r for r in test if r["id"] not in done]

    if not todo:
        raise SystemExit("没有待标的条目")

    print(f"测试集 {len(test)} 条，已标 {len(done)} 条，生成 {len(todo)} 条\n")
    out_dir = ROOT / "data/labeled"
    for who in ("A", "B"):
        build(who, todo, out_dir / f"标注表_{who}.docx")

    print("\n说明：两份内容完全相同，只是文件名不同，避免两人拿混。")
    print("     填完后需要录入回 jsonl，再算 Kappa。")


if __name__ == "__main__":
    main()
