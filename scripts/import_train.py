"""录入训练集单人标注（管道分隔格式）。

格式：序号 | id | 倾向 | 归因 | evidence
注意列序是**先倾向后归因**，与标注界面的顺序一致。

校验三件事：id 必须在训练集里、标签必须合法、evidence 必须是原文子串。
上次录测试集时出过一处 id 手误（418→410），靠校验挡住的。
"""

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ATTR = {"焦虑型", "压力型", "生理型", "环境型", "其他"}
TEND = {"有抑郁倾向", "无抑郁倾向", "无法判断"}


def norm(s):
    s = "".join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in s)
    return re.sub(r"[\s\W_]+", "", s).lower()


def main():
    import datetime
    today = datetime.date.today().isoformat()

    train = {r["id"]: r for r in
             (json.loads(l) for l in
              (ROOT / "data/splits/train.jsonl").open(encoding="utf-8") if l.strip())}

    lines = [l.strip() for l in
             (ROOT / "data/labeled/_train_raw.txt").open(encoding="utf-8") if l.strip()]
    print(f"读到 {len(lines)} 行\n")

    out, problems = [], []
    for line in lines:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5:
            problems.append(f"字段不足：{line[:40]}")
            continue
        seq, rid, tend, attr, ev = parts[0], parts[1], parts[2], parts[3], "|".join(parts[4:])

        if rid not in train:
            problems.append(f"第{seq}行 {rid} 不在训练集里")
            continue
        if tend not in TEND:
            problems.append(f"{rid} 倾向非法：{tend}")
        if attr not in ATTR:
            problems.append(f"{rid} 归因非法：{attr}")

        text = train[rid]["text"]
        if ev.upper() not in ("N/A", "NA"):
            for piece in ev.replace("｜", "|").split("|"):
                p = piece.strip().strip("「」\"'")
                if p and norm(p) not in norm(text):
                    problems.append(f"{rid} evidence 不是原文子串：{p[:30]!r}")

        out.append({
            "id": rid, "attribution": attr, "tendency": tend,
            "evidence": ev, "annotator": "A", "annotated_at": today,
            "note": "", "text": text, "n_chars": train[rid]["n_chars"],
        })

    p = ROOT / "data/labeled/train_annot_A.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"写入 {len(out)} 条 → {p}")
    if problems:
        print(f"\n⚠️ {len(problems)} 处问题：")
        for x in problems:
            print("  ✗", x)
    else:
        print("✓ 全部通过校验")

    print(f"\n覆盖训练集 {len(out)}/{len(train)} 条")
    print("归因分布：")
    for k, v in Counter(r["attribution"] for r in out).most_common():
        print(f"  {k:<6} {v:>3}")
    print("倾向分布：")
    for k, v in Counter(r["tendency"] for r in out).most_common():
        print(f"  {k:<8} {v:>3}")


if __name__ == "__main__":
    main()
