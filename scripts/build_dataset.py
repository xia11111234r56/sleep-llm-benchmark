"""把标注好的训练集构造成 QLoRA 指令微调数据集。

关键设计：**使用和 A 组（零样本）完全相同的提示词**。
这样 E 组（微调）和 A 组（大模型零样本）的差异才只来自「有没有微调」，
而不是「提示词写法不一样」——否则 RQ2 的对照就不干净。

输出格式用 Qwen 的 chat template：
    user: <v1_zeroshot 提示词 + 正文>
    assistant: <标准 JSON>
"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ann = [json.loads(l) for l in
           (ROOT / "data/labeled/train_annot_A.jsonl").open(encoding="utf-8") if l.strip()]
    if not ann:
        raise SystemExit("没有训练标注：data/labeled/train_annot_A.jsonl")

    # 用精简提示词而不是 v1 完整版。
    # 原因：4GB 显存下 max_seq_len 只能到 512 token，而 v1 提示词光标签定义就约 600 token，
    # 会把答案挤出窗口（tokenizer 默认右截断，先切掉的就是答案）。
    # 微调场景下模型从 179 个示例里学任务，不需要把规则全写在提示里。
    template = (ROOT / "prompts/ft_compact.txt").read_text(encoding="utf-8")

    # 总字数预算。4GB 显存下 max_seq_len 只能到 512 token，Qwen 分词器约 0.67 token/汉字，
    # 700 字 ≈ 470 token，留出余量保证**答案不会被挤出窗口**。
    # 超预算时截断正文尾部，而不是让训练器从右边切——右边切先切掉的就是答案。
    BUDGET = 700
    prefix_len = len(template.replace("{{POST}}", ""))

    out_path = ROOT / "data/splits/train_instruction.jsonl"
    n = truncated = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in ann:
            post = r["text"]
            answer = {
                "attribution": r["attribution"],
                "tendency": r["tendency"],
                "evidence": r["evidence"] or "N/A",
            }
            ans_len = len(json.dumps(answer, ensure_ascii=False))
            room = BUDGET - prefix_len - ans_len - 20
            if len(post) > room:
                post = post[:max(room, 100)] + "……（原文过长已截断）"
                truncated += 1

            prompt = template.replace("{{POST}}", post)
            rec = {
                "id": r["id"],
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant",
                     "content": json.dumps(answer, ensure_ascii=False)},
                ],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"构造 {n} 条 → {out_path}")

    # 长度检查：max_seq_len 设 512，超过的会被截断，要看有多少
    lens = []
    for line in out_path.open(encoding="utf-8"):
        rec = json.loads(line)
        total = sum(len(m["content"]) for m in rec["messages"])
        lens.append(total)
    lens.sort()
    over = sum(1 for x in lens if x > 512)
    print(f"\n拼好的对话总字数：中位 {lens[len(lens)//2]}，最大 {lens[-1]}")
    print(f"构造时截断过正文的：{truncated} 条 ({truncated/n:.1%})")
    over = sum(1 for x in lens if x > 1800)
    print(f"仍超过 1800 字（约 1400 token）的：{over} 条")

    print("\n标签分布：")
    print("  归因:", dict(Counter(r["attribution"] for r in ann)))
    print("  倾向:", dict(Counter(r["tendency"] for r in ann)))


if __name__ == "__main__":
    main()
