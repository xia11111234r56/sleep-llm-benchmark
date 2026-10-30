"""找「其他」候选：描述了失眠、但没有明确归因线索的帖子。"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXTERNAL = r"考试|考研|高考|工作|加班|老板|领导|论文|答辩|面试|分手|失恋|离婚|吵架|同事|离职|失业"
BODY = r"吃药|安眠药|褪黑素|中药|西药|疼|痛|手术|激素|怀孕|经期|甲状腺|鼻炎|胃|症状"
ENV = r"噪音|室友|打呼|空调|太热|太冷|宿舍|出差|楼下|邻居|吵|声音|光"
ANX = r"想太多|胡思乱想|脑子.{0,6}(停|转|响)|越想|担心|紧张|害怕|焦虑|内耗"
BAD = r"广告|微信|淘宝|购买|咨询|疗效|噶|自杀|死法|求药|平替|开的药|换药"


def main():
    train = [json.loads(l) for l in
             (ROOT / "data/splits/train.jsonl").open(encoding="utf-8") if l.strip()]

    cands = []
    for r in train:
        t = r["text"]
        if not (60 <= r["n_chars"] <= 220) or re.search(BAD, t):
            continue
        if re.search(EXTERNAL, t) or re.search(BODY, t) or re.search(ENV, t) or re.search(ANX, t):
            continue
        if not re.search(r"失眠|睡不着|睡不好|入睡", t):
            continue
        cands.append(r)

    print(f"「其他」候选 {len(cands)} 条，列前 10 条：\n")
    for r in cands[:10]:
        print("=" * 74)
        print(f"  {r['id']}  {r['n_chars']}字")
        print(f"  {r['text'][:230]}")


if __name__ == "__main__":
    main()
