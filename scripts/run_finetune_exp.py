"""用微调后的模型（E 组）在测试集上推理。

⚠️ 与 A–D 的可比性说明（报告里要写）：
   E 组用的是 prompts/ft_compact.txt（精简版），不是 A–D 用的 v1 完整版。
   原因：4GB 显存下 max_seq_len 只能到 512 token，v1 光标签定义就约 600 token，
   会把答案挤出窗口。这是硬件约束，不是设计选择。

结果写入 outputs/raw_prompt_responses.jsonl，provider 记为 local-qlora、group 记为 E，
这样 eval.py 能直接分组，和 A–D 在同一张表里比较。
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from llm_client import extract_json  # noqa: E402

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def load_jsonl(p):
    p = ROOT / p
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    # 默认走 HuggingFace id；若已用 modelscope 下载到本地，用 --base <本地路径> 覆盖
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="outputs/lora_adapter")
    ap.add_argument("--prompt", default="prompts/ft_compact.txt")
    ap.add_argument("--group", default="E")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--max-input-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    gold = load_jsonl("data/splits/test_gold.jsonl")
    if not gold:
        raise SystemExit("没有 gold")
    if args.limit:
        gold = gold[:args.limit]

    out_path = ROOT / "outputs/raw_prompt_responses.jsonl"
    done = {(r.get("provider"), r.get("group"), r["id"])
            for r in load_jsonl("outputs/raw_prompt_responses.jsonl")}
    template = (ROOT / args.prompt).read_text(encoding="utf-8")

    torch.cuda.empty_cache()
    print(f"加载基座：{args.base}")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map={"": 0},
        torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, str(ROOT / args.adapter))
    model.eval()
    free, total = torch.cuda.mem_get_info()
    print(f"加载完成，显存空闲 {free/1024**3:.2f}/{total/1024**3:.2f} GB\n")

    ok = bad = 0
    for i, item in enumerate(gold, 1):
        if ("local-qlora", args.group, item["id"]) in done:
            continue
        prompt = template.replace("{{POST}}", item["text"])
        msgs = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt",
                                      truncation=True, max_length=args.max_input_tokens)
        # transformers 5.x 返回的是 BatchEncoding，不是裸 tensor，要显式取 input_ids
        try:
            input_ids = enc["input_ids"]
        except (KeyError, TypeError):
            input_ids = enc
        input_ids = input_ids.to(model.device)
        prompt_len = int(input_ids.shape[-1])

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        dt = time.perf_counter() - t0
        gen = out[0][prompt_len:]
        text = tok.decode(gen, skip_special_tokens=True)

        obj = extract_json(text)
        valid = bool(obj) and obj.get("attribution") in {
            "焦虑型", "压力型", "生理型", "环境型", "其他"} and obj.get("tendency") in {
            "有抑郁倾向", "无抑郁倾向", "无法判断"}
        reason = "" if valid else ("无法解析有效 JSON" if not obj else "标签非法")

        rec = {
            "group": args.group, "id": item["id"],
            "provider": "local-qlora", "model_requested": "qwen2.5-1.5b+lora",
            "model_returned": "qwen2.5-1.5b+lora",
            "attribution": (obj or {}).get("attribution"),
            "tendency": (obj or {}).get("tendency"),
            "evidence": (obj or {}).get("evidence"),
            "valid": valid, "invalid_reason": reason,
            "raw_response": text,
            "prompt_tokens": prompt_len,
            "completion_tokens": int(gen.shape[-1]),
            "latency_s": round(dt, 3),
            "ran_at": datetime.now().isoformat(timespec="seconds"),
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if valid:
            ok += 1
            print(f"[{i}/{len(gold)}] {item['id']} OK "
                  f"{obj.get('attribution')}/{obj.get('tendency')}  {dt:.1f}s", flush=True)
        else:
            bad += 1
            print(f"[{i}/{len(gold)}] {item['id']} BAD {reason}  输出={text[:60]!r}",
                  flush=True)

    print(f"\n完成：有效 {ok}，无效 {bad}")


if __name__ == "__main__":
    main()
