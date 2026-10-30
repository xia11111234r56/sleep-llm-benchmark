"""QLoRA 微调 Qwen2.5-1.5B-Instruct（E 组）。

针对 4GB 显存做的取舍（每一项都会写进报告的「实现细节」）：

    方案原文                实际采用            原因
    max_seq_len 1024    →  1536（先试）/ 1024  CPU 侧实测序列中位 1183 字
    batch_size 4        →  1                  显存
    grad_accum 4        →  8                  保持等效 batch 8
    lr 2e-4 / epochs 3 / LoRA r=16 alpha=32 / dropout 0.05   与方案一致

用法：
    python finetune_qlora.py --dry-run        # 只测显存，不训练
    python finetune_qlora.py                  # 正式训练
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 国内直连 HuggingFace 常超时，默认走镜像；用户显式设了就不覆盖
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 关闭 Xet 存储后端：它会把下载重定向到 cas-bridge.xethub.hf.co，
# 那个域名的证书链在本机不被信任（实测报「根证书不受信任」），关掉后走普通 HTTP。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def vram_report(tag=""):
    import torch
    if not torch.cuda.is_available():
        print("  CUDA 不可用")
        return
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    print(f"  显存 {tag}: 已分配 {alloc/1024**3:.2f}GB / 空闲 {free/1024**3:.2f}GB "
          f"/ 合计 {total/1024**3:.2f}GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID,
                    help="模型路径或 HF id。国内建议先用 modelscope 下载到本地再指过来")
    ap.add_argument("--data", default="data/splits/train_instruction.jsonl")
    ap.add_argument("--out", default="outputs/lora_adapter")
    ap.add_argument("--max-seq-len", type=int, default=1536)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条（调试）")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, DataCollatorForLanguageModeling,
                              Trainer, TrainingArguments)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用，无法微调")

    print("=" * 64)
    print(f"模型：{args.model}")
    print(f"HF_ENDPOINT = {os.environ.get('HF_ENDPOINT')}")
    vram_report("起点")

    rows = [json.loads(l) for l in
            (ROOT / args.data).open(encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    print(f"训练样本：{len(rows)} 条")
    print(f"max_seq_len = {args.max_seq_len}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- 4-bit 量化 ----
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map={"": 0},
        trust_remote_code=True, torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    vram_report("模型加载后")

    # ---- LoRA ----
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    vram_report("LoRA 注入后")

    # ---- 数据 ----
    texts = [tok.apply_chat_template(r["messages"], tokenize=False,
                                     add_generation_prompt=False) for r in rows]
    ds = Dataset.from_dict({"text": texts})

    def encode(batch):
        out = tok(batch["text"], truncation=True, max_length=args.max_seq_len,
                  padding=False)
        out["labels"] = out["input_ids"].copy()
        return out

    ds = ds.map(encode, batched=True, remove_columns=["text"])
    print(f"编码完成，{len(ds)} 条")

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # transformers 5.x 移除了 warmup_ratio，改用 warmup_steps，按 3% 换算
    total_steps = max(1, int(len(rows) * args.epochs / (args.batch * args.grad_accum)))
    warmup_steps = max(1, int(0.03 * total_steps))
    print(f"总步数约 {total_steps}，warmup {warmup_steps} 步")

    targs = TrainingArguments(
        output_dir=str(ROOT / "outputs/_qlora_ckpt"),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=2 if args.dry_run else -1,  # dry-run 只跑 2 步
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True,
        optim="paged_adamw_8bit",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=collator)

    if args.dry_run:
        print("\n" + "=" * 64)
        print("DRY RUN：跑 2 步，只验证显存够不够")
        t0 = time.time()
        try:
            trainer.train()
        except torch.cuda.OutOfMemoryError:
            print("\n❌ 显存不足（OOM）。")
            print("   依次尝试：--max-seq-len 1024 → 768 → 512")
            sys.exit(2)
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n✓ 2 步跑通，用时 {time.time()-t0:.1f}s")
        print(f"  峰值显存 {peak:.2f} GB")
        est = (time.time() - t0) / 2 * (len(rows) * args.epochs / (args.batch * args.grad_accum))
        print(f"  按此速度，全量 {args.epochs} epoch 约需 {est/60:.0f} 分钟")
        return

    print("\n" + "=" * 64)
    print("开始训练 …")
    t0 = time.time()
    trainer.train()
    dur = time.time() - t0

    out_dir = ROOT / args.out
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"\n完成，用时 {dur/60:.1f} 分钟，峰值显存 "
          f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print(f"LoRA adapter → {out_dir}")


if __name__ == "__main__":
    main()
