"""
train_lora.py
-------------
QLoRA fine-tuning ของ Qwen2.5 สำหรับงานวิศวกร IE ด้วย PEFT + TRL SFTTrainer
รองรับ fp16 (Turing/RTX 20xx) และ gradient checkpointing สำหรับการ์ด VRAM น้อย
รัน (GPU):  python train/train_lora.py --config train/config.yaml
"""
import argparse, yaml, torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config.yaml")
    cfg = load_cfg(ap.parse_args().config)

    m, q, l, t = cfg["model"], cfg["quantization"], cfg["lora"], cfg["train"]

    # เลือก precision ตามการ์ด:
    # - Ampere+ (มี native bf16): เทรน bf16 ตรงๆ เร็วและไม่ต้องใช้ GradScaler
    # - Turing/RTX 2070 (ไม่มี native bf16): bf16 ถูก emulate -> ช้ากว่า fp16 ~4 เท่า
    #   จึงใช้ fp16 แทน. แต่ trl SFTTrainer cast LoRA adapter เป็น bf16 เสมอตอนสร้าง trainer
    #   -> ต้อง cast trainable params กลับเป็น fp32 หลังจากนั้น (ดูด้านล่าง) ไม่งั้น GradScaler
    #   ของ fp16 จะ error: "_amp_foreach_non_finite_check_and_unscale_cuda not implemented for BFloat16"
    native_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
    use_bf16 = native_bf16
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"[info] CUDA={torch.cuda.is_available()} native_bf16={native_bf16} "
          f"-> training in {'bf16' if use_bf16 else 'fp16'}")

    tokenizer = AutoTokenizer.from_pretrained(m["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = None
    if q.get("load_in_4bit"):
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        m["base_model"],
        quantization_config=bnb,
        dtype=compute_dtype,          # transformers>=5 เปลี่ยนชื่อจาก torch_dtype (ตัวเก่าถูกเพิกเฉย -> โหลดเป็น bf16 ตาม config)
        device_map="auto",
    )
    model.config.use_cache = False
    if bnb is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=t.get("gradient_checkpointing", True))

    peft_config = LoraConfig(
        r=l["r"], lora_alpha=l["alpha"], lora_dropout=l["dropout"],
        target_modules=l["target_modules"],
        bias="none", task_type="CAUSAL_LM",
    )

    ds = load_dataset("json", data_files={
        "train": f"{t['data_dir']}/train.jsonl",
        "validation": f"{t['data_dir']}/valid.jsonl",
    })

    def to_text(ex):
        return {"text": tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False)}

    ds = ds.map(to_text, remove_columns=ds["train"].column_names)

    sft_cfg = SFTConfig(
        output_dir=t["output_dir"],
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=t["lr"],
        warmup_ratio=t["warmup_ratio"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_steps=t["eval_steps"],
        eval_strategy="steps",
        fp16=not use_bf16,
        bf16=use_bf16,
        max_length=m["max_seq_len"],          # trl>=1.0 เปลี่ยนชื่อจาก max_seq_length
        dataset_text_field="text",
        optim="adamw_torch",               # LoRA trainable params เล็กมาก -> ไม่ต้อง paged/8-bit
                                           # (paged_adamw_8bit ทำให้ CPU-GPU paging thrash + step ช้าลงเรื่อยๆ)
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,           # trl>=0.12 เปลี่ยนชื่อจาก tokenizer
    )

    # fp16 path: trl cast LoRA adapter -> bf16 ตอนสร้าง trainer; ดันกลับเป็น fp32
    # (optimizer ถูกสร้างใน train() จึงจับ params เป็น fp32 -> GradScaler ของ fp16 ทำงานได้)
    if not use_bf16:
        for p in trainer.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    trainer.train()
    trainer.save_model(t["output_dir"])
    tokenizer.save_pretrained(t["output_dir"])
    print(f"Done. Adapter saved to {t['output_dir']}")


if __name__ == "__main__":
    main()
