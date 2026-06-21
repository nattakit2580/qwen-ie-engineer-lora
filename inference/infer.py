"""
infer.py
--------
โหลด base model + LoRA adapter แล้วทดสอบกับเอกสารตัวอย่าง
รัน:  python inference/infer.py --adapter outputs/qwen-business-document-lora
"""
import argparse, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SYSTEM_PROMPT = (
    "You are an Industrial Engineering (IE) assistant. Solve the problem and return ONLY valid JSON "
    "with keys: topic, given, formula, steps, result, recommendation. "
    "Show the correct formula and step-by-step calculation; never invent numbers not derivable from the input. "
    "Respond in the user's language."
)

SAMPLE = ('ช่วยคำนวณ OEE จากข้อมูลนี้: '
          '{"planned_min": 480, "downtime_min": 30, "ideal_cycle_min": 0.6, '
          '"total_count": 700, "defects": 15}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="outputs/qwen-ie-engineer-lora")
    ap.add_argument("--doc", dest="prompt", default=SAMPLE)
    args = ap.parse_args()

    # Turing (RTX 20xx) ไม่รองรับ bf16 native -> เลือก fp16 อัตโนมัติ
    dtype = torch.bfloat16 if (torch.cuda.is_available()
                               and torch.cuda.is_bf16_supported(including_emulation=False)) else torch.float16
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=dtype, device_map="auto")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": args.prompt},
    ]
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                     return_tensors="pt").to(model.device)
    out = model.generate(inputs, max_new_tokens=512, do_sample=False)
    text = tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    print("=== RAW OUTPUT ===\n", text)
    try:
        print("\n=== PARSED JSON ===\n", json.dumps(json.loads(text), ensure_ascii=False, indent=2))
    except Exception as e:
        print("\n[warn] not valid JSON:", e)


if __name__ == "__main__":
    main()
