"""
run_eval.py  (Industrial Engineering edition)
---------------------------------------------
วัดผลแบบเร็ว: (1) JSON validity (2) topic accuracy (3) numeric result accuracy
รัน:  python eval/run_eval.py --adapter outputs/qwen-business-document-lora --data data/valid.jsonl
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


def result_match(pred, gold, tol=0.02):
    """เทียบค่าตัวเลขใน result ยอมคลาดเคลื่อน 2% (เผื่อปัดเศษ)"""
    if not isinstance(pred, dict):
        return 0.0
    hit = 0
    for k, gv in gold.items():
        pv = pred.get(k)
        try:
            if abs(float(pv) - float(gv)) <= abs(float(gv)) * tol + 1e-6:
                hit += 1
        except (TypeError, ValueError):
            if str(pv) == str(gv):
                hit += 1
    return hit / max(len(gold), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="outputs/qwen-ie-engineer-lora")
    ap.add_argument("--data", default="data/valid.jsonl")
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    # Turing (RTX 20xx) ไม่รองรับ bf16 native -> เลือก fp16 อัตโนมัติ
    dtype = torch.bfloat16 if (torch.cuda.is_available()
                               and torch.cuda.is_bf16_supported(including_emulation=False)) else torch.float16
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=dtype, device_map="auto")
    model = PeftModel.from_pretrained(model, args.adapter).eval()

    # ประเมินเฉพาะตัวอย่างโหมด JSON (assistant ตอบเป็น JSON)
    rows = []
    for l in open(args.data, encoding="utf-8"):
        m = json.loads(l)["messages"]
        if m[-1]["content"].strip().startswith("{") and m[0]["content"].startswith("You are an Industrial"):
            rows.append(m)
        if len(rows) >= args.limit:
            break

    valid_json, topic_ok, result_scores = 0, 0, []
    for m in rows:
        gold = json.loads(m[-1]["content"])
        prompt = m[:-1]
        ids = tok.apply_chat_template(prompt, add_generation_prompt=True,
                                      return_tensors="pt").to(model.device)
        out = model.generate(ids, max_new_tokens=512, do_sample=False)
        text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        try:
            pred = json.loads(text)
            valid_json += 1
            topic_ok += int(pred.get("topic") == gold.get("topic"))
            result_scores.append(result_match(pred.get("result", {}), gold.get("result", {})))
        except Exception:
            result_scores.append(0.0)

    n = len(rows)
    print(f"Samples       : {n}")
    print(f"JSON validity : {valid_json}/{n} = {valid_json/n:.1%}")
    print(f"Topic acc     : {topic_ok}/{n} = {topic_ok/n:.1%}")
    print(f"Result acc    : {sum(result_scores)/n:.1%}")


if __name__ == "__main__":
    main()
