---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: peft
tags:
  - industrial-engineering
  - manufacturing
  - lean
  - oee
  - line-balancing
  - work-study
  - lora
language: [en, th]
pipeline_tag: text-generation
---

# Qwen-IE-Engineer-LoRA 🏭

**🤗 Model weights:** [nattakit2580/qwen-ie-engineer-lora](https://huggingface.co/nattakit2580/qwen-ie-engineer-lora) · **💻 Code:** [github.com/nattakit2580/qwen-ie-engineer-lora](https://github.com/nattakit2580/qwen-ie-engineer-lora)

> Enterprise-grade LoRA adapter that turns Qwen2.5-7B into an **Industrial Engineering (IE)
> assistant** for factory / manufacturing. It solves IE calculations step by step and
> suggests practical improvements (ECRS, Lean, SMED). Bilingual (EN/TH).
> Deployable on-prem / VPC for data privacy.

## Capabilities
- ✅ **Standard Time** (observed time × rating × allowance)
- ✅ **Takt Time** (available time / demand)
- ✅ **Line Balancing** (min stations, line efficiency, bottleneck)
- ✅ **OEE** (Availability × Performance × Quality)
- ✅ **Manpower requirement** & **Labor productivity**
- ✅ Improvement advice (ECRS, Lean 7 wastes, 5S, SMED)
- ✅ Two modes: **JSON** (system integration) and **natural chat** (incl. multi-turn)

## Out of Scope
- ❌ Safety-critical decisions without engineer review (assistive only)
- ❌ Fabricating numbers not derivable from the given data
- ❌ Replacing a licensed PE / formal validation

## Output Format (JSON mode)
`topic`, `given`, `formula`, `steps`, `result`, `recommendation`

## Modes
| Use | system prompt | output |
|-----|---------------|--------|
| API / app backend | `SYSTEM_PROMPT` | valid JSON |
| Chat / assistant   | `CHAT_SYSTEM_PROMPT` | natural language, multi-turn |

---

## Reproduce / Train it yourself

```bash
# 0) ติดตั้ง (แนะนำ Python 3.10/3.11 + CUDA GPU)
pip install -r requirements.txt

# 1) สร้าง synthetic IE dataset (offline, ground truth คำนวณด้วย Python)
python data/generate_dataset.py --n 3000 --out data

# 2) เทรน QLoRA (GPU 16-24GB)
python train/train_lora.py --config train/config.yaml

# 3) ทดสอบ
python inference/infer.py --adapter outputs/qwen-ie-engineer-lora

# 4) วัดผล
python eval/run_eval.py --adapter outputs/qwen-ie-engineer-lora --data data/valid.jsonl
```

## Limitations & Safety
Assistive tool — **human-in-the-loop required**. An LLM can make arithmetic slips;
the model is trained to *show its formula and steps* so an engineer can verify.
For production, pair JSON output with a calculator/validation layer.
Training data is fully synthetic (no confidential factory data).

## License
Apache-2.0. Built with PEFT/LoRA.
