"""
generate_dataset.py  (Industrial Engineering edition)
-----------------------------------------------------
สร้าง synthetic dataset สำหรับ Qwen-IE-Engineer-LoRA แบบ offline (ไม่ต้องเรียก API)

แนวคิด: เรา "สร้างโจทย์ IE + เฉลย (คำนวณด้วย Python จริง)" พร้อมกัน
จึงได้คู่ (โจทย์ -> วิธีทำ + คำตอบที่ถูกต้องแม่นยำ) สำหรับ supervised fine-tuning
ครอบคลุม: Standard Time, Takt Time, Line Balancing, OEE, Manpower, Productivity + การปรับปรุง (ECRS/Lean)

Output: train.jsonl / valid.jsonl (chat format) ใช้กับ TRL SFTTrainer ได้ทันที
รัน:  python data/generate_dataset.py --n 3000 --out data
"""
import argparse, json, random, math, os

random.seed(42)

# โหมด 1: บังคับ JSON (สำหรับเสียบเข้าระบบ/SaaS)
SYSTEM_PROMPT = (
    "You are an Industrial Engineering (IE) assistant. Solve the problem and return ONLY valid JSON "
    "with keys: topic, given, formula, steps, result, recommendation. "
    "Show the correct formula and step-by-step calculation; never invent numbers not derivable from the input. "
    "Respond in the user's language."
)

# โหมด 2: แชทผู้ช่วย IE ทั่วไป (ตอบเป็นภาษาคน)
CHAT_SYSTEM_PROMPT = (
    "You are a helpful, professional Industrial Engineering assistant for factory/manufacturing. "
    "Explain IE concepts clearly, solve calculations step by step, and suggest practical improvements "
    "(ECRS, Lean, 5S, SMED). Respond in the user's language."
)

# ---------- ตัวสร้างโจทย์ IE แต่ละชนิด พร้อมเฉลยที่คำนวณจริง ----------

def gen_standard_time(lang):
    obs = [round(random.uniform(0.40, 1.80), 2) for _ in range(5)]
    avg = round(sum(obs) / len(obs), 3)
    rating = random.choice([90, 95, 100, 105, 110])        # performance rating %
    allowance = random.choice([10, 12, 15, 18])            # % allowance
    normal = round(avg * rating / 100, 3)
    std = round(normal * (1 + allowance / 100), 3)
    given = {"observed_times_min": obs, "rating_pct": rating, "allowance_pct": allowance}
    return {
        "topic": "Standard Time",
        "given": given,
        "formula": "Normal Time = Avg observed × Rating ; Standard Time = Normal Time × (1 + Allowance)",
        "steps": [
            f"Average observed time = {avg} min",
            f"Normal Time = {avg} × {rating}% = {normal} min",
            f"Standard Time = {normal} × (1 + {allowance}%) = {std} min/unit",
        ],
        "result": {"normal_time_min": normal, "standard_time_min": std},
        "recommendation": "ใช้ Standard Time นี้ตั้งเป้า output และคิด manpower; ลด allowance ได้ด้วยการปรับ ergonomics/5S",
    }, _ie_question(lang, "หาค่า Standard Time", "find the Standard Time", given)

def gen_takt_time(lang):
    avail_min = random.choice([420, 450, 480])             # นาที/กะ
    demand = random.choice([200, 300, 400, 480, 600])      # ชิ้น/กะ
    takt = round(avail_min * 60 / demand, 1)               # วินาที/ชิ้น
    given = {"available_min_per_shift": avail_min, "demand_per_shift": demand}
    return {
        "topic": "Takt Time",
        "given": given,
        "formula": "Takt Time = Available time / Customer demand",
        "steps": [
            f"Available time = {avail_min} min = {avail_min*60} sec",
            f"Takt Time = {avail_min*60} / {demand} = {takt} sec/unit",
        ],
        "result": {"takt_time_sec": takt},
        "recommendation": "ทุกสถานีต้องมี cycle time ≤ takt; ถ้าเกินคือ bottleneck ต้องปรับปรุง",
    }, _ie_question(lang, "หา Takt Time", "find the Takt Time", given)

def gen_line_balancing(lang):
    tasks = [random.randint(8, 45) for _ in range(random.randint(5, 8))]   # วินาที
    total = sum(tasks)
    takt = max(max(tasks), random.choice([45, 50, 55, 60]))                # วินาที (>= งานที่นานสุด)
    min_stations = math.ceil(total / takt)
    efficiency = round(total / (min_stations * takt) * 100, 1)
    given = {"task_times_sec": tasks, "takt_time_sec": takt}
    return {
        "topic": "Line Balancing",
        "given": given,
        "formula": "Min stations = ceil(Σtask / Takt) ; Line efficiency = Σtask / (stations × Takt)",
        "steps": [
            f"Total work content = {total} sec",
            f"Min stations = ceil({total} / {takt}) = {min_stations}",
            f"Line efficiency = {total} / ({min_stations} × {takt}) = {efficiency}%",
        ],
        "result": {"total_work_sec": total, "min_stations": min_stations, "line_efficiency_pct": efficiency},
        "recommendation": "ใช้ ECRS รวม/ย้ายงานให้แต่ละสถานีใกล้ takt เพื่อดัน efficiency ขึ้น และลดจำนวนคน",
    }, _ie_question(lang, "จัดสมดุลสายการผลิต (line balancing)", "balance the line", given)

def gen_oee(lang):
    planned = random.choice([420, 450, 480])
    downtime = random.choice([20, 30, 45, 60])
    run = planned - downtime
    availability = run / planned
    ideal_cycle = round(random.uniform(0.5, 1.5), 2)       # นาที/ชิ้น
    total_count = random.randint(int(run / ideal_cycle * 0.7), int(run / ideal_cycle * 0.98))
    performance = min((ideal_cycle * total_count) / run, 1.0)
    defects = random.randint(2, max(3, total_count // 20))
    quality = (total_count - defects) / total_count
    oee = availability * performance * quality
    given = {"planned_min": planned, "downtime_min": downtime,
             "ideal_cycle_min": ideal_cycle, "total_count": total_count, "defects": defects}
    return {
        "topic": "OEE",
        "given": given,
        "formula": "OEE = Availability × Performance × Quality",
        "steps": [
            f"Availability = ({planned}-{downtime})/{planned} = {round(availability*100,1)}%",
            f"Performance = ({ideal_cycle}×{total_count})/{run} = {round(performance*100,1)}%",
            f"Quality = ({total_count}-{defects})/{total_count} = {round(quality*100,1)}%",
            f"OEE = {round(availability*100,1)}% × {round(performance*100,1)}% × {round(quality*100,1)}% = {round(oee*100,1)}%",
        ],
        "result": {"availability_pct": round(availability*100,1), "performance_pct": round(performance*100,1),
                   "quality_pct": round(quality*100,1), "oee_pct": round(oee*100,1)},
        "recommendation": "โฟกัสองค์ประกอบที่ต่ำสุดก่อน (เช่น Availability ต่ำ -> ลด downtime ด้วย SMED/PM)",
    }, _ie_question(lang, "คำนวณ OEE", "calculate OEE", given)

def gen_manpower(lang):
    demand = random.choice([500, 800, 1000, 1200])
    std_time = round(random.uniform(0.5, 2.0), 2)          # นาที/ชิ้น
    avail_min = random.choice([420, 450, 480])
    efficiency = random.choice([0.85, 0.90, 0.95])
    required = math.ceil(demand * std_time / (avail_min * efficiency))
    given = {"demand_per_shift": demand, "standard_time_min": std_time,
             "available_min": avail_min, "efficiency": efficiency}
    return {
        "topic": "Manpower Requirement",
        "given": given,
        "formula": "Manpower = ceil(Demand × Std time / (Available time × Efficiency))",
        "steps": [
            f"Workload = {demand} × {std_time} = {round(demand*std_time,1)} min",
            f"Effective time/person = {avail_min} × {efficiency} = {round(avail_min*efficiency,1)} min",
            f"Manpower = ceil({round(demand*std_time,1)} / {round(avail_min*efficiency,1)}) = {required} คน",
        ],
        "result": {"required_manpower": required},
        "recommendation": "ถ้าต้องลดคน ให้ลด std time ด้วยการปรับวิธีทำงาน (ECRS) หรือ automation จุด bottleneck",
    }, _ie_question(lang, "คำนวณจำนวนคนที่ต้องใช้ (manpower)", "calculate required manpower", given)

def gen_productivity(lang):
    output = random.randint(800, 2500)
    labor_hours = random.randint(40, 200)
    prod = round(output / labor_hours, 2)
    given = {"output_units": output, "labor_hours": labor_hours}
    return {
        "topic": "Labor Productivity",
        "given": given,
        "formula": "Productivity = Output / Input (labor hours)",
        "steps": [
            f"Productivity = {output} / {labor_hours} = {prod} units/man-hour",
        ],
        "result": {"productivity_units_per_manhour": prod},
        "recommendation": "เทียบกับ baseline; เพิ่ม productivity ด้วยการลด waste (Lean 7 wastes) และ line balancing",
    }, _ie_question(lang, "คำนวณ productivity", "calculate labor productivity", given)

GENERATORS = [gen_standard_time, gen_takt_time, gen_line_balancing,
              gen_oee, gen_manpower, gen_productivity]


def _ie_question(lang, th_task, en_task, given):
    g = json.dumps(given, ensure_ascii=False)
    if lang == "th":
        return f"ช่วย{th_task}จากข้อมูลนี้: {g}"
    return f"Please {en_task} from this data: {g}"


# ---------- การ render เฉลยเป็นภาษาคน / JSON ----------

def _solution_text(gt, lang):
    steps = "\n".join(f"  {i+1}) {s}" for i, s in enumerate(gt["steps"]))
    if lang == "th":
        return (f"หัวข้อ: {gt['topic']}\n"
                f"สูตร: {gt['formula']}\n"
                f"วิธีคิด:\n{steps}\n"
                f"คำตอบ: {json.dumps(gt['result'], ensure_ascii=False)}\n"
                f"ข้อเสนอแนะ: {gt['recommendation']}")
    return (f"Topic: {gt['topic']}\n"
            f"Formula: {gt['formula']}\n"
            f"Steps:\n{steps}\n"
            f"Answer: {json.dumps(gt['result'], ensure_ascii=False)}\n"
            f"Recommendation: {gt['recommendation']}")


# ---------- แชททั่วไป / ความรู้ IE (กัน catastrophic forgetting) ----------

GENERAL_CHATS = [
    ("th", "สวัสดีครับ ช่วยอะไรได้บ้าง",
     "สวัสดีครับ ผมเป็นผู้ช่วยวิศวกร IE ช่วยคำนวณ Standard Time, Takt Time, Line Balancing, OEE, "
     "จำนวนคน (manpower), productivity และแนะนำการปรับปรุงด้วย ECRS/Lean ได้ครับ บอกโจทย์มาได้เลย"),
    ("en", "Hi, what can you do?",
     "Hi! I'm an Industrial Engineering assistant. I can compute Standard Time, Takt Time, Line "
     "Balancing, OEE, manpower, and productivity, and suggest improvements (ECRS, Lean, SMED). "
     "What's your problem?"),
    ("th", "ECRS คืออะไร",
     "ECRS คือหลักปรับปรุงงาน 4 ขั้น: Eliminate (กำจัดงานที่ไม่จำเป็น), Combine (รวมขั้นตอน), "
     "Rearrange (จัดลำดับใหม่), Simplify (ทำให้ง่ายขึ้น) ใช้ลดเวลาทำงานและ waste ในไลน์ผลิต"),
    ("th", "takt time ต่างกับ cycle time ยังไง",
     "Takt time = จังหวะที่ลูกค้าต้องการ (Available time/Demand) เป็นเป้า ส่วน cycle time = เวลาจริง"
     "ที่สถานีหนึ่งใช้ผลิต 1 ชิ้น เป้าหมายคือทำให้ cycle time ≤ takt time ทุกสถานี"),
    ("en", "What is OEE?",
     "OEE (Overall Equipment Effectiveness) = Availability × Performance × Quality. "
     "It measures how effectively equipment is used; world-class is ~85%."),
]


def _multi_turn(lang):
    """แชทโต้ตอบหลายเทิร์นเกี่ยวกับโจทย์ IE เดิม"""
    gt, q = gen_line_balancing(lang)
    if lang == "th":
        msgs = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": _solution_text(gt, lang)},
            {"role": "user", "content": "ถ้าอยากเพิ่ม line efficiency ควรทำยังไง"},
            {"role": "assistant", "content":
                "แนวทางเพิ่ม efficiency:\n- ใช้ ECRS รวม/ย้ายงานให้แต่ละสถานีเวลาใกล้ takt\n"
                "- ปรับปรุงสถานี bottleneck (งานที่นานสุด) ก่อน\n- พิจารณาแบ่งงานที่นานเป็นงานย่อย"},
            {"role": "user", "content": "bottleneck คือสถานีไหน"},
            {"role": "assistant", "content":
                f"Bottleneck คือสถานีที่ใช้เวลานานสุด = {max(gt['given']['task_times_sec'])} วินาที "
                f"ควรปรับปรุงจุดนี้ก่อนเพราะเป็นตัวจำกัดกำลังผลิตของทั้งไลน์"},
        ]
    else:
        msgs = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": _solution_text(gt, lang)},
            {"role": "user", "content": "How can I improve the line efficiency?"},
            {"role": "assistant", "content":
                "To improve efficiency:\n- Apply ECRS to combine/move tasks closer to takt\n"
                "- Improve the bottleneck station (longest task) first\n- Split the longest task into sub-tasks"},
            {"role": "user", "content": "Which station is the bottleneck?"},
            {"role": "assistant", "content":
                f"The bottleneck is the longest task = {max(gt['given']['task_times_sec'])} sec. "
                f"Improve it first since it limits the whole line's throughput."},
        ]
    return {"messages": msgs}


def _compare(lang):
    """เปรียบเทียบ 2 สถานการณ์ (เช่น 2 แผนจัดไลน์) แล้วเลือกที่ดีกว่า"""
    a, _ = gen_line_balancing(lang)
    b, _ = gen_line_balancing(lang)
    ea, eb = a["result"]["line_efficiency_pct"], b["result"]["line_efficiency_pct"]
    better = "A" if ea >= eb else "B"
    if lang == "th":
        user = ("เปรียบเทียบแผนจัดไลน์ 2 แบบนี้ ควรเลือกแบบไหน\n\n"
                f"[แผน A] งาน(วินาที)={a['given']['task_times_sec']} takt={a['given']['takt_time_sec']}\n"
                f"[แผน B] งาน(วินาที)={b['given']['task_times_sec']} takt={b['given']['takt_time_sec']}")
        ans = (f"แผน A: {a['result']['min_stations']} สถานี, efficiency {ea}%\n"
               f"แผน B: {b['result']['min_stations']} สถานี, efficiency {eb}%\n"
               f"คำแนะนำ: เลือกแผน {better} เพราะ line efficiency สูงกว่า ใช้คนคุ้มกว่า")
    else:
        user = ("Compare these two line plans. Which should I choose?\n\n"
                f"[Plan A] tasks(sec)={a['given']['task_times_sec']} takt={a['given']['takt_time_sec']}\n"
                f"[Plan B] tasks(sec)={b['given']['task_times_sec']} takt={b['given']['takt_time_sec']}")
        ans = (f"Plan A: {a['result']['min_stations']} stations, efficiency {ea}%\n"
               f"Plan B: {b['result']['min_stations']} stations, efficiency {eb}%\n"
               f"Recommendation: choose Plan {better} — higher line efficiency, better labor utilization.")
    return {"messages": [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": ans},
    ]}


def build_example():
    """multi-task: คำนวณ(JSON) + อธิบาย(ภาษาคน) + แชทหลายเทิร์น + เปรียบเทียบ + ความรู้ทั่วไป"""
    roll = random.random()

    # 1) ความรู้/แชททั่วไป ~10%
    if roll < 0.10:
        lang, u, a = random.choice(GENERAL_CHATS)
        return {"messages": [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": u},
            {"role": "assistant", "content": a},
        ]}

    lang = random.choice(["th", "en"])

    # 2) เปรียบเทียบ 2 แผน ~12%
    if roll < 0.22:
        return _compare(lang)

    # 3) แชทหลายเทิร์น ~13%
    if roll < 0.35:
        return _multi_turn(lang)

    gt, q = random.choice(GENERATORS)(lang)

    # 4) โหมด JSON (เสียบเข้าระบบ) ~32%
    if roll < 0.67:
        return {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": json.dumps(gt, ensure_ascii=False)},
        ]}

    # 5) โหมดอธิบายเป็นภาษาคน (สูตร+วิธีคิด+คำตอบ+ข้อเสนอแนะ) ~33%
    return {"messages": [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": q},
        {"role": "assistant", "content": _solution_text(gt, lang)},
    ]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default="data")
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = [build_example() for _ in range(args.n)]
    random.shuffle(data)
    n_valid = int(args.n * args.valid_ratio)
    valid, train = data[:n_valid], data[n_valid:]

    for name, rows in [("train.jsonl", train), ("valid.jsonl", valid)]:
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(train)} train / {len(valid)} valid -> {args.out}/")


if __name__ == "__main__":
    main()
