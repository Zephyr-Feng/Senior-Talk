"""
全量 EATD LoRA 推理（486 条）：为融合实验准备 P_text
- 复用 eval_lora.py 的模型加载/推理逻辑（4bit + lora_adapter_dep_v1）
- 断点续跑：已存在的 (person_id, emotion) 跳过
- 输出：training_output/eval_full/eval_full.jsonl（每行一条）
"""
import json, os, sys, csv, torch
from transformers import (
    Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
)
from peft import PeftModel

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0        # 可选：只跑前 N 条（验证用）
SHARD_ID = int(sys.argv[2]) if len(sys.argv) > 2 else 0      # 分片序号
SHARD_TOTAL = int(sys.argv[3]) if len(sys.argv) > 3 else 1   # 分片总数（并行加速）

# === 配置 ===
MODEL_PATH = "/root/autodl-tmp/.cache/modelscope/qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH = "/root/autodl-tmp/senior_project/training_output/lora_adapter_dep_v1"
BASE = "/root/autodl-tmp/senior_project"
EATD_DIR = os.path.join(BASE, "data/EATD-Corpus/EATD-Corpus")
FEATURES_CSV = os.path.join(BASE, "output/eatd_analysis/all_features.csv")
OUT_DIR = os.path.join(BASE, "training_output/eval_full")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_JSONL = os.path.join(OUT_DIR, "eval_full_shard%d.jsonl" % SHARD_ID) if SHARD_TOTAL > 1 \
    else os.path.join(OUT_DIR, "eval_full.jsonl")


def acoustic_to_text(s):
    parts = []
    sr = s["speech_rate"]
    if sr < 2.5: parts.append("语速很慢（%.1f字/秒）" % sr)
    elif sr < 4.0: parts.append("语速偏慢（%.1f字/秒）" % sr)
    elif sr < 5.5: parts.append("语速正常（%.1f字/秒）" % sr)
    else: parts.append("语速偏快（%.1f字/秒）" % sr)

    pr = s["pause_ratio"]
    if pr < 0.2: parts.append("停顿很少")
    elif pr < 0.4: parts.append("停顿略多")
    elif pr < 0.6: parts.append("停顿较多")
    else: parts.append("停顿非常多")

    pv = s["pitch_variability"]
    if pv < 20: parts.append("音调变化小，语调平淡")
    elif pv < 40: parts.append("音调变化正常")
    elif pv < 60: parts.append("音调变化较大")
    else: parts.append("音调变化显著")

    ev = s["energy_variability"]
    if ev < 0.02: parts.append("发声能量极低，说话无力")
    elif ev < 0.05: parts.append("发声能量偏低")
    elif ev < 0.08: parts.append("发声能量正常")
    else: parts.append("发声能量充足")

    jit = s["jitter"]
    if jit < 0.015: parts.append("嗓音稳定性好")
    elif jit < 0.025: parts.append("嗓音稳定性一般")
    else: parts.append("嗓音稳定性较差")

    return "，".join(parts)


def extract_conclusion(response_text):
    for line in response_text.split("\n"):
        line = line.strip()
        if line.startswith("结论："):
            return line.replace("结论：", "").strip()
    return response_text  # fallback


# === 1. 准备全量样本 ===
def sf(v):
    try: return float(v) if v and v.strip() else 0.0
    except: return 0.0

samples = []
with open(FEATURES_CSV) as f:
    for row in csv.DictReader(f):
        pid, emo = row["person_id"], row["emotion"]
        txt_path = os.path.join(EATD_DIR, pid, emo + ".txt")
        if not os.path.exists(txt_path):
            continue
        text = open(txt_path).read().strip()
        if len(text) < 5:
            continue
        samples.append({
            "person_id": pid, "emotion": emo,
            "sds": sf(row["sds"]),
            "depressed": row["depressed"].strip() == "True",
            "text": text,
            "speech_rate": sf(row["speech_rate"]),
            "pause_ratio": sf(row["pause_ratio"]),
            "pitch_variability": sf(row["pitch_variability"]),
            "energy_variability": sf(row["energy_variability"]),
            "jitter": sf(row["jitter"]),
            "shimmer": sf(row["shimmer"]),
            "voiced_ratio": sf(row["voiced_ratio"]),
            "spectral_centroid": sf(row["spectral_centroid"]),
        })

# 断点续跑：跳过已处理（兼容旧全量文件 + 本分片文件）
if LIMIT:
    samples = samples[:LIMIT]
done = set()
for cand in [os.path.join(OUT_DIR, "eval_full.jsonl"), OUT_JSONL]:
    if os.path.exists(cand):
        for line in open(cand, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["person_id"], r["emotion"]))


def in_shard(pid):
    if SHARD_TOTAL <= 1:
        return True
    return int(pid.split("_")[1]) % SHARD_TOTAL == SHARD_ID


pending = [s for s in samples if (s["person_id"], s["emotion"]) not in done and in_shard(s["person_id"])]
print("总样本: %d, 已完成: %d, 待跑: %d" % (len(samples), len(done), len(pending)))

# === 2. 加载模型 ===
print("加载模型...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, quantization_config=bnb_config,
    device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager",
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
processor = AutoProcessor.from_pretrained(ADAPTER_PATH)

system_msg = "你是一个专业的心理健康评估助手。请根据对方的语音声学特征和说话内容进行综合分析，输出推理式诊断报告。注意引用具体的说话内容和声学数值作为依据。"

# === 3. 推理（断点续跑，逐条写盘） ===
print("开始推理...")
out_fh = open(OUT_JSONL, "a", encoding="utf-8")
try:
    for i, s in enumerate(pending):
        acoustic_desc = acoustic_to_text(s)
        user_msg = "【声学特征】%s\n【说话内容】%s" % (acoustic_desc, s["text"])
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_msg}]},
            {"role": "user", "content": [{"type": "text", "text": user_msg}]},
        ]
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=200, temperature=0.1, do_sample=False,
            )
        response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        conclusion = extract_conclusion(response)
        pred_dep = "关注" in conclusion or "抑郁" in conclusion

        rec = {
            "person_id": s["person_id"], "emotion": s["emotion"],
            "sds": s["sds"], "depressed": s["depressed"],
            "conclusion": conclusion, "pred_dep": pred_dep,
            "predicted": response,
        }
        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_fh.flush()

        if (i + 1) % 10 == 0:
            print("  进度: %d/%d" % (i + 1, len(pending)), flush=True)
finally:
    out_fh.close()

# === 4. 汇总 ===
import collections
total = 0
for line in open(OUT_JSONL, encoding="utf-8"):
    r = json.loads(line)
    if r["pred_dep"] == r["depressed"]:
        total += 1
print("完成 %d 条, 当前准确率 %.1f%%" % (sum(1 for _ in open(OUT_JSONL, encoding="utf-8")), total / sum(1 for _ in open(OUT_JSONL, encoding="utf-8")) * 100))
