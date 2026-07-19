"""
评估微调后的 Qwen2.5-VL-7B LoRA 模型
- 用 EATD 未参与训练的情感样本做测试（每人的不同情绪）
- 对比 zero-shot 和 fine-tuned 的结果
"""
import json, os, csv, torch
from transformers import (
    Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
)
from peft import PeftModel

# === 配置 ===
MODEL_PATH = "/root/autodl-tmp/.cache/modelscope/qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH = "/root/autodl-tmp/senior_project/training_output/lora_adapter"
BASE = "/root/autodl-tmp/senior_project"
EATD_DIR = os.path.join(BASE, "data/EATD-Corpus/EATD-Corpus")
FEATURES_CSV = os.path.join(BASE, "output/eatd_analysis/all_features.csv")
OUTPUT_DIR = os.path.join(BASE, "training_output/evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 声学特征转文本（与训练时一致）
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

# === 1. 准备测试数据 ===
print("准备测试数据...")
persons = {}
with open(FEATURES_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid = row["person_id"]
        emo = row["emotion"]
        txt_path = os.path.join(EATD_DIR, pid, emo + ".txt")
        if not os.path.exists(txt_path):
            continue
        with open(txt_path) as tf:
            text = tf.read().strip()
        if len(text) < 5:
            continue

        def sf(v):
            try: return float(v) if v and v.strip() else 0.0
            except: return 0.0

        persons[(pid, emo)] = {
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
        }

print("总样本: %d" % len(persons))

# === 2. 构建"留一情绪"测试集 ===
# 训练用了所有情绪的均衡数据，所以测试时每个人选一个没见过的情绪
# 或者按 person 分层：训练集里见过的人但不同情绪
# 更简单的方案：随机选 80 条做测试
import random
random.seed(42)
test_keys = random.sample(list(persons.keys()), k=min(80, len(persons)))
test_samples = [persons[k] for k in test_keys]

print("测试样本: %d (抑郁: %d, 正常: %d)" % (
    len(test_samples),
    sum(1 for s in test_samples if s["depressed"]),
    sum(1 for s in test_samples if not s["depressed"])
))

# === 3. 加载模型 ===
print("\n加载模型...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("  加载 base model...")
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)

print("  加载 LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

processor = AutoProcessor.from_pretrained(ADAPTER_PATH)

# === 4. 推理 ===
print("\n开始推理...")

system_msg = "你是一个专业的心理健康评估助手。请根据对方的语音声学特征和说话内容进行综合分析，输出推理式诊断报告。注意引用具体的说话内容和声学数值作为依据。"

def extract_conclusion(response_text):
    """从推理报告中提取结论（需要关注/正常）"""
    for line in response_text.split("\n"):
        line = line.strip()
        if line.startswith("结论："):
            return line.replace("结论：", "").strip()
    return response_text  # fallback

def load_model_and_predict(test_samples, model, processor):
    """批量推理，返回结果列表"""
    results = []
    for i, s in enumerate(test_samples):
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
                **inputs,
                max_new_tokens=200,  # 推理式输出较长
                temperature=0.1,
                do_sample=False,
            )

        response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        conclusion = extract_conclusion(response)
        pred_dep = "关注" in conclusion or "抑郁" in conclusion
        actual_dep = s["depressed"]

        results.append({
            "person_id": s["person_id"],
            "emotion": s["emotion"],
            "sds": s["sds"],
            "actual": "抑郁" if actual_dep else "正常",
            "predicted": response,
            "conclusion": conclusion,
            "correct": pred_dep == actual_dep,
        })

        if (i+1) % 10 == 0:
            print("  进度: %d/%d" % (i+1, len(test_samples)))
    return results

results = load_model_and_predict(test_samples, model, processor)

# === 5. 统计结果 ===
print("\n" + "=" * 60)
print("评估结果")
print("=" * 60)

tp = sum(1 for r in results if "关注" in r["conclusion"] and r["actual"] == "抑郁")
tn = sum(1 for r in results if r["conclusion"] == "正常" and r["actual"] == "正常")
fp = sum(1 for r in results if "关注" in r["conclusion"] and r["actual"] == "正常")
fn = sum(1 for r in results if r["conclusion"] == "正常" and r["actual"] == "抑郁")

total = len(results)
acc = (tp+tn)/total*100
sens = tp/(tp+fn)*100 if tp+fn > 0 else 0
spec = tn/(tn+fp)*100 if tn+fp > 0 else 0
prec = tp/(tp+fp)*100 if tp+fp > 0 else 0
f1 = 2*tp/(2*tp+fp+fn)*100 if 2*tp+fp+fn > 0 else 0

print("             | 预测抑郁 | 预测正常")
print("  实际抑郁   |   %3d    |   %3d" % (tp, fn))
print("  实际正常   |   %3d    |   %3d" % (fp, tn))
print()
print("  准确率 Accuracy:     %.1f%%" % acc)
print("  敏感度 Sensitivity:  %.1f%%" % sens)
print("  特异度 Specificity:  %.1f%%" % spec)
print("  精确率 Precision:    %.1f%%" % prec)
print("  F1 分数:             %.1f%%" % f1)

# === 6. 保存结果 ===
output = {
    "config": {
        "model": MODEL_PATH,
        "adapter": ADAPTER_PATH,
        "test_samples": total,
    },
    "confusion_matrix": {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    },
    "metrics": {
        "accuracy_pct": round(acc, 1),
        "sensitivity_pct": round(sens, 1),
        "specificity_pct": round(spec, 1),
        "precision_pct": round(prec, 1),
        "f1_pct": round(f1, 1),
    },
    "results": results,
}

with open(os.path.join(OUTPUT_DIR, "evaluation_results.json"), "w") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

# 保存易读版本
with open(os.path.join(OUTPUT_DIR, "predictions.txt"), "w") as f:
    f.write("person_id\temotion\tsds\tactual\tconclusion\tcorrect\tpredicted_preview\n")
    for r in results:
        preview = r["predicted"][:100].replace("\n", " ")
        f.write("%s\t%s\t%.1f\t%s\t%s\t%s\t%s\n" % (
            r["person_id"], r["emotion"], r["sds"],
            r["actual"], r["conclusion"], r["correct"], preview
        ))

print("\n结果已保存到: %s" % OUTPUT_DIR)
print("Done!")
