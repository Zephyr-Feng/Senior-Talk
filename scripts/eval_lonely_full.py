"""
孤独倾向 LoRA 全量评估
- 模型: lora_adapter_lonely_v1（4bit + 推理式报告，结论行"孤独倾向：明显/不明显"）
- 评估集:
    heldout: training_data_lonely/holdout.jsonl（30+30 留出，主评估）
    eatd:    EATD 479 条真实语料（预期全"不明显"→ 误报检查）
- 断点续跑：已评估的文本跳过
- 用法: python eval_lonely_full.py heldout | eatd [shard_id shard_total]
"""
import json, os, sys, csv, zlib, torch
from transformers import (
    Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
)
from peft import PeftModel

SET = sys.argv[1] if len(sys.argv) > 1 else "heldout"
SHARD_ID = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SHARD_TOTAL = int(sys.argv[3]) if len(sys.argv) > 3 else 1

# === 配置 ===
MODEL_PATH = "/root/autodl-tmp/.cache/modelscope/qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH = "/root/autodl-tmp/senior_project/training_output/lora_adapter_lonely_v1"
BASE = "/root/autodl-tmp/senior_project"
EATD_DIR = os.path.join(BASE, "data/EATD-Corpus/EATD-Corpus")
FEATURES_CSV = os.path.join(BASE, "output/eatd_analysis/all_features.csv")
HOLDOUT_JSONL = os.path.join(BASE, "training_data_lonely/holdout.jsonl")
OUT_DIR = os.path.join(BASE, "training_output/eval_lonely")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_JSONL = os.path.join(OUT_DIR, "eval_lonely_%s_shard%d.jsonl" % (SET, SHARD_ID)) if SHARD_TOTAL > 1 \
    else os.path.join(OUT_DIR, "eval_lonely_%s.jsonl" % SET)


def extract_lonely_conclusion(response_text):
    for line in response_text.split("\n"):
        line = line.strip()
        if line.startswith("孤独倾向："):
            return line.replace("孤独倾向：", "").strip()
    return response_text  # fallback


def parse_lonely_pred(conclusion):
    # 精确匹配："不明显" 包含子串 "明显"，必须排除（2026-08-07 修复："明显" in "不明显" 为 True 导致全部误报）
    c = conclusion.strip()
    if c == "明显":
        return True
    if c == "不明显":
        return False
    # 容错：正常形式如"无明显孤独迹象"→ False；只有"明显"打头且非"不明显"才判正
    return c.startswith("明显") and not c.startswith("不明显")


# === 1. 准备评估样本 ===
samples = []
if SET == "heldout":
    with open(HOLDOUT_JSONL) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            samples.append({"key": "h%d" % len(samples), "text": d["text"], "lonely": d["lonely"], "dims": d.get("lonely_dims", [])})
    print("评估集 heldout: %d 条" % len(samples))
elif SET == "eatd":
    with open(FEATURES_CSV) as f:
        for row in csv.DictReader(f):
            pid, emo = row["person_id"], row["emotion"]
            txt_path = os.path.join(EATD_DIR, pid, emo + ".txt")
            if not os.path.exists(txt_path):
                continue
            text = open(txt_path).read().strip()
            if len(text) < 5:
                continue
            samples.append({"key": "%s_%s" % (pid, emo), "text": text, "lonely": False, "dims": []})
    print("评估集 eatd: %d 条（预期全部孤独倾向不明显）" % len(samples))
else:
    print("未知评估集: %s（可选 heldout / eatd）" % SET)
    sys.exit(1)


def in_shard(key):
    if SHARD_TOTAL <= 1:
        return True
    # crc32 跨进程稳定（hash() 有随机种子，分片会不一致）
    return zlib.crc32(key.encode("utf-8")) % SHARD_TOTAL == SHARD_ID


# 断点续跑
done = set()
for cand in [os.path.join(OUT_DIR, "eval_lonely_%s.jsonl" % SET), OUT_JSONL]:
    if os.path.exists(cand):
        for line in open(cand, encoding="utf-8"):
            done.add(json.loads(line)["key"])
pending = [s for s in samples if s["key"] not in done and in_shard(s["key"])]
print("已完成: %d, 待跑: %d" % (len(done), len(pending)))

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

system_msg = "你是一个专业的心理健康评估助手。请根据对方说话的内容判断其是否存在孤独倾向，输出简短分析并给出明确结论。"

# === 3. 推理 ===
print("开始推理...")
out_fh = open(OUT_JSONL, "a", encoding="utf-8")
try:
    for i, s in enumerate(pending):
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_msg}]},
            {"role": "user", "content": [{"type": "text", "text": "【说话内容】%s" % s["text"]}]},
        ]
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=120, temperature=0.1, do_sample=False)
        response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        conclusion = extract_lonely_conclusion(response)
        pred_lonely = parse_lonely_pred(conclusion)

        rec = {
            "key": s["key"], "text": s["text"],
            "lonely": s["lonely"], "dims": s["dims"],
            "conclusion": conclusion, "pred_lonely": pred_lonely,
            "predicted": response,
        }
        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_fh.flush()
        if (i + 1) % 10 == 0:
            print("  进度: %d/%d" % (i + 1, len(pending)), flush=True)
finally:
    out_fh.close()

# === 4. 汇总 ===
recs = []
for line in open(OUT_JSONL, encoding="utf-8"):
    recs.append(json.loads(line))
# 用结论原文重新解析（jsonl 中旧 pred_lonely 可能含子串匹配 bug）
for r in recs:
    r["pred_lonely"] = parse_lonely_pred(r.get("conclusion", r.get("predicted", "")))
tp = sum(1 for r in recs if r["pred_lonely"] and r["lonely"])
tn = sum(1 for r in recs if not r["pred_lonely"] and not r["lonely"])
fp = sum(1 for r in recs if r["pred_lonely"] and not r["lonely"])
fn = sum(1 for r in recs if not r["pred_lonely"] and r["lonely"])
acc = (tp + tn) / len(recs)
sens = tp / (tp + fn) if (tp + fn) else 0
spec = tn / (tn + fp) if (tn + fp) else 0
prec = tp / (tp + fp) if (tp + fp) else 0
f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0
print("\n=== 汇总（%s, %d 条） ===" % (SET, len(recs)))
print("TP=%d TN=%d FP=%d FN=%d" % (tp, tn, fp, fn))
print("Acc=%.1f%% Sens=%.1f%% Spec=%.1f%% F1=%.1f%%" % (acc * 100, sens * 100, spec * 100, f1 * 100))

summary = {"set": SET, "n": len(recs), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
           "acc": acc, "sens": sens, "spec": spec, "f1": f1}
summary_path = os.path.join(OUT_DIR, "summary_%s.json" % SET)
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("汇总已保存: %s" % summary_path)
