"""
Qwen2.5-VL-7B LoRA 微调脚本 —— 孤独倾向二分类版
- 训练数据：合成孤独样本（300 正 + 300 负，模板注入法，标签随生成已知）
- 4-bit 量化加载以节省显存，输出：孤独倾向 明显/不明显（推理式报告）
- 正负均衡，无需加权 loss（与抑郁版 FN_WEIGHT=3.0 不同）
- 输入仅【说话内容】（孤独倾向主要靠语义判断）
- 分层留出 30+30 验证集（不参与训练，训练后评估）
"""
import json, os, random, torch
from torch.utils.data import Dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# === 配置 ===
MODEL_PATH = "/root/autodl-tmp/.cache/modelscope/qwen/Qwen2.5-VL-7B-Instruct"
DATA_PATH = "/root/autodl-tmp/senior_project/training_data_lonely/lonely_samples.jsonl"
OUTPUT_DIR = "/root/autodl-tmp/senior_project/training_output/lora_adapter_lonely_v1"
HOLDOUT_PATH = "/root/autodl-tmp/senior_project/training_data_lonely/holdout.jsonl"
HOLDOUT_PER_CLASS = 30   # 留出 30+30 验证集
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LoRA 参数（与抑郁版一致）
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# 训练参数（与抑郁版一致）
BATCH_SIZE = 2        # 4-bit + LoRA + 24GB VRAM，batch=2 安全
GRAD_ACCUM_STEPS = 4  # 等效 batch = 2*4 = 8
EPOCHS = 5
LEARNING_RATE = 2e-4
MAX_LENGTH = 1280

# === 1. 加载数据 ===
print("加载训练数据...")
with open(DATA_PATH) as f:
    raw_data = [json.loads(line) for line in f if line.strip()]

lonely = sum(1 for d in raw_data if d["lonely"])
print(f"总样本: {len(raw_data)}")
print(f"  孤独: {lonely}, 不孤独: {len(raw_data)-lonely}")

# === 分层留出验证集（30+30，不参与训练） ===
random.seed(42)
lonely_data = [d for d in raw_data if d["lonely"]]
not_lonely_data = [d for d in raw_data if not d["lonely"]]
random.shuffle(lonely_data)
random.shuffle(not_lonely_data)
holdout = lonely_data[:HOLDOUT_PER_CLASS] + not_lonely_data[:HOLDOUT_PER_CLASS]
train_data = lonely_data[HOLDOUT_PER_CLASS:] + not_lonely_data[HOLDOUT_PER_CLASS:]
random.shuffle(train_data)
with open(HOLDOUT_PATH, "w") as f:
    for h in holdout:
        f.write(json.dumps(h, ensure_ascii=False) + "\n")
print(f"留出验证集: {HOLDOUT_PER_CLASS}+{HOLDOUT_PER_CLASS} 条 → {HOLDOUT_PATH}")
print(f"训练集: {len(train_data)} 条")
raw_data = train_data

# === 2. Tokenization 数据集 ===
class LonelyDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Qwen2.5-VL 对话格式
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": item["system"]}]},
            {"role": "user", "content": [{"type": "text", "text": item["input"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": item["output"]}]},
        ]

        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        tokens = self.processor(
            text=text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].squeeze()
        labels = input_ids.clone()
        labels[tokens["attention_mask"].squeeze() == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": tokens["attention_mask"].squeeze(),
            "labels": labels,
        }


# === 3. 加载模型 ===
print("\n加载模型...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",  # 不用 flash_attn
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = prepare_model_for_kbit_training(model)

# === 4. LoRA 配置 ===
print("\n配置 LoRA...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === 5. 训练 ===
dataset = LonelyDataset(raw_data, processor)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    bf16=True,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    report_to="none",
    warmup_ratio=0.03,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
)

print("\n开始训练...")
trainer.train()

# === 6. 保存 ===
print("\n保存模型...")
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

config = {
    "task": "lonely_binary",
    "model_path": MODEL_PATH,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "train_samples": len(raw_data),
    "lonely": sum(1 for d in raw_data if d["lonely"]),
    "not_lonely": sum(1 for d in raw_data if not d["lonely"]),
    "holdout": HOLDOUT_PER_CLASS * 2,
    "data_source": "synthetic_lonely_v1",
}
with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"\n训练完成！模型已保存到: {OUTPUT_DIR}")
print(f"\n===== 训练配置 =====")
for k, v in config.items():
    print(f"  {k}: {v}")
