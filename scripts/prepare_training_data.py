"""
准备 EATD 训练数据用于 Qwen2.5-VL LoRA 微调
- 使用原始 .txt 转录（比 Whisper 准确）
- 声学特征转为文本描述
- 均衡采样（抑郁样本上采样 + 数据增强）
- ★ 输出为推理式诊断报告（引用具体文本和声学特征）
"""
import json, os, csv, random, copy, re
import numpy as np
from loneliness_signal import detect_loneliness

random.seed(42)
np.random.seed(42)

# === 配置 ===
BASE = "/root/autodl-tmp/senior_project"
EATD_DIR = os.path.join(BASE, "data/EATD-Corpus/EATD-Corpus")
FEATURES_CSV = os.path.join(BASE, "output/eatd_analysis/all_features.csv")
OUTPUT_DIR = os.path.join(BASE, "training_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SDS_THRESHOLD = 50  # SDS >= 50 视为抑郁

# === 抑郁关键词 → 量表条目映射 ===
DEPRESSION_KEYWORDS = {
    "sleep": ["睡不好", "失眠", "睡不着", "早醒", "多梦", "睡不著", "很难睡", "入睡困难", "惊醒", "整晚没睡"],
    "interest": ["没意思", "没劲", "无聊", "没兴趣", "不想动", "懒得动", "不想出门", "没热情", "不想做"],
    "fatigue": ["累", "好累", "疲惫", "疲劳", "没力气", "浑身没劲", "没精神", "觉得很累", "乏力"],
    "mood": ["难过", "伤心", "不开心", "心情不好", "低落", "郁闷", "想哭", "不高兴", "烦"],
    "anxiety": ["烦躁", "焦虑", "紧张", "担心", "不安", "心慌", "害怕", "担心", "坐立不安"],
    "loneliness": ["一个人", "孤独", "没人", "没人陪", "自己", "孤零零", "没朋友"],
    "appetite": ["没胃口", "吃不下", "不想吃", "瘦了", "吃不下饭", "没食欲", "体重下降"],
    "worthlessness": ["没用", "没价值", "拖累", "负担", "没意义", "失败", "对不起"],
    "concentration": ["记不住", "注意力", "想不起来", "发呆", "走神", "反应慢", "脑子空"],
}

PHQ9_MAP = {
    "sleep": "睡眠障碍（PHQ-9条目3）",
    "interest": "兴趣丧失（PHQ-9条目1）",
    "fatigue": "精力不足（PHQ-9条目4）",
    "mood": "情绪低落（PHQ-9条目2）",
    "anxiety": "焦虑症状（GAD-7相关）",
    "loneliness": "孤独感（ULS-8相关）",
    "appetite": "食欲改变（PHQ-9条目5）",
    "worthlessness": "自我价值感降低（PHQ-9条目6）",
    "concentration": "注意力下降（PHQ-9条目7）",
}

# === 1. 读取特征 + 标签 ===
print("读取特征文件...")
samples = []
with open(FEATURES_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid = row["person_id"]
        emotion = row["emotion"]
        sds = float(row["sds"])
        depressed = row["depressed"].strip() == "True"

        # 读取原始转录
        txt_path = os.path.join(EATD_DIR, pid, f"{emotion}.txt")
        if os.path.exists(txt_path):
            with open(txt_path) as tf:
                text = tf.read().strip()
        else:
            continue
        if len(text) < 5:
            continue

        def safe_float(v, default=0.0):
            try: return float(v) if v and v.strip() else default
            except: return default

        sample = {
            "person_id": pid,
            "emotion": emotion,
            "sds": sds,
            "depressed": depressed,
            "text": text,
            "speech_rate": safe_float(row.get("speech_rate")),
            "pause_ratio": safe_float(row.get("pause_ratio")),
            "pitch_variability": safe_float(row.get("pitch_variability")),
            "energy_variability": safe_float(row.get("energy_variability")),
            "jitter": safe_float(row.get("jitter")),
            "shimmer": safe_float(row.get("shimmer")),
            "voiced_ratio": safe_float(row.get("voiced_ratio")),
            "spectral_centroid": safe_float(row.get("spectral_centroid")),
        }
        samples.append(sample)

print(f"有效样本: {len(samples)}")
dep = sum(1 for s in samples if s["depressed"])
norm = len(samples) - dep
print(f"  抑郁: {dep}, 正常: {norm}")
lonely_all = sum(1 for s in samples if detect_loneliness(s["text"])[0])
print(f"  全量中带明显孤独信号: {lonely_all} 条（{lonely_all / len(samples):.1%}）")


# === 2. 声学特征 → 文本描述 ===
def acoustic_to_text(s):
    """将声学特征转成自然语言描述"""
    parts = []

    sr = s["speech_rate"]
    if sr < 2.5:
        parts.append(f"语速很慢（{sr:.1f}字/秒）")
    elif sr < 4.0:
        parts.append(f"语速偏慢（{sr:.1f}字/秒）")
    elif sr < 5.5:
        parts.append(f"语速正常（{sr:.1f}字/秒）")
    else:
        parts.append(f"语速偏快（{sr:.1f}字/秒）")

    pr = s["pause_ratio"]
    if pr < 0.2:
        parts.append("停顿很少")
    elif pr < 0.4:
        parts.append("停顿略多")
    elif pr < 0.6:
        parts.append("停顿较多")
    else:
        parts.append("停顿非常多")

    pv = s["pitch_variability"]
    if pv < 20:
        parts.append("音调变化小，语调平淡")
    elif pv < 40:
        parts.append("音调变化正常")
    elif pv < 60:
        parts.append("音调变化较大，情绪有一定波动")
    else:
        parts.append("音调变化显著，情绪起伏明显")

    ev = s["energy_variability"]
    if ev < 0.02:
        parts.append("发声能量极低，说话无力")
    elif ev < 0.05:
        parts.append("发声能量偏低，可能精力不足")
    elif ev < 0.08:
        parts.append("发声能量正常")
    else:
        parts.append("发声能量充足")

    jit = s["jitter"]
    if jit < 0.015:
        parts.append("嗓音稳定性好")
    elif jit < 0.025:
        parts.append("嗓音稳定性一般")
    else:
        parts.append("嗓音稳定性较差")

    shim = s["shimmer"]
    if shim < 0.08:
        parts.append("音质较干净")
    elif shim < 0.12:
        parts.append("音质一般")
    else:
        parts.append("音质有些嘶哑")

    vratio = s["voiced_ratio"]
    if vratio < 0.4:
        parts.append("浊音比例偏低，发音效率不高")
    elif vratio < 0.6:
        parts.append("浊音比例正常偏低")
    elif vratio < 0.8:
        parts.append("浊音比例正常")
    else:
        parts.append("浊音比例较高，发声充分")

    return "，".join(parts)


# === 3. 推理式诊断报告生成器 ===

def find_keywords(text):
    """在文本中查找抑郁关键词，返回 (匹配类别列表, 匹配引用列表)"""
    found_categories = set()
    matched_quotes = []
    for category, keywords in DEPRESSION_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                found_categories.add(category)
                matched_quotes.append(kw)
                break  # 每个类别只取一个关键词
    return list(found_categories), matched_quotes[:4]  # 最多引用4个


def generate_acoustic_analysis(s, depressed):
    """生成声学特征的临床分析，返回分析句列表"""
    items = []

    sr = s["speech_rate"]
    if sr < 2.5:
        items.append(f"语速很慢（{sr:.1f}字/秒），反映精力不足或情绪低落")
    elif sr < 4.0:
        items.append(f"语速偏慢（{sr:.1f}字/秒），可能反映精力不足")

    pr = s["pause_ratio"]
    if pr > 0.4:
        items.append(f"停顿较多（{pr:.0%}），提示可能存在思维迟缓或犹豫")

    pv = s["pitch_variability"]
    if pv < 20:
        items.append("音调变化小（语调平淡），常见于抑郁患者的单调发声模式")
    elif pv < 40:
        items.append("音调变化偏低，情绪表达受限")

    ev = s["energy_variability"]
    if ev < 0.05:
        items.append("发声能量偏低，说话力度不足，反映精力减退")

    jit = s["jitter"]
    if jit > 0.025:
        items.append(f"嗓音稳定性较差（jitter={jit:.3f}），反映声带控制减弱")

    shim = s["shimmer"]
    if shim > 0.12:
        items.append(f"音质嘶哑（shimmer={shim:.3f}），发声系统负担较重")

    if not items:
        items.append("声学特征基本在正常范围内")

    return items


def generate_reasoning_output(text, depressed, sds, features):
    """
    生成推理式诊断报告
    格式：分析 → 结论 → 建议
    ★ 抑郁为主判断；语义孤独信号明显时，在分析/建议中提示"可能存在孤独"
    """
    categories, quotes = find_keywords(text)
    acoustic_items = generate_acoustic_analysis(features, depressed)
    lonely_strong, lonely_dims, lonely_quotes = detect_loneliness(text)

    # === 文本分析 ===
    if depressed:
        if categories:
            scale_refs = [PHQ9_MAP[c] for c in categories if c in PHQ9_MAP]
            text_analysis = "说话内容提到" + "、".join(f'"{q}"' for q in quotes)
            if scale_refs:
                text_analysis += "，与" + "、".join(scale_refs) + "相关"
            text_analysis += "，属抑郁核心症状群"
        else:
            text_analysis = "说话内容虽未出现典型抑郁关键词，但结合声学特征需综合判断"
    else:
        if categories:
            scale_refs = [PHQ9_MAP[c] for c in categories if c in PHQ9_MAP]
            text_analysis = "说话内容提到" + "、".join(f'"{q}"' for q in quotes)
            if scale_refs:
                text_analysis += "，涉及" + "、".join(scale_refs)
            text_analysis += "，但程度较轻，不构成明显异常模式"
        else:
            text_analysis = "说话内容未检测到明显抑郁相关关键词，语义内容基本正常"

    # 孤独信号明显时，在文本分析中补充孤独倾向提示（不改变抑郁主判断）
    if lonely_strong:
        lq = "、".join(f'"{q}"' for q in lonely_quotes)
        text_analysis += f"。另外，说话内容明显体现孤独倾向（如{lq}），提示可能存在孤独情绪"

    # === 声学分析 ===
    acoustics_analysis = "；".join(acoustic_items)

    # === 综合判断 ===
    depressed_signs = sum([
        features["speech_rate"] < 4.0,     # 语速偏慢
        features["pause_ratio"] > 0.4,     # 停顿多
        features["pitch_variability"] < 20, # 音调平淡
        features["energy_variability"] < 0.05,  # 能量低
        features["jitter"] > 0.025,        # 嗓音不稳
        len(categories) > 0,               # 有关键词
    ])

    if depressed:
        if depressed_signs >= 3:
            assessment = f"文本与声学特征均有明确抑郁指向（{depressed_signs}/6项指标），综合判断为抑郁倾向"
        elif depressed_signs >= 1:
            assessment = "文本和/或声学特征存在部分抑郁指向，综合判断为轻度抑郁倾向"
        else:
            assessment = "声学特征正常但基于标签综合判断需关注"
        conclusion = "需要关注"
        if depressed_signs >= 3:
            suggestion = f"建议进一步评估PHQ-9量表（当前SDS={sds:.0f}），重点关注情绪状态和睡眠情况，建议心理科随访"
        elif len(categories) > 0:
            suggestion = f"建议进一步评估PHQ-9量表（当前SDS={sds:.0f}），关注文本中提及的相关症状"
        else:
            suggestion = f"建议保持观察（SDS={sds:.0f}），注意情绪变化趋势"
        if lonely_strong:
            suggestion += "；建议同时关注其孤独情绪，可评估 ULS-8 量表"
    else:
        if lonely_strong:
            # 抑郁为主判断：结论仍为"正常"，但孤独情绪明显需提示关注
            if categories or depressed_signs >= 1:
                assessment = "声学/文本特征存在部分异常但未达抑郁标准，说话内容明显体现孤独倾向，综合判断无抑郁倾向但需关注孤独情绪"
            else:
                assessment = "文本和声学特征未见明显抑郁信号，但说话内容明显体现孤独倾向，综合判断无抑郁倾向，需关注孤独情绪"
            suggestion = "建议关注其孤独情绪，可鼓励增加社交互动（评估 ULS-8 量表）"
        elif depressed_signs >= 3:
            assessment = f"虽声学特征有部分异常（{depressed_signs}/6项），但文本内容正常，综合判断为正常"
            suggestion = "保持日常观察即可"
        elif depressed_signs >= 1:
            assessment = f"声学特征有轻微异常但文本内容正常，综合判断无抑郁倾向"
            suggestion = "保持日常观察即可"
        else:
            assessment = "文本和声学特征均未见明显异常，综合判断为正常状态"
            suggestion = "保持日常观察即可"
        conclusion = "正常"

    # 组装完整报告
    report = f"""分析：
1. 文本方面 — {text_analysis}。
2. 声学方面 — {acoustics_analysis}。
3. 综合判断 — {assessment}。

结论：{conclusion}
建议：{suggestion}"""
    return report


# === 4. 构造训练样本 ===
def build_conversation(sample):
    """构造对话格式样本（推理式输出）"""
    acoustic_desc = acoustic_to_text(sample)

    system_msg = "你是一个专业的心理健康评估助手。请根据对方的语音声学特征和说话内容进行综合分析，输出推理式诊断报告。注意引用具体的说话内容和声学数值作为依据。"
    user_msg = f"【声学特征】{acoustic_desc}\n【说话内容】{sample['text']}"

    # 生成推理式输出
    output = generate_reasoning_output(
        sample["text"], sample["depressed"], sample["sds"], sample
    )

    return {
        "system": system_msg,
        "input": user_msg,
        "output": output,
        "person_id": sample["person_id"],
        "emotion": sample["emotion"],
        "sds": sample["sds"],
        "depressed": sample["depressed"],
        "lonely_hint": detect_loneliness(sample["text"])[0],
    }


# === 5. 数据增强（对抑郁样本做文本增强） ===
def augment_text(text):
    """对抑郁样本做轻量文本增强"""
    augments = []
    augments.append(text)
    prefixes = ["嗯，", "我觉得吧，", "怎么说呢，", ""]
    for p in prefixes:
        if p:
            augments.append(p + text)
    return augments

def augment_sample(sample):
    """对样本做增强，返回多个变体"""
    results = []
    texts = augment_text(sample["text"])
    for t in texts:
        s = copy.deepcopy(sample)
        s["text"] = t
        results.append(s)
    return results


# === 6. 构建最终训练集 ===
print("\n构建均衡训练集...")

dep_samples = [s for s in samples if s["depressed"]]
norm_samples = [s for s in samples if not s["depressed"]]

# 抑郁样本做 4x 增强
aug_dep = []
for s in dep_samples:
    aug_dep.extend(augment_sample(s))
print(f"抑郁样本增强后: {len(aug_dep)} 条")

# 正常样本做 2x 增强
aug_norm = []
for s in norm_samples:
    prefixes = ["嗯，", "我觉得吧，", ""]
    for p in prefixes:
        if p:
            ns = copy.deepcopy(s)
            ns["text"] = p + s["text"]
            aug_norm.append(ns)
        else:
            aug_norm.append(copy.deepcopy(s))
print(f"正常样本增强后: {len(aug_norm)} 条")

# 目标：各 ~200 条
TARGET_PER_CLASS = 200

random.shuffle(aug_dep)
random.shuffle(aug_norm)

final_dep = aug_dep[:TARGET_PER_CLASS]
final_norm = aug_norm[:TARGET_PER_CLASS]

if len(final_dep) < TARGET_PER_CLASS:
    print(f"  ⚠️ 抑郁样本不足，仅 {len(final_dep)} 条")
if len(final_norm) < TARGET_PER_CLASS:
    print(f"  ⚠️ 正常样本不足，仅 {len(final_norm)} 条")

train_samples = final_dep + final_norm
random.shuffle(train_samples)

print(f"\n最终训练集: {len(train_samples)} 条")
print(f"  抑郁: {sum(1 for s in train_samples if s['depressed'])}")
print(f"  正常: {sum(1 for s in train_samples if not s['depressed'])}")
lonely_hint = sum(1 for s in train_samples if detect_loneliness(s["text"])[0])
print(f"  训练集中带孤独提示: {lonely_hint} 条（{lonely_hint / len(train_samples):.1%}）")


# === 7. 保存 ===
train_jsonl = []
for s in train_samples:
    conv = build_conversation(s)
    train_jsonl.append(conv)

train_path = os.path.join(OUTPUT_DIR, "train_data.jsonl")
with open(train_path, "w") as f:
    for item in train_jsonl:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"\n训练数据已保存: {train_path}")

# 样本预览
txt_path = os.path.join(OUTPUT_DIR, "train_samples.txt")
with open(txt_path, "w") as f:
    for item in train_jsonl:
        f.write(f"=== {item['person_id']}_{item['emotion']} ===\n")
        f.write(f"SDS={item['sds']}, {'抑郁' if item['depressed'] else '正常'}\n")
        f.write(f"用户: {item['input'][:150]}...\n")
        f.write(f"助手: {item['output']}\n\n")
print(f"样本预览已保存: {txt_path}")

# 打印几个样本
print("\n=== 样本预览 ===")
for i, item in enumerate(train_jsonl[:3]):
    print(f"\n样本 {i+1}: {item['person_id']}_{item['emotion']}")
    print(f"  SDS={item['sds']}, {'抑郁' if item['depressed'] else '正常'}")
    print(f"  用户: {item['input'][:150]}...")
    print(f"  助手输出:\n{item['output']}")

print("\n数据准备完成！")
