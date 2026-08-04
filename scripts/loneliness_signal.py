"""
孤独感信号检测（ULS-8 维度映射，供训练数据报告生成使用）

设计原则：
- 维度计数制：命中 ULS-8 维度数 >= 2，或（>=1 且含明确词汇 孤独/孤单/寂寞/孤零零）才算"明显孤独倾向"
- 维度计数天然抗噪：真实孤独表达通常多维度（如"一个人在家…没人说话"→ 缺乏陪伴+无人倾诉）
- 单维度单关键词命中不触发，避免误报（已知案例：t_68 "小孩子就是一个人生的一块白板"）
- 否定/适应语境短语命中时，其覆盖的关键词不再计数（如"不是一个人"、"一个人也挺好"）
"""

# === ULS-8 维度 → 关键词映射（老年人场景） ===
# 一个关键词只归属一个维度，避免重复计数
LONELINESS_PATTERNS = {
    "缺乏陪伴": [
        "一个人", "自己一个人", "独自", "一个人在家", "一个人住", "一个人吃饭",
        "没伴", "独居", "就我一个", "孤身", "老伴走了", "老伴去世", "老伴不在了",
        "丧偶", "就剩我一个", "剩我一个人",
    ],
    "无人倾诉": [
        "没人说话", "连个说话的人都没有", "说话的人都没有", "连个聊天的人都没有",
        "找个人说话都难", "没有可以说话的人", "没人聊", "没人说", "想说话没人",
        "找人说", "没人听", "没人和我说话", "没人陪我说话",
    ],
    "被冷落": [
        "没人理", "没人关心", "没人来看", "不来看我", "被冷落", "被孤立",
        "都不理我", "不管我",
    ],
    "无法融入": [
        "融不进去", "不合群", "说不上话", "融不进", "插不上话",
    ],
    "孤单感受": [
        "孤独", "孤单", "寂寞", "孤零零", "没朋友", "朋友少", "交不到朋友", "孤孤单单",
    ],
    "被遗弃": [
        "被抛弃", "被遗弃", "没人要", "没人管",
    ],
}

# 明确孤独词汇（命中其一即满足强信号判定条件之一）
LONELINESS_EXPLICIT = ["孤独", "孤单", "寂寞", "孤零零"]

# 否定/适应语境 → 不算孤独信号（命中时剔除其覆盖的关键词）
LONELINESS_NEGATION = [
    "不是一个人", "不只我一个人", "不止我一个人", "一个人也挺好",
    "一个人也能", "一个人照样", "一个人就挺好", "一个人习惯了", "不是孤独",
]


def detect_loneliness(text):
    """检测文本中的孤独信号（严格阈值）。

    参数:
        text: 说话内容转写文本

    返回:
        (strong, dims, quotes)
        - strong: 是否构成明显孤独倾向（bool）
        - dims:   命中的 ULS-8 维度名列表
        - quotes: 每维度首个命中关键词（最多 2 个，供报告引用）
    """
    # 1. 否定语境过滤：命中否定短语时，其覆盖的关键词不再计数
    negated = set()
    for neg in LONELINESS_NEGATION:
        if neg in text:
            for kws in LONELINESS_PATTERNS.values():
                for kw in kws:
                    if kw in neg:
                        negated.add(kw)

    # 2. 按维度统计命中
    dims, quotes = [], []
    for dim, kws in LONELINESS_PATTERNS.items():
        hits = [kw for kw in kws if kw in text and kw not in negated]
        if hits:
            dims.append(dim)
            quotes.append(hits[0])

    # 3. 强信号判定：命中维度 >= 2，或（>=1 且含明确孤独词汇）
    score = len(dims)
    has_explicit = any(kw in text and kw not in negated for kw in LONELINESS_EXPLICIT)
    strong = score >= 2 or (score >= 1 and has_explicit)

    return strong, dims, quotes[:2]
