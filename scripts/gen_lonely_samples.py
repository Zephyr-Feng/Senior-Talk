"""
合成孤独倾向训练数据生成器（模板注入法，标签随生成已知）

设计：
- 正样本：老年人口吻 + ULS-8 六维度孤独表达（含蓄/明确混合，覆盖 6 维度）
- 负样本：日常社交对话（无孤独信号）
- 自检：用 loneliness_signal.detect_loneliness 验证正样本可被规则识别、负样本不触发
- 输出：JSONL（对齐 LoRA 训练格式）+ 控制台预览

用法：
  python gen_lonely_samples.py --pos 10 --neg 10 --out preview_lonely
  python gen_lonely_samples.py --pos 300 --neg 300 --out lonely_train   # 全量训练集
"""
import argparse, json, os, random, sys
from collections import Counter
from loneliness_signal import detect_loneliness

random.seed(42)

# === ULS-8 六维度 → 老年人口吻表达模板 ===
# 每维度多个句式；含蓄表达为主（真实孤独多为含蓄），穿插明确词（孤独/孤单/寂寞）
LONELY_TEMPLATES = {
    "缺乏陪伴": [
        "唉，老伴走了以后，家里就剩我一个人了",
        "孩子们都在外面忙，我一个人吃饭，一个人看电视",
        "平时就我自己一个人在家，屋里空荡荡的",
        "春节孩子们也没回来，我一个人过的年",
        "独居好几年了，饭都是自己做自己吃",
        "晚上早早把门关上，就我一个人，灯亮着也没意思",
        "孩子们都住得远，一年到头回不了几次，家里就我一个人",
        "老伴去得早，这些年都是自己一个人过来的",
        "一个人住久了，吃饭也就是对付一口",
        "星期天公园里全是人，就我是自己一个人走的",
    ],
    "无人倾诉": [
        "心里有事想跟人说，可连个说话的人都没有",
        "白天黑夜就我自己，想找个人说话都难",
        "没人听我说，电话都响不了几回",
        "老伴在的时候还能说说话，现在连个聊天的人都没有",
        "憋了一肚子话，也不知道跟谁讲",
        "有时候想说说话，拿起电话也不知道打给谁",
        "想说点心里话，翻翻手机也不知道发给谁",
        "一天到晚难得有人跟我说句话",
        "心里憋得慌，也没个人说说",
        "有时候自言自语，说给自己听",
    ],
    "被冷落": [
        "孩子们也不来看我，电话也少",
        "街坊邻居都各忙各的，也没人理我",
        "逢年过节也没人想着我",
        "家里冷冷清清的，也没人关心我吃没吃饭",
        "隔壁老刘家儿女常来，我家孩子一年到头见不到几面",
        "生病了也只能自己去医院，没人陪着",
        "过节也不见有人惦记我",
        "他们忙他们的，我是没人管的那一个",
    ],
    "无法融入": [
        "小区里那些人在聊天，我插不上话",
        "广场舞那帮人都是老姐妹一伙一伙的，我融不进去",
        "去儿子家也待不住，跟邻居也说不上话",
        "老同学聚会我也去了，他们聊的那些我接不上茬",
        "楼下棋摊上我去了几次，人家都不带我一个",
        "现在的年轻人聊的我都听不懂，插不上话",
        "老同事聚会，就我跟大家说不到一块儿",
        "社区活动我去了，总觉得格格不入",
    ],
    "孤单感受": [
        "一个人待着，心里总觉得孤单",
        "晚上躺在床上，孤单得很",
        "这日子过得寂寞，连个盼头都没有",
        "有时候觉得挺孤零零的，像被忘了似的",
        "一个人久了，心里空落落的，说不出的寂寞",
        "夜深人静的时候，心里空落落的",
        "看到别人家热热闹闹的，自己心里酸酸的，挺孤单",
        "人老了，最怕的就是寂寞",
    ],
    "被遗弃": [
        "感觉孩子们都不要我了",
        "没人管我，就像被扔下了一样",
        "像我们这样的老人，是不是就被社会忘了",
        "他们嫌我麻烦，我也不想拖累他们",
        "孩子们把我送到养老院，就很少来看我，感觉被扔在这儿了",
        "有时候觉得我就是多余的人",
        "没人需要我，也没人惦记我",
        "老了老了，落得个没人管的下场",
    ],
}

# 正样本开场白（增强对话感；不含关键词，不影响检测）
OPENERS = [
    "", "唉，你说", "嗯，", "怎么说呢，", "唉，",
]

# === 负样本：日常社交对话（有陪伴/有活动，无孤独信号） ===
NEG_TEMPLATES = [
    "今天去菜市场，买了条鱼，晚上清蒸",
    "孙子今天放学回来，我给他做了红烧肉",
    "早上跟老张在公园下棋，下了三盘",
    "我们几个老姐妹约好了周末去爬山",
    "电视里在放戏曲频道，我听着挺好",
    "下午去接孙女放学，路上她给我讲学校的事",
    "儿子打电话来，说周末带全家回来吃饭",
    "老伴在阳台浇花，我在这看电视，挺好的",
    "社区活动室今天有合唱班，我跟邻居老李一块去了",
    "刚跟女儿视频完，她说下周带外孙来看我",
    "天气好，出去遛了遛弯，碰见几个老伙计聊了会儿",
    "晚饭后跟老伴去广场走一圈，现在睡觉都香",
    "儿媳妇给买了件新衣服，穿着挺合适",
    "老战友打电话约我下个月聚一聚",
    "今天跟闺女学用智能手机，学会发照片了",
    "邻居王奶奶送来自家包的饺子，真香",
    "今天大集，买了点花生和红薯",
    "晚上跟儿子视频，他给我看了新装修的房子",
    "孙子考了满分，高兴得我晚上都睡不着",
    "老李约我明天一起去钓鱼",
    "社区组织体检，我报名了",
    "闺女教我用手机挂号，学得挺快",
    "早上公园打太极，碰见好多熟人",
    "家里养了只小猫，天天逗它玩",
    "周末孩子们都回来，一大家子吃饭，热闹",
    "我跟老姐妹每周三都去跳舞",
    "儿子说下个月带我坐高铁出去玩",
    "老伴我俩早上一起去买菜，回来我做饭他洗碗",
    "小区新开了个老年食堂，饭菜便宜，我跟邻居天天去",
    "昨天参加孙女的家长会了，老师说孙女表现不错",
    "我报了老年大学的书法班，老师夸我进步快",
    "老战友群里天天有人聊天，热闹得很",
    "天气好的时候，我跟老伴去公园散步",
    "女儿给我买了按摩椅，坐着可舒服了",
    "今天跟邻居在楼下晒太阳聊了会天",
    "儿媳妇蒸了包子给我送过来，好吃",
    "我跟老伴计划着冬天去海南过冬",
    "老同事约我下棋，我下午就去",
    "孩子们给家里装了智能电视，我学着用",
    "妹妹打电话说下周来看我",
]

# 负样本开场白
NEG_OPENERS = ["", "嗯，", "今天啊，", "对了，", ""]


def build_positive():
    """构造一条正样本：随机 1-3 个维度表达 + 可选开场白，保证至少 2 维度或无明确词时有明确词"""
    n_dims = random.choice([1, 2, 2, 3])  # 偏向多维度（真实孤独通常多维度）
    dims = random.sample(list(LONELY_TEMPLATES.keys()), min(n_dims, len(LONELY_TEMPLATES)))
    sentences = []
    for d in dims:
        sentences.append(random.choice(LONELY_TEMPLATES[d]))
    # 单维度时补一个明确词句（保证规则引擎可识别为强信号）
    if len(dims) == 1 and not any(w in sentences[0] for w in ["孤独", "孤单", "寂寞", "孤零零"]):
        sentences.append(random.choice(LONELY_TEMPLATES["孤单感受"]))
    opener = random.choice(OPENERS)
    text = opener + "，".join(sentences) if opener else "，".join(sentences)
    return text, dims


def build_negative():
    n = random.choice([1, 1, 2])
    sentences = random.sample(NEG_TEMPLATES, min(n, len(NEG_TEMPLATES)))
    opener = random.choice(NEG_OPENERS)
    text = opener + "，".join(sentences) if opener else "，".join(sentences)
    return text, []


def build_record(text, lonely, dims):
    """构造训练样本（对齐 LoRA 推理式报告格式，结论行二值）"""
    system_msg = "你是一个专业的心理健康评估助手。请根据对方说话的内容判断其是否存在孤独倾向，输出简短分析并给出明确结论。"
    user_msg = f"【说话内容】{text}"
    if lonely:
        qs = "、".join(f'"{d}"' for d in dims)
        analysis = f"分析：说话内容提及{dims[0]}等孤独相关表达，存在明显孤独倾向信号。"
        output = f"{analysis}\n孤独倾向：明显"
    else:
        output = "分析：说话内容为日常社交/家庭互动描述，未检测到孤独相关信号。\n孤独倾向：不明显"
    return {
        "system": system_msg,
        "input": user_msg,
        "output": output,
        "text": text,
        "lonely": lonely,
        "lonely_dims": dims,
        "source": "synthetic_lonely_v1",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=int, default=10)
    ap.add_argument("--neg", type=int, default=10)
    ap.add_argument("--out", default="preview_lonely")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    records, stats = [], {"pos": 0, "neg": 0, "pos_detected": 0, "neg_false_pos": 0}
    seen, dim_counter, lengths = set(), Counter(), []

    def unique_gen(fn):
        """生成不重复文本（重试上限 30 次）"""
        for _ in range(30):
            text, dims = fn()
            if text not in seen:
                return text, dims
        return text, dims  # 组合空间耗尽时接受重复（质检报告会暴露）

    # 正样本（孤独）
    while stats["pos"] < args.pos:
        text, dims = unique_gen(build_positive)
        strong, hit_dims, _ = detect_loneliness(text)
        seen.add(text)
        records.append(build_record(text, True, dims))
        stats["pos"] += 1
        stats["pos_detected"] += int(strong)
        lengths.append(len(text))
        for d in dims:
            dim_counter[d] += 1
        if stats["pos"] <= 10:
            print(f"\n[正样本 {stats['pos']}] 维度={dims} 规则引擎命中={strong} {hit_dims}")
            print(f"  「{text}」")

    # 负样本（无孤独）
    while stats["neg"] < args.neg:
        text, _ = unique_gen(build_negative)
        strong, hit_dims, _ = detect_loneliness(text)
        seen.add(text)
        records.append(build_record(text, False, []))
        stats["neg"] += 1
        stats["neg_false_pos"] += int(strong)
        lengths.append(len(text))
        if stats["neg"] <= 10:
            print(f"\n[负样本 {stats['neg']}] 规则引擎误报={strong} {hit_dims}")
            print(f"  「{text}」")

    # 保存
    jsonl = os.path.join(args.out, "lonely_samples.jsonl")
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # === 质量报告（硬性检查） ===
    total = stats["pos"] + stats["neg"]
    lines = []
    lines.append(f"正样本 {stats['pos']} 条，规则引擎识别 {stats['pos_detected']} 条（{stats['pos_detected']/stats['pos']:.0%}）")
    lines.append(f"负样本 {stats['neg']} 条，规则引擎误报 {stats['neg_false_pos']} 条")
    lines.append(f"文本唯一率: {len(seen)}/{total}（{len(seen)/total:.1%}）")
    lines.append(f"文本长度: 最短 {min(lengths)} 字 / 最长 {max(lengths)} 字 / 平均 {sum(lengths)/len(lengths):.0f} 字")
    lines.append(f"六维度分布: " + "、".join(f"{d}={dim_counter[d]}" for d in LONELY_TEMPLATES))
    missing = [d for d in LONELY_TEMPLATES if dim_counter[d] == 0]
    if missing:
        lines.append(f"⚠️ 未覆盖维度: {missing}")

    print("\n=== 质量报告 ===")
    for ln in lines:
        print("  " + ln)

    # 硬指标：负样本误报必须为 0
    if stats["neg_false_pos"] > 0:
        print("❌ 负样本误报 > 0，数据不合格！请检查负样本模板是否含孤独关键词")
        sys.exit(1)
    with open(os.path.join(args.out, "quality_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"已保存: {jsonl} + quality_report.txt")


if __name__ == "__main__":
    main()
