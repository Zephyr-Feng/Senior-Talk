"""诊断：EATD 中命中孤独维度的样本原文（判断是否误报）"""
import sys, csv, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from loneliness_signal import detect_loneliness, LONELINESS_PATTERNS

BASE = "/root/autodl-tmp/senior_project"
FEATURES_CSV = os.path.join(BASE, "output/eatd_analysis/all_features.csv")
EATD_DIR = os.path.join(BASE, "data/EATD-Corpus/EATD-Corpus")

with open(FEATURES_CSV) as f:
    rows = list(csv.DictReader(f))

total_hits = 0
for row in rows:
    txt_path = os.path.join(EATD_DIR, row["person_id"], row["emotion"] + ".txt")
    if not os.path.exists(txt_path):
        continue
    t = open(txt_path).read().strip()
    if len(t) < 5:
        continue
    strong, dims, quotes = detect_loneliness(t)
    if dims:
        total_hits += 1
        print(f"[{row['person_id']}|{row['emotion']}] strong={strong} dims={dims} quotes={quotes}")
        print(f"   原文: {t[:120]}")

print(f"\n命中维度样本总数: {total_hits}")
