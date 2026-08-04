"""
融合实验 第一步：声学通道分类器（按人分组 CV，out-of-fold 概率）
- 输入: output/eatd_analysis/all_features.csv（486 条）
- 输出: training_output/fusion/p_acoustic.json（每条 person_id|emotion → 概率）
- 分类器对比: RandomForest / SVM-RBF / GradientBoosting，取 AUC 最高者
"""
import json, os, csv
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

BASE = "/root/autodl-tmp/senior_project"
FEATURES_CSV = os.path.join(BASE, "output/eatd_analysis/all_features.csv")
OUT_DIR = os.path.join(BASE, "training_output/fusion")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_JSON = os.path.join(OUT_DIR, "p_acoustic.json")

FEATURES = ["speech_rate", "pause_ratio", "pitch_variability", "energy_variability",
            "jitter", "shimmer", "voiced_ratio", "spectral_centroid"]

# === 1. 加载数据 ===
def sf(v):
    try: return float(v) if v and v.strip() else 0.0
    except: return 0.0

rows = []
with open(FEATURES_CSV) as f:
    for r in csv.DictReader(f):
        rows.append({
            "key": "%s|%s" % (r["person_id"], r["emotion"]),
            "person_id": r["person_id"],
            "X": [sf(r[k]) for k in FEATURES],
            "y": r["depressed"].strip() == "True",
        })

X = np.array([r["X"] for r in rows])
y = np.array([r["y"] for r in rows])
groups = np.array([r["person_id"] for r in rows])
print("样本: %d（正 %d / 负 %d），人: %d" % (len(rows), y.sum(), len(y) - y.sum(), len(set(groups))))

# === 2. 按人 GroupKFold → out-of-fold 概率 ===
classifiers = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "SVM_RBF": SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

gkf = GroupKFold(n_splits=5)
best_auc, best_name = -1, None
oof = {}
for name, clf in classifiers.items():
    probs = np.zeros(len(rows))
    for tr_idx, va_idx in gkf.split(X, y, groups):
        clf.fit(X[tr_idx], y[tr_idx])
        probs[va_idx] = clf.predict_proba(X[va_idx])[:, 1]
    auc = roc_auc_score(y, probs)
    acc = accuracy_score(y, (probs >= 0.5).astype(int))
    tn, fp, fn, tp = confusion_matrix(y, (probs >= 0.5).astype(int)).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    print("%-16s AUC=%.4f Acc=%.4f Sens=%.4f Spec=%.4f" % (name, auc, acc, sens, spec))
    if auc > best_auc:
        best_auc, best_name, oof = auc, name, probs

# === 3. 保存最优分类器的 out-of-fold 概率 ===
result = {
    "best_classifier": best_name,
    "auc": best_auc,
    "features": FEATURES,
    "probabilities": {
        rows[i]["key"]: float(oof[i]) for i in range(len(rows))
    },
}
with open(OUT_JSON, "w") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("\n最优: %s (AUC=%.4f) → %s" % (best_name, best_auc, OUT_JSON))
