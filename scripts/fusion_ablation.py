"""
融合实验 第二步：决策级融合 + 消融对比（需先跑完 eval_full 全量推理）
- P_acoustic: training_output/fusion/p_acoustic.json（声学分类器 out-of-fold 概率）
- P_text:     training_output/eval_full/eval_full.jsonl（LoRA 报告版结论二值）
- 融合: P_fused(w) = w·P_acoustic + (1-w)·P_text, w∈[0,1] 网格搜索
- 消融表: 纯声学(w=1) / 纯LoRA文本(w=0) / 最优融合(w*)
- 参照: 零样本 MLLM 63 条（output_eatd_mllm/_summary.json）
"""
import json, os, csv
import numpy as np

BASE = "/root/autodl-tmp/senior_project"
P_ACOUSTIC = os.path.join(BASE, "training_output/fusion/p_acoustic.json")
EVAL_FULL = os.path.join(BASE, "training_output/eval_full/eval_full.jsonl")
ZERO_SHOT_SUMMARY = os.path.join(BASE, "output_eatd_mllm/_summary.json")


def metrics(y_true, scores):
    """阈值 0.5 分类 + AUC"""
    from sklearn.metrics import roc_auc_score
    y = np.array(y_true)
    s = np.array(scores)
    p = (s >= 0.5).astype(int)
    tp = ((p == 1) & (y == 1)).sum()
    tn = ((p == 0) & (y == 0)).sum()
    fp = ((p == 1) & (y == 0)).sum()
    fn = ((p == 0) & (y == 1)).sum()
    acc = (tp + tn) / len(y)
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0
    auc = roc_auc_score(y, s) if len(set(y)) > 1 else float("nan")
    return {"acc": acc, "sens": sens, "spec": spec, "f1": f1, "auc": auc, "n": len(y)}


# === 1. 加载 P_acoustic ===
p_aco = json.load(open(P_ACOUSTIC))["probabilities"]
print("P_acoustic 样本: %d" % len(p_aco))

# === 2. 加载 eval_full（LoRA 全量推理结果） ===
if not os.path.exists(EVAL_FULL) or sum(1 for _ in open(EVAL_FULL, encoding="utf-8")) < 100:
    print("⚠️ eval_full.jsonl 未完成或过少，当前 %d 条，等推理完成后重跑" %
          sum(1 for _ in open(EVAL_FULL, encoding="utf-8")) if os.path.exists(EVAL_FULL) else 0)
    raise SystemExit(1)

rows = []
for line in open(EVAL_FULL, encoding="utf-8"):
    r = json.loads(line)
    key = "%s|%s" % (r["person_id"], r["emotion"])
    if key not in p_aco:
        continue
    rows.append({
        "key": key,
        "y": r["depressed"],
        "p_text": 1.0 if r["pred_dep"] else 0.0,
        "p_acoustic": p_aco[key],
    })
print("对齐样本: %d（正 %d / 负 %d）" % (len(rows), sum(r["y"] for r in rows), len(rows) - sum(r["y"] for r in rows)))

y = [r["y"] for r in rows]
p_t = [r["p_text"] for r in rows]
p_a = [r["p_acoustic"] for r in rows]

# === 3. 融合网格搜索 ===
print("\n=== 融合网格（w = 声学权重） ===")
print("%-8s %-8s %-8s %-8s %-8s %-8s" % ("w", "Acc", "Sens", "Spec", "F1", "AUC"))
results = {}
for w in [i / 10 for i in range(0, 11)]:
    fused = [w * pa + (1 - w) * pt for pa, pt in zip(p_a, p_t)]
    m = metrics(y, fused)
    results[w] = m
    print("%-8.1f %-8.4f %-8.4f %-8.4f %-8.4f %-8.4f" % (w, m["acc"], m["sens"], m["spec"], m["f1"], m["auc"]))

# 最优 w：按 F1 选（兼顾敏感度/特异度）
best_w = max(results, key=lambda w: results[w]["f1"])
print("\n最优融合 w=%.1f: %s" % (best_w, results[best_w]))

# === 4. 消融表 ===
print("\n=== 消融对比（486 条全量） ===")
ablation = [
    ("纯声学分类器 (w=1)", metrics(y, p_a)),
    ("纯LoRA文本 (w=0)", metrics(y, p_t)),
    ("决策级融合 (w=%.1f)" % best_w, results[best_w]),
]
print("%-24s %-8s %-8s %-8s %-8s %-8s" % ("配置", "Acc", "Sens", "Spec", "F1", "AUC"))
for name, m in ablation:
    print("%-24s %-8.4f %-8.4f %-8.4f %-8.4f %-8.4f" % (name, m["acc"], m["sens"], m["spec"], m["f1"], m["auc"]))

# === 5. 参照：零样本 MLLM（63 条，评估集不同） ===
if os.path.exists(ZERO_SHOT_SUMMARY):
    z = json.load(open(ZERO_SHOT_SUMMARY))
    print("\n=== 参照：零样本纯文本 MLLM（63 条子集，非同一评估集） ===")
    print("Acc=%.4f Sens=%.4f Spec=%.4f F1=%.4f (TP=%d TN=%d FP=%d FN=%d)" %
          (z["accuracy"], z["recall"], z["specificity"], z["f1"], z["tp"], z["tn"], z["fp"], z["fn"]))

# === 6. 保存结果 ===
out = {
    "n": len(rows),
    "acoustic_classifier": p_aco,
    "grid": {str(w): results[w] for w in results},
    "best_w": best_w,
    "best": results[best_w],
    "ablation": [{"name": n, **m} for n, m in ablation],
}
out_path = os.path.join(BASE, "training_output/fusion/ablation_result.json")
with open(out_path, "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\n结果保存: %s" % out_path)
