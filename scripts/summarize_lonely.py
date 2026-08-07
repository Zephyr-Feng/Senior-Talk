"""
合并 eval_lonely 分片结果 + 汇总指标
用法: python summarize_lonely.py <file1.jsonl> <file2.jsonl>... [--out summary.json]
"""
import argparse, json, os


def parse_lonely_pred(conclusion):
    """精确匹配结论（2026-08-07 修复：'明显' in '不明显' 子串误判导致全部误报）"""
    c = conclusion.strip()
    if c == "明显":
        return True
    if c == "不明显":
        return False
    return c.startswith("明显") and not c.startswith("不明显")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    recs = {}
    for f in args.files:
        if not os.path.exists(f):
            continue
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            recs[r["key"]] = r
    recs = list(recs.values())
    if not recs:
        print("无结果，退出"); return

    # 从结论原文重新解析（jsonl 中 pred_lonely 可能含旧 bug）
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

    print("TP=%d TN=%d FP=%d FN=%d (n=%d)" % (tp, tn, fp, fn, len(recs)))
    print("Acc=%.1f%% Sens=%.1f%% Spec=%.1f%% F1=%.1f%%" % (acc * 100, sens * 100, spec * 100, f1 * 100))

    if args.out:
        summary = {"n": len(recs), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                   "acc": acc, "sens": sens, "spec": spec, "f1": f1}
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print("保存:", args.out)


if __name__ == "__main__":
    main()
