"""
EATD MLLM Evaluation - Fusion version (text + acoustic features)
- Loads text transcripts + acoustic features
- Builds enriched prompt for Qwen2.5-VL
- Compares with SDS labels
"""
import os
import json
import time
import re
import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
os.environ.pop("OMP_NUM_THREADS", None)

sys.path.insert(0, str(Path(__file__).parent))
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


# ---- Prompt Template ----
SYSTEM_PROMPT = """你是一位专业的老年心理健康评估助理。你的任务是根据老年人的访谈对话文本和语音声学特征，综合评估其抑郁风险。

## 声学特征说明
- 语速越慢、停顿越多 → 可能反映精力不足、情绪低落
- 音调变化（pitch_std）越小 → 可能反映情绪平淡（flat affect）
- 能量变化（energy_std）越小 → 可能反映情感迟钝
- Jitter（基频微扰）/ Shimmer（振幅微扰）偏高 → 可能反映声带控制减弱
- 说话比例（speech_ratio）越低 → 沉默越多
- 共振峰（spectral centroid）变化 → 反映声道紧张度

## 评估维度
1. 情绪状态：是否表现出持续的悲伤、焦虑、兴趣丧失等核心症状
2. 认知功能：思维是否清晰
3. 声学一致性：语音特征是否与情绪内容一致（例如笑着说话却音调平？）
4. 综合判断

请以JSON格式返回：
{
  "depression_risk": "high/moderate/low",
  "depression_probability": 0-100,
  "reasoning": "推理过程（要结合声学特征和文本）",
  "key_signs": ["症状1", "症状2"],
  "confidence": "high/medium/low"
}"""


def load_subject(data_dir, acoustic_dir, subject_id):
    """Load text transcripts and acoustic features"""
    base = os.path.join(data_dir, subject_id)
    transcripts = {}
    for emotion in ["neutral", "negative", "positive"]:
        txt_file = os.path.join(base, f"{emotion}.txt")
        if os.path.exists(txt_file):
            with open(txt_file, "r", encoding="utf-8") as f:
                transcripts[emotion] = f.read().strip()
        else:
            transcripts[emotion] = ""

    # Load acoustic features
    acoustic_file = os.path.join(acoustic_dir, f"{subject_id}.json")
    acoustic = {}
    if os.path.exists(acoustic_file):
        acoustic = json.load(open(acoustic_file)).get("emotions", {})

    # Load label
    label_file = os.path.join(base, "label.txt")
    sds_score = None
    if os.path.exists(label_file):
        with open(label_file) as f:
            lines = f.read().strip().split("\n")
            sds_score = float(lines[0].strip())

    return transcripts, acoustic, sds_score


def build_prompt(transcripts, acoustic):
    """Build enriched prompt with text + acoustic features"""
    parts = []

    for emotion in ["neutral", "negative", "positive"]:
        text = transcripts.get(emotion, "")
        ac = acoustic.get(emotion, {})
        emoji = {"neutral": "\U0001F610", "negative": "\U0001F61F", "positive": "\U0001F60A"}

        block = f"=== {emoji.get(emotion, '')} {emotion} ===\n"
        if text:
            block += f"[对话]: {text}\n"
        if ac:
            block += (
                f"[声学特征]: 音高变化={ac.get('pitch_std',0):.1f}Hz, "
                f"能量变化={ac.get('energy_std',0):.4f}, "
                f"说话比例={ac.get('speech_ratio',0):.1%}, "
                f"停顿比例={ac.get('pause_ratio',0):.1%}, "
                f"jitter={ac.get('jitter',0):.4f}, "
                f"shimmer={ac.get('shimmer',0):.4f}"
            )
        parts.append(block)

    if not any(transcripts.values()):
        return None

    prompt = SYSTEM_PROMPT + "\n\n---\n\n以下是该老年人的访谈记录（含对话文本和语音声学特征）：\n\n"
    prompt += "\n\n".join(parts)
    prompt += "\n\n---\n\n请综合对话内容和声学特征，评估该老年人的抑郁风险，以JSON格式输出。"
    return prompt


def parse_response(text):
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def evaluate(model, processor, device, subject_id, data_dir, acoustic_dir):
    transcripts, acoustic, sds_score = load_subject(data_dir, acoustic_dir, subject_id)
    if not transcripts or all(not t for t in transcripts.values()):
        return {"subject": subject_id, "error": "no transcripts", "sds_score": sds_score}

    prompt = build_prompt(transcripts, acoustic)
    if prompt is None:
        return {"subject": subject_id, "error": "empty prompt", "sds_score": sds_score}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=512, temperature=0.3,
            top_p=0.9, do_sample=True, use_cache=True,
        )

    output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    if "assistant" in output:
        output = output.split("assistant")[-1].strip()

    parsed = parse_response(output)
    pred_depressed = None
    if parsed:
        risk = parsed.get("depression_risk", "").lower()
        prob = parsed.get("depression_probability", 50)
        if risk == "high":
            pred_depressed = True
        elif risk == "low":
            pred_depressed = False
        else:
            pred_depressed = prob >= 50

    return {
        "subject": subject_id,
        "sds_score": sds_score,
        "ground_truth_depressed": sds_score >= 50 if sds_score else None,
        "pred_depressed": pred_depressed,
        "raw_output": output,
        "parsed": parsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/EATD-Corpus/EATD-Corpus")
    parser.add_argument("--acoustic-dir", default="output_eatd_acoustic")
    parser.add_argument("--output", default="output_eatd_mllm_fusion")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--model-path", default="/root/autodl-tmp/.cache/modelscope/qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB)")

    print(f"Loading model...")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    print(f"Model loaded in {time.time()-t0:.1f}s, VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    base = os.path.join(os.path.dirname(__file__), args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    acoustic_dir = os.path.join(os.path.dirname(__file__), args.acoustic_dir) if not os.path.isabs(args.acoustic_dir) else args.acoustic_dir
    output_dir = os.path.join(os.path.dirname(__file__), args.output) if not os.path.isabs(args.output) else args.output
    os.makedirs(output_dir, exist_ok=True)

    all_subjects = sorted([d for d in os.listdir(base) if d.startswith("t_")], key=lambda x: int(x.split("_")[1]))
    if args.end is not None:
        all_subjects = [s for s in all_subjects if args.start <= int(s.split("_")[1]) <= args.end]
    else:
        all_subjects = [s for s in all_subjects if int(s.split("_")[1]) >= args.start]

    print(f"Subjects: {len(all_subjects)} ({all_subjects[0]} ~ {all_subjects[-1]})")

    processed = set()
    if args.resume:
        for f in os.listdir(output_dir):
            if f.endswith(".json") and f.startswith("eatd_"):
                processed.add(f.replace("eatd_", "").replace(".json", ""))
        if processed:
            print(f"Skipping {len(processed)} already done")

    results = []
    for i, subject_id in enumerate(all_subjects):
        if args.resume and subject_id in processed:
            print(f"  [{i+1}/{len(all_subjects)}] {subject_id} skipped")
            continue

        print(f"  [{i+1}/{len(all_subjects)}] {subject_id}...", end=" ", flush=True)
        try:
            result = evaluate(model, processor, device, subject_id, base, acoustic_dir)
            results.append(result)
            out_file = os.path.join(output_dir, f"eatd_{subject_id}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            pred = result.get("pred_depressed")
            gt = result.get("ground_truth_depressed")
            if pred is not None and gt is not None:
                match = "✅" if pred == gt else "❌"
                print(f"SDS={result['sds_score']:.0f} gt={gt} pred={pred} {match}")
            else:
                print(f"pred={pred}")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"subject": subject_id, "error": str(e)})

    # Summary
    print(f"\n{'='*50}")
    valid = [r for r in results if r.get("pred_depressed") is not None and r.get("ground_truth_depressed") is not None]
    if valid:
        y_true = np.array([r["ground_truth_depressed"] for r in valid])
        y_pred = np.array([r["pred_depressed"] for r in valid])
        tp, tn = ((y_true==1)&(y_pred==1)).sum(), ((y_true==0)&(y_pred==0)).sum()
        fp, fn = ((y_true==0)&(y_pred==1)).sum(), ((y_true==1)&(y_pred==0)).sum()
        acc = (tp+tn)/len(y_true)
        print(f"Accuracy:  {acc:.1%} ({tp+tn}/{len(y_true)})")
        print(f"Precision: {tp/(tp+fp):.1%}" if (tp+fp)>0 else "Precision: N/A")
        print(f"Recall:    {tp/(tp+fn):.1%}" if (tp+fn)>0 else "Recall: N/A")
        print(f"F1:        {2*tp/(2*tp+fp+fn):.1%}" if (2*tp+fp+fn)>0 else "F1: N/A")
        print(f"\nConfusion:")
        print(f"              Pred Dep  Pred Health")
        print(f"GT Dep         {tp:>3}         {fn:>3}")
        print(f"GT Health       {fp:>3}         {tn:>3}")

        summary = {
            "total": len(valid), "accuracy": float(acc),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "precision": float(tp/(tp+fp)) if (tp+fp)>0 else None,
            "recall": float(tp/(tp+fn)) if (tp+fn)>0 else None,
            "f1": float(2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn)>0 else None,
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(output_dir, "_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary: {output_dir}/_summary.json")

    print("Done!")


if __name__ == "__main__":
    main()
