"""
EATD-Corpus MLLM Batch Evaluation (Local Qwen2.5-VL)
- Reads 3 transcripts (neutral/negative/positive) per subject
- Asks Qwen2.5-VL to assess depression risk
- Compares with SDS label (SDS >= 50 = depressed)
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

# Ensure OMP_NUM_THREADS is set properly
os.environ.pop("OMP_NUM_THREADS", None)

sys.path.insert(0, str(Path(__file__).parent))
from config import MLLMConfig

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


# ---- Prompt ----
SYSTEM_PROMPT = """你是一位专业的老年心理健康评估助理。你的任务是根据老年人的访谈对话文本，评估其抑郁风险。

老年人会谈论不同情绪状态的话题：中性、消极、积极。

请根据以下对话文本，分析老年人的心理状态，并给出评估结果。

你的分析需要关注以下方面：
1. 情绪状态：是否表现出持续的悲伤、焦虑、兴趣丧失等抑郁核心症状
2. 认知功能：思维是否清晰，有无认知衰退迹象（重复、混乱等）
3. 睡眠食欲：是否提到睡眠问题或食欲变化
4. 社交功能：社交互动和日常活动是否受影响
5. 精力水平：是否经常感到疲劳、精力不足

请以JSON格式返回评估结果（不要用markdown包裹）：
{
  "depression_risk": "high"或"moderate"或"low",
  "depression_probability": 0-100的数值,
  "reasoning": "简要推理过程",
  "key_signs": ["关键症状1", "关键症状2", ...],
  "confidence": "high"或"medium"或"low"
}

注意：depression_probability 是0-100的数值，表示你认为该老年人患有抑郁症的可能性。"""


def load_subject(data_dir, subject_id):
    """Load a subject's 3 transcripts"""
    base = os.path.join(data_dir, subject_id)
    transcripts = {}
    for emotion in ["neutral", "negative", "positive"]:
        txt_file = os.path.join(base, f"{emotion}.txt")
        if os.path.exists(txt_file):
            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
            transcripts[emotion] = text
        else:
            transcripts[emotion] = ""
    # Load label
    label_file = os.path.join(base, "label.txt")
    sds_score = None
    if os.path.exists(label_file):
        with open(label_file) as f:
            lines = f.read().strip().split("\n")
            sds_score = float(lines[0].strip())
    return transcripts, sds_score


def build_prompt(transcripts):
    """Build prompt with 3 transcripts"""
    parts = []
    for emotion in ["neutral", "negative", "positive"]:
        text = transcripts.get(emotion, "")
        if text:
            emoji = {"neutral": "\U0001F610", "negative": "\U0001F61F", "positive": "\U0001F60A"}
            parts.append(f"=== {emoji.get(emotion, '')} 情绪状态：{emotion} ===\n{text}")

    if not parts:
        return None

    prompt = SYSTEM_PROMPT + "\n\n---\n\n以下是该老年人的访谈记录：\n\n"
    prompt += "\n\n".join(parts)
    prompt += "\n\n---\n\n请分析以上对话，以JSON格式输出该老年人的抑郁风险评估结果。"
    return prompt


def parse_response(text):
    """Parse JSON from model response, stripping markdown if needed"""
    # Remove markdown code fences
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)

    # Try to find JSON object
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def evaluate(model, processor, device, subject_id, data_dir):
    """Evaluate one subject"""
    transcripts, sds_score = load_subject(data_dir, subject_id)

    if all(not t for t in transcripts.values()):
        return {"subject": subject_id, "error": "no transcripts", "sds_score": sds_score}

    prompt = build_prompt(transcripts)
    if prompt is None:
        return {"subject": subject_id, "error": "empty prompt", "sds_score": sds_score}

    # Format with chat template
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            use_cache=True,
        )

    output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Extract assistant response
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
            # moderate: use probability threshold
            pred_depressed = prob >= 50

    result = {
        "subject": subject_id,
        "sds_score": sds_score,
        "ground_truth_depressed": sds_score >= 50 if sds_score else None,
        "pred_depressed": pred_depressed,
        "raw_output": output,
        "parsed": parsed,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/EATD-Corpus/EATD-Corpus")
    parser.add_argument("--output", default="output_eatd_mllm")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model-path", default="/root/autodl-tmp/.cache/modelscope/qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--resume", action="store_true", help="Skip already processed subjects")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB)")

    # Load model
    print(f"Loading model from {args.model_path}...")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    print(f"Model loaded in {time.time()-t0:.1f}s")
    print(f"Model device: {model.device}")
    print(f"VRAM used: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    # Load subjects
    base = os.path.join(os.path.dirname(__file__), args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    all_subjects = sorted(
        [d for d in os.listdir(base) if d.startswith("t_")],
        key=lambda x: int(x.split("_")[1])
    )

    if args.end is not None:
        all_subjects = [s for s in all_subjects if args.start <= int(s.split("_")[1]) <= args.end]
    else:
        all_subjects = [s for s in all_subjects if int(s.split("_")[1]) >= args.start]

    print(f"Subjects to process: {len(all_subjects)} ({all_subjects[0]} ~ {all_subjects[-1]})")

    # Prepare output
    output_dir = os.path.join(os.path.dirname(__file__), args.output) if not os.path.isabs(args.output) else args.output
    os.makedirs(output_dir, exist_ok=True)

    # Resume check
    processed = set()
    if args.resume:
        for f in os.listdir(output_dir):
            if f.endswith(".json") and f.startswith("eatd_"):
                sid = f.replace("eatd_", "").replace(".json", "")
                processed.add(sid)
        if processed:
            print(f"Already processed: {len(processed)} subjects, skipping")

    results = []
    total = len(all_subjects)
    errors = 0

    for i, subject_id in enumerate(all_subjects):
        if args.resume and subject_id in processed:
            print(f"  [{i+1}/{total}] {subject_id} → skipped (already processed)")
            continue

        print(f"  [{i+1}/{total}] {subject_id}...", end=" ", flush=True)
        try:
            result = evaluate(model, processor, device, subject_id, base)
            results.append(result)

            # Save individual result
            out_file = os.path.join(output_dir, f"eatd_{subject_id}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            pred = result.get("pred_depressed")
            gt = result.get("ground_truth_depressed")
            if pred is not None and gt is not None:
                match = "✅" if pred == gt else "❌"
                print(f"SDS={result['sds_score']:.0f} gt={gt} pred={pred} {match}")
            else:
                print(f"SDS={result.get('sds_score')} pred={pred} (parse error?)")

        except Exception as e:
            errors += 1
            print(f"ERROR: {e}")
            results.append({"subject": subject_id, "error": str(e)})

        # VRAM check
        if (i+1) % 5 == 0 and torch.cuda.is_available():
            print(f"    VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")

    # Summary
    print(f"\n{'='*50}")
    print(f"Completed: {len(results)} subjects, {errors} errors")

    valid = [r for r in results if r.get("pred_depressed") is not None and r.get("ground_truth_depressed") is not None]
    if valid:
        y_true = np.array([r["ground_truth_depressed"] for r in valid])
        y_pred = np.array([r["pred_depressed"] for r in valid])

        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        acc = (tp + tn) / len(y_true)

        print(f"Accuracy:  {acc:.1%} ({tp+tn}/{len(y_true)})")
        print(f"Precision: {tp/(tp+fp):.1%}" if (tp+fp) > 0 else "Precision: N/A")
        print(f"Recall:    {tp/(tp+fn):.1%}" if (tp+fn) > 0 else "Recall: N/A")
        print(f"Specificity: {tn/(tn+fp):.1%}" if (tn+fp) > 0 else "Specificity: N/A")
        print(f"F1:        {2*tp/(2*tp+fp+fn):.1%}" if (2*tp+fp+fn) > 0 else "F1: N/A")
        print(f"\nConfusion Matrix:")
        print(f"              Pred Depressed  Pred Healthy")
        print(f"GT Depressed       {tp:>3}             {fn:>3}")
        print(f"GT Healthy          {fp:>3}             {tn:>3}")

        # Save summary
        summary = {
            "total": len(valid),
            "accuracy": float(acc),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "precision": float(tp/(tp+fp)) if (tp+fp) > 0 else None,
            "recall": float(tp/(tp+fn)) if (tp+fn) > 0 else None,
            "specificity": float(tn/(tn+fp)) if (tn+fp) > 0 else None,
            "f1": float(2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn) > 0 else None,
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(output_dir, "_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {output_dir}/_summary.json")

    print("Done!")


if __name__ == "__main__":
    main()
