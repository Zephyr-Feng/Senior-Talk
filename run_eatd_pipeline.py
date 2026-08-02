"""
Run full B1-B5 pipeline on all EATD audio files
Overrides: use_local=False (no Qwen2.5-VL loading, uses simulated review)
"""
import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Suppress non-critical logs
logging.basicConfig(level=logging.WARNING)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))

from config import DEFAULT_CONFIG, MLLMConfig
from scripts.utils import read_audio, save_daily_json
from scripts.b1_vad import VADProcessor
from scripts.b2_transcribe import WhisperTranscriber
from scripts.b2_acoustic import AcousticFeatureExtractor
from scripts.b3_semantic import SemanticAnalyzer
from scripts.b4_context import MLLMContextProvider
from scripts.b5_mllm_review import MLLMReviewer


def process_subject(subject_id, data_base, config, output_root):
    """Process one subject's 3 emotion WAV files"""
    subj_dir = os.path.join(data_base, subject_id)
    results = {}

    for emotion in ["neutral", "negative", "positive"]:
        wav_path = os.path.join(subj_dir, f"{emotion}.wav")
        if not os.path.exists(wav_path):
            continue

        result_key = f"{subject_id}_{emotion}"
        out_file = os.path.join(output_root, f"pipeline_{result_key}.json")
        if os.path.exists(out_file):
            results[emotion] = json.load(open(out_file))
            continue

        t_start = time.time()
        try:
            # Load audio
            audio, sr = read_audio(wav_path, config.acoustic.sample_rate)
            total_minutes = len(audio) / sr / 60

            # B1: VAD
            vad = VADProcessor(config.vad)
            vad_segments = vad.detect_voice_activity(audio, sr)

            # B1: Lightweight diarization
            try:
                diarization_segments = vad.lightweight_diarize(audio, sr, vad_segments)
                vad_stats = vad.compute_daily_stats(diarization_segments, total_minutes)
            except Exception:
                speech_mins = np.sum(vad_segments[:, 1] - vad_segments[:, 0]) / 60 if len(vad_segments) > 0 else 0
                vad_stats = {
                    "speaking_minutes": round(speech_mins, 2),
                    "interaction_count": 0,
                    "valid_audio_minutes": round(total_minutes, 2),
                    "feature_confidence": 0.5
                }

            # B2: Transcribe
            transcriber = WhisperTranscriber(config.transcription)
            try:
                transcription_result = transcriber.transcribe(wav_path)
            except Exception:
                transcription_result = {
                    "full_text": "", "segments": [], "speech_rate": 0,
                    "word_count": 0, "duration": len(audio)/sr
                }

            # B2: Acoustic features
            acoustic = AcousticFeatureExtractor(config.acoustic)
            acoustic_features = acoustic.compute_acoustic_summary(audio, sr, vad_segments)
            if transcription_result.get("speech_rate", 0) > 0:
                acoustic_features["speech_rate"] = transcription_result["speech_rate"]

            # B3: Semantic
            semantic = SemanticAnalyzer(config.semantic)
            full_text = transcription_result.get("full_text", "")
            semantic_result = semantic.analyze(full_text) if full_text else {
                "semantic_evidence": [], "evidence_count": 0,
                "repetition_count": 0, "safety_flag": False
            }

            # B4: Context
            context_provider = MLLMContextProvider()
            daily_output = context_provider.format_daily_output(
                vad_stats, transcription_result, acoustic_features, semantic_result
            )
            daily_output["full_text"] = full_text

            # B5: MLLM Review (simulated, no local model)
            reviewer = MLLMReviewer(config.mllm)
            daily_output = reviewer.review(daily_output)

            elapsed = time.time() - t_start
            result = {
                "subject": subject_id,
                "emotion": emotion,
                "file": wav_path,
                "elapsed_sec": round(elapsed, 1),
                "pipeline_output": daily_output
            }
            results[emotion] = result

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            word_count = transcription_result.get("word_count", 0)
            print(f"  {subject_id}/{emotion}: {word_count}chars, {elapsed:.0f}s")

        except Exception as e:
            elapsed = time.time() - t_start
            print(f"  {subject_id}/{emotion}: ERROR {e} ({elapsed:.0f}s)")
            results[emotion] = {"subject": subject_id, "emotion": emotion, "error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/EATD-Corpus/EATD-Corpus")
    parser.add_argument("--output", default="output_eatd_pipeline")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Override config: don't load local Qwen2.5-VL (CPU can't handle it)
    config = DEFAULT_CONFIG
    config.mllm.use_local = False
    config.mllm.simulate_without_api = True
    config.mllm.enable_review = True
    config.mllm.api_key = ""  # Clear API key to avoid 403 errors

    print(f"Config: Whisper={config.transcription.model_size.split('/')[-1]}, "
          f"MLLM={'simulated' if not config.mllm.use_local else 'local'}")

    data_base = os.path.join(os.path.dirname(__file__), args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    output_root = os.path.join(os.path.dirname(__file__), args.output) if not os.path.isabs(args.output) else args.output
    os.makedirs(output_root, exist_ok=True)

    all_subjects = sorted(
        [d for d in os.listdir(data_base) if d.startswith("t_")],
        key=lambda x: int(x.split("_")[1])
    )
    if args.end is not None:
        all_subjects = [s for s in all_subjects if args.start <= int(s.split("_")[1]) <= args.end]
    else:
        all_subjects = [s for s in all_subjects if int(s.split("_")[1]) >= args.start]

    print(f"Subjects: {len(all_subjects)} ({all_subjects[0]} ~ {all_subjects[-1]})")
    print(f"Output: {output_root}")

    # Resume
    processed = set()
    if args.resume:
        for f in os.listdir(output_root):
            if f.startswith("pipeline_") and f.endswith(".json"):
                parts = f.replace("pipeline_", "").replace(".json", "").rsplit("_", 1)
                if len(parts) == 2:
                    processed.add(parts[0])
        if processed:
            print(f"Already processed: {len(processed)} subjects")

    total_success = 0
    total_errors = 0
    t0 = time.time()

    for i, subject_id in enumerate(all_subjects):
        if args.resume and subject_id in processed:
            print(f"  [{i+1}/{len(all_subjects)}] {subject_id} → skipped")
            total_success += 1
            continue

        print(f"  [{i+1}/{len(all_subjects)}] {subject_id} processing...")
        results = process_subject(subject_id, data_base, config, output_root)

        for emotion, r in results.items():
            if "error" in r:
                total_errors += 1
            else:
                total_success += 1

        # Log subject summary
        elapsed = time.time() - t0
        rate = (i+1) / elapsed if elapsed > 0 else 0
        remaining = len(all_subjects) - i - 1
        eta = remaining / rate if rate > 0 else 0
        print(f"    → {total_success} files done, {total_errors} errors "
              f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"Complete! {total_success} files, {total_errors} errors")
    print(f"Time: {elapsed:.0f}s ({total_success/elapsed:.1f} files/s)")
    print(f"Output: {output_root}")


if __name__ == "__main__":
    import numpy as np
    main()
