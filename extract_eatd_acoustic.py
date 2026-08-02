"""
EATD Acoustic Feature Extraction (B2-based)
- Extracts prosody, spectral, voice quality features
- Simple energy-based VAD for pause info
- Output: JSON per emotion per subject
"""
import os
import json
import sys
import time
import argparse
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def extract_acoustic(audio, sr):
    """Extract all acoustic features from audio"""
    import librosa

    features = {}

    # ---- Prosody: pitch ----
    f0, voiced_flag, _ = librosa.pyin(
        audio, fmin=65, fmax=500, sr=sr
    )
    f0 = np.array(f0)
    f0_voiced = f0[voiced_flag] if any(voiced_flag) else np.array([0.0])
    features["pitch_mean"] = round(float(np.mean(f0_voiced)), 2) if len(f0_voiced) > 0 else 0.0
    features["pitch_std"] = round(float(np.std(f0_voiced)), 2) if len(f0_voiced) > 0 else 0.0
    features["voiced_ratio"] = round(float(np.mean(voiced_flag)), 4) if any(voiced_flag) else 0.0

    # ---- Prosody: energy ----
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    features["energy_mean"] = round(float(np.mean(rms)), 4)
    features["energy_std"] = round(float(np.std(rms)), 4)

    # ---- Spectral ----
    sc = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=512)[0]
    sb = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=512)[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=512)[0]
    features["spectral_centroid_mean"] = round(float(np.mean(sc)), 2)
    features["spectral_bandwidth_mean"] = round(float(np.mean(sb)), 2)
    features["zcr_mean"] = round(float(np.mean(zcr)), 4)

    # ---- Voice quality (jitter/shimmer via parselmouth) ----
    try:
        import parselmouth
        import soundfile as sf
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            fname = f.name
            sf.write(fname, audio, sr)
            snd = parselmouth.Sound(fname)
            pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
            features["jitter"] = round(parselmouth.praat.call(
                pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3), 6)
            features["shimmer"] = round(parselmouth.praat.call(
                [snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6), 6)
            os.unlink(fname)
    except Exception as e:
        features["jitter"] = 0.0
        features["shimmer"] = 0.0
        features["voice_quality_error"] = str(e)

    # ---- Simple energy-based VAD ----
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    energy = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=hop_len)[0]
    threshold = np.max(energy) * 0.05  # 5% of max energy
    is_speech = energy > threshold
    speech_frames = np.sum(is_speech)
    total_frames = len(is_speech)
    features["speech_ratio"] = round(float(speech_frames / max(total_frames, 1)), 4)
    features["pause_ratio"] = round(1.0 - features["speech_ratio"], 4)
    features["duration_sec"] = round(len(audio) / sr, 2)

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/EATD-Corpus/EATD-Corpus")
    parser.add_argument("--output", default="output_eatd_acoustic")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    import librosa

    base = os.path.join(os.path.dirname(__file__), args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    output_dir = os.path.join(os.path.dirname(__file__), args.output) if not os.path.isabs(args.output) else args.output
    os.makedirs(output_dir, exist_ok=True)

    all_subjects = sorted(
        [d for d in os.listdir(base) if d.startswith("t_")],
        key=lambda x: int(x.split("_")[1])
    )
    if args.end is not None:
        all_subjects = [s for s in all_subjects if args.start <= int(s.split("_")[1]) <= args.end]
    else:
        all_subjects = [s for s in all_subjects if int(s.split("_")[1]) >= args.start]

    print(f"Subjects: {len(all_subjects)} ({all_subjects[0]} ~ {all_subjects[-1]})")
    print(f"Output: {output_dir}")

    total_files = 0
    t0 = time.time()

    for i, subject_id in enumerate(all_subjects):
        subj_dir = os.path.join(base, subject_id)
        out_file = os.path.join(output_dir, f"{subject_id}.json")

        # Skip if already done
        if os.path.exists(out_file):
            continue

        features = {"subject": subject_id, "emotions": {}}

        for emotion in ["neutral", "negative", "positive"]:
            wav_file = os.path.join(subj_dir, f"{emotion}.wav")
            if not os.path.exists(wav_file):
                continue

            try:
                audio, sr = librosa.load(wav_file, sr=16000)
                feat = extract_acoustic(audio, sr)
                feat["emotion"] = emotion
                feat["file"] = wav_file
                features["emotions"][emotion] = feat
                total_files += 1
            except Exception as e:
                print(f"  ERROR {subject_id}/{emotion}: {e}")

        # Save per subject
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(features, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - t0
        rate = (i+1) / elapsed if elapsed > 0 else 0
        eta = (len(all_subjects) - i - 1) / rate if rate > 0 else 0
        print(f"  [{i+1}/{len(all_subjects)}] {subject_id} → {total_files} files ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone! {total_files} files in {elapsed:.0f}s ({total_files/elapsed:.1f} files/s)")


if __name__ == "__main__":
    main()
