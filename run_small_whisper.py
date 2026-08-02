import json, os, sys, numpy as np

with open("/tmp/review_manifest.json") as f:
    manifest = json.load(f)

sys.path.insert(0, "/root/autodl-tmp/senior_project")
from config import PipelineConfig, TranscriptionConfig
from scripts.b1_vad import VADProcessor
from scripts.b2_transcribe import WhisperTranscriber
from scripts.b2_acoustic import AcousticFeatureExtractor
from scripts.b3_semantic import SemanticAnalyzer
from scripts.utils import read_audio

# Use small model instead of tiny
config = PipelineConfig()
config.transcription.model_size = "/root/autodl-tmp/.cache/modelscope/XXXXRT/faster-whisper/faster-whisper-small"

print(f"Using model: {config.transcription.model_size}")
print(f"Processing {len(manifest)} files...")

results = []
for i, item in enumerate(manifest):
    wav_path = item["wav_path"]
    item_id = item["id"]
    print(f"  [{item_id}/{len(manifest)}] {item['speaker']}/{item['emotion']} ({item['duration']}s)...", end="", flush=True)

    try:
        audio, sr = read_audio(wav_path, config.acoustic.sample_rate)

        # B1: VAD
        vad = VADProcessor(config.vad)
        vad_segments = vad.detect_voice_activity(audio, sr)

        # B2: Transcription with SMALL model
        transcriber = WhisperTranscriber(config.transcription)
        transcription_result = transcriber.transcribe(wav_path)
        full_text = transcription_result.get("full_text", "")

        # B2: Acoustic
        acoustic = AcousticFeatureExtractor(config.acoustic)
        acoustic_features = acoustic.compute_acoustic_summary(audio, sr, vad_segments)
        if transcription_result.get("speech_rate", 0) > 0:
            acoustic_features["speech_rate"] = transcription_result["speech_rate"]

        # B3: Semantic
        semantic = SemanticAnalyzer(config.semantic)
        semantic_result = semantic.analyze(full_text) if full_text else {"semantic_evidence": [], "evidence_count": 0, "repetition_count": 0, "safety_flag": False}

        results.append({
            "id": item_id,
            "speaker": item["speaker"],
            "emotion": item["emotion"],
            "duration": item["duration"],
            "gt_text": item["gt_text"],
            "sds": item["sds"],
            "whisper_text": full_text,
            "speech_rate": acoustic_features.get("speech_rate", 0),
            "pause_ratio": acoustic_features.get("pause_ratio", 0),
            "evidence_count": semantic_result.get("evidence_count", 0),
            "semantic_evidence": semantic_result.get("semantic_evidence", []),
            "repetition_count": semantic_result.get("repetition_count", 0),
            "safety_flag": semantic_result.get("safety_flag", False),
        })
        print(f" {len(full_text)}chr ev={semantic_result.get('evidence_count', 0)}")
    except Exception as e:
        print(f" ERROR: {e}")
        results.append({"id": item_id, "speaker": item["speaker"], "emotion": item["emotion"], "duration": item["duration"], "gt_text": item["gt_text"], "whisper_text": f"[ERROR] {e}"})

with open("/tmp/review_results_small.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nDone: {len(results)} files")
print("Saved to /tmp/review_results_small.json")
