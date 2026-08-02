import json, os, sys, time, numpy as np

with open("/tmp/review_manifest.json") as f:
    manifest = json.load(f)

sys.path.insert(0, "/root/autodl-tmp/senior_project")
from config import PipelineConfig
from scripts.b1_vad import VADProcessor
from scripts.b2_transcribe import WhisperTranscriber
from scripts.b2_acoustic import AcousticFeatureExtractor
from scripts.b3_semantic import SemanticAnalyzer
from scripts.utils import read_audio

config = PipelineConfig()
config.transcription.model_size = "/root/autodl-tmp/.cache/modelscope/XXXXRT/faster-whisper/faster-whisper-small"

print("Processing 50 files with Whisper small...")
results = []
times = []
for i, item in enumerate(manifest):
    wav = item["wav_path"]
    sid = item["id"]
    t0 = time.time()
    print(f"  [{sid}/50] {item['speaker']}/{item['emotion']} ({item['duration']}s)...", end=" ", flush=True)
    try:
        audio, sr = read_audio(wav, config.acoustic.sample_rate)
        vad = VADProcessor(config.vad)
        vs = vad.detect_voice_activity(audio, sr)
        tr = WhisperTranscriber(config.transcription)
        tx = tr.transcribe(wav)
        ft = tx.get("full_text", "")
        ac = AcousticFeatureExtractor(config.acoustic)
        af = ac.compute_acoustic_summary(audio, sr, vs)
        if tx.get("speech_rate", 0) > 0:
            af["speech_rate"] = tx["speech_rate"]
        sm = SemanticAnalyzer(config.semantic)
        sr2 = sm.analyze(ft) if ft else {"semantic_evidence": [], "evidence_count": 0, "repetition_count": 0, "safety_flag": False}
        elapsed = time.time() - t0
        times.append(elapsed)
        results.append({
            "id": sid, "speaker": item["speaker"], "emotion": item["emotion"],
            "duration": item["duration"], "gt_text": item["gt_text"], "sds": item["sds"],
            "whisper_text": ft, "speech_rate": af.get("speech_rate", 0),
            "pause_ratio": af.get("pause_ratio", 0),
            "evidence_count": sr2.get("evidence_count", 0),
            "semantic_evidence": sr2.get("semantic_evidence", []),
            "repetition_count": sr2.get("repetition_count", 0),
            "safety_flag": sr2.get("safety_flag", False),
        })
        predicted = (50 - i - 1) * (sum(times)/len(times)) if times else 0
        print(str(len(ft)) + "chr ev=" + str(sr2.get("evidence_count", 0)) + " [" + str(round(elapsed, 1)) + "s, ~" + str(round(predicted/60, 1)) + "min left]")
    except Exception as e:
        elapsed = time.time() - t0
        print("ERROR: " + str(e))
        results.append({"id": sid, "speaker": item["speaker"], "emotion": item["emotion"], "duration": item["duration"], "gt_text": item["gt_text"], "whisper_text": "[ERROR] " + str(e)})

with open("/tmp/review_results_small.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

avg = sum(times)/len(times) if times else 0
print("Done: " + str(len(results)) + " files, avg " + str(round(avg, 1)) + "s/file")
