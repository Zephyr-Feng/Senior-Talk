#!/usr/bin/env python3
"""Test the audio module B pipeline with synthetic audio data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import soundfile as sf
from config import VADConfig
from scripts.b1_vad import VADProcessor
from scripts.b2_acoustic import AcousticFeatureExtractor
from scripts.b3_semantic import SemanticAnalyzer
from scripts.b4_context import MLLMContextProvider


def gen_audio(duration=10.0, sr=16000, pattern="speech"):
    t = np.linspace(0, duration, int(sr * duration), False)
    if pattern == "speech":
        # Generate continuous speech-like audio with long energy blocks
        sig = np.random.normal(0, 0.005, len(t))
        # Two long speech blocks (each > 1 second)
        for block_start in [0.2, 2.5]:
            bs = int(block_start * sr)
            be = min(bs + int(1.5 * sr), len(t))
            n = be - bs
            if n > 0:
                bt = np.linspace(0, n/sr, n)
                block = np.sin(2 * np.pi * (250 + 50 * np.sin(2 * np.pi * 2 * bt)) * bt) * 0.8
                block *= np.hanning(n)
                sig[bs:be] += block
        sig = sig / max(np.max(np.abs(sig)), 0.001) * 0.9
    elif pattern == "silence":
        sig = np.random.normal(0, 0.005, len(t))
    elif pattern == "tv_noise":
        sig = np.sin(2 * np.pi * 800 * t) * 0.3 + np.sin(2 * np.pi * 1200 * t) * 0.2
        sig += np.random.normal(0, 0.01, len(t))
    else:
        sig = np.random.normal(0, 0.01, len(t))
    sig = sig / max(np.max(np.abs(sig)), 0.001)
    return sig, sr


def test_b1_vad():
    print("\n=== B1 VAD Test ===")
    vad = VADProcessor(VADConfig(min_speech_duration_ms=100.0))
    audio, sr = gen_audio(5.0, pattern="speech")
    segments = vad.detect_voice_activity(audio, sr)
    print(f"  Speech audio: {len(segments)} VAD segments")
    assert len(segments) > 0, "Should detect speech"
    if len(segments) > 0:
        total_speech = np.sum(segments[:, 1] - segments[:, 0])
        print(f"  Total speech: {total_speech:.2f}s / 5.00s")
    audio_s, _ = gen_audio(5.0, pattern="silence")
    seg_s = vad.detect_voice_activity(audio_s, _)
    print(f"  Silence audio: {len(seg_s)} VAD segments")
    stats = vad.compute_daily_stats([], 10.0)
    assert stats["speaking_minutes"] == 0.0
    print("  PASSED")


def test_b2_acoustic():
    print("\n=== B2 Acoustic Test ===")
    from config import AcousticConfig
    extractor = AcousticFeatureExtractor(AcousticConfig())
    audio, sr = gen_audio(5.0, pattern="speech")
    vad = VADProcessor(VADConfig())
    segs = vad.detect_voice_activity(audio, sr)
    features = extractor.extract(audio, sr)
    print(f"  Features: {list(features.keys())}")
    assert "pitch_variability" in features
    assert features["pitch_variability"] >= 0
    pr = extractor.compute_pause_ratio(audio, sr, segs)
    print(f"  Pause ratio: {pr}")
    assert 0 <= pr <= 1
    summary = extractor.compute_acoustic_summary(audio, sr, segs)
    assert "jitter" in summary
    print(f"  Jitter: {summary['jitter']}, Shimmer: {summary['shimmer']}")
    print("  PASSED")


def test_b3_semantic():
    print("\n=== B3 Semantic Test ===")
    from config import SemanticConfig
    analyzer = SemanticAnalyzer(SemanticConfig())
    r1 = analyzer.analyze("最近总是睡不着，半夜就醒了")
    assert "sleep_complaint" in r1["evidence_labels"]
    print(f"  Sleep complaint: detected")
    r2 = analyzer.analyze("每天就我一个人在家，没人说话，很孤独")
    assert "loneliness" in r2["evidence_labels"]
    print(f"  Loneliness: detected")
    r3 = analyzer.analyze("不如死了算了")
    assert r3["safety_flag"] == True
    print(f"  Safety flag: {r3['safety_flag']}")
    r4 = analyzer.analyze("")
    assert r4["evidence_count"] == 0
    print(f"  Empty text: OK")
    print("  PASSED")


def test_b4_context():
    print("\n=== B4 Context Test ===")
    provider = MLLMContextProvider()
    tw = type("TW", (), {"start_sec": 10.0, "end_sec": 30.0})
    ctx = provider.prepare_context(
        "今天感觉不太好", "SPEAKER_00",
        {"speech_rate": 2.5, "pause_ratio": 0.3, "pitch_variability": 0.1,
         "energy_variability": 0.05, "jitter": 0.03, "shimmer": 0.1},
        {"evidence_labels": ["sleep_complaint"], "evidence_count": 1, "safety_flag": False},
        tw
    )
    assert "time_window" in ctx
    assert ctx["time_window"]["start_sec"] == 10.0
    print(f"  Context: {len(ctx)} keys")
    daily = provider.format_daily_output(
        {"speaking_minutes": 30.5, "interaction_count": 5,
         "valid_audio_minutes": 60.0, "feature_confidence": 0.85},
        {"speech_rate": 3.2},
        {"pause_ratio": 0.25, "pitch_variability": 0.15, "energy_variability": 0.08},
        {"repetition_count": 3, "semantic_evidence": [], "safety_flag": False}
    )
    assert daily["speaking_minutes"] == 30.5
    print(f"  Daily output: {list(daily.keys())}")
    print("  PASSED")


def main():
    print("=" * 40)
    print("Audio Module B - Pipeline Tests")
    print("=" * 40)
    tests = [test_b1_vad, test_b2_acoustic, test_b3_semantic, test_b4_context]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{'=' * 40}")
    print(f"Results: {passed}/{len(tests)} passed")
    print(f"{'=' * 40}")
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
