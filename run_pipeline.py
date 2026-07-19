#!/usr/bin/env python3
"""Run the full B1-B5 pipeline on audio file(s)."""
import argparse
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import DEFAULT_CONFIG, MLLMConfig
from scripts.utils import read_audio, save_daily_json
from scripts.b1_vad import VADProcessor
from scripts.b2_transcribe import WhisperTranscriber
from scripts.b2_acoustic import AcousticFeatureExtractor
from scripts.b3_semantic import SemanticAnalyzer
from scripts.b4_context import MLLMContextProvider
from scripts.b5_mllm_review import MLLMReviewer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def process_audio_file(audio_path: str, config, date=None):
    """Process a single audio file through the full pipeline"""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Processing: {audio_path}")
    audio, sr = read_audio(audio_path, config.acoustic.sample_rate)
    total_minutes = len(audio) / sr / 60
    logger.info(f"Audio loaded: {len(audio)/sr:.1f}s, {sr}Hz")

    # B1: VAD
    vad = VADProcessor(config.vad)
    vad_segments = vad.detect_voice_activity(audio, sr)
    logger.info(f"B1: {len(vad_segments)} speech segments")

    # B1: Diarization attempt (try pyannote first, fallback to lightweight)
    try:
        diarization_segments = vad.diarize(audio_path)
        vad_stats = vad.compute_daily_stats(diarization_segments, total_minutes)
    except Exception as e:
        logger.info(f"Pyannote unavailable, trying lightweight diarization: {e}")
        try:
            diarization_segments = vad.lightweight_diarize(audio, sr, vad_segments)
            vad_stats = vad.compute_daily_stats(diarization_segments, total_minutes)
            logger.info(f"Lightweight diarization: {vad_stats['interaction_count']} interactions")
        except Exception as e2:
            logger.warning(f"All diarization failed: {e2}")
            if len(vad_segments) > 0:
                speech_mins = np.sum(vad_segments[:, 1] - vad_segments[:, 0]) / 60
            else:
                speech_mins = 0
            vad_stats = {
                "speaking_minutes": round(speech_mins, 2),
                "interaction_count": 0,
                "valid_audio_minutes": round(total_minutes, 2),
                "feature_confidence": 0.5
            }

    # B2: Transcription
    transcriber = WhisperTranscriber(config.transcription)
    try:
        transcription_result = transcriber.transcribe(audio_path)
        logger.info(f"B2: Transcribed {transcription_result.get('word_count', 0)} chars")
    except Exception as e:
        logger.warning(f"Transcription failed: {e}")
        transcription_result = {
            "full_text": "", "segments": [], "speech_rate": 0,
            "word_count": 0, "duration": len(audio)/sr
        }

    # B2: Acoustic features
    acoustic = AcousticFeatureExtractor(config.acoustic)
    acoustic_features = acoustic.compute_acoustic_summary(audio, sr, vad_segments)
    if transcription_result.get("speech_rate", 0) > 0:
        acoustic_features["speech_rate"] = transcription_result["speech_rate"]
    logger.info(f"B2: {len(acoustic_features)} acoustic features")

    # B3: Semantic analysis
    semantic = SemanticAnalyzer(config.semantic)
    full_text = transcription_result.get("full_text", "")
    semantic_result = semantic.analyze(full_text) if full_text else {
        "semantic_evidence": [], "evidence_count": 0,
        "repetition_count": 0, "safety_flag": False
    }
    logger.info(f"B3: {semantic_result.get('evidence_count', 0)} evidence labels")
    if semantic_result.get("safety_flag"):
        logger.warning("SAFETY FLAG triggered!")

    # B4: Format output
    context_provider = MLLMContextProvider()
    daily_output = context_provider.format_daily_output(
        vad_stats, transcription_result, acoustic_features, semantic_result
    )
    # Add full transcription text for MLLM review
    daily_output["full_text"] = full_text
    logger.info("B4: Output formatted")

    # B5: MLLM Review (natural language summary via Qwen2.5-VL or simulated)
    reviewer = MLLMReviewer(config.mllm)
    daily_output = reviewer.review(daily_output)
    logger.info("B5: MLLM review complete")

    return daily_output


def process_directory(input_dir: str, config):
    """Process all audio files in a directory"""
    input_path = Path(input_dir)
    audio_files = []
    for ext in ["*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg"]:
        audio_files.extend(input_path.rglob(ext))
    if not audio_files:
        logger.warning(f"No audio files found in {input_dir}")
        return
    logger.info(f"Found {len(audio_files)} audio files")
    for af in sorted(audio_files):
        try:
            result = process_audio_file(str(af), config)
            path = save_daily_json(result, config.output_dir, config.user_id,
                                    datetime.now().strftime("%Y-%m-%d"))
            logger.info(f"Saved: {path}")
        except Exception as e:
            logger.error(f"Failed on {af}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Audio Module B Pipeline (B1-B5)")
    parser.add_argument("input", help="Input audio file or directory")
    parser.add_argument("--user", default="default_user", help="User ID")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--date", default=None, help="Date (YYYY-MM-DD)")
    parser.add_argument("--mllm", action="store_true", help="Enable MLLM API review")
    parser.add_argument("--api-key", default=None, help="DashScope API key")
    args = parser.parse_args()
    config = DEFAULT_CONFIG
    config.user_id = args.user
    config.output_dir = args.output

    # Apply CLI overrides for MLLM config
    if args.mllm:
        config.mllm.enable_review = True
    if args.api_key:
        config.mllm.api_key = args.api_key
        config.mllm.enable_review = True

    input_path = Path(args.input)
    if input_path.is_file():
        result = process_audio_file(str(input_path), config, args.date)
        out_path = save_daily_json(result, config.output_dir, config.user_id,
                                    args.date or "unknown")
        print(f"\nResult saved to: {out_path}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif input_path.is_dir():
        process_directory(str(input_path), config)
    else:
        print(f"Error: {args.input} not found")
        sys.exit(1)


if __name__ == "__main__":
    main()
