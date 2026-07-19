"""B4: MLLM Context Provider - prepares context for Qwen2.5-VL"""
from pathlib import Path
from typing import Dict, Any, List
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class MLLMContextProvider:
    def __init__(self):
        pass

    def prepare_context(self, transcription, speaker_label,
                         acoustic_summary, semantic_evidence, time_window):
        """Prepare context for Qwen2.5-VL video-frame review"""
        return {
            "time_window": {
                "start_sec": time_window.start_sec,
                "end_sec": time_window.end_sec
            },
            "transcription": transcription,
            "speaker": speaker_label,
            "acoustic_context": {
                "speech_rate": acoustic_summary.get("speech_rate", 0),
                "pause_ratio": acoustic_summary.get("pause_ratio", 0),
                "pitch_variability": acoustic_summary.get("pitch_variability", 0),
                "energy_variability": acoustic_summary.get("energy_variability", 0),
                "voice_quality": {
                    "jitter": acoustic_summary.get("jitter", 0),
                    "shimmer": acoustic_summary.get("shimmer", 0)
                }
            },
            "semantic_hints": {
                "evidence_labels": semantic_evidence.get("evidence_labels", []),
                "evidence_count": semantic_evidence.get("evidence_count", 0),
                "safety_flag": semantic_evidence.get("safety_flag", False)
            },
            "note": "Transcription is context only. Do NOT discard acoustic branch."
        }

    def format_daily_output(self, vad_stats, transcription_result,
                             acoustic_features, semantic_result):
        """Format the complete daily output per spec"""
        return {
            "speaking_minutes": vad_stats.get("speaking_minutes", 0),
            "interaction_count": vad_stats.get("interaction_count", 0),
            "speech_rate": transcription_result.get("speech_rate",
                           acoustic_features.get("speech_rate", 0)),
            "pause_ratio": acoustic_features.get("pause_ratio", 0),
            "pitch_variability": acoustic_features.get("pitch_variability", 0),
            "energy_variability": acoustic_features.get("energy_variability", 0),
            "repetition_count": semantic_result.get("repetition_count", 0),
            "semantic_evidence": semantic_result.get("semantic_evidence", []),
            "valid_audio_minutes": vad_stats.get("valid_audio_minutes", 0),
            "feature_confidence": vad_stats.get("feature_confidence", 0),
            "safety_flag": semantic_result.get("safety_flag", False)
        }
