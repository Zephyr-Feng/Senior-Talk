"""Utility functions for audio module B"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime


def read_audio(path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Read audio file, return (audio_array, sample_rate)"""
    import librosa
    audio, loaded_sr = librosa.load(path, sr=sr, mono=True)
    return audio, loaded_sr


def save_daily_json(output: Dict[str, Any], output_dir: str,
                     user_id: str, date: str) -> Path:
    """Save daily JSON output"""
    path = Path(output_dir) / user_id / f"{date}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    output["user_id"] = user_id
    output["date"] = date
    output["generated_at"] = datetime.now().isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return path


def aggregate_trend(daily_data: list) -> Dict[str, Any]:
    """Aggregate multi-day trends (7-day / 14-day)"""
    if not daily_data:
        return {}
    numeric_keys = [
        "speaking_minutes", "interaction_count", "speech_rate",
        "pause_ratio", "pitch_variability", "energy_variability",
        "repetition_count", "valid_audio_minutes", "feature_confidence"
    ]
    trend = {}
    for key in numeric_keys:
        values = [d.get(key, 0) or 0 for d in daily_data]
        if values:
            trend[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "trend": "up" if len(values) > 1 and values[-1] > values[0]
                        else "down" if len(values) > 1 and values[-1] < values[0]
                        else "stable"
            }
    return trend
