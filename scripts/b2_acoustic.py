"""B2: Acoustic Feature Extraction"""
import numpy as np
from pathlib import Path
from typing import Dict, Any
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AcousticConfig


class AcousticFeatureExtractor:
    def __init__(self, config: AcousticConfig = AcousticConfig()):
        self.config = config

    def extract(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract comprehensive acoustic features"""
        import librosa
        features = {}
        features.update(self._extract_prosody(audio, sr))
        features.update(self._extract_spectral(audio, sr))
        features.update(self._extract_voice_quality(audio, sr))
        return features

    def _extract_prosody(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract prosodic features: pitch, energy"""
        import librosa
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=self.config.f0_min, fmax=self.config.f0_max, sr=sr
        )
        f0 = np.array(f0)
        f0_voiced = f0[voiced_flag] if any(voiced_flag) else np.array([0.0])
        # Filter out NaN/Inf from failed pitch tracking
        f0_voiced = f0_voiced[np.isfinite(f0_voiced)]
        if len(f0_voiced) == 0:
            f0_voiced = np.array([0.0])
        rms = librosa.feature.rms(y=audio, frame_length=self.config.frame_length,
                                   hop_length=self.config.hop_length)[0]
        pitch_var = float(np.std(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        energy_var = float(np.std(rms))
        return {
            "pitch_mean": float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0,
            "pitch_std": pitch_var,
            "pitch_variability": round(pitch_var, 4),
            "energy_mean": float(np.mean(rms)),
            "energy_std": energy_var,
            "energy_variability": round(energy_var, 4),
            "voiced_ratio": float(np.mean(voiced_flag)) if any(voiced_flag) else 0.0
        }

    def _extract_spectral(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract spectral features"""
        import librosa
        sc = librosa.feature.spectral_centroid(
            y=audio, sr=sr, hop_length=self.config.hop_length
        )[0]
        sb = librosa.feature.spectral_bandwidth(
            y=audio, sr=sr, hop_length=self.config.hop_length
        )[0]
        zcr = librosa.feature.zero_crossing_rate(
            y=audio, hop_length=self.config.hop_length
        )[0]
        return {
            "spectral_centroid_mean": float(np.mean(sc)),
            "spectral_bandwidth_mean": float(np.mean(sb)),
            "zero_crossing_rate_mean": float(np.mean(zcr))
        }

    def _extract_voice_quality(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract voice quality features (Jitter, Shimmer)"""
        try:
            import parselmouth
            import soundfile as sf
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, sr)
                snd = parselmouth.Sound(f.name)
                pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
                jitter = parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                shimmer = parselmouth.praat.call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            return {"jitter": round(jitter, 6), "shimmer": round(shimmer, 6)}
        except Exception as e:
            return {"jitter": 0.0, "shimmer": 0.0, "voice_quality_error": str(e)}

    def compute_pause_ratio(self, audio: np.ndarray, sr: int, vad_segments: np.ndarray) -> float:
        """Compute pause ratio from VAD segments"""
        total = len(audio) / sr
        speech = np.sum(vad_segments[:, 1] - vad_segments[:, 0]) if len(vad_segments) > 0 else 0
        silence = total - speech
        return round(silence / max(total, 0.001), 4)

    def compute_acoustic_summary(self, audio, sr, vad_segments):
        """Compute full acoustic summary for daily output"""
        features = self.extract(audio, sr)
        pause_ratio = self.compute_pause_ratio(audio, sr, vad_segments)
        return {
            "speech_rate": features.get("speech_rate", 0),
            "pause_ratio": pause_ratio,
            "pitch_variability": features.get("pitch_variability", 0),
            "energy_variability": features.get("energy_variability", 0),
            "jitter": features.get("jitter", 0),
            "shimmer": features.get("shimmer", 0),
            "spectral_centroid": features.get("spectral_centroid_mean", 0),
            "voiced_ratio": features.get("voiced_ratio", 0)
        }
