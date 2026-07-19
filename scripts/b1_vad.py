"""B1: Voice Activity Detection & Speaker Diarization"""
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import VADConfig


class VADProcessor:
    def __init__(self, config: VADConfig = VADConfig()):
        self.config = config
        self._diarization_pipeline = None

    def _init_diarization(self):
        if self._diarization_pipeline is None:
            from pyannote.audio import Pipeline
            self._diarization_pipeline = Pipeline.from_pretrained(
                self.config.diarization_model
            )
        return self._diarization_pipeline

    def detect_voice_activity(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Simple energy-based VAD"""
        frame_len = int(sr * self.config.frame_duration_ms / 1000)
        energy = np.array([
            np.sum(audio[i:i+frame_len]**2)
            for i in range(0, len(audio) - frame_len, frame_len)
        ])
        # Use median-based adaptive threshold (more robust)
        energy_mean = np.mean(energy)
        energy_std = np.std(energy)
        threshold = energy_mean + energy_std * (self.config.threshold - 0.5) * 2
        # Fallback to mean * threshold if the above is too aggressive
        threshold = max(threshold, energy_mean * self.config.threshold * 0.5)
        segments = []
        in_speech = False
        start = 0
        for i, e in enumerate(energy):
            if e > threshold and not in_speech:
                start = i * frame_len / sr
                in_speech = True
            elif e <= threshold and in_speech:
                duration = (i * frame_len / sr) - start
                if duration >= self.config.min_speech_duration_ms / 1000:
                    segments.append([start, i * frame_len / sr])
                in_speech = False
        if in_speech:
            segments.append([start, len(audio) / sr])
        return np.array(segments) if segments else np.array([]).reshape(0, 2)

    def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> List[Dict]:
        """Speaker diarization using pyannote (requires HF token + GPU)"""
        pipeline = self._init_diarization()
        kwargs = {}
        if num_speakers:
            kwargs["num_speakers"] = num_speakers
        diarization = pipeline(audio_path, **kwargs)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start, "end": turn.end,
                "speaker": speaker, "duration": turn.end - turn.start
            })
        return segments

    def lightweight_diarize(self, audio: np.ndarray, sr: int, vad_segments: np.ndarray) -> List[Dict]:
        """Lightweight speaker diarization using acoustic features (CPU-only, no extra models).

        Extracts pitch, energy, and spectral features per VAD segment,
        then clusters segments into speaker groups using simple k-means.
        """
        if len(vad_segments) == 0:
            return []

        import librosa

        features = []       # feature vectors for valid segments
        valid_idxs = []     # original segment indices that are valid

        for i, (start, end) in enumerate(vad_segments):
            s = int(start * sr)
            e = int(end * sr)
            seg = audio[s:e]
            seg_len = len(seg) / sr

            # Skip very short segments (less than 150ms)
            if seg_len < 0.15:
                continue

            feat = []

            # 1. Pitch statistics
            try:
                f0, voiced_flag, _ = librosa.pyin(
                    seg, fmin=50, fmax=500, sr=sr
                )
                f0 = np.array(f0)
                f0_voiced = f0[voiced_flag] if any(voiced_flag) else f0
                if len(f0_voiced) > 0:
                    feat.append(float(np.mean(f0_voiced)))  # pitch mean
                    feat.append(float(np.std(f0_voiced)))   # pitch std
                else:
                    feat.extend([0.0, 0.0])
            except Exception:
                feat.extend([0.0, 0.0])

            # 2. Energy (RMS)
            try:
                rms = librosa.feature.rms(y=seg, frame_length=2048, hop_length=512)[0]
                feat.append(float(np.mean(rms)))
            except Exception:
                feat.append(0.0)

            # 3. Spectral centroid (brightness / voice quality)
            try:
                sc = librosa.feature.spectral_centroid(y=seg, sr=sr, hop_length=512)[0]
                feat.append(float(np.mean(sc)))
            except Exception:
                feat.append(0.0)

            features.append(feat)
            valid_idxs.append(i)

        n_valid = len(features)

        # If too few segments, treat as single speaker
        if n_valid < 3:
            result = []
            for start, end in vad_segments:
                result.append({
                    "start": start, "end": end,
                    "speaker": "SPEAKER_00",
                    "duration": end - start
                })
            return result

        # Normalize features
        feat_arr = np.array(features)
        f_mean = np.mean(feat_arr, axis=0)
        f_std = np.std(feat_arr, axis=0)
        f_std = np.where(f_std < 1e-10, 1.0, f_std)
        norm = (feat_arr - f_mean) / f_std

        # Simple k-means with k=2 (elderly vs other), pure numpy
        # Initialize: use min and max energy as seeds (elderly vs visitor often differ in energy)
        k = min(2, n_valid)
        energy_idx = 2  # energy is the 3rd feature (index 2)
        sorted_idxs = np.argsort(norm[:, energy_idx])
        centroids = norm[[sorted_idxs[0], sorted_idxs[-1]]]

        # Iterate k-means (max 10 iterations)
        for _ in range(10):
            distances = np.array([
                np.linalg.norm(norm - c, axis=1) for c in centroids
            ])  # shape (k, n_valid)
            labels = np.argmin(distances, axis=0)

            new_centroids = []
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    new_centroids.append(np.mean(norm[mask], axis=0))
                else:
                    new_centroids.append(centroids[j])
            new_centroids = np.array(new_centroids)

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        # Determine which cluster is the "primary" speaker (more speech time)
        speaker_time = {0: 0.0, 1: 0.0}
        for idx_in_valid, seg_i in enumerate(valid_idxs):
            spk = labels[idx_in_valid]
            dur = vad_segments[seg_i, 1] - vad_segments[seg_i, 0]
            speaker_time[int(spk)] += dur

        primary_speaker = max(speaker_time, key=speaker_time.get)
        # Map: primary → SPEAKER_00, other → SPEAKER_01
        label_to_speaker = {
            primary_speaker: "SPEAKER_00",
            1 - primary_speaker: "SPEAKER_01"
        }

        # Build result preserving original segment order
        label_by_seg = {}
        for idx_in_valid, seg_i in enumerate(valid_idxs):
            label_by_seg[seg_i] = int(labels[idx_in_valid])

        result = []
        for i, (start, end) in enumerate(vad_segments):
            spk = label_to_speaker.get(label_by_seg.get(i, primary_speaker), "SPEAKER_00")
            result.append({
                "start": start, "end": end,
                "speaker": spk,
                "duration": end - start
            })

        return result

    def compute_daily_stats(self, segments, total_audio_minutes):
        """Compute daily voice activity statistics"""
        if not segments:
            return {"speaking_minutes": 0.0, "interaction_count": 0,
                    "valid_audio_minutes": total_audio_minutes, "feature_confidence": 0.0}
        speaker_durations = {}
        for seg in segments:
            spk = seg["speaker"]
            speaker_durations[spk] = speaker_durations.get(spk, 0) + seg["duration"]
        total_speech = sum(speaker_durations.values()) / 60
        num_speakers = len(speaker_durations)

        # Count speaker transitions = interactions
        if num_speakers > 1:
            # Count how many times speaker changes between consecutive segments
            transitions = 0
            prev_spk = None
            for seg in segments:
                spk = seg["speaker"]
                if prev_spk is not None and spk != prev_spk:
                    transitions += 1
                prev_spk = spk
            interactions = max(1, transitions)
        else:
            interactions = 0

        confidence = min(1.0, total_speech / max(total_audio_minutes, 0.1))
        return {
            "speaking_minutes": round(total_speech, 2),
            "interaction_count": interactions,
            "valid_audio_minutes": round(total_audio_minutes, 2),
            "feature_confidence": round(confidence, 3)
        }
