"""B2: Speech Transcription using faster-whisper"""
from pathlib import Path
from typing import Dict, Any
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TranscriptionConfig


class WhisperTranscriber:
    def __init__(self, config: TranscriptionConfig = TranscriptionConfig()):
        self.config = config
        self._model = None

    def _init_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type
            )
        return self._model

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio file"""
        model = self._init_model()
        segments, info = model.transcribe(
            audio_path,
            language=self.config.language,
            beam_size=self.config.beam_size,
            vad_filter=self.config.vad_filter
        )
        result = {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "segments": []
        }
        for seg in segments:
            result["segments"].append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "confidence": seg.avg_logprob
            })
        result["full_text"] = " ".join(s["text"] for s in result["segments"])
        result["word_count"] = len(result["full_text"])
        result["speech_rate"] = round(
            result["word_count"] / max(result["duration"], 0.1), 2
        )
        return result
