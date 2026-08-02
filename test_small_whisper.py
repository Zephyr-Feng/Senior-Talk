import sys
sys.path.insert(0, "/root/autodl-tmp/senior_project")
from config import PipelineConfig
from scripts.b2_transcribe import WhisperTranscriber
import time

config = PipelineConfig()
config.transcription.model_size = "/root/autodl-tmp/.cache/modelscope/XXXXRT/faster-whisper/faster-whisper-small"

print("Loading small model...")
t0 = time.time()
t = WhisperTranscriber(config.transcription)
elapsed = time.time() - t0
print("Model loaded in " + str(round(elapsed, 1)) + "s")

print("Transcribing...")
t0 = time.time()
result = t.transcribe("data/EATD-Corpus/EATD-Corpus/t_1/neutral.wav")
elapsed = time.time() - t0
text = result.get("full_text", "")
print("Done in " + str(round(elapsed, 1)) + "s")
print("Text: " + text[:100])
print("Chars: " + str(len(text)))
