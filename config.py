"""Audio Module B - Configuration"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VADConfig:
    sample_rate: int = 16000
    frame_duration_ms: int = 30
    threshold: float = 0.3 # Lowered from 0.5 for more sensitive VAD
    min_speech_duration_ms: float = 300.0
    min_silence_duration_ms: float = 500.0
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    num_speakers: Optional[int] = None


@dataclass
class TranscriptionConfig:
    model_size: str = "/root/autodl-tmp/.cache/modelscope/XXXXRT/faster-whisper/faster-whisper-small"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "zh"
    beam_size: int = 5
    vad_filter: bool = True


@dataclass
class AcousticConfig:
    sample_rate: int = 16000
    frame_length: int = 2048
    hop_length: int = 512
    f0_min: float = 50.0
    f0_max: float = 500.0


@dataclass
class SemanticConfig:
    evidence_labels: List[str] = field(default_factory=lambda: [
        "sleep_complaint", "loneliness", "anxiety_worry",
        "loss_of_interest", "repeated_questions", "time_confusion"
    ])
    sensitive_keywords: List[str] = field(default_factory=lambda: [
        "不想活", "死了算了", "不如死", "自杀", "活不下去"
    ])


@dataclass
class MLLMConfig:
    """Configuration for MLLM review (Qwen2.5-VL via DashScope API or local model)"""
    # --- API 模式（Key 请通过环境变量 DASHSCOPE_API_KEY 注入，勿提交到仓库）---
    api_key: str = ""
    model: str = "qwen2.5-vl-7b-instruct"    # API 模型名
    endpoint: str = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    # --- 本地推理模式 ---
    use_local: bool = True                    # True = 加载本地 Qwen2.5-VL 推理
    local_model_path: str = "/root/autodl-tmp/.cache/modelscope/qwen/Qwen2.5-VL-7B-Instruct"
    # LoRA 微调 adapter 路径（推理式报告版）。为空字符串 = 零样本模式（用原始模型）
    lora_adapter_path: str = "/root/autodl-tmp/senior_project/training_output/lora_adapter_dep_v1"
    # --- 通用参数 ---
    max_tokens: int = 1024
    temperature: float = 0.3
    enable_review: bool = True
    simulate_without_api: bool = True


@dataclass
class PipelineConfig:
    user_id: str = "default_user"
    output_dir: str = "output"
    data_dir: str = "data"
    vad: VADConfig = field(default_factory=VADConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    acoustic: AcousticConfig = field(default_factory=AcousticConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    mllm: MLLMConfig = field(default_factory=MLLMConfig)


DEFAULT_CONFIG = PipelineConfig()
