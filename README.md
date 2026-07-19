# Senior Talk — 老年人心理健康语音筛查系统

基于语音声学特征与 MLLM 的老年人心理健康筛查系统。

## 项目结构

```
├── config.py                    # 全局配置（VAD/转写/声学/语义/Pipeline）
├── run_pipeline.py             # 完整 Pipeline 入口
├── test_pipeline.py            # B1-B4 单元测试套件
├── scripts/
│   ├── utils.py                # 工具函数
│   ├── b1_vad.py               # B1: 语音活动检测 + 说话人分离
│   ├── b2_transcribe.py        # B2: Whisper 转写
│   ├── b2_acoustic.py          # B2: 声学特征提取（librosa + parselmouth）
│   ├── b3_semantic.py          # B3: 语义证据提取（6类标签 + 敏感词）
│   ├── b4_context.py           # B4: MLLM 上下文打包
│   ├── prepare_training_data.py # LoRA 微调数据准备（推理式输出）
│   ├── train_lora.py           # Qwen2.5-VL LoRA 微调
│   └── eval_lora.py            # 微调模型评估
└── .gitignore
```

## Pipeline 流程

1. **B1** — VAD + 轻量说话人分离（pyannote 替代方案）
2. **B2** — Whisper 转写 + 声学特征（语速、停顿、音调变化、能量、jitter、shimmer）
3. **B3** — 语义证据提取（PHQ-9/GAD-7/PSQI/ULS-8 映射）
4. **B4** — MLLM 上下文打包（JSON 标准化输出）
5. **B5** — MLLM 复核（规则引擎 / API / 本地模型三种模式）

## LoRA 微调

基于 Qwen2.5-VL-7B-Instruct，使用 EATD-Corpus 数据集微调：

| 模式 | 准确率 | 敏感度 | 特异度 | F1 |
|------|:-----:|:------:|:------:|:--:|
| 二分类 | 73.8% | 89.3% | 65.4% | 70.4% |
| 推理式报告 | **80.0%** | 71.4% | **84.6%** | **71.4%** |

- 声学特征转为文本描述输入 MLLM，实现声学+语义多模态融合诊断
- 加权 loss（FN_weight=3.0）优先降低漏诊率
- 输出为带引用、分点分析、综合判断的推理式诊断报告

## 依赖

- Python 3.10+
- PyTorch 2.x
- faster-whisper
- librosa
- parselmouth (praat)
- transformers + peft (LoRA 微调)
