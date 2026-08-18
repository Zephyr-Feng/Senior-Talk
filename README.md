# Senior Talk — 老年人心理健康语音筛查系统

输入一段语音，输出心理健康诊断报告。基于声学特征提取（语速、停顿、音高、能量等）与多模态大模型（Qwen2.5-VL）的自动筛查系统。

## 运行流程（Pipeline）

```
音频 → B1 语音活动检测(VAD) → B2 Whisper转写 + 声学特征 → B3 语义证据 → B5 MLLM复核 → 诊断报告(JSON)
                              ↘ B4 上下文打包（为 MLLM 准备输入）
```

| 模块 | 作用 | 输出 |
|------|------|------|
| **B1** | 语音活动检测 + 轻量说话人分离 | 说话时长、互动次数 |
| **B2** | Whisper 转写 + 声学特征提取 | 转写文本、语速/停顿/音高/能量/jitter/shimmer |
| **B3** | 语义证据提取 | 6 类标签（睡眠抱怨/孤独/焦虑/兴趣下降/重复提问/时间混乱）+ 自杀敏感词 |
| **B4** | MLLM 上下文打包 | 标准化 JSON 上下文 |
| **B5** | MLLM 复核生成诊断报告 | feature_analysis / content_analysis / overall_assessment / summary / recommendation |

## 快速开始

### 1. 安装依赖

```bash
conda create -n psy-b python=3.10 -y && conda activate psy-b
pip install faster-whisper librosa parselmouth soundfile pandas numpy
# LoRA 微调需要（可选）：
pip install transformers peft accelerate datasets
```

### 2. 下载模型（国内用 ModelScope）

```bash
pip install modelscope

# Whisper 转写模型（small 462MB，可选 tiny 75MB / base / medium / large-v3）
modelscope download --model XXXXRT/faster-whisper \
  --local_dir ~/.cache/modelscope/XXXXRT/faster-whisper --include "faster-whisper-small/*"

# 本地 MLLM（Qwen2.5-VL-7B-Instruct，16GB）
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct \
  --local_dir ~/.cache/modelscope/qwen/Qwen2.5-VL-7B-Instruct
```

### 3. 运行 Pipeline

```bash
# 单条音频（默认本地 Qwen2.5-VL 复核；无 GPU 时自动降级为规则引擎）
python run_pipeline.py 音频.wav --user user1 --output output

# 批量目录
python run_pipeline.py /path/to/audio_dir/ --user user1 --output output

# 用 API 模式复核（需设置环境变量，勿把 Key 写进代码）
export DASHSCOPE_API_KEY=sk-xxx
python run_pipeline.py 音频.wav --user user1 --output output --mllm --api-key $DASHSCOPE_API_KEY

# 单元测试
python test_pipeline.py
```

### 4. 输出

每个文件输出一个 JSON，核心是诊断报告字段 `mllm_review`：

```json
{
  "speech_rate": 3.78, "pause_ratio": 0.765, "pitch_variability": 43.93,
  "semantic_evidence": [],
  "mllm_review": {
    "feature_analysis": "语速偏慢（1.23字/秒），可能反映精力不足或情绪低落…",
    "content_analysis": "转写文本为日常对话，未检测到明显心理问题关键词",
    "overall_assessment": "正常",
    "summary": "该老人今日语音数据整体在正常范围内",
    "recommendation": "保持日常观察即可"
  }
}
```

- `overall_assessment`：`正常` / `关注` / `异常`
- 声学特征关联量表：PHQ-9 / GAD-7 / PSQI / ULS-8 / AD8

## 数据集

| 数据集 | 说明 | 用途 |
|--------|------|------|
| **EATD-Corpus**（ICASSP 2022） | 抑郁症检测数据集：162 人 × 3 情绪 × 486 条音频，含 SDS 抑郁量表标签（SDS≥50 为抑郁） | 概念验证：声学分类器、LoRA 微调、融合消融实验 |
| **SeniorTalk** | 老年人语音对话数据集（元数据 + 示例音频） | 实际场景测试 |

> 数据集体积大，**未包含在本仓库**中，需自行下载。EATD-Corpus 为 ICASSP 2022 公开数据集，SeniorTalk 见其官方发布渠道。

## 模型

| 模型 | 用途 | 大小 | 来源 |
|------|------|------|------|
| faster-whisper-small | 语音转写 | 462MB | ModelScope（`XXXXRT/faster-whisper`） |
| Qwen2.5-VL-7B-Instruct | MLLM 复核 / 诊断报告 | 16GB | ModelScope（`Qwen/Qwen2.5-VL-7B-Instruct`） |

**LoRA 微调版**（训练权重未上传，可参考 `scripts/train_lora.py` 复现）：
基于 EATD 数据微调 Qwen2.5-VL，把声学特征转为文本描述 + 说话内容输入，输出带引证的推理式诊断报告。

| 模式 | 准确率 | 敏感度 | 特异度 | F1 |
|------|:-----:|:------:|:------:|:--:|
| 二分类 | 73.8% | 89.3% | 65.4% | 70.4% |
| 推理式报告 | 80.0% | 71.4% | 84.6% | 71.4% |

## 项目结构

```
├── config.py                    # 全局配置（VAD/转写/声学/语义/Pipeline/MLLM）
├── run_pipeline.py              # 完整 Pipeline 入口（音频 → 诊断报告）
├── test_pipeline.py             # B1-B4 单元测试
├── scripts/
│   ├── b1_vad.py                # B1: 语音活动检测 + 说话人分离
│   ├── b2_transcribe.py         # B2: Whisper 转写
│   ├── b2_acoustic.py           # B2: 声学特征（librosa + parselmouth）
│   ├── b3_semantic.py           # B3: 语义证据（6 类标签 + 敏感词）
│   ├── b4_context.py            # B4: MLLM 上下文打包
│   ├── b5_mllm_review.py        # B5: MLLM 复核（本地模型 / API / 规则引擎）
│   ├── run_eatd_pipeline.py     # EATD 批量特征提取
│   ├── prepare_training_data.py # LoRA 微调数据准备
│   ├── train_lora.py            # LoRA 微调训练
│   ├── eval_lora.py             # LoRA 评估
│   ├── loneliness_signal.py     # 孤独规则引擎
│   └── fusion_ablation.py       # 声学+语义融合消融实验
└── output/                      # 输出目录（不入库）
```

## 说明

- 默认 `config.py` 中 `use_local=True`：优先加载本地 Qwen2.5-VL；无 GPU 或模型缺失时自动降级为规则引擎模式（`simulate_without_api=True`）
- 声学分类器当前在 EATD 上区分度有限（AUC≈0.6），融合诊断以 MLLM 语义分析为主
- 本系统为筛查辅助工具，不构成医疗诊断
