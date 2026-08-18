"""B5: MLLM Review Module - generates natural language summary via Qwen2.5-VL

Three modes:
  1. Local model (use_local=True): loads Qwen2.5-VL-7B from local path
  2. API mode (use_local=False, api_key set): calls DashScope API
  3. Simulated mode (default): rule-based summary without model
"""
import json
import logging
import re
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MLLMConfig
import torch

logger = logging.getLogger(__name__)


class MLLMReviewer:
    """Generates psychological health review from pipeline features"""

    def __init__(self, config: MLLMConfig = MLLMConfig()):
        self.config = config
        self._model = None
        self._processor = None
        self._model_loaded = False
        self._is_lora = False

    def _parse_json_response(self, text: str) -> Any:
        """Parse model response as JSON, stripping markdown code fences (```json ... ```)."""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return json.loads(text)

    def _build_prompt(self, daily_output: Dict[str, Any]) -> str:
        """Build the prompt for Qwen2.5-VL from pipeline output"""
        transcription = daily_output.get("full_text", "")
        semantic = daily_output.get("semantic_evidence", [])
        safety = daily_output.get("safety_flag", False)

        semantic_text = "无"
        if semantic:
            items = []
            for s in semantic:
                items.append(f"  - {s.get('label', '')}: {s.get('evidence', '')}")
            semantic_text = "\n" + "\n".join(items)

        safety_text = "是 ⚠️" if safety else "否"

        return f"""你是一位老年心理健康筛查助理。请根据以下老年人语音数据分析其心理健康状况。

## 基本信息
- 日期: {daily_output.get('date', '未知')}
- 用户: {daily_output.get('user_id', '未知')}

## 语音活动
- 有效录音: {daily_output.get('valid_audio_minutes', 0)} 分钟
- 老人说话时长: {daily_output.get('speaking_minutes', 0)} 分钟
- 社交互动次数: {daily_output.get('interaction_count', 0)} 次

## 声学特征
- 语速: {daily_output.get('speech_rate', 0)} 字/秒
- 停顿比例: {daily_output.get('pause_ratio', 0)}（值越高表示沉默越多）
- 音高变化: {daily_output.get('pitch_variability', 0)}
- 能量变化: {daily_output.get('energy_variability', 0)}

## 转写文本
{transcription if transcription else "（无有效转写）"}

## 语义证据
{semantic_text}

## 安全标志
{safety_text}

请从以下三个方面进行分析，并以 **JSON格式** 返回（不要用 markdown 包裹）：

1. feature_analysis：声学特征反映的心理状态（语速快慢、停顿多少、音调变化等）
2. content_analysis：转写文本反映的情绪和认知状态
3. overall_assessment：综合评估（正常/关注/异常）及建议

返回格式：
{{"feature_analysis": "...", "content_analysis": "...", "overall_assessment": "正常/关注/异常", "summary": "一句话总结", "recommendation": "建议"}}"""

    def _call_api(self, prompt: str) -> Optional[str]:
        """Call DashScope Qwen2.5-VL API"""
        import http.client
        import json as json_module

        api_key = self.config.api_key
        if not api_key:
            logger.warning("MLLM API key not configured")
            return None

        payload = json_module.dumps({
            "model": self.config.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": prompt}
                        ]
                    }
                ]
            },
            "parameters": {
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "result_format": "message"
            }
        })

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        try:
            conn = http.client.HTTPSConnection("dashscope.aliyuncs.com", timeout=60)
            conn.request("POST", "/api/v1/services/aigc/multimodal-generation/generation",
                        payload.encode("utf-8"), headers)
            resp = conn.getresponse()
            body = resp.read().decode("utf-8")
            conn.close()

            if resp.status != 200:
                logger.error(f"API error {resp.status}: {body[:200]}")
                return None

            result = json_module.loads(body)
            # Parse DashScope response format
            output = result.get("output", {})
            choices = output.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content", "")
                if isinstance(content, list):
                    texts = [item.get("text", "") for item in content if isinstance(item, dict)]
                    content = "\n".join(texts)
                return content
            return None
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return None

    def _load_local_model(self):
        """Lazy-load Qwen2.5-VL model from local path.

        When lora_adapter_path is configured, loads the LoRA fine-tuned
        reasoning-report version (4-bit quantized base + adapter).
        """
        if self._model_loaded:
            return True
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            model_path = self.config.local_model_path
            lora_path = self.config.lora_adapter_path

            if lora_path:
                from peft import PeftModel
                from transformers import BitsAndBytesConfig
                logger.info(f"Loading base Qwen2.5-VL + LoRA adapter from {lora_path}...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
                self._model = PeftModel.from_pretrained(base_model, lora_path)
                self._processor = AutoProcessor.from_pretrained(lora_path)
                self._is_lora = True
                logger.info(f"LoRA model loaded OK (device={self._model.device})")
            else:
                logger.info(f"Loading local Qwen2.5-VL from {model_path}...")
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    device_map="auto",
                )
                self._processor = AutoProcessor.from_pretrained(model_path)
                self._is_lora = False
                logger.info(f"Local model loaded OK (device={self._model.device})")
            self._model_loaded = True
            return True
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            self._model_loaded = False
            return False

    def _call_local_model(self, prompt: str) -> Optional[str]:
        """Run inference with locally loaded Qwen2.5-VL"""
        if not self._load_local_model():
            return None
        try:
            messages = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(text=[text], padding=True, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    pad_token_id=self._processor.tokenizer.pad_token_id,
                )
            # Trim input tokens from output
            input_len = inputs["input_ids"].shape[1]
            output = self._processor.decode(
                generated_ids[0][input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return output.strip()
        except Exception as e:
            logger.error(f"Local inference failed: {e}")
            return None

    # --- LoRA 推理式报告版 ---

    def _features_to_text(self, daily_output: Dict[str, Any]) -> str:
        """Convert acoustic features to natural language description
        (same style as prepare_training_data.acoustic_to_text, keeping the
        LoRA model's training distribution)."""
        f = daily_output
        parts = []

        sr = f.get("speech_rate", 0)
        if sr < 2.5:
            parts.append(f"语速很慢（{sr:.1f}字/秒）")
        elif sr < 4.0:
            parts.append(f"语速偏慢（{sr:.1f}字/秒）")
        elif sr < 5.5:
            parts.append(f"语速正常（{sr:.1f}字/秒）")
        else:
            parts.append(f"语速偏快（{sr:.1f}字/秒）")

        pr = f.get("pause_ratio", 0)
        if pr < 0.2:
            parts.append("停顿很少")
        elif pr < 0.4:
            parts.append("停顿略多")
        elif pr < 0.6:
            parts.append("停顿较多")
        else:
            parts.append("停顿非常多")

        pv = f.get("pitch_variability", 0)
        if pv < 20:
            parts.append("音调变化小，语调平淡")
        elif pv < 40:
            parts.append("音调变化正常")
        elif pv < 60:
            parts.append("音调变化较大，情绪有一定波动")
        else:
            parts.append("音调变化显著，情绪起伏明显")

        ev = f.get("energy_variability", 0)
        if ev < 0.02:
            parts.append("发声能量极低，说话无力")
        elif ev < 0.05:
            parts.append("发声能量偏低，可能精力不足")
        elif ev < 0.08:
            parts.append("发声能量正常")
        else:
            parts.append("发声能量充足")

        jit = f.get("jitter", 0)
        if jit and jit < 0.015:
            parts.append("嗓音较稳定")
        elif jit and jit < 0.03:
            parts.append("嗓音稳定性一般")
        else:
            parts.append("嗓音稳定性较差")

        sh = f.get("shimmer", 0)
        if sh and sh < 0.03:
            parts.append("音质干净")
        elif sh and sh < 0.06:
            parts.append("音质一般")
        else:
            parts.append("音质较嘈杂")

        return "，".join(parts)

    def _call_lora_model(self, daily_output: Dict[str, Any]) -> Optional[str]:
        """Run inference with the LoRA reasoning-report model."""
        if not self._load_local_model():
            return None
        try:
            system_msg = ("你是一个专业的心理健康评估助手。请根据对方的语音声学特征和"
                          "说话内容进行综合分析，输出推理式诊断报告。注意引用具体的说话"
                          "内容和声学数值作为依据。")
            user_msg = "【声学特征】%s\n【说话内容】%s" % (
                self._features_to_text(daily_output),
                daily_output.get("full_text", ""),
            )
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": system_msg}]},
                {"role": "user", "content": [{"type": "text", "text": user_msg}]},
            ]
            text = self._processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(text=text, return_tensors="pt", padding=True)
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=200,  # 推理式输出较长
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self._processor.tokenizer.pad_token_id,
                )
            input_len = inputs["input_ids"].shape[1]
            response = self._processor.decode(
                outputs[0][input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            return response
        except Exception as e:
            logger.error(f"LoRA inference failed: {e}")
            return None

    def _parse_lora_report(self, response: str) -> Dict[str, Any]:
        """Parse the reasoning-report format (分析/结论/建议) into structured fields."""
        result: Dict[str, Any] = {}

        # 结论行 → overall_assessment
        conclusion = ""
        for line in response.split("\n"):
            if line.strip().startswith("结论："):
                conclusion = line.replace("结论：", "").strip()
                break
        if "关注" in conclusion or "异常" in conclusion or "抑郁" in conclusion:
            result["overall_assessment"] = "关注"
        elif "正常" in conclusion:
            result["overall_assessment"] = "正常"
        else:
            result["overall_assessment"] = conclusion or "正常"

        # 建议行 → recommendation
        rec = ""
        for line in response.split("\n"):
            if line.strip().startswith("建议："):
                rec = line.replace("建议：", "").strip()
                break
        result["recommendation"] = rec

        # 分析段落（"分析"到"结论"之间）→ summary（综合判断）
        m = re.search(r"分析[:：](.*?)(?:\n\s*结论[:：]|$)", response, re.S)
        if m:
            result["summary"] = m.group(1).strip()
        else:
            result["summary"] = conclusion or response[:100]

        result["report"] = response          # 完整推理式报告原文
        result["mode"] = "lora"
        return result

    def _generate_simulated_review(self, daily_output: Dict[str, Any]) -> Dict[str, str]:
        """Generate rule-based review when API is unavailable"""
        sr = daily_output.get("speech_rate", 0)
        pr = daily_output.get("pause_ratio", 0)
        pv = daily_output.get("pitch_variability", 0)
        ev = daily_output.get("energy_variability", 0)
        semantic = daily_output.get("semantic_evidence", [])
        safety = daily_output.get("safety_flag", False)
        interaction = daily_output.get("interaction_count", 0)
        speaking = daily_output.get("speaking_minutes", 0)

        # Feature analysis
        feature_parts = []
        if sr < 1.0:
            feature_parts.append(f"语速极慢（{sr}字/秒），可能存在语言运动迟缓")
        elif sr < 1.8:
            feature_parts.append(f"语速偏慢（{sr}字/秒），可能反映精力不足或情绪低落")
        elif sr < 3.0:
            feature_parts.append(f"语速正常（{sr}字/秒）")
        else:
            feature_parts.append(f"语速偏快（{sr}字/秒），可能存在焦虑情绪")

        if pr > 0.5:
            feature_parts.append(f"停顿比例偏高（{pr}），可能存在认知搜索困难或犹豫")
        elif pr > 0.35:
            feature_parts.append(f"停顿比例略高（{pr}）")
        else:
            feature_parts.append(f"停顿比例正常（{pr}）")

        if pv < 30:
            feature_parts.append(f"音调变化单调（{pv}），可能存在情绪表达减弱（情感平淡）")
        elif pv > 80:
            feature_parts.append(f"音调变化较大（{pv}），情绪表达丰富")
        else:
            feature_parts.append(f"音调变化在正常范围（{pv}）")

        if ev < 0.1:
            feature_parts.append("能量变化偏低，语音活力不足")
        else:
            feature_parts.append("能量变化正常")

        if interaction == 0 and speaking > 0:
            feature_parts.append("未检测到多人社交互动")
        elif interaction > 0:
            feature_parts.append(f"检测到{interaction}次社交互动")

        feature_analysis = "；".join(feature_parts)

        # Content analysis
        if safety:
            content_analysis = "⚠️ 转写文本检测到敏感词汇（自杀相关），需要立即关注！"
        elif semantic:
            labels = [s.get("label", "") for s in semantic]
            content_parts = []
            label_names = {
                "sleep_complaint": "睡眠抱怨",
                "loneliness": "孤独表达",
                "anxiety_worry": "焦虑担忧",
                "loss_of_interest": "兴趣下降",
                "repeated_questions": "重复提问",
                "time_confusion": "时间混乱"
            }
            for l in labels:
                name = label_names.get(l, l)
                content_parts.append(f"存在{name}")
            content_analysis = "，".join(content_parts)
        else:
            content_analysis = "转写文本为日常对话，未检测到明显心理问题关键词"

        # Overall assessment
        confidence = daily_output.get("feature_confidence", 0.5)
        risk_score = 0
        if safety:
            risk_score += 3
        if sr < 1.0:
            risk_score += 1
        if pr > 0.5:
            risk_score += 1
        if pv < 30:
            risk_score += 1
        if semantic:
            risk_score += len(semantic)

        # Low-confidence audio should not trigger flags unless safety is real
        if confidence < 0.3 and not safety and not semantic:
            overall = "正常"
        elif risk_score >= 4:
            overall = "异常"
        elif risk_score >= 2:
            overall = "关注"
        else:
            overall = "正常"

        # Summary
        if overall == "异常":
            summary = f"该老人今日语音数据存在异常指标，建议及时关注"
        elif overall == "关注":
            summary = f"该老人今日语音数据部分指标偏离正常范围，建议持续观察"
        else:
            summary = f"该老人今日语音数据整体在正常范围内"

        # Recommendation
        if safety:
            recommendation = "检测到敏感词汇，请立即联系相关人员确认安全状况"
        elif overall == "异常":
            recommendation = "建议增加观察频率，必要时安排专业心理评估"
        elif overall == "关注":
            recommendation = "建议持续监测语音特征变化趋势"
        else:
            recommendation = "保持日常观察即可"

        return {
            "feature_analysis": feature_analysis,
            "content_analysis": content_analysis,
            "overall_assessment": overall,
            "summary": summary,
            "recommendation": recommendation
        }

    def review(self, daily_output: Dict[str, Any]) -> Dict[str, Any]:
        """Run MLLM review on daily pipeline output

        Priority:
          1. use_local=True -> load Qwen2.5-VL from local path
          2. enable_review + api_key -> call DashScope API
          3. simulate_without_api=True -> rule-based simulated review
          4. otherwise -> no review
        """
        if not self.config.enable_review and not self.config.simulate_without_api:
            logger.info("MLLM review disabled")
            daily_output["mllm_review"] = {"note": "MLLM review not enabled"}
            return daily_output

        review_result = None

        # Mode 1: Local model (LoRA reasoning-report version if configured)
        if self.config.use_local:
            if self.config.lora_adapter_path:
                logger.info("Using local Qwen2.5-VL + LoRA (reasoning-report version)...")
                response = self._call_lora_model(daily_output)
                if response:
                    review_result = self._parse_lora_report(response)
                    logger.info("LoRA MLLM review successful")
                else:
                    logger.warning("LoRA model inference failed, falling back")
            else:
                logger.info("Using local Qwen2.5-VL (zero-shot)...")
                prompt = self._build_prompt(daily_output)
                response = self._call_local_model(prompt)
                if response:
                    try:
                        review_result = self._parse_json_response(response)
                        logger.info("Local MLLM review successful")
                    except json.JSONDecodeError:
                        logger.warning(f"Local model response not valid JSON: {response[:100]}")
                        review_result = {"raw_response": response}
                    else:
                        review_result["mode"] = "local"
                else:
                    logger.warning("Local model inference failed, falling back")

        # Mode 2: API
        if review_result is None and self.config.enable_review and self.config.api_key:
            logger.info("Calling Qwen2.5-VL API for review...")
            prompt = self._build_prompt(daily_output)
            response = self._call_api(prompt)
            if response:
                try:
                    review_result = self._parse_json_response(response)
                    logger.info("MLLM API review successful")
                except json.JSONDecodeError:
                    logger.warning(f"MLLM response not valid JSON, using raw: {response[:100]}")
                    review_result = {"raw_response": response}
                else:
                    review_result["mode"] = "api"

        # Mode 3: Simulated
        if review_result is None:
            logger.info("Using simulated review")
            review_result = self._generate_simulated_review(daily_output)
            review_result["mode"] = "simulated"

        review_result["generated_at"] = datetime.now().isoformat()
        daily_output["mllm_review"] = review_result
        return daily_output
