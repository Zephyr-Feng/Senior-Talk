"""B3: Semantic Evidence Extraction from Transcriptions"""
import re
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SemanticConfig


@dataclass
class EvidenceLabel:
    name: str
    zh_name: str
    keywords: List[str]
    patterns: List[str]
    associated_scales: List[str]


class SemanticAnalyzer:
    def __init__(self, config: SemanticConfig = SemanticConfig()):
        self.config = config
        self._init_evidence_labels()

    def _init_evidence_labels(self):
        self.labels = [
            EvidenceLabel("sleep_complaint", "睡眠抱怨", [
                "睡不着", "失眠", "难入睡", "早醒", "睡不好",
                "整夜睡不着", "半夜醒", "做梦多"
            ], [r"(睡不着|失眠|没睡好|睡得不好|整夜没睡)"],
                          ["PSQI", "PHQ-9"]),
            EvidenceLabel("loneliness", "孤独表达", [
                "一个人", "没人陪", "孤单", "孤独", "没人说话",
                "寂寞", "独自", "没人关心"
            ], [r"(一个人|没人陪|孤单|孤独|寂寞|没人说话)"],
                          ["ULS-8"]),
            EvidenceLabel("anxiety_worry", "焦虑担忧", [
                "担心", "焦虑", "紧张", "害怕", "不安", "烦躁",
                "坐立不安", "心慌", "着急"
            ], [r"(担心|焦虑|紧张|害怕|不安|烦躁|心慌)"],
                          ["GAD-7"]),
            EvidenceLabel("loss_of_interest", "兴趣下降", [
                "没意思", "没兴趣", "不想动", "懒得", "不想做",
                "没劲", "提不起劲"
            ], [r"(没意思|没兴趣|不想动|懒得|没劲|提不起)"],
                          ["PHQ-9"]),
            EvidenceLabel("repeated_questions", "重复问题", [
                "刚刚说过", "又问一遍", "反复说", "重复问"
            ], [r"(又问|重复|反复说|刚刚才|说过好几次|又忘了)"],
                          ["AD8"]),
            EvidenceLabel("time_confusion", "时间混乱", [
                "今天几号", "什么日子", "分不清",
                "搞不清时间", "白天晚上分不清"
            ], [r"(几号|什么日子|分不清|什么时间).*(分不清|不知道)"],
                          ["AD8"]),
        ]

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text for evidence of mental health indicators"""
        if not text or not text.strip():
            return {"semantic_evidence": [], "evidence_count": 0}

        evidence_list = []
        found_labels = set()

        for label in self.labels:
            matches = []
            for pattern in label.patterns:
                found = re.findall(pattern, text)
                for m in found:
                    if isinstance(m, tuple):
                        m = " ".join(filter(None, m))
                    matches.append(m)
            keyword_hits = [kw for kw in label.keywords if kw in text]
            if matches or keyword_hits:
                evidence = {
                    "label": label.name,
                    "label_zh": label.zh_name,
                    "keyword_hits": keyword_hits[:5],
                    "pattern_matches": matches[:3],
                    "associated_scales": label.associated_scales
                }
                evidence_list.append(evidence)
                found_labels.add(label.name)

        sensitive_hits = [kw for kw in self.config.sensitive_keywords if kw in text]
        safety_flag = len(sensitive_hits) > 0

        return {
            "semantic_evidence": evidence_list,
            "evidence_count": len(evidence_list),
            "evidence_labels": list(found_labels),
            "safety_flag": safety_flag,
            "sensitive_matches": sensitive_hits if safety_flag else [],
            "repetition_count": self._count_repetitions(text)
        }

    def _count_repetitions(self, text: str) -> int:
        """Estimate repetition count from repeated phrases"""
        sentences = re.split(r"[.!?,;!?，。！？、]", text)
        seen = set()
        repetitions = 0
        for s in sentences:
            s = s.strip()
            if len(s) > 2:
                if s in seen:
                    repetitions += 1
                seen.add(s)
        return repetitions

    def analyze_daily(self, all_transcriptions: List[str]) -> Dict[str, Any]:
        """Aggregate daily semantic analysis across multiple segments"""
        combined_text = " ".join(all_transcriptions)
        result = self.analyze(combined_text)
        total_repetitions = sum(self._count_repetitions(t) for t in all_transcriptions)
        result["repetition_count"] = total_repetitions
        return result
