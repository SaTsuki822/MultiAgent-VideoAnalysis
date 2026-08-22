"""评测集标注 schema 与样例。

标注结构：每段视频标注异常类型（规则 id）、起止时间、严重级别。
诚实标注：下方 MOCK_ANNOTATIONS 是样例，真实评测集（60~80 段）需另行采集/合成，
并用 FFmpeg 合成数据占比不超过 30%（见实现计划的「风险与应对」）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Annotation:
    """一段视频里的一处真实异常标注。"""
    clip_id: str
    rule_id: str           # 对应巡检规则 id
    start_seconds: float
    end_seconds: float
    severity: str          # high / medium / low


@dataclass
class Prediction:
    """系统输出的一条告警（时间戳用于与标注对齐）。"""
    clip_id: str
    rule_id: str
    timestamp_seconds: float
    confidence: float = 0.0


# 样例标注：仅用于跑通评测代码；真实数据见 docs / dataset 目录
MOCK_ANNOTATIONS: list[Annotation] = [
    Annotation(clip_id="cam_001.mp4", rule_id="rule_helmet", start_seconds=1.5, end_seconds=4.5, severity="high"),
    Annotation(clip_id="cam_001.mp4", rule_id="rule_fire", start_seconds=6.0, end_seconds=9.0, severity="high"),
]
