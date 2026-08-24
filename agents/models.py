"""核心数据模型。

设计原则：
1. 用 pydantic 做运行时校验 + JSON 序列化 —— 告警 / 报告 / 误报签名要跨进程传输
   （MCP 工具返回值、Qdrant metadata、前端展示），序列化友好是硬需求而非装饰。
2. 所有跨模块流转的对象都用这里的模型，避免 dict 满天飞导致字段拼写错误难以发现。
3. 字段命名与 plan 文档（PatrolState 字段、MCP 工具 schema）保持一致，面试时能直接对应。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

# ---- 枚举（Literal 提供类型级约束）----
Severity = Literal["high", "medium", "low"]
AlarmStatus = Literal["pending_review", "confirmed", "false_positive", "suppressed", "dismissed"]
Verdict = Literal["confirmed", "rejected", "uncertain"]
FeedbackDecision = Literal["confirm", "false_positive", "change_severity", "dismiss"]


# ============================================================
# 规则 / 台账层
# ============================================================
class Rule(BaseModel):
    """自然语言巡检规则。核心卖点：新增规则 = 加一行描述，无需重训。"""
    id: str
    name: str
    description: str
    severity: Severity = "medium"
    # 以下字段由 planner 编译时填充（结构化判定依据）
    target_objects: list[str] = Field(default_factory=list)
    evidence_hints: list[str] = Field(default_factory=list)
    # 持续型异常要求的最短持续时间（秒），如「物料堵塞需持续 5 分钟」
    duration_threshold_seconds: Optional[float] = None


class Camera(BaseModel):
    """摄像头台账（camera-registry 的 mock 数据，预留真实 NVR 接入）。"""
    id: str
    name: str
    area: str
    rtsp_url: str = ""


class Clip(BaseModel):
    """一段待分析的视频片段。"""
    camera_id: str
    path: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float


class AnalysisTask(BaseModel):
    """planner 编译出的最小分析子任务：每个 camera × 每个 rule = 一个 task。"""
    id: str
    camera_id: str
    rule: Rule
    clip_path: str
    output_schema: str = "finding"
    # 时序聚合需要绝对时间信息
    clip_start_time: Optional[datetime] = None
    clip_end_time: Optional[datetime] = None
    duration_seconds: float = 0.0


# ============================================================
# 感知 / 复核层
# ============================================================
class FrameEvidence(BaseModel):
    """VLM 命中时引用的关键帧证据（answer with evidence）。"""
    frame_index: int
    timestamp_seconds: float
    description: str = ""


class Finding(BaseModel):
    """L2 初筛结果。hit 表示该片段是否命中规则。"""
    id: str
    task_id: str
    rule_id: str
    camera_id: str
    hit: bool
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[FrameEvidence] = Field(default_factory=list)
    hit_frame_indices: list[int] = Field(default_factory=list)
    # 成本记账（对应面试 Q14）
    cost_tokens: int = 0
    cost_currency: float = 0.0
    # 时序聚合需要绝对时间信息（由 router 从 AnalysisTask 带入）
    clip_start_time: Optional[datetime] = None
    clip_end_time: Optional[datetime] = None
    duration_seconds: float = 0.0


class Verification(BaseModel):
    """verify 节点对 finding 的复核结论。"""
    finding_id: str
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    sop_references: list[str] = Field(default_factory=list)


# ============================================================
# 时序聚合 / 持续异常跟踪层
# ============================================================

class OngoingAnomaly(BaseModel):
    """持续型异常的片段级状态机跟踪记录。

    以 (camera_id, rule_id) 为唯一键，跨片段累计异常持续时间。
    """
    camera_id: str
    rule_id: str
    rule_name: str = ""
    severity: Severity = "medium"
    duration_threshold_seconds: float = 0.0

    # 状态机状态
    state: Literal["idle", "detected", "accumulating", "confirmed", "closed"] = "idle"

    # 时间轴（绝对时间，来自 clip_start_time / clip_end_time）
    first_seen_at: Optional[datetime] = None   # 首次命中片段的开始时间
    last_seen_at: Optional[datetime] = None    # 最新命中片段的结束时间
    accumulated_seconds: float = 0.0           # 累计持续秒数（视频时间轴）

    # 证据累积（取各片段的关键帧证据，避免无限增长可限制条数）
    evidence_snapshots: list[FrameEvidence] = Field(default_factory=list)

    # 关联告警（confirmed 后生成）
    alarm_id: Optional[str] = None

    # 元数据
    hit_count: int = 0          # 累计命中片段数
    miss_count: int = 0         # 连续未命中片段数（用于超时关闭）
    updated_at: datetime = Field(default_factory=datetime.now)

    def compute_duration(self) -> float:
        """基于 first_seen_at 与 last_seen_at 计算视频时间轴跨度（秒）。"""
        if self.first_seen_at and self.last_seen_at:
            return (self.last_seen_at - self.first_seen_at).total_seconds()
        return self.accumulated_seconds
class Alarm(BaseModel):
    """一条（待复核 / 已确认 / 已抑制）告警。"""
    id: str
    camera_id: str
    rule_id: str
    rule_name: str
    severity: Severity
    status: AlarmStatus = "pending_review"
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[FrameEvidence] = Field(default_factory=list)
    # 记忆层抑制标记
    suppressed: bool = False
    suppression_reason: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


class Feedback(BaseModel):
    """HITL 人工反馈。"""
    alarm_id: str
    decision: FeedbackDecision
    new_severity: Optional[Severity] = None
    comment: str = ""


class FalsePositiveSignature(BaseModel):
    """误报结构化签名，LLM 从人工标记的误报中抽取。

    维度：scene（场景）/ object（对象）/ lighting（光照）/ description（描述）。
    抽取结构化签名的原因：让相似误报可在向量空间比对，而非逐字匹配文本。
    """
    alarm_id: str
    rule_id: str
    scene: str
    object: str
    lighting: str = ""
    description: str
    occurrence_count: int = 1  # 防污染：>= activation_count 才生效


class EventCard(BaseModel):
    """已确认真实告警生成的事件卡片，报告时检索「近 30 天同类事件」。"""
    id: str
    alarm_id: str
    camera_id: str
    rule_id: str
    summary: str
    occurred_at: datetime


class Report(BaseModel):
    """最终巡检报告。"""
    patrol_id: str
    generated_at: datetime = Field(default_factory=datetime.now)
    summary: str = ""
    alarms: list[Alarm] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)


class LogEntry(BaseModel):
    """审计日志条目（reducer 追加式收集）。"""
    node: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
