"""执行轨迹（Trace）数据模型。

轨迹分析的对象是「一次巡检里每个节点 / Agent 的一次执行」——即 span：
- TraceSpan   : 单个执行片段（节点名、起止时间、耗时、状态、token/成本、摘要）
- TraceRecord : 一次巡检（trace）的完整 span 集合
- NodeAggregate / TraceSummary : 分析结果（延迟 / 成本 / 错误 / 瓶颈 / 时间轴）

设计原则与 agents/models.py 一致：pydantic 做校验 + 序列化友好；
span 的 input_summary / output_summary 只存「摘要」而非完整大对象，避免轨迹膨胀。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

SpanStatus = Literal["ok", "error", "running"]


class TraceSpan(BaseModel):
    """一个节点 / Agent 的一次执行片段。"""

    span_id: str
    name: str                          # 节点名或 agent 名，如 "plan" / "dispatch"
    agent: str = ""                    # 所属 Agent（多 Agent 版用），单 Agent 版留空
    parent_id: Optional[str] = None    # 父 span（支持嵌套，当前流水线为扁平）
    start_at: datetime = Field(default_factory=datetime.now)
    end_at: Optional[datetime] = None
    duration_ms: float = 0.0
    status: SpanStatus = "running"
    error: str = ""
    input_summary: dict = Field(default_factory=dict)
    output_summary: dict = Field(default_factory=dict)
    tokens: int = 0                    # 该 span 消耗的 token（估算口径，见 router 说明）
    cost: float = 0.0                  # 该 span 的成本（估算口径）
    attributes: dict = Field(default_factory=dict)


class TraceRecord(BaseModel):
    """一次巡检（patrol / trace）的完整执行轨迹。"""

    trace_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    spans: list[TraceSpan] = Field(default_factory=list)

    def add_span(self, span: TraceSpan) -> None:
        self.spans.append(span)


class NodeAggregate(BaseModel):
    """单个节点（或 agent）维度的聚合统计。"""

    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    avg_ms: float = 0.0
    tokens: int = 0
    cost: float = 0.0
    errors: int = 0


class TraceSummary(BaseModel):
    """轨迹分析结果：延迟 / 成本 / 错误 / 瓶颈 / 可读时间轴。"""

    trace_id: str
    total_spans: int = 0
    total_duration_ms: float = 0.0    # 墙钟：最早开始到最晚结束
    sum_duration_ms: float = 0.0      # 各 span 时长之和（顺序执行时 ≈ 墙钟）
    ok_count: int = 0
    error_count: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    per_node: dict[str, NodeAggregate] = Field(default_factory=dict)
    bottleneck: str = ""              # 耗时最长的节点名
    bottleneck_ms: float = 0.0        # 该节点耗时
    timeline: str = ""                # 可读文本时间轴
