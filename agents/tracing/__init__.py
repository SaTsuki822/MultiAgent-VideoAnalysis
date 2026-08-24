"""执行轨迹采集与分析（对应面试「Agent 执行轨迹分析」）。

定位：补上单 Agent 流水线的可观测性——每次巡检把每个节点的耗时 / 状态 / token / 成本
记录为 span，聚合出「瓶颈、成本、错误率、时间轴」，供调优与回归对比。

生产可替换为 Langfuse / OpenTelemetry 后端（见 store.TraceStore 接口），当前先用
内存 + JSONL 落盘跑通闭环。
"""

from agents.tracing.models import NodeAggregate, TraceRecord, TraceSpan, TraceSummary
from agents.tracing.store import InMemoryTraceStore, JsonlTraceStore, TraceStore
from agents.tracing.tracer import Tracer, analyze, render_timeline, summarize_state

__all__ = [
    "TraceSpan",
    "TraceRecord",
    "TraceSummary",
    "NodeAggregate",
    "TraceStore",
    "InMemoryTraceStore",
    "JsonlTraceStore",
    "Tracer",
    "analyze",
    "render_timeline",
    "summarize_state",
]
