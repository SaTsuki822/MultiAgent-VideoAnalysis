"""执行轨迹采集器（Tracer）与轨迹分析。

采集：用 `with tracer.span("plan") as s: ...` 包住节点执行，自动记耗时、状态、
      异常；span 的 output_summary 由调用方写入（或由 workflow 的 _wrap_node 自动填充）。
分析：analyze() 把一条 TraceRecord 聚合为 TraceSummary（延迟 / 成本 / 错误 / 瓶颈 / 时间轴）。

诚实边界：
- token / cost 是「估算口径」，与 router 的 CostBreakdown 一致，真实数字需接 Langfuse / 实测回填；
- LangGraph 的 interrupt 会被 span 记成 error（GraphInterrupt 异常），但异常会原样重抛，
  不破坏图的中断语义——这是当前实现的已知简化。
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

from agents.tracing.models import NodeAggregate, TraceRecord, TraceSpan, TraceSummary
from agents.tracing.store import InMemoryTraceStore, TraceStore


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def summarize_value(value, max_items: int = 8):
    """把一个状态字段压缩成可读摘要，避免把大对象 / 长文本塞进 span。"""
    if isinstance(value, list):
        return f"{len(value)} items"
    if isinstance(value, dict):
        return {k: summarize_value(v) for k, v in list(value.items())[:max_items]}
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            return type(value).__name__
    if isinstance(value, str):
        return value[:64]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return type(value).__name__


def summarize_state(state: dict) -> dict:
    """把 state / node update 压成摘要 dict（供 span 的 input/output_summary 使用）。"""
    return {k: summarize_value(v) for k, v in state.items()}


class Tracer:
    """执行轨迹采集器。

    典型用法：
        tracer = Tracer("patrol_001")
        with tracer.span("plan") as span:
            ...   # 抛异常会自动标记 error 并重抛
        summary = tracer.analyze()   # 得到 TraceSummary
        tracer.save()                # 持久化（若传了 store）
    """

    def __init__(self, trace_id: str, store: TraceStore | None = None, agent: str = "") -> None:
        self.trace_id = trace_id
        self._store = store or InMemoryTraceStore()
        self._agent = agent
        self._spans: list[TraceSpan] = []

    @contextmanager
    def span(
        self,
        name: str,
        agent: str | None = None,
        parent: TraceSpan | None = None,
        **attributes,
    ) -> Iterator[TraceSpan]:
        """开启一个执行片段。进入 / 退出自动计时；异常自动标记 error 并重抛。"""
        span = TraceSpan(
            span_id=_new_id("span"),
            name=name,
            agent=agent or self._agent,
            parent_id=parent.span_id if parent else None,
            attributes=dict(attributes),
        )
        start = time.perf_counter()
        try:
            yield span
            span.status = "ok"
        except Exception as exc:  # noqa: BLE001 —— 记录后原样重抛，不吞异常
            span.status = "error"
            span.error = str(exc)
            raise
        finally:
            span.end_at = datetime.now()
            span.duration_ms = (time.perf_counter() - start) * 1000.0
            self._spans.append(span)

    def record(self) -> TraceRecord:
        """返回本次采集的完整轨迹（快照）。"""
        record = TraceRecord(trace_id=self.trace_id)
        record.spans = [s.model_copy(deep=True) for s in self._spans]
        return record

    def analyze(self) -> TraceSummary:
        """把已采集的 span 聚合为分析摘要。"""
        return analyze(self.record())

    def save(self) -> None:
        """持久化到绑定的 store。"""
        self._store.save(self.record())


def analyze(record: TraceRecord) -> TraceSummary:
    """核心分析：延迟分布、成本、错误、瓶颈、时间轴。"""
    summary = TraceSummary(trace_id=record.trace_id, total_spans=len(record.spans))
    spans = record.spans
    if not spans:
        return summary

    starts = [s.start_at for s in spans]
    ends = [s.end_at for s in spans if s.end_at is not None]
    if starts and ends:
        summary.total_duration_ms = (max(ends) - min(starts)).total_seconds() * 1000.0
    summary.sum_duration_ms = sum(s.duration_ms for s in spans)

    for s in spans:
        if s.status == "error":
            summary.error_count += 1
        else:
            summary.ok_count += 1
        summary.total_tokens += s.tokens
        summary.total_cost += s.cost

        agg = summary.per_node.setdefault(s.name, NodeAggregate())
        agg.count += 1
        agg.total_ms += s.duration_ms
        agg.max_ms = max(agg.max_ms, s.duration_ms)
        agg.tokens += s.tokens
        agg.cost += s.cost
        if s.status == "error":
            agg.errors += 1

    for agg in summary.per_node.values():
        if agg.count:
            agg.avg_ms = agg.total_ms / agg.count

    slowest = max(spans, key=lambda s: s.duration_ms)
    summary.bottleneck = slowest.name
    summary.bottleneck_ms = slowest.duration_ms
    summary.timeline = render_timeline(record)
    return summary


def render_timeline(record: TraceRecord) -> str:
    """渲染可读文本时间轴（轨迹）。"""
    spans = sorted(record.spans, key=lambda s: s.start_at)
    lines = [f"[{record.trace_id}] {len(spans)} spans"]
    for s in spans:
        status = "ERR" if s.status == "error" else " ok"
        token_part = f", {s.tokens}tok" if s.tokens else ""
        err_part = f"  err={s.error}" if s.status == "error" else ""
        lines.append(f"  {s.name:<18} {s.duration_ms:>9.1f}ms  {status}{token_part}{err_part}")
    return "\n".join(lines)
