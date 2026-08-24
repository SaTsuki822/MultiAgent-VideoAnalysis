"""执行轨迹采集与分析单测。

覆盖：span 采集（耗时 / 状态 / 异常）、聚合分析（瓶颈 / 成本 / 时间轴）、
      run_pipeline 接入 tracer、JSONL 持久化回读。
"""

from __future__ import annotations

import pytest

from agents.memory.vector_store import reset_vector_store
from agents.tracing import JsonlTraceStore, Tracer
from agents.workflow import run_pipeline


def test_tracer_records_span_duration_and_status():
    tracer = Tracer("t1")
    with tracer.span("plan") as span:
        span.output_summary = {"tasks": 3}
    record = tracer.record()
    assert record.trace_id == "t1"
    assert len(record.spans) == 1
    span = record.spans[0]
    assert span.name == "plan"
    assert span.status == "ok"
    assert span.duration_ms >= 0
    assert span.output_summary == {"tasks": 3}


def test_tracer_marks_error_on_exception():
    tracer = Tracer("t2")
    with pytest.raises(ValueError):
        with tracer.span("verify"):
            raise ValueError("boom")
    span = tracer.record().spans[0]
    assert span.status == "error"
    assert "boom" in span.error


def test_analyze_computes_bottleneck_and_totals():
    tracer = Tracer("t3")
    with tracer.span("plan") as span:
        span.tokens = 10
    with tracer.span("dispatch") as span:
        span.tokens = 100
    summary = tracer.analyze()
    assert summary.total_spans == 2
    assert summary.total_tokens == 110
    assert summary.error_count == 0
    assert summary.bottleneck in {"plan", "dispatch"}
    assert summary.per_node["dispatch"].tokens == 100
    assert summary.timeline


def test_run_pipeline_with_tracer(fake_toolbox, medium_rule, camera):
    reset_vector_store()
    tracer = Tracer("patrol_trace")
    initial = {"patrol_id": "p_t", "rules": [medium_rule], "cameras": [camera]}
    state = run_pipeline(
        initial,
        fake_toolbox,
        hitl_handler=lambda p: [{"alarm_id": a.id, "decision": "confirm"} for a in p],
        tracer=tracer,
    )
    summary = tracer.analyze()
    expected_nodes = [
        "plan",
        "fetch",
        "dispatch",
        "verify",
        "temporal_aggregate",
        "memory_filter",
        "hitl",
        "report",
        "notify",
    ]
    assert list(summary.per_node.keys()) == expected_nodes
    assert summary.total_spans == 9
    assert summary.error_count == 0
    assert state["report"] is not None


def test_jsonl_store_roundtrip(tmp_path):
    store = JsonlTraceStore(path=tmp_path / "traces.jsonl")
    tracer = Tracer("t4", store=store)
    with tracer.span("plan"):
        pass
    tracer.save()
    loaded = store.load("t4")
    assert loaded is not None
    assert loaded.trace_id == "t4"
    assert len(loaded.spans) == 1
    assert loaded.spans[0].name == "plan"
    assert "t4" in store.list_ids()
