"""主状态图：plan → fetch → dispatch → verify → memory_filter → hitl → report → notify。

提供两种运行形态（对应面试 Q3「为什么用 LangGraph」）：
1. build_graph()：LangGraph StateGraph + checkpointer + interrupt —— 生产形态，
   可讲 checkpoint 断点续跑、interrupt 人工介入、状态 reducer 合并；
2. run_pipeline()：纯 Python 顺序执行 —— 降级形态，不依赖 LangGraph 也能端到端跑，
   用于 demo 与单测（保证「无 GPU / 无 API key / 无 LangGraph 时仍可跑通闭环」）。

两者共用同一批节点函数，行为一致；差异只在「有无 checkpoint 持久化与中断恢复能力」。
"""

from __future__ import annotations

from functools import partial
from typing import Callable

from agents.nodes import dispatcher, fetcher, hitl, memory_filter, notifier, planner, reporter, temporal_aggregator, verifier
from agents.state import PatrolState, merge_by_id
from agents.toolbox import Toolbox
from agents.tracing import Tracer, summarize_state

# 追加式 reducer 字段与按 id 合并字段（与 state.py 保持一致）
_APPEND_FIELDS = {"findings", "verifications", "feedback", "logs"}
_MERGE_FIELDS = {"alarms", "pending_review"}


def merge_state(state: dict, update: dict) -> dict:
    """按 reducer 语义合并节点更新（run_pipeline 用，模拟 LangGraph 的 reducer 行为）。"""
    new = dict(state)
    for key, value in update.items():
        if key in _APPEND_FIELDS:
            new[key] = list(new.get(key, [])) + list(value)
        elif key in _MERGE_FIELDS:
            new[key] = merge_by_id(list(new.get(key, [])), list(value))
        else:
            new[key] = value
    return new


def _wrap_node(name: str, fn: Callable, tracer: "Tracer | None") -> Callable:
    """若有 tracer，给节点包一层执行轨迹采集（计时 + 输入/输出摘要 + 状态）。"""
    if tracer is None:
        return fn

    def wrapped(state: dict) -> dict:
        with tracer.span(name) as span:
            span.input_summary = {"state_keys": list(state.keys())}
            update = fn(state) or {}
            span.output_summary = summarize_state(update)
            return update

    return wrapped


def _interrupt_handler(pending):
    """LangGraph 图里的人工介入：调 interrupt 挂起，等待 Command(resume=...) 恢复。"""
    from langgraph.types import interrupt

    return interrupt({"pending_review": [a.model_dump() for a in pending]})


def build_graph(toolbox: Toolbox, hitl_handler: Callable | None = None, checkpointer=None, tracer: "Tracer | None" = None):
    """构建 LangGraph 状态图。checkpointer 默认 MemorySaver，可传 SqliteSaver 持久化。"""
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(PatrolState)
    builder.add_node("plan", _wrap_node("plan", partial(planner.planner_node, toolbox=toolbox), tracer))
    builder.add_node("fetch", _wrap_node("fetch", partial(fetcher.fetcher_node, toolbox=toolbox), tracer))
    builder.add_node("dispatch", _wrap_node("dispatch", partial(dispatcher.dispatch_node, toolbox=toolbox), tracer))
    builder.add_node("verify", _wrap_node("verify", partial(verifier.verifier_node, toolbox=toolbox), tracer))
    builder.add_node("temporal_aggregate", _wrap_node("temporal_aggregate", partial(temporal_aggregator.temporal_aggregate_node, toolbox=toolbox), tracer))
    builder.add_node("memory_filter", _wrap_node("memory_filter", partial(memory_filter.memory_filter_node, toolbox=toolbox), tracer))
    builder.add_node("hitl", _wrap_node("hitl", partial(hitl.hitl_node, toolbox=toolbox, hitl_handler=hitl_handler or _interrupt_handler), tracer))
    builder.add_node("report", _wrap_node("report", partial(reporter.reporter_node, toolbox=toolbox), tracer))
    builder.add_node("notify", _wrap_node("notify", partial(notifier.notifier_node, toolbox=toolbox), tracer))

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "fetch")
    builder.add_edge("fetch", "dispatch")
    builder.add_edge("dispatch", "verify")
    builder.add_edge("verify", "temporal_aggregate")
    builder.add_edge("temporal_aggregate", "memory_filter")
    builder.add_edge("memory_filter", "hitl")
    builder.add_edge("hitl", "report")
    builder.add_edge("report", "notify")
    builder.add_edge("notify", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


def run_pipeline(initial_state: dict, toolbox: Toolbox, hitl_handler: Callable | None = None, tracer: "Tracer | None" = None) -> dict:
    """纯 Python 顺序执行节点（降级形态）。返回完整 state。

    可选传入 tracer：采集每个节点的执行轨迹（耗时/状态/摘要），用于轨迹分析。
    """
    state = dict(initial_state)
    nodes: list[tuple[str, Callable]] = [
        ("plan", partial(planner.planner_node, toolbox=toolbox)),
        ("fetch", partial(fetcher.fetcher_node, toolbox=toolbox)),
        ("dispatch", partial(dispatcher.dispatch_node, toolbox=toolbox)),
        ("verify", partial(verifier.verifier_node, toolbox=toolbox)),
        ("temporal_aggregate", partial(temporal_aggregator.temporal_aggregate_node, toolbox=toolbox)),
        ("memory_filter", partial(memory_filter.memory_filter_node, toolbox=toolbox)),
        ("hitl", partial(hitl.hitl_node, toolbox=toolbox, hitl_handler=hitl_handler)),
        ("report", partial(reporter.reporter_node, toolbox=toolbox)),
        ("notify", partial(notifier.notifier_node, toolbox=toolbox)),
    ]
    for name, node in nodes:
        update = _wrap_node(name, node, tracer)(state) or {}
        state = merge_state(state, update)
    return state
