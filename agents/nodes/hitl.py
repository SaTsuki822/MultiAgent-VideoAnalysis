"""HITL 人工复核节点（对应面试 Q8/Q12「什么时候让人介入」）。

人工只介入「高价值、不确定」的告警：经过三级路由 + 一致性 + 记忆过滤后仍待确认的告警。
人工决策四选一：confirm（确认）/ false_positive（误报，进入记忆学习）/ change_severity（改级）/ dismiss。

关键闭环：标记误报 → 写入误报记忆 → 下次同类告警被自动抑制（这是整个 demo 最打动人之处）。
本模块不 import langgraph，interrupt 由 workflow 层通过 hitl_handler 注入，保证节点可独立单测。
"""

from __future__ import annotations

from agents.config import get_settings
from agents.llm import get_llm_client
from agents.memory.embedding import get_embedder
from agents.memory.event_memory import remember_event
from agents.memory.false_positive import remember_false_positive
from agents.memory.vector_store import get_vector_store
from agents.models import Alarm, Feedback, LogEntry
from agents.toolbox import Toolbox


def apply_decisions(
    decisions: list[dict],
    pending: list[Alarm],
    toolbox: Toolbox,
    store=None,
    embedder=None,
    llm=None,
) -> dict:
    """把人工决策应用到告警，并触发记忆层更新。返回 (feedback, 更新后的 alarms, logs)。"""
    settings = get_settings()
    store = store or get_vector_store(settings)
    embedder = embedder or get_embedder(settings)
    llm = llm or get_llm_client()

    by_id = {a.id: a for a in pending}
    feedback: list[Feedback] = []
    updated: list[Alarm] = []

    for d in decisions:
        alarm = by_id.get(d.get("alarm_id"))
        if alarm is None:
            continue
        decision = d.get("decision", "dismiss")

        if decision == "confirm":
            alarm.status = "confirmed"
            remember_event(alarm, store, embedder, llm, settings)
        elif decision == "false_positive":
            alarm.status = "false_positive"
            remember_false_positive(alarm, d.get("comment", ""), store, embedder, llm, settings)
        elif decision == "change_severity":
            new_sev = d.get("new_severity")
            if new_sev in {"high", "medium", "low"}:
                alarm.severity = new_sev
            alarm.status = "confirmed"
        else:  # dismiss
            alarm.status = "dismissed"

        feedback.append(
            Feedback(
                alarm_id=alarm.id,
                decision=decision,
                new_severity=d.get("new_severity"),
                comment=d.get("comment", ""),
            )
        )
        updated.append(alarm)

    log = LogEntry(node="hitl", message=f"人工复核 {len(updated)} 条告警")
    return {"feedback": feedback, "alarms": updated, "logs": [log]}


def hitl_node(state: dict, toolbox: Toolbox, hitl_handler=None) -> dict:
    """挂起等待人工复核（若有待复核告警）。

    hitl_handler(pending) -> decisions。在 LangGraph 图中由 workflow 注入 interrupt 实现；
    在 demo / 单测中由调用方注入回调（如命令行交互、脚本预置决策）。
    """
    pending: list[Alarm] = state.get("pending_review", [])
    if not pending:
        return {"logs": [LogEntry(node="hitl", message="无待复核告警，跳过人工介入")]}

    if hitl_handler is None:
        raise RuntimeError("存在待复核告警，但未提供 hitl_handler（人工决策入口）")

    decisions = hitl_handler(pending)
    return apply_decisions(decisions, pending, toolbox)
