"""memory_filter 节点：用误报记忆抑制已知误报。

对 verify 产出的每条告警，检索误报记忆库；命中「已激活」的误报签名则标记 suppressed，
否则进入 pending_review 等待人工复核。防污染规则在 memory/false_positive.py 的 should_suppress 中。
"""

from __future__ import annotations

from agents.config import get_settings
from agents.llm import get_llm_client
from agents.memory.embedding import get_embedder
from agents.memory.false_positive import lookup_similar_false_positive, should_suppress
from agents.memory.vector_store import get_vector_store
from agents.models import Alarm, LogEntry
from agents.toolbox import Toolbox


def memory_filter_node(
    state: dict,
    toolbox: Toolbox,
    store=None,
    embedder=None,
    llm=None,
) -> dict:
    alarms: list[Alarm] = state.get("alarms", [])
    settings = get_settings()
    store = store or get_vector_store(settings)
    embedder = embedder or get_embedder(settings)
    llm = llm or get_llm_client()

    pending: list[Alarm] = []
    suppressed_count = 0
    for alarm in alarms:
        hits = lookup_similar_false_positive(alarm, store, embedder, llm, settings)
        suppress, reason = should_suppress(alarm, hits, settings)
        if suppress:
            alarm.suppressed = True
            alarm.suppression_reason = reason
            alarm.status = "suppressed"
            suppressed_count += 1
        else:
            pending.append(alarm)

    log = LogEntry(
        node="memory_filter",
        message=f"{len(alarms)} 条告警中抑制 {suppressed_count} 条已知误报，{len(pending)} 条进入人工复核",
    )
    return {"alarms": alarms, "pending_review": pending, "logs": [log]}
