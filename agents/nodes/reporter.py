"""report 节点：生成结构化巡检报告。"""

from __future__ import annotations

from agents.llm import MockLLMClient, get_llm_client
from agents.models import Alarm, LogEntry, Report
from agents.toolbox import Toolbox


def _summary_mock(alarms: list[Alarm], stats: dict) -> str:
    return (
        f"本次巡检共产生 {stats['total_alarms']} 条告警："
        f"确认 {stats['confirmed']} 条，抑制 {stats['suppressed']} 条，误报 {stats['false_positive']} 条。"
    )


def reporter_node(state: dict, toolbox: Toolbox) -> dict:
    alarms: list[Alarm] = state.get("alarms", [])
    findings = state.get("findings", [])

    stats = {
        "total_findings": len(findings),
        "hit_findings": sum(1 for f in findings if f.hit),
        "total_alarms": len(alarms),
        "confirmed": sum(1 for a in alarms if a.status == "confirmed"),
        "suppressed": sum(1 for a in alarms if a.suppressed),
        "false_positive": sum(1 for a in alarms if a.status == "false_positive"),
        "pending": sum(1 for a in alarms if a.status == "pending_review"),
    }

    llm = get_llm_client()
    if isinstance(llm, MockLLMClient):
        summary = _summary_mock(alarms, stats)
    else:
        lines = "\n".join(
            f"- [{a.severity}] {a.rule_name} @ {a.camera_id}（{a.status}）" for a in alarms
        )
        raw = llm.complete(
            system="你是巡检报告助手，用 2~3 句话概括一次巡检的结果。",
            user=f"统计：{stats}\n告警列表：\n{lines or '无'}",
        )
        summary = raw.strip() or _summary_mock(alarms, stats)

    report = Report(
        patrol_id=state.get("patrol_id", ""),
        summary=summary,
        alarms=alarms,
        stats=stats,
    )
    log = LogEntry(node="report", message=f"生成巡检报告（{stats['total_alarms']} 条告警）")
    return {"report": report, "logs": [log]}
