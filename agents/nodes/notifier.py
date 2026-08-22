"""notify 节点：对已确认告警创建工单并发送通知。"""

from __future__ import annotations

from agents.models import Alarm, LogEntry
from agents.toolbox import Toolbox


def notifier_node(state: dict, toolbox: Toolbox) -> dict:
    alarms: list[Alarm] = state.get("alarms", [])
    confirmed = [a for a in alarms if a.status == "confirmed"]
    logs: list[LogEntry] = []

    for a in confirmed:
        ticket = toolbox.create_ticket(alarm_id=a.id, assignee="安全员")
        toolbox.notify(
            channel="webhook",
            payload={"ticket_id": ticket.get("ticket_id"), "alarm_id": a.id, "severity": a.severity},
        )
        logs.append(LogEntry(node="notify", message=f"告警 {a.id} 已创建工单 {ticket.get('ticket_id')} 并通知"))

    return {"logs": logs}
