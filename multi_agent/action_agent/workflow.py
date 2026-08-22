"""执行 Agent（Action Agent）：业务闭环。

职责：
- HITL 人工复核中断与恢复；
- 告警创建、工单派发、通知推送；
- 报告生成；
- 误报记忆写入、事件记忆写入。

部署特点：
- 轻量，调用 MCP Server（alarm/ticket）；
- 与人工复核台（Streamlit/Next.js）交互。
"""

from __future__ import annotations

from typing import Any

from agents.nodes.hitl import apply_decisions
from agents.nodes.reporter import reporter_node
from agents.nodes.notifier import notifier_node
from agents.toolbox import Toolbox


class ActionAgent:
    """执行 Agent：人工复核 + 告警工单 + 报告。"""

    def __init__(self, toolbox: Toolbox | None = None):
        self.toolbox = toolbox

    def execute(self, alarms: list[dict], patrol_id: str, hitl_decisions: list[dict] | None = None) -> dict:
        """执行业务动作。"""
        from agents.models import Alarm, LogEntry

        alarm_objs = [Alarm(**a) for a in alarms]
        state = {"alarms": alarm_objs, "pending_review": alarm_objs, "patrol_id": patrol_id}

        # HITL：如果有决策则应用，否则 pending_review 保留
        if hitl_decisions:
            hitl_result = apply_decisions(hitl_decisions, alarm_objs, self.toolbox)
            state.update(hitl_result)

        # 报告
        rep_result = reporter_node(state, self.toolbox)
        state.update(rep_result)

        # 通知
        not_result = notifier_node(state, self.toolbox)
        state.update(not_result)

        return {
            "report": state.get("report", {}).model_dump() if state.get("report") else None,
            "alarms": [a.model_dump() for a in state.get("alarms", [])],
            "logs": [l.model_dump() for l in state.get("logs", [])],
        }

    def run(self, task: str, payload: dict) -> dict:
        """HTTP 入口。"""
        if task == "execute":
            return {
                "status": "success",
                "result": self.execute(
                    payload.get("alarms", []),
                    payload.get("patrol_id", ""),
                    payload.get("hitl_decisions"),
                ),
            }
        return {"status": "error", "error": f"未知任务: {task}"}
