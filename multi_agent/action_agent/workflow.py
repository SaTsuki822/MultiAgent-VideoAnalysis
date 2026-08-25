"""执行 Agent（Action Agent）：业务闭环。

职责：
- HITL 人工复核决策应用；
- 告警创建、工单派发、通知推送；
- 报告生成；
- 误报记忆写入、事件记忆写入。

设计要点：
- 内部基于 LangGraph 子图编排：hitl_apply → report → notify；
- 部署特点：轻量，调用 MCP Server（alarm/ticket）；与人工复核台交互。
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

import operator

from agents.models import Alarm, LogEntry
from agents.nodes.hitl import apply_decisions
from agents.nodes.notifier import notifier_node
from agents.nodes.reporter import reporter_node
from agents.state import merge_by_id
from agents.toolbox import Toolbox


# ============================================================
# Action Agent 子图状态
# ============================================================
class ActionState(TypedDict, total=False):
    """执行 Agent 内部状态。"""

    alarms: Annotated[list[Alarm], merge_by_id]
    pending_review: Annotated[list[Alarm], merge_by_id]
    patrol_id: str
    hitl_decisions: list[dict] | None
    report: Any
    logs: Annotated[list[LogEntry], operator.add]


# ============================================================
# Node 包装器
# ============================================================
def _hitl_apply_node(state: ActionState, toolbox: Toolbox) -> dict:
    """应用人工复核决策（若有）。"""
    decisions = state.get("hitl_decisions")
    pending = state.get("pending_review", [])
    if not decisions or not pending:
        return {"logs": [LogEntry(node="hitl", message="无人工决策，跳过 HITL 应用")]}
    return apply_decisions(decisions, pending, toolbox)


def _reporter_node(state: ActionState, toolbox: Toolbox) -> dict:
    return reporter_node(state, toolbox)


def _notifier_node(state: ActionState, toolbox: Toolbox) -> dict:
    return notifier_node(state, toolbox)


# ============================================================
# LangGraph 子图构建
# ============================================================
def build_action_graph(toolbox: Toolbox):
    """构建执行 Agent 的 LangGraph 子图：hitl_apply → report → notify。"""
    from functools import partial

    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(ActionState)
    builder.add_node("hitl_apply", partial(_hitl_apply_node, toolbox=toolbox))
    builder.add_node("report", partial(_reporter_node, toolbox=toolbox))
    builder.add_node("notify", partial(_notifier_node, toolbox=toolbox))

    builder.add_edge(START, "hitl_apply")
    builder.add_edge("hitl_apply", "report")
    builder.add_edge("report", "notify")
    builder.add_edge("notify", END)

    return builder.compile()


# ============================================================
# Agent 封装
# ============================================================
class ActionAgent:
    """执行 Agent：人工复核 + 告警工单 + 报告。内部由 LangGraph 子图编排。"""

    def __init__(self, toolbox: Toolbox | None = None):
        self.toolbox = toolbox
        self._graph = build_action_graph(toolbox) if toolbox else None

    def execute(
        self,
        alarms: list[dict],
        patrol_id: str,
        hitl_decisions: list[dict] | None = None,
        pending_review: list[dict] | None = None,
    ) -> dict:
        """执行业务动作（通过 LangGraph 子图）。

        pending_review：决策 Agent 分流出的「待人工复核」告警（未命中误报抑制的告警）；
        若未显式传入（旧调用），回退到全部 alarms，保证无 HITL 分隔时也能跑通。
        """
        alarm_objs = [Alarm(**a) for a in alarms]
        pending_objs = (
            [Alarm(**a) for a in pending_review] if pending_review is not None else alarm_objs
        )
        initial_state: ActionState = {
            "alarms": alarm_objs,
            "pending_review": pending_objs,
            "patrol_id": patrol_id,
            "hitl_decisions": hitl_decisions,
        }

        # 使用 LangGraph 子图执行
        if self._graph:
            final_state = self._graph.invoke(initial_state)
        else:
            # 降级：直接串调
            final_state = dict(initial_state)
            final_state.update(_hitl_apply_node(final_state, self.toolbox))
            final_state.update(_reporter_node(final_state, self.toolbox))
            final_state.update(_notifier_node(final_state, self.toolbox))

        return {
            "report": final_state.get("report", {}).model_dump() if final_state.get("report") else None,
            "alarms": [a.model_dump() for a in final_state.get("alarms", [])],
            "logs": [l.model_dump() for l in final_state.get("logs", [])],
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
                    payload.get("pending_review"),
                ),
            }
        return {"status": "error", "error": f"未知任务: {task}"}
