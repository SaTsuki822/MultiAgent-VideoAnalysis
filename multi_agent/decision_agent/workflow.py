"""决策 Agent（Decision Agent）：准确率核心。

职责：
- 跨帧一致性校验；
- 大模型二次确认（带 SOP 依据）；
- 误报记忆检索与抑制；
- 时序聚合、风险定级。

设计要点：
- 内部基于 LangGraph 子图编排：verify → memory_filter，支持后续扩展更多复核节点；
- 部署特点：中量，调用大模型 API + Qdrant 检索；无需常驻 GPU。
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

import operator

from agents.models import LogEntry
from agents.nodes.memory_filter import memory_filter_node
from agents.nodes.temporal_aggregator import temporal_aggregate_node
from agents.nodes.verifier import verifier_node
from agents.state import merge_by_id
from agents.toolbox import Toolbox


# ============================================================
# Decision Agent 子图状态
# ============================================================
class DecisionState(TypedDict, total=False):
    """决策 Agent 内部状态：只需持有 verifier + memory_filter 所需字段。"""

    findings: list
    tasks: list
    verifications: Annotated[list, operator.add]
    alarms: Annotated[list, merge_by_id]
    pending_review: Annotated[list, merge_by_id]
    logs: Annotated[list[LogEntry], operator.add]


# ============================================================
# Node 包装器（把原 node 函数适配为 DecisionState -> partial update）
# ============================================================
def _verify_node(state: DecisionState, toolbox: Toolbox) -> dict:
    return verifier_node(state, toolbox)


def _temporal_aggregate_node(state: DecisionState, toolbox: Toolbox) -> dict:
    return temporal_aggregate_node(state, toolbox)


def _memory_filter_node(state: DecisionState, toolbox: Toolbox) -> dict:
    return memory_filter_node(state, toolbox)


# ============================================================
# LangGraph 子图构建
# ============================================================
def build_decision_graph(toolbox: Toolbox):
    """构建决策 Agent 的 LangGraph 子图：verify → temporal_aggregate → memory_filter。"""
    from functools import partial

    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(DecisionState)
    builder.add_node("verify", partial(_verify_node, toolbox=toolbox))
    builder.add_node("temporal_aggregate", partial(_temporal_aggregate_node, toolbox=toolbox))
    builder.add_node("memory_filter", partial(_memory_filter_node, toolbox=toolbox))

    builder.add_edge(START, "verify")
    builder.add_edge("verify", "temporal_aggregate")
    builder.add_edge("temporal_aggregate", "memory_filter")
    builder.add_edge("memory_filter", END)

    return builder.compile()


# ============================================================
# Agent 封装
# ============================================================
class DecisionAgent:
    """决策 Agent：复核确认 + 记忆检索。内部由 LangGraph 子图编排。"""

    def __init__(self, toolbox: Toolbox | None = None):
        self.toolbox = toolbox
        self._graph = build_decision_graph(toolbox) if toolbox else None

    def verify(self, findings: list[dict], tasks: list[dict]) -> dict:
        """执行复核与记忆过滤（通过 LangGraph 子图）。"""
        from agents.models import AnalysisTask, Camera, Finding, Rule

        finding_objs = [Finding(**f) for f in findings]
        task_objs = []
        for t in tasks:
            task_objs.append(
                AnalysisTask(
                    id=t["id"],
                    camera_id=t["camera_id"],
                    rule=Rule(**t.get("rule", {})),
                    clip_path=t.get("clip_path", ""),
                )
            )

        initial_state: DecisionState = {
            "findings": finding_objs,
            "tasks": task_objs,
        }

        # 使用 LangGraph 子图执行
        if self._graph:
            final_state = self._graph.invoke(initial_state)
        else:
            # 降级：直接串调（兼容无 toolbox 的单测场景）
            final_state = dict(initial_state)
            final_state.update(verifier_node(final_state, self.toolbox))
            final_state.update(temporal_aggregate_node(final_state, self.toolbox))
            final_state.update(memory_filter_node(final_state, self.toolbox))

        return {
            "verifications": [v.model_dump() for v in final_state.get("verifications", [])],
            "alarms": [a.model_dump() for a in final_state.get("alarms", [])],
            "pending_review": [a.model_dump() for a in final_state.get("pending_review", [])],
            "logs": [l.model_dump() for l in final_state.get("logs", [])],
        }

    def run(self, task: str, payload: dict) -> dict:
        """HTTP 入口。"""
        if task == "verify":
            return {"status": "success", "result": self.verify(payload.get("findings", []), payload.get("tasks", []))}
        return {"status": "error", "error": f"未知任务: {task}"}
