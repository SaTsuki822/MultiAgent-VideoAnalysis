"""规划 Agent（Planner Agent）：轻量文本推理。

职责：
- 接收自然语言规则 + 摄像头台账；
- LLM 编译规则为结构化配置（含 ReAct 自主探索）；
- 生成 camera × rule 子任务 DAG；
- 从 knowledge-server 检索相关 SOP。

设计要点：
- 内部基于 LangGraph 子图编排：plan 节点（当前单节点，可扩展为 validate → plan → enrich）；
- 部署特点：轻量，调用大模型 API，无需 GPU；可与协调 Agent 同进程，也可独立部署。
"""

from __future__ import annotations

from typing import Annotated, TypedDict

import operator

from agents.models import AnalysisTask, Camera, LogEntry, Rule
from agents.nodes.planner import planner_node
from agents.toolbox import Toolbox


# ============================================================
# Planner Agent 子图状态
# ============================================================
class PlannerState(TypedDict, total=False):
    """规划 Agent 内部状态。"""

    rules: list[Rule]
    cameras: list[Camera]
    tasks: list[AnalysisTask]
    logs: Annotated[list[LogEntry], operator.add]


# ============================================================
# Node 包装器
# ============================================================
def _plan_node(state: PlannerState, toolbox: Toolbox) -> dict:
    return planner_node(state, toolbox)


# ============================================================
# LangGraph 子图构建
# ============================================================
def build_planner_graph(toolbox: Toolbox):
    """构建规划 Agent 的 LangGraph 子图：plan（当前单节点，预留扩展）。"""
    from functools import partial

    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(PlannerState)
    builder.add_node("plan", partial(_plan_node, toolbox=toolbox))
    builder.add_edge(START, "plan")
    builder.add_edge("plan", END)
    return builder.compile()


# ============================================================
# Agent 封装
# ============================================================
class PlannerAgent:
    """规划 Agent：规则解析与任务分解。内部由 LangGraph 子图编排。"""

    def __init__(self, toolbox: Toolbox | None = None):
        self.toolbox = toolbox
        self._graph = build_planner_graph(toolbox) if toolbox else None

    def plan(self, rules: list[dict], cameras: list[dict]) -> dict:
        """生成结构化巡检任务（通过 LangGraph 子图）。"""
        rule_objs = [Rule(**r) for r in rules]
        camera_objs = [Camera(**c) for c in cameras]
        initial_state: PlannerState = {"rules": rule_objs, "cameras": camera_objs}

        # 使用 LangGraph 子图执行
        if self._graph:
            final_state = self._graph.invoke(initial_state)
        else:
            # 降级：直接串调
            final_state = planner_node(initial_state, self.toolbox)

        return {
            "tasks": [t.model_dump() for t in final_state.get("tasks", [])],
            "cameras": [c.model_dump() for c in final_state.get("cameras", [])],
            "logs": [l.model_dump() for l in final_state.get("logs", [])],
        }

    def run(self, task: str, payload: dict) -> dict:
        """HTTP 入口：接收 task 名称，执行对应逻辑。"""
        if task == "plan":
            return {"status": "success", "result": self.plan(payload.get("rules", []), payload.get("cameras", []))}
        return {"status": "error", "error": f"未知任务: {task}"}
