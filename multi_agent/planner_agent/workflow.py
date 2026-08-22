"""规划 Agent（Planner Agent）：轻量文本推理。

职责：
- 接收自然语言规则 + 摄像头台账；
- LLM 编译规则为结构化配置；
- 生成 camera × rule 子任务 DAG；
- 从 knowledge-server 检索相关 SOP。

部署特点：
- 轻量，调用大模型 API，无需 GPU；
- 可与协调 Agent 同进程，也可独立部署。
"""

from __future__ import annotations

from typing import Any

from agents.nodes.planner import planner_node
from agents.toolbox import Toolbox


class PlannerAgent:
    """规划 Agent：规则解析与任务分解。"""

    def __init__(self, toolbox: Toolbox | None = None):
        self.toolbox = toolbox

    def plan(self, rules: list[dict], cameras: list[dict]) -> dict:
        """生成结构化巡检任务。"""
        from agents.models import Rule, Camera

        rule_objs = [Rule(**r) for r in rules]
        camera_objs = [Camera(**c) for c in cameras]
        state = {"rules": rule_objs, "cameras": camera_objs}
        result = planner_node(state, self.toolbox)
        return {
            "tasks": [t.model_dump() for t in result.get("tasks", [])],
            "cameras": [c.model_dump() for c in result.get("cameras", [])],
            "logs": [l.model_dump() for l in result.get("logs", [])],
        }

    def run(self, task: str, payload: dict) -> dict:
        """HTTP 入口：接收 task 名称，执行对应逻辑。"""
        if task == "plan":
            return {"status": "success", "result": self.plan(payload.get("rules", []), payload.get("cameras", []))}
        return {"status": "error", "error": f"未知任务: {task}"}
