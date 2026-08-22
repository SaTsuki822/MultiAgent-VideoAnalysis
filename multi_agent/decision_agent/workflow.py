"""决策 Agent（Decision Agent）：准确率核心。

职责：
- 跨帧一致性校验；
- 大模型二次确认（带 SOP 依据）；
- 误报记忆检索与抑制；
- 时序聚合、风险定级。

部署特点：
- 中量，调用大模型 API + Qdrant 检索；
- 无需常驻 GPU。
"""

from __future__ import annotations

from typing import Any

from agents.nodes.verifier import verifier_node
from agents.nodes.memory_filter import memory_filter_node
from agents.toolbox import Toolbox


class DecisionAgent:
    """决策 Agent：复核确认 + 记忆检索。"""

    def __init__(self, toolbox: Toolbox | None = None):
        self.toolbox = toolbox

    def verify(self, findings: list[dict], tasks: list[dict]) -> dict:
        """执行复核与记忆过滤。"""
        from agents.models import Finding, AnalysisTask, Rule, Camera

        finding_objs = [Finding(**f) for f in findings]
        task_objs = []
        for t in tasks:
            task_objs.append(AnalysisTask(
                id=t["id"],
                camera_id=t["camera_id"],
                rule=Rule(**t.get("rule", {})),
                clip_path=t.get("clip_path", ""),
            ))

        state = {"findings": finding_objs, "tasks": task_objs}

        # 先 verifier
        v_result = verifier_node(state, self.toolbox)
        state.update(v_result)

        # 再 memory_filter
        mf_result = memory_filter_node(state, self.toolbox)
        state.update(mf_result)

        return {
            "verifications": [v.model_dump() for v in state.get("verifications", [])],
            "alarms": [a.model_dump() for a in state.get("alarms", [])],
            "pending_review": [a.model_dump() for a in state.get("pending_review", [])],
            "logs": [l.model_dump() for l in state.get("logs", [])],
        }

    def run(self, task: str, payload: dict) -> dict:
        """HTTP 入口。"""
        if task == "verify":
            return {"status": "success", "result": self.verify(payload.get("findings", []), payload.get("tasks", []))}
        return {"status": "error", "error": f"未知任务: {task}"}
