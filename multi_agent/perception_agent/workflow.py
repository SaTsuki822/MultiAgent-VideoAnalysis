"""感知 Agent（Perception Agent）：GPU 密集型，水平扩展核心。

职责：
- 接收协调 Agent 分发的视频片段 + 规则；
- 执行 L0→L1→L2 三级模型路由；
- 返回候选 Finding 列表 + 成本明细。

设计要点：
- 内部基于 LangGraph 子图编排：analyze 节点（内部封装 L0/L1/L2 三级路由）；
- 无状态化：输入只有 clip + rule，输出只有 finding，不保存中间帧；
- 幂等性：同一任务执行多次结果一致。
"""

from __future__ import annotations

import time
from typing import Annotated, Any, TypedDict

import operator

from a2a_protocol.message_schema import A2AMessage, A2AResult
from a2a_protocol.redis_stream import RedisStreamClient


# ============================================================
# Perception Agent 子图状态
# ============================================================
class PerceptionState(TypedDict, total=False):
    """感知 Agent 内部状态。"""

    task: dict  # AnalysisTask 的字典形式
    finding: dict | None
    cost: dict
    logs: Annotated[list, operator.add]


# ============================================================
# Node 包装器（内部走三级路由）
# ============================================================
def _analyze_node(state: PerceptionState) -> dict:
    """执行三级路由分析（L0 运动检测 → L1 抽帧去重 → L2 VLM 初筛）。"""
    from agents.models import AnalysisTask, Finding, Rule
    from agents.prescreen.router import Router

    task_dict = state.get("task", {})
    if not task_dict:
        return {"finding": None, "cost": {}, "logs": [{"node": "perception", "message": "空任务"}]}

    task = AnalysisTask(
        id=task_dict.get("id", ""),
        camera_id=task_dict.get("camera_id", ""),
        rule=Rule(**task_dict.get("rule", {})),
        clip_path=task_dict.get("clip_path", ""),
        clip_start_time=task_dict.get("clip_start_time"),
        clip_end_time=task_dict.get("clip_end_time"),
        duration_seconds=task_dict.get("duration_seconds", 0.0),
    )

    # 使用 Router 走三级路由（L0→L1→L2）
    router = Router()
    finding, breakdown = router.route(task)

    return {
        "finding": finding.model_dump(),
        "cost": {
            "l1_kept_frames": breakdown.l1_kept_frames,
            "l2_screened_frames": breakdown.l2_screened_frames,
            "l2_hit_frames": breakdown.l2_hit_frames,
            "total_tokens": breakdown.total_tokens,
            "total_cost": breakdown.total_cost,
        },
        "logs": [{"node": "perception", "message": f"分析完成 {task.id} hit={finding.hit}"}],
    }


# ============================================================
# LangGraph 子图构建
# ============================================================
def build_perception_graph():
    """构建感知 Agent 的 LangGraph 子图：analyze（内部 L0→L1→L2）。

    当前为单节点封装，三级路由在 _analyze_node 内部通过 Router 顺序执行。
    未来可按 L0/L1/L2 拆分为独立 node，实现更细粒度的 checkpoint 与并发采样。
    """
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(PerceptionState)
    builder.add_node("analyze", _analyze_node)
    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", END)
    return builder.compile()


# ============================================================
# Agent 封装
# ============================================================
class PerceptionAgent:
    """感知 Agent：三级路由视频分析。内部由 LangGraph 子图编排。"""

    def __init__(self, agent_id: str, redis_client=None):
        self.agent_id = agent_id
        self.redis = RedisStreamClient(redis_client=redis_client)
        self._graph = build_perception_graph()

    def analyze(self, clip_path: str, rule: dict, camera_id: str) -> dict:
        """执行三级路由分析（通过 LangGraph 子图）。"""
        from agents.models import AnalysisTask, Rule

        task = AnalysisTask(
            id=f"task_{camera_id}_{rule.get('id', 'unknown')}",
            camera_id=camera_id,
            rule=Rule(**rule),
            clip_path=clip_path,
        )

        initial_state: PerceptionState = {"task": task.model_dump()}
        final_state = self._graph.invoke(initial_state)

        finding = final_state.get("finding")
        if finding is None:
            return {"findings": [], "cost": final_state.get("cost", {})}
        return {
            "findings": [finding],
            "cost": final_state.get("cost", {}),
        }

    def process_message(self, message: A2AMessage) -> A2AResult:
        """处理协调 Agent 分发的任务。"""
        payload = message.payload
        try:
            result = self.analyze(
                clip_path=payload.get("clip_path", ""),
                rule=payload.get("rule", {}),
                camera_id=payload.get("camera_id", ""),
            )
            return A2AResult(
                message_id=message.message_id,
                from_agent=self.agent_id,
                to_agent="orchestrator",
                status="success",
                result=result,
            )
        except Exception as e:
            return A2AResult(
                message_id=message.message_id,
                from_agent=self.agent_id,
                to_agent="orchestrator",
                status="error",
                error=str(e),
            )

    def run_loop(self, max_idle_sec: float = 30.0):
        """持续消费 Redis Stream 中的任务（生产模式）。"""
        print(f"[PerceptionAgent {self.agent_id}] 启动，等待任务...")
        idle_since = time.time()
        while True:
            tasks = self.redis.consume_task(self.agent_id, count=1, block_ms=5000)
            if tasks:
                idle_since = time.time()
                for stream_id, msg in tasks:
                    print(f"[PerceptionAgent {self.agent_id}] 处理任务 {msg.message_id}")
                    result = self.process_message(msg)
                    self.redis.ack_task(stream_id)
                    self.redis.send_result(result)
                    print(f"[PerceptionAgent {self.agent_id}] 任务 {msg.message_id} 完成")
            else:
                if time.time() - idle_since > max_idle_sec:
                    print(f"[PerceptionAgent {self.agent_id}] 空闲超时，退出")
                    break


def main():
    """CLI 入口：python -m multi_agent.perception_agent.workflow perception-1"""
    import sys

    agent_id = sys.argv[1] if len(sys.argv) > 1 else "perception-1"
    agent = PerceptionAgent(agent_id)
    agent.run_loop()


if __name__ == "__main__":
    main()
