"""感知 Agent（Perception Agent）：GPU 密集型，水平扩展核心。

职责：
- 接收协调 Agent 分发的视频片段 + 规则；
- 执行 L0→L1→L2 三级路由；
- 返回候选 Finding 列表。

设计原则：
- **无状态化**：输入只有 clip + rule，输出只有 finding，不保存中间帧；
- **幂等性**：同一任务执行多次结果一致；
- **成本记账**：每帧记录处理成本，回传给协调 Agent。
"""

from __future__ import annotations

import time
from typing import Any

from a2a_protocol.message_schema import A2AMessage, A2AResult
from a2a_protocol.redis_stream import RedisStreamClient


class PerceptionAgent:
    """感知 Agent：三级路由视频分析。"""

    def __init__(self, agent_id: str, redis_client=None):
        self.agent_id = agent_id
        self.redis = RedisStreamClient(redis_client=redis_client)

    def analyze(self, clip_path: str, rule: dict, camera_id: str) -> dict:
        """执行三级路由分析（复用现有 prescreen 逻辑）。"""
        # 复用现有 agents/prescreen/ 下的实现
        from agents.nodes.prescreen import analyze_task
        from agents.models import AnalysisTask, Rule

        task = AnalysisTask(
            id=f"task_{camera_id}_{rule.get('id', 'unknown')}",
            camera_id=camera_id,
            rule=Rule(**rule),
            clip_path=clip_path,
        )
        # analyze_task 内部调用 toolbox，这里简化为直接调用
        # 生产环境中，感知 Agent 应有自己的 Toolbox 实例（只连接 video_analysis MCP）
        finding = analyze_task(task, toolbox=None)
        if finding is None:
            return {"findings": [], "cost": {"l0": 0, "l1": 0, "l2": 0}}
        return {
            "findings": [finding.model_dump()],
            "cost": {"l0": 10, "l1": 5, "l2": 1},  # 占位，真实应从 router 记账
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
