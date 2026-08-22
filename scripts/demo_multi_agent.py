"""GuardEye 多 Agent 分布式 Demo。

演示 1 个协调 Agent + 2 个感知 Agent 的并行工作流程。
"""

from __future__ import annotations

import sys
import threading
import time

# 将项目根目录加入路径
sys.path.insert(0, "F:/其他/LYK/求职/LLM/VideoAnalysis")

from multi_agent.orchestrator.workflow import Orchestrator
from multi_agent.perception_agent.workflow import PerceptionAgent


def run_perception_agent(agent_id: str, duration_sec: float = 60.0):
    """在独立线程中启动感知 Agent。"""
    agent = PerceptionAgent(agent_id)
    print(f"[Demo] 启动感知 Agent {agent_id}")
    agent.run_loop(max_idle_sec=duration_sec)


def main():
    print("=" * 60)
    print("GuardEye 多 Agent 分布式 Demo")
    print("=" * 60)

    # 启动 2 个感知 Agent（模拟多实例）
    perception_ids = ["perception-1", "perception-2"]
    threads = []
    for pid in perception_ids:
        t = threading.Thread(target=run_perception_agent, args=(pid, 30.0))
        t.start()
        threads.append(t)
        time.sleep(0.5)

    # 等待感知 Agent 就绪
    time.sleep(1.0)

    # 启动协调 Agent，执行巡检
    orch = Orchestrator(
        planner_url="http://localhost:8001",
        decision_url="http://localhost:8003",
        action_url="http://localhost:8004",
        perception_agents=perception_ids,
    )

    # Mock 规则和摄像头
    rules = [
        {"id": "rule_1", "name": "未佩戴安全帽", "description": "检查是否佩戴安全帽", "severity": "high"},
        {"id": "rule_2", "name": "明火检测", "description": "检测画面中是否有明火", "severity": "high"},
    ]
    cameras = [
        {"id": "cam_01", "name": "1号出入口", "location": "A区"},
        {"id": "cam_02", "name": "2号输送带", "location": "B区"},
        {"id": "cam_03", "name": "3号堆放区", "location": "C区"},
    ]

    print(f"\n[Demo] 协调 Agent 启动巡检：{len(rules)} 条规则 × {len(cameras)} 路摄像头")
    result = orch.run_patrol(rules, cameras)
    print(f"\n[Demo] 巡检结果：{result}")

    # 等待感知 Agent 线程结束
    for t in threads:
        t.join(timeout=5.0)

    print("\n[Demo] 结束")


if __name__ == "__main__":
    main()
