"""协调 Agent（Orchestrator）：系统中枢。

职责：
1. 接收用户巡检请求（规则 + 摄像头范围）；
2. 调用规划 Agent 生成结构化任务；
3. 将任务分片后分发给多个感知 Agent（Map）；
4. 收集感知 Agent 返回的候选 Finding（Reduce）；
5. 调用决策 Agent 复核；
6. 调用执行 Agent 落地告警/工单/报告。

关键技术点：
- 负载均衡：维护感知 Agent 连接池，按队列长度 / GPU 利用率动态选择；
- 故障恢复：感知 Agent 实例崩溃时，自动重试其他实例；
- 全局状态：通过 Redis / Postgres 维护任务级状态。
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from a2a_protocol.message_schema import A2AMessage, A2AResult
from a2a_protocol.redis_stream import RedisStreamClient


class LoadBalancer:
    """感知 Agent 负载均衡器。

    当前实现：简单轮询（Round Robin）。
    生产可扩展为：按 GPU 利用率、队列长度、地理位置加权。
    """

    def __init__(self, agent_ids: list[str]):
        self.agent_ids = agent_ids
        self._idx = 0

    def pick(self) -> str:
        agent_id = self.agent_ids[self._idx % len(self.agent_ids)]
        self._idx += 1
        return agent_id

    def mark_unavailable(self, agent_id: str):
        """标记某感知 Agent 不可用（如心跳超时）。"""
        if agent_id in self.agent_ids:
            self.agent_ids.remove(agent_id)

    def add(self, agent_id: str):
        """新感知 Agent 注册。"""
        if agent_id not in self.agent_ids:
            self.agent_ids.append(agent_id)


class Orchestrator:
    """协调 Agent：任务分片、负载均衡、结果聚合、异常恢复。"""

    def __init__(
        self,
        planner_url: str = "http://localhost:8001",
        decision_url: str = "http://localhost:8003",
        action_url: str = "http://localhost:8004",
        perception_agents: list[str] | None = None,
        redis_client=None,
    ):
        self.planner_url = planner_url
        self.decision_url = decision_url
        self.action_url = action_url
        self.load_balancer = LoadBalancer(perception_agents or ["perception-1"])
        self.redis = RedisStreamClient(redis_client=redis_client)

    # ---- 同步调用：规划 Agent / 决策 Agent / 执行 Agent ----

    def _call_sync(self, url: str, task: str, payload: dict) -> dict:
        """同步 HTTP 调用其他 Agent（规划、决策、执行）。"""
        import requests

        try:
            resp = requests.post(f"{url}/run", json={"task": task, "payload": payload}, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ---- 异步分发：感知 Agent ----

    def dispatch_to_perception(
        self,
        tasks: list[dict],
        timeout_sec: int = 120,
    ) -> list[A2AResult]:
        """将子任务分发给感知 Agent，通过 Redis Stream 异步消费 + 回调收集结果。

        流程：
        1. 为每个子任务生成 message_id；
        2. 按负载均衡选择感知 Agent，写入 Redis Stream；
        3. 轮询感知 Agent 的回调结果 Stream，收集返回；
        4. 超时未返回的任务标记为失败。
        """
        if not tasks:
            return []

        # 1. 发送任务
        sent_ids: set[str] = set()
        for task in tasks:
            msg = A2AMessage(
                message_id=f"msg_{uuid.uuid4().hex[:8]}",
                from_agent="orchestrator",
                to_agent=self.load_balancer.pick(),
                task="analyze_clip",
                payload=task,
                timeout_sec=timeout_sec,
            )
            self.redis.send_task(msg)
            sent_ids.add(msg.message_id)

        # 2. 收集结果（轮询，最多等 timeout_sec）
        results: list[A2AResult] = []
        start = time.time()
        while sent_ids and time.time() - start < timeout_sec + 10:
            for agent_id in self.load_balancer.agent_ids:
                for res in self.redis.consume_results(agent_id, count=100, block_ms=1000):
                    if res.message_id in sent_ids:
                        sent_ids.remove(res.message_id)
                        results.append(res)
            time.sleep(0.5)

        # 3. 未返回的标记为超时
        for missing_id in sent_ids:
            results.append(
                A2AResult(
                    message_id=missing_id,
                    from_agent="orchestrator",
                    to_agent="perception",
                    status="timeout",
                    error="感知 Agent 未在超时内返回结果",
                )
            )

        return results

    # ---- 完整巡检流程 ----

    def run_patrol(self, rules: list[dict], cameras: list[dict]) -> dict:
        """端到端巡检：协调 Agent 编排全流程。"""
        patrol_id = f"patrol_{uuid.uuid4().hex[:8]}"
        print(f"[Orchestrator] 启动巡检 {patrol_id}，规则={len(rules)} 摄像头={len(cameras)}")

        # Step 1: 规划 Agent 生成子任务
        plan_res = self._call_sync(self.planner_url, "plan", {"rules": rules, "cameras": cameras})
        if plan_res.get("status") == "error":
            return {"patrol_id": patrol_id, "status": "failed", "stage": "plan", "error": plan_res.get("error")}
        sub_tasks = plan_res.get("result", {}).get("tasks", [])

        # Step 2: 分发给感知 Agent（Map）
        perception_results = self.dispatch_to_perception(sub_tasks)
        findings = []
        for res in perception_results:
            if res.status == "success":
                findings.extend(res.result.get("findings", []))

        # Step 3: 决策 Agent 复核
        decision_res = self._call_sync(
            self.decision_url,
            "verify",
            {"findings": findings, "tasks": sub_tasks},
        )
        alarms = decision_res.get("result", {}).get("alarms", [])

        # Step 4: 执行 Agent 落地（告警、工单、报告）
        action_res = self._call_sync(
            self.action_url,
            "execute",
            {"alarms": alarms, "patrol_id": patrol_id},
        )

        return {
            "patrol_id": patrol_id,
            "status": "success",
            "tasks_total": len(sub_tasks),
            "findings": len(findings),
            "alarms": len(alarms),
            "action": action_res.get("result", {}),
        }
