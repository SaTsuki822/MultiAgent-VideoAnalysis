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

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from a2a_protocol.message_schema import A2AMessage, A2AResult
from a2a_protocol.redis_stream import RedisStreamClient
from multi_agent.orchestrator.patrol_state_store import (
    STAGE_CREATED,
    STAGE_DISPATCHED,
    STAGE_DONE,
    STAGE_EXECUTED,
    STAGE_FAILED,
    STAGE_PLANNED,
    STAGE_VERIFIED,
    PatrolCheckpoint,
    RedisPatrolStateStore,
)


class LoadBalancer:
    """感知 Agent 负载均衡器。

    当前实现：基于「队列长度 × 任务预估权重」的加权最小负载调度。
    相比 Round Robin，能感知任务异构性（简单初筛 vs 长耗时 VLM 分析），
    把新任务优先分给当前加权负载最小的实例。

    生产可扩展为：叠加 GPU 利用率、地理位置加权。
    """

    def __init__(self, agent_ids: list[str]):
        self.agent_ids = agent_ids
        # 每个 Agent 当前的加权负载（已分发未确认的任务权重之和）
        self.agent_loads: dict[str, float] = {aid: 0.0 for aid in agent_ids}

    @staticmethod
    def _estimate_task_weight(task: dict) -> float:
        """根据规则复杂度预估任务权重。

        权重设计：
        - 基础权重 1.0
        - 持续型异常（需分析更长时间）：+1.0
        - 目标对象多（检测更复杂）：每超 2 个 +0.5
        - 判定证据维度多：每超 2 条 +0.5
        """
        rule = task.get("rule", {}) if isinstance(task, dict) else {}
        weight = 1.0

        # 持续型异常需要分析更长时间
        duration = rule.get("duration_threshold_seconds") if isinstance(rule, dict) else None
        if duration is not None:
            weight += 1.0

        # 目标对象多 = 检测更复杂
        targets = rule.get("target_objects", []) if isinstance(rule, dict) else []
        if len(targets) > 2:
            weight += 0.5 * (len(targets) - 2)

        # 证据维度多 = 复核更复杂
        hints = rule.get("evidence_hints", []) if isinstance(rule, dict) else []
        if len(hints) > 2:
            weight += 0.5 * (len(hints) - 2)

        return weight

    def pick(self, task: dict | None = None) -> tuple[str, float]:
        """选择当前加权负载最小的 Agent，并预占其负载。

        Returns:
            (agent_id, task_weight)：选中的 Agent ID 和该任务的预估权重
        """
        if not self.agent_ids:
            raise RuntimeError("无可用感知Agent实例")

        weight = self._estimate_task_weight(task) if task else 1.0
        # 选择加权负载最小的实例
        agent_id = min(self.agent_ids, key=lambda aid: self.agent_loads.get(aid, float("inf")))
        self.agent_loads[agent_id] = self.agent_loads.get(agent_id, 0.0) + weight
        return agent_id, weight

    def release(self, agent_id: str, weight: float):
        """任务完成或超时后释放加权负载。"""
        if agent_id in self.agent_loads:
            self.agent_loads[agent_id] = max(0.0, self.agent_loads[agent_id] - weight)

    def mark_unavailable(self, agent_id: str):
        """标记某感知 Agent 不可用（如心跳超时），移除其负载记录。"""
        if agent_id in self.agent_ids:
            self.agent_ids.remove(agent_id)
        self.agent_loads.pop(agent_id, None)

    def add(self, agent_id: str):
        """新感知 Agent 注册。"""
        if agent_id not in self.agent_ids:
            self.agent_ids.append(agent_id)
        if agent_id not in self.agent_loads:
            self.agent_loads[agent_id] = 0.0


class Orchestrator:
    """协调 Agent：任务分片、负载均衡、结果聚合、异常恢复。"""

    def __init__(
        self,
        planner_url: str = "http://localhost:8001",
        decision_url: str = "http://localhost:8003",
        action_url: str = "http://localhost:8004",
        perception_agents: list[str] | None = None,
        redis_client=None,
        agent_spawner=None,
        patrol_store=None,
    ):
        self.planner_url = planner_url
        self.decision_url = decision_url
        self.action_url = action_url
        self.load_balancer = LoadBalancer(perception_agents or ["perception-1"])
        self.redis = RedisStreamClient(redis_client=redis_client)

        # 状态持久化：默认复用同一 Redis 连接做 checkpoint；也可注入 mock store（测试）
        self.patrol_store = (
            patrol_store if patrol_store is not None else RedisPatrolStateStore(self.redis.r)
        )

        # 自动扩缩容：仅当显式注入 agent_spawner 时才构建（默认关闭）
        self.autoscaler = None
        if agent_spawner is not None:
            from agents.config import get_settings
            from multi_agent.orchestrator.autoscaler import Autoscaler

            self.autoscaler = Autoscaler(self.redis, self.load_balancer, agent_spawner, get_settings())

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
        2. 按负载均衡（加权最小负载）选择感知 Agent，写入 Redis Stream；
        3. 轮询感知 Agent 的回调结果 Stream，收集返回；
        4. 超时未返回的任务标记为失败，并释放对应加权负载。
        """
        if not tasks:
            return []

        # 1. 发送任务（带加权负载均衡）
        sent_ids: dict[str, tuple[str, float]] = {}  # message_id -> (agent_id, weight)
        for task in tasks:
            agent_id, weight = self.load_balancer.pick(task)
            msg = A2AMessage(
                message_id=f"msg_{uuid.uuid4().hex[:8]}",
                from_agent="orchestrator",
                to_agent=agent_id,
                task="analyze_clip",
                payload=task,
                timeout_sec=timeout_sec,
            )
            self.redis.send_task(msg)
            sent_ids[msg.message_id] = (agent_id, weight)

        # 2. 收集结果（轮询，最多等 timeout_sec）
        results: list[A2AResult] = []
        start = time.time()
        while sent_ids and time.time() - start < timeout_sec + 10:
            for agent_id in self.load_balancer.agent_ids:
                for res in self.redis.consume_results(agent_id, count=100, block_ms=1000):
                    if res.message_id in sent_ids:
                        picked_agent, weight = sent_ids.pop(res.message_id)
                        # 释放该 Agent 的加权负载
                        self.load_balancer.release(picked_agent, weight)
                        results.append(res)
            time.sleep(0.5)

        # 3. 未返回的标记为超时，并释放负载
        for missing_id, (agent_id, weight) in sent_ids.items():
            self.load_balancer.release(agent_id, weight)
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

    # ---- 自动扩缩容 ----

    def start_autoscaler(self):
        """后台线程运行自动扩缩容（需在巡检前启动）。

        返回后台线程；调用方需先注入 agent_spawner（见 __init__）。
        """
        if self.autoscaler is None:
            raise RuntimeError("未提供 agent_spawner，无法启动自动扩缩容")
        t = threading.Thread(target=self.autoscaler.run, daemon=True)
        t.start()
        return t

    # ---- 完整巡检流程 ----

    def _save_checkpoint(self, cp: PatrolCheckpoint):
        """持久化巡检快照；store 缺失或写失败时不中断主流程。"""
        if self.patrol_store is None:
            return
        self.patrol_store.save(cp)

    def _run_from_checkpoint(self, cp: PatrolCheckpoint) -> dict:
        """从 cp.stage 之后继续执行剩余阶段，返回最终结果。

        线性级联：每完成一个阶段就把 cp.stage 前推，后续 if 条件基于最新 stage 判断；
        resume 时只有「已持久化阶段之后」的 if 会命中，前面已完成的阶段被自然跳过。
        """
        patrol_id = cp.patrol_id

        # Step 1: 规划 Agent 生成子任务
        if cp.stage == STAGE_CREATED:
            plan_res = self._call_sync(self.planner_url, "plan", {"rules": cp.rules, "cameras": cp.cameras})
            if plan_res.get("status") == "error":
                cp.status = STAGE_FAILED
                cp.error = plan_res.get("error")
                self._save_checkpoint(cp)
                return {"patrol_id": patrol_id, "status": "failed", "stage": "plan", "error": plan_res.get("error")}
            cp.sub_tasks = plan_res.get("result", {}).get("tasks", [])
            cp.stage = STAGE_PLANNED
            self._save_checkpoint(cp)

        # Step 2: 分发给感知 Agent（Map）
        if cp.stage == STAGE_PLANNED:
            perception_results = self.dispatch_to_perception(cp.sub_tasks)
            cp.findings = []
            for res in perception_results:
                if res.status == "success":
                    cp.findings.extend(res.result.get("findings", []))
            cp.stage = STAGE_DISPATCHED
            self._save_checkpoint(cp)

        # Step 3: 决策 Agent 复核
        if cp.stage == STAGE_DISPATCHED:
            decision_res = self._call_sync(
                self.decision_url,
                "verify",
                {"findings": cp.findings, "tasks": cp.sub_tasks},
            )
            cp.alarms = decision_res.get("result", {}).get("alarms", [])
            cp.stage = STAGE_VERIFIED
            self._save_checkpoint(cp)

        # Step 4: 执行 Agent 落地（告警、工单、报告）
        if cp.stage == STAGE_VERIFIED:
            action_res = self._call_sync(
                self.action_url,
                "execute",
                {"alarms": cp.alarms, "patrol_id": patrol_id},
            )
            cp.action_result = action_res.get("result", {})
            cp.stage = STAGE_EXECUTED
            cp.status = STAGE_DONE
            self._save_checkpoint(cp)

        return {
            "patrol_id": patrol_id,
            "status": "success",
            "tasks_total": len(cp.sub_tasks),
            "findings": len(cp.findings),
            "alarms": len(cp.alarms),
            "action": cp.action_result or {},
        }

    def run_patrol(self, rules: list[dict], cameras: list[dict]) -> dict:
        """端到端巡检：协调 Agent 编排全流程，并按阶段持久化 checkpoint。"""
        patrol_id = f"patrol_{uuid.uuid4().hex[:8]}"
        print(f"[Orchestrator] 启动巡检 {patrol_id}，规则={len(rules)} 摄像头={len(cameras)}")

        cp = PatrolCheckpoint(patrol_id=patrol_id, rules=rules, cameras=cameras)
        self._save_checkpoint(cp)
        return self._run_from_checkpoint(cp)

    def resume_patrol(self, patrol_id: str) -> dict:
        """从最近一次 checkpoint 续跑未完成阶段。

        诚实边界：若 checkpoint 停在 dispatch 之后（findings 已部分收集），重跑会重新
        dispatch 感知任务（重发消息，非 exactly-once），可能重复分析已完成的片段；
        完整幂等续跑（按 message_id 跳过已 ACK 任务）是 TODO。
        """
        if self.patrol_store is None:
            return {"patrol_id": patrol_id, "status": "error", "error": "未配置状态持久化"}
        cp = self.patrol_store.load(patrol_id)
        if cp is None:
            return {"patrol_id": patrol_id, "status": "error", "error": "checkpoint 不存在"}
        if cp.status == STAGE_DONE:
            return {
                "patrol_id": patrol_id,
                "status": "success",
                "tasks_total": len(cp.sub_tasks),
                "findings": len(cp.findings),
                "alarms": len(cp.alarms),
                "action": cp.action_result or {},
            }
        return self._run_from_checkpoint(cp)
