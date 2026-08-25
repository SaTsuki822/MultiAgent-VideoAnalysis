"""感知 Agent 自动扩缩容器（自研轻量版，路径 A）。

职责：
1. 定期采样任务流积压量（待处理任务数）；
2. 积压 >= 高水位 → 扩容（起新感知 Agent 进程 + LoadBalancer.add）；
3. 积压 <= 低水位 → 缩容（优雅摘除 + LoadBalancer.mark_unavailable）；
4. 冷却窗口防抖，避免负载抖动导致频繁扩缩。

设计要点（对应面试「自动扩缩容」）：
- 指标：待处理任务数（`RedisStreamClient.task_backlog`，语义 = 已发送未完成）；
- 边界：min_instances / max_instances 硬约束，max 受 GPU 供给约束（此处不感知 GPU，用上限兜底）；
- 三段式：AgentSpawner 接口（真实 subprocess / mock），Autoscaler 只依赖该接口，可单测；
- 与 K8s HPA / KEDA 的关系：本模块是「无 K8s 的自研实现」，生产可替换为 KEDA Redis Stream Scaler，
  对外接口（积压 → 决策 → 增删实例）保持兼容。

诚实边界：
- 缩容是「摘除路由 + 终止进程」，不是完整优雅下线（未等待 in-flight 任务排空）；
  因缩容仅在积压低水位触发，in-flight 极少，残留风险可控；完整排空是 TODO。
- GPU 容量、Orchestrator 单点瓶颈不在本模块处理范围内。
"""

from __future__ import annotations

import re
import threading
import time
from abc import ABC, abstractmethod


class AgentSpawner(ABC):
    """感知 Agent 实例生命周期管理接口（三段式：真实 subprocess / mock）。"""

    @abstractmethod
    def spawn(self, agent_id: str) -> bool:
        """启动一个感知 Agent 实例，成功返回 True。"""

    @abstractmethod
    def terminate(self, agent_id: str) -> bool:
        """停止一个感知 Agent 实例，成功返回 True。"""

    @abstractmethod
    def running(self) -> list[str]:
        """返回当前仍存活的 agent_id 列表。"""


class SubprocessAgentSpawner(AgentSpawner):
    """真实实现：subprocess 起 / 停感知 Agent 进程。

    感知 Agent 以 `python -m multi_agent.perception_agent.workflow <agent_id>` 启动，
    agent_id 决定其 Redis Stream 消费者名（见 perception_agent/workflow.py 的 main()）。
    """

    def __init__(self, module: str = "multi_agent.perception_agent.workflow"):
        self.module = module
        self._procs: dict[str, "subprocess.Popen"] = {}

    def spawn(self, agent_id: str) -> bool:
        import subprocess
        import sys

        existing = self._procs.get(agent_id)
        if existing is not None and existing.poll() is None:
            return False  # 已存在且存活
        proc = subprocess.Popen([sys.executable, "-m", self.module, agent_id])
        self._procs[agent_id] = proc
        return True

    def terminate(self, agent_id: str) -> bool:
        proc = self._procs.pop(agent_id, None)
        if proc is None or proc.poll() is not None:
            return False  # 未管理或已退出
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        return True

    def running(self) -> list[str]:
        return [aid for aid, p in self._procs.items() if p.poll() is None]


class MockAgentSpawner(AgentSpawner):
    """测试用：内存中维护实例集合，无真实子进程。"""

    def __init__(self):
        self.alive: set[str] = set()
        self.spawned: list[str] = []
        self.terminated: list[str] = []

    def spawn(self, agent_id: str) -> bool:
        if agent_id in self.alive:
            return False
        self.alive.add(agent_id)
        self.spawned.append(agent_id)
        return True

    def terminate(self, agent_id: str) -> bool:
        if agent_id not in self.alive:
            return False
        self.alive.discard(agent_id)
        self.terminated.append(agent_id)
        return True

    def running(self) -> list[str]:
        return sorted(self.alive)


class Autoscaler:
    """根据任务积压量扩缩感知 Agent 实例。

    依赖注入：redis（提供 task_backlog）、load_balancer（add/mark_unavailable）、
    spawner（起/停进程）、settings（阈值与边界），全部可 mock，便于单测。
    """

    def __init__(
        self,
        redis,          # RedisStreamClient 或任意提供 task_backlog() 的对象
        load_balancer,  # LoadBalancer（含 add / mark_unavailable / agent_ids / agent_loads）
        spawner: AgentSpawner,
        settings,       # Settings（含 autoscaler_* 配置）
    ):
        self.redis = redis
        self.lb = load_balancer
        self.spawner = spawner
        self.settings = settings
        self._last_scale_time = 0.0  # 上次扩缩容时间戳（冷却窗口用）
        self._stop_event = threading.Event()

    # ---- 指标 ----

    def backlog(self) -> int:
        """待处理任务数。"""
        return int(self.redis.task_backlog() or 0)

    def current_instances(self) -> int:
        """当前 LoadBalancer 中登记的感知 Agent 实例数。"""
        return len(self.lb.agent_ids)

    # ---- 决策 ----

    def decide(self) -> str:
        """根据积压量与实例数返回决策：'scale_up' | 'scale_down' | 'hold'。"""
        backlog = self.backlog()
        n = self.current_instances()
        s = self.settings

        if backlog >= s.autoscaler_scale_up_backlog and n < s.autoscaler_max_instances:
            return "scale_up"
        if backlog <= s.autoscaler_scale_down_backlog and n > s.autoscaler_min_instances:
            return "scale_down"
        return "hold"

    # ---- 执行 ----

    def _numeric_suffix(self, agent_id: str) -> int:
        """取 agent_id 末尾数字（如 perception-2 → 2，无数字则 0）。"""
        m = re.search(r"(\d+)$", agent_id)
        return int(m.group(1)) if m else 0

    def _next_agent_id(self) -> str:
        """从现有 agent_id 中取最大数字后缀 +1 生成新 id（如 perception-2 → perception-3）。"""
        max_n = max((self._numeric_suffix(aid) for aid in self.lb.agent_ids), default=0)
        return f"perception-{max_n + 1}"

    def scale_up(self) -> bool:
        """起一个新实例并注册进负载均衡。"""
        new_id = self._next_agent_id()
        if not self.spawner.spawn(new_id):
            return False
        self.lb.add(new_id)
        return True

    def scale_down(self) -> bool:
        """摘除「负载最小、且编号最大（最新）」的实例并停止其进程。"""
        if self.current_instances() <= self.settings.autoscaler_min_instances:
            return False
        # 优先退掉空闲且最后加入的实例：负载升序，同负载取编号最大者
        victim = min(
            self.lb.agent_ids,
            key=lambda aid: (self.lb.agent_loads.get(aid, 0.0), -self._numeric_suffix(aid)),
        )
        if not self.spawner.terminate(victim):
            return False
        self.lb.mark_unavailable(victim)
        return True

    # ---- 循环 ----

    def tick(self) -> str:
        """单次采样 + 决策 + 执行。返回实际执行的决策（冷却中则返回 'hold'）。

        拆成独立方法便于单测：直接调 tick() 就能驱动一轮扩缩容，无需起线程。
        """
        if self._in_cooldown():
            return "hold"

        decision = self.decide()
        if decision == "scale_up":
            self.scale_up()
        elif decision == "scale_down":
            self.scale_down()
        else:
            return "hold"

        self._last_scale_time = time.time()
        return decision

    def _in_cooldown(self) -> bool:
        cooldown = self.settings.autoscaler_cooldown_sec
        if cooldown <= 0:
            return False
        return time.time() - self._last_scale_time < cooldown

    def run(self, stop_event: threading.Event | None = None):
        """监控循环：定期 tick。可传 stop_event 或调 self.stop() 结束。"""
        stop_event = stop_event or self._stop_event
        while not stop_event.is_set():
            self.tick()
            stop_event.wait(self.settings.autoscaler_poll_interval_sec)

    def stop(self):
        self._stop_event.set()
