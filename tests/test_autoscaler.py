"""自动扩缩容单测（路径 A：自研 Autoscaler）。

用 MockAgentSpawner + 假 backlog 驱动 Autoscaler，验证：
- 积压高水位 → 扩容（起实例 + LoadBalancer.add）
- 积压低水位 → 缩容（摘除 + LoadBalancer.mark_unavailable）
- 中间区间 → hold
- min / max 边界不越界
- 冷却窗口防抖
"""

from __future__ import annotations

import os
import sys

# multi_agent 内部以 `a2a_protocol` 作为顶层包导入（见 orchestrator/workflow.py），
# 需把 multi_agent/ 加入 sys.path 才能解析；项目根目录已由 conftest / pyproject 加入。
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "multi_agent"),
)

from agents.config import Settings
from multi_agent.orchestrator.autoscaler import Autoscaler, MockAgentSpawner
from multi_agent.orchestrator.workflow import LoadBalancer


class FakeRedisBacklog:
    """只提供 autoscaler 需要的 task_backlog()，避免真实 Redis 依赖。"""

    def __init__(self, backlog: int = 0):
        self._backlog = backlog

    def set(self, n: int):
        self._backlog = n

    def task_backlog(self) -> int:
        return self._backlog


def make_autoscaler(backlog=0, agents=None, **settings_overrides):
    defaults = dict(
        autoscaler_min_instances=1,
        autoscaler_max_instances=4,
        autoscaler_scale_up_backlog=4,
        autoscaler_scale_down_backlog=0,
        autoscaler_poll_interval_sec=5.0,
        autoscaler_cooldown_sec=0.0,  # 单测关闭冷却，避免时间依赖
    )
    defaults.update(settings_overrides)
    settings = Settings(**defaults)
    lb = LoadBalancer(agents or ["perception-1"])
    spawner = MockAgentSpawner()
    # 初始实例已在 spawner 中标记存活（模拟已启动）
    for aid in lb.agent_ids:
        spawner.spawn(aid)
    redis = FakeRedisBacklog(backlog)
    return Autoscaler(redis, lb, spawner, settings), spawner, redis


def test_scale_up_when_backlog_high():
    autoscaler, spawner, redis = make_autoscaler(backlog=10)
    assert autoscaler.tick() == "scale_up"
    assert "perception-2" in autoscaler.lb.agent_ids
    assert spawner.running() == ["perception-1", "perception-2"]


def test_scale_down_when_idle():
    autoscaler, spawner, redis = make_autoscaler(backlog=0, agents=["perception-1", "perception-2"])
    assert autoscaler.tick() == "scale_down"
    # 缩容退掉编号最大（最新）的空闲实例
    assert autoscaler.lb.agent_ids == ["perception-1"]
    assert spawner.running() == ["perception-1"]


def test_hold_within_band():
    autoscaler, _, _ = make_autoscaler(backlog=2)  # 0 < 2 < 4
    assert autoscaler.tick() == "hold"
    assert autoscaler.lb.agent_ids == ["perception-1"]


def test_never_below_min_instances():
    autoscaler, _, _ = make_autoscaler(backlog=0, agents=["perception-1"])
    assert autoscaler.tick() == "hold"  # 已是 min，不缩


def test_never_above_max_instances():
    agents = ["perception-1", "perception-2", "perception-3", "perception-4"]
    autoscaler, _, _ = make_autoscaler(backlog=10, agents=agents)
    assert autoscaler.tick() == "hold"  # 已是 max，不扩


def test_scale_up_respects_cooldown():
    autoscaler, _, _ = make_autoscaler(backlog=10, autoscaler_cooldown_sec=3600.0)
    assert autoscaler.tick() == "scale_up"
    assert autoscaler.tick() == "hold"  # 冷却期内不再扩


def test_next_agent_id_increments_max():
    autoscaler, _, _ = make_autoscaler(backlog=0, agents=["perception-1", "perception-5"])
    assert autoscaler._next_agent_id() == "perception-6"
