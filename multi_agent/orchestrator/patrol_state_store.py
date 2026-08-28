"""协调器巡检状态持久化（Checkpoint）。

解决「多 Agent 版协调器状态只存在内存局部变量、崩溃即丢」的短板：
- 把 run_patrol 的上下文按阶段落 Redis，使崩溃后可检查、可审计、可续跑；
- 三段式：PatrolStateStore 接口 + RedisPatrolStateStore（真实）+ MemoryPatrolStateStore（测试）；
- 失败安全：所有 Redis 操作 try/except 兜底，持久化挂了不影响主巡检流程。

设计要点（对应面试「协调器状态持久化 / 单一事实源」）：
- key 约定：`guardeye:patrol:{patrol_id}`（JSON 序列化巡检上下文），
  另有 `guardeye:patrols`（SET）记录所有 patrol_id 供 list_active；
- 阶段常量：created → planned → dispatched → verified → hitl → executed → done（failed 表示失败）；
- 诚实边界：resume 重跑会重发感知任务（非 exactly-once），仅适合 Demo；完整幂等续跑是 TODO。
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


# ---- 阶段常量 ----
STAGE_CREATED = "created"        # 已创建，尚未规划
STAGE_PLANNED = "planned"        # 规划 Agent 已产出子任务
STAGE_DISPATCHED = "dispatched"  # 感知 Agent 已返回 findings
STAGE_VERIFIED = "verified"      # 决策 Agent 已产出 alarms
STAGE_HITL = "hitl"              # 有待复核告警，等待人工提交决策
STAGE_EXECUTED = "executed"      # 执行 Agent 已落地
STAGE_DONE = "done"              # 全流程完成
STAGE_FAILED = "failed"          # 失败（stage 字段记录失败在哪个阶段）


@dataclass
class PatrolCheckpoint:
    """一次巡检的可持久化上下文快照。"""

    patrol_id: str
    status: str = STAGE_CREATED
    stage: str = STAGE_CREATED
    rules: list[dict] = field(default_factory=list)
    cameras: list[dict] = field(default_factory=list)
    sub_tasks: list[dict] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)
    alarms: list[dict] = field(default_factory=list)
    pending_review: list[dict] = field(default_factory=list)
    hitl_decisions: list[dict] | None = None
    action_result: dict | None = None
    error: str | None = None
    # Phase 1：结构化异常日志（为 Phase 2 LLM 决策积累上下文）
    exception_log: list[dict] = field(default_factory=list)
    # Phase 3：安全关键异常的人工复核（ESCALATE → HITL）
    exception_review: list[dict] = field(default_factory=list)
    exception_hitl_decisions: list[dict] | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PatrolCheckpoint":
        return cls(
            patrol_id=data["patrol_id"],
            status=data.get("status", STAGE_CREATED),
            stage=data.get("stage", STAGE_CREATED),
            rules=data.get("rules", []),
            cameras=data.get("cameras", []),
            sub_tasks=data.get("sub_tasks", []),
            findings=data.get("findings", []),
            alarms=data.get("alarms", []),
            pending_review=data.get("pending_review", []),
            hitl_decisions=data.get("hitl_decisions"),
            action_result=data.get("action_result"),
            error=data.get("error"),
            exception_log=data.get("exception_log", []),
            exception_review=data.get("exception_review", []),
            exception_hitl_decisions=data.get("exception_hitl_decisions"),
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
        )


class PatrolStateStore(ABC):
    """巡检状态持久化接口。"""

    @abstractmethod
    def save(self, checkpoint: PatrolCheckpoint) -> bool:
        """保存（覆盖写）一个巡检快照，成功返回 True。"""

    @abstractmethod
    def load(self, patrol_id: str) -> PatrolCheckpoint | None:
        """读取巡检快照，不存在返回 None。"""

    @abstractmethod
    def delete(self, patrol_id: str) -> bool:
        """删除巡检快照。"""

    @abstractmethod
    def list_active(self) -> list[str]:
        """返回所有已记录（未删除）的 patrol_id 列表。"""


class RedisPatrolStateStore(PatrolStateStore):
    """真实实现：巡检上下文 JSON 序列化后写入 Redis。

    key 约定：
    - `guardeye:patrol:{patrol_id}` → JSON 字符串
    - `guardeye:patrols` → SET，记录所有 patrol_id（供 list_active）
    """

    PREFIX = "guardeye:patrol:"
    INDEX = "guardeye:patrols"

    def __init__(self, redis_client=None):
        if redis_client is None:
            import redis

            redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.r = redis_client

    def save(self, checkpoint: PatrolCheckpoint) -> bool:
        try:
            checkpoint.updated_at = time.time()
            self.r.set(
                self.PREFIX + checkpoint.patrol_id,
                json.dumps(checkpoint.to_dict(), ensure_ascii=False),
            )
            self.r.sadd(self.INDEX, checkpoint.patrol_id)
            return True
        except Exception:
            return False

    def load(self, patrol_id: str) -> PatrolCheckpoint | None:
        try:
            raw = self.r.get(self.PREFIX + patrol_id)
            if raw is None:
                return None
            return PatrolCheckpoint.from_dict(json.loads(raw))
        except Exception:
            return None

    def delete(self, patrol_id: str) -> bool:
        try:
            self.r.delete(self.PREFIX + patrol_id)
            self.r.srem(self.INDEX, patrol_id)
            return True
        except Exception:
            return False

    def list_active(self) -> list[str]:
        try:
            return sorted(self.r.smembers(self.INDEX))
        except Exception:
            return []


class MemoryPatrolStateStore(PatrolStateStore):
    """测试用：内存字典实现，无外部依赖。"""

    def __init__(self):
        self._data: dict[str, dict] = {}

    def save(self, checkpoint: PatrolCheckpoint) -> bool:
        checkpoint.updated_at = time.time()
        self._data[checkpoint.patrol_id] = checkpoint.to_dict()
        return True

    def load(self, patrol_id: str) -> PatrolCheckpoint | None:
        data = self._data.get(patrol_id)
        return PatrolCheckpoint.from_dict(data) if data else None

    def delete(self, patrol_id: str) -> bool:
        return self._data.pop(patrol_id, None) is not None

    def list_active(self) -> list[str]:
        return sorted(self._data.keys())
