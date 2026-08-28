"""协调器状态持久化单测。

用 MemoryPatrolStateStore + monkeypatch 掉 _call_sync / dispatch_to_perception，
验证：
- PatrolCheckpoint 序列化往返
- store save/load/delete/list_active
- run_patrol 分阶段落 checkpoint，最终 status=done
- resume_patrol 从中途 checkpoint 续跑剩余阶段
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

# multi_agent 内部以 `a2a_protocol` 作为顶层包导入（见 orchestrator/workflow.py），
# 需把 multi_agent/ 加入 sys.path；项目根目录已由 conftest / pyproject 加入。
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "multi_agent"),
)

from multi_agent.orchestrator.patrol_state_store import (
    STAGE_DISPATCHED,
    STAGE_DONE,
    STAGE_EXECUTED,
    MemoryPatrolStateStore,
    PatrolCheckpoint,
)
from multi_agent.orchestrator.workflow import Orchestrator


def make_orchestrator():
    store = MemoryPatrolStateStore()
    # redis_client 用 dummy，避免真实 redis 导入；store 注入 mock，避免真实 Redis 写
    orch = Orchestrator(redis_client=object(), patrol_store=store)
    return orch, store


def test_checkpoint_roundtrip():
    cp = PatrolCheckpoint(
        patrol_id="patrol_x",
        rules=[{"id": "r1"}],
        sub_tasks=[{"id": "t1", "camera_id": "c1", "rule": {}}],
        findings=[{"id": "f1", "hit": True}],
    )
    store = MemoryPatrolStateStore()
    assert store.save(cp) is True
    loaded = store.load("patrol_x")
    assert loaded is not None
    assert loaded.patrol_id == "patrol_x"
    assert loaded.rules == [{"id": "r1"}]
    assert loaded.sub_tasks == [{"id": "t1", "camera_id": "c1", "rule": {}}]
    assert loaded.findings == [{"id": "f1", "hit": True}]
    assert store.list_active() == ["patrol_x"]
    assert store.delete("patrol_x") is True
    assert store.load("patrol_x") is None
    assert store.list_active() == []


def test_run_patrol_persists_stages(monkeypatch):
    orch, store = make_orchestrator()

    def fake_call_sync_with_retry(cp, url, task, payload, source_agent="", max_retries=2):
        if task == "plan":
            return {"status": "success", "result": {"tasks": [{"id": "t1", "camera_id": "c1", "rule": {}}]}}
        if task == "verify":
            return {"status": "success", "result": {"alarms": [{"id": "a1", "severity": "high"}]}}
        if task == "execute":
            return {"status": "success", "result": {"report": "ok"}}
        return {"status": "error", "error": "unknown"}

    def fake_dispatch(tasks):
        return [SimpleNamespace(status="success", result={"findings": [{"id": "f1", "hit": True}]})]

    monkeypatch.setattr(orch, "_call_sync_with_retry", fake_call_sync_with_retry)
    monkeypatch.setattr(orch, "dispatch_to_perception", fake_dispatch)

    result = orch.run_patrol([{"id": "r1"}], [{"id": "cam1"}])
    assert result["status"] == "success"
    patrol_id = result["patrol_id"]

    cp = store.load(patrol_id)
    assert cp is not None
    assert cp.status == STAGE_DONE
    assert cp.stage == STAGE_EXECUTED
    assert len(cp.sub_tasks) == 1
    assert len(cp.findings) == 1
    assert len(cp.alarms) == 1
    assert cp.action_result == {"report": "ok"}


def test_resume_patrol_from_mid_checkpoint(monkeypatch):
    orch, store = make_orchestrator()
    # 模拟崩溃在 dispatch 之后：已持久化 sub_tasks/findings，尚未 verify
    cp = PatrolCheckpoint(
        patrol_id="patrol_x",
        stage=STAGE_DISPATCHED,
        sub_tasks=[{"id": "t1", "camera_id": "c1", "rule": {}}],
        findings=[{"id": "f1", "hit": True}],
    )
    store.save(cp)

    def fake_call_sync_with_retry(cp, url, task, payload, source_agent="", max_retries=2):
        if task == "verify":
            return {"status": "success", "result": {"alarms": [{"id": "a1"}]}}
        if task == "execute":
            return {"status": "success", "result": {"report": "ok"}}
        return {"status": "error", "error": "unknown"}

    monkeypatch.setattr(orch, "_call_sync_with_retry", fake_call_sync_with_retry)

    result = orch.resume_patrol("patrol_x")
    assert result["status"] == "success"
    assert result["alarms"] == 1

    reloaded = store.load("patrol_x")
    assert reloaded.stage == STAGE_EXECUTED
    assert reloaded.status == STAGE_DONE


def test_resume_patrol_missing_checkpoint():
    orch, store = make_orchestrator()
    result = orch.resume_patrol("patrol_nonexistent")
    assert result["status"] == "error"
    assert "不存在" in result["error"]
