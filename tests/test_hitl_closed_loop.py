"""多 Agent 版 HITL 闭环单测。

验证「决策分流 → 协调器停等 → 人工回填 → 执行落地」这条链路在协调器侧是否真正闭合：
- 决策 Agent 返回 pending_review 时，run_patrol 停在 waiting_hitl（checkpoint 落 hitl 阶段）
- submit_hitl_decisions 提交决策后继续执行，checkpoint 推进到 done
- 无 pending_review 时跳过 HITL，直接 done
- execute 收到协调器传来的 pending_review + hitl_decisions（不再丢字段）
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "multi_agent"),
)

from multi_agent.orchestrator.patrol_state_store import (
    STAGE_DONE,
    STAGE_EXECUTED,
    STAGE_HITL,
    MemoryPatrolStateStore,
    PatrolCheckpoint,
)
from multi_agent.orchestrator.workflow import Orchestrator


def make_orchestrator():
    store = MemoryPatrolStateStore()
    orch = Orchestrator(redis_client=object(), patrol_store=store)
    return orch, store


def _plan_verify_fakes(verify_result, execute_capture):
    """构造 plan→verify→execute 的假 HTTP 调用；execute 的参数会被写入 execute_capture。"""

    def fake_call_sync(url, task, payload):
        if task == "plan":
            return {"status": "success", "result": {"tasks": [{"id": "t1", "camera_id": "c1", "rule": {}}]}}
        if task == "verify":
            return {"status": "success", "result": verify_result}
        if task == "execute":
            execute_capture["payload"] = payload
            return {"status": "success", "result": {"report": "ok"}}
        return {"status": "error", "error": "unknown"}

    return fake_call_sync


def _fake_dispatch(tasks):
    return [SimpleNamespace(status="success", result={"findings": [{"id": "f1", "hit": True}]})]


def test_run_patrol_stops_at_hitl(monkeypatch):
    orch, store = make_orchestrator()
    execute_capture = {}
    verify_result = {
        "alarms": [
            {"id": "a1", "severity": "low"},
            {"id": "a2", "severity": "low"},
        ],
        "pending_review": [{"id": "a1", "severity": "low"}],
    }

    monkeypatch.setattr(orch, "_call_sync", _plan_verify_fakes(verify_result, execute_capture))
    monkeypatch.setattr(orch, "dispatch_to_perception", _fake_dispatch)

    result = orch.run_patrol([{"id": "r1"}], [{"id": "cam1"}])
    assert result["status"] == "waiting_hitl"
    assert result["stage"] == STAGE_HITL
    assert result["pending_review"] == [{"id": "a1", "severity": "low"}]
    assert execute_capture == {}  # 尚未执行

    cp = store.load(result["patrol_id"])
    assert cp.stage == STAGE_HITL
    assert cp.pending_review == [{"id": "a1", "severity": "low"}]
    assert cp.hitl_decisions is None


def test_submit_hitl_decisions_continues_to_done(monkeypatch):
    orch, store = make_orchestrator()
    cp = PatrolCheckpoint(
        patrol_id="patrol_x",
        stage=STAGE_HITL,
        alarms=[
            {"id": "a1", "severity": "low"},
            {"id": "a2", "severity": "low"},
        ],
        pending_review=[{"id": "a1", "severity": "low"}],
    )
    store.save(cp)

    execute_capture = {}

    def fake_call_sync(url, task, payload):
        if task == "execute":
            execute_capture["payload"] = payload
            return {"status": "success", "result": {"report": "ok"}}
        return {"status": "error", "error": "unknown"}

    monkeypatch.setattr(orch, "_call_sync", fake_call_sync)

    decisions = [{"alarm_id": "a1", "decision": "confirm"}]
    result = orch.submit_hitl_decisions("patrol_x", decisions)
    assert result["status"] == "success"

    # 关键断言：pending_review 与 hitl_decisions 都被传给执行 Agent
    assert execute_capture["payload"]["alarms"] == cp.alarms
    assert execute_capture["payload"]["pending_review"] == [{"id": "a1", "severity": "low"}]
    assert execute_capture["payload"]["hitl_decisions"] == decisions

    reloaded = store.load("patrol_x")
    assert reloaded.stage == STAGE_EXECUTED
    assert reloaded.status == STAGE_DONE


def test_submit_hitl_decisions_empty_list_still_proceeds(monkeypatch):
    """空决策列表（人工已复核、无需处置）≠ 未提交，仍应推进到执行。"""
    orch, store = make_orchestrator()
    cp = PatrolCheckpoint(
        patrol_id="patrol_x",
        stage=STAGE_HITL,
        alarms=[{"id": "a1", "severity": "low"}],
        pending_review=[{"id": "a1", "severity": "low"}],
    )
    store.save(cp)

    def fake_call_sync(url, task, payload):
        if task == "execute":
            return {"status": "success", "result": {"report": "ok"}}
        return {"status": "error", "error": "unknown"}

    monkeypatch.setattr(orch, "_call_sync", fake_call_sync)

    result = orch.submit_hitl_decisions("patrol_x", [])
    assert result["status"] == "success"
    assert store.load("patrol_x").status == STAGE_DONE


def test_no_pending_review_skips_hitl(monkeypatch):
    orch, store = make_orchestrator()
    execute_capture = {}
    verify_result = {"alarms": [{"id": "a1", "severity": "high"}], "pending_review": []}

    monkeypatch.setattr(orch, "_call_sync", _plan_verify_fakes(verify_result, execute_capture))
    monkeypatch.setattr(orch, "dispatch_to_perception", _fake_dispatch)

    result = orch.run_patrol([{"id": "r1"}], [{"id": "cam1"}])
    assert result["status"] == "success"
    assert result["alarms"] == 1
    # 直接走 execute，未停在 hitl
    assert execute_capture["payload"]["pending_review"] == []


def test_submit_hitl_decisions_wrong_stage():
    orch, store = make_orchestrator()
    cp = PatrolCheckpoint(patrol_id="patrol_y", stage=STAGE_EXECUTED)
    store.save(cp)
    result = orch.submit_hitl_decisions("patrol_y", [{"alarm_id": "a1", "decision": "confirm"}])
    assert result["status"] == "error"
    assert "非 hitl" in result["error"]


def test_submit_hitl_decisions_missing_checkpoint():
    orch, store = make_orchestrator()
    result = orch.submit_hitl_decisions("patrol_nonexistent", [])
    assert result["status"] == "error"
    assert "不存在" in result["error"]
