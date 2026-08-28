"""Phase 3 异常 HITL 闭环单测。

验证安全关键异常（ESCALATE）触发人工复核的完整链路：
- 异常处理引擎决策为 ESCALATE → 协调 Agent 触发 HITL
- 人工提交 abort → 巡检标记失败
- 人工提交 retry → 从异常发生阶段重试
- 人工提交 skip → 跳过当前阶段继续执行
- PatrolCheckpoint 新字段序列化

设计原则：
- 通过 monkeypatch _exc_handler.handle 强制返回 ESCALATE，隔离测试目标（HITL 闭环）与异常策略细节；
- 纯内存 store，零外部依赖。
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
    STAGE_CREATED,
    STAGE_DISPATCHED,
    STAGE_DONE,
    STAGE_EXECUTED,
    STAGE_FAILED,
    STAGE_HITL,
    STAGE_PLANNED,
    STAGE_VERIFIED,
    MemoryPatrolStateStore,
    PatrolCheckpoint,
)
from multi_agent.orchestrator.workflow import Orchestrator
from multi_agent.orchestrator.exception_handler import (
    AgentExceptionEvent,
    ExceptionDecision,
    L0L1ExceptionHandler,
    make_exception_event,
)


def make_orchestrator():
    store = MemoryPatrolStateStore()
    orch = Orchestrator(redis_client=object(), patrol_store=store)
    return orch, store


class TestExceptionEscalateToHitl:
    def test_escalate_triggers_hitl_at_plan_stage(self, monkeypatch):
        """plan 阶段异常处理返回 ESCALATE → 触发 HITL 等待人工复核。"""
        orch, store = make_orchestrator()

        def fake_call_sync_with_retry(cp, url, task, payload, source_agent="", max_retries=2):
            if task == "plan":
                return {"status": "error", "error": "mock plan error"}
            return {"status": "error", "error": "unknown"}

        monkeypatch.setattr(orch, "_call_sync_with_retry", fake_call_sync_with_retry)

        # 强制异常处理返回 ESCALATE
        original_handle = orch._exc_handler.handle

        def force_escalate(event):
            event = original_handle(event)
            event.final_decision = ExceptionDecision.ESCALATE
            event.final_reason = "mock escalate for test"
            return event

        monkeypatch.setattr(orch._exc_handler, "handle", force_escalate)

        result = orch.run_patrol([{"id": "r1"}], [{"id": "cam1"}])
        assert result["status"] == "waiting_hitl"
        assert result["stage"] == STAGE_HITL
        assert "exception_review" in result
        assert len(result["exception_review"]) == 1
        assert result["exception_review"][0]["stage"] == STAGE_CREATED

        cp = store.load(result["patrol_id"])
        assert cp.stage == STAGE_HITL
        assert len(cp.exception_review) == 1
        assert cp.exception_hitl_decisions is None

    def test_hitl_abort_confirms_failure(self, monkeypatch):
        """人工提交 abort → 巡检标记为 FAILED。"""
        orch, store = make_orchestrator()

        def fake_call_sync_with_retry(cp, url, task, payload, source_agent="", max_retries=2):
            if task == "plan":
                return {"status": "error", "error": "mock plan error"}
            return {"status": "error", "error": "unknown"}

        monkeypatch.setattr(orch, "_call_sync_with_retry", fake_call_sync_with_retry)
        monkeypatch.setattr(
            orch._exc_handler,
            "handle",
            lambda e: _force_decision(e, ExceptionDecision.ESCALATE, "mock escalate"),
        )

        result = orch.run_patrol([{"id": "r1"}], [{"id": "cam1"}])
        patrol_id = result["patrol_id"]

        decisions = [
            {"exception_event_id": result["exception_review"][0]["event_id"], "decision": "abort", "reason": "人工确认中断"}
        ]
        result2 = orch.submit_hitl_decisions(patrol_id, decisions)
        assert result2["status"] == "failed"
        assert result2["stage"] == "exception_hitl"

        cp = store.load(patrol_id)
        assert cp.status == STAGE_FAILED
        assert cp.exception_review == []
        assert cp.exception_hitl_decisions is None

    def test_hitl_retry_resumes_from_same_stage(self, monkeypatch):
        """人工提交 retry → 从异常发生阶段（STAGE_CREATED）重新执行 plan。"""
        orch, store = make_orchestrator()
        attempt = [0]

        def fake_call_sync_with_retry(cp, url, task, payload, source_agent="", max_retries=2):
            if task == "plan":
                attempt[0] += 1
                if attempt[0] == 1:
                    return {"status": "error", "error": "mock plan error"}
                return {"status": "success", "result": {"tasks": [{"id": "t1"}]}}
            if task == "verify":
                return {"status": "success", "result": {"alarms": [], "pending_review": []}}
            if task == "execute":
                return {"status": "success", "result": {"report": "ok"}}
            return {"status": "error", "error": "unknown"}

        monkeypatch.setattr(orch, "_call_sync_with_retry", fake_call_sync_with_retry)
        monkeypatch.setattr(
            orch._exc_handler,
            "handle",
            lambda e: _force_decision(e, ExceptionDecision.ESCALATE, "mock escalate"),
        )
        monkeypatch.setattr(
            orch, "dispatch_to_perception", lambda tasks: []
        )

        result = orch.run_patrol([{"id": "r1"}], [{"id": "cam1"}])
        patrol_id = result["patrol_id"]
        assert result["status"] == "waiting_hitl"

        decisions = [
            {
                "exception_event_id": result["exception_review"][0]["event_id"],
                "decision": "retry",
                "stage": STAGE_CREATED,
            }
        ]
        result2 = orch.submit_hitl_decisions(patrol_id, decisions)
        assert result2["status"] == "success"
        assert attempt[0] == 2  # 重试了一次 plan

        cp = store.load(patrol_id)
        assert cp.status == STAGE_DONE
        assert cp.exception_review == []
        assert cp.exception_hitl_decisions is None

    def test_hitl_skip_advances_to_next_stage(self, monkeypatch):
        """人工提交 skip → 跳过 plan 阶段，使用空任务继续执行后续流程。"""
        orch, store = make_orchestrator()

        def fake_call_sync_with_retry(cp, url, task, payload, source_agent="", max_retries=2):
            if task == "verify":
                return {"status": "success", "result": {"alarms": [], "pending_review": []}}
            if task == "execute":
                return {"status": "success", "result": {"report": "ok"}}
            return {"status": "error", "error": "unknown"}

        monkeypatch.setattr(orch, "_call_sync_with_retry", fake_call_sync_with_retry)
        monkeypatch.setattr(
            orch._exc_handler,
            "handle",
            lambda e: _force_decision(e, ExceptionDecision.ESCALATE, "mock escalate"),
        )
        monkeypatch.setattr(
            orch, "dispatch_to_perception", lambda tasks: []
        )

        result = orch.run_patrol([{"id": "r1"}], [{"id": "cam1"}])
        patrol_id = result["patrol_id"]
        assert result["status"] == "waiting_hitl"

        decisions = [
            {
                "exception_event_id": result["exception_review"][0]["event_id"],
                "decision": "skip",
                "stage": STAGE_CREATED,
            }
        ]
        result2 = orch.submit_hitl_decisions(patrol_id, decisions)
        assert result2["status"] == "success"

        cp = store.load(patrol_id)
        assert cp.status == STAGE_DONE
        assert cp.sub_tasks == []  # plan 被跳过，空任务
        assert cp.stage == STAGE_EXECUTED

    def test_checkpoint_serialization_with_exception_review(self):
        """PatrolCheckpoint 新增字段序列化往返。"""
        cp = PatrolCheckpoint(
            patrol_id="patrol_x",
            exception_review=[
                {"event_id": "exc_01", "stage": STAGE_PLANNED, "source_agent": "perception"}
            ],
            exception_hitl_decisions=[{"exception_event_id": "exc_01", "decision": "abort"}],
        )
        store = MemoryPatrolStateStore()
        assert store.save(cp) is True
        loaded = store.load("patrol_x")
        assert loaded is not None
        assert len(loaded.exception_review) == 1
        assert loaded.exception_review[0]["event_id"] == "exc_01"
        assert loaded.exception_hitl_decisions is not None
        assert loaded.exception_hitl_decisions[0]["decision"] == "abort"


def _force_decision(event: AgentExceptionEvent, decision: ExceptionDecision, reason: str) -> AgentExceptionEvent:
    """测试辅助：强制设置异常事件的最终决策。"""
    event.final_decision = decision
    event.final_reason = reason
    event.action_params = {}
    return event
