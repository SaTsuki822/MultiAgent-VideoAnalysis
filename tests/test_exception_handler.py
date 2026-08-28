"""异常处理引擎单测 — Phase 1（L0/L1 硬编码策略层）。

覆盖：
1. ExceptionClassifier：基于关键词的 L0 分类
2. SafetyGuard：安全基线校验（high severity 拦截、基础设施错误升级等）
3. L0L1ExceptionHandler：策略表查询 + 重试耗尽升级 + 安全基线集成
4. 异常事件序列化往返

设计原则：
- 零外部依赖（不调用 LLM / Redis / HTTP）；
- 纯函数为主，状态变化明确；
- 策略表变更时应能在此单测中快速发现回归。
"""

from __future__ import annotations

import pytest

from agents.config import Settings
from agents.llm import LLMClient, MockLLMClient
from multi_agent.orchestrator.exception_handler import (
    AgentExceptionEvent,
    ExceptionClassifier,
    ExceptionDecision,
    LLMExceptionAdvisor,
    L0L1ExceptionHandler,
    SafetyGuard,
    make_exception_event,
)


# ============================================================
# ExceptionClassifier
# ============================================================
class TestExceptionClassifier:
    def test_timeout_classification(self):
        assert ExceptionClassifier.classify("Request timeout after 30s") == "timeout"
        assert ExceptionClassifier.classify("Connection timed out") == "timeout"
        assert ExceptionClassifier.classify("连接超时，请重试") == "timeout"

    def test_connection_error_classification(self):
        assert ExceptionClassifier.classify("Connection refused") == "connection_error"
        assert ExceptionClassifier.classify("Network unreachable") == "connection_error"
        assert ExceptionClassifier.classify("无法连接到服务器") == "connection_error"

    def test_agent_crash_classification(self):
        assert ExceptionClassifier.classify("Agent process killed by signal 9") == "agent_crash"
        assert ExceptionClassifier.classify("OOM: out of memory") == "agent_crash"
        assert ExceptionClassifier.classify("进程崩溃，退出码-1") == "agent_crash"

    def test_empty_result_classification(self):
        assert ExceptionClassifier.classify("Planner returned empty tasks") == "empty_result"
        assert ExceptionClassifier.classify("No findings in result") == "empty_result"

    def test_infrastructure_error_classification(self):
        assert ExceptionClassifier.classify("Redis connection lost") == "infrastructure_error"
        assert ExceptionClassifier.classify("Postgres pool exhausted") == "infrastructure_error"

    def test_unknown_fallback(self):
        assert ExceptionClassifier.classify("Something weird happened") == "unknown"
        assert ExceptionClassifier.classify("") == "unknown"

    def test_all_agents_failed_explicit(self):
        assert (
            ExceptionClassifier.classify("all agents failed to respond", source_agent="perception")
            == "all_agents_failed"
        )


# ============================================================
# SafetyGuard
# ============================================================
class TestSafetyGuard:
    @pytest.fixture
    def guard(self):
        return SafetyGuard()

    def test_high_severity_ignore_blocked(self, guard):
        event = make_exception_event(
            patrol_id="p1",
            source_agent="perception",
            error_message="timeout",
            context={"rule_severity": "high"},
        )
        passed, decision, reason = guard.validate(event, ExceptionDecision.IGNORE)
        assert passed is False
        assert decision == ExceptionDecision.RETRY
        assert "安全基线拦截" in reason
        assert "severity=high" in reason

    def test_high_severity_skip_task_blocked(self, guard):
        event = make_exception_event(
            patrol_id="p1",
            source_agent="perception",
            error_message="timeout",
            context={"rule_severity": "high"},
        )
        passed, decision, reason = guard.validate(event, ExceptionDecision.SKIP_TASK)
        assert passed is False
        assert decision == ExceptionDecision.ESCALATE
        assert "禁止 SKIP_TASK" in reason

    def test_medium_severity_ignore_allowed(self, guard):
        event = make_exception_event(
            patrol_id="p1",
            source_agent="perception",
            error_message="minor glitch",
            context={"rule_severity": "medium"},
        )
        passed, decision, reason = guard.validate(event, ExceptionDecision.IGNORE)
        assert passed is True
        assert decision == ExceptionDecision.IGNORE

    def test_infrastructure_error_retry_exhausted(self, guard):
        event = make_exception_event(
            patrol_id="p1",
            source_agent="orchestrator",
            error_message="Redis connection lost",
            context={"retry_count": 1, "rule_severity": "medium"},
        )
        event.exception_type = "infrastructure_error"
        passed, decision, reason = guard.validate(event, ExceptionDecision.RETRY)
        assert passed is False
        assert decision == ExceptionDecision.ABORT
        assert "基础设施异常重试" in reason

    def test_infrastructure_error_first_retry_allowed(self, guard):
        event = make_exception_event(
            patrol_id="p1",
            source_agent="orchestrator",
            error_message="Redis connection lost",
            context={"retry_count": 0, "rule_severity": "medium"},
        )
        event.exception_type = "infrastructure_error"
        passed, decision, reason = guard.validate(event, ExceptionDecision.RETRY)
        assert passed is True  # 首次重试，不拦截

    def test_single_agent_crash_skip_blocked(self, guard):
        event = make_exception_event(
            patrol_id="p1",
            source_agent="perception",
            error_message="agent crash",
            context={"total_perception_agents": 1, "rule_severity": "medium"},
        )
        event.exception_type = "agent_crash"
        passed, decision, reason = guard.validate(event, ExceptionDecision.SKIP_TASK)
        assert passed is False
        assert decision == ExceptionDecision.ABORT
        assert "唯一感知 Agent 崩溃" in reason

    def test_multi_agent_crash_skip_allowed(self, guard):
        event = make_exception_event(
            patrol_id="p1",
            source_agent="perception",
            error_message="agent crash",
            context={"total_perception_agents": 3, "rule_severity": "medium"},
        )
        event.exception_type = "agent_crash"
        passed, decision, reason = guard.validate(event, ExceptionDecision.SKIP_TASK)
        assert passed is True


# ============================================================
# L0L1ExceptionHandler（集成策略表 + 安全基线）
# ============================================================
class TestL0L1ExceptionHandler:
    @pytest.fixture
    def handler(self):
        return L0L1ExceptionHandler()

    def test_planner_timeout_policy(self, handler):
        event = make_exception_event(
            patrol_id="p1", source_agent="planner", error_message="Request timeout"
        )
        result = handler.handle(event)
        assert result.final_decision == ExceptionDecision.RETRY
        assert result.action_params["max_retries"] == 2
        assert result.exception_type == "timeout"

    def test_perception_timeout_policy(self, handler):
        event = make_exception_event(
            patrol_id="p1", source_agent="perception", error_message="Connection timed out"
        )
        result = handler.handle(event)
        assert result.final_decision == ExceptionDecision.RETRY
        assert result.action_params["max_retries"] == 1

    def test_empty_result_planner_skip_stage(self, handler):
        event = make_exception_event(
            patrol_id="p1",
            source_agent="planner",
            error_message="Planner returned empty tasks",
        )
        result = handler.handle(event)
        assert result.final_decision == ExceptionDecision.SKIP_STAGE

    def test_infrastructure_error_abort(self, handler):
        event = make_exception_event(
            patrol_id="p1",
            source_agent="orchestrator",
            error_message="Redis connection lost",
        )
        result = handler.handle(event)
        assert result.final_decision == ExceptionDecision.ABORT

    def test_high_severity_timeout_escalated_by_guard(self, handler):
        """策略表建议 RETRY，但 SafetyGuard 对 high severity 做额外约束的场景。"""
        event = make_exception_event(
            patrol_id="p1",
            source_agent="perception",
            error_message="timeout",
            context={"rule_severity": "high"},
        )
        result = handler.handle(event)
        # 策略表对 perception timeout 建议 RETRY，不触发安全基线
        assert result.final_decision == ExceptionDecision.RETRY

    def test_high_severity_skip_task_escalated_by_guard(self, handler):
        """策略表建议 SKIP_TASK（empty_result + perception），但 high severity 被拦截为 ESCALATE。"""
        event = make_exception_event(
            patrol_id="p1",
            source_agent="perception",
            error_message="empty result",
            context={"rule_severity": "high"},
        )
        result = handler.handle(event)
        assert result.final_decision == ExceptionDecision.ESCALATE
        assert "安全基线拦截" in result.final_reason

    def test_retry_exhausted_upgrade(self, handler):
        """感知 Agent 超时且重试次数已达上限 → 升级为 SKIP_TASK。"""
        event = make_exception_event(
            patrol_id="p1",
            source_agent="perception",
            error_message="timeout",
            context={"retry_count": 1, "rule_severity": "medium"},
        )
        result = handler.handle(event)
        # 策略表 max_retries=1，retry_count=1 已耗尽 → 升级为 SKIP_TASK
        assert result.final_decision == ExceptionDecision.SKIP_TASK

    def test_event_serialization(self, handler):
        event = make_exception_event(
            patrol_id="p1", source_agent="planner", error_message="network error"
        )
        result = handler.handle(event)
        d = result.to_dict()
        assert d["patrol_id"] == "p1"
        assert d["source_agent"] == "planner"
        assert d["exception_type"] == "connection_error"
        assert d["final_decision"] == "retry"
        assert "event_id" in d
        assert "timestamp" in d


# ============================================================
# make_exception_event 便捷函数
# ============================================================
class TestMakeExceptionEvent:
    def test_auto_generates_id_and_timestamp(self):
        event = make_exception_event(
            patrol_id="p1", source_agent="planner", error_message="error"
        )
        assert event.event_id.startswith("exc_")
        assert len(event.event_id) == 12  # exc_ + 8 hex
        assert event.timestamp is not None
        assert event.patrol_id == "p1"
        assert event.exception_type == "unknown"  # 未预分类

    def test_accepts_context(self):
        event = make_exception_event(
            patrol_id="p1",
            source_agent="perception",
            error_message="timeout",
            context={"retry_count": 2, "camera_id": "cam_01"},
        )
        assert event.context["retry_count"] == 2
        assert event.context["camera_id"] == "cam_01"


# ============================================================
# Phase 2：LLM 异常顾问
# ============================================================
class FakeLLMClient(LLMClient):
    """测试中用于模拟 LLM 返回预设 JSON 的客户端。"""

    def __init__(self, response_json: str):
        self.response_json = response_json

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        return self.response_json

    def complete_vision(self, system: str, user: str, images_b64: list[str], json_mode: bool = False) -> str:
        return self.complete(system, user, json_mode)


class TestLLMExceptionAdvisor:
    def test_mock_llm_returns_fallback(self):
        """MockLLMClient 应直接降级到 L1，不消耗 token。"""
        advisor = LLMExceptionAdvisor(llm=MockLLMClient(), settings=Settings())
        event = make_exception_event(patrol_id="p1", source_agent="planner", error_message="timeout")
        decision, reason, params, confidence = advisor.advise(event, ExceptionDecision.RETRY, "L1 reason")
        assert decision == ExceptionDecision.RETRY
        assert confidence == 0.0
        assert "mock" in reason

    def test_build_prompt_contains_key_info(self):
        advisor = LLMExceptionAdvisor(llm=MockLLMClient(), settings=Settings())
        event = make_exception_event(
            patrol_id="p1",
            source_agent="perception",
            error_message="timeout",
            stage="verify",
            context={"rules_count": 5, "retry_count": 2, "history_same_type_1h": 4},
        )
        prompt = advisor._build_prompt(event, ExceptionDecision.RETRY, "L1 reason")
        assert "p1" in prompt
        assert "verify" in prompt
        assert "perception" in prompt
        assert "timeout" in prompt
        assert "5" in prompt
        assert "2" in prompt
        assert "4" in prompt
        assert "RETRY" in prompt
        assert "L1 reason" in prompt

    def test_parse_advice_success(self):
        advisor = LLMExceptionAdvisor(llm=MockLLMClient(), settings=Settings())
        raw = '{"decision": "SKIP_TASK", "reasoning": "非关键路径", "confidence": 0.85}'
        decision, reason, params, confidence = advisor._parse_advice(raw, ExceptionDecision.RETRY)
        assert decision == ExceptionDecision.SKIP_TASK
        assert reason == "非关键路径"
        assert confidence == 0.85
        assert params == {}

    def test_parse_advice_invalid_json(self):
        advisor = LLMExceptionAdvisor(llm=MockLLMClient(), settings=Settings())
        decision, reason, params, confidence = advisor._parse_advice("not json", ExceptionDecision.RETRY)
        assert decision == ExceptionDecision.RETRY
        assert "降级" in reason
        assert confidence == 0.0

    def test_parse_advice_unknown_decision(self):
        advisor = LLMExceptionAdvisor(llm=MockLLMClient(), settings=Settings())
        raw = '{"decision": "DO_NOTHING", "confidence": 0.9}'
        decision, reason, params, confidence = advisor._parse_advice(raw, ExceptionDecision.ESCALATE)
        assert decision == ExceptionDecision.ESCALATE
        assert confidence == 0.9

    def test_parse_advice_missing_confidence(self):
        advisor = LLMExceptionAdvisor(llm=MockLLMClient(), settings=Settings())
        raw = '{"decision": "ABORT", "reasoning": "严重故障"}'
        decision, reason, params, confidence = advisor._parse_advice(raw, ExceptionDecision.RETRY)
        assert decision == ExceptionDecision.ABORT
        assert confidence == 0.0

    def test_advise_with_fake_llm_high_confidence(self):
        raw = '{"decision": "ESCALATE", "reasoning": "检测到安全风险", "confidence": 0.92}'
        advisor = LLMExceptionAdvisor(llm=FakeLLMClient(raw), settings=Settings())
        event = make_exception_event(patrol_id="p1", source_agent="perception", error_message="weird")
        decision, reason, params, confidence = advisor.advise(event, ExceptionDecision.RETRY, "L1")
        assert decision == ExceptionDecision.ESCALATE
        assert "安全风险" in reason
        assert confidence == 0.92

    def test_advise_with_llm_exception_fallback(self):
        class BadLLM(LLMClient):
            def complete(self, system, user, json_mode=False):
                raise RuntimeError("模型服务不可用")
            def complete_vision(self, system, user, images_b64, json_mode=False):
                raise RuntimeError("模型服务不可用")

        advisor = LLMExceptionAdvisor(llm=BadLLM(), settings=Settings())
        event = make_exception_event(patrol_id="p1", source_agent="planner", error_message="timeout")
        decision, reason, params, confidence = advisor.advise(event, ExceptionDecision.RETRY, "L1")
        assert decision == ExceptionDecision.RETRY
        assert "降级" in reason
        assert "模型服务不可用" in reason
        assert confidence == 0.0


class TestL0L1ExceptionHandlerWithLLM:
    @pytest.fixture
    def enabled_settings(self):
        return Settings(exception_llm_advisor_enabled=True, exception_llm_advisor_threshold=0.7)

    def test_should_use_llm_abort_recoverable(self, enabled_settings):
        advisor = LLMExceptionAdvisor(llm=MockLLMClient(), settings=enabled_settings)
        handler = L0L1ExceptionHandler(llm_advisor=advisor)
        event = make_exception_event(
            patrol_id="p1", source_agent="action", error_message="weird failure"
        )
        # 构造一个不在策略表中的异常类型，使 L1 fallback 到 ABORT
        event.exception_type = "custom_error"
        decision, _ = handler._lookup_policy(event.exception_type, event.source_agent)
        assert decision == ExceptionDecision.ABORT
        assert handler._should_use_llm(event, decision) is True

    def test_should_use_llm_abort_infrastructure_not_triggered(self, enabled_settings):
        advisor = LLMExceptionAdvisor(llm=MockLLMClient(), settings=enabled_settings)
        handler = L0L1ExceptionHandler(llm_advisor=advisor)
        event = make_exception_event(
            patrol_id="p1", source_agent="orchestrator", error_message="Redis connection lost"
        )
        event.exception_type = "infrastructure_error"
        assert handler._should_use_llm(event, ExceptionDecision.ABORT) is False

    def test_should_use_llm_semantic_mismatch(self, enabled_settings):
        advisor = LLMExceptionAdvisor(llm=MockLLMClient(), settings=enabled_settings)
        handler = L0L1ExceptionHandler(llm_advisor=advisor)
        event = make_exception_event(
            patrol_id="p1", source_agent="decision", error_message="语义不匹配"
        )
        event.exception_type = "semantic_mismatch"
        assert handler._should_use_llm(event, ExceptionDecision.RETRY) is True

    def test_should_use_llm_high_frequency(self, enabled_settings):
        advisor = LLMExceptionAdvisor(llm=MockLLMClient(), settings=enabled_settings)
        handler = L0L1ExceptionHandler(llm_advisor=advisor)
        event = make_exception_event(
            patrol_id="p1", source_agent="perception", error_message="timeout",
            context={"history_same_type_1h": 3},
        )
        assert handler._should_use_llm(event, ExceptionDecision.RETRY) is True

    def test_should_use_llm_disabled(self):
        disabled_settings = Settings(exception_llm_advisor_enabled=False)
        advisor = LLMExceptionAdvisor(llm=MockLLMClient(), settings=disabled_settings)
        handler = L0L1ExceptionHandler(llm_advisor=advisor)
        event = make_exception_event(
            patrol_id="p1", source_agent="planner", error_message="timeout"
        )
        assert handler._should_use_llm(event, ExceptionDecision.ABORT) is False

    def test_handler_adopts_high_confidence_llm(self, enabled_settings):
        # 场景：semantic_mismatch 触发 LLM，LLM 建议 RETRY（与 L1 ABORT 不同），高置信度采纳
        raw = '{"decision": "RETRY", "reasoning": "临时故障", "confidence": 0.85}'
        advisor = LLMExceptionAdvisor(llm=FakeLLMClient(raw), settings=enabled_settings)
        handler = L0L1ExceptionHandler(llm_advisor=advisor)
        event = make_exception_event(
            patrol_id="p2", source_agent="decision", error_message="semantic mismatch"
        )
        event.exception_type = "semantic_mismatch"
        result = handler.handle(event)
        assert result.final_decision == ExceptionDecision.RETRY
        assert "LLM 建议" in result.final_reason
        assert "0.85" in result.final_reason

    def test_handler_fallback_low_confidence_llm(self, enabled_settings):
        # 场景：semantic_mismatch 触发 LLM，LLM 建议 ABORT（低置信度），fallback 到 L1 ABORT
        raw = '{"decision": "ABORT", "reasoning": "不确定", "confidence": 0.3}'
        advisor = LLMExceptionAdvisor(llm=FakeLLMClient(raw), settings=enabled_settings)
        handler = L0L1ExceptionHandler(llm_advisor=advisor)
        event = make_exception_event(
            patrol_id="p1", source_agent="decision", error_message="semantic mismatch"
        )
        event.exception_type = "semantic_mismatch"
        result = handler.handle(event)
        # L1 对 semantic_mismatch 无命中策略表 -> fallback ABORT
        # LLM 建议 ABORT 但置信度低，不采纳，仍回退到 L1 ABORT
        assert result.final_decision == ExceptionDecision.ABORT
        assert "未采纳" in result.final_reason
        assert "0.30" in result.final_reason
        assert "0.7" in result.final_reason  # 阈值

    def test_handler_safety_guard_overrides_llm(self, enabled_settings):
        # LLM 建议 IGNORE，但 high severity 被 SafetyGuard 拦截为 RETRY
        raw = '{"decision": "IGNORE", "reasoning": "看起来是误报", "confidence": 0.95}'
        advisor = LLMExceptionAdvisor(llm=FakeLLMClient(raw), settings=enabled_settings)
        handler = L0L1ExceptionHandler(llm_advisor=advisor)
        event = make_exception_event(
            patrol_id="p1",
            source_agent="perception",
            error_message="semantic mismatch",
            context={"rule_severity": "high"},
        )
        event.exception_type = "semantic_mismatch"
        result = handler.handle(event)
        # SafetyGuard 拦截 high severity IGNORE -> RETRY
        assert result.final_decision == ExceptionDecision.RETRY
        assert "安全基线拦截" in result.final_reason
