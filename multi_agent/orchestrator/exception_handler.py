"""协调 Agent 异常处理引擎 — Phase 1 + Phase 2（L0/L1 硬编码策略层 + LLM 辅助决策层）。

设计目标：
1. 将散落的异常处理 if/else 收拢为统一入口；
2. 异常事件结构化（模型化），为 LLM 决策积累上下文；
3. L0/L1 硬编码策略覆盖 80% 高频异常，零 token 成本、毫秒级响应；
4. Phase 2 LLM 顾问在复杂/模糊场景提供第二意见，提升决策质量；
5. 安全基线兜底：high severity 场景禁止静默忽略，LLM 建议须经 SafetyGuard 校验。

架构：
  子 Agent 异常 → ExceptionClassifier(L0 分类) → L0L1ExceptionHandler(策略决策)
    → [可选] LLMExceptionAdvisor(L2 建议) → SafetyGuard(安全基线校验)
    → 执行动作 → 记录异常日志
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from agents.config import Settings, get_settings
from agents.llm import LLMClient, MockLLMClient, get_llm_client


# ============================================================
# 决策枚举
# ============================================================
class ExceptionDecision(str, Enum):
    """异常处理决策。"""

    RETRY = "retry"              # 重试当前阶段/任务
    SKIP_TASK = "skip_task"      # 跳过该子任务，继续其他任务
    SKIP_STAGE = "skip_stage"    # 跳过整个阶段（如规划返回空任务）
    ABORT = "abort"              # 中断整个巡检，标记 failed
    ESCALATE = "escalate"        # 升级处理（提高 severity + 立即人工告警）
    IGNORE = "ignore"            # 不做处理，继续（仅低风险场景）


# ============================================================
# 异常事件模型
# ============================================================
@dataclass
class AgentExceptionEvent:
    """一次子 Agent 异常的结构化记录。

    字段设计原则：足够丰富以支撑 Phase 2 LLM 决策，但 Phase 1 中只使用部分字段。
    """

    event_id: str
    timestamp: float = field(default_factory=time.time)
    patrol_id: str = ""

    # 异常来源
    source_agent: Literal["planner", "perception", "decision", "action", "orchestrator"] = "orchestrator"
    source_agent_id: str = ""          # 具体实例标识，如 perception-3
    stage: str = ""                    # 当前巡检阶段

    # 异常分类（L0 硬编码分类器产出）
    exception_type: Literal[
        "timeout",
        "connection_error",
        "agent_crash",
        "empty_result",
        "semantic_mismatch",
        "cost_limit",
        "infrastructure_error",
        "all_agents_failed",
        "unknown",
    ] = "unknown"

    # 异常详情
    error_message: str = ""
    raw_response: str = ""             # 子 Agent 原始返回（可选，用于调试）

    # 上下文（供 LLM 决策用，Phase 1 用于安全基线判断）
    context: dict[str, Any] = field(default_factory=dict)
    # context 建议包含的键：
    #   - rules_count, tasks_count, findings_count
    #   - rule_severity: high/medium/low（当前规则 severity）
    #   - retry_count: 已重试次数
    #   - camera_id: 关联摄像头
    #   - history_same_type_1h: 近1小时同类异常次数

    # 决策结果（由 Handler 写入）
    final_decision: ExceptionDecision | None = None
    final_reason: str = ""
    action_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "patrol_id": self.patrol_id,
            "source_agent": self.source_agent,
            "source_agent_id": self.source_agent_id,
            "stage": self.stage,
            "exception_type": self.exception_type,
            "error_message": self.error_message,
            "raw_response": self.raw_response,
            "context": self.context,
            "final_decision": self.final_decision.value if self.final_decision else None,
            "final_reason": self.final_reason,
            "action_params": self.action_params,
        }


# ============================================================
# L0 异常分类器（规则硬编码）
# ============================================================
class ExceptionClassifier:
    """基于错误消息和来源的 L0 分类器。

    设计为纯函数 + 静态方法，零状态、零依赖，便于单测。
    """

    # 关键词 → 异常类型 映射表
    _TIMEOUT_PATTERNS = ("timeout", "timed out", "time out", "连接超时")
    _CONN_PATTERNS = ("connection refused", "refused", "network unreachable", "network error", "无法连接", "连接断开", "断开")
    _CRASH_PATTERNS = ("crash", "killed", "signal", "segmentation", "oom", "进程崩溃")
    _EMPTY_PATTERNS = ("empty", "no tasks", "no findings", "空任务", "无结果")
    _INFRA_PATTERNS = ("redis", "postgres", "database", "db", "qdrant", "基础设施")

    @classmethod
    def classify(cls, error_message: str, source_agent: str = "") -> str:
        """根据错误消息分类异常类型。"""
        msg = error_message.lower()

        if any(p in msg for p in cls._TIMEOUT_PATTERNS):
            return "timeout"
        if any(p in msg for p in cls._CONN_PATTERNS):
            return "connection_error"
        if any(p in msg for p in cls._CRASH_PATTERNS):
            return "agent_crash"
        if any(p in msg for p in cls._EMPTY_PATTERNS):
            return "empty_result"
        if any(p in msg for p in cls._INFRA_PATTERNS):
            return "infrastructure_error"

        # 感知 Agent 全失败（由 orchestrator 显式标记）
        if source_agent == "perception" and "all agents failed" in msg:
            return "all_agents_failed"

        return "unknown"


# ============================================================
# 安全基线校验器
# ============================================================
class SafetyGuard:
    """安全基线：覆盖 LLM/策略决策，防止安全关键流程被静默跳过。

    规则硬编码，不可配置（避免配置错误导致安全逃逸）。
    """

    @staticmethod
    def validate(event: AgentExceptionEvent, proposed: ExceptionDecision) -> tuple[bool, ExceptionDecision, str]:
        """校验 proposed 决策是否符合安全基线。

        Returns:
            (是否通过, 最终决策, 原因)
        """
        rule_severity = event.context.get("rule_severity", "medium")

        # 基线 1：high severity 场景禁止 IGNORE
        if rule_severity == "high" and proposed == ExceptionDecision.IGNORE:
            return False, ExceptionDecision.RETRY, (
                f"安全基线拦截：severity=high 的异常禁止 IGNORE，"
                f"已降级为 RETRY（event={event.event_id}）"
            )

        # 基线 2：high severity 场景禁止 SKIP_TASK（必须重试或人工介入）
        if rule_severity == "high" and proposed == ExceptionDecision.SKIP_TASK:
            return False, ExceptionDecision.ESCALATE, (
                f"安全基线拦截：severity=high 的异常禁止 SKIP_TASK，"
                f"已升级为 ESCALATE（event={event.event_id}）"
            )

        # 基线 3：infrastructure_error 且影响全局 → 禁止 RETRY（重试无意义）
        if event.exception_type == "infrastructure_error" and proposed == ExceptionDecision.RETRY:
            retry_count = event.context.get("retry_count", 0)
            if retry_count >= 1:
                return False, ExceptionDecision.ABORT, (
                    f"安全基线拦截：基础设施异常重试 {retry_count} 次仍失败，"
                    f"禁止继续 RETRY，已升级为 ABORT（event={event.event_id}）"
                )

        # 基线 4：agent_crash 且为唯一实例 → 禁止 SKIP（无法完成巡检）
        if event.exception_type == "agent_crash" and proposed in (ExceptionDecision.SKIP_TASK, ExceptionDecision.SKIP_STAGE):
            total_agents = event.context.get("total_perception_agents", 999)
            if total_agents <= 1:
                return False, ExceptionDecision.ABORT, (
                    f"安全基线拦截：唯一感知 Agent 崩溃，"
                    f"禁止 SKIP，已升级为 ABORT（event={event.event_id}）"
                )

        return True, proposed, "安全基线通过"


# ============================================================
# LLM 异常决策顾问（Phase 2）
# ============================================================
class LLMExceptionAdvisor:
    """基于 LLM 的异常分析与决策建议（L2 层）。

    设计原则：
    - 只做"建议"，不做最终决策（最终由 SafetyGuard + 策略引擎把关）；
    - 仅在 L1 策略不明确或需要业务理解时触发；
    - LLM 调用失败时透明降级，不影响原有 L1 决策；
    - 建议包含置信度，低置信度时 fallback 到 L1。
    """

    def __init__(self, llm: "LLMClient | None" = None, settings: "Settings | None" = None):
        self.llm = llm or get_llm_client()
        self.settings = settings or get_settings()

    def advise(
        self,
        event: AgentExceptionEvent,
        l1_decision: ExceptionDecision,
        l1_reason: str,
    ) -> tuple[ExceptionDecision, str, dict, float]:
        """基于 LLM 分析给出异常处理建议。

        Returns:
            (建议决策, 建议理由, 参数, 置信度)
        """
        # mock 后端直接返回 L1 决策（不消耗 token）
        if isinstance(self.llm, MockLLMClient):
            return l1_decision, "[mock] 降级到 L1 策略", {}, 0.0

        prompt = self._build_prompt(event, l1_decision, l1_reason)
        try:
            raw = self.llm.complete(
                system="你是 GuardEye 巡检系统的异常处理专家。你只输出 JSON，不输出其他内容。",
                user=prompt,
                json_mode=True,
            )
            return self._parse_advice(raw, l1_decision)
        except Exception as exc:
            # LLM 调用失败：透明降级到 L1
            return l1_decision, f"[LLM 降级] 调用失败：{exc}", {}, 0.0

    def _build_prompt(self, event: AgentExceptionEvent, l1_decision: ExceptionDecision, l1_reason: str) -> str:
        ctx = event.context
        return (
            "你是 GuardEye 巡检系统的异常处理专家。请根据以下信息给出异常处理建议。\n\n"
            "## 当前巡检状态\n"
            f"- 巡检ID: {event.patrol_id}\n"
            f"- 当前阶段: {event.stage}\n"
            f"- 规则数: {ctx.get('rules_count', 'unknown')}\n"
            f"- 任务数: {ctx.get('tasks_count', 'unknown')}\n"
            f"- 已完成任务: {ctx.get('completed_tasks', 'unknown')}\n\n"
            "## 异常事件\n"
            f"- 来源: {event.source_agent}"
            f"{f' ({event.source_agent_id})' if event.source_agent_id else ''}\n"
            f"- 类型: {event.exception_type}\n"
            f"- 错误信息: {event.error_message}\n"
            f"- 已重试次数: {ctx.get('retry_count', 0)}\n\n"
            "## 历史模式\n"
            f"- 近1小时同类异常次数: {ctx.get('history_same_type_1h', 0)}\n\n"
            "## L1 策略建议\n"
            f"- 建议决策: {l1_decision.value}\n"
            f"- 建议理由: {l1_reason}\n\n"
            "## 可选决策说明\n"
            "1. RETRY: 重试当前操作（适合临时性故障如网络抖动）\n"
            "2. SKIP_TASK: 跳过该子任务（适合非关键路径的单任务失败）\n"
            "3. SKIP_STAGE: 跳过整个阶段（适合配置错误导致的空结果）\n"
            "4. ABORT: 中断整个巡检（适合严重系统故障）\n"
            "5. ESCALATE: 升级处理（适合安全风险或需要人工判断的场景）\n\n"
            "## 约束\n"
            f"- 当前规则安全级别: {ctx.get('rule_severity', 'medium')}\n"
            "- high severity 规则相关的异常不可静默忽略\n"
            "- 如果判断是临时性故障，优先 RETRY\n"
            "- 如果判断是配置/规则问题，优先 SKIP_STAGE 或 ESCALATE\n\n"
            '请以 JSON 输出：{"decision": "RETRY|SKIP_TASK|SKIP_STAGE|ABORT|ESCALATE", '
            '"reasoning": "简要说明（中文）", "confidence": 0.0-1.0}'
        )

    def _parse_advice(
        self, raw: str, fallback_decision: ExceptionDecision
    ) -> tuple[ExceptionDecision, str, dict, float]:
        """解析 LLM 返回的 JSON 建议。"""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return fallback_decision, "[LLM 降级] 返回非 JSON", {}, 0.0

        decision_str = data.get("decision", "").upper()
        decision_map = {
            "RETRY": ExceptionDecision.RETRY,
            "SKIP_TASK": ExceptionDecision.SKIP_TASK,
            "SKIP_STAGE": ExceptionDecision.SKIP_STAGE,
            "ABORT": ExceptionDecision.ABORT,
            "ESCALATE": ExceptionDecision.ESCALATE,
            "IGNORE": ExceptionDecision.IGNORE,
        }
        decision = decision_map.get(decision_str, fallback_decision)
        reasoning = data.get("reasoning", "")
        confidence = float(data.get("confidence", 0.0))
        return decision, reasoning, {}, confidence


# ============================================================
# L0/L1 异常处理器（Phase 1 + Phase 2 集成）
# ============================================================
class L0L1ExceptionHandler:
    """基于硬编码策略 + 可选 LLM 顾问的异常处理器。

    决策逻辑：
    - 先按 (exception_type, source_agent) 查策略表（L1）；
    - [Phase 2] 复杂场景触发 LLM 顾问给出第二意见（L2）；
    - 再经 SafetyGuard 校验；
    - 最终输出决策 + 执行参数。
    """

    # 策略表：(exception_type, source_agent) → (决策, 默认参数)
    # source_agent 为 "" 表示通配
    _POLICY: dict[tuple[str, str], tuple[ExceptionDecision, dict]] = {
        # 超时策略
        ("timeout", "perception"): (ExceptionDecision.RETRY, {"max_retries": 1, "backoff_sec": 0}),
        ("timeout", "planner"): (ExceptionDecision.RETRY, {"max_retries": 2, "backoff_sec": 1}),
        ("timeout", "decision"): (ExceptionDecision.RETRY, {"max_retries": 2, "backoff_sec": 1}),
        ("timeout", "action"): (ExceptionDecision.RETRY, {"max_retries": 2, "backoff_sec": 1}),
        # 连接错误
        ("connection_error", ""): (ExceptionDecision.RETRY, {"max_retries": 2, "backoff_sec": 2}),
        # Agent 崩溃
        ("agent_crash", "perception"): (ExceptionDecision.RETRY, {"max_retries": 1, "remove_agent": True}),
        # 空结果
        ("empty_result", "planner"): (ExceptionDecision.SKIP_STAGE, {}),
        ("empty_result", "perception"): (ExceptionDecision.SKIP_TASK, {}),
        # 基础设施错误
        ("infrastructure_error", ""): (ExceptionDecision.ABORT, {}),
        # 全失败
        ("all_agents_failed", ""): (ExceptionDecision.ABORT, {}),
        # 未知异常（保守策略）
        ("unknown", ""): (ExceptionDecision.RETRY, {"max_retries": 1, "backoff_sec": 1}),
    }

    def __init__(self, llm_advisor: LLMExceptionAdvisor | None = None):
        self.classifier = ExceptionClassifier()
        self.guard = SafetyGuard()
        self.llm_advisor = llm_advisor

    def handle(self, event: AgentExceptionEvent) -> AgentExceptionEvent:
        """处理异常事件，写入决策结果，返回更新后的事件。

        这是统一入口。调用方应：
        1. 构造 AgentExceptionEvent；
        2. 调用 handler.handle(event)；
        3. 根据 event.final_decision 执行动作。
        """
        # Step 1：L0 分类（如未预分类）
        if event.exception_type == "unknown" and event.error_message:
            event.exception_type = self.classifier.classify(event.error_message, event.source_agent)

        # Step 2：查策略表
        proposed, params = self._lookup_policy(event.exception_type, event.source_agent)
        reason = f"L1 策略：({event.exception_type}, {event.source_agent}) → {proposed.value}"

        # Step 3：检查已重试次数，决定是否升级决策
        retry_count = event.context.get("retry_count", 0)
        max_retries = params.get("max_retries", 0)
        if proposed == ExceptionDecision.RETRY and retry_count >= max_retries:
            proposed, params = self._upgrade_after_exhausted(event)
            reason = f"L1 升级：重试耗尽 ({retry_count}/{max_retries}) → {proposed.value}"

        # Step 4：[Phase 2] 可选 LLM 顾问
        if self.llm_advisor is not None and self._should_use_llm(event, proposed):
            llm_decision, llm_reason, llm_params, confidence = self.llm_advisor.advise(
                event, proposed, reason
            )
            threshold = self.llm_advisor.settings.exception_llm_advisor_threshold
            if confidence >= threshold:
                proposed = llm_decision
                params = llm_params
                reason = f"[LLM 建议] {llm_reason} (confidence={confidence:.2f})"
            else:
                reason += f" | [LLM 未采纳] 置信度 {confidence:.2f} < 阈值 {threshold}"

        # Step 5：安全基线校验
        passed, final_decision, guard_reason = self.guard.validate(event, proposed)
        if not passed:
            # 基线拦截后，清空原参数（避免与升级决策冲突）
            params = {}
            reason = guard_reason

        # Step 6：写入事件
        event.final_decision = final_decision
        event.final_reason = reason
        event.action_params = params

        return event

    def _lookup_policy(self, exception_type: str, source_agent: str) -> tuple[ExceptionDecision, dict]:
        """查策略表，优先精确匹配， fallback 通配。"""
        key = (exception_type, source_agent)
        if key in self._POLICY:
            return self._POLICY[key]
        # fallback：通配 source_agent
        key_wildcard = (exception_type, "")
        if key_wildcard in self._POLICY:
            return self._POLICY[key_wildcard]
        # 最终 fallback：未知策略
        return ExceptionDecision.ABORT, {}

    def _upgrade_after_exhausted(self, event: AgentExceptionEvent) -> tuple[ExceptionDecision, dict]:
        """重试次数耗尽后升级决策。"""
        # 感知层超时耗尽 → 跳过该任务（不影响整体巡检）
        if event.source_agent == "perception":
            return ExceptionDecision.SKIP_TASK, {}
        # 其他 Agent 超时耗尽 → 中断巡检（关键路径不可缺失）
        return ExceptionDecision.ABORT, {}

    def _should_use_llm(self, event: AgentExceptionEvent, l1_decision: ExceptionDecision) -> bool:
        """判断是否需要 LLM 辅助决策。

        触发条件（满足任一即触发）：
        1. L1 策略保守地给出 ABORT，但异常可能可恢复；
        2. 语义级异常；
        3. 高频重复异常（模式识别）。
        """
        if self.llm_advisor is None:
            return False
        settings = self.llm_advisor.settings
        if not settings.exception_llm_advisor_enabled:
            return False

        # 场景 1：L1 策略保守地给出 ABORT，但异常可能可恢复
        if l1_decision == ExceptionDecision.ABORT and event.exception_type not in (
            "infrastructure_error", "all_agents_failed", "agent_crash"
        ):
            return True

        # 场景 2：语义级异常
        if event.exception_type == "semantic_mismatch":
            return True

        # 场景 3：高频重复异常（模式识别）
        same_type_count = event.context.get("history_same_type_1h", 0)
        if same_type_count >= 3:
            return True

        return False


# ============================================================
# 便捷函数
# ============================================================
def make_exception_event(
    patrol_id: str,
    source_agent: str,
    error_message: str,
    stage: str = "",
    source_agent_id: str = "",
    context: dict | None = None,
) -> AgentExceptionEvent:
    """便捷构造异常事件，自动生成 event_id 和 timestamp。"""
    import uuid

    return AgentExceptionEvent(
        event_id=f"exc_{uuid.uuid4().hex[:8]}",
        patrol_id=patrol_id,
        source_agent=source_agent,
        source_agent_id=source_agent_id,
        stage=stage,
        error_message=error_message,
        context=context or {},
    )
