"""verify 节点：准确率核心（对应面试 Q7「VLM 幻觉怎么控制」）。

两道防线：
1. 跨帧一致性：单帧命中不采信，滑动窗口内连续 N 帧命中才升级——
   因为单帧 VLM 很容易幻觉，多帧一致才可信；
2. 大模型二次确认：通过一致性的 finding，注入 SOP 判定依据，让大模型最终定夺。

诚实标注：mock 后端下「二次确认」是透传（只依赖一致性），真实后端才走 LLM 复核。
"""

from __future__ import annotations

import json

from agents.config import get_settings
from agents.llm import LLMClient, MockLLMClient, get_llm_client
from agents.models import Alarm, Finding, LogEntry, Verification
from agents.toolbox import Toolbox


def _get_field(obj, key: str, default=None):
    """兼容 dict 与 Pydantic 模型的字段读取（避免 getattr 默认参数被提前求值）。"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def passes_consistency(finding: Finding, window_size: int, max_gap_seconds: float) -> bool:
    """滑动窗口跨帧一致性：是否存在 window_size 个命中帧，且首尾时间差 <= max_gap_seconds。

    命中帧按时间排序，遍历长度为 window_size 的滑动窗口，任一窗口满足时间跨度约束即通过。
    """
    times = sorted(e.timestamp_seconds for e in finding.evidence)
    for i in range(len(times) - window_size + 1):
        if times[i + window_size - 1] - times[i] <= max_gap_seconds:
            return True
    return False


def verify_finding(
    finding: Finding,
    consistent: bool,
    rule_name: str,
    toolbox: Toolbox,
    llm: LLMClient,
) -> Verification:
    if not consistent:
        return Verification(
            finding_id=finding.id,
            verdict="rejected",
            confidence=0.0,
            reasoning="命中帧不足或跨帧不一致，单帧/偶发命中不采信",
        )
    if isinstance(llm, MockLLMClient):
        return Verification(
            finding_id=finding.id,
            verdict="confirmed",
            confidence=finding.confidence,
            reasoning="跨帧一致性通过（mock 二次确认透传，真实后端走 LLM 复核）",
        )

    # 真实后端：检索 SOP 作为判定依据注入 prompt
    sop_hits = toolbox.search_sop(rule_name, limit=2)
    sop_text = "\n".join(f"- {s['title']}: {s['content']}" for s in sop_hits) or "无相关 SOP"
    evidence_text = "\n".join(
        f"帧@{e.frame_index}(t={e.timestamp_seconds:.1f}s): {e.description}" for e in finding.evidence
    )
    prompt = (
        "你是巡检复核专家。根据初筛命中证据与 SOP，判断是否确认为真实告警。\n"
        f"规则：{rule_name}\n证据：\n{evidence_text}\nSOP 判定依据：\n{sop_text}\n"
        '只输出 JSON：{"verdict": "confirmed|rejected|uncertain", "confidence": 0.0~1.0, "reasoning": "..."}'
    )
    raw = llm.complete(system="你是巡检复核助手。", user=prompt, json_mode=True)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return Verification(finding_id=finding.id, verdict="uncertain", confidence=0.0, reasoning="复核输出不可解析")
    verdict = data.get("verdict", "uncertain")
    if verdict not in {"confirmed", "rejected", "uncertain"}:
        verdict = "uncertain"
    return Verification(
        finding_id=finding.id,
        verdict=verdict,
        confidence=float(data.get("confidence", 0.0)),
        reasoning=str(data.get("reasoning", "")),
        sop_references=[s["id"] for s in sop_hits],
    )


def verifier_node(state: dict, toolbox: Toolbox) -> dict:
    findings: list[Finding] = state.get("findings", [])
    settings = get_settings()
    llm = get_llm_client()

    # 从 tasks 建立 rule_id -> (severity, rule_name, duration_threshold) 映射
    rule_meta: dict[str, tuple[str, str, float | None]] = {}
    for t in state.get("tasks", []):
        if hasattr(t, "rule"):
            rule = t.rule
        else:
            rule = t.get("rule", {})
        rid = _get_field(rule, "id", "")
        severity = _get_field(rule, "severity", "medium")
        name = _get_field(rule, "name", rid)
        duration = _get_field(rule, "duration_threshold_seconds")
        if rid:
            rule_meta[rid] = (severity, name, duration)

    max_gap = settings.consecutive_hit_window / settings.base_fps
    verifications: list[Verification] = []
    alarms: list[Alarm] = []

    for f in findings:
        severity, rule_name, duration_threshold = rule_meta.get(f.rule_id, ("medium", f.rule_id, None))
        consistent = passes_consistency(f, settings.consecutive_hit_window, max_gap)
        ver = verify_finding(f, consistent, rule_name, toolbox, llm)
        verifications.append(ver)

        # 持续型异常：不在 verifier 层创建 Alarm，留给 temporal_aggregate 做时序聚合
        if duration_threshold:
            continue

        if ver.verdict == "confirmed":
            evidence = [e.model_dump() for e in f.evidence]
            alarm_dict = toolbox.create_alarm(
                camera_id=f.camera_id,
                rule_id=f.rule_id,
                rule_name=rule_name,
                severity=severity,
                confidence=ver.confidence,
                evidence=evidence,
            )
            alarms.append(Alarm(**alarm_dict))

    log = LogEntry(
        node="verify",
        message=f"复核 {len(findings)} 个 finding，确认告警 {len(alarms)} 条",
    )
    return {"verifications": verifications, "alarms": alarms, "logs": [log]}
