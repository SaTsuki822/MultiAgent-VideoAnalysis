"""temporal_aggregate 节点：持续型异常的片段级状态机 + 时序聚合。

职责：
1. 对 confirmed 的 finding，按 (camera_id, rule_id) 维护活跃异常表；
2. 跨片段累计异常持续时间（视频时间轴，基于 clip_start_time / clip_end_time）；
3. 当累计持续时间 >= rule.duration_threshold_seconds 时，升级为正式 Alarm；
4. 对 rejected 的 finding，递增 miss_count，连续 miss 则关闭活跃异常；
5. 无 duration_threshold 的规则：直接透传 verifier 产出的 Alarm，不做时序聚合。

状态机：
    IDLE ──confirmed──→ DETECTED ──confirmed──→ ACCUMULATING ──duration≥threshold──→ CONFIRMED
                          │                         │                              │
                          └────rejected×2───────────┘──────────────────────────────┘──→ CLOSED
"""

from __future__ import annotations

from datetime import datetime

from agents.config import get_settings
from agents.models import Alarm, Finding, LogEntry, OngoingAnomaly, Verification
from agents.temporal_store import close, get, put
from agents.toolbox import Toolbox

# 连续未命中几次后关闭活跃异常（可配置）
_MAX_MISS_COUNT = 2


def _find_task_for_finding(finding: Finding, tasks: list) -> dict | None:
    """通过 task_id 在 tasks 列表中找到对应的 task dict。"""
    for t in tasks:
        if t.get("id") == finding.task_id:
            return t
    return None


def _build_rule_map(tasks: list) -> dict[tuple[str, str], dict]:
    """建立 (camera_id, rule_id) → rule dict 的映射。"""
    result: dict[tuple[str, str], dict] = {}
    for t in tasks:
        cam = t.get("camera_id", "")
        rule = t.get("rule", {})
        rid = rule.get("id", "")
        if cam and rid:
            result[(cam, rid)] = rule
    return result


def temporal_aggregate_node(state: dict, toolbox: Toolbox) -> dict:
    """时序聚合节点。

    输入 state 需包含：
    - findings: list[Finding | dict]（原始初筛结果）
    - tasks: list[dict | AnalysisTask]（原始子任务，含 rule.duration_threshold_seconds）
    - verifications: list[Verification | dict]（verify 节点产出）
    - alarms: list[Alarm | dict]（verify 节点产出的告警）

    输出：
    - alarms: list[Alarm]（时序聚合后的告警：单帧透传 + 持续型阈值满足后才生成）
    - logs: list[LogEntry]
    """
    # ---- 类型归一化（兼容 Pydantic 对象与 dict） ----
    def _norm_task(t):
        if hasattr(t, "model_dump"):
            return t.model_dump()
        return dict(t)

    def _norm_finding(f):
        if isinstance(f, Finding):
            return f
        return Finding(**f)

    def _norm_verification(v):
        if isinstance(v, Verification):
            return v
        return Verification(**v)

    def _norm_alarm(a):
        if isinstance(a, Alarm):
            return a
        return Alarm(**a)

    findings: list[Finding] = [_norm_finding(f) for f in state.get("findings", [])]
    tasks: list[dict] = [_norm_task(t) for t in state.get("tasks", [])]
    verifications: list[Verification] = [_norm_verification(v) for v in state.get("verifications", [])]
    verifier_alarms: list[Alarm] = [_norm_alarm(a) for a in state.get("alarms", [])]

    rule_map = _build_rule_map(tasks)
    finding_map = {f.id: f for f in findings}

    # 输出告警列表
    output_alarms: list[Alarm] = []

    # 1. 无 duration_threshold 的规则：直接透传 verifier 的 alarms
    for alarm in verifier_alarms:
        key = (alarm.camera_id, alarm.rule_id)
        rule = rule_map.get(key, {})
        threshold = rule.get("duration_threshold_seconds")
        if threshold is None or threshold == 0:
            output_alarms.append(alarm)

    # 2. 有 duration_threshold 的规则：走状态机累计
    confirmed_count = 0
    closed_count = 0

    for ver in verifications:
        finding = finding_map.get(ver.finding_id)
        if finding is None:
            continue

        task = _find_task_for_finding(finding, tasks)
        if task is None:
            continue

        rule = task.get("rule", {})
        duration_threshold = rule.get("duration_threshold_seconds")
        if duration_threshold is None or duration_threshold == 0:
            continue

        camera_id = finding.camera_id
        rule_id = finding.rule_id
        rule_name = rule.get("name", "")
        severity = rule.get("severity", "medium")

        anomaly = get(camera_id, rule_id)

        if ver.verdict == "confirmed":
            # 首次命中或重新命中（之前已 closed）
            if anomaly is None or anomaly.state == "closed":
                anomaly = OngoingAnomaly(
                    camera_id=camera_id,
                    rule_id=rule_id,
                    rule_name=rule_name,
                    severity=severity,
                    duration_threshold_seconds=float(duration_threshold),
                    state="detected",
                    first_seen_at=finding.clip_start_time,
                    last_seen_at=finding.clip_end_time,
                    accumulated_seconds=finding.duration_seconds or 0.0,
                    evidence_snapshots=list(finding.evidence[:3]),
                    hit_count=1,
                    miss_count=0,
                )
            else:
                # 继续累加
                anomaly.state = "accumulating"
                anomaly.last_seen_at = finding.clip_end_time
                # 基于视频时间轴计算累计时长（first → last 的跨度）
                anomaly.accumulated_seconds = anomaly.compute_duration()
                # 追加证据（限制总量避免无限增长）
                remaining = max(0, 10 - len(anomaly.evidence_snapshots))
                anomaly.evidence_snapshots.extend(list(finding.evidence[:remaining]))
                anomaly.hit_count += 1
                anomaly.miss_count = 0

            put(anomaly)

            # 检查是否满足持续阈值
            if anomaly.accumulated_seconds >= float(duration_threshold) and anomaly.state != "confirmed":
                anomaly.state = "confirmed"
                put(anomaly)

                alarm = Alarm(
                    id=f"alarm_{camera_id}_{rule_id}_{int(datetime.now().timestamp())}",
                    camera_id=camera_id,
                    rule_id=rule_id,
                    rule_name=rule_name,
                    severity=severity,
                    confidence=ver.confidence,
                    evidence=anomaly.evidence_snapshots,
                )
                anomaly.alarm_id = alarm.id
                put(anomaly)
                output_alarms.append(alarm)
                confirmed_count += 1

        elif ver.verdict == "rejected":
            if anomaly is not None and anomaly.state not in ("closed", "confirmed"):
                anomaly.miss_count += 1
                if anomaly.miss_count >= _MAX_MISS_COUNT:
                    anomaly.state = "closed"
                    close(camera_id, rule_id, reason=f"连续 {anomaly.miss_count} 次未命中，关闭活跃异常")
                    closed_count += 1
                else:
                    put(anomaly)

    log = LogEntry(
        node="temporal_aggregate",
        message=(
            f"时序聚合：透传单帧告警 {len(output_alarms) - confirmed_count} 条，"
            f"持续型确认 {confirmed_count} 条，关闭 {closed_count} 条"
        ),
    )
    return {"alarms": output_alarms, "logs": [log]}
