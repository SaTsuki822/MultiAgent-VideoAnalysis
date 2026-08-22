"""PatrolState 与 reducer 定义。

为什么单独一个文件（对应面试 Q6「状态怎么设计、并发怎么控制」）：
- reducer 是纯函数，不依赖 LangGraph，可独立单测；
- LangGraph 的 reducer 机制本质是 Annotated[type, reducer]，这里把 reducer 显式写出，
  能讲清楚「并发节点写同一字段时如何合并」。

Reducer 设计：
- findings / verifications / feedback / logs：追加式（operator.add），并发安全；
- alarms / pending_review：按 id 去重合并（merge_by_id）—— dispatch 并行时同一告警
  可能被多次 push，需要按 id 收敛而非重复出现。
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from agents.models import (
    Alarm,
    AnalysisTask,
    Camera,
    Clip,
    Feedback,
    Finding,
    LogEntry,
    Report,
    Rule,
    Verification,
)


def merge_by_id(existing: list, incoming: list) -> list:
    """按 id 去重合并：同 id 新值覆盖旧值，顺序保持首次出现顺序。

    适用 alarms —— 同一告警跨节点被多次写入时，收敛为一条且保留最新状态。
    """
    merged: dict = {item.id: item for item in existing}
    for item in incoming:
        merged[item.id] = item
    return list(merged.values())


class PatrolState(TypedDict, total=False):
    # ---- 输入（初始态设置，覆盖语义）----
    patrol_id: str
    rules: list[Rule]
    cameras: list[Camera]
    clips: list[Clip]

    # ---- planner 产出 ----
    tasks: list[AnalysisTask]

    # ---- dispatch / prescreen / verify 产出（追加式，并发安全）----
    findings: Annotated[list[Finding], operator.add]
    verifications: Annotated[list[Verification], operator.add]

    # ---- 告警（按 id 去重合并）----
    alarms: Annotated[list[Alarm], merge_by_id]
    pending_review: Annotated[list[Alarm], merge_by_id]

    # ---- HITL ----
    feedback: Annotated[list[Feedback], operator.add]

    # ---- 最终产物 ----
    report: Optional[Report]

    # ---- 审计 ----
    logs: Annotated[list[LogEntry], operator.add]
