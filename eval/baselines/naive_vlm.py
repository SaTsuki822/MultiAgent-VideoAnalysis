"""基线系统：单帧直连大 VLM（无三级路由）。

用于对比「三级路由」的成本与准确性差异。诚实标注：本基线给出的是「成本与误报的结构性估算」，
真实数字需接大 VLM 后回填；其价值在于展示「为什么直连大模型又贵又容易误报」的账（对应面试 Q14）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaselineResult:
    frames_sent_to_big_vlm: int
    est_cost: float
    est_false_positive: float


def estimate_naive_baseline(total_frames: int, fps: float = 30.0, cost_per_frame: float = 0.0) -> BaselineResult:
    """估算单帧直连大 VLM 的处理成本与误报。

    total_frames：视频总帧数。直连方案对每一帧都调用大模型，无运动过滤、无去重、无初筛。
    误报按历史经验单帧方案 ~25% 的常量估算（占位，需实测回填）。
    """
    frames = total_frames  # 每一帧都送
    est_cost = frames * cost_per_frame
    est_fp = frames * 0.25  # 占位常量，见 plan「单帧方案误报率 ~25%」
    return BaselineResult(frames_sent_to_big_vlm=frames, est_cost=est_cost, est_false_positive=est_fp)
