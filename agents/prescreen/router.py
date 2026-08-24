"""三级路由调度器：L0 → L1 → L2，并做成本记账。

设计要点（对应面试 Q11「三级路由的依据」、Q14「成本账怎么算」）：
- 按「处理成本 × 信息密度」分层：运动检测免费过滤静止，抽帧去重降帧数，VLM 只处理保留帧；
- 每层记录处理量，最终给出 CostBreakdown，让「成本降了一个数量级」有账可算；
- Router 本身串行处理单个 task，并发由 dispatcher 节点在 task 维度做（见 nodes/dispatcher.py）。

诚实标注：total_tokens 与 total_cost 是估算口径，真实单价需在 Phase 6 用 Langfuse Trace 回填；
这里把「单位成本常量」集中在 cost_unit 字段，便于替换，不写死魔法数字散落各处。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agents.config import Settings, get_settings
from agents.llm import LLMClient, get_llm_client
from agents.models import AnalysisTask, Finding, FrameEvidence
from agents.prescreen.l0_motion import detect_motion_intervals
from agents.prescreen.l1_sampler import sample_frames
from agents.prescreen.l2_vlm import ScreenFn, frame_to_b64, screen_frame


@dataclass
class CostBreakdown:
    """单 task 的成本明细（帧数 + token + 金额）。"""
    l1_kept_frames: int = 0       # L1 去重后保留帧数
    l2_screened_frames: int = 0   # L2 VLM 实际处理帧数
    l2_hit_frames: int = 0        # L2 命中帧数
    total_tokens: int = 0         # 估算 token（TODO: 用 Langfuse 实测回填）
    total_cost: float = 0.0       # 估算金额（单位成本见 Router.cost_unit）

    def merge(self, other: "CostBreakdown") -> "CostBreakdown":
        return CostBreakdown(
            l1_kept_frames=self.l1_kept_frames + other.l1_kept_frames,
            l2_screened_frames=self.l2_screened_frames + other.l2_screened_frames,
            l2_hit_frames=self.l2_hit_frames + other.l2_hit_frames,
            total_tokens=self.total_tokens + other.total_tokens,
            total_cost=self.total_cost + other.total_cost,
        )


class Router:
    """把单个 AnalysisTask 走完 L0→L1→L2，产出 Finding 与成本明细。"""

    # 单位成本常量（占位，需按真实模型单价替换，对应面试 Q14）
    cost_per_token: float = 0.0  # 元/token，mock 阶段为 0，真实后端按模型单价填
    est_tokens_per_frame: int = 600  # 单帧 VLM 请求的估算 token（含图 + 输出）

    def __init__(self, settings: Settings | None = None, llm: LLMClient | None = None, screen_fn: ScreenFn | None = None):
        self.settings = settings or get_settings()
        self.llm = llm or get_llm_client()
        # 可注入自定义初筛函数（如 ScriptedScreen），默认走 screen_frame
        self._screen_fn = screen_fn

    def _screen(self, rule, frame_b64: str, camera_id: str, timestamp: float):
        if self._screen_fn is not None:
            return self._screen_fn(rule, frame_b64, camera_id, timestamp)
        return screen_frame(rule, frame_b64, camera_id, timestamp, llm=self.llm)

    def route(self, task: AnalysisTask) -> tuple[Finding, CostBreakdown]:
        rule = task.rule
        clip_path = task.clip_path
        breakdown = CostBreakdown()

        # L0：运动检测（零成本，只过滤静止）
        motion = detect_motion_intervals(clip_path, self.settings)

        # L1：抽帧 + 去重
        frames = sample_frames(clip_path, motion, self.settings)
        breakdown.l1_kept_frames = len(frames)

        # L2：VLM 初筛
        hit_evidence: list[FrameEvidence] = []
        hit_confidences: list[float] = []
        for f in frames:
            b64 = frame_to_b64(f.image)
            res = self._screen(rule, b64, task.camera_id, f.timestamp_seconds)
            breakdown.l2_screened_frames += 1
            if res.hit:
                breakdown.l2_hit_frames += 1
                hit_confidences.append(res.confidence)
                hit_evidence.append(
                    FrameEvidence(
                        frame_index=f.frame_index,
                        timestamp_seconds=f.timestamp_seconds,
                        description=res.evidence,
                    )
                )

        breakdown.total_tokens = breakdown.l2_screened_frames * self.est_tokens_per_frame
        breakdown.total_cost = breakdown.total_tokens * self.cost_per_token

        hit = len(hit_evidence) > 0
        confidence = (sum(hit_confidences) / len(hit_confidences)) if hit_confidences else 0.0
        finding = Finding(
            id=f"finding_{task.id}",
            task_id=task.id,
            rule_id=rule.id,
            camera_id=task.camera_id,
            hit=hit,
            confidence=confidence,
            evidence=hit_evidence,
            hit_frame_indices=[e.frame_index for e in hit_evidence],
            cost_tokens=breakdown.total_tokens,
            cost_currency=breakdown.total_cost,
            # 带入绝对时间信息，供下游时序聚合使用
            clip_start_time=task.clip_start_time,
            clip_end_time=task.clip_end_time,
            duration_seconds=task.duration_seconds,
        )
        return finding, breakdown
