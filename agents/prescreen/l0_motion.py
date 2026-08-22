"""L0 运动检测：零成本过滤静止片段。

为什么用帧差法而非复杂背景建模：
- 帧差法简单、确定、零模型，符合 L0「零成本过滤 ~50% 静止帧」的定位；
- 复杂背景建模（如 MOG2）在真实部署可替换，但 demo 阶段帧差足够且可解释；
- 面试可讲：L0 是「信息密度」判断——静止画面几乎不含新信息，直接丢弃是成本最优解。

分层：
- compute_motion_ratio：纯 numpy，不依赖 cv2，可单测；
- detect_motion_intervals：视频 IO（cv2.VideoCapture），把逐帧比值合并成运动区间。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents.config import Settings, get_settings


@dataclass
class MotionInterval:
    start_seconds: float
    end_seconds: float


def to_gray(frame: np.ndarray) -> np.ndarray:
    """彩色转灰度（numpy 加权，避免额外依赖 cv2 才能单测）。"""
    if frame.ndim == 3:
        return (
            frame[..., 0] * 0.299 + frame[..., 1] * 0.587 + frame[..., 2] * 0.114
        )
    return frame


def compute_motion_ratio(prev: np.ndarray, curr: np.ndarray, pixel_threshold: float = 25.0) -> float:
    """相邻两帧的「运动像素比例」。

    做法：灰度化 → 帧差绝对值 → 超过 pixel_threshold 的像素占比。
    返回 [0, 1]，越大表示画面变化越剧烈。
    """
    prev_g = to_gray(prev).astype(np.float32)
    curr_g = to_gray(curr).astype(np.float32)
    diff = np.abs(prev_g - curr_g)
    motion_mask = diff > pixel_threshold
    return float(motion_mask.mean())


def detect_motion_intervals(
    video_path: str,
    settings: Settings | None = None,
    sample_every: float = 0.5,
) -> list[MotionInterval]:
    """扫描视频，返回运动区间列表。

    采样帧做相邻帧差，连续超过阈值的采样点合并为一个区间。
    sample_every 控制采样粒度（秒），值越小越精细、计算量越大。
    """
    import cv2

    settings = settings or get_settings()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(fps * sample_every))

    intervals: list[MotionInterval] = []
    current_start: float | None = None
    prev_gray = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / fps
        gray = to_gray(frame).astype(np.float32)

        if prev_gray is not None:
            diff = np.abs(gray - prev_gray)
            ratio = float((diff > 25.0).mean())
            if ratio >= settings.motion_ratio_threshold:
                if current_start is None:
                    current_start = timestamp
            else:
                if current_start is not None:
                    intervals.append(MotionInterval(current_start, timestamp))
                    current_start = None

        prev_gray = gray
        frame_idx += 1

    if current_start is not None:
        intervals.append(MotionInterval(current_start, frame_idx / fps))

    cap.release()
    return intervals
