"""L1 抽帧策略 + 帧去重。

两个职责：
1. 采样：基线 base_fps，运动区间动态升帧至 motion_max_fps——
   静止画面信息密度低就少采，运动画面信息密度高就多采（对应面试 Q8「抽帧策略怎么定」）。
2. 去重：相邻帧相似度过高则丢弃，降低下游 VLM 调用量。

去重的无模型实现用「平均哈希（aHash）+ 汉明距离」——真实、确定、可解释；
生产可用 CLIP embedding 替换（接口一致），aHash 只是离线降级，不编造语义能力。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents.config import Settings, get_settings
from agents.prescreen.l0_motion import MotionInterval, to_gray


@dataclass
class SampledFrame:
    frame_index: int
    timestamp_seconds: float
    image: np.ndarray


def average_hash_bits(frame: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """计算平均哈希位向量（bool 数组，长度 hash_size^2）。

    纯 numpy 实现（块平均缩放 + 相邻列比较），不依赖 cv2/PIL，可单测。
    思路：把图缩到 (hash_size+1) x hash_size，逐行比较相邻列像素大小。
    """
    gray = to_gray(frame)
    h, w = gray.shape
    row_blocks = h // (hash_size + 1)
    col_blocks = w // hash_size
    if row_blocks == 0 or col_blocks == 0:
        # 帧过小，退化为按最小块处理
        return np.zeros(hash_size * hash_size, dtype=bool)
    small = gray[: row_blocks * (hash_size + 1), : col_blocks * hash_size]
    small = small.reshape(hash_size + 1, row_blocks, hash_size, col_blocks).mean(axis=(1, 3))
    return (small[:-1, :] > small[1:, :]).flatten()


def hash_similarity(bits_a: np.ndarray, bits_b: np.ndarray) -> float:
    """由汉明距离换算相似度：1 - 汉明距离 / 总位数。"""
    if bits_a.shape != bits_b.shape:
        return 0.0
    total = bits_a.size
    if total == 0:
        return 1.0
    hamming = int(np.sum(bits_a != bits_b))
    return 1.0 - hamming / total


def _fps_at(timestamp: float, motion_intervals: list[MotionInterval], base_fps: float, motion_max_fps: float) -> float:
    in_motion = any(iv.start_seconds <= timestamp <= iv.end_seconds for iv in motion_intervals)
    return motion_max_fps if in_motion else base_fps


def sample_frames(
    video_path: str,
    motion_intervals: list[MotionInterval],
    settings: Settings | None = None,
    dedup: bool = True,
) -> list[SampledFrame]:
    """按「基线 1fps + 运动升帧」采样，并对相邻帧去重。

    采样逻辑：维护 next_sample_time，根据当前时间点所处区间决定采样间隔；
    去重逻辑：新帧与上一保留帧的 aHash 相似度 >= dedup_threshold 则丢弃。
    """
    import cv2

    settings = settings or get_settings()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[SampledFrame] = []
    next_sample_time = 0.0
    prev_bits = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = frame_idx / fps
        if timestamp >= next_sample_time:
            if dedup:
                bits = average_hash_bits(frame)
                if prev_bits is not None and hash_similarity(prev_bits, bits) >= settings.dedup_threshold:
                    # 与上一保留帧几乎相同，跳过（降本）
                    interval = 1.0 / _fps_at(timestamp, motion_intervals, settings.base_fps, settings.motion_max_fps)
                    next_sample_time = timestamp + interval
                    frame_idx += 1
                    continue
                prev_bits = bits

            frames.append(SampledFrame(frame_index=frame_idx, timestamp_seconds=timestamp, image=frame))
            interval = 1.0 / _fps_at(timestamp, motion_intervals, settings.base_fps, settings.motion_max_fps)
            next_sample_time = timestamp + interval

        frame_idx += 1

    cap.release()
    return frames
