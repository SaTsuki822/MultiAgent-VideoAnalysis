"""测试感知层纯逻辑：运动检测、aHash 去重、跨帧一致性。"""

import numpy as np

from agents.models import Finding, FrameEvidence
from agents.nodes.verifier import passes_consistency
from agents.prescreen.l0_motion import compute_motion_ratio
from agents.prescreen.l1_sampler import average_hash_bits, hash_similarity


def test_motion_ratio_identical_frames_is_zero():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert compute_motion_ratio(frame, frame) == 0.0


def test_motion_ratio_changed_frames_positive():
    a = np.zeros((100, 100, 3), dtype=np.uint8)
    b = np.full((100, 100, 3), 255, dtype=np.uint8)
    assert compute_motion_ratio(a, b) > 0.5


def test_hash_similarity_identical_frame_is_one():
    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    assert hash_similarity(average_hash_bits(frame), average_hash_bits(frame)) == 1.0


def test_passes_consistency_consecutive_hits():
    f = Finding(
        id="f1", task_id="t1", rule_id="r1", camera_id="c1", hit=True, confidence=0.9,
        evidence=[FrameEvidence(frame_index=i, timestamp_seconds=float(i)) for i in range(3)],
    )
    assert passes_consistency(f, window_size=3, max_gap_seconds=3.0) is True


def test_passes_consistency_sparse_hits_fails():
    f = Finding(
        id="f1", task_id="t1", rule_id="r1", camera_id="c1", hit=True, confidence=0.9,
        evidence=[
            FrameEvidence(frame_index=0, timestamp_seconds=0.0),
            FrameEvidence(frame_index=10, timestamp_seconds=10.0),
        ],
    )
    assert passes_consistency(f, window_size=3, max_gap_seconds=3.0) is False
