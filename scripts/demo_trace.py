"""执行轨迹分析 Demo：跑一轮巡检 → 采集 span → 输出时间轴与聚合摘要。

演示「Agent 执行轨迹分析」能力：每个节点执行的耗时、状态、token/成本都被记录，
最终输出瓶颈节点与每节点耗时分布，用于定位慢在哪、成本花在哪。

运行：python scripts/demo_trace.py
"""

from __future__ import annotations

import os
import sys

# 保证能 import 项目包（从任意目录运行）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.models import Rule
from agents.prescreen.l2_vlm import ScriptedScreen
from agents.toolbox import build_default_toolbox
from agents.tracing import JsonlTraceStore, Tracer
from agents.workflow import run_pipeline

CLIP_PATH = "data/clips/cam_001.mp4"
CAMERA = {"id": "cam_001", "name": "东门入口", "area": "gate_east", "rtsp_url": ""}
RULE = Rule(id="rule_intrusion", name="区域入侵", description="非工作时间有人进入危险区域", severity="medium")


def make_synthetic_clip(path: str, duration: float = 10.0, fps: int = 10, size=(320, 240)) -> None:
    """生成合成视频：移动白块模拟运动（真实文件，供 L0/L1 处理）。"""
    import cv2
    import numpy as np

    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError("无法创建合成视频，请确认 opencv-python 已安装")
    h, w = size[1], size[0]
    for i in range(int(duration * fps)):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        x = (i * 6) % (w - 40)
        frame[90:150, x : x + 40] = 255
        writer.write(frame)
    writer.release()
    print(f"[demo_trace] 已生成合成视频: {path}")


def main() -> None:
    make_synthetic_clip(CLIP_PATH)

    # 命中表：cam_001 在 2.0/3.0/4.0 秒命中「区域入侵」（连续 3 帧，通过一致性）
    hit_map = {"cam_001": {"rule_intrusion": [2.0, 3.0, 4.0]}}
    toolbox = build_default_toolbox(screen_fn=ScriptedScreen(hit_map))

    trace_id = "trace_demo_1"
    tracer = Tracer(trace_id, store=JsonlTraceStore())

    initial = {"patrol_id": trace_id, "rules": [RULE], "cameras": [CAMERA]}
    state = run_pipeline(
        initial,
        toolbox,
        hitl_handler=lambda p: [{"alarm_id": a.id, "decision": "confirm"} for a in p],
        tracer=tracer,
    )
    tracer.save()

    summary = tracer.analyze()
    print("\n" + "=" * 60)
    print(summary.timeline)
    print("=" * 60)
    print(f"墙钟耗时   : {summary.total_duration_ms:.1f} ms")
    print(f"节点耗时和 : {summary.sum_duration_ms:.1f} ms")
    print(f"错误数     : {summary.error_count}")
    print(f"瓶颈节点   : {summary.bottleneck} ({summary.bottleneck_ms:.1f} ms)")
    print("节点聚合   :")
    for name, agg in summary.per_node.items():
        print(f"  {name:<18} count={agg.count}  avg={agg.avg_ms:7.1f}ms  max={agg.max_ms:7.1f}ms  tokens={agg.tokens}")
    print(f"\n告警 {len(state.get('alarms', []))} 条，轨迹已落盘 data/traces.jsonl")


if __name__ == "__main__":
    main()
