"""端到端 demo：合成视频 + ScriptedScreen + 误报记忆闭环。

演示 plan 文档里「最打动人」的闭环：误报 → 人工标记 → 下次自动抑制。

三轮巡检（保持默认 fp_activation_count=2，完整展示「积累 N 次才生效」机制）：
- 第 1 轮：检出告警 → 人工标记为误报（误报签名 occurrence_count=1）；
- 第 2 轮：同类告警再次检出，但 count=1 < 2，不抑制 → 人工再标记误报（count=2）；
- 第 3 轮：同类告警被误报记忆自动抑制，无需人工复核。

诚实标注：ScriptedScreen 是「数据注入」模拟真实 VLM 命中，非真实视觉理解；
真实部署请配置真实 VLM（见 infra/sglang.sh 与 .env.example）。
运行：python scripts/demo.py（需安装依赖：pip install -e ".[dev]"）
"""

from __future__ import annotations

import os
import sys

# 保证能 import 项目包（从任意目录运行）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agents.models import Rule
from agents.prescreen.l2_vlm import ScriptedScreen
from agents.toolbox import build_default_toolbox
from agents.workflow import run_pipeline

CLIP_PATH = "data/clips/cam_001.mp4"
CAMERA = {"id": "cam_001", "name": "东门入口", "area": "gate_east", "rtsp_url": ""}
# 用 medium 级别规则演示误报抑制——因为防污染规则规定 high 级别永不自动抑制（见 false_positive.py）
RULE = Rule(id="rule_intrusion", name="区域入侵", description="非工作时间有人进入危险区域", severity="medium")


def make_synthetic_clip(path: str, duration: float = 10.0, fps: int = 10, size=(320, 240)) -> None:
    """用 OpenCV 生成合成视频：移动白块模拟运动（真实文件，供 L0/L1 处理）。"""
    import cv2

    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError("无法创建合成视频，请确认 opencv-python 已安装")
    h, w = size[1], size[0]
    for i in range(int(duration * fps)):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        x = (i * 6) % (w - 40)  # 移动白块 → 产生帧差 → L0 判为运动
        frame[90:150, x : x + 40] = 255
        writer.write(frame)
    writer.release()
    print(f"[demo] 已生成合成视频: {path}（{duration}s @ {fps}fps）")


def make_handler(decisions: list[dict]):
    """构造人工复核回调：把决策套到当前 pending 的第一条告警上。"""
    def handler(pending):
        print(f"  [HITL] 待复核告警 {len(pending)} 条: {[a.id for a in pending]}")
        out = []
        for d in decisions:
            out.append({"alarm_id": pending[0].id, "decision": d["decision"], "comment": d.get("comment", "")})
        return out
    return handler


def run_round(round_no: int, toolbox, decisions: list[dict]):
    initial = {
        "patrol_id": f"patrol_{round_no}",
        "rules": [RULE],
        "cameras": [CAMERA],
    }
    handler = make_handler(decisions)
    state = run_pipeline(initial, toolbox, hitl_handler=handler)

    print(f"\n===== 第 {round_no} 轮巡检结果 =====")
    for a in state.get("alarms", []):
        tag = f"suppressed({a.suppression_reason})" if a.suppressed else a.status
        print(f"  - 告警 {a.id} [{a.rule_name}] 状态={tag}")
    report = state.get("report")
    if report:
        print(f"  报告: {report.summary}")
    return state


def main() -> None:
    make_synthetic_clip(CLIP_PATH)

    # 命中表：cam_001 在 2.0/3.0/4.0 秒命中「区域入侵」（连续 3 帧，通过一致性）
    hit_map = {"cam_001": {"rule_intrusion": [2.0, 3.0, 4.0]}}
    toolbox = build_default_toolbox(screen_fn=ScriptedScreen(hit_map))

    run_round(1, toolbox, [{"decision": "false_positive", "comment": "夜间误报，实际是保安巡逻"}])
    run_round(2, toolbox, [{"decision": "false_positive", "comment": "同类夜间误报"}])
    run_round(3, toolbox, [{"decision": "confirm"}])  # 期望：第 3 轮已无待复核，被自动抑制

    print("\n[demo] 完成。第 3 轮告警应被误报记忆自动抑制，无需人工复核。")


if __name__ == "__main__":
    main()
