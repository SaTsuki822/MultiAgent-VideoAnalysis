"""pytest 共享 fixture。

FakeToolbox：注入到节点的假工具层，让 workflow 编排逻辑在无视频 / 无 MCP 进程 /
无 GPU 的情况下单测。这正体现了「节点依赖 Toolbox 接口而非具体实现」的设计价值。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from agents.models import Rule


class FakeToolbox:
    """模拟 5 个 MCP Server 的假工具层，analyze_clip 返回预设命中（不读真实视频）。"""

    def __init__(self, camera_id: str = "cam_test"):
        self.camera_id = camera_id
        self.counter = 0

    def list_cameras(self):
        return [{"id": self.camera_id, "name": "测试摄像头", "area": "test", "rtsp_url": ""}]

    def get_clip(self, camera_id, start, end):
        return {
            "clip_path": f"data/clips/{camera_id}.mp4",
            "start_time": "2026-08-17T08:00:00",
            "end_time": "2026-08-17T09:00:00",
            "duration_seconds": 3600.0,
        }

    def analyze_clip(self, clip_path, camera_id, rule):
        rule_id = rule["id"]
        task_id = f"task_{camera_id}_{rule_id}"
        return {
            "finding": {
                "id": f"finding_{task_id}",
                "task_id": task_id,
                "rule_id": rule_id,
                "camera_id": camera_id,
                "hit": True,
                "confidence": 0.9,
                "evidence": [
                    {"frame_index": 1, "timestamp_seconds": 1.0, "description": "x"},
                    {"frame_index": 2, "timestamp_seconds": 2.0, "description": "x"},
                    {"frame_index": 3, "timestamp_seconds": 3.0, "description": "x"},
                ],
                "hit_frame_indices": [1, 2, 3],
            },
            "cost": {},
        }

    def search_sop(self, query, limit=3):
        return []

    def search_similar_events(self, query, limit=3):
        return []

    def create_alarm(self, camera_id, rule_id, rule_name, severity, confidence, evidence):
        self.counter += 1
        return {
            "id": f"alarm_{self.counter}",
            "camera_id": camera_id,
            "rule_id": rule_id,
            "rule_name": rule_name,
            "severity": severity,
            "confidence": confidence,
            "evidence": evidence,
            "status": "pending_review",
        }

    def suppress_alarm(self, alarm_id, reason):
        return {}

    def create_ticket(self, alarm_id, assignee):
        return {"ticket_id": "tk_test"}

    def notify(self, channel, payload):
        return {"sent": True}


@pytest.fixture
def fake_toolbox():
    return FakeToolbox()


@pytest.fixture
def medium_rule():
    # medium 级别：可被误报记忆抑制（high 会触发「永不抑制」白名单）
    return Rule(id="rule_intrusion", name="区域入侵", description="非工作时间有人进入危险区域", severity="medium")


@pytest.fixture
def camera():
    return {"id": "cam_test", "name": "测试摄像头", "area": "test", "rtsp_url": ""}
