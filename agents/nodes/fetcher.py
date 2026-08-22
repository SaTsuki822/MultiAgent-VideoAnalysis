"""fetch 节点：从 camera-registry 拉取待分析视频片段元数据。"""

from __future__ import annotations

from datetime import datetime, timedelta

from agents.models import Clip, LogEntry
from agents.toolbox import Toolbox


def fetcher_node(state: dict, toolbox: Toolbox, start_time: str | None = None, end_time: str | None = None) -> dict:
    """为每路摄像头拉取巡检时间段内的片段元数据。

    默认时间段为「今天 08:00-09:00」（demo 用）；真实场景由调度层传入具体巡检窗口。
    """
    cameras = state.get("cameras", []) or toolbox.list_cameras()

    if start_time is None:
        now = datetime.now()
        start_time = now.replace(hour=8, minute=0, second=0, microsecond=0).isoformat()
        end_time = (now.replace(hour=9, minute=0, second=0, microsecond=0)).isoformat()

    clips: list[Clip] = []
    for cam in cameras:
        meta = toolbox.get_clip(cam["id"], start_time, end_time)
        clips.append(
            Clip(
                camera_id=cam["id"],
                path=meta["clip_path"],
                start_time=datetime.fromisoformat(meta["start_time"]),
                end_time=datetime.fromisoformat(meta["end_time"]),
                duration_seconds=meta["duration_seconds"],
            )
        )

    log = LogEntry(node="fetcher", message=f"拉取 {len(clips)} 段视频片段元数据（{start_time} ~ {end_time}）")
    return {"clips": clips, "logs": [log]}
