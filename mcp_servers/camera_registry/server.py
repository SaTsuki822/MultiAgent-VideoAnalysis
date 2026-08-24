"""camera-registry：摄像头台账管理。

工具：list_cameras / get_clip / get_metadata / update_camera。
支持从本地 JSON 文件加载真实台账（`data/cameras.json`），不存在时回退到 mock 数据并自动保存。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from mcp_servers._core import MCPServer

# mock 台账：10 路摄像头覆盖工地典型点位
MOCK_CAMERAS: list[dict] = [
    {"id": "cam_001", "name": "东门入口", "area": "gate_east", "rtsp_url": "rtsp://mock/nvr/cam_001"},
    {"id": "cam_002", "name": "西门入口", "area": "gate_west", "rtsp_url": "rtsp://mock/nvr/cam_002"},
    {"id": "cam_003", "name": "1号楼基坑", "area": "zone_a_excavation", "rtsp_url": "rtsp://mock/nvr/cam_003"},
    {"id": "cam_004", "name": "2号楼塔吊", "area": "zone_b_crane", "rtsp_url": "rtsp://mock/nvr/cam_004"},
    {"id": "cam_005", "name": "材料堆放区", "area": "storage_area", "rtsp_url": "rtsp://mock/nvr/cam_005"},
    {"id": "cam_006", "name": "钢筋加工棚", "area": "rebar_workshop", "rtsp_url": "rtsp://mock/nvr/cam_006"},
    {"id": "cam_007", "name": "临时用电区", "area": "power_area", "rtsp_url": "rtsp://mock/nvr/cam_007"},
    {"id": "cam_008", "name": "生活区宿舍", "area": "dormitory", "rtsp_url": "rtsp://mock/nvr/cam_008"},
    {"id": "cam_009", "name": "围墙周界", "area": "perimeter", "rtsp_url": "rtsp://mock/nvr/cam_009"},
    {"id": "cam_010", "name": "消防通道", "area": "fire_lane", "rtsp_url": "rtsp://mock/nvr/cam_010"},
]

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_CAMERA_FILE = _DATA_DIR / "cameras.json"


def _load_cameras() -> list[dict]:
    """优先从本地 JSON 加载，不存在则回退 mock 并自动保存。"""
    if _CAMERA_FILE.exists():
        try:
            with open(_CAMERA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            cameras = data.get("cameras", [])
            if cameras:
                return cameras
        except Exception:
            pass
    # 回退 mock 并持久化
    _save_cameras(MOCK_CAMERAS)
    return list(MOCK_CAMERAS)


def _save_cameras(cameras: list[dict]) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(_CAMERA_FILE, "w", encoding="utf-8") as f:
            json.dump({"cameras": cameras}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# 启动时加载
_CAMERAS: list[dict] = _load_cameras()


def _find(camera_id: str) -> dict | None:
    return next((c for c in _CAMERAS if c["id"] == camera_id), None)


def build_server() -> MCPServer:
    server = MCPServer(name="camera-registry", version="1.1.0", instructions="工地摄像头台账查询与更新")

    server.register(
        name="list_cameras",
        description="列出所有可用摄像头及其元数据。当需要确定巡检范围、按区域筛选摄像头时调用。无参数。",
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=lambda args: {"cameras": _CAMERAS},
    )

    server.register(
        name="get_metadata",
        description="查询单个摄像头的元数据（区域、RTSP 地址等）。参数 camera_id 必填；摄像头不存在时返回 isError。",
        input_schema={
            "type": "object",
            "properties": {"camera_id": {"type": "string", "description": "摄像头 ID，如 cam_001"}},
            "required": ["camera_id"],
        },
        handler=lambda args: _find(args["camera_id"]) or (_raise(f"camera not found: {args['camera_id']}")),
    )

    server.register(
        name="get_clip",
        description=(
            "获取指定摄像头在某个时间段内的视频片段元数据。start_time/end_time 为 ISO 格式字符串；"
            "返回片段路径与时长。用于巡检任务拉取待分析片段。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "camera_id": {"type": "string"},
                "start_time": {"type": "string", "description": "ISO 时间，如 2026-08-17T08:00:00"},
                "end_time": {"type": "string"},
            },
            "required": ["camera_id", "start_time", "end_time"],
        },
        handler=lambda args: _get_clip(args),
    )

    server.register(
        name="update_camera",
        description=(
            "更新或新增摄像头台账。参数 camera_id 必填，name / area / rtsp_url 可选；"
            "camera_id 已存在则更新，不存在则新增。返回更新后的摄像头对象。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "camera_id": {"type": "string"},
                "name": {"type": "string"},
                "area": {"type": "string"},
                "rtsp_url": {"type": "string"},
            },
            "required": ["camera_id"],
        },
        handler=lambda args: _update_camera(args),
    )

    return server


def _raise(msg: str) -> None:
    raise ValueError(msg)


def _get_clip(args: dict) -> dict:
    camera = _find(args["camera_id"])
    if camera is None:
        raise ValueError(f"camera not found: {args['camera_id']}")
    start = datetime.fromisoformat(args["start_time"])
    end = datetime.fromisoformat(args["end_time"])
    duration = (end - start).total_seconds()
    return {
        "camera_id": camera["id"],
        "clip_path": f"data/clips/{camera['id']}.mp4",
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "duration_seconds": duration,
    }


def _update_camera(args: dict) -> dict:
    camera_id = args["camera_id"]
    existing = _find(camera_id)
    if existing:
        existing.update({k: v for k, v in args.items() if k != "camera_id" and v is not None})
        camera = existing
    else:
        camera = {
            "id": camera_id,
            "name": args.get("name", camera_id),
            "area": args.get("area", "unknown"),
            "rtsp_url": args.get("rtsp_url", ""),
        }
        _CAMERAS.append(camera)
    _save_cameras(_CAMERAS)
    return camera


def main() -> None:
    build_server().run_stdio()


if __name__ == "__main__":
    main()
