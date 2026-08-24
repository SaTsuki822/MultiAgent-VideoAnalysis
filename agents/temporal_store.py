"""活跃异常存储层：内存 + 线程锁 + JSON 持久化。

为「持续型异常」的片段级状态机提供状态存储：
- 以 (camera_id, rule_id) 为唯一键，跟踪异常的生命周期；
- 支持内存查询、更新、关闭、列表；
- 自动持久化到 data/ongoing_anomalies.json（与 camera_registry / ticket 等保持一致）。

生产可替换为 Redis / Postgres，接口不变。
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from agents.models import FrameEvidence, OngoingAnomaly

# 内存存储
_STORE: dict[str, OngoingAnomaly] = {}
_LOCK = threading.Lock()

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_ONGOING_FILE = _DATA_DIR / "ongoing_anomalies.json"


def _key(camera_id: str, rule_id: str) -> str:
    return f"{camera_id}:{rule_id}"


def _serialize(obj: OngoingAnomaly) -> dict:
    """Pydantic model → 可 JSON 序列化的 dict（datetime 转 ISO 字符串）。"""
    data = obj.model_dump()
    for field in ("first_seen_at", "last_seen_at", "updated_at"):
        v = data.get(field)
        if isinstance(v, datetime):
            data[field] = v.isoformat()
    return data


def _deserialize(data: dict) -> OngoingAnomaly:
    """dict → OngoingAnomaly（ISO 字符串 转 datetime）。"""
    for field in ("first_seen_at", "last_seen_at", "updated_at"):
        v = data.get(field)
        if isinstance(v, str):
            data[field] = datetime.fromisoformat(v)
    return OngoingAnomaly(**data)


def _load() -> None:
    """启动时从 JSON 文件加载。"""
    global _STORE
    if not _ONGOING_FILE.exists():
        return
    try:
        with open(_ONGOING_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for item in raw.get("anomalies", []):
            anomaly = _deserialize(item)
            _STORE[_key(anomaly.camera_id, anomaly.rule_id)] = anomaly
    except Exception:
        pass


def _save() -> None:
    """持久化到 JSON 文件。"""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with _LOCK:
            data = {"anomalies": [_serialize(a) for a in _STORE.values()]}
        with open(_ONGOING_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# 模块加载时自动恢复
_load()


# ============================================================
# 公开 API
# ============================================================

def get(camera_id: str, rule_id: str) -> OngoingAnomaly | None:
    """查询指定 (camera_id, rule_id) 的活跃异常。"""
    with _LOCK:
        return _STORE.get(_key(camera_id, rule_id))


def put(anomaly: OngoingAnomaly) -> OngoingAnomaly:
    """创建或更新活跃异常，并持久化。"""
    with _LOCK:
        anomaly.updated_at = datetime.now()
        _STORE[_key(anomaly.camera_id, anomaly.rule_id)] = anomaly
    _save()
    return anomaly


def close(camera_id: str, rule_id: str, reason: str = "") -> OngoingAnomaly | None:
    """关闭指定活跃异常。"""
    with _LOCK:
        anomaly = _STORE.get(_key(camera_id, rule_id))
        if anomaly is None:
            return None
        anomaly.state = "closed"
        anomaly.updated_at = datetime.now()
        _save()
        return anomaly


def list_all(
    camera_id: str | None = None,
    rule_id: str | None = None,
    state: str | None = None,
) -> list[OngoingAnomaly]:
    """列出活跃异常，支持过滤。"""
    with _LOCK:
        results = list(_STORE.values())
    if camera_id:
        results = [a for a in results if a.camera_id == camera_id]
    if rule_id:
        results = [a for a in results if a.rule_id == rule_id]
    if state:
        results = [a for a in results if a.state == state]
    return results


def reset() -> None:
    """清空全部活跃异常（单测 / 调试用）。"""
    global _STORE
    with _LOCK:
        _STORE = {}
    _save()
