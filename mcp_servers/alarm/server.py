"""alarm：告警生命周期管理。

工具：create_alarm / query_history / suppress_similar。
内存存储 + 线程锁（HTTP 传输多线程安全）。生产接数据库 / 消息队列，工具 schema 不变。
"""

from __future__ import annotations

import threading
import uuid

from mcp_servers._core import MCPServer

_STORE: dict[str, dict] = {}
_LOCK = threading.Lock()


def build_server() -> MCPServer:
    server = MCPServer(name="alarm", version="1.0.0", instructions="告警创建、查询与相似抑制")

    server.register(
        name="create_alarm",
        description=(
            "创建一条告警。参数：camera_id、rule_id、rule_name、severity（high/medium/low）、"
            "confidence（0~1）、evidence（关键帧证据列表）。返回含 alarm_id 的完整告警。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "camera_id": {"type": "string"},
                "rule_id": {"type": "string"},
                "rule_name": {"type": "string"},
                "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                "confidence": {"type": "number"},
                "evidence": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["camera_id", "rule_id", "rule_name", "severity", "confidence"],
        },
        handler=lambda args: _create(args),
    )

    server.register(
        name="query_history",
        description=(
            "按摄像头与规则查询历史告警，用于判断是否重复告警。"
            "参数 camera_id、rule_id 均可选；都不传则返回全部。"
        ),
        input_schema={
            "type": "object",
            "properties": {"camera_id": {"type": "string"}, "rule_id": {"type": "string"}},
        },
        handler=lambda args: _query(args),
    )

    server.register(
        name="suppress_similar",
        description=(
            "把指定告警标记为 suppressed 并记录原因（对应误报记忆的「只降权不删除」）。"
            "参数 alarm_id 必填、reason 建议填写。返回更新后的告警。"
        ),
        input_schema={
            "type": "object",
            "properties": {"alarm_id": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["alarm_id"],
        },
        handler=lambda args: _suppress(args),
    )

    return server


def _create(args: dict) -> dict:
    alarm = {
        "id": f"alarm_{uuid.uuid4().hex[:12]}",
        "camera_id": args["camera_id"],
        "rule_id": args["rule_id"],
        "rule_name": args["rule_name"],
        "severity": args["severity"],
        "confidence": args.get("confidence", 0.0),
        "evidence": args.get("evidence", []),
        "status": "pending_review",
        "suppressed": False,
        "suppression_reason": "",
    }
    with _LOCK:
        _STORE[alarm["id"]] = alarm
    return alarm


def _query(args: dict) -> dict:
    camera_id = args.get("camera_id")
    rule_id = args.get("rule_id")
    with _LOCK:
        results = [
            a
            for a in _STORE.values()
            if (camera_id is None or a["camera_id"] == camera_id)
            and (rule_id is None or a["rule_id"] == rule_id)
        ]
    return {"alarms": results}


def _suppress(args: dict) -> dict:
    with _LOCK:
        alarm = _STORE.get(args["alarm_id"])
        if alarm is None:
            raise ValueError(f"alarm not found: {args['alarm_id']}")
        alarm["status"] = "suppressed"
        alarm["suppressed"] = True
        alarm["suppression_reason"] = args.get("reason", "")
        return alarm


def main() -> None:
    build_server().run_stdio()


if __name__ == "__main__":
    main()
