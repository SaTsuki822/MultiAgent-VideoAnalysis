"""ticket：工单创建与通知。

工具：create_ticket / query_tickets / update_ticket / notify。
- create_ticket / query_tickets / update_ticket：内存存储 + 线程锁 + JSON 文件持久化（HTTP 多线程安全）；
- notify：优先真实 webhook 调用，失败降级为 mock 记录。
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path

import httpx

from mcp_servers._core import MCPServer

# ---- 内存存储 + 线程锁 ----
_STORE: dict[str, dict] = {}
_LOCK = threading.Lock()
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_TICKET_FILE = _DATA_DIR / "tickets.json"
_NOTIFY_LOG = _DATA_DIR / "notify_log.jsonl"


def _load() -> None:
    """从 JSON 文件加载已持久化的工单。"""
    global _STORE
    if _TICKET_FILE.exists():
        try:
            with open(_TICKET_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            with _LOCK:
                _STORE = {t["ticket_id"]: t for t in data.get("tickets", [])}
        except Exception:
            pass


def _save() -> None:
    """将当前工单写入 JSON 文件（幂等）。"""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        snapshot = list(_STORE.values())
    try:
        with open(_TICKET_FILE, "w", encoding="utf-8") as f:
            json.dump({"tickets": snapshot}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _append_notify_log(record: dict) -> None:
    """追加通知记录到 JSONL。"""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(_NOTIFY_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


# 启动时加载历史数据
_load()


def build_server() -> MCPServer:
    server = MCPServer(name="ticket", version="1.1.0", instructions="工单创建、查询、更新与通知分发")

    server.register(
        name="create_ticket",
        description=(
            "为一条已确认告警创建工单。参数 alarm_id、assignee（处理人）、"
            "priority（可选，默认 medium）、description（可选）。"
            "返回工单对象，含 ticket_id，并自动持久化到 data/tickets.json。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "alarm_id": {"type": "string"},
                "assignee": {"type": "string"},
                "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                "description": {"type": "string"},
            },
            "required": ["alarm_id", "assignee"],
        },
        handler=lambda args: _create(args),
    )

    server.register(
        name="query_tickets",
        description=(
            "查询工单列表。支持按 alarm_id、assignee、status 过滤；"
            "都不传则返回全部。返回工单数组。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "alarm_id": {"type": "string"},
                "assignee": {"type": "string"},
                "status": {"type": "string", "enum": ["open", "in_progress", "resolved", "closed"]},
            },
        },
        handler=lambda args: _query(args),
    )

    server.register(
        name="update_ticket",
        description=(
            "更新工单状态或备注。参数 ticket_id 必填，status 和 note 可选。"
            "返回更新后的工单。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string"},
                "status": {"type": "string", "enum": ["open", "in_progress", "resolved", "closed"]},
                "note": {"type": "string"},
            },
            "required": ["ticket_id"],
        },
        handler=lambda args: _update(args),
    )

    server.register(
        name="notify",
        description=(
            "发送通知。优先尝试真实 webhook 调用（channel 为完整 URL 时）；"
            "失败或 channel 不是 URL 时，降级为本地记录（写入 data/notify_log.jsonl）。"
            "参数 channel（webhook URL 或通道标识）、payload（任意 JSON）。"
        ),
        input_schema={
            "type": "object",
            "properties": {"channel": {"type": "string"}, "payload": {"type": "object"}},
            "required": ["channel", "payload"],
        },
        handler=lambda args: _notify(args),
    )

    return server


def _create(args: dict) -> dict:
    ticket = {
        "ticket_id": f"tk_{uuid.uuid4().hex[:12]}",
        "alarm_id": args["alarm_id"],
        "assignee": args["assignee"],
        "priority": args.get("priority", "medium"),
        "description": args.get("description", ""),
        "status": "open",
        "created_at": _now(),
        "updated_at": _now(),
    }
    with _LOCK:
        _STORE[ticket["ticket_id"]] = ticket
    _save()
    return ticket


def _query(args: dict) -> dict:
    alarm_id = args.get("alarm_id")
    assignee = args.get("assignee")
    status = args.get("status")
    with _LOCK:
        results = [
            t
            for t in _STORE.values()
            if (alarm_id is None or t["alarm_id"] == alarm_id)
            and (assignee is None or t["assignee"] == assignee)
            and (status is None or t["status"] == status)
        ]
    return {"tickets": results, "total": len(results)}


def _update(args: dict) -> dict:
    ticket_id = args["ticket_id"]
    with _LOCK:
        ticket = _STORE.get(ticket_id)
        if ticket is None:
            raise ValueError(f"ticket not found: {ticket_id}")
        if "status" in args:
            ticket["status"] = args["status"]
        if "note" in args:
            ticket["description"] = ticket.get("description", "") + f"\n[更新] {args['note']}"
        ticket["updated_at"] = _now()
        _STORE[ticket_id] = ticket
    _save()
    return ticket


def _notify(args: dict) -> dict:
    channel = args["channel"]
    payload = args["payload"]
    record = {
        "id": f"ntf_{uuid.uuid4().hex[:8]}",
        "channel": channel,
        "payload": payload,
        "timestamp": _now(),
    }

    # 如果 channel 是合法 URL，尝试真实 webhook 调用
    if channel.startswith("http://") or channel.startswith("https://"):
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(channel, json=payload)
                resp.raise_for_status()
            record["sent"] = True
            record["mode"] = "webhook"
            record["http_status"] = resp.status_code
            _append_notify_log(record)
            return {
                "sent": True,
                "mode": "webhook",
                "channel": channel,
                "http_status": resp.status_code,
            }
        except Exception as exc:
            # 真实发送失败，降级为本地记录
            record["sent"] = False
            record["mode"] = "local_log"
            record["error"] = str(exc)
            _append_notify_log(record)
            return {
                "sent": False,
                "mode": "local_log",
                "channel": channel,
                "error": str(exc),
                "note": "已写入 data/notify_log.jsonl，生产环境请配置真实 webhook",
            }

    # 非 URL 通道，直接本地记录
    record["sent"] = False
    record["mode"] = "local_log"
    _append_notify_log(record)
    return {
        "sent": False,
        "mode": "local_log",
        "channel": channel,
        "note": "已写入 data/notify_log.jsonl，生产环境请配置真实 webhook",
    }


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    build_server().run_stdio()


if __name__ == "__main__":
    main()
