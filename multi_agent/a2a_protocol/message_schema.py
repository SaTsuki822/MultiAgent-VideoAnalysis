"""A2A 协议消息格式定义。

多 Agent 之间通信的统一消息 schema，支持同步 HTTP/gRPC 和异步 Redis Stream。
所有消息必须包含 message_id（用于 Trace 追踪和幂等去重）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class A2AMessage:
    """Agent 间标准消息格式。"""

    message_id: str
    from_agent: str          # 发送方 Agent 标识，如 "orchestrator"
    to_agent: str            # 接收方 Agent 标识，如 "perception"
    task: str                # 任务类型，如 "analyze_clip"
    payload: dict[str, Any] = field(default_factory=dict)
    callback_url: str | None = None   # 异步回调地址
    timeout_sec: int = 120             # 任务超时时间
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "from": self.from_agent,
            "to": self.to_agent,
            "task": self.task,
            "payload": self.payload,
            "callback_url": self.callback_url,
            "timeout_sec": self.timeout_sec,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "A2AMessage":
        return cls(
            message_id=data["message_id"],
            from_agent=data["from"],
            to_agent=data["to"],
            task=data["task"],
            payload=data.get("payload", {}),
            callback_url=data.get("callback_url"),
            timeout_sec=data.get("timeout_sec", 120),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
        )


@dataclass
class A2AResult:
    """Agent 任务返回结果。"""

    message_id: str          # 对应请求的 message_id
    from_agent: str
    to_agent: str
    status: str              # "success" | "error" | "timeout"
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "from": self.from_agent,
            "to": self.to_agent,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "A2AResult":
        return cls(
            message_id=data["message_id"],
            from_agent=data["from"],
            to_agent=data["to"],
            status=data["status"],
            result=data.get("result", {}),
            error=data.get("error"),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
        )
