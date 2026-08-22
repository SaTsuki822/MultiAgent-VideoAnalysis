"""ticket：工单创建与通知。

工具：create_ticket / notify。
notify 为 mock（不真实发送），返回「发送目标 + payload」，生产接真实 webhook / 邮件。
"""

from __future__ import annotations

import uuid

from mcp_servers._core import MCPServer


def build_server() -> MCPServer:
    server = MCPServer(name="ticket", version="1.0.0", instructions="工单创建与通知分发")

    server.register(
        name="create_ticket",
        description=(
            "为一条已确认告警创建工单。参数 alarm_id、assignee（处理人）、priority（可选，默认 medium）。"
            "返回工单对象，含 ticket_id。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "alarm_id": {"type": "string"},
                "assignee": {"type": "string"},
                "priority": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": ["alarm_id", "assignee"],
        },
        handler=lambda args: {
            "ticket_id": f"tk_{uuid.uuid4().hex[:12]}",
            "alarm_id": args["alarm_id"],
            "assignee": args["assignee"],
            "priority": args.get("priority", "medium"),
            "status": "open",
        },
    )

    server.register(
        name="notify",
        description=(
            "发送通知（mock）。参数 channel（webhook/email 等）、payload（任意 JSON）。"
            "返回发送结果说明，不真实发送——生产接真实渠道。"
        ),
        input_schema={
            "type": "object",
            "properties": {"channel": {"type": "string"}, "payload": {"type": "object"}},
            "required": ["channel", "payload"],
        },
        handler=lambda args: {
            "sent": True,
            "channel": args["channel"],
            "message": "mock notify: 生产环境将真实投递",
            "payload": args["payload"],
        },
    )

    return server


def main() -> None:
    build_server().run_stdio()


if __name__ == "__main__":
    main()
