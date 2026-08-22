"""ticket MCP Server：工单 + 通知（webhook/邮件 mock）。"""

from mcp_servers.ticket.server import build_server, main

__all__ = ["build_server", "main"]
