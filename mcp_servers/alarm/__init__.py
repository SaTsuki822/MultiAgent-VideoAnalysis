"""alarm MCP Server：告警生命周期管理（创建 / 查询 / 相似抑制）。"""

from mcp_servers.alarm.server import build_server, main

__all__ = ["build_server", "main"]
