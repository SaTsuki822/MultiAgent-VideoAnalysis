"""MCP（Model Context Protocol）工具层。

自研轻量协议核心（_core.py）+ 5 个独立工具 Server + 客户端封装。
每个 Server 都可通过 stdio / Streamable HTTP 两种传输独立运行。
"""

from mcp_servers._core import MCPServer, Tool

__all__ = ["MCPServer", "Tool"]
