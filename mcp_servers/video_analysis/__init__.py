"""video-analysis MCP Server：封装 VLM 推理，输入片段 + 巡检项，输出结构化结果。"""

from mcp_servers.video_analysis.server import build_server, main

__all__ = ["build_server", "main"]
