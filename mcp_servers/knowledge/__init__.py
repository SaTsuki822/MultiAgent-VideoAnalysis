"""knowledge MCP Server：包装 Qdrant RAG（SOP 检索 + 相似事件检索）。"""

from mcp_servers.knowledge.server import SOP_DOCS, build_server, main, seed_sop

__all__ = ["SOP_DOCS", "build_server", "main", "seed_sop"]
