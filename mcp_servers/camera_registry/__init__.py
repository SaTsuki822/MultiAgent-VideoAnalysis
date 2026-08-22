"""camera-registry MCP Server：摄像头台账（mock 10 路，预留真实 NVR 接入）。"""

from mcp_servers.camera_registry.server import MOCK_CAMERAS, build_server, main

__all__ = ["MOCK_CAMERAS", "build_server", "main"]
