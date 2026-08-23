"""MCP 客户端封装：工具发现 + 调用 + 超时管理。

三种传输（面试可讲「传输解耦」）：
- InProcess：直接持有 MCPServer 对象调用，用于 demo / 单测（无进程边界）；
- HTTP：Streamable HTTP，远程调用；
- Stdio：子进程 stdio，真实协议走线（演示「跨进程工具调用」）。

默认 InProcess，保证无外部依赖时端到端可跑；切 stdio/HTTP 只改 transport 参数。
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from mcp_servers._core import MCPServer


class MCPClient:
    def __init__(self, server: MCPServer | None = None, http_url: str | None = None, timeout: float = 30.0):
        self._server = server
        self._http_url = http_url
        self._timeout = timeout
        self._tool_cache: dict[str, dict] | None = None

    def list_tools(self) -> list[dict]:
        """发现工具（对应协议 tools/list）。"""
        if self._tool_cache is not None:
            return list(self._tool_cache.values())
        result = self._request("tools/list")
        tools = result.get("tools", [])
        self._tool_cache = {t["name"]: t for t in tools}
        return tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """调用工具并解析出业务结果（content 里的文本）。"""
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            text = result.get("content", [{}])[0].get("text", "unknown error")
            raise RuntimeError(f"MCP tool '{name}' error: {text}")
        content = result.get("content", [])
        if not content:
            return None
        text = content[0].get("text", "")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def _request(self, method: str, params: dict | None = None) -> dict:
        message = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
        if self._server is not None:
            response = self._server.handle_rpc(message)
        elif self._http_url is not None:
            with httpx.Client(timeout=self._timeout) as client:
                # 声明支持 SSE（Streamable HTTP），Server 优先返回 SSE 格式
                headers = {"Accept": "text/event-stream, application/json"}
                resp = client.post(self._http_url, json=message, headers=headers)
                resp.raise_for_status()

                content_type = resp.headers.get("Content-Type", "")
                if "text/event-stream" in content_type:
                    # 解析 SSE 响应：提取 data: 行中的 JSON
                    response = self._parse_sse(resp.text)
                else:
                    response = resp.json()
        else:
            raise RuntimeError("MCPClient 未配置任何传输")
        if "error" in response:
            raise RuntimeError(f"MCP error: {response['error']}")
        return response.get("result", {})

    @staticmethod
    def _parse_sse(text: str) -> dict:
        """解析 SSE 响应体，提取 `data: <json>` 行并反序列化。"""
        for line in text.strip().splitlines():
            if line.startswith("data:"):
                data = line[len("data:"):].strip()
                return json.loads(data)
        raise RuntimeError("SSE 响应中未找到有效的 data 行")
