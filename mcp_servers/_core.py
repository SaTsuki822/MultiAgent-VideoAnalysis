"""自研轻量 MCP 协议核心。

为什么自研而非直接调官方 mcp SDK（对应面试 Q5「MCP 和 Function Calling 区别」）：
- MCP 本质 = JSON-RPC 2.0 + 三个核心方法（initialize / tools/list / tools/call）；
- 手写一遍能下潜到协议细节（消息格式、传输、能力协商、错误码），面试时能讲清「MCP 到底是什么」；
- 工具 handler 与传输解耦，生产可无缝替换为官方 SDK，业务逻辑零改动。

覆盖：
- JSON-RPC 2.0 请求 / 响应 / 错误码（-32700 / -32600 / -32601 / -32602 / -32603）；
- initialize 能力协商（声明 tools 能力）、tools/list 工具发现、tools/call 工具调用；
- 两种传输：stdio（逐行 JSON）与 Streamable HTTP（标准库 http.server 实现）。

协议事实依据（非臆造）：MCP 工具调用返回 {content: [{type:"text", text}], isError}，
tools/list 返回 {tools: [{name, description, inputSchema}]}——均为公开协议规范字段。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

# JSON-RPC 2.0 标准错误码
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

PROTOCOL_VERSION = "2024-11-05"


@dataclass
class Tool:
    """一个可被模型调用的工具。description 是「工具描述工程」的载体。"""
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict], Any]


class MCPServer:
    """协议级 MCP Server：注册工具 + 处理 JSON-RPC 消息。"""

    def __init__(self, name: str, version: str = "1.0.0", instructions: str = ""):
        self.name = name
        self.version = version
        self.instructions = instructions
        self._tools: dict[str, Tool] = {}
        self._initialized = False

    # ---- 工具注册 ----
    def register(self, name: str, description: str, input_schema: dict[str, Any], handler: Callable[[dict], Any]) -> None:
        self._tools[name] = Tool(name=name, description=description, input_schema=input_schema, handler=handler)

    # ---- JSON-RPC 分发 ----
    def handle_rpc(self, message: dict) -> dict | None:
        """处理一条 JSON-RPC 消息；通知（无 id）返回 None，请求返回响应。"""
        try:
            method = message.get("method")
            params = message.get("params", {}) or {}
            rpc_id = message.get("id")
            is_notification = "id" not in message

            if method == "initialize":
                result = self._handle_initialize(params)
            elif method == "notifications/initialized":
                self._initialized = True
                return None  # 通知，不响应
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = self._handle_list_tools()
            elif method == "tools/call":
                result = self._handle_call_tool(params)
            else:
                return self._error(rpc_id, METHOD_NOT_FOUND, f"unknown method: {method}")

            if is_notification:
                return None
            return {"jsonrpc": "2.0", "id": rpc_id, "result": result}
        except Exception as exc:  # 内部错误兜底，避免进程崩溃
            return self._error(message.get("id"), INTERNAL_ERROR, str(exc))

    # ---- 各方法实现 ----
    def _handle_initialize(self, params: dict) -> dict:
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": self.name, "version": self.version},
            "instructions": self.instructions,
        }

    def _handle_list_tools(self) -> dict:
        tools = [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            }
            for t in self._tools.values()
        ]
        return {"tools": tools}

    def _handle_call_tool(self, params: dict) -> dict:
        name = params.get("name", "")
        arguments = params.get("arguments", {}) or {}
        tool = self._tools.get(name)
        if tool is None:
            # 工具不存在按业务错误返回，不抛异常
            return {"content": [{"type": "text", "text": f"unknown tool: {name}"}], "isError": True}
        try:
            result = tool.handler(arguments)
        except Exception as exc:
            return {"content": [{"type": "text", "text": f"tool error: {exc}"}], "isError": True}
        # 统一包装为 MCP content 块
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        return {"content": [{"type": "text", "text": text}], "isError": False}

    @staticmethod
    def _error(rpc_id: Any, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}}

    # ---- 传输 ----
    def run_stdio(self) -> None:
        """stdio 传输：逐行读 JSON，逐行写 JSON。"""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                sys.stdout.write(json.dumps(self._error(None, PARSE_ERROR, "parse error")) + "\n")
                sys.stdout.flush()
                continue
            response = self.handle_rpc(message)
            if response is not None:
                sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                sys.stdout.flush()

    def run_http(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Streamable HTTP 传输（SSE 实现，标准库，无框架依赖）。

        支持两种响应模式：
        1. 客户端请求头带 `Accept: text/event-stream` → 返回 SSE 流式格式；
        2. 客户端不支持 SSE → 返回普通 JSON（向后兼容）。
        """
        from http.server import BaseHTTPRequestHandler, HTTPServer

        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)

                # 判断客户端是否支持 SSE
                accept = self.headers.get("Accept", "")
                supports_sse = "text/event-stream" in accept

                try:
                    message = json.loads(body)
                    response = server.handle_rpc(message)
                except Exception as exc:
                    response = server._error(None, PARSE_ERROR, str(exc))

                if supports_sse and response is not None:
                    # SSE 格式：text/event-stream，data: <json>\n\n
                    payload = json.dumps(response, ensure_ascii=False)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                else:
                    # 普通 JSON 返回（向后兼容）
                    payload = json.dumps(response, ensure_ascii=False).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

            def log_message(self, *args):
                pass  # 静默访问日志

        httpd = HTTPServer((host, port), Handler)
        print(f"[{self.name}] MCP server listening on http://{host}:{port} (Streamable HTTP/SSE)", file=sys.stderr)
        httpd.serve_forever()
