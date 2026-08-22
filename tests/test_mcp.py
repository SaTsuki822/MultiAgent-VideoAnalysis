"""测试 MCP 协议核心：initialize / tools/list / tools/call / 错误码 / 通知。"""

import json

from mcp_servers._core import MCPServer


def test_initialize_negotiates_tools_capability():
    server = MCPServer(name="test")
    resp = server.handle_rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert resp["result"]["serverInfo"]["name"] == "test"
    assert resp["result"]["capabilities"]["tools"] == {}


def test_list_tools():
    server = MCPServer(name="test")
    server.register("echo", "echo tool", {"type": "object"}, lambda args: args.get("x"))
    resp = server.handle_rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = resp["result"]["tools"]
    assert tools[0]["name"] == "echo"
    assert tools[0]["description"] == "echo tool"


def test_call_tool_returns_content_block():
    server = MCPServer(name="test")
    server.register("echo", "echo", {"type": "object"}, lambda args: {"got": args.get("x")})
    resp = server.handle_rpc(
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "echo", "arguments": {"x": 1}}}
    )
    assert resp["result"]["isError"] is False
    content = json.loads(resp["result"]["content"][0]["text"])
    assert content["got"] == 1


def test_call_tool_unknown_tool_is_business_error():
    server = MCPServer(name="test")
    resp = server.handle_rpc(
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "nope", "arguments": {}}}
    )
    assert resp["result"]["isError"] is True


def test_unknown_method_returns_method_not_found():
    server = MCPServer(name="test")
    resp = server.handle_rpc({"jsonrpc": "2.0", "id": 5, "method": "nope"})
    assert resp["error"]["code"] == -32601


def test_notification_returns_none():
    server = MCPServer(name="test")
    resp = server.handle_rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert resp is None
