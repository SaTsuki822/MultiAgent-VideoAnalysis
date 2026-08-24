"""Function Calling 接入单测。

覆盖：mcp→openai schema 转换、mock 默认不支持、_react_compile 分发路由、
      tool-call 循环端到端编译、文本 ReAct 路径回归、LLM 异常回退。
"""

from __future__ import annotations

import pytest

from agents.llm import LLMClient, MockLLMClient, ToolCall, ToolCallResponse, mcp_tools_to_openai
from agents.models import Rule
from agents.nodes.planner import _react_compile


def _rule() -> Rule:
    # 未命中关键词表的陌生规则，才会走 ReAct 编译路径
    return Rule(id="rule_x", name="高空坠物", description="高处有物体掉落风险", severity="high")


class FakeToolCallingLLM(LLMClient):
    """按脚本返回 tool_call / 最终答案的假客户端，supports_tool_calling=True。"""

    supports_tool_calling = True

    def __init__(self, script):
        self.script = list(script)
        self.messages_seen: list[list[dict]] = []

    def complete(self, system, user, json_mode=False):
        return "{}"

    def complete_vision(self, system, user, images_b64, json_mode=False):
        return "{}"

    def complete_with_tools(self, messages, tools, json_mode=False):
        self.messages_seen.append(list(messages))
        return self.script.pop(0)


class TextOnlyLLM(LLMClient):
    """不支持工具调用的假客户端（模拟旧文本 ReAct 路径）。"""

    supports_tool_calling = False

    def __init__(self):
        self.calls = 0

    def complete(self, system, user, json_mode=False):
        self.calls += 1
        return 'Final Answer: {"target_objects": ["person"], "evidence_hints": ["越界"]}'

    def complete_vision(self, system, user, images_b64, json_mode=False):
        return "{}"


class FailingToolCallingLLM(FakeToolCallingLLM):
    def complete_with_tools(self, messages, tools, json_mode=False):
        raise RuntimeError("network down")


def test_mcp_tools_to_openai_shape():
    mcp_tools = [
        {
            "name": "search_sop",
            "description": "检索 SOP",
            "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        }
    ]
    openai_tools = mcp_tools_to_openai(mcp_tools)
    fn = openai_tools[0]["function"]
    assert openai_tools[0]["type"] == "function"
    assert fn["name"] == "search_sop"
    assert fn["description"] == "检索 SOP"
    assert fn["parameters"]["required"] == ["query"]


def test_mock_client_does_not_support_tool_calling():
    assert MockLLMClient.supports_tool_calling is False
    with pytest.raises(NotImplementedError):
        MockLLMClient().complete_with_tools([], [])


def test_react_compile_uses_function_calling_when_supported(fake_toolbox):
    llm = FakeToolCallingLLM(
        [
            ToolCallResponse(tool_calls=[ToolCall(id="c1", name="search_sop", arguments={"query": "高空"})]),
            ToolCallResponse(content='{"target_objects": ["object"], "evidence_hints": ["掉落"], "duration_threshold_seconds": null}'),
        ]
    )
    compiled = _react_compile(_rule(), llm, fake_toolbox)
    assert len(llm.messages_seen) == 2
    assert compiled.target_objects == ["object"]
    assert compiled.evidence_hints == ["掉落"]


def test_react_compile_builds_tool_result_messages(fake_toolbox):
    llm = FakeToolCallingLLM(
        [
            ToolCallResponse(tool_calls=[ToolCall(id="c1", name="search_sop", arguments={"query": "x"})]),
            ToolCallResponse(tool_calls=[ToolCall(id="c2", name="search_sop", arguments={"query": "y"})]),
            ToolCallResponse(content='{"target_objects": ["a"], "evidence_hints": ["b"]}'),
        ]
    )
    compiled = _react_compile(_rule(), llm, fake_toolbox)
    assert len(llm.messages_seen) == 3
    # 第二轮应包含 assistant.tool_calls 与 tool 角色结果
    second_roles = [m["role"] for m in llm.messages_seen[1]]
    assert "assistant" in second_roles and "tool" in second_roles
    assert compiled.target_objects == ["a"]


def test_react_compile_text_path_still_works(fake_toolbox):
    llm = TextOnlyLLM()
    compiled = _react_compile(_rule(), llm, fake_toolbox)
    assert llm.calls == 1
    assert compiled.target_objects == ["person"]


def test_react_compile_falls_back_on_llm_error(fake_toolbox):
    llm = FailingToolCallingLLM([])
    compiled = _react_compile(_rule(), llm, fake_toolbox)
    # 未命中关键词表 → _deterministic_compile 退化为规则名本身
    assert compiled.target_objects == ["高空坠物"]
