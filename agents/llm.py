"""LLM / VLM 客户端抽象。

设计原则（对应面试 Q5/Q7 的「接口解耦」思路）：
- 三段式：接口（ABC）+ 真实实现（OpenAICompatibleClient）+ mock 兜底（MockLLMClient）；
- 所有需要模型推理的节点（planner / verify / 误报签名抽取 / 报告）都依赖 LLMClient 接口，
  因此无 API key / 无 GPU 时整套流程仍可离线端到端跑通（由各语义函数走确定性 fallback）。

注意：MockLLMClient 只作为「安全网」，返回显式标记；真正的确定性逻辑分布在各语义函数里
（compile_rule / verify / extract_fp_signature 等），它们通过 isinstance 判断是否 mock，
避免 mock 客户端里塞一堆与业务耦合的启发式。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import httpx

from agents.config import Settings, get_settings


@dataclass
class ToolCall:
    """一次原生 Function Calling 的工具调用（归一化后的形态）。"""

    id: str
    name: str
    arguments: dict


@dataclass
class ToolCallResponse:
    """complete_with_tools 的返回：最终文本 + 模型要调用的工具。"""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


def mcp_tools_to_openai(tools: list[dict]) -> list[dict]:
    """把 MCP tools/list 的结果转换为 OpenAI Function Calling 的 tools 参数。

    MCP  : {"name", "description", "inputSchema"}
    OpenAI: {"type": "function", "function": {"name", "description", "parameters"}}

    二者字段几乎一一对应——这正是「MCP 暴露的工具能无缝对接 Function Calling」的体现。
    """
    result = []
    for t in tools:
        result.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("inputSchema", t.get("input_schema", {"type": "object", "properties": {}})),
                },
            }
        )
    return result


class LLMClient(ABC):
    """文本 / 多模态推理的统一抽象。"""

    # 是否支持原生 Function Calling（真实 OpenAI 兼容客户端置 True，mock 默认 False）
    supports_tool_calling: bool = False

    @abstractmethod
    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        """纯文本补全，返回字符串（json_mode 时返回 JSON 文本，由调用方解析）。"""

    @abstractmethod
    def complete_vision(self, system: str, user: str, images_b64: list[str], json_mode: bool = False) -> str:
        """带图补全，images_b64 为 JPEG 的 base64 字符串列表。"""

    def complete_with_tools(self, messages: list[dict], tools: list[dict], json_mode: bool = False) -> ToolCallResponse:
        """原生 Function Calling：传入完整对话 + 工具 schema，返回文本与 tool_calls。

        默认实现抛 NotImplementedError（mock 不支持）；支持原生工具调用的客户端需覆盖。
        上层用 supports_tool_calling 判断走这条路径还是走文本 ReAct。
        """
        raise NotImplementedError("该客户端不支持原生 Function Calling")


class OpenAICompatibleClient(LLMClient):
    """走 OpenAI 兼容 /chat/completions 端点的真实实现。

    覆盖 DeepSeek / Qwen-Max / SGLang 暴露的 VLM 端点——它们都实现了该兼容协议，
    这正是不把模型 provider 写死、靠协议解耦的好处。
    """

    supports_tool_calling = True

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _post(self, messages: list[dict], json_mode: bool) -> str:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            # 结构化输出用低温，降低发散；文本生成给一点创造性
            "temperature": 0.0 if json_mode else 0.2,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self._post(messages, json_mode)

    def complete_vision(self, system: str, user: str, images_b64: list[str], json_mode: bool = False) -> str:
        content: list[dict] = [{"type": "text", "text": user}]
        for b64 in images_b64:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        return self._post(messages, json_mode)

    def complete_with_tools(self, messages: list[dict], tools: list[dict], json_mode: bool = False) -> ToolCallResponse:
        """原生 Function Calling：传 tools + tool_choice=auto，返回归一化的 tool_calls。"""
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0 if json_mode else 0.2,
            "tools": tools,
            "tool_choice": "auto",
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        message = data["choices"][0]["message"]
        tool_calls = [_normalize_tool_call(tc) for tc in message.get("tool_calls", [])]
        return ToolCallResponse(content=message.get("content"), tool_calls=tool_calls)


def _normalize_tool_call(tc: dict) -> ToolCall:
    """把 OpenAI 返回的 tool_call 归一化为 ToolCall（arguments 字符串 → dict）。"""
    fn = tc.get("function", {})
    arguments = fn.get("arguments", "{}")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    return ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=arguments)


class MockLLMClient(LLMClient):
    """无模型后端时的占位实现。

    它本身不做任何「智能」——只返回显式标记，避免静默返回空串导致下游解析出错。
    真正的离线确定性逻辑在语义函数里（见 nodes/planner.py、memory/false_positive.py 等）。
    """

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        return '{"_mock": true, "note": "no LLM backend; deterministic fallback should be used"}'

    def complete_vision(self, system: str, user: str, images_b64: list[str], json_mode: bool = False) -> str:
        return self.complete(system, user, json_mode)


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """工厂：按 settings.llm_backend 返回对应客户端。"""
    settings = settings or get_settings()
    if settings.llm_backend == "openai_compatible":
        return OpenAICompatibleClient(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
    return MockLLMClient()
