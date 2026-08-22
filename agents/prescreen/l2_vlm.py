"""L2 小 VLM 初筛。

职责：对抽帧后的单帧做「是否命中巡检规则」的粗判，约束输出 JSON schema
{hit, confidence, evidence}（对应面试 Q7「输出约束 JSON schema + 要求引用帧证据」）。

统一 ScreenFn 签名：screen(rule, frame_b64, camera_id, timestamp_seconds) -> VLMResult。
- screen_frame：真实 VLM 调用（走 LLMClient.complete_vision）+ mock 分支（无视觉能力，恒不命中）；
- ScriptedScreen：测试 / demo 用可注入实现，按 (camera, rule, 时间戳) 预置命中表，
  用于演示完整告警闭环——是显式的数据注入，不是编造智能，README 会注明。
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Callable

import numpy as np

from agents.llm import LLMClient, MockLLMClient, get_llm_client
from agents.models import Rule

VLM_PROMPT_TEMPLATE = (
    "你是工地安全巡检的视觉初筛模型。判断画面是否命中以下巡检规则。\n"
    "规则：{rule_name}（{rule_desc}）\n"
    '只输出 JSON：{{"hit": true/false, "confidence": 0.0~1.0, "evidence": "一句话描述所见"}}。\n'
    "命中（hit=true）意味着画面中确实出现了规则描述的目标；不确定时 hit=false。"
)


@dataclass
class VLMResult:
    hit: bool
    confidence: float
    evidence: str


# (rule, frame_b64, camera_id, timestamp_seconds) -> VLMResult
ScreenFn = Callable[[Rule, str, str, float], VLMResult]


def frame_to_b64(frame: np.ndarray) -> str:
    """numpy 帧 → JPEG base64 字符串（走 cv2 编码）。"""
    import cv2

    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("帧编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def screen_frame(
    rule: Rule,
    frame_b64: str,
    camera_id: str = "",
    timestamp_seconds: float = 0.0,
    llm: LLMClient | None = None,
) -> VLMResult:
    """对单帧做 VLM 初筛（真实 / mock 分支）。

    真实后端：构造约束 prompt，强制 JSON 输出；解析失败兜底为不命中——
    这是「宁漏勿误报」的保守策略：漏检可在 verify 层靠大模型兜回，误报则直接消耗人工复核成本。
    mock 后端：无视觉能力，恒返回 hit=False。
    """
    llm = llm or get_llm_client()
    if isinstance(llm, MockLLMClient):
        return VLMResult(hit=False, confidence=0.0, evidence="mock backend: no vision capability")

    prompt = VLM_PROMPT_TEMPLATE.format(rule_name=rule.name, rule_desc=rule.description)
    raw = llm.complete_vision(system="你是视觉初筛助手。", user=prompt, images_b64=[frame_b64], json_mode=True)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return VLMResult(hit=False, confidence=0.0, evidence="VLM output unparsable")
    return VLMResult(
        hit=bool(data.get("hit", False)),
        confidence=float(data.get("confidence", 0.0)),
        evidence=str(data.get("evidence", "")),
    )


class ScriptedScreen:
    """测试 / demo 用：按预置命中表返回结果，模拟真实 VLM 输出。

    用法：ScriptedScreen({camera_id: {rule_id: [命中帧时间戳...]}})。
    匹配窗口 ±0.5s，因为采样帧时间戳未必精确等于预置时间。
    """

    def __init__(self, hit_map: dict[str, dict[str, list[float]]], default_confidence: float = 0.9, window: float = 0.5):
        self._hit_map = hit_map
        self._default_confidence = default_confidence
        self._window = window

    def __call__(self, rule: Rule, frame_b64: str, camera_id: str, timestamp_seconds: float) -> VLMResult:
        rule_hits = self._hit_map.get(camera_id, {}).get(rule.id, [])
        matched = any(abs(t - timestamp_seconds) <= self._window for t in rule_hits)
        if matched:
            return VLMResult(
                hit=True,
                confidence=self._default_confidence,
                evidence=f"scripted hit on '{rule.name}' at {timestamp_seconds:.1f}s",
            )
        return VLMResult(hit=False, confidence=0.0, evidence="scripted: no hit")
