"""planner 节点：把自然语言规则编译成结构化巡检任务（DAG）。

核心卖点（对应面试 Q1/Q2「规则即配置」）：新增巡检项 = 加一行自然语言描述，
无需重训模型。这里把描述编译为 {target_objects, evidence_hints, duration_threshold}，
供后续感知 / 复核使用。

编译有两种实现：mock 用关键词表（确定性），真实后端用 LLM。
"""

from __future__ import annotations

import json

from agents.llm import LLMClient, MockLLMClient, get_llm_client
from agents.models import AnalysisTask, LogEntry, Rule
from agents.toolbox import Toolbox

# 关键词表：规则名/描述 → (目标对象, 证据要点)。仅 mock 兜底用。
_KEYWORD_MAP: list[tuple[list[str], list[str], list[str]]] = [
    (["安全帽", "helmet", "hat", "帽"], ["helmet", "hard hat"], ["未佩戴安全帽", "无帽", "下颌带未系"]),
    (["明火", "火", "flame", "fire", "烟", "smoke"], ["fire", "flame", "smoke"], ["明火", "火焰", "浓烟"]),
    (["入侵", "闯入", "intrusion", "intruder"], ["person", "intruder"], ["非工作时间进入", "闯入危险区域"]),
    (["物料", "堆放", "堵塞", "material", "obstruction"], ["material", "obstruction"], ["通道堵塞", "物料堆放超时"]),
]


def _deterministic_compile(rule: Rule) -> Rule:
    """确定性编译：按关键词表匹配。无 LLM 时的兜底。"""
    text = f"{rule.name} {rule.description}".lower()
    for keywords, objects, hints in _KEYWORD_MAP:
        if any(k in text for k in keywords):
            rule.target_objects = objects
            rule.evidence_hints = hints
            return rule
    # 未命中任何已知规则：目标对象退化为规则名本身
    rule.target_objects = [rule.name]
    rule.evidence_hints = [rule.description]
    return rule


def _llm_compile(rule: Rule, llm: LLMClient) -> Rule:
    """LLM 编译：输出结构化判定依据。"""
    prompt = (
        "把以下巡检规则编译成结构化判定依据。\n"
        f"规则：{rule.name}（{rule.description}）\n"
        '只输出 JSON：{"target_objects": [...], "evidence_hints": [...], "duration_threshold_seconds": null}\n'
        "target_objects 是画面中要检测的目标对象；evidence_hints 是判定命中的视觉证据要点；"
        "若规则要求持续一段时间才算异常，duration_threshold_seconds 填秒数，否则 null。"
    )
    raw = llm.complete(system="你是巡检规则编译助手。", user=prompt, json_mode=True)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _deterministic_compile(rule)
    rule.target_objects = list(data.get("target_objects", []))
    rule.evidence_hints = list(data.get("evidence_hints", []))
    duration = data.get("duration_threshold_seconds")
    rule.duration_threshold_seconds = float(duration) if duration else None
    return rule


def compile_rule(rule: Rule, llm: LLMClient | None = None) -> Rule:
    llm = llm or get_llm_client()
    if isinstance(llm, MockLLMClient):
        return _deterministic_compile(rule)
    return _llm_compile(rule, llm)


def planner_node(state: dict, toolbox: Toolbox) -> dict:
    """规划：编译规则 + 拉取摄像头台账 + 生成 camera × rule 子任务。"""
    rules: list[Rule] = state.get("rules", [])
    llm = get_llm_client()

    # 摄像头台账：优先用 state 里的，否则从 camera-registry 拉取
    cameras = state.get("cameras") or toolbox.list_cameras()

    tasks: list[AnalysisTask] = []
    for cam in cameras:
        for rule in rules:
            compiled = compile_rule(rule, llm)
            tasks.append(
                AnalysisTask(
                    id=f"task_{cam['id']}_{rule.id}",
                    camera_id=cam["id"],
                    rule=compiled,
                    clip_path=f"data/clips/{cam['id']}.mp4",
                )
            )

    log = LogEntry(node="planner", message=f"编译 {len(rules)} 条规则 × {len(cameras)} 路摄像头 = {len(tasks)} 个子任务")
    return {"tasks": tasks, "cameras": cameras, "logs": [log]}
