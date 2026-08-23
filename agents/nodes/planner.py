"""planner 节点：把自然语言规则编译成结构化巡检任务（DAG）。

核心卖点（对应面试 Q1/Q2「规则即配置」）：新增巡检项 = 加一行自然语言描述，
无需重训模型。这里把描述编译为 {target_objects, evidence_hints, duration_threshold}，
供后续感知 / 复核使用。

编译有三种实现：
1. 关键词表匹配（确定性，零成本）
2. ReAct 自主探索（陌生规则查 SOP → 推理 → 输出结构化配置）
3. Mock 兜底（无 LLM 时的确定性回退）
"""

from __future__ import annotations

import json
import re

from agents.llm import LLMClient, MockLLMClient, get_llm_client
from agents.models import AnalysisTask, LogEntry, Rule
from agents.toolbox import Toolbox

# 关键词表：规则名/描述 → (目标对象, 证据要点)。仅 mock / 已知规则兜底用。
_KEYWORD_MAP: list[tuple[list[str], list[str], list[str]]] = [
    (["安全帽", "helmet", "hat", "帽"], ["helmet", "hard hat"], ["未佩戴安全帽", "无帽", "下颌带未系"]),
    (["明火", "火", "flame", "fire", "烟", "smoke"], ["fire", "flame", "smoke"], ["明火", "火焰", "浓烟"]),
    (["入侵", "闯入", "intrusion", "intruder"], ["person", "intruder"], ["非工作时间进入", "闯入危险区域"]),
    (["物料", "堆放", "堵塞", "material", "obstruction"], ["material", "obstruction"], ["通道堵塞", "物料堆放超时"]),
]


def _is_known_rule(rule: Rule) -> bool:
    """检查规则是否在预置关键词表中。"""
    text = f"{rule.name} {rule.description}".lower()
    for keywords, _, _ in _KEYWORD_MAP:
        if any(k in text for k in keywords):
            return True
    return False


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


def _build_react_prompt(rule: Rule, history: list[dict]) -> str:
    """构造 ReAct 循环的 prompt。"""
    history_text = ""
    for i, h in enumerate(history, 1):
        history_text += f"Step {i}:\nThought: {h['thought']}\nAction: {h['action']}\nObservation: {json.dumps(h['observation'], ensure_ascii=False)}\n\n"

    return (
        "你是一个巡检规则编译助手。请通过「思考 → 查询 SOP → 观察结果」的循环，"
        "理解以下规则，最终输出结构化判定依据。\n\n"
        f"规则名称：{rule.name}\n"
        f"规则描述：{rule.description}\n\n"
        "你可以使用以下工具：\n"
        "- search_sop(query): 查询 SOP 知识库，获取相关判定依据（返回 title + content 列表）\n\n"
        "请严格按以下格式输出（每步一行）：\n"
        "Thought: <你的推理，分析规则需要检测什么目标、依据什么证据>\n"
        "Action: search_sop(<查询词>)\n"
        "...（系统会自动填充 Observation，你继续下一步 Thought/Action）\n"
        "当你足够了解规则后，输出最终答案：\n"
        'Final Answer: {"target_objects": [...], "evidence_hints": [...], "duration_threshold_seconds": null}\n\n'
        "约束：\n"
        "- target_objects 是英文标识符（如 person, helmet, fire），供下游目标检测使用\n"
        "- evidence_hints 是中文视觉证据描述\n"
        "- duration_threshold_seconds 为 null 表示单帧即判定；有持续要求时填秒数\n"
        "- 最多 3 步查询，若仍不确定则保守输出\n\n"
        f"{history_text}"
        "请继续：\n"
    )


def _extract_thought(text: str) -> str:
    m = re.search(r"Thought:\s*(.+?)(?:\nAction:|\nFinal Answer:|$)", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_action(text: str) -> str:
    m = re.search(r"Action:\s*(.+?)(?:\n|$)", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_final_answer(text: str) -> dict | None:
    m = re.search(r"Final Answer:\s*(\{.*\})", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _parse_search_sop_action(action_str: str) -> str | None:
    """从 Action: search_sop(查询词) 中提取查询词。"""
    m = re.search(r"search_sop\([\"']?(.+?)[\"']?\)", action_str)
    return m.group(1).strip() if m else None


def _react_compile(rule: Rule, llm: LLMClient, toolbox: Toolbox, max_steps: int = 3) -> Rule:
    """ReAct 编译：对陌生规则自主探索，查 SOP → 推理 → 输出结构化配置。

    循环：Thought → Action(search_sop) → Observation → ...
    超步或解析失败则回退到 _deterministic_compile。
    """
    history: list[dict] = []

    for step in range(max_steps):
        prompt = _build_react_prompt(rule, history)
        try:
            raw = llm.complete(
                system="你是巡检规则编译助手。严格按 Thought/Action/Final Answer 格式输出。",
                user=prompt,
                temperature=0.2,
            )
        except Exception:
            # LLM 调用失败，立即回退
            return _deterministic_compile(rule)

        # 检查是否已经输出最终答案
        final = _extract_final_answer(raw)
        if final is not None:
            try:
                rule.target_objects = list(final.get("target_objects", []))
                rule.evidence_hints = list(final.get("evidence_hints", []))
                duration = final.get("duration_threshold_seconds")
                rule.duration_threshold_seconds = float(duration) if duration is not None else None
                return rule
            except (TypeError, ValueError):
                # 解析成功但字段类型不对，继续回退
                return _deterministic_compile(rule)

        # 否则提取 Thought + Action，执行工具调用
        thought = _extract_thought(raw)
        action = _extract_action(raw)

        if not action:
            # LLM 没有输出有效 Action，结束循环
            break

        query = _parse_search_sop_action(action)
        if not query:
            # 不是 search_sop 动作，结束循环
            break

        try:
            observation = toolbox.search_sop(query, limit=3)
        except Exception:
            observation = {"error": "SOP 查询失败"}

        history.append({"thought": thought, "action": action, "observation": observation})

    # 超步或中间失败 → 回退确定性编译
    return _deterministic_compile(rule)


def _llm_compile(rule: Rule, llm: LLMClient) -> Rule:
    """LLM 直接编译：输出结构化判定依据。（保留给已知规则的非 ReAct 路径使用）"""
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


def compile_rule(rule: Rule, llm: LLMClient | None = None, toolbox: Toolbox | None = None) -> Rule:
    """分层规则编译：已知规则走确定性编译，陌生规则走 ReAct 探索，失败兜底。

    三层策略：
    1. 关键词表命中 → 确定性编译（零成本、零延迟）
    2. 关键词表未命中 + 有 toolbox → ReAct 探索（查 SOP → 推理，最多 3 步）
    3. ReAct 超步/失败/mock 后端 → 回退确定性编译（保证流程不阻塞）
    """
    llm = llm or get_llm_client()

    if isinstance(llm, MockLLMClient):
        return _deterministic_compile(rule)

    # 已知规则：确定性编译
    if _is_known_rule(rule):
        return _deterministic_compile(rule)

    # 陌生规则 + 有 toolbox：ReAct 探索
    if toolbox is not None:
        return _react_compile(rule, llm, toolbox)

    # 陌生规则 + 无 toolbox：直接 LLM 编译（降级）
    return _llm_compile(rule, llm)


def planner_node(state: dict, toolbox: Toolbox) -> dict:
    """规划：编译规则 + 拉取摄像头台账 + 生成 camera × rule 子任务。"""
    rules: list[Rule] = state.get("rules", [])
    llm = get_llm_client()

    # 摄像头台账：优先用 state 里的，否则从 camera-registry 拉取
    cameras = state.get("cameras") or toolbox.list_cameras()

    tasks: list[AnalysisTask] = []
    react_used = 0
    for cam in cameras:
        for rule in rules:
            compiled = compile_rule(rule, llm, toolbox)
            # 判断是否走了 ReAct：如果原规则不在关键词表但编译结果有结构化字段，视为 ReAct 成功
            if not _is_known_rule(rule) and compiled.target_objects:
                react_used += 1
            tasks.append(
                AnalysisTask(
                    id=f"task_{cam['id']}_{rule.id}",
                    camera_id=cam["id"],
                    rule=compiled,
                    clip_path=f"data/clips/{cam['id']}.mp4",
                )
            )

    log_msg = f"编译 {len(rules)} 条规则 × {len(cameras)} 路摄像头 = {len(tasks)} 个子任务"
    if react_used:
        log_msg += f"（其中 {react_used} 条陌生规则通过 ReAct 探索编译）"

    log = LogEntry(node="planner", message=log_msg)
    return {"tasks": tasks, "cameras": cameras, "logs": [log]}
