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

from agents.llm import LLMClient, MockLLMClient, ToolCall, get_llm_client, mcp_tools_to_openai
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


def _make_fingerprint(action: str, observation: dict) -> str:
    """生成工具调用的指纹，用于死循环检测。

    指纹由查询词 + Observation 前 120 字符组成。
    同样的查询拿到同样的返回 → 连续重复即判定死循环。
    """
    obs_text = json.dumps(observation, ensure_ascii=False)[:120]
    return f"{action}|{obs_text}"


def _detect_loop(fingerprints: list[str]) -> bool:
    """检测最后两次调用是否完全相同（死循环）。"""
    if len(fingerprints) < 2:
        return False
    return fingerprints[-1] == fingerprints[-2]


def _compress_text_history(history: list[dict]) -> list[dict]:
    """对文本 ReAct 的 history 做轻量压缩：保留最后一条完整，前面的做摘要。

    压缩策略：
    - thought 截断到 80 字符
    - observation 只保留标题列表或错误摘要
    """
    if len(history) <= 1:
        return history
    compressed = []
    for i, h in enumerate(history[:-1]):
        obs = h["observation"]
        if isinstance(obs, dict):
            if "results" in obs:
                titles = [r.get("title", "") for r in obs["results"][:2]]
                obs_summary = f"查到: {', '.join(titles) or '无结果'}"
            elif "error" in obs:
                obs_summary = f"失败: {obs['error']}"
            else:
                obs_summary = str(obs)[:80]
        else:
            obs_summary = str(obs)[:80]
        compressed.append({
            "thought": f"[Step{i+1}] {h['thought'][:80]}...",
            "action": h["action"],
            "observation": obs_summary,
        })
    compressed.append(history[-1])
    return compressed


def _compress_fc_messages(messages: list[dict]) -> list[dict]:
    """对 Function Calling 的消息列表做压缩：把早期的 assistant/tool 轮次摘要成一条。"""
    if len(messages) <= 4:
        return messages
    compressed = [messages[0], messages[1]]  # 保留 system + user
    # 遍历中间的 assistant/tool 对，做摘要
    i = 2
    while i < len(messages) - 2:
        if messages[i].get("role") == "assistant" and i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
            tool_calls = messages[i].get("tool_calls", [])
            if tool_calls:
                fn = tool_calls[0]["function"]
                args = json.loads(fn["arguments"])
                query = args.get("query", "")[:40]
                result = messages[i + 1].get("content", "")[:80]
                compressed.append({
                    "role": "assistant",
                    "content": f"[历史步骤] 调用 {fn['name']}('{query}') → {result}",
                })
            i += 2
        else:
            compressed.append(messages[i])
            i += 1
    compressed.extend(messages[-2:])  # 保留最后两条完整
    return compressed


def _apply_final(rule: Rule, final: dict) -> Rule:
    """把编译结果 dict 写回 rule；字段类型不对则回退确定性编译。"""
    try:
        rule.target_objects = list(final.get("target_objects", []))
        rule.evidence_hints = list(final.get("evidence_hints", []))
        duration = final.get("duration_threshold_seconds")
        rule.duration_threshold_seconds = float(duration) if duration is not None else None
        return rule
    except (TypeError, ValueError):
        return _deterministic_compile(rule)


def _parse_final(raw: str) -> dict | None:
    """从最终文本解析结构化结果：兼容 "Final Answer: {...}" 与纯 JSON 两种形态。"""
    if not raw:
        return None
    final = _extract_final_answer(raw)
    if final is not None:
        return final
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _react_compile_text(rule: Rule, llm: LLMClient, toolbox: Toolbox, max_steps: int = 3) -> Rule:
    """文本 ReAct 编译：模型输出 Thought/Action 文本，代码解析并执行 search_sop。

    循环：Thought → Action(search_sop) → Observation → ...
    超步、解析失败、死循环检测触发则回退到 _deterministic_compile。
    """
    history: list[dict] = []
    fingerprints: list[str] = []

    for step in range(max_steps):
        # ---- 上下文压缩：第 2 步起对早期 history 做摘要，防止 prompt 膨胀 ----
        active_history = _compress_text_history(history) if step >= 1 else history

        prompt = _build_react_prompt(rule, active_history)
        try:
            raw = llm.complete(
                system="你是巡检规则编译助手。严格按 Thought/Action/Final Answer 格式输出。",
                user=prompt,
            )
        except Exception:
            # LLM 调用失败，立即回退
            return _deterministic_compile(rule)

        # 检查是否已经输出最终答案
        final = _parse_final(raw)
        if final is not None:
            return _apply_final(rule, final)

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

        # ---- 死循环检测：记录指纹，连续重复则中断 ----
        fingerprint = _make_fingerprint(action, observation)
        fingerprints.append(fingerprint)
        if _detect_loop(fingerprints):
            # 检测到死循环（连续两次相同查询+相同返回），立即退出 ReAct
            break

        history.append({"thought": thought, "action": action, "observation": observation})

    # 超步、中间失败或死循环 → 回退确定性编译
    return _deterministic_compile(rule)


def _build_fc_user_prompt(rule: Rule) -> str:
    """Function Calling 编译的用户 prompt：要求模型调 search_sop 后输出纯 JSON。"""
    return (
        f"规则名称：{rule.name}\n"
        f"规则描述：{rule.description}\n\n"
        "请通过调用 search_sop 工具查询相关判定依据，理解规则后输出最终答案（纯 JSON）：\n"
        '{"target_objects": [...], "evidence_hints": [...], "duration_threshold_seconds": null}\n'
        "约束：target_objects 是英文标识符（如 person, helmet, fire）；evidence_hints 是中文视觉证据描述；"
        "duration_threshold_seconds 为 null 表示单帧即判定，有持续要求时填秒数；最多调用 3 次工具。"
    )


def _assistant_message(resp: "ToolCallResponse") -> dict:
    """把 ToolCallResponse 转回 OpenAI 消息格式的 assistant 消息（含 tool_calls）。"""
    tool_calls = [
        {
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
        }
        for tc in resp.tool_calls
    ]
    return {"role": "assistant", "content": resp.content, "tool_calls": tool_calls}


def _execute_tool_call(tool_call: ToolCall, toolbox: Toolbox):
    """执行模型请求的 tool_call。当前只暴露 search_sop，走 Toolbox 语义方法。"""
    if tool_call.name == "search_sop":
        query = tool_call.arguments.get("query", "")
        limit = int(tool_call.arguments.get("limit", 3))
        return toolbox.search_sop(query, limit=limit)
    return {"error": f"unknown tool: {tool_call.name}"}


def _react_compile_with_tools(rule: Rule, llm: LLMClient, toolbox: Toolbox, max_steps: int = 3) -> Rule:
    """原生 Function Calling 编译：模型直接输出 search_sop 的 tool_call，框架执行后喂回结果。

    与 _react_compile_text 的区别：不再用文本解析 "Action: search_sop(...)",
    而是模型原生输出 tool_calls；多轮循环按 OpenAI 协议组织
    （assistant.tool_calls → tool 角色结果 → 下一轮）。

    本实现已集成：
    - 死循环检测：同样的查询+返回连续重复 2 次即中断
    - 上下文压缩：第 2 步起对早期消息做摘要，防止上下文膨胀
    """
    # 从 MCP tools/list 拉取 search_sop 的 schema，转成 OpenAI tools 格式
    tools = mcp_tools_to_openai(toolbox.list_tools())
    messages: list[dict] = [
        {"role": "system", "content": "你是巡检规则编译助手。可调用 search_sop 查询 SOP，最终输出结构化判定依据。"},
        {"role": "user", "content": _build_fc_user_prompt(rule)},
    ]
    fingerprints: list[str] = []

    for step in range(max_steps):
        # ---- 上下文压缩：第 2 步起对早期 messages 做摘要 ----
        active_messages = _compress_fc_messages(messages) if step >= 1 else messages

        try:
            resp = llm.complete_with_tools(active_messages, tools)
        except Exception:
            return _deterministic_compile(rule)

        # 没有 tool_call → 模型认为已可作答，尝试解析最终答案
        if not resp.tool_calls:
            final = _parse_final(resp.content or "")
            if final is not None:
                return _apply_final(rule, final)
            return _deterministic_compile(rule)

        # 有 tool_call → 记入消息历史，执行并把结果作为 tool 消息喂回
        messages.append(_assistant_message(resp))
        for tc in resp.tool_calls:
            observation = _execute_tool_call(tc, toolbox)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(observation, ensure_ascii=False)})

            # ---- 死循环检测：只针对 search_sop 生成指纹 ----
            if tc.name == "search_sop":
                action_repr = f"search_sop({tc.arguments.get('query', '')})"
                fingerprint = _make_fingerprint(action_repr, observation)
                fingerprints.append(fingerprint)
                if _detect_loop(fingerprints):
                    # 检测到死循环，直接回退确定性编译
                    return _deterministic_compile(rule)

    return _deterministic_compile(rule)



def _react_compile(rule: Rule, llm: LLMClient, toolbox: Toolbox, max_steps: int = 3) -> Rule:
    """ReAct 编译分发：真实客户端支持原生 Function Calling 则走 tool-call 循环，
    否则走文本 ReAct。两者都以 _deterministic_compile 兜底。
    """
    if getattr(llm, "supports_tool_calling", False):
        return _react_compile_with_tools(rule, llm, toolbox, max_steps)
    return _react_compile_text(rule, llm, toolbox, max_steps)


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
    from datetime import datetime

    rules: list[Rule] = state.get("rules", [])
    llm = get_llm_client()

    # 摄像头台账：优先用 state 里的，否则从 camera-registry 拉取
    cameras = state.get("cameras") or toolbox.list_cameras()

    # Demo 默认巡检窗口（与 fetcher_node 保持一致）
    now = datetime.now()
    start_time = now.replace(hour=8, minute=0, second=0, microsecond=0)
    end_time = now.replace(hour=9, minute=0, second=0, microsecond=0)
    duration = (end_time - start_time).total_seconds()

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
                    clip_start_time=start_time,
                    clip_end_time=end_time,
                    duration_seconds=duration,
                )
            )

    log_msg = f"编译 {len(rules)} 条规则 × {len(cameras)} 路摄像头 = {len(tasks)} 个子任务"
    if react_used:
        log_msg += f"（其中 {react_used} 条陌生规则通过 ReAct 探索编译）"

    log = LogEntry(node="planner", message=log_msg)
    return {"tasks": tasks, "cameras": cameras, "logs": [log]}
