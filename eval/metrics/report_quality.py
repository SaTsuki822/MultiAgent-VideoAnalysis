"""报告质量：LLM-as-Judge 按 rubric 打分。

已知偏差（对应面试 Q13）：位置偏差、长度偏好——所以要做成对比较 + 20% 人工抽检校准，
不能只看单条绝对分。这里实现打分接口，mock 用规则化打分（结构完整性 + 关键信息覆盖），
真实后端走 LLM。rubric 维度：完整性 / 准确性 / 可读性。
"""

from __future__ import annotations

from agents.llm import LLMClient, MockLLMClient, get_llm_client
from agents.models import Report

RUBRIC = {
    "completeness": "是否包含统计数字、告警明细、确认/抑制/误报分类",
    "accuracy": "告警是否与证据一致，无明显编造",
    "readability": "是否结构化、可读",
}


def score_report(report: Report, llm: LLMClient | None = None) -> dict:
    llm = llm or get_llm_client()
    if isinstance(llm, MockLLMClient):
        # 规则化打分：结构完整性 + 信息覆盖（无 LLM 时的确定性基线）
        completeness = 1.0 if (report.summary and report.stats) else 0.5
        accuracy = 1.0 if all(a.rule_name for a in report.alarms) else 0.5
        readability = 1.0
        return {
            "completeness": completeness,
            "accuracy": accuracy,
            "readability": readability,
            "note": "mock 规则化打分，真实打分需 LLM-as-Judge + 人工抽检校准",
        }

    prompt = (
        "按以下 rubric 给巡检报告打分（每项 0~1）：\n"
        f"{RUBRIC}\n"
        f"报告摘要：{report.summary}\n统计：{report.stats}\n"
        '只输出 JSON：{"completeness": 0.0, "accuracy": 0.0, "readability": 0.0}'
    )
    raw = llm.complete(system="你是报告质量评审。", user=prompt, json_mode=True)
    import json

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    return {
        "completeness": float(data.get("completeness", 0.0)),
        "accuracy": float(data.get("accuracy", 0.0)),
        "readability": float(data.get("readability", 0.0)),
    }
