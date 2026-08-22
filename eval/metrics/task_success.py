"""任务层指标：端到端任务成功率。

定义：一次巡检是否「成功」，取决于两件事——
1. 流程完整走完（产出了 Report）；
2. 对标注里的异常，至少产生了一条对应的告警（检出）。

返回成功率（0~1）与逐次结果明细，供回归对比。
"""

from __future__ import annotations

from eval.dataset.annotations import Annotation


def compute_task_success(runs: list[dict]) -> dict:
    """runs: [{"produced_report": bool, "detected_rule_ids": set[str], "expected_rule_ids": set[str]}]"""
    success = 0
    details = []
    for i, run in enumerate(runs):
        produced = run.get("produced_report", False)
        detected = set(run.get("detected_rule_ids", []))
        expected = set(run.get("expected_rule_ids", []))
        covered = expected.issubset(detected)
        ok = produced and covered
        success += int(ok)
        details.append({"run": i, "success": ok, "produced_report": produced, "detected": sorted(detected), "expected": sorted(expected)})
    rate = success / len(runs) if runs else 0.0
    return {"task_success_rate": rate, "success_count": success, "total_runs": len(runs), "details": details}


def expected_rule_ids(annotations: list[Annotation]) -> set[str]:
    return {a.rule_id for a in annotations}
