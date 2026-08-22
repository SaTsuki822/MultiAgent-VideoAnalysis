"""端到端 workflow 编排测试（用 FakeToolbox，不依赖视频 / GPU / MCP 进程）。

验证：三轮巡检中，误报记忆在第 3 轮自动抑制同类告警（默认 activation_count=2）。
"""

from agents.memory.vector_store import reset_vector_store
from agents.workflow import run_pipeline


def make_initial(patrol_id, rule, camera):
    return {"patrol_id": patrol_id, "rules": [rule], "cameras": [camera]}


def fp_handler(pending):
    return [{"alarm_id": a.id, "decision": "false_positive", "comment": "夜间误报"} for a in pending]


def confirm_handler(pending):
    return [{"alarm_id": a.id, "decision": "confirm"} for a in pending]


def test_three_round_suppression_loop(fake_toolbox, medium_rule, camera):
    reset_vector_store()

    # 第 1 轮：检出告警 → 人工标记误报（occurrence_count=1）
    s1 = run_pipeline(make_initial("p1", medium_rule, camera), fake_toolbox, hitl_handler=fp_handler)
    assert len(s1["alarms"]) == 1
    assert s1["alarms"][0].status == "false_positive"

    # 第 2 轮：再次检出，count=1 < 2 不抑制 → 人工再标记误报（count=2）
    s2 = run_pipeline(make_initial("p2", medium_rule, camera), fake_toolbox, hitl_handler=fp_handler)
    assert len(s2["alarms"]) == 1
    assert s2["alarms"][0].status == "false_positive"

    # 第 3 轮：count=2 >= 2 → 自动抑制，无需人工复核
    s3 = run_pipeline(make_initial("p3", medium_rule, camera), fake_toolbox, hitl_handler=confirm_handler)
    assert len(s3["alarms"]) == 1
    assert s3["alarms"][0].suppressed is True
    assert s3["pending_review"] == []


def test_pipeline_produces_report(fake_toolbox, medium_rule, camera):
    reset_vector_store()
    s = run_pipeline(make_initial("p1", medium_rule, camera), fake_toolbox, hitl_handler=confirm_handler)
    assert s["report"] is not None
    assert s["report"].stats["total_alarms"] == 1
