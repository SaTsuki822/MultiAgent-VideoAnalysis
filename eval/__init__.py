"""评测体系：标注集 + 指标 + 基线 + 回归入口。

三层指标（对应面试 Q13「怎么评估这个系统」）：
- 感知层：检出率 Recall / 误报率（perception.py）
- 任务层：端到端成功率（task_success.py）
- 报告层：LLM-as-Judge 质量打分（report_quality.py，需人工抽检校准）
"""

__all__: list[str] = []
