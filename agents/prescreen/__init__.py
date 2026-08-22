"""感知层：三级模型路由（成本控制核心）。

L0 运动检测（免费，过滤静止）→ L1 抽帧 + 去重（低成本）→ L2 小 VLM 初筛（有成本）。
大模型二次确认在 verify 节点，不属于感知层。
"""

from agents.prescreen.router import CostBreakdown, Router

__all__ = ["Router", "CostBreakdown"]
