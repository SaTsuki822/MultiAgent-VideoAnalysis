"""GuardEye —— 多模态自主巡检 Agent 核心包。

分层（自上而下）：
- models.py   : 数据模型（跨层传输的对象）
- state.py    : PatrolState 与 reducer（LangGraph 状态）
- config.py   : 集中配置
- llm.py      : LLM/VLM 客户端抽象（真实 + mock）
- memory/     : 误报记忆 / 事件记忆 / 向量存储
- prescreen/  : 三级模型路由（L0 运动检测 → L1 抽帧去重 → L2 VLM 初筛）
- nodes/      : LangGraph 各节点
- workflow.py : 主状态图（checkpoint + interrupt）
- hitl.py     : 人工复核交互
"""

__version__ = "0.1.0"
