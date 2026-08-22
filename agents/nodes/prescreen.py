"""prescreen 的 worker 逻辑：对单个 AnalysisTask 做三级路由分析。

dispatcher 节点负责并发（map-reduce），这里的 analyze_task 是单个 worker。
之所以把 worker 独立成文件：既能被 dispatcher 并发调用，又能单独单测。
"""

from __future__ import annotations

from agents.models import AnalysisTask, Finding
from agents.toolbox import Toolbox


def analyze_task(task: AnalysisTask, toolbox: Toolbox) -> Finding | None:
    """对单个 task 调 video-analysis 工具，返回 Finding。

    若视频文件不存在等业务错误，工具层会返回 isError；这里捕获后返回 None 并在上层记录，
    避免单个失败子任务阻塞整体（对应面试 Q6「失败子任务独立重试不阻塞整体」）。
    """
    try:
        result = toolbox.analyze_clip(
            clip_path=task.clip_path,
            camera_id=task.camera_id,
            rule=task.rule.model_dump(),
        )
    except Exception as exc:
        return None  # 失败交由 dispatcher 记录日志
    return Finding(**result["finding"])
