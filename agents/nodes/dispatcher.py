"""dispatch 节点：map-reduce 并发派发子任务到 prescreen。

并发控制（对应面试 Q6「并发怎么控制」）：
- 用 ThreadPoolExecutor + max_workers 上限，上限来自 settings.max_concurrency；
- 上限依据：GPU 显存（SGLang 小 VLM）与 API 限流，是「资源约束」而非拍脑袋；
- 单个子任务失败不抛异常、不阻塞整体，失败被记录到日志，findings 只收成功项。

为什么用线程池而不是 LangGraph Send API：两者都能做 map-reduce，但线程池的并发上限
显式可控、可单测，且不依赖 LangGraph 版本细节；Send API 作为可替换实现写入注释。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.config import get_settings
from agents.models import LogEntry
from agents.nodes.prescreen import analyze_task
from agents.toolbox import Toolbox


def dispatch_node(state: dict, toolbox: Toolbox, max_concurrency: int | None = None) -> dict:
    tasks = state.get("tasks", [])
    if not tasks:
        return {"findings": [], "logs": [LogEntry(node="dispatch", message="无子任务可派发")]}

    settings = get_settings()
    max_workers = max_concurrency or settings.max_concurrency

    findings = []
    failures = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(analyze_task, task, toolbox): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                finding = future.result()
            except Exception:
                finding = None
            if finding is not None:
                findings.append(finding)
            else:
                failures += 1

    log = LogEntry(
        node="dispatch",
        message=f"并发派发 {len(tasks)} 个子任务（max_workers={max_workers}），成功 {len(findings)}，失败 {failures}",
    )
    return {"findings": findings, "logs": [log]}
