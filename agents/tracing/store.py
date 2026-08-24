"""执行轨迹的持久化抽象。

三段式（接口 + 内存实现 + 落盘实现）：
- InMemoryTraceStore : 单测 / Demo 用，进程内 dict；
- JsonlTraceStore    : 追加写 data/traces.jsonl（每行一个 span），可离线回放分析；
- 生产可替换为 Langfuse / OpenTelemetry 后端，接口不变（对应「Langfuse Trace 回填」预留）。
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from agents.tracing.models import TraceRecord, TraceSpan


class TraceStore(ABC):
    """轨迹存储接口。"""

    @abstractmethod
    def save(self, record: TraceRecord) -> None:
        """持久化一条完整轨迹。"""

    @abstractmethod
    def load(self, trace_id: str) -> TraceRecord | None:
        """按 trace_id 读取轨迹。"""

    @abstractmethod
    def list_ids(self) -> list[str]:
        """列出已持久化的 trace_id。"""


class InMemoryTraceStore(TraceStore):
    """内存存储（单测 / Demo）。"""

    def __init__(self) -> None:
        self._records: dict[str, TraceRecord] = {}
        self._lock = threading.Lock()

    def save(self, record: TraceRecord) -> None:
        with self._lock:
            self._records[record.trace_id] = record.model_copy(deep=True)

    def load(self, trace_id: str) -> TraceRecord | None:
        with self._lock:
            record = self._records.get(trace_id)
            return record.model_copy(deep=True) if record else None

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._records.keys())


class JsonlTraceStore(TraceStore):
    """JSONL 落盘存储：每行一个 span，便于流式读取与外部工具分析。"""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = Path(__file__).resolve().parent.parent.parent / "data" / "traces.jsonl"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, record: TraceRecord) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            for span in record.spans:
                line = {"trace_id": record.trace_id, **span.model_dump(mode="json")}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

    def load(self, trace_id: str) -> TraceRecord | None:
        if not self.path.exists():
            return None
        spans: list[TraceSpan] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("trace_id") != trace_id:
                    continue
                span_data = {k: v for k, v in obj.items() if k != "trace_id"}
                spans.append(TraceSpan(**span_data))
        if not spans:
            return None
        return TraceRecord(trace_id=trace_id, spans=spans)

    def list_ids(self) -> list[str]:
        if not self.path.exists():
            return []
        ids: list[str] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                tid = obj.get("trace_id")
                if tid and tid not in ids:
                    ids.append(tid)
        return ids
