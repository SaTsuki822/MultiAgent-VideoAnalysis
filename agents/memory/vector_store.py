"""向量存储抽象。

设计原则：
- 记忆层只依赖 VectorStore 接口，Qdrant 是生产实现、InMemory 是离线降级；
- InMemoryVectorStore 用 numpy 算余弦相似度，行为与 Qdrant 对齐（score 越大越相似），
  保证「无 Docker 时也能跑通 demo + 单测」，且切换后端只改配置不改业务代码。

诚实标注：InMemoryVectorStore 面向小规模 demo/测试，无持久化、无近似索引；生产必须用 Qdrant。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agents.config import Settings, get_settings


@dataclass
class SearchHit:
    """一次向量检索命中的结果。"""
    id: str
    score: float          # 余弦相似度，越大越相似
    payload: dict = field(default_factory=dict)


class VectorStore(ABC):
    @abstractmethod
    def ensure_collection(self, collection: str, dim: int) -> None:
        """确保集合存在（幂等）。"""

    @abstractmethod
    def upsert(self, collection: str, id: str, vector: list[float], payload: dict[str, Any]) -> None:
        """写入 / 覆盖一条向量。"""

    @abstractmethod
    def search(self, collection: str, vector: list[float], limit: int = 5) -> list[SearchHit]:
        """按余弦相似度检索 top-k。"""


class InMemoryVectorStore(VectorStore):
    """纯 Python 内存实现，无外部依赖，用于离线 demo 与单测。"""

    def __init__(self):
        self._collections: dict[str, dict[str, tuple[list[float], dict]]] = {}

    def ensure_collection(self, collection: str, dim: int) -> None:
        self._collections.setdefault(collection, {})

    def upsert(self, collection: str, id: str, vector: list[float], payload: dict[str, Any]) -> None:
        self._collections.setdefault(collection, {})
        self._collections[collection][id] = (list(vector), dict(payload))

    def search(self, collection: str, vector: list[float], limit: int = 5) -> list[SearchHit]:
        import numpy as np

        items = self._collections.get(collection, {})
        if not items:
            return []
        query = np.asarray(vector, dtype=float)
        q_norm = float(np.linalg.norm(query)) or 1.0
        hits: list[SearchHit] = []
        for id, (vec, payload) in items.items():
            v = np.asarray(vec, dtype=float)
            v_norm = float(np.linalg.norm(v)) or 1.0
            score = float(np.dot(query, v) / (q_norm * v_norm))
            hits.append(SearchHit(id=id, score=score, payload=payload))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]


class QdrantVectorStore(VectorStore):
    """基于 qdrant-client 的生产实现。

    三个集合三用途（对应 plan）：sop_kb（SOP 知识）、false_positive_memory（误报记忆）、
    event_memory（事件检索）。帧去重集合 frame_embeddings 由感知层单独管理。
    """

    def __init__(self, url: str, api_key: str = ""):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = QdrantClient(url=url, api_key=api_key or None)
        self._Distance = Distance
        self._VectorParams = VectorParams

    def ensure_collection(self, collection: str, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        try:
            self._client.get_collection(collection)
        except Exception:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, collection: str, id: str, vector: list[float], payload: dict[str, Any]) -> None:
        from qdrant_client.models import PointStruct

        self._client.upsert(
            collection_name=collection,
            points=[PointStruct(id=id, vector=vector, payload=payload)],
        )

    def search(self, collection: str, vector: list[float], limit: int = 5) -> list[SearchHit]:
        res = self._client.search(
            collection_name=collection,
            query_vector=vector,
            limit=limit,
        )
        return [SearchHit(id=r.id, score=float(r.score), payload=dict(r.payload or {})) for r in res]


_memory_store: InMemoryVectorStore | None = None


def get_vector_store(settings: Settings | None = None) -> VectorStore:
    """工厂：memory 后端返回进程级单例，qdrant 每次新建客户端连接。

    单例的原因：误报记忆 / 事件记忆是有状态数据，跨巡检轮次必须持久在同一实例上，
    否则「第 1 轮写入误报、第 2 轮检索抑制」的闭环会因每次新建空 store 而失效。
    """
    settings = settings or get_settings()
    if settings.vector_backend == "qdrant":
        return QdrantVectorStore(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    global _memory_store
    if _memory_store is None:
        _memory_store = InMemoryVectorStore()
    return _memory_store


def reset_vector_store() -> None:
    """清空 memory 后端单例（测试隔离用）。"""
    global _memory_store
    _memory_store = None
