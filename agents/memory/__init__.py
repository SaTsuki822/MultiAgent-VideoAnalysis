"""记忆层：误报自学习记忆 + 事件记忆 + 向量存储。

核心设计（对应面试 Q12「误报记忆怎么实现、会不会累积错误」）：
- 误报不直接删除告警，而是「只降权」——命中相似签名时把告警标记 suppressed 并附解释；
- 新签名需积累 occurrence_count >= activation_count 次才真正生效，避免单次偶发误判污染记忆；
- 高级别告警永不自动抑制（白名单兜底，防抑制过度）。
"""

from agents.memory.embedding import Embedder, HashingEmbedder, get_embedder
from agents.memory.vector_store import InMemoryVectorStore, QdrantVectorStore, SearchHit, VectorStore, get_vector_store

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "get_embedder",
    "VectorStore",
    "InMemoryVectorStore",
    "QdrantVectorStore",
    "SearchHit",
    "get_vector_store",
]
