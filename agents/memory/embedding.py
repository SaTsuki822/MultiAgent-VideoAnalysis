"""文本 embedding 抽象。

设计原则：
- 记忆层（SOP / 误报签名 / 事件卡片）依赖 Embedder 接口，而非直接绑某个模型；
- 无模型后端用 HashingEmbedder（特征哈希 / hashing trick），确定性、可解释、共享 token 越多
  余弦相似度越高——足以让离线 demo 演示「同一误报被抑制」的闭环；
- 生产环境换成 bge-m3 等语义模型时，只需实现 ModelEmbedder，调用方零改动。

诚实标注：HashingEmbedder 不是语义模型，它没有真正理解文本，只做 token 级相似度；
这正是 plan 里「embedding 相似 ≠ 语义等价」风险的来源，防污染机制据此做了降权而非删除。
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

from agents.config import Settings, get_settings

_TOKEN_RE = re.compile(r"[\w一-鿿]+")


class Embedder(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """把文本映射为向量。"""

    @abstractmethod
    def dim(self) -> int:
        """向量维度。"""


class HashingEmbedder(Embedder):
    """特征哈希：把每个 token 哈希到固定维度并带符号累加，再 L2 归一化。

    无模型、无随机种子依赖（确定性），可离线单测。共享 token 越多，余弦相似度越高。
    """

    def __init__(self, dim: int = 1024):
        self._dim = dim

    def dim(self) -> int:
        return self._dim

    def _tokenize(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def embed_text(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for tok in self._tokenize(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class ModelEmbedder(Embedder):
    """语义模型 embedder 占位。

    TODO(未验证)：接入 BAAI/bge-m3（多语言，维度 1024）做文本 embedding。
    需要先在本地或远程部署 embedding 服务，再实现 embed_text。
    保留此类的目的是让接口边界清晰，避免调用方感知「模型 vs 哈希」差异。
    """

    def __init__(self, dim: int = 1024):
        self._dim = dim

    def dim(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError("ModelEmbedder 未接入真实模型，请改用 HashingEmbedder 或实现模型调用")


def get_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    # 记忆层文本 embedding 当前统一用特征哈希降级；语义模型为 TODO
    return HashingEmbedder(dim=settings.embedding_dim)
