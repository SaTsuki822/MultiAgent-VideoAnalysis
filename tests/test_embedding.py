"""测试 HashingEmbedder 的相似度行为。"""

import numpy as np

from agents.memory.embedding import HashingEmbedder


def cosine(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) or 1.0))


def test_identical_text_max_similarity():
    emb = HashingEmbedder(dim=256)
    v1 = emb.embed_text("未佩戴安全帽")
    v2 = emb.embed_text("未佩戴安全帽")
    assert cosine(v1, v2) > 0.99


def test_shared_tokens_more_similar_than_unrelated():
    emb = HashingEmbedder(dim=256)
    a = emb.embed_text("区域入侵 东门")
    b = emb.embed_text("区域入侵 西门")
    c = emb.embed_text("明火 烟雾")
    assert cosine(a, b) > cosine(a, c)


def test_dim_consistency():
    emb = HashingEmbedder(dim=128)
    assert len(emb.embed_text("任意文本")) == 128
    assert emb.dim() == 128
