"""误报自学习记忆（差异化亮点，对应面试 Q12）。

核心机制：
1. 人工标记误报 → LLM/确定性抽取结构化签名 {scene, object, lighting, description}；
2. 签名以「稳定检索键」写入向量库 false_positive_memory，携带 occurrence_count；
3. 新告警产生时，用同样的检索键去查相似误报，命中则降权/抑制并附解释。

检索键设计（关键）：误报抑制的「相似」定义为「同一摄像头 × 同一规则的同类误报」，
因此检索向量只依赖稳定维度 rule_id + camera_id + rule_name；description 等易变字段
只作为 payload 用于解释，不参与相似度计算——这保证 remember 与 lookup 的向量一致，
相似度可达 1.0，否则特征哈希对文本差异敏感会导致抑制失效。

防污染三规则（防止记忆被错误累积污染）：
- 只降权不删除：抑制只是把告警标为 suppressed 并附原因，告警本身仍在，可被人工重新确认；
- 积累生效：新签名 occurrence_count 需 >= activation_count 才真正触发抑制，单次偶发不算数；
- 高级别白名单：never_suppress_severity（默认 high）的告警永不自动抑制。

诚实标注：稳定键匹配是 mock 降级下的「同摄像头同规则」匹配，跨场景泛化需语义 embedding；
「embedding 相似 ≠ 语义等价」的风险由防污染三规则兜底。
"""

from __future__ import annotations

import json

from agents.config import Settings, get_settings
from agents.llm import LLMClient, MockLLMClient, get_llm_client
from agents.memory.embedding import Embedder, get_embedder
from agents.memory.vector_store import SearchHit, VectorStore, get_vector_store
from agents.models import Alarm, FalsePositiveSignature


def signature_to_text(sig: FalsePositiveSignature) -> str:
    """签名 → 人类可读文本（用于 payload 展示 / 解释，不用于相似度检索）。"""
    return " ".join([sig.scene, sig.object, sig.lighting, sig.description, sig.rule_id])


def retrieval_text(alarm: Alarm) -> str:
    """检索键：稳定的 rule_id + camera_id + rule_name。"""
    return f"{alarm.rule_id} {alarm.camera_id} {alarm.rule_name}"


def _deterministic_signature(alarm: Alarm, comment: str) -> FalsePositiveSignature:
    """无 LLM 时的确定性签名抽取（离线 demo / 单测用）。"""
    return FalsePositiveSignature(
        alarm_id=alarm.id,
        rule_id=alarm.rule_id,
        scene=alarm.camera_id,
        object=alarm.rule_name,
        lighting="unknown",
        description=comment or f"false positive on rule '{alarm.rule_name}'",
    )


def _llm_signature(alarm: Alarm, comment: str, llm: LLMClient) -> FalsePositiveSignature:
    """用 LLM 从误报中抽取更精细的结构化签名。"""
    prompt = (
        "从以下被人工标记为误报的告警中，抽取结构化签名，输出 JSON：\n"
        '{"scene": "...", "object": "...", "lighting": "...", "description": "..."}\n'
        "scene 描述场景（区域/环境），object 描述被误判的对象，lighting 描述光照条件，"
        "description 用一句话概括误报原因。只输出 JSON。\n"
        f"告警信息：规则={alarm.rule_name}，摄像头={alarm.camera_id}，人工备注={comment or '无'}"
    )
    raw = llm.complete(system="你是误报分析助手。", user=prompt, json_mode=True)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _deterministic_signature(alarm, comment)
    return FalsePositiveSignature(
        alarm_id=alarm.id,
        rule_id=alarm.rule_id,
        scene=str(data.get("scene", alarm.camera_id)),
        object=str(data.get("object", alarm.rule_name)),
        lighting=str(data.get("lighting", "unknown")),
        description=str(data.get("description", comment or "")),
    )


def build_signature(alarm: Alarm, comment: str = "", llm: LLMClient | None = None) -> FalsePositiveSignature:
    """统一入口：mock 走确定性，真实后端走 LLM。"""
    llm = llm or get_llm_client()
    if isinstance(llm, MockLLMClient):
        return _deterministic_signature(alarm, comment)
    return _llm_signature(alarm, comment, llm)


def remember_false_positive(
    alarm: Alarm,
    comment: str,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
) -> FalsePositiveSignature:
    """把一条人工确认的误报写入记忆库，并维护 occurrence_count。

    检索到几乎相同的既有键（score >= 0.95）则累加计数而非新增——
    这正是「积累 N 次才生效」的计数来源。
    """
    settings = settings or get_settings()
    store = store or get_vector_store(settings)
    embedder = embedder or get_embedder(settings)
    llm = llm or get_llm_client()

    sig = build_signature(alarm, comment, llm)
    vec = embedder.embed_text(retrieval_text(alarm))
    store.ensure_collection(settings.collection_fp, embedder.dim())

    hits = store.search(settings.collection_fp, vec, limit=1)
    if hits and hits[0].score >= 0.95:
        count = int(hits[0].payload.get("occurrence_count", 1)) + 1
        payload = {**sig.model_dump(), "occurrence_count": count}
        store.upsert(settings.collection_fp, hits[0].id, vec, payload)
        sig.occurrence_count = count
    else:
        payload = {**sig.model_dump(), "occurrence_count": sig.occurrence_count}
        store.upsert(settings.collection_fp, alarm.id, vec, payload)

    return sig


def lookup_similar_false_positive(
    alarm: Alarm,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
    top_k: int = 3,
) -> list[SearchHit]:
    """检索与新告警相似的已记忆误报（用稳定检索键）。"""
    settings = settings or get_settings()
    store = store or get_vector_store(settings)
    embedder = embedder or get_embedder(settings)

    vec = embedder.embed_text(retrieval_text(alarm))
    return store.search(settings.collection_fp, vec, limit=top_k)


def should_suppress(
    alarm: Alarm,
    hits: list[SearchHit],
    settings: Settings | None = None,
) -> tuple[bool, str]:
    """判断一条告警是否应被记忆抑制。

    返回 (是否抑制, 原因)。三条防污染规则在此显式体现：
    1. 高级别白名单：never_suppress_severity 永不自动抑制；
    2. 相似度阈值：hit.score >= fp_similarity_threshold 才视为命中；
    3. 积累生效：命中的签名 occurrence_count >= fp_activation_count 才触发。
    """
    settings = settings or get_settings()

    if alarm.severity == settings.never_suppress_severity:
        return False, f"severity={alarm.severity} is on the never-suppress whitelist"

    for hit in hits:
        if hit.score < settings.fp_similarity_threshold:
            continue
        count = int(hit.payload.get("occurrence_count", 0))
        if count >= settings.fp_activation_count:
            desc = hit.payload.get("description", "")
            return True, f"matched known false-positive '{desc}' (score={hit.score:.3f}, count={count})"
    return False, ""
