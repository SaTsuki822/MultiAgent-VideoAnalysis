"""事件记忆：已确认真实告警 → 事件卡片 → 报告时检索历史。

与误报记忆的区别（面试可讲的边界）：
- 误报记忆：负面样本，用于「抑制」；
- 事件记忆：正面样本（已确认真实事件），用于「上下文增强」——报告生成时检索
  该区域近 N 天同类事件，让报告具备历史对比能力。

复用同一个 VectorStore，只是用不同 collection，避免引入第二套存储。
"""

from __future__ import annotations

from datetime import datetime, timedelta

from agents.config import Settings, get_settings
from agents.llm import LLMClient, MockLLMClient, get_llm_client
from agents.memory.embedding import Embedder, get_embedder
from agents.memory.vector_store import SearchHit, VectorStore, get_vector_store
from agents.models import Alarm, EventCard


def _summarize(alarm: Alarm) -> str:
    """无 LLM 时的事件摘要：模板拼装，保证离线可跑。"""
    return f"{alarm.rule_name} 于 {alarm.camera_id}（置信度 {alarm.confidence:.2f}）"


def remember_event(
    alarm: Alarm,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
) -> EventCard:
    """把一条已确认告警落成事件卡片并写入事件记忆库。"""
    settings = settings or get_settings()
    store = store or get_vector_store(settings)
    embedder = embedder or get_embedder(settings)
    llm = llm or get_llm_client()

    if isinstance(llm, MockLLMClient):
        summary = _summarize(alarm)
    else:
        raw = llm.complete(
            system="你是事件摘要助手，用一句话概括一起安全事件。",
            user=f"规则：{alarm.rule_name}，位置：{alarm.camera_id}，置信度：{alarm.confidence:.2f}",
        )
        summary = raw.strip() or _summarize(alarm)

    card = EventCard(
        id=f"evt_{alarm.id}",
        alarm_id=alarm.id,
        camera_id=alarm.camera_id,
        rule_id=alarm.rule_id,
        summary=summary,
        occurred_at=alarm.created_at,
    )

    text = f"{card.camera_id} {card.rule_id} {card.summary}"
    vec = embedder.embed_text(text)
    store.ensure_collection(settings.collection_events, embedder.dim())
    payload = {**card.model_dump(mode="json"), "occurred_at": card.occurred_at.isoformat()}
    store.upsert(settings.collection_events, card.id, vec, payload)
    return card


def search_recent_events(
    alarm: Alarm,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    settings: Settings | None = None,
    top_k: int = 3,
    days: int = 30,
) -> list[SearchHit]:
    """检索该区域近 N 天同类事件（按相似度 + 时间窗口过滤）。"""
    settings = settings or get_settings()
    store = store or get_vector_store(settings)
    embedder = embedder or get_embedder(settings)

    text = f"{alarm.camera_id} {alarm.rule_id}"
    vec = embedder.embed_text(text)
    hits = store.search(settings.collection_events, vec, limit=max(top_k * 3, top_k))

    cutoff = datetime.now() - timedelta(days=days)
    recent: list[SearchHit] = []
    for h in hits:
        occurred = h.payload.get("occurred_at", "")
        try:
            dt = datetime.fromisoformat(occurred)
        except (TypeError, ValueError):
            dt = None
        # 无时间信息的事件不因时间窗误伤，保留但排在后面（防御性处理）
        if dt is None or dt >= cutoff:
            recent.append(h)
        if len(recent) >= top_k:
            break
    return recent
