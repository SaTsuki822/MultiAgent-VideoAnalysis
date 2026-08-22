"""knowledge：SOP 知识库 + 相似事件检索（复用记忆层的向量存储）。

工具：search_sop / search_similar_events。
- search_sop：planner 检索相关 SOP 补充判定依据（复用项目1的 RAG 能力）；
- search_similar_events：报告生成时检索历史同类事件。

SOP 数据为 mock 样例（真实工地 SOP 需按现场手册补充），embedding 用特征哈希降级。
"""

from __future__ import annotations

from mcp_servers._core import MCPServer

from agents.config import get_settings
from agents.memory.embedding import get_embedder
from agents.memory.vector_store import get_vector_store

# mock SOP 样例：判定依据，供 planner / verify 注入 prompt
SOP_DOCS: list[dict] = [
    {"id": "sop_helmet", "title": "安全帽佩戴标准", "content": "施工区域所有人员必须佩戴安全帽，下颌带系紧；未佩戴或佩戴不规范判定为违规。"},
    {"id": "sop_fire", "title": "明火与烟雾处置", "content": "施工区域禁止明火；发现明火或明显烟雾立即上报，按消防预案处置。"},
    {"id": "sop_intrusion", "title": "危险区域入侵", "content": "非工作时间（18:00-06:00）禁止人员进入基坑、塔吊等危险区域。"},
    {"id": "sop_material", "title": "物料堆放规范", "content": "通道物料堆放不得超过 2 小时，且需留出消防通道宽度。"},
]


def seed_sop(store=None, embedder=None) -> None:
    """把 SOP 样例写入向量库（幂等：已存在则跳过）。"""
    settings = get_settings()
    store = store or get_vector_store(settings)
    embedder = embedder or get_embedder(settings)
    store.ensure_collection(settings.collection_sop, embedder.dim())
    if store.search(settings.collection_sop, embedder.embed_text("安全帽"), limit=1):
        return
    for doc in SOP_DOCS:
        vec = embedder.embed_text(f"{doc['title']} {doc['content']}")
        store.upsert(settings.collection_sop, doc["id"], vec, doc)


def build_server(store=None, embedder=None) -> MCPServer:
    settings = get_settings()
    store = store or get_vector_store(settings)
    embedder = embedder or get_embedder(settings)
    seed_sop(store, embedder)

    server = MCPServer(name="knowledge", version="1.0.0", instructions="SOP 知识库与相似事件检索")

    server.register(
        name="search_sop",
        description=(
            "检索与查询相关的 SOP 判定依据，返回带相似度分数的条目。"
            "用于为巡检规划 / 复核补充判定标准。参数 query、limit（默认 3）。"
        ),
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        },
        handler=lambda args: _search_sop(args, store, embedder, settings),
    )

    server.register(
        name="search_similar_events",
        description=(
            "检索历史相似事件卡片，用于报告生成时提供「该区域近 30 天同类事件」上下文。"
            "参数 query、limit（默认 3）。"
        ),
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        },
        handler=lambda args: _search_events(args, store, embedder, settings),
    )

    return server


def _search_sop(args, store, embedder, settings) -> dict:
    vec = embedder.embed_text(args["query"])
    hits = store.search(settings.collection_sop, vec, limit=args.get("limit", 3))
    return {
        "results": [
            {"id": h.id, "score": h.score, "title": h.payload.get("title", ""), "content": h.payload.get("content", "")}
            for h in hits
        ]
    }


def _search_events(args, store, embedder, settings) -> dict:
    vec = embedder.embed_text(args["query"])
    hits = store.search(settings.collection_events, vec, limit=args.get("limit", 3))
    return {"results": [{"id": h.id, "score": h.score, "summary": h.payload.get("summary", "")} for h in hits]}


def main() -> None:
    build_server().run_stdio()


if __name__ == "__main__":
    main()
