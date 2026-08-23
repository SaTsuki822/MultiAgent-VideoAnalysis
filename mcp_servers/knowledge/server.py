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
# 为支持 ReAct 规则编译，每条 SOP 增加编译辅助字段（target_objects / evidence_hints / related_keywords）
SOP_DOCS: list[dict] = [
    {
        "id": "sop_helmet",
        "title": "安全帽佩戴标准",
        "content": "施工区域所有人员必须佩戴安全帽，下颌带系紧；未佩戴或佩戴不规范判定为违规。",
        "target_objects": ["person", "helmet"],
        "evidence_hints": ["未佩戴安全帽", "下颌带未系"],
        "related_keywords": ["安全帽", "头盔", "佩戴", "帽"],
    },
    {
        "id": "sop_fire",
        "title": "明火与烟雾处置",
        "content": "施工区域禁止明火；发现明火或明显烟雾立即上报，按消防预案处置。",
        "target_objects": ["fire", "flame", "smoke"],
        "evidence_hints": ["明火", "火焰", "浓烟", "火花"],
        "related_keywords": ["明火", "火", "烟", "燃烧"],
    },
    {
        "id": "sop_intrusion",
        "title": "危险区域入侵",
        "content": "非工作时间（18:00-06:00）禁止人员进入基坑、塔吊等危险区域。",
        "target_objects": ["person", "intruder"],
        "evidence_hints": ["非工作时间进入", "闯入危险区域", "越界"],
        "related_keywords": ["入侵", "闯入", "进入", "越界"],
    },
    {
        "id": "sop_material",
        "title": "物料堆放规范",
        "content": "通道物料堆放不得超过 2 小时，且需留出消防通道宽度。",
        "target_objects": ["material", "obstruction"],
        "evidence_hints": ["通道堵塞", "物料堆放超时", "消防通道被占"],
        "related_keywords": ["物料", "堆放", "堵塞", "通道"],
    },
    {
        "id": "sop_fall_protection",
        "title": "高空作业与防坠落",
        "content": "距坠落高度基准面 2 米及以上的作业为高空作业，必须系挂安全带，设置安全网。",
        "target_objects": ["person", "safety_belt", "safety_net"],
        "evidence_hints": ["未系安全带", "悬空作业", "无安全防护"],
        "related_keywords": ["高空", "坠落", "安全带", "安全网"],
    },
    {
        "id": "sop_electric",
        "title": "临时用电安全",
        "content": "配电箱需上锁、接地完好；禁止私拉乱接电线；潮湿环境使用漏电保护器。",
        "target_objects": ["electrical_box", "cable"],
        "evidence_hints": ["配电箱未上锁", "电线私拉", "无漏电保护"],
        "related_keywords": ["用电", "电线", "配电箱", "漏电"],
    },
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
