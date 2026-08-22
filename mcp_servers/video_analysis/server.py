"""video-analysis：封装三级路由推理，对外暴露为 MCP 工具。

工具：analyze_clip / analyze_frames / compare_baseline。
build_server 接受可选的 screen_fn，便于 demo 注入 ScriptedScreen 演示告警闭环；
不注入时走默认 screen_frame（真实 VLM 或 mock 恒不命中）。
"""

from __future__ import annotations

import uuid

from mcp_servers._core import MCPServer

from agents.models import AnalysisTask, Rule
from agents.prescreen.l2_vlm import ScreenFn, frame_to_b64, screen_frame
from agents.prescreen.router import Router


def build_server(screen_fn: ScreenFn | None = None) -> MCPServer:
    router = Router(screen_fn=screen_fn)
    server = MCPServer(name="video-analysis", version="1.0.0", instructions="视频片段 / 帧的视觉初筛分析")

    server.register(
        name="analyze_clip",
        description=(
            "对一段视频片段做三级路由分析（运动检测→抽帧去重→VLM 初筛），"
            "判断是否命中某条巡检规则。参数 clip_path（视频文件路径）、camera_id、"
            "rule（对象，含 id/name/description/severity）。返回结构化 finding 与成本明细。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "clip_path": {"type": "string"},
                "camera_id": {"type": "string"},
                "rule": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "severity": {"type": "string"},
                    },
                    "required": ["id", "name", "description"],
                },
            },
            "required": ["clip_path", "camera_id", "rule"],
        },
        handler=lambda args: _analyze_clip(args, router),
    )

    server.register(
        name="analyze_frames",
        description=(
            "对已抽帧的若干图片路径做逐帧初筛（跳过运动检测与抽帧，直接 VLM）。"
            "参数 frame_paths（图片文件路径列表）、rule。返回逐帧结果。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "frame_paths": {"type": "array", "items": {"type": "string"}},
                "rule": {"type": "object"},
            },
            "required": ["frame_paths", "rule"],
        },
        handler=lambda args: _analyze_frames(args, router),
    )

    server.register(
        name="compare_baseline",
        description=(
            "对比「单帧直连大 VLM」基线与「三级路由」的成本/准确性差异（mock 说明，"
            "真实对比数据见 docs/baseline_comparison.md）。无参数。"
        ),
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=lambda args: {
            "baseline": "单帧直连大 VLM",
            "routed": "三级路由（L0 运动检测 → L1 抽帧去重 → L2 小 VLM）",
            "note": "路由后大模型仅处理少量可疑帧，成本下降一个数量级（详见评测报告，指标待回填）",
        },
    )

    return server


def _rule_from(args: dict) -> Rule:
    r = args["rule"]
    return Rule(id=r["id"], name=r["name"], description=r["description"], severity=r.get("severity", "medium"))


def _analyze_clip(args: dict, router: Router) -> dict:
    rule = _rule_from(args)
    task = AnalysisTask(
        id=f"task_{uuid.uuid4().hex[:12]}",
        camera_id=args["camera_id"],
        rule=rule,
        clip_path=args["clip_path"],
    )
    finding, cost = router.route(task)
    return {"finding": finding.model_dump(), "cost": cost.__dict__}


def _analyze_frames(args: dict, router: Router) -> dict:
    import cv2

    rule = _rule_from(args)
    results = []
    for path in args["frame_paths"]:
        frame = cv2.imread(path)
        if frame is None:
            results.append({"path": path, "error": "cannot read image"})
            continue
        b64 = frame_to_b64(frame)
        # 复用 Router 的初筛函数（默认 screen_frame，或注入的 scripted）
        res = router._screen(rule, b64, args.get("camera_id", ""), 0.0)
        results.append({"path": path, "hit": res.hit, "confidence": res.confidence, "evidence": res.evidence})
    return {"results": results}


def main() -> None:
    build_server().run_stdio()


if __name__ == "__main__":
    main()
