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
            "对比「单帧直连大 VLM」基线与「三级路由」的成本差异。"
            "可选参数 clip_path + rule，提供则实际分析并输出真实对比；"
            "不提供则基于成本模型输出理论估算。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "clip_path": {"type": "string", "description": "视频路径（可选）"},
                "rule": {"type": "object", "description": "规则对象（可选，与 clip_path 同时提供时走真实分析）"},
                "camera_id": {"type": "string", "description": "摄像头 ID（可选，真实分析时必填）"},
            },
        },
        handler=lambda args: _compare_baseline(args, router),
    )

    return server


def _compare_baseline(args: dict, router: Router) -> dict:
    """对比基线与三级路由的成本。"""
    clip_path = args.get("clip_path")
    rule_dict = args.get("rule")
    camera_id = args.get("camera_id", "compare_cam")

    # 成本模型常量（与 Router 对齐）
    est_tokens_per_frame = router.est_tokens_per_frame  # 默认 600
    cost_per_token = router.cost_per_token or 0.003     # 若 mock 为 0，用占位单价 0.003 元/token 估算

    if clip_path and rule_dict:
        # 走真实分析，获取三级路由实际成本
        from agents.models import AnalysisTask, Rule
        rule = Rule(
            id=rule_dict.get("id", "compare_rule"),
            name=rule_dict.get("name", "compare"),
            description=rule_dict.get("description", ""),
            severity=rule_dict.get("severity", "medium"),
        )
        task = AnalysisTask(
            id=f"task_compare_{uuid.uuid4().hex[:8]}",
            camera_id=camera_id,
            rule=rule,
            clip_path=clip_path,
        )
        finding, cost = router.route(task)
        routed_frames = cost.l2_screened_frames
        total_frames = cost.l1_kept_frames
        if total_frames == 0:
            total_frames = routed_frames  # 兜底
    else:
        # 无视频路径，基于典型场景理论估算
        total_frames = 300   # 假设 10 秒 30fps = 300 帧
        routed_frames = 30   # 三级路由后保留约 10%

    # 基线成本 = 所有帧直连大模型
    baseline_tokens = total_frames * est_tokens_per_frame
    baseline_cost = baseline_tokens * cost_per_token

    # 三级路由成本 = 只处理保留帧
    routed_tokens = routed_frames * est_tokens_per_frame
    routed_cost = routed_tokens * cost_per_token

    savings_ratio = (1 - routed_cost / baseline_cost) if baseline_cost > 0 else 0.0

    return {
        "mode": "real" if (clip_path and rule_dict) else "estimated",
        "baseline": {
            "strategy": "单帧直连大 VLM",
            "total_frames": total_frames,
            "screened_frames": total_frames,
            "est_tokens": baseline_tokens,
            "est_cost_cny": round(baseline_cost, 4),
        },
        "routed": {
            "strategy": "三级路由（L0 运动检测 → L1 抽帧去重 → L2 小 VLM）",
            "total_frames": total_frames,
            "screened_frames": routed_frames,
            "l0_filtered": total_frames - routed_frames,  # 简化估算
            "est_tokens": routed_tokens,
            "est_cost_cny": round(routed_cost, 4),
        },
        "savings": {
            "saved_frames": total_frames - routed_frames,
            "saved_tokens": baseline_tokens - routed_tokens,
            "saved_cost_cny": round(baseline_cost - routed_cost, 4),
            "savings_ratio": f"{savings_ratio:.1%}",
        },
        "note": "cost_per_token 为估算单价，真实成本按实际模型定价调整",
    }


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
