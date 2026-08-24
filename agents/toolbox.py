"""工具层访问封装（Toolbox）。

为什么封装一层（对应面试 Q5「MCP 解耦工具提供方与使用方」）：
- 节点通过 Toolbox 调用 5 个 MCP Server，而非直接 import server 实现，
  保持「工具提供方 vs 使用方」的协议级解耦；
- 测试可注入 fake Toolbox，无需启动任何进程 / 服务；
- 默认 InProcess 传输（直接持有 server 对象），demo 与单测零外部依赖可跑。
"""

from __future__ import annotations

from mcp_servers import alarm, camera_registry, knowledge, ticket, video_analysis
from mcp_servers.mcp_client import MCPClient

from agents.prescreen.l2_vlm import ScreenFn


class Toolbox:
    """聚合 5 个 MCP Server 的客户端，节点只依赖这一层。"""

    def __init__(self, camera: MCPClient, video: MCPClient, alarm_cli: MCPClient, ticket_cli: MCPClient, knowledge_cli: MCPClient):
        self.camera = camera
        self.video = video
        self.alarm = alarm_cli
        self.ticket = ticket_cli
        self.knowledge = knowledge_cli

    # ---- camera-registry ----
    def list_cameras(self) -> list[dict]:
        return self.camera.call_tool("list_cameras", {})["cameras"]

    def get_metadata(self, camera_id: str) -> dict:
        return self.camera.call_tool("get_metadata", {"camera_id": camera_id})

    def get_clip(self, camera_id: str, start: str, end: str) -> dict:
        return self.camera.call_tool("get_clip", {"camera_id": camera_id, "start_time": start, "end_time": end})

    def update_camera(self, camera_id: str, name: str | None = None, area: str | None = None, rtsp_url: str | None = None) -> dict:
        args = {"camera_id": camera_id}
        if name is not None:
            args["name"] = name
        if area is not None:
            args["area"] = area
        if rtsp_url is not None:
            args["rtsp_url"] = rtsp_url
        return self.camera.call_tool("update_camera", args)

    # ---- video-analysis ----
    def analyze_clip(self, clip_path: str, camera_id: str, rule: dict) -> dict:
        return self.video.call_tool("analyze_clip", {"clip_path": clip_path, "camera_id": camera_id, "rule": rule})

    def compare_baseline(self, clip_path: str | None = None, rule: dict | None = None, camera_id: str | None = None) -> dict:
        args: dict = {}
        if clip_path is not None:
            args["clip_path"] = clip_path
        if rule is not None:
            args["rule"] = rule
        if camera_id is not None:
            args["camera_id"] = camera_id
        return self.video.call_tool("compare_baseline", args)

    # ---- knowledge ----
    def search_sop(self, query: str, limit: int = 3) -> list[dict]:
        return self.knowledge.call_tool("search_sop", {"query": query, "limit": limit})["results"]

    def search_similar_events(self, query: str, limit: int = 3) -> list[dict]:
        return self.knowledge.call_tool("search_similar_events", {"query": query, "limit": limit})["results"]

    def list_tools(self) -> list[dict]:
        """返回规则编译可用的工具 schema（MCP tools/list 格式），供 Function Calling 使用。"""
        return [t for t in self.knowledge.list_tools() if t["name"] == "search_sop"]

    # ---- alarm ----
    def create_alarm(self, camera_id: str, rule_id: str, rule_name: str, severity: str, confidence: float, evidence: list) -> dict:
        return self.alarm.call_tool(
            "create_alarm",
            {
                "camera_id": camera_id,
                "rule_id": rule_id,
                "rule_name": rule_name,
                "severity": severity,
                "confidence": confidence,
                "evidence": evidence,
            },
        )

    def suppress_alarm(self, alarm_id: str, reason: str) -> dict:
        return self.alarm.call_tool("suppress_similar", {"alarm_id": alarm_id, "reason": reason})

    # ---- ticket ----
    def create_ticket(self, alarm_id: str, assignee: str, priority: str = "medium", description: str = "") -> dict:
        args = {"alarm_id": alarm_id, "assignee": assignee}
        if priority:
            args["priority"] = priority
        if description:
            args["description"] = description
        return self.ticket.call_tool("create_ticket", args)

    def query_tickets(self, alarm_id: str | None = None, assignee: str | None = None, status: str | None = None) -> list[dict]:
        args: dict = {}
        if alarm_id is not None:
            args["alarm_id"] = alarm_id
        if assignee is not None:
            args["assignee"] = assignee
        if status is not None:
            args["status"] = status
        return self.ticket.call_tool("query_tickets", args).get("tickets", [])

    def update_ticket(self, ticket_id: str, status: str | None = None, note: str | None = None) -> dict:
        args: dict = {"ticket_id": ticket_id}
        if status is not None:
            args["status"] = status
        if note is not None:
            args["note"] = note
        return self.ticket.call_tool("update_ticket", args)

    def notify(self, channel: str, payload: dict) -> dict:
        return self.ticket.call_tool("notify", {"channel": channel, "payload": payload})


def build_default_toolbox(screen_fn: ScreenFn | None = None) -> Toolbox:
    """用 InProcess 传输组装 5 个 Server 的默认 Toolbox。

    screen_fn 用于注入 ScriptedScreen（demo 演示告警闭环）；生产传 None 走真实/默认初筛。
    """
    return Toolbox(
        camera=MCPClient(camera_registry.build_server()),
        video=MCPClient(video_analysis.build_server(screen_fn=screen_fn)),
        alarm_cli=MCPClient(alarm.build_server()),
        ticket_cli=MCPClient(ticket.build_server()),
        knowledge_cli=MCPClient(knowledge.build_server()),
    )
