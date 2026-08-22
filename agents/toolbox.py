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

    def get_clip(self, camera_id: str, start: str, end: str) -> dict:
        return self.camera.call_tool("get_clip", {"camera_id": camera_id, "start_time": start, "end_time": end})

    # ---- video-analysis ----
    def analyze_clip(self, clip_path: str, camera_id: str, rule: dict) -> dict:
        return self.video.call_tool("analyze_clip", {"clip_path": clip_path, "camera_id": camera_id, "rule": rule})

    # ---- knowledge ----
    def search_sop(self, query: str, limit: int = 3) -> list[dict]:
        return self.knowledge.call_tool("search_sop", {"query": query, "limit": limit})["results"]

    def search_similar_events(self, query: str, limit: int = 3) -> list[dict]:
        return self.knowledge.call_tool("search_similar_events", {"query": query, "limit": limit})["results"]

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
    def create_ticket(self, alarm_id: str, assignee: str) -> dict:
        return self.ticket.call_tool("create_ticket", {"alarm_id": alarm_id, "assignee": assignee})

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
