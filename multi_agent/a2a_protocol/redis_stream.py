"""Redis Stream 异步通信封装。

基于 Redis Stream 实现 Agent 间的异步消息分发：
- 生产者（协调 Agent）向 Stream 写入任务；
- 消费者（感知 Agent）以消费者组方式竞争消费，天然实现负载均衡；
- ACK 机制保证至少一次交付；
- PEL（Pending Entries List）便于追踪未完成任务。

为什么选 Redis Stream：
- 轻量，与现有 Redis（Checkpoint / 状态存储）可共用；
- 支持 Consumer Group，多感知 Agent 实例可竞争消费；
- 支持 ACK 和 PEL，便于故障恢复。
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from a2a_protocol.message_schema import A2AMessage, A2AResult


def _get_redis():
    """懒加载 redis 客户端，避免导入时无 redis 就报错。"""
    import redis

    return redis.Redis(host="localhost", port=6379, decode_responses=True)


class RedisStreamClient:
    """Redis Stream 客户端：生产和消费 A2A 消息。"""

    def __init__(self, stream_name: str = "guardeye_tasks", group_name: str = "perception_group", redis_client=None):
        self.r = redis_client or _get_redis()
        self.stream_name = stream_name
        self.group_name = group_name

    # ---- 生产者 API（协调 Agent 用）----

    def send_task(self, message: A2AMessage) -> str:
        """向 Stream 发送任务，返回消息 ID。"""
        msg_dict = message.to_dict()
        # Redis Stream 的 xadd 要求 value 为字符串
        entry = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v) for k, v in msg_dict.items()}
        msg_id = self.r.xadd(self.stream_name, entry)
        return msg_id

    def broadcast_task(self, messages: list[A2AMessage]) -> list[str]:
        """批量发送任务（pipeline 优化）。"""
        pipe = self.r.pipeline()
        for msg in messages:
            entry = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v) for k, v in msg.to_dict().items()}
            pipe.xadd(self.stream_name, entry)
        return pipe.execute()

    # ---- 消费者 API（感知 Agent 用）----

    def ensure_group(self, consumer_name: str):
        """确保消费者组存在。"""
        try:
            self.r.xgroup_create(self.stream_name, self.group_name, id="0", mkstream=True)
        except Exception:
            pass  # 组已存在

    def consume_task(
        self,
        consumer_name: str,
        count: int = 1,
        block_ms: int = 5000,
    ) -> list[tuple[str, A2AMessage]]:
        """从消费者组读取任务，返回 (stream_id, message) 列表。

        调用方处理完任务后，必须调用 ack_task(stream_id)。
        """
        self.ensure_group(consumer_name)
        entries = self.r.xreadgroup(
            groupname=self.group_name,
            consumername=consumer_name,
            streams={self.stream_name: ">"},
            count=count,
            block=block_ms,
        )
        results: list[tuple[str, A2AMessage]] = []
        for stream_key, items in entries or []:
            for stream_id, fields in items:
                # fields 是 dict，值可能是 json 字符串
                data = {}
                for k, v in fields.items():
                    try:
                        data[k] = json.loads(v)
                    except (json.JSONDecodeError, TypeError):
                        data[k] = v
                results.append((stream_id, A2AMessage.from_dict(data)))
        return results

    def ack_task(self, stream_id: str):
        """确认任务已完成。"""
        self.r.xack(self.stream_name, self.group_name, stream_id)

    def get_pending(self, consumer_name: str) -> list[tuple[str, str, int, int]]:
        """获取本消费者的 PEL（未确认任务）。

        返回 [(stream_id, consumer_name, idle_ms, delivery_count), ...]
        """
        return self.r.xpending_range(
            self.stream_name,
            self.group_name,
            min="-",
            max="+",
            count=100,
            consumername=consumer_name,
        ) or []

    # ---- 回调结果（感知 Agent → 协调 Agent）----

    def send_result(self, result: A2AResult) -> str:
        """将结果写回回调 Stream（topic = {from_agent}_results）。"""
        result_stream = f"{result.from_agent}_results"
        entry = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v) for k, v in result.to_dict().items()}
        return self.r.xadd(result_stream, entry)

    def consume_results(self, agent_name: str, count: int = 100, block_ms: int = 1000) -> list[A2AResult]:
        """协调 Agent 读取各感知 Agent 返回的结果。"""
        result_stream = f"{agent_name}_results"
        entries = self.r.xread({result_stream: "0-0"}, count=count, block=block_ms)
        results: list[A2AResult] = []
        for stream_key, items in entries or []:
            for stream_id, fields in items:
                data = {}
                for k, v in fields.items():
                    try:
                        data[k] = json.loads(v)
                    except (json.JSONDecodeError, TypeError):
                        data[k] = v
                results.append(A2AResult.from_dict(data))
                # 读取后删除，避免堆积
                self.r.xdel(result_stream, stream_id)
        return results

    def close(self):
        self.r.close()
