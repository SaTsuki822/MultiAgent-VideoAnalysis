"""集中配置。

设计原则（对应面试 Q14「成本数字怎么算」、Q11「路由依据」）：
- 所有魔法值（阈值、模型 ID、集合名、并发上限）收拢到这里，代码内不散落常量；
- 阈值是 Phase 3/6 要扫参的实验对象，集中一处便于评测时统一调整；
- 环境变量 + 默认值双层：裸 clone 后不配置任何东西也能直接跑 mock 闭环，
  配了 .env 则无缝切换到真实后端。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    # ---- LLM / VLM ----
    llm_backend: str = "mock"          # "mock" | "openai_compatible"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""                # 规划 / 复核 / 报告用大模型
    vlm_model: str = ""                # L2 初筛小 VLM（经 SGLang 暴露）

    # ---- 向量库 ----
    vector_backend: str = "memory"     # "memory" | "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    collection_sop: str = "sop_kb"
    collection_fp: str = "false_positive_memory"
    collection_events: str = "event_memory"
    collection_frames: str = "frame_embeddings"
    embedding_dim: int = 1024          # bge-m3 维度；memory 后端用 hash 时忽略

    # ---- 三级路由阈值（Phase 3 实验对象）----
    base_fps: float = 1.0              # L1 基线采样率
    motion_max_fps: float = 4.0        # 运动区域动态升帧上限
    motion_ratio_threshold: float = 0.02  # L0 判定「有运动」的帧差比例阈值
    dedup_threshold: float = 0.95      # CLIP 相邻帧去重阈值

    # ---- verify 跨帧一致性 ----
    consecutive_hit_window: int = 3    # 滑动窗口内连续命中 N 帧才升级

    # ---- 记忆防污染 ----
    fp_similarity_threshold: float = 0.85  # 误报签名检索相似度阈值
    fp_activation_count: int = 2       # 新误报积累 N 次才生效
    never_suppress_severity: str = "high"  # 高级别告警永不自动抑制

    # ---- 并发 ----
    max_concurrency: int = 4           # dispatch 并发上限（受 GPU 显存 / API 限流约束）

    # ---- 自动扩缩容（路径 A：自研轻量 autoscaler，见 multi_agent/orchestrator/autoscaler.py）----
    autoscaler_enabled: bool = False           # 是否启用（默认关，需显式开启）
    autoscaler_min_instances: int = 1          # 最小感知 Agent 实例数
    autoscaler_max_instances: int = 4          # 最大感知 Agent 实例数（受 GPU 供给约束）
    autoscaler_scale_up_backlog: int = 4       # 积压 >= 该值则扩容
    autoscaler_scale_down_backlog: int = 0     # 积压 <= 该值则缩容
    autoscaler_poll_interval_sec: float = 5.0  # 采样间隔（秒）
    autoscaler_cooldown_sec: float = 10.0      # 扩缩容冷却窗口（秒，防抖）

    # ---- 异常处理 LLM 顾问（Phase 2）----
    exception_llm_advisor_enabled: bool = False   # 是否启用 LLM 异常顾问
    exception_llm_advisor_threshold: float = 0.7  # LLM 建议置信度阈值，低于此值 fallback 到 L1

    # ---- 路径 ----
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")

    @classmethod
    def from_env(cls) -> "Settings":
        """从环境变量 / .env 覆盖默认值，未设置则用 dataclass 默认值。"""
        load_dotenv(PROJECT_ROOT / ".env")
        env = os.environ
        s = cls()
        for field_name in cls.__dataclass_fields__:
            key = f"GUARDEYE_{field_name.upper()}"
            if key not in env:
                continue
            current = getattr(s, field_name)
            raw = env[key]
            if isinstance(current, bool):
                setattr(s, field_name, raw.lower() in {"1", "true", "yes"})
            elif isinstance(current, int):
                setattr(s, field_name, int(raw))
            elif isinstance(current, float):
                setattr(s, field_name, float(raw))
            elif isinstance(current, Path):
                setattr(s, field_name, Path(raw))
            else:
                setattr(s, field_name, raw)
        return s


_settings: Settings | None = None


def get_settings() -> Settings:
    """进程级单例。测试可用 reset_settings() 强制重载。"""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def reset_settings() -> None:
    """清空缓存，下次 get_settings() 重新读取环境。供测试与热更新使用。"""
    global _settings
    _settings = None
