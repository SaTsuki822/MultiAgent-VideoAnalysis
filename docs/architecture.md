# GuardEye 架构与设计决策

本文说明各模块职责与关键设计决策（「为什么这么做」而非「做了什么」），供面试深挖时对照。

---

## 1. 分层架构（单 Agent 版）

```
规则配置层（自然语言规则 + SOP）
      ↓
LangGraph 编排层（plan → fetch → dispatch → verify → memory_filter → hitl → report → notify）
      ↓
工具层（5 个 MCP Server，stdio / Streamable HTTP）
      ↓
三级模型路由 + 记忆库（Qdrant / 内存降级）
```

每层边界清晰：编排层不关心工具内部实现（只通过 Toolbox 调 MCP），工具层不关心业务语义
（只暴露 schema + handler），记忆层独立可替换（VectorStore 接口）。

## 2. 多 Agent 协作架构（Phase 8 扩展版）

```
规则配置层
      ↓
协调 Agent（Orchestrator）—— 任务分片、负载均衡、结果聚合、异常恢复
      ↓
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ 规划 Agent    │ 感知 Agent ×N │ 决策 Agent    │ 执行 Agent    │
│ LangGraph 子图 │ LangGraph 子图 │ LangGraph 子图 │ LangGraph 子图 │
│ 轻量文本推理   │ GPU 密集型    │ 中量推理+检索  │ 轻量业务动作   │
└─────────────┴─────────────┴─────────────┴─────────────┘
      ↓
工具层（5 个 MCP Server）
      ↓
三级模型路由 + 记忆库（Qdrant）
```

多 Agent 的核心设计原则：
- **职责隔离**：感知 Agent 只做视频分析，决策 Agent 只做复核定级，各 Agent 不越界；
- **无状态化**：感知 Agent 输入 clip+rule → 输出 finding，不保存中间帧，便于水平扩展；
- **通信解耦**：同步 HTTP/gRPC（规划→协调→决策，需要立即响应）+ 异步 Redis Stream（协调→感知，分析时长不确定）；
- **故障隔离**：单个感知 Agent 崩溃，协调 Agent 重试其他实例，不影响整体流程。

---

## 3. 关键设计决策

### 3.1 为什么用 LangGraph 而不是手写 ReAct 循环 / Dify

- 需要 checkpoint 断点续跑 + interrupt 人工介入 + 显式状态控制，这些自研成本高；
- Dify 类低代码在细粒度状态控制、自定义记忆、评估集成上受限；
- 简单线性场景其实手写循环更轻，这里是因为「断点 + 人工介入 + 并发 reducer」三需求才选状态图。

**但为什么规划 Agent 内部又引入了 ReAct？**

这是「分层架构」思维：
- **外层**（LangGraph 状态图）：固定流程保证可靠性、可审计、可恢复——plan → fetch → dispatch → ... → notify 的走向由代码写死，不受 LLM 临场发挥影响；
- **内层**（规划 Agent 的 ReAct 循环）：规则编译是项目中「最不确定」的环节，陌生规则（如"检查高空作业安全带"）需要查 SOP → 推理 → 确定检测目标，ReAct 的自主探索能力正好覆盖这个场景。

具体实现：
- 已知规则（关键词表命中）→ 确定性编译，零成本；
- 陌生规则 → ReAct 循环（最多 3 步查询 SOP）→ 输出结构化配置；
- ReAct 超步/失败 → 回退关键词表兜底，保证流程不阻塞。

这体现了「该确定的地方确定，该灵活的地方灵活」的架构分层原则。

### 3.2 为什么从单 Agent 演进为多 Agent

**单 Agent 的瓶颈**：
- 感知层（GPU 密集型）与规划/决策（轻量文本）耦合，资源争抢；
- 单进程并行上限受限于单机 GPU 显存；
- 一个节点 Bug 可能导致全图失败。

**多 Agent 的收益**：
- 感知 Agent 独立水平扩展（N 个无状态实例）；
- 故障隔离（单实例崩溃不影响整体）；
- 独立迭代（感知 Agent 升级模型版本不需重启其他 Agent）。

**演进策略**：先单 Agent 验证业务逻辑，再按「无状态化 → 状态分区 → 消息契约 → 独立部署」四步拆分。

### 3.3 MCP vs Function Calling

MCP 是工具层标准化协议（发现 / 调用 / 资源 / prompt 模板），解耦工具提供方与使用方；
Function Calling 是模型能力。本项目自研了 JSON-RPC 核心（`mcp_servers/_core.py`），
覆盖 `initialize` 能力协商、`tools/list` 发现、`tools/call` 调用，支持 stdio 与 HTTP 双传输。

在多 Agent 架构中，MCP 的价值更大：各 Agent 可复用同一套 MCP Server，新增 Agent 无需重复实现工具逻辑。

### 3.4 A2A 通信：为什么同步 + 异步混合

| 通信方向 | 模式 | 原因 |
|----------|------|------|
| 规划 Agent → 协调 Agent | 同步 HTTP | 规则解析结果需立即返回才能下一步分片 |
| 协调 Agent → 感知 Agent | 异步 Redis Stream | 分析时长不确定（10s~2min），HTTP 长连接易超时 |
| 感知 Agent → 协调 Agent | 异步回调 | 分析完后回调，天然支持重试和背压 |
| 协调 Agent → 决策 Agent | 同步 HTTP | 需等待复核结果才能决定是否需要人工确认 |

Redis Stream 选型理由：轻量、与现有 Redis（Checkpoint）可共用、支持 Consumer Group（多感知 Agent 竞争消费）。

### 3.5 三级模型路由的依据

按「处理成本 × 信息密度」分层：
- L0 运动检测：帧差法，零成本，过滤静止画面（信息密度低）；
- L1 抽帧 + aHash 去重：低成本，降帧数（运动区域升帧、相似帧去重）；
- L2 小 VLM 初筛：有成本，粗判命中；
- 大模型二次确认：最贵，只处理少量可疑流量。

每层处理量都记入 `CostBreakdown`，成本账可审计。

### 3.6 跨帧一致性（防 VLM 幻觉）

单帧 VLM 容易幻觉。工程防御：滑动窗口内连续 N 帧命中才升级为告警（`passes_consistency`），
再让大模型带 SOP 依据二次确认。四层防御（schema 约束 / 跨帧一致 / 二次确认 / 记忆抑制）。

### 3.7 误报记忆防污染

误报只降权不删除；新签名积累 N 次（activation_count）才生效；高级别告警永不自动抑制。
检索用「稳定键」（rule_id + camera_id + rule_name），description 只用于解释不参与匹配——
这是特征哈希降级下的可靠选择，真实场景可换语义 embedding。

---

## 4. 模块职责

| 模块 | 职责 |
|---|---|
| `models.py` | 跨层数据模型（pydantic），序列化友好 |
| `state.py` | PatrolState + reducer（findings 追加、alarms 按 id 合并） |
| `llm.py` | LLM/VLM 客户端抽象（ABC + OpenAI 兼容 + mock） |
| `prescreen/` | L0/L1/L2 三级路由与成本记账 |
| `memory/` | 误报记忆、事件记忆、向量存储、embedding |
| `nodes/` | LangGraph 各节点（纯函数，可独立单测） |
| `workflow.py` | 状态图组装 + 纯 Python pipeline 降级 |
| `mcp_servers/` | 自研 MCP 核心 + 5 Server + 客户端 |
| `eval/` | 标注 schema、指标、基线、回归入口 |
| `multi_agent/a2a_protocol/` | A2A 消息格式 + Redis Stream 封装 |
| `multi_agent/orchestrator/` | 协调 Agent：负载均衡、任务分片、结果聚合 |
| `multi_agent/perception_agent/` | 感知 Agent：三级路由，无状态，可水平扩展 |
| `multi_agent/planner_agent/` | 规划 Agent：规则解析、任务分解 |
| `multi_agent/decision_agent/` | 决策 Agent：复核、记忆检索、风险定级 |
| `multi_agent/action_agent/` | 执行 Agent：人工确认、告警工单、报告 |

---

## 5. 「接口 + 真实实现 + mock 降级」清单

| 能力 | 接口 | 真实实现 | mock/降级 |
|---|---|---|---|
| LLM/VLM | `LLMClient` | `OpenAICompatibleClient` | `MockLLMClient`（语义函数走确定性） |
| 向量存储 | `VectorStore` | `QdrantVectorStore` | `InMemoryVectorStore` |
| embedding | `Embedder` | `ModelEmbedder`(TODO) | `HashingEmbedder` |
| 视频初筛 | `ScreenFn` | `screen_frame` | `ScriptedScreen`（数据注入） |
| MCP 传输 | `MCPClient` | HTTP / stdio | InProcess |
| 运行形态 | — | `build_graph()`（LangGraph） | `run_pipeline()`（纯 Python） |
| 多 Agent 通信 | `A2AMessage` | Redis Stream | 内存队列（单测用） |
| 负载均衡 | `LoadBalancer` | Round Robin + 健康检查 | 单实例（降级） |

---

## 6. 已知不足与改进方向

- 评测集规模与多样性有限（真实工地数据难获取）；
- 夜间 / 恶劣天气检出率预期下降；
- 记忆系统长期演化（季节性场景变化）未做；
- **多 Agent 分布式版**：当前为架构验证代码，未在真实大规模场景下测试；
- 事件相机 / 主动抽帧可替代固定采样；
- 感知 Agent 负载均衡当前为简单轮询，生产可扩展为按 GPU 利用率加权。

每点都可对应一条改进方案（面试 Q17/Q19/Q23.5 的素材）。
