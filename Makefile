.PHONY: install test demo demo-multi lint qdrant redis infra

# 安装（含开发依赖）
install:
	pip install -e ".[dev]"

# 运行单元测试（不依赖 GPU / API key）
test:
	pytest

# 跑端到端 demo（单 Agent，合成视频 + 误报记忆闭环）
demo:
	python scripts/demo.py

# 跑多 Agent 分布式 demo（1 协调 + 2 感知 Agent 线程）
demo-multi:
	python scripts/demo_multi_agent.py

# 起基础设施（Qdrant + Redis）
infra:
	docker compose up -d qdrant redis

# 起向量库（memory 后端无需）
qdrant:
	docker compose up -d qdrant

# 起 Redis（多 Agent 通信必备）
redis:
	docker compose up -d redis

# 起完整多 Agent 集群（需构建镜像）
multi-agent-up:
	docker compose --profile multi-agent up -d

# 停多 Agent 集群
multi-agent-down:
	docker compose --profile multi-agent down
