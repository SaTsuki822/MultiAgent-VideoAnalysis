# GuardEye 多 Agent 基础镜像
# 用法：
#   docker build -f multi_agent/Dockerfile --target planner -t guardeye-planner .
#   docker build -f multi_agent/Dockerfile --target perception -t guardeye-perception .

FROM python:3.11-slim AS base

WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e "."

COPY agents/ ./agents/
COPY mcp_servers/ ./mcp_servers/
COPY multi_agent/ ./multi_agent/
COPY scripts/ ./scripts/
COPY data/ ./data/

# ---- 规划 Agent ----
FROM base AS planner
EXPOSE 8000
CMD ["python", "-m", "multi_agent.planner_agent.workflow"]

# ---- 感知 Agent ----
FROM base AS perception
ENV AGENT_ID=perception-1
CMD ["python", "-m", "multi_agent.perception_agent.workflow", "${AGENT_ID}"]

# ---- 决策 Agent ----
FROM base AS decision
EXPOSE 8000
CMD ["python", "-m", "multi_agent.decision_agent.workflow"]

# ---- 执行 Agent ----
FROM base AS action
EXPOSE 8000
CMD ["python", "-m", "multi_agent.action_agent.workflow"]
