#!/usr/bin/env bash
# SGLang 部署小 VLM（L2 初筛模型）。
# 诚实标注：SGLang CLI 随版本有差异，以下命令为当前主流用法，实际以官方文档为准。
# 显存 <16GB 建议换量化版：Qwen/Qwen2.5-VL-7B-Instruct-AWQ。
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-VL-7B-Instruct}"
PORT="${2:-30000}"

echo "[sglang] 启动模型 ${MODEL}，端口 ${PORT} ..."
python -m sglang.launch_server \
  --model-path "${MODEL}" \
  --port "${PORT}" \
  --served-model-name qwen2.5-vl-7b

# 启动后，L2 VLM 通过 OpenAI 兼容端点调用：
#   GUARDEYE_LLM_BACKEND=openai_compatible
#   GUARDEYE_LLM_BASE_URL=http://localhost:30000/v1
#   GUARDEYE_VLM_MODEL=qwen2.5-vl-7b
