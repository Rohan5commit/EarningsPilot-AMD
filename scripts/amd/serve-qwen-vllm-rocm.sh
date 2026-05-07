#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${AMD_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_KEY="${AMD_OPENAI_API_KEY:-earningspilot-local-dev-key}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

cat <<BANNER
Starting EarningsPilot AMD ROCm model server
Model: ${MODEL_ID}
Host: ${HOST}
Port: ${PORT}
Max model length: ${MAX_MODEL_LEN}
GPU memory utilization: ${GPU_MEMORY_UTILIZATION}
Endpoint: http://${HOST}:${PORT}/v1/chat/completions
BANNER

python -m vllm.entrypoints.openai.api_server \
  --host "${HOST}" \
  --port "${PORT}" \
  --api-key "${API_KEY}" \
  --model "${MODEL_ID}" \
  --served-model-name "${MODEL_ID}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code
