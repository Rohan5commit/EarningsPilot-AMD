#!/usr/bin/env bash
set -euo pipefail

# EarningsPilot AMD - serve the base Qwen model through vLLM/OpenAI-compatible API.
# Run inside the ROCm/vLLM environment. For vllm/vllm-openai-rocm images, the
# expected launcher is `vllm serve <model>`, not passing `python -m ...` to the
# Docker image entrypoint.

MODEL_ID="${AMD_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_ID}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_KEY="${AMD_OPENAI_API_KEY:-earningspilot-local-dev-key}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_SERVER_CMD="${VLLM_SERVER_CMD:-auto}"

if [[ "$VLLM_SERVER_CMD" == "auto" ]]; then
  if command -v vllm >/dev/null 2>&1; then
    VLLM_SERVER_CMD="vllm-serve"
  else
    VLLM_SERVER_CMD="python-module"
  fi
fi

cat <<BANNER
Starting EarningsPilot AMD ROCm model server
Model: ${MODEL_ID}
Served model name: ${SERVED_MODEL_NAME}
Host: ${HOST}
Port: ${PORT}
Max model length: ${MAX_MODEL_LEN}
GPU memory utilization: ${GPU_MEMORY_UTILIZATION}
vLLM launcher: ${VLLM_SERVER_CMD}
Endpoint: http://${HOST}:${PORT}/v1/chat/completions
BANNER

common_args=(
  --host "${HOST}"
  --port "${PORT}"
  --api-key "${API_KEY}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --trust-remote-code
)

case "$VLLM_SERVER_CMD" in
  vllm-serve)
    exec vllm serve "${MODEL_ID}" "${common_args[@]}"
    ;;
  python-module)
    exec python -m vllm.entrypoints.openai.api_server \
      --model "${MODEL_ID}" \
      "${common_args[@]}"
    ;;
  *)
    echo "[ERROR] Unknown VLLM_SERVER_CMD=${VLLM_SERVER_CMD}. Use auto, vllm-serve, or python-module." >&2
    exit 1
    ;;
esac
