#!/usr/bin/env bash
set -euo pipefail

# EarningsPilot AMD - serve base Qwen plus optional EarningsPilot LoRA adapter via vLLM.
# Run inside the ROCm/vLLM environment. On the DigitalOcean one-click image, use:
#   docker exec -it rocm /bin/bash
# then run this script from the repo root.

BASE_MODEL="${BASE_MODEL:-${AMD_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}}"
ADAPTER_PATH="${ADAPTER_PATH:-artifacts/lora/earningspilot-qwen-7b-lora-10h-forced}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-EarningsPilot-Qwen-7B-LoRA}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_KEY="${AMD_OPENAI_API_KEY:-earningspilot-local-dev-key}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
ENABLE_LORA="${ENABLE_LORA:-auto}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"

if [[ "$ENABLE_LORA" == "auto" ]]; then
  if [[ -d "$ADAPTER_PATH" && -f "$ADAPTER_PATH/adapter_config.json" ]]; then
    ENABLE_LORA="true"
  else
    ENABLE_LORA="false"
  fi
fi

cat <<BANNER
Starting EarningsPilot AMD vLLM server
Base model: ${BASE_MODEL}
Served model name: ${SERVED_MODEL_NAME}
LoRA enabled: ${ENABLE_LORA}
Adapter path: ${ADAPTER_PATH}
Host: ${HOST}
Port: ${PORT}
Max model length: ${MAX_MODEL_LEN}
GPU memory utilization: ${GPU_MEMORY_UTILIZATION}
Endpoint: http://${HOST}:${PORT}/v1/chat/completions
BANNER

args=(
  python -m vllm.entrypoints.openai.api_server
  --host "${HOST}"
  --port "${PORT}"
  --api-key "${API_KEY}"
  --model "${BASE_MODEL}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --trust-remote-code
)

if [[ "$ENABLE_LORA" == "true" ]]; then
  if [[ ! -f "$ADAPTER_PATH/adapter_config.json" ]]; then
    echo "[ERROR] ENABLE_LORA=true but adapter_config.json not found at $ADAPTER_PATH" >&2
    exit 1
  fi
  args+=(
    --enable-lora
    --max-lora-rank "${MAX_LORA_RANK}"
    --lora-modules "${SERVED_MODEL_NAME}=${ADAPTER_PATH}"
  )
fi

exec "${args[@]}"
