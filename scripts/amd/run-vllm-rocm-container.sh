#!/usr/bin/env bash
set -euo pipefail

# EarningsPilot AMD - run vLLM in a fresh ROCm Docker container while bypassing
# the image entrypoint. This avoids the vllm/vllm-openai-rocm entrypoint parsing
# the entire command string as one positional argument.
#
# Run on the AMD GPU host from the repo root:
#   ./scripts/amd/run-vllm-rocm-container.sh

VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai-rocm:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-earningspilot-vllm}"
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
HF_HOME_DIR="${HF_HOME_DIR:-$HOME/.cache/huggingface}"

if [[ "$ENABLE_LORA" == "auto" ]]; then
  if [[ -f "$ADAPTER_PATH/adapter_config.json" ]]; then
    ENABLE_LORA="true"
  else
    ENABLE_LORA="false"
  fi
fi

repo_root="$(pwd)"
mkdir -p artifacts/runtime "$HF_HOME_DIR"
inner_script="artifacts/runtime/start-vllm-inside-container.sh"
cat > "$inner_script" <<'INNER'
#!/usr/bin/env bash
set -euo pipefail

args=(
  vllm serve "${BASE_MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --api-key "${API_KEY}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --trust-remote-code
)

if [[ "${ENABLE_LORA}" == "true" ]]; then
  if [[ ! -f "${ADAPTER_PATH}/adapter_config.json" ]]; then
    echo "[ERROR] ENABLE_LORA=true but adapter_config.json not found at ${ADAPTER_PATH}" >&2
    exit 1
  fi
  args+=(
    --enable-lora
    --max-lora-rank "${MAX_LORA_RANK}"
    --lora-modules "${SERVED_MODEL_NAME}=${ADAPTER_PATH}"
  )
fi

echo "[INFO] Starting vLLM with argv array:"
printf '  %q' "${args[@]}"
echo
exec "${args[@]}"
INNER
chmod +x "$inner_script"

cat <<BANNER
Starting fresh ROCm vLLM Docker container
Image: ${VLLM_IMAGE}
Container: ${CONTAINER_NAME}
Base model: ${BASE_MODEL}
Served model name: ${SERVED_MODEL_NAME}
LoRA enabled: ${ENABLE_LORA}
Adapter path: ${ADAPTER_PATH}
Port: ${PORT}
Entrypoint bypass: --entrypoint /bin/bash
BANNER

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "[INFO] Removing existing container $CONTAINER_NAME"
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

exec docker run --rm -it \
  --name "$CONTAINER_NAME" \
  --entrypoint /bin/bash \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size=16g \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -p "${PORT}:${PORT}" \
  -e BASE_MODEL="${BASE_MODEL}" \
  -e ADAPTER_PATH="${ADAPTER_PATH}" \
  -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
  -e HOST="${HOST}" \
  -e PORT="${PORT}" \
  -e API_KEY="${API_KEY}" \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  -e ENABLE_LORA="${ENABLE_LORA}" \
  -e MAX_LORA_RANK="${MAX_LORA_RANK}" \
  -e HF_HOME=/root/.cache/huggingface \
  -v "${repo_root}:${repo_root}" \
  -v "${HF_HOME_DIR}:/root/.cache/huggingface" \
  -w "${repo_root}" \
  "$VLLM_IMAGE" \
  "${repo_root}/${inner_script}"
