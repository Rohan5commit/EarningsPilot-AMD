#!/usr/bin/env bash
set -euo pipefail

# EarningsPilot AMD - quick MI300X utilization sampler.
# Works on bare ROCm hosts or the DigitalOcean vLLM one-click container named "rocm".
# Usage:
#   ./scripts/amd/gpu-utilization.sh
#   SAMPLES=10 INTERVAL=2 ./scripts/amd/gpu-utilization.sh

SAMPLES="${SAMPLES:-5}"
INTERVAL="${INTERVAL:-1}"
CONTAINER_NAME="${CONTAINER_NAME:-rocm}"

run_rocm_smi() {
  if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi "$@"
    return 0
  fi

  if command -v docker >/dev/null 2>&1 && docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    docker exec "$CONTAINER_NAME" rocm-smi "$@"
    return 0
  fi

  return 1
}

if ! run_rocm_smi --showproductname >/dev/null 2>&1; then
  cat >&2 <<ERR
[ERROR] Could not find rocm-smi on the host or inside Docker container '$CONTAINER_NAME'.
Try one of these manually:
  rocm-smi --showuse --showmemuse --showtemp --showpower
  docker exec -it $CONTAINER_NAME rocm-smi --showuse --showmemuse --showtemp --showpower
ERR
  exit 1
fi

echo "[INFO] Sampling AMD GPU utilization: samples=$SAMPLES interval=${INTERVAL}s"
echo "[INFO] If GPU use is near 90-100%, training is compute-bound. If it jumps low/high, tune dataloader or batch settings."
echo

for i in $(seq 1 "$SAMPLES"); do
  echo "===== sample $i/$SAMPLES $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  run_rocm_smi --showproductname --showuse --showmemuse --showtemp --showpower || true
  if [[ "$i" != "$SAMPLES" ]]; then
    sleep "$INTERVAL"
  fi
  echo
done
