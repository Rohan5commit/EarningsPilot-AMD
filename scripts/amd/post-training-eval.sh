#!/usr/bin/env bash
set -euo pipefail

# EarningsPilot AMD - post-training evaluation artifact collector.
# Run on the GPU host after stopping or pausing LoRA training.
# It collects checkpoint metadata, training progress, optional app eval, optional AMD endpoint benchmark, and an archive.

OUTPUT_DIR="${OUTPUT_DIR:-artifacts/lora/earningspilot-qwen-7b-lora-10h-forced}"
LOG_FILE="${LOG_FILE:-artifacts/logs/lora-train-forced-10h.log}"
EVAL_DIR="${EVAL_DIR:-artifacts/eval/$(date -u +%Y%m%dT%H%M%SZ)}"
ARCHIVE_ADAPTER="${ARCHIVE_ADAPTER:-true}"
RUN_GPU_UTIL="${RUN_GPU_UTIL:-true}"
RUN_APP_EVAL="${RUN_APP_EVAL:-true}"
RUN_AMD_BENCHMARK="${RUN_AMD_BENCHMARK:-true}"
BENCHMARK_RUNS="${BENCHMARK_RUNS:-3}"
AUTO_RESOLVE_PATHS="${AUTO_RESOLVE_PATHS:-true}"

requested_output_dir="$OUTPUT_DIR"
requested_log_file="$LOG_FILE"

resolve_output_dir() {
  if [[ -d "$OUTPUT_DIR" || "$AUTO_RESOLVE_PATHS" != "true" ]]; then
    return
  fi

  local candidates=()
  if [[ "$OUTPUT_DIR" != *-forced ]]; then
    candidates+=("${OUTPUT_DIR}-forced")
  fi
  candidates+=(
    "artifacts/lora/earningspilot-qwen-7b-lora-10h-forced"
    "artifacts/lora/earningspilot-qwen-7b-lora-10h"
  )

  if [[ -d artifacts/lora ]]; then
    while IFS= read -r adapter_config; do
      candidates+=("$(dirname "$adapter_config")")
    done < <(find artifacts/lora -type f -name adapter_config.json | sort -V -r)
  fi

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate" ]]; then
      echo "[WARN] OUTPUT_DIR not found: $requested_output_dir" >&2
      echo "[WARN] Auto-resolved OUTPUT_DIR to: $candidate" >&2
      OUTPUT_DIR="$candidate"
      return
    fi
  done
}

resolve_log_file() {
  if [[ -f "$LOG_FILE" || "$AUTO_RESOLVE_PATHS" != "true" ]]; then
    return
  fi

  local candidates=()
  if [[ "$OUTPUT_DIR" == *-forced* ]]; then
    candidates+=("artifacts/logs/lora-train-forced-10h.log")
  fi
  candidates+=(
    "artifacts/logs/lora-train.log"
    "artifacts/logs/lora-train-forced-10h.log"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      echo "[WARN] LOG_FILE not found: $requested_log_file" >&2
      echo "[WARN] Auto-resolved LOG_FILE to: $candidate" >&2
      LOG_FILE="$candidate"
      return
    fi
  done
}

resolve_output_dir
resolve_log_file

mkdir -p "$EVAL_DIR"

echo "[INFO] EarningsPilot post-training eval"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"
echo "[INFO] LOG_FILE=$LOG_FILE"
echo "[INFO] EVAL_DIR=$EVAL_DIR"
echo "[INFO] AUTO_RESOLVE_PATHS=$AUTO_RESOLVE_PATHS"

if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "[ERROR] Adapter output directory not found: $OUTPUT_DIR" >&2
  echo "[ERROR] Set OUTPUT_DIR to the directory that contains adapter_config.json or checkpoint-* directories." >&2
  echo "[ERROR] To disable path fallback attempts, set AUTO_RESOLVE_PATHS=false." >&2
  exit 1
fi

checkpoint_count="0"
latest_checkpoint=""
if compgen -G "$OUTPUT_DIR/checkpoint-*" >/dev/null; then
  find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V > "$EVAL_DIR/checkpoints.txt"
  checkpoint_count="$(wc -l < "$EVAL_DIR/checkpoints.txt" | tr -d ' ')"
  latest_checkpoint="$(tail -n 1 "$EVAL_DIR/checkpoints.txt")"
else
  : > "$EVAL_DIR/checkpoints.txt"
fi

echo "[INFO] checkpoint_count=$checkpoint_count"
echo "[INFO] latest_checkpoint=${latest_checkpoint:-none}"

if [[ -f "$latest_checkpoint/trainer_state.json" ]]; then
  cp "$latest_checkpoint/trainer_state.json" "$EVAL_DIR/latest-trainer-state.json"
  python3 - "$latest_checkpoint/trainer_state.json" > "$EVAL_DIR/latest-trainer-summary.json" <<'PY'
import json, sys
from pathlib import Path
state = json.loads(Path(sys.argv[1]).read_text())
log_history = state.get("log_history") or []
last = log_history[-1] if log_history else {}
print(json.dumps({
    "global_step": state.get("global_step"),
    "max_steps": state.get("max_steps"),
    "epoch": state.get("epoch"),
    "best_metric": state.get("best_metric"),
    "last_log": last,
}, indent=2))
PY
fi

if [[ -f "$OUTPUT_DIR/training-summary.json" ]]; then
  cp "$OUTPUT_DIR/training-summary.json" "$EVAL_DIR/training-summary.json"
fi
if [[ -f "$OUTPUT_DIR/training-run-status.json" ]]; then
  cp "$OUTPUT_DIR/training-run-status.json" "$EVAL_DIR/training-run-status.json"
fi

if [[ -f "$LOG_FILE" ]]; then
  tail -n 200 "$LOG_FILE" > "$EVAL_DIR/training-log-tail.txt"
else
  echo "[WARN] Training log not found: $LOG_FILE" | tee "$EVAL_DIR/training-log-tail.txt"
fi

if [[ -x ./scripts/amd/training-progress.sh ]]; then
  OUTPUT_DIR="$OUTPUT_DIR" LOG_FILE="$LOG_FILE" ./scripts/amd/training-progress.sh | tee "$EVAL_DIR/training-progress.txt" || true
fi

if [[ "$RUN_GPU_UTIL" == "true" && -x ./scripts/amd/gpu-utilization.sh ]]; then
  SAMPLES="${GPU_UTIL_SAMPLES:-5}" INTERVAL="${GPU_UTIL_INTERVAL:-1}" ./scripts/amd/gpu-utilization.sh | tee "$EVAL_DIR/gpu-utilization.txt" || true
fi

if [[ "$RUN_APP_EVAL" == "true" ]]; then
  if [[ -n "${EARNINGSPILOT_BASE_URL:-}" ]]; then
    echo "[INFO] Running app eval against $EARNINGSPILOT_BASE_URL"
    EARNINGSPILOT_BASE_URL="$EARNINGSPILOT_BASE_URL" npm run eval:sample | tee "$EVAL_DIR/app-eval.json"
  else
    echo "[WARN] Skipping app eval because EARNINGSPILOT_BASE_URL is not set." | tee "$EVAL_DIR/app-eval.json"
  fi
fi

if [[ "$RUN_AMD_BENCHMARK" == "true" ]]; then
  if [[ -n "${AMD_OPENAI_BASE_URL:-}" ]]; then
    echo "[INFO] Running AMD endpoint benchmark against $AMD_OPENAI_BASE_URL"
    BENCHMARK_RUNS="$BENCHMARK_RUNS" npm run benchmark:amd | tee "$EVAL_DIR/amd-benchmark.json"
  else
    echo "[WARN] Skipping AMD benchmark because AMD_OPENAI_BASE_URL is not set." | tee "$EVAL_DIR/amd-benchmark.json"
  fi
fi

archive_path=""
if [[ "$ARCHIVE_ADAPTER" == "true" ]]; then
  archive_path="$EVAL_DIR/$(basename "$OUTPUT_DIR")-adapter-and-logs.tar.gz"
  tar -czf "$archive_path" "$OUTPUT_DIR" "$LOG_FILE" 2>/dev/null || tar -czf "$archive_path" "$OUTPUT_DIR"
  echo "[INFO] archive=$archive_path"
fi

python3 - "$EVAL_DIR" "$OUTPUT_DIR" "$LOG_FILE" "$checkpoint_count" "$latest_checkpoint" "$archive_path" <<'PY'
import json, sys
from pathlib import Path

eval_dir, output_dir, log_file, checkpoint_count, latest_checkpoint, archive_path = sys.argv[1:]
manifest = {
    "eval_dir": eval_dir,
    "output_dir": output_dir,
    "log_file": log_file,
    "checkpoint_count": int(checkpoint_count),
    "latest_checkpoint": latest_checkpoint or None,
    "archive_path": archive_path or None,
    "files": sorted(p.name for p in Path(eval_dir).iterdir()),
}
Path(eval_dir, "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
PY

echo "[INFO] Post-training eval artifacts written to $EVAL_DIR"
