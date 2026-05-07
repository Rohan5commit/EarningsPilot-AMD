#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-artifacts/lora/earningspilot-qwen-7b-lora}"
LOG_FILE="${LOG_FILE:-artifacts/logs/lora-train.log}"

latest_checkpoint=""
if compgen -G "$OUTPUT_DIR/checkpoint-*" >/dev/null; then
  latest_checkpoint="$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
fi

if [[ -z "$latest_checkpoint" ]]; then
  echo "[progress] No checkpoints found yet in $OUTPUT_DIR"
else
  echo "[progress] Latest checkpoint: $latest_checkpoint"
  if [[ -f "$latest_checkpoint/trainer_state.json" ]]; then
    python3 - "$latest_checkpoint/trainer_state.json" <<'PY'
import json, sys
from pathlib import Path
state_path = Path(sys.argv[1])
state = json.loads(state_path.read_text())
step = state.get("global_step")
max_steps = state.get("max_steps")
progress = None
if isinstance(step, int) and isinstance(max_steps, int) and max_steps:
    progress = round(step / max_steps * 100, 2)
last = (state.get("log_history") or [{}])[-1]
print(json.dumps({
    "global_step": step,
    "max_steps": max_steps,
    "progress_percent": progress,
    "epoch": state.get("epoch"),
    "best_metric": state.get("best_metric"),
    "last_log": last,
}, indent=2))
PY
  fi
fi

if [[ -f "$OUTPUT_DIR/training-summary.json" ]]; then
  echo "[progress] Final training summary exists: $OUTPUT_DIR/training-summary.json"
  cat "$OUTPUT_DIR/training-summary.json"
fi

if [[ -f "$LOG_FILE" ]]; then
  echo "[progress] Last 30 log lines from $LOG_FILE"
  tail -n 30 "$LOG_FILE"
else
  echo "[progress] Log file not found yet: $LOG_FILE"
fi
