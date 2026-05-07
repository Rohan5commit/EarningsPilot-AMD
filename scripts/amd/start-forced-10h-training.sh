#!/usr/bin/env bash
set -euo pipefail

# EarningsPilot AMD - emergency wall-clock LoRA runner.
# This script is intentionally separate from start-lora-training.sh so GPU hosts
# can run a known-good forced 10h path without accidentally using an older smoke script.

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SOURCE_TRAIN_FILE="${TRAIN_FILE:-training-data/earningspilot-sft-10h.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/lora/earningspilot-qwen-7b-lora-10h-forced}"
EXPANDED_TRAIN_FILE="${EXPANDED_TRAIN_FILE:-artifacts/training/earningspilot-sft-forced-10h.jsonl}"
TRAIN_HOURS="${TRAIN_HOURS:-10}"
MAX_STEPS="${MAX_STEPS:-100000000}"
MIN_TRAIN_ROWS="${MIN_TRAIN_ROWS:-1000000}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-100}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-24}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-auto}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "$SOURCE_TRAIN_FILE" ]]; then
  echo "[ERROR] Source training file not found: $SOURCE_TRAIN_FILE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$(dirname "$EXPANDED_TRAIN_FILE")" artifacts/logs

SOURCE_ROWS="$(wc -l < "$SOURCE_TRAIN_FILE" | tr -d ' ')"
if [[ "$SOURCE_ROWS" -le 0 ]]; then
  echo "[ERROR] Source training file is empty: $SOURCE_TRAIN_FILE" >&2
  exit 1
fi

cat <<INFO
[INFO] EarningsPilot forced 10h training launcher v2
[INFO] BASE_MODEL=$BASE_MODEL
[INFO] SOURCE_TRAIN_FILE=$SOURCE_TRAIN_FILE
[INFO] SOURCE_ROWS=$SOURCE_ROWS
[INFO] EXPANDED_TRAIN_FILE=$EXPANDED_TRAIN_FILE
[INFO] MIN_TRAIN_ROWS=$MIN_TRAIN_ROWS
[INFO] OUTPUT_DIR=$OUTPUT_DIR
[INFO] TRAIN_HOURS=$TRAIN_HOURS
[INFO] MAX_STEPS=$MAX_STEPS
[INFO] CHECKPOINT_STEPS=$CHECKPOINT_STEPS
[INFO] KEEP_CHECKPOINTS=$KEEP_CHECKPOINTS
[INFO] RESUME_FROM_CHECKPOINT=$RESUME_FROM_CHECKPOINT
INFO

# Materialize a large JSONL file on disk. This removes ambiguity around whether
# the Trainer saw the tiny 5-row smoke file or an in-memory repeated dataset.
if [[ ! -f "$EXPANDED_TRAIN_FILE" ]] || [[ "$(wc -l < "$EXPANDED_TRAIN_FILE" | tr -d ' ')" -lt "$MIN_TRAIN_ROWS" ]]; then
  echo "[INFO] Building forced expanded training file..."
  "$PYTHON_BIN" - "$SOURCE_TRAIN_FILE" "$EXPANDED_TRAIN_FILE" "$MIN_TRAIN_ROWS" <<'PY'
import sys
from pathlib import Path
src = Path(sys.argv[1])
out = Path(sys.argv[2])
target = int(sys.argv[3])
rows = src.read_text().splitlines()
if not rows:
    raise SystemExit(f"No rows in {src}")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    for i in range(target):
        f.write(rows[i % len(rows)] + "\n")
print({"source_rows": len(rows), "expanded_rows": target, "output": str(out)})
PY
else
  echo "[INFO] Existing expanded file has $(wc -l < "$EXPANDED_TRAIN_FILE" | tr -d ' ') rows; reusing it."
fi

EXPANDED_ROWS="$(wc -l < "$EXPANDED_TRAIN_FILE" | tr -d ' ')"
if [[ "$EXPANDED_ROWS" -lt "$MIN_TRAIN_ROWS" ]]; then
  echo "[ERROR] Expanded training file too small: $EXPANDED_ROWS < $MIN_TRAIN_ROWS" >&2
  exit 1
fi

echo "[INFO] Installing training dependencies..."
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install "transformers>=4.45" "datasets>=2.20" "peft>=0.12" "trl>=0.10" "accelerate>=0.33"

cat > "$OUTPUT_DIR/run_forced_lora_sft.py" <<'PY'
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

base_model = os.environ["BASE_MODEL"]
train_file = os.environ["EXPANDED_TRAIN_FILE"]
output_dir = os.environ["OUTPUT_DIR"]
max_steps = int(os.environ["MAX_STEPS"])
learning_rate = float(os.environ["LR"])
batch_size = int(os.environ["BATCH_SIZE"])
grad_accum = int(os.environ["GRAD_ACCUM"])
lora_r = int(os.environ["LORA_R"])
lora_alpha = int(os.environ["LORA_ALPHA"])
checkpoint_steps = int(os.environ["CHECKPOINT_STEPS"])
logging_steps = int(os.environ["LOGGING_STEPS"])
keep_checkpoints = int(os.environ["KEEP_CHECKPOINTS"])
resume_setting = os.environ.get("RESUME_FROM_CHECKPOINT", "auto")

os.makedirs(output_dir, exist_ok=True)

def latest_checkpoint(path: str):
    checkpoints = []
    for child in Path(path).glob("checkpoint-*"):
        if child.is_dir():
            try:
                checkpoints.append((int(child.name.split("-")[-1]), child))
            except ValueError:
                continue
    return str(sorted(checkpoints)[-1][1]) if checkpoints else None

resume_checkpoint = None
if resume_setting.lower() == "auto":
    resume_checkpoint = latest_checkpoint(output_dir)
elif resume_setting.lower() not in {"", "false", "none", "0"}:
    resume_checkpoint = resume_setting

print(f"[INFO] Loading materialized train_file={train_file}", flush=True)
dataset = load_dataset("json", data_files=train_file, split="train")
row_count = len(dataset)
print(f"[INFO] Loaded dataset rows={row_count}", flush=True)
if row_count < 100000:
    raise RuntimeError(f"Refusing short run: dataset has only {row_count} rows")

def to_text(ex):
    messages = ex.get("messages", [])
    return {"text": "\n".join(f"{m['role']}: {m['content']}" for m in messages if "role" in m and "content" in m)}

dataset = dataset.map(to_text, desc="Formatting chat rows")

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
if torch.cuda.is_available():
    model = model.to("cuda")

peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
)

args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    learning_rate=learning_rate,
    max_steps=max_steps,
    num_train_epochs=1000000,
    logging_steps=logging_steps,
    save_strategy="steps",
    save_steps=checkpoint_steps,
    save_total_limit=keep_checkpoints,
    fp16=False,
    bf16=True,
    dataloader_pin_memory=False,
    gradient_checkpointing=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    peft_config=peft_config,
    args=args,
)

print(f"[INFO] Starting trainer.train(max_steps={max_steps}, resume={resume_checkpoint})", flush=True)
trainer.train(resume_from_checkpoint=resume_checkpoint)
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

with open(os.path.join(output_dir, "training-summary.json"), "w") as f:
    json.dump({
        "base_model": base_model,
        "train_file": train_file,
        "output_dir": output_dir,
        "dataset_rows": row_count,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "checkpoint_steps": checkpoint_steps,
        "logging_steps": logging_steps,
        "keep_checkpoints": keep_checkpoints,
        "resume_from_checkpoint": resume_checkpoint,
    }, f, indent=2)
PY

export BASE_MODEL EXPANDED_TRAIN_FILE OUTPUT_DIR MAX_STEPS LR BATCH_SIZE GRAD_ACCUM LORA_R LORA_ALPHA CHECKPOINT_STEPS LOGGING_STEPS KEEP_CHECKPOINTS RESUME_FROM_CHECKPOINT

set +e
timeout --signal=INT --kill-after=120s "${TRAIN_HOURS}h" "$PYTHON_BIN" "$OUTPUT_DIR/run_forced_lora_sft.py" 2>&1 | tee -a "artifacts/logs/lora-train-forced-10h.log"
status=${PIPESTATUS[0]}
set -e

cat > "$OUTPUT_DIR/training-run-status.json" <<EOF_STATUS
{
  "exit_code": $status,
  "train_hours": "$TRAIN_HOURS",
  "max_steps": "$MAX_STEPS",
  "source_train_file": "$SOURCE_TRAIN_FILE",
  "source_rows": "$SOURCE_ROWS",
  "expanded_train_file": "$EXPANDED_TRAIN_FILE",
  "expanded_rows": "$EXPANDED_ROWS",
  "checkpoint_steps": "$CHECKPOINT_STEPS",
  "resume_from_checkpoint": "$RESUME_FROM_CHECKPOINT",
  "status_note": "0 means max_steps/completion; 124 means timeout reached; 130 means interrupted after a saved checkpoint can be resumed."
}
EOF_STATUS

echo "[INFO] Forced training process exited with code $status."
echo "[INFO] Adapter artifacts: $OUTPUT_DIR"
echo "[INFO] Checkpoints: $OUTPUT_DIR/checkpoint-*"
echo "[INFO] Log file: artifacts/logs/lora-train-forced-10h.log"
echo "[INFO] Run status: $OUTPUT_DIR/training-run-status.json"
exit "$status"
