#!/usr/bin/env bash
set -euo pipefail

# EarningsPilot AMD - time-boxed LoRA/QLoRA runner for AMD MI300X hosts
# Usage:
#   TRAIN_HOURS=10 MAX_STEPS=100000 BASE_MODEL=Qwen/Qwen2.5-7B-Instruct ./scripts/amd/start-lora-training.sh

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-training-data/earningspilot-sft-10h.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/lora/earningspilot-qwen-7b-lora-10h}"
TRAIN_HOURS="${TRAIN_HOURS:-10}"
MAX_STEPS="${MAX_STEPS:-100000}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-4}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-250}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-8}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-auto}"
MIN_TRAIN_ROWS="${MIN_TRAIN_ROWS:-250000}"

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "[ERROR] Training file not found: $TRAIN_FILE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p artifacts/logs

echo "[INFO] Starting time-boxed LoRA training"
echo "[INFO] BASE_MODEL=$BASE_MODEL"
echo "[INFO] TRAIN_FILE=$TRAIN_FILE"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"
echo "[INFO] TRAIN_HOURS=$TRAIN_HOURS"
echo "[INFO] MAX_STEPS=$MAX_STEPS"
echo "[INFO] CHECKPOINT_STEPS=$CHECKPOINT_STEPS"
echo "[INFO] KEEP_CHECKPOINTS=$KEEP_CHECKPOINTS"
echo "[INFO] RESUME_FROM_CHECKPOINT=$RESUME_FROM_CHECKPOINT"
echo "[INFO] MIN_TRAIN_ROWS=$MIN_TRAIN_ROWS"
echo "[INFO] TRAIN_FILE_ROWS=$(wc -l < "$TRAIN_FILE" | tr -d ' ')"

python3 -m pip install --upgrade pip
python3 -m pip install "transformers>=4.45" "datasets>=2.20" "peft>=0.12" "trl>=0.10" "accelerate>=0.33"

cat > "$OUTPUT_DIR/run_lora_sft.py" <<'PY'
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import torch
from peft import LoraConfig
from trl import SFTTrainer
import os
from pathlib import Path

base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
train_file = os.environ.get("TRAIN_FILE", "training-data/earningspilot-sft-10h.jsonl")
output_dir = os.environ.get("OUTPUT_DIR", "artifacts/lora/earningspilot-qwen-7b-lora-10h")
max_steps = int(os.environ.get("MAX_STEPS", "100000"))
learning_rate = float(os.environ.get("LR", "2e-4"))
batch_size = int(os.environ.get("BATCH_SIZE", "1"))
grad_accum = int(os.environ.get("GRAD_ACCUM", "8"))
lora_r = int(os.environ.get("LORA_R", "16"))
lora_alpha = int(os.environ.get("LORA_ALPHA", "32"))
checkpoint_steps = int(os.environ.get("CHECKPOINT_STEPS", "250"))
logging_steps = int(os.environ.get("LOGGING_STEPS", "10"))
keep_checkpoints = int(os.environ.get("KEEP_CHECKPOINTS", "8"))
resume_setting = os.environ.get("RESUME_FROM_CHECKPOINT", "auto")
min_train_rows = int(os.environ.get("MIN_TRAIN_ROWS", "250000"))

os.makedirs(output_dir, exist_ok=True)

def latest_checkpoint(path: str):
    root = Path(path)
    checkpoints = []
    for child in root.glob("checkpoint-*"):
        if child.is_dir():
            try:
                step = int(child.name.split("-")[-1])
            except ValueError:
                continue
            checkpoints.append((step, child))
    if not checkpoints:
        return None
    return str(sorted(checkpoints)[-1][1])

resume_checkpoint = None
if resume_setting.lower() == "auto":
    resume_checkpoint = latest_checkpoint(output_dir)
elif resume_setting.lower() not in {"", "false", "none", "0"}:
    resume_checkpoint = resume_setting

if resume_checkpoint:
    print(f"[INFO] Resuming from checkpoint: {resume_checkpoint}", flush=True)
else:
    print("[INFO] Starting without checkpoint resume", flush=True)

dataset = load_dataset("json", data_files=train_file, split="train")
original_rows = len(dataset)
if original_rows <= 0:
    raise RuntimeError(f"Training file has no rows: {train_file}")
if original_rows < min_train_rows:
    print(f"[INFO] Expanding training dataset from {original_rows} to {min_train_rows} rows by deterministic repetition", flush=True)
    dataset = dataset.select([i % original_rows for i in range(min_train_rows)])
else:
    print(f"[INFO] Training dataset rows: {original_rows}; no repetition needed", flush=True)
effective_rows = len(dataset)
print(f"[INFO] Effective training rows: {effective_rows}", flush=True)

def to_text(ex):
    messages = ex.get("messages", [])
    text = "\n".join([f"{m['role']}: {m['content']}" for m in messages if 'role' in m and 'content' in m])
    return {"text": text}

dataset = dataset.map(to_text)

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
    num_train_epochs=100000,
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

trainer.train(resume_from_checkpoint=resume_checkpoint)
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

with open(os.path.join(output_dir, "training-summary.json"), "w") as f:
    json.dump({
        "base_model": base_model,
        "train_file": train_file,
        "output_dir": output_dir,
        "max_steps": max_steps,
        "original_dataset_rows": original_rows,
        "effective_dataset_rows": effective_rows,
        "min_train_rows": min_train_rows,
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

export BASE_MODEL TRAIN_FILE OUTPUT_DIR MAX_STEPS LR BATCH_SIZE GRAD_ACCUM LORA_R LORA_ALPHA CHECKPOINT_STEPS LOGGING_STEPS KEEP_CHECKPOINTS RESUME_FROM_CHECKPOINT MIN_TRAIN_ROWS

# Use SIGINT first so Transformers has a chance to unwind cleanly; periodic checkpoints are the durable resume boundary.
set +e
timeout --signal=INT --kill-after=120s "${TRAIN_HOURS}h" python3 "$OUTPUT_DIR/run_lora_sft.py" 2>&1 | tee -a "artifacts/logs/lora-train.log"
status=${PIPESTATUS[0]}
set -e

cat > "$OUTPUT_DIR/training-run-status.json" <<EOF_STATUS
{
  "exit_code": $status,
  "train_hours": "$TRAIN_HOURS",
  "max_steps": "$MAX_STEPS",
  "checkpoint_steps": "$CHECKPOINT_STEPS",
  "min_train_rows": "$MIN_TRAIN_ROWS",
  "resume_from_checkpoint": "$RESUME_FROM_CHECKPOINT",
  "status_note": "0 means max_steps/completion; 124 means timeout reached; 130 means interrupted after a saved checkpoint can be resumed."
}
EOF_STATUS

echo "[INFO] Training process exited with code $status."
echo "[INFO] Adapter artifacts: $OUTPUT_DIR"
echo "[INFO] Checkpoints: $OUTPUT_DIR/checkpoint-*"
echo "[INFO] Log file: artifacts/logs/lora-train.log"
echo "[INFO] Run status: $OUTPUT_DIR/training-run-status.json"
exit "$status"
