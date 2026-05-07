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
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-1000}"
LOGGING_STEPS="${LOGGING_STEPS:-25}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-auto}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_LENGTH="${MAX_LENGTH:-512}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
PIN_MEMORY="${PIN_MEMORY:-true}"
ATTENTION_IMPL="${ATTENTION_IMPL:-sdpa}"
COMPILE_MODEL="${COMPILE_MODEL:-false}"
TOKEN_CACHE_LIMIT="${TOKEN_CACHE_LIMIT:-50000}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-false}"

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
[INFO] BATCH_SIZE=$BATCH_SIZE
[INFO] GRAD_ACCUM=$GRAD_ACCUM
[INFO] MAX_LENGTH=$MAX_LENGTH
[INFO] DATALOADER_NUM_WORKERS=$DATALOADER_NUM_WORKERS
[INFO] ATTENTION_IMPL=$ATTENTION_IMPL
[INFO] COMPILE_MODEL=$COMPILE_MODEL
[INFO] LOAD_IN_4BIT=$LOAD_IN_4BIT
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
"$PYTHON_BIN" -m pip install "transformers>=4.45,<5" "peft>=0.12,<0.13" "accelerate>=1.1,<2"
if [[ "$LOAD_IN_4BIT" == "true" ]]; then
  "$PYTHON_BIN" -m pip install "bitsandbytes>=0.43"
fi

cat > "$OUTPUT_DIR/run_forced_lora_sft.py" <<'PY'
import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

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
max_length = int(os.environ.get("MAX_LENGTH", "512"))
dataloader_num_workers = int(os.environ.get("DATALOADER_NUM_WORKERS", "2"))
pin_memory = os.environ.get("PIN_MEMORY", "true").lower() == "true"
attention_impl = os.environ.get("ATTENTION_IMPL", "sdpa")
compile_model = os.environ.get("COMPILE_MODEL", "false").lower() == "true"
token_cache_limit = int(os.environ.get("TOKEN_CACHE_LIMIT", "50000"))
load_in_4bit = os.environ.get("LOAD_IN_4BIT", "false").lower() == "true"

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

row_count = sum(1 for _ in open(train_file, "r"))
print(f"[INFO] Loading materialized train_file={train_file}", flush=True)
print(f"[INFO] Loaded dataset rows={row_count}", flush=True)
if row_count < 100000:
    raise RuntimeError(f"Refusing short run: dataset has only {row_count} rows")

def row_to_text(row):
    messages = row.get("messages", [])
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages if "role" in m and "content" in m)

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"[INFO] Pre-tokenizing up to {token_cache_limit} unique rows at max_length={max_length}", flush=True)
seen = set()
features = []
with open(train_file, "r") as f:
    for line in f:
        if not line.strip():
            continue
        text = row_to_text(json.loads(line))
        if text in seen:
            continue
        seen.add(text)
        enc = tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        features.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})
        if len(features) >= token_cache_limit:
            break
if not features:
    raise RuntimeError("No tokenized features were built")
print(f"[INFO] Tokenized feature cache size={len(features)}", flush=True)

class RepeatingTokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, features, length):
        self.features = features
        self.length = max(int(length), len(features))
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return self.features[idx % len(self.features)]

model_kwargs = {
    "trust_remote_code": True,
    "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    "low_cpu_mem_usage": False,
}
if attention_impl and attention_impl.lower() not in {"default", "none", "false"}:
    model_kwargs["attn_implementation"] = attention_impl
if load_in_4bit:
    try:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print("[INFO] LOAD_IN_4BIT=true; using BitsAndBytesConfig QLoRA path", flush=True)
    except Exception as exc:
        print(f"[WARN] LOAD_IN_4BIT requested but unavailable: {exc}; falling back to bf16 LoRA", flush=True)

model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
if torch.cuda.is_available() and not load_in_4bit:
    model = model.to("cuda")

preferred_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj", "c_attn", "c_proj"]
module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
resolved_targets = [name for name in preferred_targets if name in module_names]
if not resolved_targets:
    raise ValueError(f"Could not resolve LoRA target modules from model. Available sample: {sorted(list(module_names))[:30]}")

peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=resolved_targets,
)
if load_in_4bit:
    model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.config.use_cache = False
if compile_model and hasattr(torch, "compile"):
    print("[INFO] Compiling model with torch.compile", flush=True)
    model = torch.compile(model)

use_cuda = torch.cuda.is_available()
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    learning_rate=learning_rate,
    max_steps=max_steps,
    logging_steps=logging_steps,
    save_strategy="steps",
    save_steps=checkpoint_steps,
    save_total_limit=keep_checkpoints,
    fp16=False,
    bf16=use_cuda,
    use_cpu=not use_cuda,
    dataloader_num_workers=dataloader_num_workers,
    dataloader_pin_memory=pin_memory,
    gradient_checkpointing=False,
    report_to="none",
)

train_dataset = RepeatingTokenizedDataset(features, row_count)
trainer = Trainer(model=model, train_dataset=train_dataset, args=args)

print(f"[INFO] Starting Trainer.train(max_steps={max_steps}, resume={resume_checkpoint})", flush=True)
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
        "target_modules": resolved_targets,
        "checkpoint_steps": checkpoint_steps,
        "max_length": max_length,
        "dataloader_num_workers": dataloader_num_workers,
        "pin_memory": pin_memory,
        "attention_impl": attention_impl,
        "compile_model": compile_model,
        "token_cache_size": len(features),
        "load_in_4bit": load_in_4bit,
        "logging_steps": logging_steps,
        "keep_checkpoints": keep_checkpoints,
        "resume_from_checkpoint": resume_checkpoint,
        "trainer": "transformers.Trainer",
    }, f, indent=2)

PY

export BASE_MODEL EXPANDED_TRAIN_FILE OUTPUT_DIR MAX_STEPS LR BATCH_SIZE GRAD_ACCUM LORA_R LORA_ALPHA CHECKPOINT_STEPS LOGGING_STEPS KEEP_CHECKPOINTS RESUME_FROM_CHECKPOINT MAX_LENGTH DATALOADER_NUM_WORKERS PIN_MEMORY ATTENTION_IMPL COMPILE_MODEL TOKEN_CACHE_LIMIT LOAD_IN_4BIT

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
