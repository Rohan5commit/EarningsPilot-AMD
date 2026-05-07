#!/usr/bin/env bash
set -euo pipefail

# EarningsPilot AMD - time-boxed LoRA/QLoRA runner for AMD MI300X hosts
# Usage:
#   TRAIN_HOURS=10 MAX_STEPS=20000 BASE_MODEL=Qwen/Qwen2.5-7B-Instruct ./scripts/amd/start-lora-training.sh

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-training-data/earningspilot-sft-10h.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/lora/earningspilot-qwen-7b-lora}"
TRAIN_HOURS="${TRAIN_HOURS:-10}"
MAX_STEPS="${MAX_STEPS:-20000}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-4}"

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

base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
train_file = os.environ.get("TRAIN_FILE", "training-data/earningspilot-sft-10h.jsonl")
output_dir = os.environ.get("OUTPUT_DIR", "artifacts/lora/earningspilot-qwen-7b-lora")
max_steps = int(os.environ.get("MAX_STEPS", "20000"))
learning_rate = float(os.environ.get("LR", "2e-4"))
batch_size = int(os.environ.get("BATCH_SIZE", "1"))
grad_accum = int(os.environ.get("GRAD_ACCUM", "8"))
lora_r = int(os.environ.get("LORA_R", "16"))
lora_alpha = int(os.environ.get("LORA_ALPHA", "32"))

os.makedirs(output_dir, exist_ok=True)

dataset = load_dataset("json", data_files=train_file, split="train")

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
    logging_steps=20,
    save_steps=100,
    save_total_limit=2,
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

trainer.train()
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

with open(os.path.join(output_dir, "training-summary.json"), "w") as f:
    json.dump({
        "base_model": base_model,
        "train_file": train_file,
        "output_dir": output_dir,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
    }, f, indent=2)
PY

export BASE_MODEL TRAIN_FILE OUTPUT_DIR MAX_STEPS LR BATCH_SIZE GRAD_ACCUM LORA_R LORA_ALPHA

timeout "${TRAIN_HOURS}h" python3 "$OUTPUT_DIR/run_lora_sft.py" 2>&1 | tee "artifacts/logs/lora-train.log"

echo "[INFO] Training time-box completed (or process exited)."
echo "[INFO] Adapter artifacts: $OUTPUT_DIR"
echo "[INFO] Log file: artifacts/logs/lora-train.log"
