#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-training-data/earningspilot-sft.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/lora/earningspilot-qwen-7b-lora}"
TRAIN_HOURS="${TRAIN_HOURS:-15}"
MAX_STEPS="${MAX_STEPS:-800}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-4}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if command -v python3.11 >/dev/null 2>&1 && [[ "${PYTHON_BIN}" == "python3" ]]; then
  PYTHON_BIN="python3.11"
fi

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "[ERROR] Training file not found: $TRAIN_FILE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" artifacts/logs

echo "[INFO] Starting time-boxed LoRA training"
echo "[INFO] PYTHON_BIN=$PYTHON_BIN"

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install "transformers>=4.45,<5" "peft>=0.12,<0.13" "accelerate>=1.1,<2"

cat > "$OUTPUT_DIR/run_lora_sft.py" <<'PY'
import json
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
train_file = os.environ.get("TRAIN_FILE", "training-data/earningspilot-sft.jsonl")
output_dir = os.environ.get("OUTPUT_DIR", "artifacts/lora/earningspilot-qwen-7b-lora")
max_steps = int(os.environ.get("MAX_STEPS", "800"))
learning_rate = float(os.environ.get("LR", "2e-4"))
batch_size = int(os.environ.get("BATCH_SIZE", "1"))
grad_accum = int(os.environ.get("GRAD_ACCUM", "8"))
lora_r = int(os.environ.get("LORA_R", "16"))
lora_alpha = int(os.environ.get("LORA_ALPHA", "32"))

os.makedirs(output_dir, exist_ok=True)

def read_text_rows(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            messages = ex.get("messages", [])
            text = "\n".join([f"{m['role']}: {m['content']}" for m in messages if 'role' in m and 'content' in m])
            if text:
                rows.append(text)
    if not rows:
        raise ValueError("No training rows found in training file")
    return rows

texts = read_text_rows(train_file)
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def _dtype():
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    trust_remote_code=True,
    torch_dtype=_dtype(),
    low_cpu_mem_usage=False,
)

if torch.cuda.is_available():
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
model = get_peft_model(model, peft_config)
model.config.use_cache = False

class CausalTextDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=1024):
        self.samples = []
        for text in samples:
            enc = tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
            input_ids = enc["input_ids"][0]
            attention_mask = enc["attention_mask"][0]
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            self.samples.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

dataset = CausalTextDataset(texts, tokenizer)

use_cuda = torch.cuda.is_available()
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
    bf16=use_cuda,
    use_cpu=not use_cuda,
    dataloader_pin_memory=False,
    gradient_checkpointing=False,
    report_to="none",
)

trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()
model.save_pretrained(output_dir)
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

if timeout "${TRAIN_HOURS}h" "$PYTHON_BIN" "$OUTPUT_DIR/run_lora_sft.py" 2>&1 | tee "artifacts/logs/lora-train.log"; then
  train_status=0
else
  train_status=$?
fi

if [ "$train_status" -ne 0 ] && [ "$train_status" -ne 124 ]; then
  exit "$train_status"
fi

echo "[INFO] Training time-box completed (or process exited)."
