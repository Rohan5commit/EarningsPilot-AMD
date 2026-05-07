#!/usr/bin/env bash
set -euo pipefail

# EarningsPilot AMD - vLLM-free OpenAI-compatible fallback server.
# Use this when vLLM ROCm containers/entrypoints are unstable. It serves the
# base model plus an optional PEFT LoRA adapter with FastAPI + Transformers.
# It is slower than vLLM, but reliable enough for hackathon benchmark/demo proof.

BASE_MODEL="${BASE_MODEL:-${AMD_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}}"
ADAPTER_PATH="${ADAPTER_PATH:-artifacts/lora/earningspilot-qwen-7b-lora-10h-forced}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-EarningsPilot-Qwen-7B-LoRA}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_KEY="${AMD_OPENAI_API_KEY:-local-dev-key}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NEW_TOKENS_DEFAULT="${MAX_NEW_TOKENS_DEFAULT:-96}"
MODEL_LOAD_MODE="${MODEL_LOAD_MODE:-auto}"
ATTENTION_IMPL="${ATTENTION_IMPL:-sdpa}"
MERGE_LORA="${MERGE_LORA:-true}"
ENABLE_LORA="${ENABLE_LORA:-auto}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SERVER_PY="${SERVER_PY:-artifacts/runtime/transformers_openai_server.py}"

if [[ "$ENABLE_LORA" == "auto" ]]; then
  if [[ -f "$ADAPTER_PATH/adapter_config.json" ]]; then
    ENABLE_LORA="true"
  else
    ENABLE_LORA="false"
  fi
fi

mkdir -p "$(dirname "$SERVER_PY")"

cat <<BANNER
Starting EarningsPilot Transformers OpenAI-compatible fallback server
Base model: ${BASE_MODEL}
Served model name: ${SERVED_MODEL_NAME}
LoRA enabled: ${ENABLE_LORA}
Adapter path: ${ADAPTER_PATH}
Host: ${HOST}
Port: ${PORT}
Max prompt tokens: ${MAX_MODEL_LEN}
Default max new tokens: ${MAX_NEW_TOKENS_DEFAULT}
Model load mode: ${MODEL_LOAD_MODE}
Attention implementation: ${ATTENTION_IMPL}
Merge LoRA before serving: ${MERGE_LORA}
Endpoint: http://${HOST}:${PORT}/v1/chat/completions
BANNER

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install "fastapi>=0.110,<1" "uvicorn[standard]>=0.27,<1" "transformers>=4.45,<5" "peft>=0.12,<0.13" "accelerate>=1.1,<2" "safetensors>=0.4"

cat > "$SERVER_PY" <<'PY'
import os
import time
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:  # pragma: no cover - optional unless adapter exists
    PeftModel = None

BASE_MODEL = os.environ["BASE_MODEL"]
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "")
SERVED_MODEL_NAME = os.environ["SERVED_MODEL_NAME"]
API_KEY = os.environ.get("API_KEY", "")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "1024"))
MAX_NEW_TOKENS_DEFAULT = int(os.environ.get("MAX_NEW_TOKENS_DEFAULT", "96"))
MODEL_LOAD_MODE = os.environ.get("MODEL_LOAD_MODE", "auto")
ATTENTION_IMPL = os.environ.get("ATTENTION_IMPL", "sdpa")
MERGE_LORA = os.environ.get("MERGE_LORA", "true").lower() == "true"
ENABLE_LORA = os.environ.get("ENABLE_LORA", "false").lower() == "true"

app = FastAPI(title="EarningsPilot AMD Transformers OpenAI-compatible server")

def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def _dtype():
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32

def _load_kwargs():
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": _dtype(),
        "low_cpu_mem_usage": True,
    }
    if ATTENTION_IMPL:
        kwargs["attn_implementation"] = ATTENTION_IMPL
    if torch.cuda.is_available() and MODEL_LOAD_MODE == "auto":
        kwargs["device_map"] = "auto"
    return kwargs

print({
    "event": "loading_model",
    "base_model": BASE_MODEL,
    "served_model_name": SERVED_MODEL_NAME,
    "enable_lora": ENABLE_LORA,
    "adapter_path": ADAPTER_PATH,
    "device": _device(),
    "dtype": str(_dtype()),
    "model_load_mode": MODEL_LOAD_MODE,
    "attention_impl": ATTENTION_IMPL,
    "max_prompt_tokens": MAX_MODEL_LEN,
    "default_max_new_tokens": MAX_NEW_TOKENS_DEFAULT,
}, flush=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **_load_kwargs())
if ENABLE_LORA:
    if not ADAPTER_PATH or not os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
        raise RuntimeError(f"ENABLE_LORA=true but adapter_config.json not found at {ADAPTER_PATH}")
    if PeftModel is None:
        raise RuntimeError("peft is not available but ENABLE_LORA=true")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    if MERGE_LORA and hasattr(model, "merge_and_unload"):
        print({"event": "merging_lora_adapter"}, flush=True)
        model = model.merge_and_unload()

if MODEL_LOAD_MODE != "auto" or not torch.cuda.is_available():
    model.to(_device())
model.eval()
print({"event": "model_ready", "device": _device()}, flush=True)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = Field(default=None)
    response_format: Optional[Dict[str, Any]] = None


def require_auth(authorization: Optional[str]):
    if not API_KEY:
        return
    expected = f"Bearer {API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def render_prompt(messages: List[ChatMessage]) -> str:
    raw = [{"role": m.role, "content": m.content} for m in messages]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(raw, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return "\n".join(f"{m.role}: {m.content}" for m in messages) + "\nassistant:"

@app.get("/v1/models")
def models(authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    return {
        "object": "list",
        "data": [{"id": SERVED_MODEL_NAME, "object": "model", "owned_by": "earningspilot-amd"}],
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": SERVED_MODEL_NAME}

@app.post("/v1/chat/completions")
def chat(req: ChatRequest, authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    started = time.time()
    prompt = render_prompt(req.messages)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_MODEL_LEN,
    ).to(_device())
    max_new_tokens = min(int(req.max_tokens or MAX_NEW_TOKENS_DEFAULT), MAX_NEW_TOKENS_DEFAULT)
    temperature = float(req.temperature or 0.0)
    do_sample = temperature > 0
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
    with torch.no_grad():
        output = model.generate(**encoded, **generate_kwargs)
    generated = output[0][encoded["input_ids"].shape[-1]:]
    content = tokenizer.decode(generated, skip_special_tokens=True).strip()
    prompt_tokens = int(encoded["input_ids"].numel())
    completion_tokens = int(generated.numel())
    return {
        "id": f"chatcmpl-epamd-{int(started * 1000)}",
        "object": "chat.completion",
        "created": int(started),
        "model": SERVED_MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8000")))
PY

export BASE_MODEL ADAPTER_PATH SERVED_MODEL_NAME API_KEY MAX_MODEL_LEN MAX_NEW_TOKENS_DEFAULT MODEL_LOAD_MODE ATTENTION_IMPL MERGE_LORA ENABLE_LORA HOST PORT
exec "$PYTHON_BIN" "$SERVER_PY"
