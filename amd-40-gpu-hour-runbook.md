# 40 AMD Instinct MI300X GPU-Hour Execution Runbook

## Objective

Use the full 40 MI300X GPU-hour budget to make EarningsPilot AMD feel like a real AMD cloud product, not a CPU-only demo. The goal is to produce three winning artifacts:

1. A live AMD-hosted open-source model endpoint connected to the app.
2. A credible EarningsPilot-Qwen-7B-LoRA adapter training story.
3. Benchmark/evaluation evidence that can be shown in the README, video, and slides.

## Access preference

Use SSH into an already-provisioned AMD GPU host. A DigitalOcean token is only useful if it provisions the GPU or manages DNS/deployment. For the highest chance of winning, prioritize direct SSH to the ROCm/MI300X machine.

## 40 GPU-hour allocation

| Phase | GPU-hours | Output |
| --- | ---: | --- |
| ROCm/vLLM bring-up | 4 | Running OpenAI-compatible Qwen endpoint |
| App integration + eval | 4 | `amd-openai-compatible` mode verified in EarningsPilot |
| SFT dataset expansion checks | 2 | Valid JSONL examples and prompt coverage |
| LoRA/QLoRA adapter training (time-boxed) | 15 | EarningsPilot-Qwen-7B-LoRA best adapter within strict budget |
| Serving base + adapter | 5 | Endpoint serving the tuned adapter or best base model |
| Benchmark and golden evaluation | 5 | Latency, output throughput, quality gate results |
| Demo capture buffer | 3 | Screenshots/video proof, fallback testing |
| Final contingency | 2 | Deadline/debug buffer |
| **Total** | **40** | Winning AMD proof package |

## Step 1 — Verify GPU host

Run on the GPU host:

```bash
hostname
uname -a
rocm-smi || true
rocminfo | head -80 || true
python --version
```

Record GPU count, memory, ROCm version, and driver status in `benchmark-notes.md`.

## Step 2 — Install model serving stack

Preferred stack:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install vllm transformers accelerate huggingface_hub
```

If the host image already includes ROCm-ready vLLM, use that environment instead.

## Step 3 — Serve Qwen 7B on AMD

From the repo root on the GPU host:

```bash
export AMD_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
export AMD_OPENAI_API_KEY=replace-with-temporary-key
./scripts/amd/serve-qwen-vllm-rocm.sh
```

Expose the endpoint as:

```bash
AMD_OPENAI_BASE_URL=http://<gpu-host>:8000/v1
```

## Step 4 — Benchmark endpoint

From any machine that can reach the endpoint:

```bash
AMD_OPENAI_BASE_URL=http://<gpu-host>:8000/v1 \
AMD_OPENAI_API_KEY=replace-with-temporary-key \
AMD_MODEL_ID=Qwen/Qwen2.5-7B-Instruct \
npm run benchmark:amd
```

Paste the JSON output into `benchmark-notes.md` and the slide deck.

## Step 5 — Connect EarningsPilot

Set these secrets in Vercel/Hugging Face Space or local `.env.local`:

```bash
AMD_OPENAI_BASE_URL=http://<gpu-host>:8000/v1
AMD_OPENAI_API_KEY=replace-with-temporary-key
AMD_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
AMD_GPU_NAME=AMD Instinct MI300X
AMD_GPU_HOURS_BUDGET=40
```

Run:

```bash
npm run build
PORT=3000 npm run start
npm run eval:sample
```

The app should report `amd-openai-compatible` mode when the endpoint is reachable.

## Step 6 — Train EarningsPilot-Qwen-7B-LoRA

Use `finetuning.md` as the recipe. The owned model artifact is:

```text
EarningsPilot-Qwen-7B-LoRA
```

Training success criteria:

- Adapter is saved.
- At least one before/after eval is recorded.
- Adapter does not reduce JSON validity or evidence grounding.
- If adapter is weaker than the base model, use base model for the live demo and present adapter as the reproducible training path.

## Step 7 — Final judging assets

By the end of the 40 GPU-hour budget, capture:

- Screenshot of GPU model endpoint logs.
- Screenshot or terminal output from `npm run benchmark:amd`.
- Screenshot or terminal output from `npm run eval:sample` in AMD model mode.
- One slide showing the 40 GPU-hour allocation.
- One sentence in the video: "EarningsPilot AMD uses AMD Instinct MI300X to serve open-source Qwen inference and train the EarningsPilot-Qwen-7B-LoRA finance adapter."

## Fallback rule

If the AMD endpoint becomes unstable, keep the public app in deterministic mode and use the benchmark/training artifacts as proof of AMD usage. Do not let a cold GPU endpoint break the judge demo.
