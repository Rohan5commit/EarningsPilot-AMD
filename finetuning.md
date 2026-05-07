# Fine-Tuning and Evaluation Plan

## Short answer

EarningsPilot AMD does **not** need a fully trained custom model to be hackathon-submission ready. The core product value is the agent workflow, grounded evidence, and AMD-compatible open-source inference path. However, the project now includes a reproducible fine-tuning and evaluation path so the submission can credibly explain how the model improves on AMD Developer Cloud after the demo.

## Why not train before the demo?

Full fine-tuning is risky for a public hackathon demo because:

- A small synthetic dataset can overfit and make the model worse outside the sample scenario.
- Training and deployment can consume time that is better spent making the end-to-end product reliable.
- Judges need to see the workflow working instantly; they should not wait on a GPU job.
- Source grounding, schema validation, and agent decomposition often improve reliability more than premature fine-tuning.

The recommended strategy is:

1. Ship deterministic + model-backed orchestration.
2. Collect failure cases from real transcripts and filings.
3. Evaluate against golden tasks.
4. Fine-tune a small adapter only after enough labeled examples exist.
5. Serve the tuned adapter on AMD Developer Cloud.

## What is included now

- `training-data/earningspilot-sft.jsonl`: seed supervised fine-tuning examples in chat format.
- `eval/golden-sample.json`: golden expectations for the seeded demo package.
- `scripts/evaluate-sample.mjs`: an end-to-end regression/evaluation harness that checks KPI, risk, evidence, company, and ticker outputs.

## Recommended AMD fine-tuning stack

Use AMD Developer Cloud with a ROCm-enabled GPU instance and one of these open-source model families:

- `Qwen/Qwen2.5-7B-Instruct`
- Llama-family instruct models where licensing fits the deployment
- Mistral-family instruct models
- DeepSeek-family instruct models where hardware and memory allow

Recommended training approach:

- **Method:** LoRA or QLoRA adapter fine-tuning.
- **Libraries:** TRL + PEFT + Transformers on ROCm, or Axolotl on ROCm.
- **Objective:** Improve JSON adherence, evidence grounding, KPI normalization, and risk-title consistency.
- **Base model first choice:** Qwen2.5 7B Instruct because it is strong at structured extraction and multilingual/document tasks.


## Expanded SFT dataset for the remaining GPU window

The repo includes two dataset tiers:

- `training-data/earningspilot-sft.jsonl`: 5 hand-written seed examples used for smoke tests.
- `training-data/earningspilot-sft-expanded.jsonl`: 5,000 deterministic synthetic finance-agent conversations for smoke or shorter adapter runs.
- `training-data/earningspilot-sft-10h.jsonl`: 50,000 deterministic synthetic finance-agent conversations (~52 MB) for the urgent 10-hour MI300X training window.

Regenerate or resize the expanded dataset with:

```bash
npm run generate:sft:10h
```

Restart the AMD host training run with the expanded dataset and a hard timeout:

```bash
TRAIN_HOURS=9 \
MAX_STEPS=100000000 \
BATCH_SIZE=4 \
GRAD_ACCUM=4 \
MAX_LENGTH=512 \
DATALOADER_NUM_WORKERS=2 \
ATTENTION_IMPL=sdpa \
CHECKPOINT_STEPS=1000 \
KEEP_CHECKPOINTS=8 \
MIN_TRAIN_ROWS=250000 \
RESUME_FROM_CHECKPOINT=auto \
TRAIN_FILE=training-data/earningspilot-sft-10h.jsonl \
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
./scripts/amd/start-lora-training.sh
```

The launcher saves every `CHECKPOINT_STEPS` optimizer steps and resumes from the newest `checkpoint-*` directory when `RESUME_FROM_CHECKPOINT=auto`. Use `./scripts/amd/training-progress.sh` to print the latest checkpoint step, percent progress against `MAX_STEPS`, the last Trainer log entry, and the last 30 log lines.



Emergency forced wall-clock run if the host keeps executing a short smoke path:

```bash
TRAIN_HOURS=9 \
MAX_STEPS=100000000 \
MIN_TRAIN_ROWS=1000000 \
BATCH_SIZE=4 \
GRAD_ACCUM=4 \
MAX_LENGTH=512 \
DATALOADER_NUM_WORKERS=2 \
ATTENTION_IMPL=sdpa \
CHECKPOINT_STEPS=1000 \
KEEP_CHECKPOINTS=8 \
RESUME_FROM_CHECKPOINT=auto \
TRAIN_FILE=training-data/earningspilot-sft-10h.jsonl \
OUTPUT_DIR=artifacts/lora/earningspilot-qwen-7b-lora-10h-forced \
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
./scripts/amd/start-forced-10h-training.sh
```



For the 9-hour rescue run, the fastest safe optimization is not more synthetic rows; it is eliminating per-step tokenization. The current AMD launchers pre-tokenize a bounded cache, repeat tokenized samples through a map-style dataset, shorten sequences to 512 tokens by default, use larger micro-batches, and checkpoint every 1,000 steps to avoid save overhead. Set `LOAD_IN_4BIT=true` only if the AMD host has a working ROCm-compatible bitsandbytes install; otherwise the default bf16 LoRA path is safer on MI300X.



## vLLM serving note for ROCm containers

For `vllm/vllm-openai-rocm` images, launch from inside the running container shell and use the modern vLLM CLI shape: `vllm serve <model> ...`. The repo helpers default to that CLI when available and only fall back to `python -m vllm.entrypoints.openai.api_server` for older local installs. If Docker logs show `unrecognized arguments` for `python -m` or `vllm.entrypoints`, enter the container first with `docker exec -it rocm /bin/bash`, then run `./scripts/amd/serve-trained-adapter-vllm-rocm.sh` from the repo root.


If the existing one-click container still rejects arguments, use `./scripts/amd/run-vllm-rocm-container.sh` from the host. It starts a fresh `vllm/vllm-openai-rocm:latest` container with `--entrypoint /bin/bash`, mounts the repo, writes an inner command script, and launches `vllm serve` from an argv array so the Docker entrypoint cannot collapse arguments.


If vLLM remains blocked by container entrypoint issues, use `./scripts/amd/serve-transformers-openai-rocm.sh`. It loads the base model plus optional PEFT adapter with Transformers/FastAPI and exposes the same OpenAI-compatible `/v1/models` and `/v1/chat/completions` endpoints expected by `npm run benchmark:amd`. The fallback is intentionally capped for demo reliability (`MAX_MODEL_LEN=1024`, `MAX_NEW_TOKENS_DEFAULT=96` by default), uses `device_map=auto` on GPU hosts, merges the LoRA adapter before serving when possible, and exposes `/health`; wait for `curl http://<gpu-host>:8000/health` to return `ok` before launching the benchmark. If the fallback still approaches the benchmark timeout, lower `BENCHMARK_MAX_TOKENS` (for example `BENCHMARK_MAX_TOKENS=48`) or raise `BENCHMARK_TIMEOUT_MS` for the first run.

## Post-training evaluation handoff

After the adapter reaches the target checkpoint, stop training and run:

```bash
EARNINGSPILOT_BASE_URL=https://earningspilot-amd.vercel.app \
AMD_OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
AMD_OPENAI_API_KEY=epamd-temp-key \
AMD_MODEL_ID=Qwen/Qwen2.5-7B-Instruct \
OUTPUT_DIR=artifacts/lora/earningspilot-qwen-7b-lora-10h-forced \
LOG_FILE=artifacts/logs/lora-train-forced-10h.log \
./scripts/amd/post-training-eval.sh
```

The script writes checkpoint inventory, latest Trainer state, `npm run eval:sample` output, AMD endpoint benchmark output, GPU utilization samples, and a tarball containing the adapter/logs into `artifacts/eval/<timestamp>/`.

## Current PEFT / Trainer recipe

> Treat this as the production recipe for AMD Developer Cloud, not something required for the deterministic public demo.

The active AMD launchers now avoid `trl.SFTTrainer` API drift by using `transformers.Trainer` directly. They apply LoRA with PEFT first, pre-tokenize a bounded cache, and then train against a repeating map-style dataset so throughput is controlled by `MAX_STEPS`, `BATCH_SIZE`, `MAX_LENGTH`, and the wall-clock timeout.

Key defaults for the 9-hour rescue run:

```bash
BATCH_SIZE=4
GRAD_ACCUM=4
MAX_LENGTH=512
DATALOADER_NUM_WORKERS=2
ATTENTION_IMPL=sdpa
CHECKPOINT_STEPS=1000
LOAD_IN_4BIT=false  # set true only if ROCm bitsandbytes works on the host
```

QLoRA is available through `LOAD_IN_4BIT=true`; when enabled, the scripts install bitsandbytes, pass a `BitsAndBytesConfig`, and call `prepare_model_for_kbit_training` before applying LoRA. On MI300X, the safer default remains bf16 LoRA unless the host confirms ROCm-compatible bitsandbytes support.

## Evaluation gate before deployment

Run the regression harness against the local or deployed app:

```bash
npm run build
PORT=3000 npm run start
npm run eval:sample
```

Or test a deployed URL:

```bash
EARNINGSPILOT_BASE_URL=https://earningspilot-amd.vercel.app npm run eval:sample
```

Minimum acceptance criteria:

- Company and ticker detected correctly.
- At least six KPI rows extracted.
- At least four risk rows extracted.
- Required risk categories are present.
- Required KPI categories are present.
- Parser, KPI, and Risk agents are represented in evidence.

## Deployment after fine-tuning

1. Train and save the LoRA adapter on AMD Developer Cloud.
2. Merge the adapter or serve base + adapter with vLLM/TGI where supported.
3. Expose an OpenAI-compatible `/chat/completions` endpoint.
4. Set `AMD_OPENAI_BASE_URL`, `AMD_OPENAI_API_KEY`, and `AMD_MODEL_ID` in Vercel or Hugging Face Space secrets.
5. Re-run `npm run eval:sample` against the public URL.

## What to say in the video

"We intentionally did not make the public demo depend on a newly trained model because reliability matters for judges. The architecture is training-ready: we include seed SFT data, a golden evaluation harness, and an AMD ROCm LoRA plan. In production, we would collect analyst corrections, fine-tune a Qwen/Llama/Mistral adapter on AMD Developer Cloud, and serve it through the same OpenAI-compatible endpoint already supported by the app."
