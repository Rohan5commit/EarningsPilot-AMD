# Benchmark and AMD Workload Notes

## Benchmark objective

Show why AMD Developer Cloud matters for EarningsPilot AMD: the expensive portion of the workload is not the UI; it is repeated structured inference over long financial documents and multiple specialized agent prompts.

## Workload profile

A typical analyst session includes:

- 1 earnings transcript: 8,000-18,000 tokens
- 1 SEC filing excerpt or 10-K section: 5,000-25,000 tokens
- 1 KPI CSV or investor-presentation extract: 500-3,000 tokens
- 5 agent stages: parser, KPI, risk, thesis, report
- 8-20 evidence-grounded extraction and synthesis calls in production mode


## 40 MI300X GPU-hour benchmark plan

The project now assumes a 40 AMD Instinct MI300X GPU-hour ceiling. Spend it on visible winning proof, not exploratory dead ends:

- 4 hours: ROCm/vLLM bring-up and Qwen endpoint smoke test.
- 4 hours: app integration and `amd-openai-compatible` mode validation.
- 16 hours: two LoRA/QLoRA adapter attempts for EarningsPilot-Qwen-7B-LoRA.
- 4 hours: serve the best base/adapted model endpoint.
- 4 hours: benchmark endpoint latency and output throughput with `npm run benchmark:amd`.
- 4 hours: run golden eval and capture screenshots/video proof.
- 4 hours: contingency and final demo capture.

Use `amd-40-gpu-hour-runbook.md` as the operating checklist.

## Suggested AMD benchmark

Run an OpenAI-compatible model server on AMD Developer Cloud with ROCm and an open-source model such as `Qwen/Qwen2.5-7B-Instruct`.

Measure:

```bash
curl -s "$AMD_OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $AMD_OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"Qwen/Qwen2.5-7B-Instruct",
    "temperature":0.1,
    "messages":[
      {"role":"system","content":"Return compact JSON only."},
      {"role":"user","content":"Extract KPIs and risks from this earnings excerpt: revenue grew 18%, gross margin was 43.2%, capacity is constrained, export controls may reduce sales."}
    ]
  }'
```

Track:

- Time to first token
- Total latency
- Output tokens per second
- GPU utilization
- Cost per analyzed filing packet

## Observed MI300X benchmark run (May 7, 2026)

From a live AMD endpoint (`http://127.0.0.1:8000/v1`) using `Qwen/Qwen2.5-7B-Instruct` on AMD Instinct MI300X:

- `status`: pass
- `runs`: 3
- `avgLatencyMs`: 391
- `avgOutputCharsPerSecond`: 789
- per-run latency: 408 ms, 385 ms, 380 ms

Related end-to-end app validation from the same run:

- `modelMode`: `amd-openai-compatible`
- Golden eval: pass
- Observed extraction: 8 KPIs, 5 risks, 32 evidence items

These metrics are suitable as submission evidence that EarningsPilot AMD was run against a real AMD GPU-backed endpoint, not only deterministic fallback mode.

## Why AMD matters

AMD GPU capacity enables the product to run open-source finance workflows without sending sensitive company materials to closed hosted systems. The agent architecture can scale horizontally: multiple document chunks and extraction agents can be batched or parallelized across AMD GPU workers.


## Training benchmark extension

For a stronger AMD Developer Cloud story, run a small LoRA adapter experiment with `training-data/earningspilot-sft.jsonl` and record:

- Adapter training wall-clock time.
- Peak GPU memory.
- Tokens per second during SFT.
- JSON-validity rate before vs. after fine-tuning.
- Golden-sample pass/fail from `npm run eval:sample`.

The point is not that five seed examples create a production finance model. The point is that the repo has the instrumentation and workflow needed to collect analyst corrections, fine-tune adapters on AMD GPUs, and prove quality improvements before deployment.

## Demo fallback note

The Hugging Face Space uses deterministic local analysis by default so judges can test instantly even without provisioned AMD secrets. This is intentional product engineering: the public demo cannot fail because a GPU endpoint is cold or unavailable, while the architecture still cleanly supports AMD-backed inference.
