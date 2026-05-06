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

## Why AMD matters

AMD GPU capacity enables the product to run open-source finance workflows without sending sensitive company materials to closed hosted systems. The agent architecture can scale horizontally: multiple document chunks and extraction agents can be batched or parallelized across AMD GPU workers.

## Demo fallback note

The Hugging Face Space uses deterministic local analysis by default so judges can test instantly even without provisioned AMD secrets. This is intentional product engineering: the public demo cannot fail because a GPU endpoint is cold or unavailable, while the architecture still cleanly supports AMD-backed inference.
