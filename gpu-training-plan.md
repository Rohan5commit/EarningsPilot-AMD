# AMD GPU and Model Ownership Plan

## Did the original prompt require a custom model?

The original product goal required **open-source model usage on AMD cloud GPUs** and an **agent architecture**, not training a frontier model from scratch. For a hackathon product, "our own model" should mean a domain-adapted open-source model or adapter that EarningsPilot AMD owns and serves, not a from-scratch foundation model.

The practical interpretation is:

1. Start with a strong open-source base model such as Qwen, Llama, DeepSeek, or Mistral.
2. Collect finance-agent examples and analyst corrections.
3. Train a small LoRA/QLoRA adapter for EarningsPilot-specific behavior.
4. Serve the base model plus EarningsPilot adapter on AMD Developer Cloud.
5. Keep the deterministic path as a reliable public fallback.

This is the right balance for the hackathon: it demonstrates AMD GPU usage and model ownership without risking the demo on an expensive, under-trained model.

## What GPUs are needed?

AMD Developer Cloud currently centers on AMD Instinct MI300X GPU access. AMD describes Developer Cloud as providing access to MI300X GPUs, and AMD's MI300X product materials list 192 GB HBM3 memory per accelerator with high AI throughput. That makes MI300X the natural target for both inference and adapter tuning.

## Recommended GPU tiers for EarningsPilot AMD

| Workload | Recommended AMD GPU | Why | Expected use |
| --- | --- | --- | --- |
| Public UI + deterministic demo | CPU only | No model serving required | Hugging Face Space / Vercel reliability |
| 7B open-source inference | 1x MI300X | Plenty of memory for Qwen/Llama/Mistral 7B-class models and long financial context | Live Report Agent summaries |
| 7B LoRA/QLoRA fine-tuning | 1x MI300X | Adapter training is memory-light compared with full fine-tuning | Create EarningsPilot adapter |
| 14B-32B inference or heavier eval | 1x MI300X, potentially quantized | 192 GB HBM3 gives room for larger open-source models | Better quality demos and batch eval |
| 70B-class inference | 1-8x MI300X depending quantization/context/concurrency | Larger models benefit from tensor parallelism and aggregate memory | Premium deployment path |
| Full fine-tuning large models | 8x MI300X node | Full fine-tuning is far more expensive than LoRA | Not needed for hackathon; future enterprise R&D |

## Recommended hackathon GPU plan

For this project, use **one AMD Instinct MI300X** for the GPU-backed part of the story:

- Serve `Qwen/Qwen2.5-7B-Instruct` or similar with vLLM/TGI/LiteLLM on ROCm.
- Run the seeded eval harness against that endpoint.
- If time allows, train a small LoRA adapter from `training-data/earningspilot-sft.jsonl` plus any additional examples.
- Deploy the endpoint behind `AMD_OPENAI_BASE_URL`.

If a full 8-GPU MI300X node is available, use it for throughput, larger model experiments, and benchmark screenshots. It is not required for the judge demo.

## What to call "our model"

Use this naming in the submission:

**EarningsPilot-Qwen-7B-LoRA**

Definition:

- Base: `Qwen/Qwen2.5-7B-Instruct`
- Adapter: EarningsPilot AMD LoRA adapter trained on finance-agent JSON tasks
- Serving: AMD Developer Cloud, ROCm, OpenAI-compatible endpoint
- Purpose: JSON-grounded KPI extraction, risk classification, thesis synthesis, and action memo generation

This is honest and technically credible: we own the task adapter and evaluation harness, while leveraging a proven open-source base model.

## Minimum training dataset target

The current repo includes seed examples so the path is real. Before claiming quality improvement, expand to at least:

- 50 KPI extraction examples
- 50 risk classification examples
- 50 bull/bear thesis examples
- 50 report/action memo examples
- 25 negative examples where the correct answer is "not supported by evidence"

The first hackathon milestone is not a production-grade model; it is a reproducible AMD GPU training loop.

## Success metrics

Track these before and after adapter tuning:

- JSON-valid response rate
- Required KPI recall
- Required risk recall
- Evidence-support rate
- Hallucinated-claim rate
- Latency and tokens/sec on MI300X
- Cost per analyzed document packet

The included `npm run eval:sample` command is the first version of this gate.
