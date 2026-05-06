# EarningsPilot AMD Architecture

## System goal

EarningsPilot AMD turns raw earnings and filings material into a decision-ready analyst packet. It is designed to run as a Hugging Face Docker Space and to optionally call an AMD Developer Cloud hosted open-source model endpoint.

## Request flow

1. The user uploads files or runs the seeded sample demo.
2. `/api/analyze` converts form uploads or JSON documents into normalized `DocumentInput` objects.
3. `runEarningsPilot` orchestrates five specialized agents.
4. Agents produce typed structures: `Kpi`, `Risk`, `ThesisPoint`, `Evidence`, and `AgentStep`.
5. The frontend renders dashboard cards, KPI tables, bull/bear panels, risk cards, an action memo, and clickable evidence snippets.

## Agent responsibilities

### Parser Agent

- Normalizes text and chunks documents.
- Identifies company and ticker hints.
- Creates the first evidence anchors.

### KPI Extraction Agent

- Searches for financial and operating metric language.
- Extracts values, periods, and trend direction.
- Links every KPI row to evidence.

### Risk Agent

- Detects forward-looking risk language.
- Classifies severity.
- Produces a focused risk register.

### Thesis Agent

- Builds separate bull and bear cases.
- Uses only source-grounded findings.
- Preserves evidence IDs for judge-friendly review.

### Report Agent

- Synthesizes executive summary, tone assessment, and action memo.
- In deterministic demo mode, uses reliable extractive summaries.
- In AMD model mode, calls an OpenAI-compatible endpoint for compact JSON summaries.



## Custom model interpretation

EarningsPilot AMD treats "our model" as a domain-adapted open-source model, not a from-scratch foundation model. The intended owned artifact is **EarningsPilot-Qwen-7B-LoRA**: an EarningsPilot adapter trained on finance-agent tasks and served with its base model on AMD Developer Cloud.

The GPU requirement for the hackathon version is intentionally modest: one AMD Instinct MI300X is enough for 7B-class inference and adapter tuning, while an 8-GPU MI300X node is reserved for larger-model inference, throughput benchmarking, or full fine-tuning research.

## Model improvement loop

EarningsPilot AMD separates demo reliability from model training. The public app can run deterministically, while the production path can improve model behavior through an AMD-hosted fine-tuning loop:

1. Capture analyst corrections and failed extractions.
2. Convert them to chat-format SFT examples like `training-data/earningspilot-sft.jsonl`.
3. Run LoRA/QLoRA fine-tuning on AMD Developer Cloud with ROCm.
4. Evaluate with `scripts/evaluate-sample.mjs` and future golden filing packs.
5. Serve the tuned adapter behind the same OpenAI-compatible endpoint used by the Report Agent.

This keeps the hackathon demo reliable while showing a credible path to domain adaptation.

## AMD Developer Cloud integration

The app is deliberately split between orchestration and inference:

- **Orchestration:** Next.js API route and TypeScript agent pipeline.
- **Inference:** Optional OpenAI-compatible endpoint backed by AMD Developer Cloud GPUs and ROCm-serving infrastructure.

Recommended serving stack:

```text
AMD Developer Cloud GPU VM
└── ROCm runtime
    └── vLLM / TGI / LiteLLM gateway
        └── Qwen2.5-7B-Instruct, Llama, DeepSeek, or Mistral model
```

The application reads:

- `AMD_OPENAI_BASE_URL`
- `AMD_OPENAI_API_KEY`
- `AMD_MODEL_ID`

If these are unset, the public demo remains fully functional and deterministic.

## Reliability choices

- Deterministic fallback avoids demo outages.
- Structured TypeScript types keep dashboard rendering stable.
- Evidence IDs make unsupported claims obvious.
- Prompts are narrow and JSON-only when model mode is enabled.
- The sample dataset ensures judges can complete an end-to-end run in under 60 seconds.

## Deployment targets

- **Primary:** Hugging Face Docker Space on port `7860`.
- **Optional:** Vercel frontend/server deployment.
- **AMD model serving:** AMD Developer Cloud GPU endpoint configured through secrets.
