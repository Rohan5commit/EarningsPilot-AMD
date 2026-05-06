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
