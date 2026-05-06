# Submission Description

## Product name

EarningsPilot AMD

## One-liner

A multi-agent earnings and filings intelligence system that converts transcripts, SEC excerpts, investor materials, and KPI CSVs into a cited analyst memo.

## What it does

Users upload one or more financial documents or run the seeded sample demo. EarningsPilot AMD parses and chunks the materials, extracts KPIs, flags forward-looking risks, separates bull and bear cases, evaluates management tone, and generates an action memo with an evidence panel.

## Why it matters

Financial research teams spend hours converting long-form source material into decision-ready notes. EarningsPilot AMD reduces that work to a repeatable workflow that is grounded in source snippets, making it easier to audit and share.

## AMD Developer Cloud usage

The application is designed around AMD Developer Cloud as the inference layer for open-source models. In production, the Report Agent can call an OpenAI-compatible endpoint backed by AMD GPUs and ROCm-compatible model serving. The public demo includes deterministic local mode for reliability, while `.env.example` and `architecture.md` document the AMD endpoint configuration.


## Custom model and GPU plan

The project does not claim to train a foundation model from scratch. Instead, the model-ownership path is an EarningsPilot LoRA/QLoRA adapter on top of a strong open-source instruction model, served on AMD Developer Cloud. The recommended hackathon configuration is one AMD Instinct MI300X for 7B-class inference and adapter tuning, with an optional 8x MI300X node for larger model experiments.

## Open-source model strategy

Recommended models include Qwen2.5 Instruct, Llama, DeepSeek, and Mistral-family instruction models. Prompts are narrow, JSON-oriented, and source-grounded.

## Differentiation

EarningsPilot AMD is not a generic PDF chatbot. It is a workflow product with specialized agents, structured outputs, KPI extraction, risk flags, bull/bear analysis, action memo generation, and clickable evidence.

## Target users

- Public-market investors
- Sell-side and buy-side research teams
- Corporate strategy teams
- Investor relations teams
- Operators monitoring competitors and suppliers
