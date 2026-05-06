# EarningsPilot AMD Slide Deck Outline

## Slide 1 — Title

EarningsPilot AMD: Multi-Agent Earnings and Filings Intelligence

## Slide 2 — Problem

Analysts must read transcripts, 10-K risk factors, investor decks, and KPI tables under time pressure. Generic chatbots summarize text but do not reliably produce cited, decision-ready research packets.

## Slide 3 — Solution

Upload financial materials and receive:

- Executive summary
- KPI table
- Bull vs. bear case
- Risk register
- Tone read
- Action memo
- Evidence snippets

## Slide 4 — Agent Workflow

Parser Agent → KPI Extraction Agent → Risk Agent → Thesis Agent → Report Agent.

Each agent has a narrow role and emits structured output.

## Slide 5 — AMD Developer Cloud Architecture

Next.js app + typed orchestration + optional OpenAI-compatible endpoint backed by AMD Developer Cloud GPUs, ROCm, and open-source models such as Qwen, Llama, DeepSeek, or Mistral. Include the model-improvement loop: seed SFT data → LoRA on AMD → golden eval → endpoint deployment.

## Slide 6 — Live Demo Screens

Show landing page, upload area, dashboard, KPI table, evidence viewer, and final memo.

## Slide 7 — Business Value

- Reduces time to first analyst memo
- Improves research consistency
- Preserves source evidence
- Helps investors, IR, corp-dev, finance, and strategy teams

## Slide 8 — Originality

Unlike a PDF chatbot, EarningsPilot AMD creates a structured research packet and exposes the agent trace and source evidence.

## Slide 9 — Benchmark / Workload Note

Long financial documents create repeated inference calls. AMD GPU-backed open-source serving supports private, scalable, cost-conscious analysis. Optional LoRA fine-tuning uses analyst corrections to improve JSON validity and evidence grounding.

## Slide 10 — Submission

- Public GitHub repo
- Hugging Face Space
- Demo video
- Slide deck
- Application URL
