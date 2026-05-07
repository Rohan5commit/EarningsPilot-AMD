# EarningsPilot AMD Demo Script

## 60-second judge demo

**0:00-0:10 — Landing page**

"EarningsPilot AMD is a multi-agent earnings and filings intelligence system for investors and operators. It is not a chatbot; it produces a structured analyst memo with citations."

**0:10-0:20 — Run sample**

Click **Run instant sample demo**.

"I am using the seeded transcript, KPI CSV, and risk-factor excerpt so the demo is reproducible on Hugging Face Spaces."

**0:20-0:35 — Dashboard**

Point to the company brief, executive summary, tone badge, and agent trace.

"Five agents collaborate: Parser, KPI Extraction, Risk, Thesis, and Report. Each agent contributes a typed output used by the dashboard."

**0:35-0:48 — Evidence**

Click a KPI evidence chip and a risk card.

"Every KPI and thesis point is grounded in a source snippet. This prevents the generic PDF-chatbot problem where the system sounds confident but cannot show its work."

**0:48-1:00 — Business and AMD story**

"In production, our custom model path is EarningsPilot-Qwen-7B-LoRA: a Qwen-class open-source base model plus an EarningsPilot finance adapter trained and served on AMD Developer Cloud. One MI300X is enough for the 7B inference and LoRA path; larger MI300X nodes are reserved for bigger models or throughput benchmarks."

## 3-minute video outline

1. Problem: earnings materials are long, repetitive, and time-sensitive.
2. Product: upload materials and receive a source-grounded action memo.
3. Agent architecture: five specialized agents with typed outputs.
4. Demo: run sample, inspect KPI table, click evidence, review risks and memo.
5. AMD: ROCm-compatible serving path, optional LoRA fine-tuning, and open-source models on AMD Developer Cloud.
6. Business value: faster diligence, better handoffs, repeatable research packets.
7. Closing: Hugging Face Space and GitHub repo are ready for judges.
