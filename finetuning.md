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
TRAIN_HOURS=10 \
MAX_STEPS=20000 \
CHECKPOINT_STEPS=250 \
KEEP_CHECKPOINTS=12 \
RESUME_FROM_CHECKPOINT=auto \
TRAIN_FILE=training-data/earningspilot-sft-10h.jsonl \
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
./scripts/amd/start-lora-training.sh
```

The launcher saves every `CHECKPOINT_STEPS` optimizer steps and resumes from the newest `checkpoint-*` directory when `RESUME_FROM_CHECKPOINT=auto`. Use `./scripts/amd/training-progress.sh` to print the latest checkpoint step, percent progress against `MAX_STEPS`, the last Trainer log entry, and the last 30 log lines.

## Example TRL / PEFT recipe

> Treat this as the production recipe for AMD Developer Cloud, not something required for the deterministic public demo.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
pip install transformers peft trl accelerate datasets bitsandbytes
```

```python
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

model_id = "Qwen/Qwen2.5-7B-Instruct"
dataset = load_dataset("json", data_files="training-data/earningspilot-sft.jsonl", split="train")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto", trust_remote_code=True)

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"
)

args = TrainingArguments(
    output_dir="outputs/earningspilot-qwen-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=5,
    save_steps=25,
    bf16=True,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    peft_config=lora,
    args=args,
    max_seq_length=2048
)
trainer.train()
trainer.save_model()
```

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
