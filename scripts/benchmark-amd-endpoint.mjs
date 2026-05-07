import { performance } from 'node:perf_hooks';

const baseUrl = process.env.AMD_OPENAI_BASE_URL;
const apiKey = process.env.AMD_OPENAI_API_KEY || 'local-dev-key';
const model = process.env.AMD_MODEL_ID || 'Qwen/Qwen2.5-7B-Instruct';
const runs = Number(process.env.BENCHMARK_RUNS || 3);
const maxTokens = Number(process.env.BENCHMARK_MAX_TOKENS || 96);
const requestTimeoutMs = Number(process.env.BENCHMARK_TIMEOUT_MS || 120000);

if (!baseUrl) {
  console.error('Missing AMD_OPENAI_BASE_URL. Point it at your AMD-hosted OpenAI-compatible /v1 endpoint.');
  process.exit(1);
}

const prompt = `Extract KPI, risk, and action memo JSON from this evidence only:\nRevenue grew 18% to $2.84 billion. Gross margin expanded to 43.2%. Data center revenue grew 41% to $3.7 billion. The company remains capacity constrained in advanced packaging and is watching export controls.`;

async function runOnce(index) {
  const started = performance.now();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), requestTimeoutMs);
  let response;
  try {
    response = await fetch(`${baseUrl.replace(/\/$/, '')}/chat/completions`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${apiKey}`
      },
      signal: controller.signal,
      body: JSON.stringify({
        model,
        temperature: 0.1,
        max_tokens: maxTokens,
        response_format: { type: 'json_object' },
        messages: [
          { role: 'system', content: 'You are EarningsPilot-Qwen-7B-LoRA. Return compact valid JSON only. Do not invent unsupported claims.' },
          { role: 'user', content: prompt }
        ]
      })
    });
  } catch (error) {
    if (error?.name === 'AbortError') {
      throw new Error(`Run ${index} timed out after ${requestTimeoutMs}ms. If this is the Transformers fallback, lower BENCHMARK_MAX_TOKENS or wait for /health before running.`);
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
  const latencyMs = Math.round(performance.now() - started);
  const json = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(`Run ${index} failed with HTTP ${response.status}: ${JSON.stringify(json).slice(0, 500)}`);
  }
  const content = json?.choices?.[0]?.message?.content || '';
  const usage = json?.usage || {};
  return {
    run: index,
    latencyMs,
    outputChars: content.length,
    promptTokens: usage.prompt_tokens ?? null,
    completionTokens: usage.completion_tokens ?? null,
    totalTokens: usage.total_tokens ?? null,
    approxOutputCharsPerSecond: Math.round((content.length / Math.max(latencyMs, 1)) * 1000)
  };
}

const results = [];
for (let i = 1; i <= runs; i += 1) {
  results.push(await runOnce(i));
}

const avgLatencyMs = Math.round(results.reduce((sum, result) => sum + result.latencyMs, 0) / results.length);
const avgOutputCharsPerSecond = Math.round(results.reduce((sum, result) => sum + result.approxOutputCharsPerSecond, 0) / results.length);

console.log(JSON.stringify({
  status: 'pass',
  endpoint: baseUrl.replace(/\/v1\/?$/, '/v1'),
  model,
  runs,
  maxTokens,
  requestTimeoutMs,
  avgLatencyMs,
  avgOutputCharsPerSecond,
  gpu: process.env.AMD_GPU_NAME || 'AMD Instinct MI300X',
  results
}, null, 2));
