import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

export async function GET() {
  const baseUrl = process.env.AMD_OPENAI_BASE_URL;
  const apiKey = process.env.AMD_OPENAI_API_KEY;
  const model = process.env.AMD_MODEL_ID || 'Qwen/Qwen2.5-7B-Instruct';

  if (!baseUrl || !apiKey) {
    return NextResponse.json({
      connected: false,
      configured: false,
      model,
      latencyMs: null,
      status: 'AMD endpoint not configured; deterministic fallback is active.'
    });
  }

  const normalizedBase = baseUrl.replace(/\/$/, '');
  const started = Date.now();

  try {
    const response = await fetch(`${normalizedBase}/chat/completions`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model,
        temperature: 0,
        max_tokens: 12,
        response_format: { type: 'json_object' },
        messages: [
          { role: 'system', content: 'Return compact valid JSON only.' },
          { role: 'user', content: 'Return {"ok":true}.' }
        ]
      })
    });

    const latencyMs = Date.now() - started;
    if (!response.ok) {
      return NextResponse.json({
        connected: false,
        configured: true,
        model,
        latencyMs,
        status: `AMD endpoint responded with HTTP ${response.status}.`
      });
    }

    return NextResponse.json({
      connected: true,
      configured: true,
      model,
      latencyMs,
      status: 'AMD endpoint reachable and healthy.'
    });
  } catch (error) {
    console.error('AMD health check failed:', error);

    return NextResponse.json({
      connected: false,
      configured: true,
      model,
      latencyMs: null,
      status: 'Unable to reach AMD endpoint.'
    });
  }
}
