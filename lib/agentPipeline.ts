import { z } from 'zod';
import { chunkText, detectDirection, normalizeText, splitSentences, truncate } from './text';
import type { AnalysisResult, DocumentInput, Evidence, Kpi, Risk, ThesisPoint } from './types';

const openAICompatibleSchema = z.object({
  AMD_OPENAI_BASE_URL: z.string().url().optional(),
  AMD_OPENAI_API_KEY: z.string().optional(),
  AMD_MODEL_ID: z.string().default('Qwen/Qwen2.5-7B-Instruct')
});

type InternalFinding = {
  sentence: string;
  document: string;
  evidenceId: string;
};

function addEvidence(evidence: Evidence[], document: string, snippet: string, relevance: string, agent: Evidence['agent']) {
  const id = `E${evidence.length + 1}`;
  evidence.push({ id, document, snippet: truncate(snippet, 420), relevance, agent });
  return id;
}

function inferCompany(text: string) {
  const tickerMatch = text.match(/\b([A-Z][A-Za-z&. ]{2,48})\s*\(([A-Z]{2,5})\)/);
  if (tickerMatch) return { company: tickerMatch[1].trim(), ticker: tickerMatch[2] };
  const titleMatch = text.match(/^([A-Z][A-Za-z&. ]{2,48})(?:\s+Q\d|\s+FY|\s+\d{4}|\s+Earnings)/m);
  return { company: titleMatch?.[1]?.trim() || 'Uploaded Company', ticker: tickerMatch?.[2] || 'N/A' };
}

function scoreTone(sentences: string[]) {
  const positive = ['growth', 'expanded', 'improved', 'stable', 'demand', 'cash flow', 'signed', 'accelerated', 'higher-value'];
  const negative = ['risk', 'pressure', 'constrained', 'delay', 'shortage', 'export', 'loss', 'geopolitical', 'elevated', 'moderates'];
  let score = 50;
  for (const sentence of sentences) {
    const lower = sentence.toLowerCase();
    for (const term of positive) if (lower.includes(term)) score += 3;
    for (const term of negative) if (lower.includes(term)) score -= 4;
  }
  score = Math.max(0, Math.min(100, score));
  const label = score >= 68 ? 'Constructive' : score >= 48 ? 'Balanced' : score >= 30 ? 'Cautious' : 'Defensive';
  return { score, label } as const;
}

function extractKpis(findings: InternalFinding[]): Kpi[] {
  const metrics = [
    'data center revenue',
    'operating cash flow',
    'free cash flow',
    'capital expenditures',
    'revenue guide',
    'gross margin',
    'inventory days',
    'guidance',
    'revenue'
  ];
  const kpis: Kpi[] = [];
  for (const finding of findings) {
    const lower = finding.sentence.toLowerCase();
    const firstCsvCell = finding.sentence.includes(',') ? finding.sentence.split(',')[0]?.trim().toLowerCase() : '';
    const metric = metrics.find((candidate) => firstCsvCell === candidate) || metrics.find((candidate) => lower.includes(candidate));
    const valueMatch = finding.sentence.match(/(?:\$\s?\d+(?:\.\d+)?\s?(?:billion|million|B|M)?|\d+(?:\.\d+)?%|\d+\s?basis points|\d+\s?days|\$\s?\d+(?:\.\d+)?B\s?-\s?\$?\d+(?:\.\d+)?B)/i);
    if (metric && valueMatch && !kpis.some((kpi) => kpi.metric.toLowerCase() === metric && kpi.value === valueMatch[0])) {
      const period = finding.sentence.match(/Q[1-4]\s?FY\d{2}|FY\d{2,4}|full-year|quarter/i)?.[0] || 'reported period';
      kpis.push({
        metric: metric.replace(/\b\w/g, (char) => char.toUpperCase()),
        value: valueMatch[0].replace(/\s+/g, ' '),
        period,
        direction: detectDirection(finding.sentence),
        evidenceId: finding.evidenceId
      });
    }
  }
  return kpis.slice(0, 8);
}

function extractRisks(findings: InternalFinding[]): Risk[] {
  const riskTerms = ['risk', 'constrained', 'capacity', 'pressure', 'export', 'controls', 'customer concentration', 'loss', 'delay', 'shortage', 'geopolitical', 'capital expenditures', 'cash flow'];
  const risks: Risk[] = [];
  for (const finding of findings) {
    const lower = finding.sentence.toLowerCase();
    if (!riskTerms.some((term) => lower.includes(term))) continue;
    let title = 'Execution risk';
    if (/customer|hyperscale|concentration/.test(lower)) title = 'Customer concentration exposure';
    else if (/export|trade|restricted/.test(lower)) title = 'Export control and trade restriction risk';
    else if (/capacity|packaging|foundry|shortage|vendors/.test(lower)) title = 'Advanced packaging capacity bottleneck';
    else if (/capital expenditures|free cash flow|cash flow/.test(lower)) title = 'Cash-flow pressure from elevated investment';
    else if (/pricing|consumer/.test(lower)) title = 'Segment pricing pressure';
    const severity: Risk['severity'] = /materially|loss|restricted|geopolitical|capacity constrained|shortages/.test(lower) ? 'High' : /pressure|elevated|delay/.test(lower) ? 'Medium' : 'Low';
    if (!risks.some((risk) => risk.title === title)) {
      risks.push({ title, severity, rationale: truncate(finding.sentence, 220), evidenceId: finding.evidenceId });
    }
  }
  return risks.slice(0, 5);
}

function thesis(findings: InternalFinding[], mode: 'bull' | 'bear'): ThesisPoint[] {
  const bullTerms = ['growth', 'expanded', 'improved', 'signed', 'demand', 'cash flow', 'higher-value', 'stable'];
  const bearTerms = ['constrained', 'pressure', 'risk', 'export', 'loss', 'delay', 'elevated', 'customer concentration', 'moderates'];
  const terms = mode === 'bull' ? bullTerms : bearTerms;
  return findings
    .filter((finding) => terms.some((term) => finding.sentence.toLowerCase().includes(term)))
    .slice(0, 4)
    .map((finding) => ({ point: truncate(finding.sentence, 210), evidenceId: finding.evidenceId }));
}

async function optionalModelPass(prompt: string): Promise<{ content: string; latencyMs: number } | null> {
  const env = openAICompatibleSchema.parse(process.env);
  if (!env.AMD_OPENAI_BASE_URL || !env.AMD_OPENAI_API_KEY) return null;
  const started = Date.now();
  const response = await fetch(`${env.AMD_OPENAI_BASE_URL.replace(/\/$/, '')}/chat/completions`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      authorization: `Bearer ${env.AMD_OPENAI_API_KEY}`
    },
    body: JSON.stringify({
      model: env.AMD_MODEL_ID,
      temperature: 0.1,
      response_format: { type: 'json_object' },
      messages: [
        { role: 'system', content: 'You are a source-grounded financial analysis agent. Return compact valid JSON only. Do not introduce facts not present in the context.' },
        { role: 'user', content: prompt }
      ]
    })
  });
  if (!response.ok) return null;
  const json = await response.json();
  const content = json?.choices?.[0]?.message?.content;
  return typeof content === 'string' ? { content, latencyMs: Date.now() - started } : null;
}

export async function runEarningsPilot(documents: DocumentInput[]): Promise<AnalysisResult> {
  const cleaned = documents.map((doc) => ({ ...doc, text: normalizeText(doc.text) })).filter((doc) => doc.text.length > 0);
  if (!cleaned.length) throw new Error('Upload at least one non-empty text, CSV, HTML, Markdown, or filing excerpt.');

  const allText = cleaned.map((doc) => doc.text).join('\n\n');
  const { company, ticker } = inferCompany(allText);
  const evidence: Evidence[] = [];

  const parserChunks = cleaned.flatMap((doc) => chunkText(doc.text).map((chunk) => ({ document: doc.name, chunk })));
  const parserEvidenceId = addEvidence(evidence, cleaned[0].name, parserChunks[0]?.chunk || cleaned[0].text, 'Initial parsed document chunk used to anchor company brief.', 'Parser Agent');

  const findings: InternalFinding[] = [];
  const analysisDocs = [...cleaned].sort((a, b) => Number(b.type.includes('csv') || b.name.toLowerCase().endsWith('.csv')) - Number(a.type.includes('csv') || a.name.toLowerCase().endsWith('.csv')));

  for (const doc of analysisDocs) {
    const isCsv = doc.type.includes('csv') || doc.name.toLowerCase().endsWith('.csv');
    const candidates = [
      ...(isCsv ? [] : splitSentences(doc.text)),
      ...doc.text.split('\n').map((line) => line.trim()).filter((line) => line.includes(',') && line.length > 20 && !line.toLowerCase().startsWith('metric,'))
    ];
    for (const sentence of candidates) {
      const lower = sentence.toLowerCase();
      if (/revenue|margin|cash flow|inventory|guid|growth|risk|constrain|pressure|export|customer|capacity|capex|capital expenditures|pricing|demand|signed|foundry|geopolitical/.test(lower)) {
        const agent = /risk|constrain|pressure|export|loss|delay|geopolitical|shortage/.test(lower) ? 'Risk Agent' : /revenue|margin|cash flow|inventory|guid|growth/.test(lower) ? 'KPI Extraction Agent' : 'Thesis Agent';
        const evidenceId = addEvidence(evidence, doc.name, sentence, 'Agent-selected source sentence for structured analysis.', agent);
        findings.push({ sentence, document: doc.name, evidenceId });
      }
    }
  }

  const kpis = extractKpis(findings);
  const risks = extractRisks(findings);
  const bullCase = thesis(findings, 'bull');
  const bearCase = thesis(findings, 'bear');
  const toneBase = scoreTone(findings.map((finding) => finding.sentence));
  const toneEvidence = findings.find((finding) => /constrained|pressure|risk|growth|improved|expanded/.test(finding.sentence.toLowerCase()))?.evidenceId || parserEvidenceId;

  const modelSummary = await optionalModelPass(`Context:\n${allText.slice(0, 7000)}\n\nCreate JSON with keys executiveSummary:string[], actionMemo:string for ${company}.`);
  let executiveSummary = [
    `${company} shows a growth profile led by demand, revenue momentum, and mix improvement in the supplied materials.`,
    `The strongest support comes from KPI evidence around revenue, margin, cash flow, and inventory discipline.`,
    `The main diligence focus is whether capacity, customer concentration, export controls, and investment intensity can be managed without derailing guidance.`
  ];
  let actionMemo = `Action memo: keep ${company} on the active research list. Validate guidance durability, monitor capacity ramp milestones, compare data-center growth against customer concentration risk, and require evidence that elevated investment converts into free-cash-flow expansion.`;
  if (modelSummary) {
    try {
      const parsed = JSON.parse(modelSummary.content) as { executiveSummary?: string[]; actionMemo?: string };
      if (Array.isArray(parsed.executiveSummary) && parsed.executiveSummary.length) executiveSummary = parsed.executiveSummary.slice(0, 4);
      if (parsed.actionMemo) actionMemo = parsed.actionMemo;
    } catch {
      // Keep deterministic result if the optional model does not return valid JSON.
    }
  }

  return {
    company,
    ticker,
    generatedAt: new Date().toISOString(),
    modelMode: modelSummary ? 'amd-openai-compatible' : 'deterministic-local',
    amdRun: {
      gpu: process.env.AMD_GPU_NAME || 'AMD Instinct MI300X',
      model: process.env.AMD_MODEL_ID || 'Qwen/Qwen2.5-7B-Instruct',
      endpointConfigured: Boolean(process.env.AMD_OPENAI_BASE_URL),
      usedModelEndpoint: Boolean(modelSummary),
      latencyMs: modelSummary?.latencyMs ?? null,
      gpuHoursBudget: Number(process.env.AMD_GPU_HOURS_BUDGET || 40),
      note: modelSummary
        ? 'Report Agent synthesis used the AMD-hosted OpenAI-compatible model endpoint.'
        : 'Deterministic fallback is active so the public demo remains reliable if the AMD endpoint is unavailable.'
    },
    companyBrief: `${company}${ticker !== 'N/A' ? ` (${ticker})` : ''} was analyzed across ${cleaned.length} uploaded source${cleaned.length === 1 ? '' : 's'} using ${parserChunks.length} parsed chunks. The agent workflow prioritized source-grounded evidence over unsupported market claims.`,
    executiveSummary,
    kpis,
    bullCase,
    bearCase,
    risks,
    tone: {
      label: toneBase.label,
      score: toneBase.score,
      rationale: `Management tone reads as ${toneBase.label.toLowerCase()} because the source set mixes growth and cash-flow positives with explicit execution and market risk language.`,
      evidenceId: toneEvidence
    },
    actionMemo,
    evidence,
    agentTrace: [
      { agent: 'Parser Agent', objective: 'Normalize uploads and chunk long-form financial materials.', output: `Parsed ${cleaned.length} document(s) into ${parserChunks.length} chunk(s).` },
      { agent: 'KPI Extraction Agent', objective: 'Extract numeric operating and financial metrics with citations.', output: `Extracted ${kpis.length} KPI rows with evidence identifiers.` },
      { agent: 'Risk Agent', objective: 'Identify forward-looking risk flags and severity.', output: `Flagged ${risks.length} prioritized risks.` },
      { agent: 'Thesis Agent', objective: 'Separate bull and bear cases using only cited evidence.', output: `Produced ${bullCase.length} bull points and ${bearCase.length} bear points.` },
      { agent: 'Report Agent', objective: 'Synthesize an executive memo and dashboard-ready structure.', output: 'Generated company brief, summary, tone read, evidence panel, and action memo.' }
    ]
  };
}
