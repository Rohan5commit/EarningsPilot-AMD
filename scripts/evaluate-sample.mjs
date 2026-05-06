import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const root = process.cwd();
const baseUrl = process.env.EARNINGSPILOT_BASE_URL || 'http://localhost:3000';
const golden = JSON.parse(readFileSync(join(root, 'eval', 'golden-sample.json'), 'utf8'));

const docs = [
  ['transcript', 'atlas-components-earnings-transcript.txt', 'text/plain'],
  ['kpis', 'atlas-components-kpis.csv', 'text/csv'],
  ['risks', 'atlas-components-risk-factors.txt', 'text/plain']
].map(([id, name, type]) => ({ id, name, type, text: readFileSync(join(root, 'sample-data', name), 'utf8') }));

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

const response = await fetch(`${baseUrl.replace(/\/$/, '')}/api/analyze`, {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ documents: docs })
});

const result = await response.json();
if (!response.ok) {
  console.error(result);
  process.exit(1);
}

const metricNames = new Set(result.kpis.map((kpi) => kpi.metric));
const riskTitles = new Set(result.risks.map((risk) => risk.title));
const evidenceAgents = new Set(result.evidence.map((item) => item.agent));

const checks = [
  ['company', () => assert(result.company === golden.company, `Expected company ${golden.company}, got ${result.company}`)],
  ['ticker', () => assert(result.ticker === golden.ticker, `Expected ticker ${golden.ticker}, got ${result.ticker}`)],
  ['kpi_count', () => assert(result.kpis.length >= golden.minimumKpis, `Expected at least ${golden.minimumKpis} KPIs, got ${result.kpis.length}`)],
  ['risk_count', () => assert(result.risks.length >= golden.minimumRisks, `Expected at least ${golden.minimumRisks} risks, got ${result.risks.length}`)],
  ...golden.requiredKpiMetrics.map((metric) => [`kpi:${metric}`, () => assert(metricNames.has(metric), `Missing KPI metric: ${metric}`)]),
  ...golden.requiredRiskTitles.map((title) => [`risk:${title}`, () => assert(riskTitles.has(title), `Missing risk title: ${title}`)]),
  ...golden.requiredEvidenceAgents.map((agent) => [`evidence:${agent}`, () => assert(evidenceAgents.has(agent), `Missing evidence agent: ${agent}`)])
];

const passed = [];
for (const [name, check] of checks) {
  check();
  passed.push(name);
}

console.log(JSON.stringify({
  status: 'pass',
  baseUrl,
  modelMode: result.modelMode,
  passed,
  observed: {
    company: result.company,
    ticker: result.ticker,
    kpis: result.kpis.length,
    risks: result.risks.length,
    evidenceItems: result.evidence.length
  }
}, null, 2));
