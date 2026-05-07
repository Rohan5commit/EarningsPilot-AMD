import fs from 'node:fs';
import path from 'node:path';

const output = process.env.SFT_OUTPUT || 'training-data/earningspilot-sft-expanded.jsonl';
const count = Number(process.env.SFT_COUNT || 5000);

const companies = [
  ['Atlas Components', 'ATLS', 'AI server power modules', 'advanced packaging'],
  ['Northstar Compute', 'NSTC', 'accelerated servers', 'HBM supply'],
  ['VectorGrid Systems', 'VGRD', 'data center networking', 'optical module capacity'],
  ['HelioSemi', 'HLSM', 'edge AI processors', 'foundry cycle time'],
  ['Meridian Devices', 'MRDN', 'industrial automation controllers', 'customer qualification delays'],
  ['Summit Memory', 'SMMT', 'high-bandwidth memory subsystems', 'wafer allocation'],
  ['Copperline Robotics', 'CPRB', 'warehouse robotics platforms', 'component shortages'],
  ['Apex Cloud Tools', 'APXT', 'AI observability software', 'enterprise renewal timing'],
  ['IonDrive Power', 'IOND', 'EV power electronics', 'battery-pack production timing'],
  ['Pioneer Fabrication', 'PNFB', 'specialty substrates', 'Malaysia line ramp']
];

const periods = ['Q1 FY26', 'Q2 FY26', 'Q3 FY26', 'Q4 FY25', 'FY25', 'reported quarter'];
const riskCatalog = [
  ['Customer concentration exposure', 'High', 'hyperscale customers represented {pct}% of revenue, and loss or delay of a major customer program could materially affect results'],
  ['Advanced packaging capacity bottleneck', 'High', '{constraint} remains capacity constrained through the first half of fiscal 2026'],
  ['Export control and trade restriction risk', 'High', 'export controls and trade restrictions may reduce sales into restricted AI infrastructure markets'],
  ['Cash-flow pressure from elevated investment', 'Medium', 'capital expenditures are expected to remain elevated during the {facility} ramp'],
  ['Segment pricing pressure', 'Medium', 'consumer and industrial products remain under pricing pressure despite stable data center pricing'],
  ['Supply-chain execution risk', 'Medium', 'third-party foundry partners and outsourced assembly vendors may experience delays or quality excursions'],
  ['Demand normalization risk', 'Medium', 'management said demand may moderate after a pull-forward in enterprise orders'],
  ['Inventory correction risk', 'Low', 'channel inventory remains above target and may take two quarters to normalize']
];

const metricCatalog = [
  ['Revenue', '${revenue} billion', 'up {growth}% year over year', 'up'],
  ['Gross margin', '{margin}%', 'expanded {bps} basis points year over year', 'up'],
  ['Free cash flow', '${fcf} million', 'improved from the prior quarter', 'up'],
  ['Operating cash flow', '${ocf} million', 'increased on working-capital discipline', 'up'],
  ['Data center revenue', '${dc} billion', 'grew {dcGrowth}% year over year', 'up'],
  ['Inventory days', '{inventory} days', 'improved sequentially from {priorInventory} days', 'down'],
  ['Revenue guide', '${guideLow}B-${guideHigh}B', 'management guidance for next quarter', 'unknown'],
  ['Capital expenditures', '${capex} million', 'rose with capacity expansion', 'up']
];

function num(seed, min, max, decimals = 0) {
  const x = Math.sin(seed * 9301 + 49297) * 233280;
  const frac = x - Math.floor(x);
  const value = min + frac * (max - min);
  return decimals ? value.toFixed(decimals) : Math.round(value).toString();
}

function fill(template, seed, companyTuple) {
  const [, , product, constraint] = companyTuple;
  const values = {
    revenue: num(seed + 1, 1.2, 8.8, 2),
    growth: num(seed + 2, 4, 42),
    margin: num(seed + 3, 32, 64, 1),
    bps: num(seed + 4, 80, 520),
    fcf: num(seed + 5, 80, 950),
    ocf: num(seed + 6, 120, 1400),
    dc: num(seed + 7, 0.8, 7.4, 2),
    dcGrowth: num(seed + 8, 12, 72),
    inventory: num(seed + 9, 46, 92),
    priorInventory: num(seed + 10, 74, 120),
    guideLow: num(seed + 11, 1.4, 7.9, 2),
    guideHigh: num(seed + 12, 2.0, 8.8, 2),
    capex: num(seed + 13, 100, 1600),
    pct: num(seed + 14, 24, 48),
    product,
    constraint,
    facility: ['Malaysia line', 'Arizona test facility', 'Dresden module plant', 'Penang assembly line'][seed % 4]
  };
  return template.replace(/\{(\w+)\}/g, (_, key) => values[key] ?? `{${key}}`);
}

function example(messages) {
  return JSON.stringify({ messages });
}

function system(agent = 'financial-document agent') {
  return {
    role: 'system',
    content: `You are an EarningsPilot AMD ${agent}. Return compact valid JSON grounded only in supplied evidence. Do not invent unsupported claims.`
  };
}

function kpiExample(i) {
  const c = companies[i % companies.length];
  const [company, ticker] = c;
  const period = periods[i % periods.length];
  const m1 = metricCatalog[i % metricCatalog.length];
  const m2 = metricCatalog[(i + 3) % metricCatalog.length];
  const s1 = `${company} (${ticker}) reported ${m1[0].toLowerCase()} of ${fill(m1[1], i, c)}, ${fill(m1[2], i, c)} in ${period}.`;
  const s2 = `${m2[0]} was ${fill(m2[1], i + 20, c)}, and management said it ${fill(m2[2], i + 20, c)}.`;
  return example([
    system('KPI Extraction Agent'),
    { role: 'user', content: `Agent: KPI Extraction Agent\nDocument: ${s1} ${s2}\nReturn KPI JSON rows with metric, value, period, direction, and exact evidence.` },
    { role: 'assistant', content: JSON.stringify({ kpis: [
      { metric: m1[0], value: fill(m1[1], i, c), period, direction: m1[3], evidence: s1 },
      { metric: m2[0], value: fill(m2[1], i + 20, c), period: 'reported period', direction: m2[3], evidence: s2 }
    ] }) }
  ]);
}

function riskExample(i) {
  const c = companies[i % companies.length];
  const [company] = c;
  const r1 = riskCatalog[i % riskCatalog.length];
  const r2 = riskCatalog[(i + 2) % riskCatalog.length];
  const e1 = `${company} disclosed that ${fill(r1[2], i, c)}.`;
  const e2 = `Management also warned that ${fill(r2[2], i + 40, c)}.`;
  return example([
    system('Risk Agent'),
    { role: 'user', content: `Agent: Risk Agent\nDocument: ${e1} ${e2}\nReturn risk JSON with title, severity, rationale, and exact evidence. Do not add risks not stated in the evidence.` },
    { role: 'assistant', content: JSON.stringify({ risks: [
      { title: r1[0], severity: r1[1], rationale: fill(r1[2], i, c), evidence: e1 },
      { title: r2[0], severity: r2[1], rationale: fill(r2[2], i + 40, c), evidence: e2 }
    ] }) }
  ]);
}

function thesisExample(i) {
  const c = companies[i % companies.length];
  const [company, ticker, product, constraint] = c;
  const dc = fill('${dc} billion', i, c);
  const growth = fill('{dcGrowth}%', i, c);
  const bull = `${company} (${ticker}) said demand for ${product} lifted data center revenue to ${dc}, up ${growth} year over year.`;
  const bear = `${company} remains exposed because ${constraint} could limit shipments through the first half of fiscal 2026.`;
  return example([
    system('Thesis Agent'),
    { role: 'user', content: `Agent: Thesis Agent\nDocument: ${bull} ${bear}\nReturn bullCase and bearCase JSON using only the supplied evidence.` },
    { role: 'assistant', content: JSON.stringify({ bullCase: [{ point: bull, evidence: bull }], bearCase: [{ point: bear, evidence: bear }] }) }
  ]);
}

function reportExample(i) {
  const c = companies[i % companies.length];
  const [company, ticker] = c;
  const rev = fill('${revenue} billion', i, c);
  const margin = fill('{margin}%', i, c);
  const risk = fill(riskCatalog[i % riskCatalog.length][2], i, c);
  const evidence = `${company} (${ticker}) reported revenue of ${rev}, gross margin of ${margin}, and disclosed that ${risk}.`;
  return example([
    system('Report Agent'),
    { role: 'user', content: `Agent: Report Agent\nContext: ${evidence}\nReturn executiveSummary:string[] and actionMemo:string JSON. Do not add unsupported market claims.` },
    { role: 'assistant', content: JSON.stringify({
      executiveSummary: [
        `${company} reported revenue of ${rev} and gross margin of ${margin} in the supplied evidence.`,
        `The key diligence item is that ${risk}.`
      ],
      actionMemo: `Keep ${company} on the active research list. Validate revenue durability and margin quality while monitoring the disclosed risk: ${risk}.`
    }) }
  ]);
}

function toneExample(i) {
  const c = companies[i % companies.length];
  const [company, , product, constraint] = c;
  const positive = `${company} highlighted improving demand for ${product}, better inventory days, and stronger operating cash flow.`;
  const negative = `Management also cited ${constraint}, export controls, and pricing pressure in non-core products.`;
  return example([
    system('Tone Agent'),
    { role: 'user', content: `Agent: Tone Agent\nDocument: ${positive} ${negative}\nReturn tone JSON with label, score, rationale, and exact evidence.` },
    { role: 'assistant', content: JSON.stringify({ tone: {
      label: 'Balanced',
      score: 58 + (i % 7),
      rationale: `The source balances demand and cash-flow positives against ${constraint}, export-control, and pricing-pressure risks.`,
      evidence: `${positive} ${negative}`
    } }) }
  ]);
}

function unsupportedExample(i) {
  const c = companies[i % companies.length];
  const [company] = c;
  const evidence = `${company} reported revenue growth and noted capacity constraints. The document did not discuss dividend policy, buybacks, debt maturity, or market share.`;
  return example([
    system('Evidence Grounding Agent'),
    { role: 'user', content: `Agent: Evidence Grounding Agent\nDocument: ${evidence}\nQuestion: Should the report claim dividend growth, buyback acceleration, or market-share gains? Return JSON.` },
    { role: 'assistant', content: JSON.stringify({ supportedClaims: ['revenue growth', 'capacity constraints'], unsupportedClaims: ['dividend growth', 'buyback acceleration', 'market-share gains'], instruction: 'Do not include unsupported claims in the final memo.' }) }
  ]);
}

const builders = [kpiExample, riskExample, thesisExample, reportExample, toneExample, unsupportedExample];
const rows = [];
for (let i = 0; i < count; i += 1) {
  rows.push(builders[i % builders.length](i));
}

fs.mkdirSync(path.dirname(output), { recursive: true });
fs.writeFileSync(output, `${rows.join('\n')}\n`);
console.log(JSON.stringify({ output, count, bytes: fs.statSync(output).size }, null, 2));
