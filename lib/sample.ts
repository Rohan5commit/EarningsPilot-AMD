import type { DocumentInput } from './types';

export const sampleDocuments: DocumentInput[] = [
  {
    id: 'sample-transcript',
    name: 'Atlas Components Q4 FY25 Earnings Call Transcript.txt',
    type: 'text/plain',
    text: `Atlas Components (ATLS) Q4 FY25 Earnings Call\n\nCEO Maya Rao: We closed fiscal 2025 with revenue of $2.84 billion, up 18% year over year, driven by accelerated demand for AI server power modules and a recovery in industrial automation. Gross margin expanded to 43.2%, up 310 basis points from last year, as mix shifted toward higher-value data center products.\n\nCFO Daniel Kim: Operating cash flow was $412 million for the quarter and free cash flow was $286 million. Full-year revenue was $10.4 billion, up 15%. Data center revenue grew 41% year over year to $3.7 billion. Inventory days improved to 68 from 81 last quarter.\n\nCEO Maya Rao: Customers are asking us for more secure supply and faster qualification cycles. We signed two multi-year supply agreements with hyperscale customers, but we remain capacity constrained on advanced packaging through the first half of fiscal 2026.\n\nCFO Daniel Kim: For Q1 FY26, we expect revenue between $2.65 billion and $2.78 billion and gross margin near 42%. We expect elevated capital expenditures as we bring our Malaysia line online.\n\nAnalyst Q&A: Management noted that pricing is stable in AI server modules, but consumer power products remain under pressure. The company is watching export controls, foundry cycle times, and customer concentration in cloud accounts.\n`
  },
  {
    id: 'sample-kpi',
    name: 'atlas-components-kpis.csv',
    type: 'text/csv',
    text: `metric,period,value,prior_period,comment\nRevenue,Q4 FY25,$2.84B,$2.41B,18% YoY growth\nGross margin,Q4 FY25,43.2%,40.1%,mix shifted toward data center\nData center revenue,FY25,$3.7B,$2.62B,41% YoY growth\nFree cash flow,Q4 FY25,$286M,$205M,higher operating cash flow\nInventory days,Q4 FY25,68,81,improved sequentially\nQ1 FY26 revenue guide,Q1 FY26,$2.65B-$2.78B,,management guidance\n`
  },
  {
    id: 'sample-filing',
    name: 'Atlas Components 10-K Risk Factors Excerpt.txt',
    type: 'text/plain',
    text: `Risk Factors Excerpt\n\nA limited number of hyperscale customers represented approximately 36% of fiscal 2025 revenue, and loss or delay of a major customer program could materially affect results.\n\nOur advanced packaging supply depends on third-party foundry partners and outsourced assembly vendors. Delays, capacity shortages, quality excursions, or geopolitical disruption may limit our ability to fulfill demand.\n\nWe are subject to export controls and trade restrictions that may reduce our ability to sell certain AI infrastructure products into restricted markets.\n\nCapital expenditures are expected to remain elevated during the Malaysia line ramp, which may pressure free cash flow if demand moderates or qualification schedules slip.\n`
  }
];
