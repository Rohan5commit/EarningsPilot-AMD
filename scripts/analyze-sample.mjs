import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const root = process.cwd();
const docs = [
  ['transcript', 'atlas-components-earnings-transcript.txt', 'text/plain'],
  ['kpis', 'atlas-components-kpis.csv', 'text/csv'],
  ['risks', 'atlas-components-risk-factors.txt', 'text/plain']
].map(([id, name, type]) => ({ id, name, type, text: readFileSync(join(root, 'sample-data', name), 'utf8') }));

const response = await fetch('http://localhost:3000/api/analyze', {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ documents: docs })
});

const json = await response.json();
if (!response.ok) {
  console.error(json);
  process.exit(1);
}
console.log(JSON.stringify({ company: json.company, kpis: json.kpis.length, risks: json.risks.length, mode: json.modelMode }, null, 2));
