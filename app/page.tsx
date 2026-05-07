'use client';

import { useMemo, useState } from 'react';
import { AlertTriangle, ArrowUpRight, BarChart3, BrainCircuit, CheckCircle2, FileText, Loader2, Rocket, ShieldAlert, Sparkles, UploadCloud } from 'lucide-react';
import { Badge } from '@/components/Badge';
import { Section } from '@/components/Section';
import type { AnalysisResult, DocumentInput, Evidence } from '@/lib/types';

function evidenceLabel(id: string, evidence: Evidence[]) {
  return evidence.find((item) => item.id === id);
}

export default function Home() {
  const [files, setFiles] = useState<File[]>([]);
  const [manualText, setManualText] = useState('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [selectedEvidence, setSelectedEvidence] = useState<Evidence | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'error'>('idle');
  const [error, setError] = useState('');

  const canAnalyze = files.length > 0 || manualText.trim().length > 20;

  async function analyzeSample() {
    setStatus('loading');
    setError('');
    try {
      const response = await fetch('/api/analyze', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ useSample: true }) });
      const json = await response.json();
      if (!response.ok) throw new Error(json.error || 'Sample analysis failed.');
      setResult(json);
      setSelectedEvidence(json.evidence?.[0] || null);
    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err.message : 'Unexpected error.');
      return;
    }
    setStatus('idle');
  }

  async function analyzeUploads() {
    setStatus('loading');
    setError('');
    try {
      let response: Response;
      if (files.length) {
        const formData = new FormData();
        files.forEach((file) => formData.append('files', file));
        response = await fetch('/api/analyze', { method: 'POST', body: formData });
      } else {
        const documents: DocumentInput[] = [{ id: 'manual', name: 'Pasted analyst material.txt', type: 'text/plain', text: manualText }];
        response = await fetch('/api/analyze', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ documents }) });
      }
      const json = await response.json();
      if (!response.ok) throw new Error(json.error || 'Upload analysis failed.');
      setResult(json);
      setSelectedEvidence(json.evidence?.[0] || null);
    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err.message : 'Unexpected error.');
      return;
    }
    setStatus('idle');
  }

  const riskTone = useMemo(() => {
    if (!result) return 'neutral';
    if (result.tone.label === 'Constructive') return 'green';
    if (result.tone.label === 'Balanced') return 'blue';
    if (result.tone.label === 'Cautious') return 'amber';
    return 'red';
  }, [result]);

  return (
    <main className="relative overflow-hidden px-4 py-6 sm:px-6 lg:px-8">
      <div className="grid-fade pointer-events-none absolute inset-0 opacity-60" />
      <div className="relative mx-auto max-w-7xl">
        <nav className="mb-10 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-emerald-400 text-slate-950 shadow-glow"><BrainCircuit className="h-6 w-6" /></div>
            <div>
              <p className="font-black tracking-tight">EarningsPilot AMD</p>
              <p className="text-xs text-slate-400">Agentic filings intelligence</p>
            </div>
          </div>
          <a href="https://www.amd.com/en/developer/cloud.html" target="_blank" className="hidden items-center gap-2 rounded-full border border-white/10 px-4 py-2 text-sm text-slate-200 hover:border-emerald-300/40 sm:flex">
            AMD Developer Cloud <ArrowUpRight className="h-4 w-4" />
          </a>
        </nav>

        <section className="grid gap-8 lg:grid-cols-[1.05fr_0.95fr] lg:items-center">
          <div>
            <Badge tone="green"><Sparkles className="mr-1 h-3.5 w-3.5" /> Built for AMD Developer Hackathon</Badge>
            <h1 className="mt-6 max-w-4xl text-4xl font-black leading-tight tracking-tight text-white sm:text-6xl">
              Multi-agent earnings intelligence in one judge-friendly workflow.
            </h1>
            <p className="mt-5 max-w-2xl text-lg leading-8 text-slate-300">
              Upload transcripts, SEC excerpts, investor notes, or KPI CSVs. Parser, KPI, Risk, Thesis, and Report agents collaborate to produce a source-grounded analyst memo—without behaving like a generic PDF chatbot.
            </p>
            <div className="mt-7 flex flex-wrap gap-3">
              <button onClick={analyzeSample} disabled={status === 'loading'} className="rounded-2xl bg-emerald-400 px-5 py-3 font-bold text-slate-950 shadow-glow transition hover:bg-emerald-300 disabled:cursor-wait disabled:opacity-70">
                {status === 'loading' ? 'Running agents…' : 'Run instant sample demo'}
              </button>
              <a href="#upload" className="rounded-2xl border border-white/15 px-5 py-3 font-bold text-white hover:border-emerald-300/50">Upload documents</a>
            </div>
          </div>

          <div className="glass rounded-[2rem] p-5">
            <div className="grid gap-3 sm:grid-cols-2">
              {['Parser Agent', 'KPI Extraction Agent', 'Risk Agent', 'Thesis Agent', 'Report Agent'].map((agent, index) => (
                <div key={agent} className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                  <p className="text-xs text-slate-400">0{index + 1}</p>
                  <p className="mt-2 font-bold text-white">{agent}</p>
                  <p className="mt-2 text-sm text-slate-400">{['Chunks and normalizes source material.', 'Finds financial and operating metrics.', 'Flags forward-looking risk language.', 'Builds bull and bear cases.', 'Writes the final action memo.'][index]}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section id="upload" className="mt-10 grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
          <div className="glass rounded-3xl p-6">
            <div className="flex items-center gap-3 text-emerald-300"><UploadCloud className="h-5 w-5" /><p className="text-sm font-bold uppercase tracking-[0.2em]">Upload flow</p></div>
            <h2 className="mt-3 text-2xl font-black">Drop in analyst materials</h2>
            <p className="mt-2 text-slate-400">Use text, CSV, Markdown, HTML, filing excerpts, or the seeded sample package. PDF uploads are accepted with a clear lightweight-build note; exported PDF text works best.</p>
            <input
              type="file"
              multiple
              accept=".txt,.md,.csv,.json,.html,.htm,.pdf,text/*"
              onChange={(event) => setFiles(Array.from(event.target.files || []))}
              className="mt-5 block w-full cursor-pointer rounded-2xl border border-dashed border-slate-500/50 bg-slate-950/50 p-4 text-sm text-slate-300 file:mr-4 file:rounded-xl file:border-0 file:bg-emerald-400 file:px-4 file:py-2 file:font-bold file:text-slate-950"
            />
            {files.length ? <p className="mt-3 text-sm text-slate-300">Selected: {files.map((file) => file.name).join(', ')}</p> : null}
            <textarea
              value={manualText}
              onChange={(event) => setManualText(event.target.value)}
              placeholder="Or paste an earnings-call excerpt, SEC risk factor, investor presentation text, or KPI table here…"
              className="mt-4 h-36 w-full rounded-2xl border-white/10 bg-slate-950/70 text-slate-100 placeholder:text-slate-500 focus:border-emerald-300 focus:ring-emerald-300"
            />
            <button onClick={analyzeUploads} disabled={!canAnalyze || status === 'loading'} className="mt-4 w-full rounded-2xl bg-white px-5 py-3 font-bold text-slate-950 transition hover:bg-emerald-100 disabled:cursor-not-allowed disabled:opacity-40">
              Analyze uploaded material
            </button>
            {status === 'error' ? <div className="mt-4 rounded-2xl border border-rose-400/30 bg-rose-400/10 p-4 text-sm text-rose-100"><AlertTriangle className="mr-2 inline h-4 w-4" />{error}</div> : null}
          </div>

          <div className="glass rounded-3xl p-6">
            <div className="flex items-center gap-3 text-sky-300"><Rocket className="h-5 w-5" /><p className="text-sm font-bold uppercase tracking-[0.2em]">AMD story</p></div>
            <h2 className="mt-3 text-2xl font-black">Designed for AMD GPU-backed open-source inference</h2>
            <p className="mt-2 text-slate-300">Production mode points the Report Agent at an OpenAI-compatible endpoint serving Qwen, Llama, DeepSeek, or Mistral on AMD Developer Cloud. Demo mode remains deterministic so judges can test instantly on Hugging Face Spaces.</p>
            <div className="mt-5 grid gap-3 sm:grid-cols-3">
              <Badge tone="green">ROCm-ready serving</Badge>
              <Badge tone="blue">Structured outputs</Badge>
              <Badge tone="amber">Source citations</Badge>
            </div>
          </div>
        </section>

        {status === 'loading' ? (
          <div className="mt-8 glass flex items-center gap-4 rounded-3xl p-6 text-slate-200"><Loader2 className="h-6 w-6 animate-spin text-emerald-300" /> Agents are parsing, extracting KPIs, triaging risk, debating thesis, and drafting the memo…</div>
        ) : null}

        {result ? (
          <section className="mt-10 space-y-6">
            <div className="glass rounded-3xl p-6">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <p className="text-sm font-bold uppercase tracking-[0.24em] text-emerald-300">Analysis dashboard</p>
                  <h2 className="mt-2 text-3xl font-black">{result.company} {result.ticker !== 'N/A' ? `(${result.ticker})` : ''}</h2>
                  <p className="mt-2 max-w-3xl text-slate-300">{result.companyBrief}</p>
                </div>
                <div className="flex flex-wrap gap-2"><Badge tone={riskTone as never}>Tone: {result.tone.label} · {result.tone.score}/100</Badge><Badge tone="blue">Mode: {result.modelMode}</Badge><Badge tone="green">GPU budget: {result.amdRun.gpuHoursBudget} MI300X hrs</Badge></div>
              </div>
            </div>

            <div className="grid gap-6 lg:grid-cols-3">
              <Section title="AMD acceleration" eyebrow="MI300X model path">
                <div className="space-y-3 text-sm text-slate-300">
                  <p><span className="font-bold text-white">GPU:</span> {result.amdRun.gpu}</p>
                  <p><span className="font-bold text-white">Model:</span> {result.amdRun.model}</p>
                  <p><span className="font-bold text-white">Endpoint:</span> {result.amdRun.usedModelEndpoint ? 'AMD model endpoint used' : result.amdRun.endpointConfigured ? 'Configured, fallback used' : 'Not configured in public demo'}</p>
                  <p><span className="font-bold text-white">Latency:</span> {result.amdRun.latencyMs ? `${result.amdRun.latencyMs} ms` : 'fallback mode'}</p>
                  <p className="rounded-2xl border border-emerald-400/20 bg-emerald-400/5 p-3 text-emerald-100">{result.amdRun.note}</p>
                </div>
              </Section>
              <Section title="Executive summary" eyebrow="Report Agent">
                <ul className="space-y-3 text-slate-300">{result.executiveSummary.map((item) => <li key={item} className="flex gap-3"><CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-300" />{item}</li>)}</ul>
              </Section>
              <Section title="Action memo" eyebrow="Decision support"><p className="leading-8 text-slate-200">{result.actionMemo}</p></Section>
            </div>

            <Section title="KPI extraction table" eyebrow="KPI Extraction Agent">
              <div className="overflow-x-auto">
                <table className="w-full min-w-[720px] text-left text-sm">
                  <thead className="text-slate-400"><tr><th className="pb-3">Metric</th><th className="pb-3">Value</th><th className="pb-3">Period</th><th className="pb-3">Trend</th><th className="pb-3">Evidence</th></tr></thead>
                  <tbody className="divide-y divide-white/10">{result.kpis.map((kpi) => <tr key={`${kpi.metric}-${kpi.value}`}><td className="py-3 font-semibold text-white">{kpi.metric}</td><td className="py-3 text-emerald-200">{kpi.value}</td><td className="py-3 text-slate-300">{kpi.period}</td><td className="py-3 text-slate-300">{kpi.direction}</td><td className="py-3"><button onClick={() => setSelectedEvidence(evidenceLabel(kpi.evidenceId, result.evidence) || null)} className="rounded-full bg-slate-800 px-3 py-1 text-xs text-sky-200 hover:bg-slate-700">{kpi.evidenceId}</button></td></tr>)}</tbody>
                </table>
              </div>
            </Section>

            <div className="grid gap-6 lg:grid-cols-2">
              <Section title="Bull case" eyebrow="Thesis Agent"><ul className="space-y-3">{result.bullCase.map((point) => <li key={point.point} className="rounded-2xl border border-emerald-400/15 bg-emerald-400/5 p-4 text-slate-200">{point.point} <button onClick={() => setSelectedEvidence(evidenceLabel(point.evidenceId, result.evidence) || null)} className="ml-2 text-xs font-bold text-emerald-300">{point.evidenceId}</button></li>)}</ul></Section>
              <Section title="Bear case" eyebrow="Thesis Agent"><ul className="space-y-3">{result.bearCase.map((point) => <li key={point.point} className="rounded-2xl border border-amber-400/15 bg-amber-400/5 p-4 text-slate-200">{point.point} <button onClick={() => setSelectedEvidence(evidenceLabel(point.evidenceId, result.evidence) || null)} className="ml-2 text-xs font-bold text-amber-300">{point.evidenceId}</button></li>)}</ul></Section>
            </div>

            <div className="grid gap-6 lg:grid-cols-[1fr_0.85fr]">
              <Section title="Top risks" eyebrow="Risk Agent"><div className="space-y-3">{result.risks.map((risk) => <button key={risk.title} onClick={() => setSelectedEvidence(evidenceLabel(risk.evidenceId, result.evidence) || null)} className="block w-full rounded-2xl border border-rose-400/15 bg-rose-400/5 p-4 text-left hover:border-rose-300/40"><div className="flex items-center justify-between gap-3"><p className="font-bold text-white"><ShieldAlert className="mr-2 inline h-4 w-4 text-rose-300" />{risk.title}</p><Badge tone={risk.severity === 'High' ? 'red' : risk.severity === 'Medium' ? 'amber' : 'neutral'}>{risk.severity}</Badge></div><p className="mt-2 text-sm text-slate-300">{risk.rationale}</p></button>)}</div></Section>
              <Section title="Evidence viewer" eyebrow="Citations & snippets">
                {selectedEvidence ? <div><Badge tone="blue">{selectedEvidence.id} · {selectedEvidence.agent}</Badge><p className="mt-4 text-sm text-slate-400">{selectedEvidence.document}</p><blockquote className="mt-3 rounded-2xl border-l-4 border-emerald-300 bg-slate-950/60 p-4 leading-7 text-slate-100">{selectedEvidence.snippet}</blockquote><p className="mt-3 text-sm text-slate-400">{selectedEvidence.relevance}</p></div> : <p className="text-slate-400">Select an evidence chip to inspect the supporting source snippet.</p>}
              </Section>
            </div>

            <Section title="Agent trace" eyebrow="Orchestration">
              <div className="grid gap-3 md:grid-cols-5">{result.agentTrace.map((step) => <div key={step.agent} className="rounded-2xl border border-white/10 bg-white/[0.03] p-4"><BarChart3 className="mb-3 h-5 w-5 text-emerald-300" /><p className="font-bold text-white">{step.agent}</p><p className="mt-2 text-xs text-slate-400">{step.objective}</p><p className="mt-3 text-sm text-slate-300">{step.output}</p></div>)}</div>
            </Section>
          </section>
        ) : (
          <div className="mt-10 glass rounded-3xl p-8 text-center text-slate-300"><FileText className="mx-auto mb-3 h-10 w-10 text-slate-500" />No analysis yet. Run the instant sample demo for a complete seeded workflow.</div>
        )}
      </div>
    </main>
  );
}
