export type DocumentInput = {
  id: string;
  name: string;
  type: string;
  text: string;
};

export type Evidence = {
  id: string;
  document: string;
  snippet: string;
  relevance: string;
  agent: 'Parser Agent' | 'KPI Extraction Agent' | 'Risk Agent' | 'Thesis Agent' | 'Report Agent';
};

export type Kpi = {
  metric: string;
  value: string;
  period: string;
  direction: 'up' | 'down' | 'flat' | 'unknown';
  evidenceId: string;
};

export type Risk = {
  title: string;
  severity: 'Low' | 'Medium' | 'High';
  rationale: string;
  evidenceId: string;
};

export type ThesisPoint = {
  point: string;
  evidenceId: string;
};

export type AgentStep = {
  agent: string;
  objective: string;
  output: string;
};

export type AnalysisResult = {
  company: string;
  ticker: string;
  generatedAt: string;
  modelMode: 'deterministic-local' | 'amd-openai-compatible';
  companyBrief: string;
  executiveSummary: string[];
  kpis: Kpi[];
  bullCase: ThesisPoint[];
  bearCase: ThesisPoint[];
  risks: Risk[];
  tone: {
    label: 'Constructive' | 'Balanced' | 'Cautious' | 'Defensive';
    score: number;
    rationale: string;
    evidenceId: string;
  };
  actionMemo: string;
  evidence: Evidence[];
  agentTrace: AgentStep[];
};
