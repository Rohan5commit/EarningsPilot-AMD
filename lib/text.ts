export function normalizeText(text: string) {
  return text.replace(/\r/g, '\n').replace(/[\t ]+/g, ' ').replace(/\n{3,}/g, '\n\n').trim();
}

export function chunkText(text: string, maxChars = 850) {
  const paragraphs = normalizeText(text).split(/\n\s*\n/).filter(Boolean);
  const chunks: string[] = [];
  let current = '';
  for (const paragraph of paragraphs) {
    if ((current + '\n\n' + paragraph).length > maxChars && current) {
      chunks.push(current.trim());
      current = paragraph;
    } else {
      current = current ? `${current}\n\n${paragraph}` : paragraph;
    }
  }
  if (current) chunks.push(current.trim());
  return chunks.length ? chunks : [normalizeText(text).slice(0, maxChars)];
}

export function splitSentences(text: string) {
  return normalizeText(text)
    .split(/(?<=[.!?])\s+(?=[A-Z0-9])/)
    .map((sentence) => sentence.trim())
    .filter((sentence) => sentence.length > 30);
}

export function detectDirection(text: string): 'up' | 'down' | 'flat' | 'unknown' {
  const lower = text.toLowerCase();
  if (/up|grew|growth|expanded|improved|increased|higher|accelerated/.test(lower)) return 'up';
  if (/down|declined|pressure|lower|reduced|moderated|constrained|delay/.test(lower)) return 'down';
  if (/stable|flat|unchanged/.test(lower)) return 'flat';
  return 'unknown';
}

export function truncate(text: string, max = 260) {
  const clean = normalizeText(text);
  return clean.length > max ? `${clean.slice(0, max - 1).trim()}…` : clean;
}
