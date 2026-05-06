import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import { runEarningsPilot } from '@/lib/agentPipeline';
import { sampleDocuments } from '@/lib/sample';
import type { DocumentInput } from '@/lib/types';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

const jsonSchema = z.object({
  useSample: z.boolean().optional(),
  documents: z.array(z.object({ id: z.string(), name: z.string(), type: z.string(), text: z.string() })).optional()
});

async function docsFromFormData(request: NextRequest): Promise<DocumentInput[]> {
  const data = await request.formData();
  const files = data.getAll('files').filter((item): item is File => item instanceof File);
  const docs: DocumentInput[] = [];
  for (const file of files) {
    const buffer = Buffer.from(await file.arrayBuffer());
    const isPdf = file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');
    const text = isPdf
      ? `PDF upload detected: ${file.name}. For the live hackathon demo, export PDF text or use the sample dataset. Binary PDF bytes were received (${buffer.length} bytes), but this lightweight Space build keeps dependencies minimal for reliable deployment.`
      : buffer.toString('utf8');
    docs.push({ id: crypto.randomUUID(), name: file.name, type: file.type || 'application/octet-stream', text });
  }
  return docs;
}

export async function POST(request: NextRequest) {
  try {
    const contentType = request.headers.get('content-type') || '';
    let documents: DocumentInput[] = [];
    if (contentType.includes('multipart/form-data')) {
      documents = await docsFromFormData(request);
    } else {
      const body = jsonSchema.parse(await request.json());
      documents = body.useSample ? sampleDocuments : body.documents || [];
    }
    const result = await runEarningsPilot(documents);
    return NextResponse.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Analysis failed.';
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
