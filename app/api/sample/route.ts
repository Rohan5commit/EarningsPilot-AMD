import { NextResponse } from 'next/server';
import { sampleDocuments } from '@/lib/sample';

export const dynamic = 'force-dynamic';

export async function GET() {
  return NextResponse.json({ documents: sampleDocuments });
}
