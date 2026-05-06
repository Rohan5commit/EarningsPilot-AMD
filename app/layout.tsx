import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'EarningsPilot AMD — Agentic Earnings Intelligence',
  description: 'Multi-agent earnings, filings, and KPI intelligence for investors, researchers, and operators, designed for AMD Developer Cloud.'
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
