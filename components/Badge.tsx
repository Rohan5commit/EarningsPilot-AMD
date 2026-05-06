import { clsx } from 'clsx';

export function Badge({ children, tone = 'neutral' }: { children: React.ReactNode; tone?: 'green' | 'blue' | 'amber' | 'red' | 'neutral' }) {
  return (
    <span
      className={clsx(
        'inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold',
        tone === 'green' && 'border-emerald-400/30 bg-emerald-400/10 text-emerald-200',
        tone === 'blue' && 'border-sky-400/30 bg-sky-400/10 text-sky-200',
        tone === 'amber' && 'border-amber-400/30 bg-amber-400/10 text-amber-200',
        tone === 'red' && 'border-rose-400/30 bg-rose-400/10 text-rose-200',
        tone === 'neutral' && 'border-slate-500/30 bg-slate-500/10 text-slate-200'
      )}
    >
      {children}
    </span>
  );
}
