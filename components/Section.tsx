export function Section({ title, eyebrow, children }: { title: string; eyebrow?: string; children: React.ReactNode }) {
  return (
    <section className="glass rounded-3xl p-5 sm:p-7">
      {eyebrow ? <p className="mb-2 text-xs font-bold uppercase tracking-[0.24em] text-emerald-300">{eyebrow}</p> : null}
      <h2 className="mb-5 text-xl font-bold text-white sm:text-2xl">{title}</h2>
      {children}
    </section>
  );
}
