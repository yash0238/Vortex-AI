import type { ReactNode } from "react";
import { G } from "../glossary";
import { fmt } from "./charts";
import type { Any, BenchRow, RefMetric } from "../lib/data";
import { famColor } from "../lib/data";

export const Term = ({ k, children }: { k: string; children?: ReactNode }) => (
  <span className="term" tabIndex={0} data-tip={G[k]?.short ?? ""}>{children ?? G[k]?.name ?? k}</span>
);
export const Stat = ({ k, v, s }: { k: ReactNode; v: ReactNode; s?: ReactNode }) => (
  <div className="stat"><span className="k">{k}</span><span className="v num">{v}</span>{s && <span className="s">{s}</span>}</div>
);
export const Card = ({ title, sub, right, children, flat }: { title?: ReactNode; sub?: ReactNode; right?: ReactNode; children?: ReactNode; flat?: boolean }) => (
  <section className={"card" + (flat ? " flat" : "")}>
    {(title || right) && <div className="card-head"><div>{title && <h3>{title}</h3>}{sub && <p>{sub}</p>}</div>{right}</div>}
    {children}
  </section>
);
export const Head = ({ eyebrow, title, children }: { eyebrow: string; title: string; children?: ReactNode }) => (
  <header><div className="eyebrow">{eyebrow}</div><h2 style={{ fontSize: "2rem", marginTop: 6 }}>{title}</h2>{children && <p className="lede">{children}</p>}</header>
);
export const Empty = ({ what }: { what: string }) => (<div className="empty">{what} has not been run yet. This section fills in automatically after the experiment finishes and <code>python -m crisisbench.export_web</code> is run.</div>);
export const Loading = ({ error }: { error?: string | null }) => (<div className="empty">{error ? `Could not load results: ${error}. Run python -m crisisbench.export_web first.` : "Loading results…"}</div>);
export function Seg<T extends string>({ value, set, options }: { value: T; set: (v: T) => void; options: { v: T; label: string }[] }) {
  return (<div className="seg" role="group">{options.map((o) => (<button key={o.v} aria-pressed={o.v === value} onClick={() => set(o.v)}>{o.label}</button>))}</div>);
}
export const Verdict = ({ v }: { v?: string }) => {
  const s = (v ?? "not run").toLowerCase();
  const cls = s.startsWith("pass") || s.startsWith("supported") ? "pass" : s.startsWith("fail") || s.startsWith("not supported") ? "fail" : s.includes("inconclusive") ? "warn" : "";
  return <span className={"pill " + cls}>{v ?? "not run"}</span>;
};

/* Generic table over an array of records. */
export function Records({ rows, cols, first }: { rows: Any[]; cols?: { k: string; label?: string; d?: number }[]; first?: string }) {
  if (!rows?.length) return null;
  const c: { k: string; label?: string; d?: number }[] = cols ?? Object.keys(rows[0]).map((k) => ({ k, label: k }));
  return (
    <div className="tbl-wrap"><table className="tbl"><thead><tr>{c.map((x) => (<th key={x.k}>{x.label ?? x.k}</th>))}</tr></thead>
      <tbody>{rows.map((r, i) => (<tr key={i} className={r[first ?? c[0].k]?.toString().includes("CG-NSDE (ours)") ? "ours" : ""}>{c.map((x) => {
        const v = r[x.k];
        return <td key={x.k}>{typeof v === "number" ? fmt(v, x.d ?? 3) : v === null || v === undefined ? "–" : String(v)}</td>;
      })}</tr>))}</tbody></table></div>
  );
}

/* Benchmark table: one row per generator, cells green when inside the real band. */
export function BenchTable({ rows, refm, metrics }: { rows: BenchRow[]; refm: Record<string, RefMetric>; metrics: string[] }) {
  const cell = (r: BenchRow, m: string) => {
    const v = r[m], sd = r[m + "_sd"], inside = (r[m + "_inside"] ?? 0) >= 0.5;
    if (v === null || v === undefined) return <td key={m} className="out" title="too few crisis windows were produced for this metric to exist">n/p</td>;
    return <td key={m} className={inside ? "in" : "out"}>{fmt(v)}{sd ? <small> ±{fmt(sd, 2)}</small> : null}</td>;
  };
  return (
    <div className="tbl-wrap"><table className="tbl">
      <thead><tr><th>Model</th>{metrics.map((m) => (<th key={m}><span className="term" data-tip={G[m]?.short ?? ""}>{G[m]?.name ?? m}</span></th>))}</tr></thead>
      <tbody>
        <tr className="real"><td>Real data (95% band)</td>{metrics.map((m) => { const x = refm[m]; if (!x) return <td key={m}>–</td>;
          return <td key={m}>{x.kind === "dist" ? `≤ ${fmt(x.ceiling)}` : <>{fmt(x.real)} <small>[{fmt(x.lo)}, {fmt(x.hi)}]</small></>}</td>; })}</tr>
        {rows.map((r) => (<tr key={r.generator} className={r.family_tag === "neural-ours" ? "ours" : ""}>
          <td><span className="dot" style={{ background: famColor(r.family_tag) }} />{r.name}</td>{metrics.map((m) => cell(r, m))}</tr>))}
      </tbody></table></div>
  );
}
