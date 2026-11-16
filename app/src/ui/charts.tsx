import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

/* Small SVG chart kit. Every chart: thin marks, recessive axes, hover tooltip, colours from CSS tokens. */

export type Num = number | null | undefined;
const ok = (v: Num): v is number => typeof v === "number" && Number.isFinite(v);

export function fmt(v: Num, d = 2): string {
  if (!ok(v)) return "n/a";
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(1);
  return v.toLocaleString("en-US", { maximumFractionDigits: a >= 100 ? 0 : a >= 10 ? 1 : d });
}

function useWidth(): [React.RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement | null>(null);
  const [w, setW] = useState(600);
  useLayoutEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver((e) => setW(Math.max(260, e[0].contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

function ticks(lo: number, hi: number, n = 5): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / n;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

function extent(vals: Num[], pad = 0.06): [number, number] {
  const v = vals.filter(ok);
  if (!v.length) return [0, 1];
  let lo = Math.min(...v), hi = Math.max(...v);
  if (lo === hi) { lo -= 1; hi += 1; }
  const p = (hi - lo) * pad;
  return [lo - p, hi + p];
}

type Tip = { x: number; y: number; body: ReactNode } | null;
const TipBox = ({ tip }: { tip: Tip }) => (tip ? <div className="tip" style={{ left: tip.x, top: tip.y }}>{tip.body}</div> : null);

/* ------------------------------------------------------------------ line */
export type Series = { name: string; color: string; y: Num[]; width?: number; dash?: string; faint?: boolean };
export function LineChart(p: {
  series: Series[]; labels?: string[]; height?: number; yFmt?: (v: number) => string; yLabel?: string;
  bands?: { lo: Num[]; hi: Num[]; color: string }[]; regions?: { from: number; to: number }[];
  refs?: { y: number; label: string; color?: string }[]; yDomain?: [number, number]; xTicks?: number; legend?: boolean;
}) {
  const [ref, W] = useWidth();
  const [tip, setTip] = useState<Tip>(null);
  const [hi, setHi] = useState<number | null>(null);
  const H = p.height ?? 260, m = { l: 46, r: 14, t: 10, b: 24 };
  const n = Math.max(...p.series.map((s) => s.y.length), 1);
  const all: Num[] = [...p.series.flatMap((s) => s.y), ...(p.bands ?? []).flatMap((b) => [...b.lo, ...b.hi]), ...(p.refs ?? []).map((r) => r.y)];
  const [y0, y1] = p.yDomain ?? extent(all);
  const sx = (i: number) => m.l + (i / Math.max(n - 1, 1)) * (W - m.l - m.r);
  const sy = (v: number) => m.t + (1 - (v - y0) / (y1 - y0)) * (H - m.t - m.b);
  const yf = p.yFmt ?? ((v: number) => fmt(v));
  const path = (y: Num[]) => {
    let d = "", pen = false;
    y.forEach((v, i) => { if (ok(v)) { d += `${pen ? "L" : "M"}${sx(i).toFixed(1)},${sy(v).toFixed(1)}`; pen = true; } else pen = false; });
    return d;
  };
  const area = (lo: Num[], hi2: Num[]) => {
    const idx = lo.map((_, i) => i).filter((i) => ok(lo[i]) && ok(hi2[i]));
    if (!idx.length) return "";
    return "M" + idx.map((i) => `${sx(i).toFixed(1)},${sy(hi2[i] as number).toFixed(1)}`).join("L") + "L" +
      idx.reverse().map((i) => `${sx(i).toFixed(1)},${sy(lo[i] as number).toFixed(1)}`).join("L") + "Z";
  };
  const nx = p.xTicks ?? Math.min(6, n);
  const xt = Array.from({ length: nx }, (_, k) => Math.round((k / Math.max(nx - 1, 1)) * (n - 1)));
  const move = (e: React.MouseEvent) => {
    const r = (e.currentTarget as SVGElement).getBoundingClientRect();
    const i = Math.max(0, Math.min(n - 1, Math.round(((e.clientX - r.left - m.l) / (W - m.l - m.r)) * (n - 1))));
    setHi(i);
    const named = p.series.filter((s) => !s.faint && ok(s.y[i]));
    setTip({ x: sx(i), y: m.t + 8, body: (<><b>{p.labels?.[i] ?? `day ${i + 1}`}</b>{named.slice(0, 6).map((s) => (
      <div key={s.name}><span className="dot" style={{ background: s.color }} />{s.name}: {yf(s.y[i] as number)}</div>))}</>) });
  };
  return (
    <div className="chart" ref={ref}>
      <svg height={H} onMouseMove={move} onMouseLeave={() => { setTip(null); setHi(null); }} role="img">
        {(p.regions ?? []).map((r, i) => (<rect key={i} x={sx(r.from)} width={Math.max(1.5, sx(r.to) - sx(r.from))} y={m.t} height={H - m.t - m.b} fill="var(--crisis)" opacity={0.13} />))}
        {ticks(y0, y1).map((t) => (<g key={t}><line x1={m.l} x2={W - m.r} y1={sy(t)} y2={sy(t)} stroke="var(--line)" /><text x={m.l - 6} y={sy(t) + 3.5} textAnchor="end">{yf(t)}</text></g>))}
        {xt.map((i) => (<text key={i} x={sx(i)} y={H - 6} textAnchor={i === 0 ? "start" : i === n - 1 ? "end" : "middle"}>{p.labels?.[i] ?? i + 1}</text>))}
        {(p.bands ?? []).map((b, i) => (<path key={i} d={area(b.lo, b.hi)} fill={b.color} stroke="none" />))}
        {(p.refs ?? []).map((r) => (<g key={r.label}><line x1={m.l} x2={W - m.r} y1={sy(r.y)} y2={sy(r.y)} stroke={r.color ?? "var(--axis)"} strokeDasharray="4 4" /><text x={W - m.r} y={sy(r.y) - 4} textAnchor="end">{r.label}</text></g>))}
        {p.series.map((s, i) => (<path key={s.name + i} d={path(s.y)} fill="none" stroke={s.color} strokeWidth={s.width ?? (s.faint ? 1 : 2)} strokeDasharray={s.dash} opacity={s.faint ? 0.45 : 1} strokeLinejoin="round" />))}
        {hi !== null && <line x1={sx(hi)} x2={sx(hi)} y1={m.t} y2={H - m.b} stroke="var(--axis)" />}
        {p.yLabel && <text x={m.l} y={m.t + 2} className="lbl" dy={-0}>{p.yLabel}</text>}
      </svg>
      <TipBox tip={tip} />
      {p.legend !== false && p.series.filter((s) => !s.faint).length > 1 && (
        <div className="legend">{p.series.filter((s) => !s.faint).map((s) => (<span key={s.name}><i style={{ background: s.color }} />{s.name}</span>))}</div>)}
    </div>
  );
}

/* -------------------------------------------- interval plot (metric strip) */
export type Row = { label: string; color: string; value: Num; sd?: Num; note?: string; bold?: boolean };
export function IntervalPlot(p: { rows: Row[]; band?: { lo: number; hi: number; mid?: number; label?: string }; ceiling?: number;
  xFmt?: (v: number) => string; xLabel?: string; domain?: [number, number]; zero?: boolean }) {
  const [ref, W] = useWidth();
  const [tip, setTip] = useState<Tip>(null);
  const rowH = 26, m = { l: Math.min(170, W * 0.36), r: 54, t: 22, b: 26 };
  const H = m.t + m.b + p.rows.length * rowH;
  const vals: Num[] = [...p.rows.flatMap((r) => [r.value, ok(r.value) && ok(r.sd) ? r.value + r.sd : null, ok(r.value) && ok(r.sd) ? r.value - r.sd : null]),
    p.band?.lo, p.band?.hi, p.ceiling, p.zero ? 0 : null];
  const [x0, x1] = p.domain ?? extent(vals, 0.08);
  const sx = (v: number) => m.l + ((Math.max(x0, Math.min(x1, v)) - x0) / (x1 - x0)) * (W - m.l - m.r);
  const xf = p.xFmt ?? ((v: number) => fmt(v));
  return (
    <div className="chart" ref={ref}>
      <svg height={H} onMouseLeave={() => setTip(null)} role="img">
        {p.band && <rect x={sx(p.band.lo)} width={Math.max(2, sx(p.band.hi) - sx(p.band.lo))} y={m.t - 6} height={H - m.t - m.b + 8} fill="var(--band)" rx={4} />}
        {p.band && <text x={(sx(p.band.lo) + sx(p.band.hi)) / 2} y={11} textAnchor="middle" className="lbl">{p.band.label ?? "real data, 95% band"}</text>}
        {p.band?.mid !== undefined && <line x1={sx(p.band.mid)} x2={sx(p.band.mid)} y1={m.t - 6} y2={H - m.b + 2} stroke="var(--accent)" strokeWidth={1.5} />}
        {p.ceiling !== undefined && <><rect x={sx(x0)} width={Math.max(2, sx(p.ceiling) - sx(x0))} y={m.t - 6} height={H - m.t - m.b + 8} fill="var(--band)" rx={4} />
          <text x={sx(p.ceiling)} y={11} textAnchor="end" className="lbl">within real-vs-real noise</text></>}
        {ticks(x0, x1, 5).map((t) => (<g key={t}><line x1={sx(t)} x2={sx(t)} y1={H - m.b} y2={H - m.b + 4} stroke="var(--axis)" /><text x={sx(t)} y={H - 8} textAnchor="middle">{xf(t)}</text></g>))}
        {p.rows.map((r, i) => {
          const y = m.t + i * rowH + rowH / 2;
          const has = ok(r.value);
          return (
            <g key={r.label} onMouseEnter={() => setTip({ x: has ? sx(r.value as number) : m.l + 30, y, body: (<><b>{r.label}</b><div>{has ? xf(r.value as number) : "not produced"}{ok(r.sd) && r.sd ? ` ± ${xf(r.sd)} (seeds)` : ""}</div>{r.note && <div>{r.note}</div>}</>) })}>
              <rect x={0} y={y - rowH / 2} width={W} height={rowH} fill="transparent" />
              <line x1={m.l} x2={W - m.r} y1={y} y2={y} stroke="var(--line)" />
              <text x={m.l - 10} y={y + 4} textAnchor="end" className="lbl" style={{ fontWeight: r.bold ? 750 : 500, fill: r.bold ? "var(--ink)" : undefined }}>{r.label}</text>
              {has && ok(r.sd) && r.sd > 0 && <line x1={sx((r.value as number) - r.sd)} x2={sx((r.value as number) + r.sd)} y1={y} y2={y} stroke={r.color} strokeWidth={2} />}
              {has ? <circle cx={sx(r.value as number)} cy={y} r={r.bold ? 6 : 5} fill={r.color} stroke="var(--surface)" strokeWidth={2} />
                : <text x={m.l + 6} y={y + 4}>no crisis windows produced</text>}
              {has && <text x={W - m.r + 6} y={y + 4} className="num">{xf(r.value as number)}</text>}
            </g>
          );
        })}
        {p.xLabel && <text x={W - m.r} y={H - 8} textAnchor="end" />}
      </svg>
      <TipBox tip={tip} />
    </div>
  );
}

/* ---------------------------------------------------------------- scatter */
export type Pt = { x: Num; y: Num; label: string; color: string; bold?: boolean; note?: string };
export function Scatter(p: { points: Pt[]; xLabel: string; yLabel: string; height?: number; xBand?: [number, number]; yBand?: [number, number];
  xFmt?: (v: number) => string; yFmt?: (v: number) => string; xLog?: boolean }) {
  const [ref, W] = useWidth();
  const [tip, setTip] = useState<Tip>(null);
  const H = p.height ?? 360, m = { l: 52, r: 20, t: 14, b: 40 };
  const tx = (v: number) => (p.xLog ? Math.log10(Math.max(v, 1e-6)) : v);
  const pts = p.points.filter((q) => ok(q.x) && ok(q.y));
  const [x0, x1] = extent([...pts.map((q) => tx(q.x as number)), ...(p.xBand ?? []).map(tx)], 0.1);
  const [y0, y1] = extent([...pts.map((q) => q.y), ...(p.yBand ?? [])], 0.1);
  const sx = (v: number) => m.l + ((tx(v) - x0) / (x1 - x0)) * (W - m.l - m.r);
  const sy = (v: number) => m.t + (1 - (v - y0) / (y1 - y0)) * (H - m.t - m.b);
  const xf = p.xFmt ?? ((v: number) => fmt(v)), yf = p.yFmt ?? ((v: number) => fmt(v));
  const xt = p.xLog ? ticks(x0, x1, 4).map((t) => Math.pow(10, t)) : ticks(x0, x1, 5);
  return (
    <div className="chart" ref={ref}>
      <svg height={H} onMouseLeave={() => setTip(null)} role="img">
        {p.yBand && <rect x={m.l} width={W - m.l - m.r} y={sy(p.yBand[1])} height={Math.max(2, sy(p.yBand[0]) - sy(p.yBand[1]))} fill="var(--band)" />}
        {p.xBand && <rect x={sx(p.xBand[0])} width={Math.max(2, sx(p.xBand[1]) - sx(p.xBand[0]))} y={m.t} height={H - m.t - m.b} fill="var(--band)" />}
        {ticks(y0, y1).map((t) => (<g key={t}><line x1={m.l} x2={W - m.r} y1={sy(t)} y2={sy(t)} stroke="var(--line)" /><text x={m.l - 6} y={sy(t) + 3.5} textAnchor="end">{yf(t)}</text></g>))}
        {xt.map((t) => (<text key={t} x={sx(t)} y={H - m.b + 15} textAnchor="middle">{xf(t)}</text>))}
        <text x={(m.l + W - m.r) / 2} y={H - 6} textAnchor="middle" className="lbl">{p.xLabel}</text>
        <text transform={`translate(13 ${(m.t + H - m.b) / 2}) rotate(-90)`} textAnchor="middle" className="lbl">{p.yLabel}</text>
        {pts.map((q) => (
          <g key={q.label} onMouseEnter={() => setTip({ x: sx(q.x as number), y: sy(q.y as number), body: (<><b>{q.label}</b><div>{p.xLabel}: {xf(q.x as number)}</div><div>{p.yLabel}: {yf(q.y as number)}</div>{q.note && <div>{q.note}</div>}</>) })}>
            <circle cx={sx(q.x as number)} cy={sy(q.y as number)} r={14} fill="transparent" />
            <circle cx={sx(q.x as number)} cy={sy(q.y as number)} r={q.bold ? 7 : 5.5} fill={q.color} stroke="var(--surface)" strokeWidth={2} />
            <text x={sx(q.x as number) + 9} y={sy(q.y as number) + 4} className="lbl" style={{ fontWeight: q.bold ? 750 : 500 }}>{q.label}</text>
          </g>))}
      </svg>
      <TipBox tip={tip} />
    </div>
  );
}

/* ---------------------------------------------------------------- heatmap */
const lerp = (a: number[], b: number[], t: number) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
const rgb = (c: number[]) => `rgb(${c[0]},${c[1]},${c[2]})`;
export function heatColor(v: number, lo: number, hi: number, mode: "seq" | "div", dark: boolean): string {
  if (mode === "div") {
    const mid = dark ? [56, 56, 53] : [240, 239, 236];
    const t = Math.max(-1, Math.min(1, v / Math.max(Math.abs(lo), Math.abs(hi), 1e-9)));
    return rgb(t >= 0 ? lerp(mid, [208, 59, 59], t) : lerp(mid, [42, 120, 214], -t));
  }
  const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo || 1)));
  return rgb(dark ? lerp([26, 38, 56], [109, 167, 236], t) : lerp([236, 243, 252], [13, 54, 107], t));
}
export function Heatmap(p: { matrix: Num[][]; rowLabels?: string[]; colLabels?: string[]; lo: number; hi: number; mode?: "seq" | "div";
  size?: number; showValues?: boolean; groups?: string[]; fmtV?: (v: number) => string }) {
  const [ref, W] = useWidth();
  const cv = useRef<HTMLCanvasElement | null>(null);
  const [tip, setTip] = useState<Tip>(null);
  const n = p.matrix.length, k = p.matrix[0]?.length ?? 0;
  const labelW = p.showValues ? Math.min(150, W * 0.3) : 0, top = p.showValues ? 74 : 0;
  const side = Math.min(p.size ?? 520, W) - labelW;
  const cell = side / Math.max(k, 1);
  const dark = typeof document !== "undefined" && document.documentElement.dataset.theme === "dark";
  const f = p.fmtV ?? ((v: number) => v.toFixed(2));
  useEffect(() => {
    const c = cv.current; if (!c || p.showValues) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = side * dpr; c.height = cell * n * dpr;
    const g = c.getContext("2d"); if (!g) return;
    g.scale(dpr, dpr);
    for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) {
      const v = p.matrix[i][j];
      g.fillStyle = ok(v) ? heatColor(v, p.lo, p.hi, p.mode ?? "seq", dark) : "transparent";
      g.fillRect(j * cell, i * cell, cell + 0.5, cell + 0.5);
    }
    if (p.groups) { g.strokeStyle = dark ? "#0d0d0d" : "#fcfcfb"; g.lineWidth = 2;
      for (let i = 1; i < n; i++) if (p.groups[i] !== p.groups[i - 1]) { g.beginPath(); g.moveTo(0, i * cell); g.lineTo(side, i * cell); g.moveTo(i * cell, 0); g.lineTo(i * cell, cell * n); g.stroke(); } }
  }, [p.matrix, side, cell, n, k, dark, p.lo, p.hi, p.mode, p.groups, p.showValues]);
  const hover = (e: React.MouseEvent) => {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const j = Math.floor((e.clientX - r.left) / cell), i = Math.floor((e.clientY - r.top) / cell);
    const v = p.matrix[i]?.[j];
    if (!ok(v)) return setTip(null);
    setTip({ x: e.clientX - r.left + labelW, y: e.clientY - r.top + top, body: (<><b>{p.rowLabels?.[i]} × {p.colLabels?.[j]}</b><div>{f(v)}</div></>) });
  };
  if (p.showValues) {
    return (
      <div className="chart" ref={ref}>
        <svg height={top + cell * n + 4} role="img">
          {p.colLabels?.map((l, j) => (<text key={l} transform={`translate(${labelW + j * cell + cell / 2} ${top - 6}) rotate(-45)`} className="lbl">{l}</text>))}
          {p.matrix.map((row, i) => (<g key={i}>
            <text x={labelW - 8} y={top + i * cell + cell / 2 + 4} textAnchor="end" className="lbl">{p.rowLabels?.[i]}</text>
            {row.map((v, j) => ok(v) ? (<g key={j}><rect x={labelW + j * cell + 1} y={top + i * cell + 1} width={cell - 2} height={cell - 2} rx={3} fill={heatColor(v, p.lo, p.hi, p.mode ?? "seq", dark)}><title>{`${p.rowLabels?.[i]} × ${p.colLabels?.[j]}: ${f(v)}`}</title></rect>
              {cell > 26 && <text x={labelW + j * cell + cell / 2} y={top + i * cell + cell / 2 + 3.5} textAnchor="middle" style={{ fill: Math.abs(v) > 0.55 * Math.max(Math.abs(p.lo), Math.abs(p.hi)) ? "#fff" : "var(--ink)", fontSize: 10 }}>{f(v)}</text>}</g>) : null)}
          </g>))}
        </svg>
      </div>
    );
  }
  return (
    <div className="chart" ref={ref}>
      <canvas ref={cv} style={{ width: side, height: cell * n, borderRadius: 8, display: "block" }} onMouseMove={hover} onMouseLeave={() => setTip(null)} />
      <TipBox tip={tip} />
    </div>
  );
}

/* ---------------------------------------------------------------- columns */
export function Columns(p: { values: number[]; values2?: number[]; labels?: string[]; height?: number; color: string; color2?: string; names?: [string, string] }) {
  const [ref, W] = useWidth();
  const H = p.height ?? 160, m = { l: 8, r: 8, t: 8, b: 20 };
  const norm = (a: number[]) => { const s = a.reduce((x, y) => x + y, 0) || 1; return a.map((v) => v / s); };
  const a = useMemo(() => norm(p.values), [p.values]);
  const b = useMemo(() => (p.values2 ? norm(p.values2) : null), [p.values2]);
  const mx = Math.max(...a, ...(b ?? [0]), 1e-9);
  const bw = (W - m.l - m.r) / a.length;
  const sy = (v: number) => (v / mx) * (H - m.t - m.b);
  const line = b ? b.map((v, i) => `${i ? "L" : "M"}${(m.l + i * bw + bw / 2).toFixed(1)},${(H - m.b - sy(v)).toFixed(1)}`).join("") : "";
  return (
    <div className="chart" ref={ref}>
      <svg height={H} role="img">
        {a.map((v, i) => (<rect key={i} x={m.l + i * bw + 1} width={Math.max(1, bw - 2)} y={H - m.b - sy(v)} height={sy(v)} fill={p.color} rx={2} />))}
        {b && <path d={line} fill="none" stroke={p.color2 ?? "var(--real)"} strokeWidth={2} />}
        <line x1={m.l} x2={W - m.r} y1={H - m.b} y2={H - m.b} stroke="var(--axis)" />
        {p.labels?.map((l, i) => (l ? <text key={i} x={m.l + i * bw + bw / 2} y={H - 5} textAnchor="middle">{l}</text> : null))}
      </svg>
      {p.names && <div className="legend"><span><i style={{ background: p.color }} />{p.names[0]}</span><span><i style={{ background: p.color2 ?? "var(--real)" }} />{p.names[1]}</span></div>}
    </div>
  );
}
