import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";

// ------------------------------------------------------------ animated number
export function useCountUp(target: number, ms = 900) {
  const [v, setV] = useState(target);
  const from = useRef(target);
  useEffect(() => {
    const a = from.current, t0 = performance.now();
    let raf = 0;
    const tick = (t: number) => {
      const f = Math.min(1, (t - t0) / ms), e = 1 - Math.pow(1 - f, 3);
      setV(a + (target - a) * e);
      if (f < 1) raf = requestAnimationFrame(tick); else from.current = target;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return v;
}

export function Count({ value, fmt }: { value: number; fmt: (v: number) => string }) {
  const v = useCountUp(value);
  return <>{fmt(v)}</>;
}

// ------------------------------------------------------------ progress ring
export function Ring({ value, size = 168, stroke = 14, children, color = "url(#ringgrad)" }:
  { value: number; size?: number; stroke?: number; children?: ReactNode; color?: string }) {
  const r = (size - stroke) / 2, c = 2 * Math.PI * r;
  const [shown, setShown] = useState(0);
  useEffect(() => { const t = setTimeout(() => setShown(Math.max(0, Math.min(1, value))), 60); return () => clearTimeout(t); }, [value]);
  return (
    <div className="ring-wrap" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <defs>
          <linearGradient id="ringgrad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#c6f432" /><stop offset="100%" stopColor="#8fd14f" />
          </linearGradient>
        </defs>
        <circle cx={size / 2} cy={size / 2} r={r} stroke="#1b1e23" strokeWidth={stroke} fill="none" />
        <circle cx={size / 2} cy={size / 2} r={r} stroke={color} strokeWidth={stroke} fill="none" strokeLinecap="round"
          strokeDasharray={c} strokeDashoffset={c * (1 - shown)} transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ transition: "stroke-dashoffset 1.2s cubic-bezier(.2,.8,.2,1)" }} />
      </svg>
      <div className="center">{children}</div>
    </div>
  );
}

// ------------------------------------------------------------ SVG charts
type Series = { values: number[]; color: string; width?: number; dash?: string; label?: string };
type Band = { lo: number[]; hi: number[]; color: string; opacity?: number };

// monotone cubic (Fritsch-Carlson): smooth without overshooting the data
function smooth(pts: [number, number][], move = true): string {
  const n = pts.length;
  if (n < 2) return "";
  const dx = pts.slice(1).map((p, i) => p[0] - pts[i][0]);
  const m = pts.slice(1).map((p, i) => (p[1] - pts[i][1]) / (dx[i] || 1));
  const t = pts.map((_, i) => (i === 0 ? m[0] : i === n - 1 ? m[n - 2] : m[i - 1] * m[i] <= 0 ? 0 : (m[i - 1] + m[i]) / 2));
  for (let i = 0; i < n - 1; i++) {
    if (m[i] === 0) { t[i] = 0; t[i + 1] = 0; continue; }
    const a = t[i] / m[i], b = t[i + 1] / m[i], h = a * a + b * b;
    if (h > 9) { const k = 3 / Math.sqrt(h); t[i] = k * a * m[i]; t[i + 1] = k * b * m[i]; }
  }
  let d = `${move ? "M" : "L"}${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
  for (let i = 0; i < n - 1; i++) {
    const h = dx[i] / 3;
    d += `C${(pts[i][0] + h).toFixed(1)},${(pts[i][1] + h * t[i]).toFixed(1)} ${(pts[i + 1][0] - h).toFixed(1)},${(pts[i + 1][1] - h * t[i + 1]).toFixed(1)} ${pts[i + 1][0].toFixed(1)},${pts[i + 1][1].toFixed(1)}`;
  }
  return d;
}

export function AreaChart({ series, bands = [], height = 240, xLabels, fmt, hline, hlabel, fillUnder }:
  { series: Series[]; bands?: Band[]; height?: number; xLabels: string[]; fmt: (v: number) => string; hline?: number; hlabel?: string; fillUnder?: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(600);
  const [hover, setHover] = useState<number | null>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    setW(el.clientWidth);
    return () => ro.disconnect();
  }, []);
  const n = xLabels.length, padL = 4, padR = 58, padT = 16, padB = 26;
  const all = [...series.flatMap((s) => s.values), ...bands.flatMap((b) => [...b.lo, ...b.hi]), ...(hline !== undefined ? [hline] : [])];
  const lo = Math.min(...all), hi = Math.max(...all);
  const y0 = lo - (hi - lo) * 0.08, y1 = hi + (hi - lo) * 0.08;
  const X = (i: number) => padL + (i / Math.max(n - 1, 1)) * (w - padL - padR);
  const Y = (v: number) => padT + (1 - (v - y0) / Math.max(y1 - y0, 1e-9)) * (height - padT - padB);
  const pts = (vs: number[]) => vs.map((v, i) => [X(i), Y(v)] as [number, number]);
  const area = (top: number[], bot: number[]) => `${smooth(pts(top))}${smooth(pts(bot).reverse(), false)}Z`;
  const onMove = (e: React.PointerEvent) => {
    const r = (e.currentTarget as SVGElement).getBoundingClientRect();
    const i = Math.round(((e.clientX - r.left - padL) / Math.max(w - padL - padR, 1)) * (n - 1));
    setHover(Math.max(0, Math.min(n - 1, i)));
  };
  const xt = useMemo(() => { const k = Math.max(2, Math.min(5, n, Math.floor(w / 120))); return Array.from({ length: k }, (_, j) => Math.round((j / Math.max(k - 1, 1)) * (n - 1))); }, [n, w]);
  const yt = [0, 0.5, 1].map((f) => y0 + (y1 - y0) * (0.1 + 0.8 * f));
  const main = series[series.length - 1];
  return (
    <div ref={ref} style={{ position: "relative", width: "100%" }}>
      <svg width={w} height={height} onPointerMove={onMove} onPointerLeave={() => setHover(null)} style={{ display: "block", touchAction: "pan-y" }}>
        <defs>
          {bands.map((b, i) => (
            <linearGradient key={i} id={`bg${i}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={b.color} stopOpacity={b.opacity ?? 0.28} /><stop offset="100%" stopColor={b.color} stopOpacity={(b.opacity ?? 0.28) * 0.4} />
            </linearGradient>
          ))}
          <linearGradient id="under" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={main?.color ?? "#74a214"} stopOpacity={0.28} /><stop offset="100%" stopColor={main?.color ?? "#74a214"} stopOpacity={0} />
          </linearGradient>
        </defs>
        {yt.map((v, i) => (
          <g key={i}>
            <line x1={padL} x2={w - padR} y1={Y(v)} y2={Y(v)} stroke="rgba(255,255,255,.05)" />
            <text x={w - padR + 10} y={Y(v) + 4} fill="#6b707b" fontSize="11">{fmt(v)}</text>
          </g>
        ))}
        {bands.map((b, i) => <path key={i} d={area(b.hi, b.lo)} fill={`url(#bg${i})`} />)}
        {fillUnder && main && <path d={`${smooth(pts(main.values))}L${X(n - 1)},${height - padB}L${X(0)},${height - padB}Z`} fill="url(#under)" />}
        {hline !== undefined && (
          <g>
            <line x1={padL} x2={w - padR} y1={Y(hline)} y2={Y(hline)} stroke="#c47c0c" strokeDasharray="4 4" strokeWidth={1.5} />
            {hlabel && <text x={w - padR - 6} y={Y(hline) - 7} textAnchor="end" fill="#a4a9b3" fontSize="11.5" fontWeight="600">{hlabel}</text>}
          </g>
        )}
        {series.map((s, i) => (
          <path key={i} d={smooth(pts(s.values))} fill="none" stroke={s.color} strokeWidth={s.width ?? 2} strokeDasharray={s.dash}
            strokeLinejoin="round" strokeLinecap="round"
            style={s.dash ? undefined : ({ strokeDasharray: 3000, ["--len" as string]: 3000, animation: "draw 1.2s ease-out both" } as React.CSSProperties)} />
        ))}
        {xt.map((i) => (
          <text key={i} x={X(i)} y={height - 7} textAnchor={i === 0 ? "start" : i === n - 1 ? "end" : "middle"} fill="#6b707b" fontSize="11">{xLabels[i]}</text>
        ))}
        {hover !== null && (
          <g>
            <line x1={X(hover)} x2={X(hover)} y1={padT} y2={height - padB} stroke="rgba(255,255,255,.18)" />
            {series.map((s, i) => <circle key={i} cx={X(hover)} cy={Y(s.values[hover])} r={5} fill={s.color} stroke="#0f1013" strokeWidth={2} />)}
          </g>
        )}
      </svg>
      {hover !== null && (
        <div className="tooltip" style={{ left: Math.min(Math.max(X(hover), 80), w - 120), top: Math.min(...series.map((s) => Y(s.values[hover]))) }}>
          <div className="t">{xLabels[hover]}</div>
          {series.filter((s) => s.label).map((s, i) => (
            <div key={i} className="row" style={{ gap: 6, flexWrap: "nowrap" }}><span className="dot" style={{ background: s.color }} />{s.label}<b style={{ marginLeft: "auto", paddingLeft: 12 }}>{fmt(s.values[hover])}</b></div>
          ))}
          {bands.length > 0 && <div className="t" style={{ marginTop: 4, marginBottom: 0 }}>Range {fmt(bands[0].lo[hover])} – {fmt(bands[0].hi[hover])}</div>}
        </div>
      )}
    </div>
  );
}

export function Spark({ values, color, height = 44, width = 140 }: { values: number[]; color: string; height?: number; width?: number }) {
  const lo = Math.min(...values), hi = Math.max(...values);
  const p = values.map((v, i) => [(i / Math.max(values.length - 1, 1)) * width, 3 + (1 - (v - lo) / Math.max(hi - lo, 1e-9)) * (height - 6)] as [number, number]);
  return <svg width={width} height={height} style={{ display: "block" }}><path d={smooth(p)} fill="none" stroke={color} strokeWidth={2} strokeLinecap="round" /></svg>;
}

// ------------------------------------------------------------ slider with live value
export function Slider({ label, value, min, max, step, onChange, fmt }:
  { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; fmt: (v: number) => string }) {
  return (
    <div>
      <div className="sl-head"><span>{label}</span><b>{fmt(value)}</b></div>
      <input className="slider" type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  );
}

export function Toggle({ on, onChange }: { on: boolean; onChange: (v: boolean) => void }) {
  return <button className={`toggle ${on ? "on" : ""}`} onClick={() => onChange(!on)} aria-pressed={on} />;
}

// ------------------------------------------------------------ confetti burst
export function Confetti({ fire }: { fire: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (!fire) return;
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    cv.width = innerWidth; cv.height = innerHeight;
    const cols = ["#c6f432", "#8fd14f", "#7480e8", "#f5f6f7", "#f0b429"];
    const ps = Array.from({ length: 140 }, () => ({
      x: innerWidth / 2, y: innerHeight * 0.35, vx: (Math.random() - 0.5) * 14, vy: -Math.random() * 13 - 4,
      s: 4 + Math.random() * 6, c: cols[Math.floor(Math.random() * cols.length)], r: Math.random() * 6, vr: (Math.random() - 0.5) * 0.3,
    }));
    let raf = 0, f = 0;
    const tick = () => {
      ctx.clearRect(0, 0, cv.width, cv.height);
      for (const p of ps) {
        p.vy += 0.35; p.x += p.vx; p.y += p.vy; p.r += p.vr; p.vx *= 0.99;
        ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.r); ctx.fillStyle = p.c; ctx.fillRect(-p.s / 2, -p.s / 4, p.s, p.s / 2); ctx.restore();
      }
      if (++f < 150) raf = requestAnimationFrame(tick); else ctx.clearRect(0, 0, cv.width, cv.height);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [fire]);
  return <canvas ref={ref} className="confetti" />;
}

export function Sheet({ open, onClose, children }: { open: boolean; onClose: () => void; children: ReactNode }) {
  if (!open) return null;
  return (
    <div className="sheet-back" onClick={onClose}>
      <div className="sheet" onClick={(e) => e.stopPropagation()}>{children}</div>
    </div>
  );
}
