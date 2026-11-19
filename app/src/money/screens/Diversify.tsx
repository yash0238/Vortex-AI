import { useEffect, useMemo, useState } from "react";
import { FlaskConical, Pencil } from "lucide-react";
import { diversification, type Engine, type Holding } from "../engine";
import { catalogue, type Profile } from "../store";
import { Count } from "../ui";
import type { Results, Track } from "../App";

function NetworkGraph({ hs, pairs, mode }: { hs: Holding[]; pairs: Results["div"]["pairs"]; mode: "calm" | "crash" }) {
  const risky = hs.map((h, i) => ({ h, i })).filter((x) => x.h.kind !== "safe" && x.h.amount > 0);
  const W = 420, H = 300, cx = W / 2, cy = H / 2, R = Math.min(W, H) / 2 - 46;
  const tot = risky.reduce((a, x) => a + x.h.amount, 0) || 1;
  const pos = new Map(risky.map((x, k) => {
    const a = (k / Math.max(risky.length, 1)) * Math.PI * 2 - Math.PI / 2;
    return [x.i, { x: cx + R * Math.cos(a), y: cy + R * Math.sin(a), r: 7 + 16 * Math.sqrt(x.h.amount / tot) }];
  }));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }}>
      {pairs.map((p, k) => {
        const a = pos.get(p.a), b = pos.get(p.b);
        if (!a || !b) return null;
        const v = Math.max(0, mode === "calm" ? p.calm : p.crash);
        return <line key={k} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={mode === "crash" ? "#e2475a" : "#7480e8"}
          strokeOpacity={0.15 + 0.85 * v} strokeWidth={1 + 9 * v * v} style={{ transition: "all .6s ease" }} />;
      })}
      {risky.map(({ h, i }) => {
        const p = pos.get(i)!;
        return (
          <g key={i}>
            <circle cx={p.x} cy={p.y} r={p.r} fill="#1b1e23" stroke="#c6f432" strokeWidth={2} />
            <text x={p.x} y={p.y + p.r + 16} textAnchor="middle" fill="#a4a9b3" fontSize="11">{h.label.length > 18 ? h.label.slice(0, 17) + "…" : h.label}</text>
          </g>
        );
      })}
      <defs><radialGradient id="nodeg"><stop offset="0%" stopColor="#c6f432" /><stop offset="100%" stopColor="#74a214" /></radialGradient></defs>
    </svg>
  );
}

export function Diversify({ eng, profile, r, edit, track }: { eng: Engine; profile: Profile; r: Results; edit: () => void; track: Track; update: (p: Partial<Profile>) => void }) {
  useEffect(() => { track("divers"); }, [track]);
  const [mode, setMode] = useState<"calm" | "crash">("crash");
  const [extra, setExtra] = useState<string[]>([]);
  const d = r.div;
  const twins = [...d.pairs].sort((a, b) => b.crash - a.crash).slice(0, 4);
  const held = new Set(profile.holdings.map((h) => h.id));
  const cands = catalogue(eng).filter((h) => !held.has(h.id) && h.kind !== "safe");
  const total = profile.holdings.reduce((a, h) => a + h.amount, 0);
  const whatIf = useMemo(() => {
    if (!extra.length) return null;
    const add = cands.filter((c) => extra.includes(c.id)).map((c) => ({ ...c, amount: total * 0.1 }));
    return diversification(eng, [...profile.holdings, ...add]);
  }, [extra, eng, profile.holdings, total, cands]);
  const lost = d.calm > 0 ? 1 - d.crash / d.calm : 0;

  return (
    <>
    <div className="pagehead"><div><div className="eyebrow">How many independent bets you really hold</div><h1 className="title">Diversification</h1></div></div>
    <div className="grid">
      <div className="card hero c12">
        <div className="row" style={{ gap: 40 }}>
          <div className="kpi"><span>Holdings</span><b style={{ fontSize: 52, letterSpacing: "-0.045em" }}>{d.nHold}</b></div>
          <div className="kpi"><span>Act like, calm days</span><b style={{ fontSize: 52, letterSpacing: "-0.045em" }} className="goodtext">{d.calm.toFixed(1)}</b></div>
          <div className="kpi"><span>Act like, in a crash</span><b style={{ fontSize: 52, letterSpacing: "-0.045em" }} className="badtext">
            <Count value={d.crash} fmt={(v) => v.toFixed(1)} /></b></div>
          <div className="kpi"><span>Protection lost in a crash</span><b style={{ fontSize: 32 }}>{Math.round(lost * 100)}%</b></div>
        </div>
        <p className="muted" style={{ marginTop: 14 }}>When markets panic, holdings that normally move separately fall together. That's the core finding of the VORTEX-AI research.</p>
      </div>

      <div className="card c7 delay1">
        <div className="row between"><h3>How your holdings are tied together</h3>
          <div className="row" style={{ gap: 6 }}>
            <button className={`chip ${mode === "calm" ? "on" : ""}`} onClick={() => setMode("calm")}>Calm market</button>
            <button className={`chip ${mode === "crash" ? "on" : ""}`} onClick={() => setMode("crash")}>Crash</button></div></div>
        <NetworkGraph hs={profile.holdings} pairs={d.pairs} mode={mode} />
        <p className="muted">Thicker, brighter lines = move more together. Flip between calm and crash to watch the ties tighten.
          Safe money isn't shown: it doesn't fall with stocks.</p>
      </div>

      <div className="card c5 delay2 stack">
        <h3>Crash twins in your portfolio</h3>
        {twins.length === 0 && <p className="sub">Add at least two stock or fund holdings to see this.</p>}
        {twins.map((t, k) => (
          <div key={k} className="list-row">
            <span><b>{profile.holdings[t.a].label}</b> + <b>{profile.holdings[t.b].label}</b>
              <div className="muted">move together {Math.round(Math.max(0, t.calm) * 100)}% on calm days → <span className="badtext">{Math.round(Math.max(0, t.crash) * 100)}% in a crash</span></div></span>
          </div>
        ))}
        <button className="btn ghost small" onClick={edit}><Pencil size={14} /> Change holdings</button>
      </div>

      <div className="card c12 delay3">
        <div className="row between"><h3><FlaskConical size={14} style={{ verticalAlign: -2 }} /> Experiment: what if you added…</h3>
          {whatIf && <span className="pill good">In a crash: {d.crash.toFixed(1)} → {whatIf.crash.toFixed(1)}</span>}</div>
        <div className="row" style={{ marginTop: 10 }}>
          {cands.map((c) => (
            <button key={c.id} className={`chip ${extra.includes(c.id) ? "on" : ""}`}
              onClick={() => setExtra((x) => (x.includes(c.id) ? x.filter((y) => y !== c.id) : [...x, c.id]))}>+ {c.label}</button>
          ))}
        </div>
        <p className="muted" style={{ marginTop: 10 }}>Each experiment adds a 10% slice. This is a learning tool, not a recommendation: diversification
          lowers risk, it doesn't remove it, and safe money (FD / debt) remains the only thing that truly doesn't crash with stocks.</p>
      </div>
    </div>
    </>
  );
}
