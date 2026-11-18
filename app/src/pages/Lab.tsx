import { useState } from "react";
import type { Any, Core, Samples } from "../lib/data";
import { API, useData } from "../lib/data";
import { Columns, LineChart, fmt } from "../ui/charts";
import { Card, Empty, Head, Loading, Seg, Stat, Term } from "../ui/kit";

const Q = ["5%", "25%", "median", "75%", "95%"];

function Panel({ d, color, title }: { d: Any; color: string; title: string }) {
  if (!d) return <Card title={title}><div className="empty">This model produced fewer than 5 windows of this kind.</div></Card>;
  return (
    <Card title={title} sub="Cumulative return of an equal-weight portfolio over 60 trading days. Shaded: middle 50% and 90% of windows. Thin lines: individual windows.">
      <LineChart height={280} yFmt={(v) => v.toFixed(0) + "%"} legend={false} labels={Array.from({ length: 60 }, (_, i) => `day ${i + 1}`)}
        bands={[{ lo: d.fan[0], hi: d.fan[4], color: `color-mix(in srgb, ${color} 14%, transparent)` }, { lo: d.fan[1], hi: d.fan[3], color: `color-mix(in srgb, ${color} 22%, transparent)` }]}
        series={[...d.paths.slice(0, 8).map((y: number[], i: number) => ({ name: "path " + i, color, y, faint: true })), { name: Q[2], color, y: d.fan[2], width: 2.4 }]} />
      <div className="grid g3" style={{ marginTop: 12 }}>
        <Stat k="Annualised volatility" v={fmt(d.vol, 1) + "%"} />
        <Stat k={<Term k="boost_pct">Average pair correlation</Term>} v={fmt(d.mac)} />
        <Stat k="1-in-100 worst day" v={fmt(d.worst_day, 1) + "%"} />
      </div>
    </Card>
  );
}

export function Lab() {
  const sm = useData<Samples>("samples"), core = useData<Core>("core");
  const [fit, setFit] = useState("full");
  const [gen, setGen] = useState("regime_gaussian");
  const [live, setLive] = useState<Any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  if (!sm.data || !core.data) return <Loading error={sm.error ?? core.error} />;
  const S = sm.data[`nse/${fit}`] ?? {};
  const names = core.data.pretty;
  const gens = Object.keys(S).filter((k) => k !== "real");
  const g = S[gen] ? gen : gens[0];
  const cur = live ?? S[g];
  const real = S.real;
  const runLive = async () => {
    setBusy(true); setErr(null);
    try {
      const r = await fetch(`${API}/api/generate`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ n: 400, seed: Math.floor(Math.random() * 1e9), fit }) });
      if (!r.ok) {
        let msg = `Server answered HTTP ${r.status}`;
        try {
          const body = await r.json();
          if (body?.detail) msg = body.detail;
        } catch {}
        throw new Error(msg);
      }
      setLive(await r.json());
    } catch (e) {
      setErr(`Live generation requires the FastAPI server (python -m uvicorn api.server:app --port 8000) and a trained CG-NSDE v2 checkpoint (.pt) in artifacts/models/nse/cgnsde2/${fit}/seed0.pt. ${String(e)}`);
    }
    setBusy(false);
  };
  const bins = Array.from({ length: 41 }, (_, i) => (i % 10 === 0 ? `${-8 + i * 0.4}%` : ""));
  return (
    <>
      <Head eyebrow="Scenario lab" title="Look at the artificial markets yourself">
        Pick a model. Left: what its calm windows look like. Right: its crisis windows. Compare with real history underneath. A good model's crisis panel should look like the real crisis panel: wider, lower, and with stocks more tightly linked.
      </Head>
      <div className="callout info" style={{ marginBottom: 16 }}>
        <b>How the Lab works:</b> Precomputed 60-day paths and quantile fans are displayed below for the 6 classical generators across both fit regimes. Comprehensive benchmark evaluations for CG-NSDE v1 and CG-NSDE v2 (post hoc) are available in the <b>Benchmark</b> and <b>Model</b> pages. Live generation from the trained neural SDE is supported via FastAPI with both <b>in-sample (full)</b> and <b>pre-2020</b> checkpoints active on disk.
      </div>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
        <Seg value={fit} set={(v) => { setFit(v); setLive(null); }} options={[{ v: "full", label: "Trained on all history (in-sample)" }, { v: "pre2020", label: "Trained to 2019 only (pre-2020)" }]} />
        <select className="btn" value={g} onChange={(e) => { setGen(e.target.value); setLive(null); }}>{gens.map((k) => (<option key={k} value={k}>{names[k] ?? k}</option>))}</select>
        <button className="btn primary" onClick={runLive} disabled={busy}>{busy ? "Generating…" : "Generate fresh CG-NSDE v2 scenarios (live)"}</button>
        {live && (
          <>
            <span className="pill info">live sample: {live.n} windows, {live.n_crisis} crisis</span>
            <button className="btn" onClick={() => setLive(null)}>Reset to precomputed</button>
          </>
        )}
      </div>
      {err && <div className="callout crisis" style={{ marginTop: 12 }}>{err}</div>}
      {!cur ? <Empty what="Generated samples for this model" /> : (
        <>
          <h2>{live ? "CG-NSDE v2 (live sample)" : names[g] ?? g}<span className="pill" style={{ marginLeft: 10 }}>{cur.n_crisis} of {cur.n} windows labelled crisis by the frozen rule</span></h2>
          <div className="grid g2"><Panel d={cur.normal} color="var(--s-ours)" title="Calm windows" /><Panel d={cur.crisis} color="var(--crisis)" title="Crisis windows" /></div>
          <h2>Real history<span className="pill" style={{ marginLeft: 10 }}>{real.n_crisis} of {real.n} windows are crisis windows</span></h2>
          <div className="grid g2"><Panel d={real.normal} color="var(--ink-2)" title="Real calm windows" /><Panel d={real.crisis} color="var(--crisis)" title="Real crisis windows" /></div>
          <div className="grid g2">
            {(["normal", "crisis"] as const).map((k) => (cur[k] && real[k] ? (
              <Card key={k} title={`Daily returns in ${k === "normal" ? "calm" : "crisis"} windows`} sub="Bars: the model. Line: real data. Fat tails show up as bars far from the centre.">
                <Columns values={cur[k].hist} values2={real[k].hist} labels={bins} color={k === "normal" ? "var(--s-ours)" : "var(--crisis)"} names={[live ? "CG-NSDE v2 (live)" : names[g] ?? g, "Real"]} />
              </Card>) : null))}
          </div>
        </>)}
    </>
  );
}
