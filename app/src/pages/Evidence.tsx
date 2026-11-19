import { useMemo, useState } from "react";
import { G, nameOf } from "../glossary";
import type { Any, Bench, Core, Market } from "../lib/data";
import { FAMILY, famColor, useData } from "../lib/data";
import { Heatmap, IntervalPlot, LineChart, Scatter, fmt } from "../ui/charts";
import { BenchTable, Card, Empty, Head, Loading, Records, Seg, Stat, Term, Verdict } from "../ui/kit";

const FAM_METRICS: Record<string, string[]> = {
  C: ["crisis_freq", "boost_pct", "tail_coexc", "cvar_ratio", "pc1_share_gap", "connect_gap"],
  B: ["corr_fro", "corr_mae", "eig_dist", "pc1_share"],
  A: ["std_ratio", "kurt_med", "acf1_ret", "acf1_sq", "acf5_sq_mean", "skew_ew"],
  D: ["disc_feat", "disc_raw"],
};
const FAM_NAME: Record<string, string> = { C: "Crisis behaviour", B: "Correlation fidelity", A: "Single-stock behaviour", D: "Detectability" };
const FITS = [{ v: "full", label: "Whole history (in-sample)" }, { v: "pre2020", label: "Trained to 2019, judged on 2020–24" }];

function Legend({ rows }: { rows: Any[] }) {
  return (<div className="legend">{Object.entries(FAMILY).filter(([k]) => rows.some((r) => r.family_tag === k)).map(([k, f]) => (<span key={k} title={f.blurb}><i style={{ background: f.color }} />{f.label}</span>))}</div>);
}

function MetricStrip({ b, m }: { b: Bench; m: string }) {
  const r = b.reference.metrics[m];
  if (!r) return null;
  const rows = b.rows.map((x) => ({ label: x.name, color: famColor(x.family_tag), value: x[m] as number | null, sd: x[m + "_sd"] as number | null, bold: x.generator === "cgnsde2" }));
  return (
    <Card flat title={<Term k={m} />} sub={G[m]?.short}>
      {r.kind === "dist" ? <IntervalPlot rows={rows} ceiling={r.ceiling} zero />
        : r.kind === "disc" ? <IntervalPlot rows={rows} band={{ lo: 0.5, hi: r.hi as number, label: "real vs real" }} domain={[0.3, 1.02]} />
          : <IntervalPlot rows={rows} band={{ lo: r.lo as number, hi: r.hi as number, mid: r.real }} />}
    </Card>
  );
}

/* ------------------------------------------------------------ Benchmark */
export function Benchmark({ market = "nse" }: { market?: string }) {
  const core = useData<Core>("core");
  const [fit, setFit] = useState("full");
  const [fam, setFam] = useState("C");
  if (!core.data) return <Loading error={core.error} />;
  const b = core.data.bench[`${market}/${fit}`];
  const sum = core.data.summary[`${market}/${fit}`];
  return (
    <>
      {market === "nse" && <Head eyebrow="Evidence · CrisisBench" title="The benchmark: every model, same rules">
        Each model was trained on the same NSE data and asked for 2,000 new 60-day windows, five times with different random seeds. Green cells lie inside the real-data band.
      </Head>}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
        <Seg value={fit} set={setFit} options={FITS} />
        {b && <span className="pill">{b.reference.n_windows} real windows, {b.reference.n_crisis} crisis</span>}
      </div>
      {!b ? <Empty what={`The ${market} benchmark for this setting`} /> : (
        <>
          <Card title="Scoreboard" sub={<><Term k="score">Family scores</Term>: average error in units of real sampling noise. Under 1.0 is as close as real data can tell. Rank in brackets.</>}>
            <div className="tbl-wrap"><table className="tbl"><thead><tr><th>Model</th>{["C", "B", "A", "D"].map((f) => (<th key={f}>{FAM_NAME[f]}</th>))}<th>Metrics inside real band</th><th>Train time</th><th>Seeds</th><th style={{ textAlign: "left" }}>Implementation</th></tr></thead>
              <tbody>{[...b.rows].sort((x, y) => x.score_C - y.score_C).map((r) => (<tr key={r.generator} className={r.generator === "cgnsde2" ? "ours" : ""}>
                <td><span className="dot" style={{ background: famColor(r.family_tag) }} />{r.name}</td>
                {["C", "B", "A", "D"].map((f) => (<td key={f} className={r[`score_${f}`] <= 1 ? "in" : "out"}>{fmt(r[`score_${f}`])} <small>({r[`rank_${f}`]})</small></td>))}
                <td>{r.n_inside} / 18</td><td>{r.train_minutes < 0.05 ? "< 3 s" : fmt(r.train_minutes, 1) + " min"}</td><td>{r.seeds}</td><td style={{ textAlign: "left" }}>{r.impl_source}</td></tr>))}</tbody></table></div>
            <Legend rows={b.rows} />
          </Card>
          <Card title="Do the metric families agree on who is best?" sub={<>Agreement between rankings (<Term k="kendall" />). Low or negative agreement between correlation fidelity (B) and crisis behaviour (C) means the usual score cannot be used to pick a crisis-realistic model.</>}
            right={sum?.tau_B_vs_C?.tau !== undefined ? <span className="pill info">B vs C: tau = {fmt(sum.tau_B_vs_C.tau)} [{fmt(sum.tau_B_vs_C.lo)}, {fmt(sum.tau_B_vs_C.hi)}]</span> : undefined}>
            <div className="grid g2">
              <Scatter height={320} xLabel="Correlation fidelity score (B)" yLabel="Crisis behaviour score (C)" xBand={[0, 1]} yBand={[0, 1]}
                points={b.rows.map((r) => ({ x: r.score_B, y: r.score_C, label: r.name, color: famColor(r.family_tag), bold: r.generator === "cgnsde2" }))} />
              <div>
                {sum?.family_taus && <Records rows={Object.entries(sum.family_taus as Record<string, number>).map(([k, v]) => ({ "Families compared": k.replace("_vs_", " vs "), "Kendall tau": v }))} />}
                {sum?.tau_B_vs_C?.loo_min !== undefined && <p style={{ fontSize: 13 }}>Dropping any single model, tau(B, C) stays between {fmt(sum.tau_B_vs_C.loo_min)} and {fmt(sum.tau_B_vs_C.loo_max)}. With the alternative crisis rule: {fmt(sum.tau_B_vs_C_alt_rule)}.</p>}
              </div>
            </div>
          </Card>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
            <h2>Metric by metric</h2>
            <Seg value={fam} set={setFam} options={["C", "B", "A", "D"].map((f) => ({ v: f, label: FAM_NAME[f] }))} />
          </div>
          <div className="grid g2">{FAM_METRICS[fam].map((m) => (<MetricStrip key={m} b={b} m={m} />))}</div>
          <Card title="Full table" sub="Mean ± standard deviation over seeds. n/p = the model produced too few crisis windows for the metric to exist.">
            <BenchTable rows={b.rows} refm={b.reference.metrics} metrics={FAM_METRICS[fam]} />
          </Card>
          {b.metric_tau && <Card title="Which individual metrics rank models alike?" sub="Kendall tau between the model rankings produced by each pair of metrics. Red = agree, blue = disagree.">
            <Heatmap matrix={b.metric_tau.matrix} rowLabels={b.metric_tau.metrics.map(nameOf)} colLabels={b.metric_tau.metrics.map(nameOf)} lo={-1} hi={1} mode="div" showValues size={760} />
          </Card>}
        </>)}
    </>
  );
}

/* ----------------------------------------------------------------- Data */
export function DataPage() {
  const mk = useData<Record<string, Market>>("markets");
  const [m, setM] = useState("nse");
  const [view, setView] = useState("diff");
  const d = mk.data?.[m];
  const regions = useMemo(() => {
    if (!d) return [];
    const idx = new Map<string, number>(d.series.date.map((x: string, i: number) => [x, i]));
    const find = (s: string) => { let best = 0; for (const [k, i] of idx) { if (k <= s) best = i; else break; } return best; };
    return d.episodes.map((e: Any) => ({ from: find(e.start), to: find(e.end) + 1 }));
  }, [d]);
  if (!mk.data) return <Loading error={mk.error} />;
  const opts = Object.entries(mk.data).filter(([, v]) => v).map(([k, v]) => ({ v: k, label: (v as Any).label }));
  if (!d) return <Empty what="This market panel" />;
  const diff = d.corr_crisis.map((r: number[], i: number) => r.map((v, j) => v - d.corr_normal[i][j]));
  const mat = view === "diff" ? diff : view === "crisis" ? d.corr_crisis : d.corr_normal;
  return (
    <>
      <Head eyebrow="Evidence · data" title="The real data behind everything">Daily prices from Yahoo Finance, cleaned by fixed, logged rules. Red bands mark the crisis episodes found by the frozen rule.</Head>
      <Seg value={m} set={setM} options={opts} />
      <div className="grid g4">
        <Card flat><Stat k="Stocks" v={d.n_assets} s={d.label} /></Card>
        <Card flat><Stat k="Trading days" v={d.n_days.toLocaleString()} s={`${d.first} to ${d.last}`} /></Card>
        <Card flat><Stat k="Crisis episodes" v={d.episodes.length} s={`${fmt((100 * d.windows.n_crisis) / d.windows.n, 1)}% of windows`} /></Card>
        <Card flat><Stat k={<Term k="boost_pct">Average pair correlation</Term>} v={`${fmt(d.mac.normal)} → ${fmt(d.mac.crisis)}`} s="calm windows → crisis windows" /></Card>
      </div>
      <Card title="Market level with crisis episodes" sub="Equal-weight index of all stocks (start = 100).">
        <LineChart height={280} labels={d.series.date} regions={regions} series={[{ name: "Equal-weight index", color: "var(--accent)", y: d.series.index }]} yFmt={(v) => v.toFixed(0)} />
      </Card>
      <div className="grid g2">
        <Card title="Volatility" sub="20-day annualised volatility of the index, %.">
          <LineChart height={220} labels={d.series.date} regions={regions} series={[{ name: "Volatility", color: "var(--s-dyn)", y: d.series.vol }]} yFmt={(v) => v.toFixed(0) + "%"} />
        </Card>
        <Card title="How correlated the stocks are, window by window" sub="Average absolute pairwise correlation inside each 60-day window. It jumps in crises: this jump is what models must reproduce.">
          <LineChart height={220} labels={d.windows.end} series={[{ name: "Average pair correlation", color: "var(--crisis)", y: d.windows.mac }]}
            regions={d.windows.label.reduce((acc: Any[], v: number, i: number) => { if (v) { const l = acc[acc.length - 1]; if (l && l.to === i) l.to = i + 1; else acc.push({ from: i, to: i + 1 }); } return acc; }, [])} />
        </Card>
      </div>
      <Card title="Correlation between every pair of stocks" sub={m.startsWith("nse") ? "Stocks are grouped by sector (white lines)." : undefined}
        right={<Seg value={view} set={setView} options={[{ v: "normal", label: "Calm" }, { v: "crisis", label: "Crisis" }, { v: "diff", label: "Crisis minus calm" }]} />}>
        <Heatmap matrix={mat} rowLabels={d.tickers} colLabels={d.tickers} lo={view === "diff" ? -0.4 : 0} hi={view === "diff" ? 0.4 : 0.9} mode={view === "diff" ? "div" : "seq"} groups={d.sectors} size={620} />
        <p style={{ fontSize: 13 }}>{view === "diff" ? "Red = the pair is more correlated in crises than in calm periods. Nearly the whole map turns red: in a crisis, diversification across stocks and sectors weakens." : "Darker = more correlated. Hover for the pair and value."}</p>
      </Card>
      <div className="grid g2">
        <Card title="Crisis episodes"><Records rows={d.episodes} cols={[{ k: "start", label: "First window ends" }, { k: "end", label: "Last window ends" }, { k: "windows", label: "Windows" }]} /></Card>
        <Card title="Cleaning log" sub={<>Checksum of the cleaned return matrix: <code>{String(d.sha256).slice(0, 16)}…</code></>}>
          <Records rows={d.cleaning_log.filter((x: Any) => x.action !== "NOT_USED")} cols={[{ k: "ticker", label: "Ticker" }, { k: "action", label: "Action" }, { k: "date", label: "Date" }, { k: "detail", label: "Detail" }]} />
        </Card>
      </div>
    </>
  );
}

/* ---------------------------------------------------------- Crisis test */
export function CrisisTest() {
  const core = useData<Core>("core");
  if (!core.data) return <Loading error={core.error} />;
  const h = core.data.heldout, b = core.data.bench["nse/pre2020"], ext = core.data.bench["nse_ext/upto2024"];
  const eps = h ? [...new Set(h.map((r) => r.episode as string))] : [];
  return (
    <>
      <Head eyebrow="Evidence · held-out crashes" title="Crashes the models never saw">
        Every model here was trained only on data up to December 2019. COVID-2020, the 2022 shock and mid-2024 were never shown to them. Does their idea of a crisis match what then happened?
      </Head>
      <div className="callout warn" style={{ marginBottom: 16 }}>
        <b>Gate G2 Verdict (FAIL for all models):</b> On the held-out COVID-2020 crash, real cross-stock correlation surged by +101.1% (real 95% interval: 65.11% to 140.1%); the 2022 shock was +49.42% (real 95% interval: 31.44% to 71.67%). CG-NSDE v2 (post hoc, pre-2020 fit) generated +22.15% (±2.08%) and CG-NSDE v1 generated +2.91% (±0.91%) — both outside the real intervals on 0 of 2 episodes (G2 FAIL; v2's Family C error rises ~5x out-of-sample from 0.145 in-sample to 0.742 pre-2020). Models trained purely on mild historical periods cannot extrapolate to unprecedented systemic shocks.
      </div>
      {b && <div className="grid g2"><MetricStrip b={b} m="boost_pct" /><MetricStrip b={b} m="cvar_ratio" /><MetricStrip b={b} m="tail_coexc" /><MetricStrip b={b} m="crisis_freq" /></div>}
      {!h ? <Empty what="The per-episode held-out table" /> : eps.map((e) => {
        const rows = h.filter((r) => r.episode === e);
        const r0 = rows[0];
        return (<Card key={e} title={`Episode: ${r0.episode_label ?? e}`} sub={`${r0.n_real_crisis} real crisis windows. Band = real 95% interval for this episode.`}>
          <IntervalPlot band={{ lo: r0.lo, hi: r0.hi, mid: r0.real }} xFmt={(v) => v.toFixed(0) + "%"}
            rows={rows.map((r) => ({ label: r.name, color: famColor(r.family_tag), value: r.gen, sd: r.gen_sd, bold: r.generator === "cgnsde2", note: r.inside ? "inside the real band" : "outside the real band" }))} />
        </Card>);
      })}
      <Card title="An untouched out-of-time check: 2025 to October 2026" sub="Models trained on data up to December 2024, judged on the following 22 months, which were downloaded only after the protocol was frozen.">
        {ext ? <BenchTable rows={ext.rows} refm={ext.reference.metrics} metrics={["boost_pct", "cvar_ratio", "tail_coexc", "corr_fro", "kurt_med", "acf1_sq"]} /> : <Empty what="The 2025–2026 out-of-time run" />}
      </Card>
    </>
  );
}

/* ----------------------------------------------------------------- Risk */
export function Risk() {
  const core = useData<Core>("core");
  const [level, setLevel] = useState("0.99");
  const [pf, setPf] = useState("equal_weight");
  if (!core.data) return <Loading error={core.error} />;
  const r = core.data.risk, s = core.data.risk_series, mv = core.data.metric_validity;
  const rows = (r ?? []).filter((x) => String(x.level) === level && x.portfolio === pf).sort((a, b) => a.fz0 - b.fz0);
  return (
    <>
      <Head eyebrow="Evidence · downstream use" title="Would a risk desk trust these scenarios?">
        Each model's synthetic windows are used to forecast tomorrow's <Term k="var" /> and <Term k="es" /> for a portfolio of the 47 stocks, every trading day from 2020 to 2024, using only data available at the time. Then we count how often real losses broke the forecast.
      </Head>
      {!r ? <Empty what="The VaR / Expected Shortfall backtest" /> : (<>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          <Seg value={level} set={setLevel} options={[{ v: "0.95", label: "95% VaR" }, { v: "0.975", label: "97.5%" }, { v: "0.99", label: "99% VaR" }]} />
          <Seg value={pf} set={setPf} options={[{ v: "equal_weight", label: "Equal-weight portfolio" }, { v: "dirichlet_seed7", label: "Random-weight portfolio" }]} />
        </div>
        <Card title="Backtest results" sub={<>Sorted by <Term k="fz0" /> (best first). <Term k="kupiec" />, <Term k="christoffersen" />: p-values below 0.05 mean the model fails that test. <Term k="traffic" /> applies to the 99% level.</>}>
          <div className="tbl-wrap"><table className="tbl"><thead><tr><th>Method</th><th>Breaches</th><th>Expected</th><th>Kupiec p</th><th>Independence p</th><th>Cond. coverage p</th><th>Traffic light</th><th>ES test Z2</th><th>FZ0 loss</th></tr></thead>
            <tbody>{rows.map((x) => (<tr key={x.method} className={x.method === "cgnsde2" ? "ours" : x.kind === "benchmark" ? "real" : ""}>
              <td><span className="dot" style={{ background: x.kind === "benchmark" ? "var(--real)" : famColor(x.family_tag) }} />{x.name}</td>
              <td>{x.violations}</td><td>{fmt(x.expected, 1)}</td>
              {["kupiec_p", "ind_p", "cc_p"].map((k) => (<td key={k} className={x[k] >= 0.05 ? "in" : "out"}>{fmt(x[k], 3)}</td>))}
              <td>{level === "0.99" ? <span className={"pill " + (x.traffic === "green" ? "pass" : x.traffic === "red" ? "fail" : "warn")}>{x.traffic}</span> : "–"}</td>
              <td>{fmt(x.z2, 2)}</td><td>{fmt(x.fz0, 3)}</td></tr>))}</tbody></table></div>
        </Card>
        {s && <Card title="99% VaR forecasts against what happened" sub="Equal-weight portfolio. Grey bars are real daily returns; each line is a model's loss limit. A return below the line is a breach.">
          <LineChart height={340} labels={s.dates} yFmt={(v) => v.toFixed(0) + "%"}
            series={[{ name: "Daily return", color: "var(--muted)", y: s.ret, faint: true }, ...Object.entries(s.methods as Record<string, Any>).map(([k, v]) => ({ name: v.name, color: v.color ?? "var(--accent)", y: v.var99, width: k === "cgnsde2" ? 2.4 : 1.6 }))]} />
        </Card>}
        <Card title="Which metric family predicts risk quality?" sub={<>For each family: <Term k="kendall" /> between the benchmark ranking of the models and their backtest ranking. This is the test of whether crisis metrics matter in practice.</>}>
          {mv ? <IntervalPlot zero domain={[-1, 1]} xFmt={(v) => v.toFixed(1)} rows={mv.map((x) => ({ label: x.label, color: x.kind === "family" ? "var(--accent)" : "var(--s-static)", value: x.tau, sd: null, note: `95% interval ${fmt(x.lo)} to ${fmt(x.hi)}`, bold: x.kind === "family" }))} /> : <Empty what="The metric-validity analysis" />}
        </Card>
      </>)}
    </>
  );
}

/* ---------------------------------------------------------------- Model */
export function Model() {
  const core = useData<Core>("core");
  if (!core.data) return <Loading error={core.error} />;
  const cg = core.data.cgnsde, b = core.data.bench["nse/full"], abl = b?.ablation;
  const hist: Any[] = cg?.info?.history ?? [];
  const VAR: Record<string, string> = { "cgnsde2": "Full model (v2)", "cgnsde2@no_gat": "without graph attention", "cgnsde2@no_supcon": "without contrastive loss",
    "cgnsde2@no_graph_loss": "without graph loss", "cgnsde2@no_sde": "without the SDE (frozen hidden state)", "cgnsde2@scalar_vol": "one volatility state for everything (as in v1)",
    "cgnsde2@no_voldyn": "without volatility memory", "cgnsde2@no_leverage": "without the leverage effect", "cgnsde2@gauss": "Gaussian likelihood instead of fat tails",
    "cgnsde2@param_innov": "drawn shocks instead of replayed residuals", "cgnsde2@beta01": "weaker bottleneck (KL weight 0.1)", "cgnsde2@K1": "1 factor", "cgnsde2@K4": "4 factors", "cgnsde2@K16": "16 factors" };
  const v1 = cg?.v1_info;
  return (
    <>
      <Head eyebrow="Evidence · our model" title="CG-NSDE under the microscope">
        Two model generations, four distinct architectural issues across iterations: legacy codebase defects (sampler mismatch and likelihood dominance) were resolved during protocol standardization, while CrisisBench v1 failed against classical regime models due to overlapping-window memorisation and scalar volatility scaling. CG-NSDE v2 was redesigned post hoc to solve all four.
      </Head>
      <div className="grid g2">
        <Card flat title="Version 1: what went wrong">
          <p><b>Diagnosis of failure:</b> The first model memorised its training windows (best validation at epoch {v1?.best_epoch ?? "2"} of {v1?.epochs_run ?? "10"}). Four distinct architectural issues occurred across iterations:</p>
          <ul style={{ margin: "8px 0 0 16px", padding: 0, fontSize: 13.5, color: "var(--ink-2)" }}>
            <li><b>Legacy code defects (resolved in protocol audit):</b> Earlier prototypes in <code>training/vortex_model.py</code> suffered from <i>sampler mismatch</i> (deterministic generation training vs noisy inference) and <i>likelihood dominance</i> (unweighted reconstruction overpowering latent dynamics).</li>
            <li><b>CrisisBench v1 — Overlapping-window memorisation:</b> Validation NLL peaked at epoch 2 of 10. Weak KL bottleneck (beta=0.01) and unconstrained graph embeddings allowed the decoder to memorise heavily overlapping 60-day training windows instead of learning generalisable dynamics.</li>
            <li><b>CrisisBench v1 — Scalar volatility scaling:</b> A single global volatility state scaled the whole covariance matrix by one scalar, making it mathematically impossible to model cross-asset correlation surges during crises.</li>
          </ul>
          <p className="muted" style={{ marginTop: 8 }}><b>Gate verdicts:</b> G1 FAIL (score_C 1.22, rank 11 of 15), G2 FAIL (0/2 held-out episodes).</p>
        </Card>
        <Card flat title="Version 2: architectural redesign (post hoc)">
          <p><b>How v2 solves each defect:</b> Redesigned after understanding v1's failure, using 2010–2019 fit-period data only:</p>
          <ul style={{ margin: "8px 0 0 16px", padding: 0, fontSize: 13.5, color: "var(--ink-2)" }}>
            <li><b>Factor EGARCH dynamics (fixes scalar vol):</b> 8 static PCA-initialised factor loadings (rank_K=8) + idiosyncratic variance, whose log-variances evolve via a discrete-time EGARCH-type recursion on standardised shocks (with leverage theta on factor shocks), modulated by the 16-d latent SDE path. Correlated stocks surge together in crises.</li>
            <li><b>Strict posterior isolation (fixes memorisation):</b> Graph attention (GAT) is restricted strictly to the posterior encoder during training; the generator samples from an unconditioned latent prior without real-data leakage.</li>
            <li><b>Filtered Historical Simulation (FHS innovations):</b> Standardised empirical residuals replace rigid parametric distributions, natively capturing kurtosis and extreme daily tail shocks.</li>
            <li><b>Parameter efficiency &amp; balanced objective:</b> 41,396 parameters (73% fewer than v1's 152,889) with a balanced loss objective, eliminating capacity for window memorisation and resolving likelihood dominance.</li>
          </ul>
          <p className="muted" style={{ marginTop: 8 }}><b>Gate verdicts:</b> G1 PASS (score_C 0.145, rank 1 in-sample), G2 FAIL (0/2 held-out episodes: ~5&times; OOS degradation to 0.742).</p>
        </Card>
      </div>

      <Card title="Architecture comparison: v1 vs v2">
        <div className="tbl-wrap"><table className="tbl"><thead><tr><th>Property</th><th>CG-NSDE v1 (preregistered)</th><th>CG-NSDE v2 (ours, post hoc)</th><th>Significance</th></tr></thead><tbody>
          <tr><td>Parameters</td><td>152,889</td><td>41,396 (-73%)</td><td>Vastly reduced capacity prevents training memorisation</td></tr>
          <tr><td>Volatility state</td><td>Single global scalar</td><td>K=8 factor log-vols + idiosyncratic</td><td>Factor volatility allows correlation to surge in crises</td></tr>
          <tr><td>Graph attention (GAT)</td><td>In generator conditioning</td><td>Posterior encoder only</td><td>Eliminates data leakage into generated paths</td></tr>
          <tr><td>Innovation noise</td><td>Parametric Student-t</td><td>Filtered Historical Simulation (FHS)</td><td>Accurately matches empirical kurtosis and fat tails</td></tr>
          <tr><td>Convergence</td><td>Overfits by epoch 2</td><td>Steady progress to epoch 5/13</td><td>Balanced weighted objective resolves likelihood dominance</td></tr>
          <tr><td>G1 Verdict</td><td><span className="pill fail">FAIL</span> (score_C: 1.22, rank 11)</td><td><span className="pill pass">PASS</span> (score_C: 0.145, rank 1)</td><td>v2 outperforms all classical regime models in-sample</td></tr>
          <tr><td>G2 Verdict</td><td><span className="pill fail">FAIL</span> (0/2 episodes)</td><td><span className="pill fail">FAIL</span> (0/2 episodes)</td><td>Both fail held-out 2020 crash — Family C error rises ~5&times; OOS (0.145 &rarr; 0.742)</td></tr>
        </tbody></table></div>
      </Card>

      {!cg ? <Empty what="CG-NSDE training" /> : (
        <div className="grid g4">
          <Card flat><Stat k="Training time (CPU)" v={`${fmt(cg.train_minutes, 1)} min`} s={`${cg.info?.epochs_run ?? "?"} epochs, best at ${cg.info?.best_epoch ?? "?"}`} /></Card>
          <Card flat><Stat k="Tail thickness learned (ν)" v={fmt(cg.info?.nu, 1)} s="lower = fatter tails; a bell curve is ∞" /></Card>
          <Card flat><Stat k="Common factors" v={cg.hparams?.rank_K ?? "8"} s="market-wide drivers, each with its own volatility memory" /></Card>
          <Card flat><Stat k="Requested-regime boost" v={cg.conditional ? `+${fmt(cg.conditional.boost_pct, 0)}%` : "n/a"} s="when asked for crisis vs calm directly (not used for ranking)" /></Card>
        </div>)}
      {hist.length > 0 && <Card title="Training curves" sub="Negative log-likelihood per epoch (lower is better). Training stops when the validation curve stops improving.">
        <LineChart height={240} labels={hist.map((_, i) => `epoch ${i + 1}`)} series={[{ name: "train", color: "var(--s-ours)", y: hist.map((h) => h.train_nll ?? h.train) }, { name: "validation (2018–19)", color: "var(--crisis)", y: hist.map((h) => h.val_nll ?? h.val) }]} />
      </Card>}
      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}><h2>Ablations</h2><p className="muted" style={{ marginLeft: 12, fontSize: 12 }}>All 39 runs (13 variants × 3 seeds + 5 full v2 seeds) evaluated on the full NSE-47 panel (nse/full)</p></div>
      {!abl?.length ? <Empty what="The ablation study" /> : (<>
        <div className="grid g2">
          {["score_C", "score_B", "score_A", "score_D"].map((k) => (<Card flat key={k} title={FAM_NAME[k.slice(-1)] + " score"} sub="Lower is better; under 1 is within real noise.">
            <IntervalPlot zero ceiling={1} rows={abl.map((r) => ({ label: VAR[r.generator] ?? r.name, color: r.generator === "cgnsde2" ? "var(--s-ours)" : "var(--s-static)", value: r[k], sd: r[k + "_sd"], bold: r.generator === "cgnsde2" }))} />
          </Card>))}
        </div>
        <Card title="Ablation table"><BenchTable rows={abl.map((r) => ({ ...r, name: VAR[r.generator] ?? r.name }))} refm={b.reference.metrics} metrics={["boost_pct", "cvar_ratio", "tail_coexc", "corr_fro", "kurt_med", "acf1_sq", "disc_feat"]} /></Card>
      </>)}
      {cg?.explain?.latent && <Card title="Does the hidden space separate crisis from calm?" sub="Each dot is one real training window, placed by the first two principal components of its hidden starting state. Red = crisis windows.">
        <Scatter height={360} xLabel="hidden component 1" yLabel="hidden component 2"
          points={cg.explain.latent.map((p: Any, i: number) => ({ x: p[0], y: p[1], label: "", color: p[2] ? "var(--crisis)" : "var(--s-static)", note: String(i) }))} />
      </Card>}
    </>
  );
}

/* Gate markdown files can start with a heading (## GATE_ACFr2) and carry a table
   row like | **Pass** | **True** | or | **Pass** | **False** |. Parse that into a verdict;
   fall back to an older "Verdict: ..." first line so no gate shows "not run". */
function gateVerdict(md: string): string | undefined {
  const m = md.match(/\|\s*\*\*Pass\*\*\s*\|\s*\*\*(True|False)\*\*\s*\|/);
  if (m) return m[1] === "True" ? "PASS" : "FAIL";
  const first = md.split("\n")[0];
  const i = first.indexOf(": ");
  return i >= 0 ? first.slice(i + 2).trim() : undefined;
}

/* -------------------------------------------------------------- Markets */
export function Markets() {
  const core = useData<Core>("core");
  const [m, setM] = useState("us");
  if (!core.data) return <Loading error={core.error} />;
  const S = core.data.summary;
  const rows = ["nse", "us", "kr"].flatMap((k) => ["full", "pre2020"].map((f) => ({ k, f, t: S[`${k}/${f}`]?.tau_B_vs_C }))).filter((x) => x.t?.tau !== undefined);
  const NAME: Record<string, string> = { nse: "India (NSE)", us: "United States", kr: "South Korea (KOSPI)" };
  return (
    <>
      <Head eyebrow="Evidence · replication" title="Is this only an India effect?">The same models with the same settings, frozen after the NSE runs, on 47 large US stocks and 47 large Korean stocks.</Head>
      <Card title="Agreement between correlation fidelity and crisis behaviour, by market" sub="Kendall tau between the two rankings with a 95% seed-bootstrap interval. Near or below zero everywhere would mean the finding is general.">
        {rows.length ? <IntervalPlot zero domain={[-1, 1]} xFmt={(v) => v.toFixed(1)} rows={rows.map((x) => ({ label: `${NAME[x.k]} · ${x.f === "full" ? "in-sample" : "held-out"}`, color: x.k === "nse" ? "var(--s-ours)" : "var(--s-regime)", value: x.t.tau, sd: null, note: `95% interval ${fmt(x.t.lo)} to ${fmt(x.t.hi)}, ${x.t.n_generators} models` }))} /> : <Empty what="Multi-market replication" />}
      </Card>
      <Seg value={m} set={setM} options={[{ v: "us", label: NAME.us }, { v: "kr", label: NAME.kr }]} />
      <Benchmark market={m} key={m} />
    </>
  );
}

/* --------------------------------------------------------------- Checks */
export function Checks() {
  const core = useData<Core>("core");
  if (!core.data) return <Loading error={core.error} />;
  const d = core.data.dgp, c = core.data.contamination, b = core.data.bench["nse/full"];
  const gates = core.data.gates ?? {};
  const crisisGates = Object.entries(gates).filter(([k]) => k.startsWith("GATE_G"));
  const legacyGates = Object.entries(gates).filter(([k]) => !k.startsWith("GATE_G"));
  return (
    <>
      <Head eyebrow="Evidence · sanity checks" title="Can the measuring stick itself be trusted?">Before trusting a metric on models, we test it where the right answer is known, and we test how fragile it is to dirty data.</Head>
      <Card title="Check 1 · Simulated markets with a known answer" sub="We simulate artificial markets where we set the true crisis correlation boost ourselves (0%, 20%, 40%, 80%) and one with no regimes at all, then ask the benchmark to measure it.">
        {d ? (<>
          <Scatter height={300} xLabel="True boost built into the simulation, %" yLabel="Boost measured by the benchmark, %"
            points={d.map((r) => ({ x: r.true_boost_pct ?? r.population_boost ?? r.true_boost, y: r.rule_boost_pct ?? r.benchmark_boost ?? r.rule_boost, label: String(r.panel ?? r.name), color: "var(--accent)" }))} />
          <Records rows={d} />
        </>) : <Empty what="The controlled-simulation check" />}
      </Card>
      <Card title="Check 2 · What one bad price does" sub="We deliberately inject data errors into the clean NSE panel (a single 76% bad tick, 1% bad ticks, one unadjusted stock split) and re-run the models.">
        {c ? <Records rows={c} /> : <Empty what="The contamination study" />}
      </Card>
      <Card title="Check 3 · A different definition of crisis" sub="Rankings recomputed with the alternative rule (crisis = high 20-day volatility of the equal-weight portfolio).">
        {b?.alt_rule ? <BenchTable rows={b.alt_rule} refm={{}} metrics={["crisis_freq", "boost_pct", "cvar_ratio", "pc1_share_gap", "connect_gap"]} /> : <Empty what="The label-rule robustness table" />}
      </Card>
      <Card title="CrisisBench decision gates and post-hoc audits" sub="Preregistered decision gates (G1–G4) and v2 post-hoc evaluations generated directly by crisisbench.analysis.">
        {crisisGates.length ? crisisGates.map(([k, v]) => (<details key={k} style={{ marginBottom: 8 }}><summary style={{ cursor: "pointer", fontWeight: 600 }}>{k} <Verdict v={gateVerdict(v)} /></summary><pre style={{ whiteSpace: "pre-wrap", fontSize: 12, color: "var(--ink-2)" }}>{v}</pre></details>)) : <Empty what="CrisisBench gate evaluation" />}
      </Card>
      <Card title="Legacy six-gate statistical diagnostics" sub="Single-metric statistical tolerance tests (ACF, Frobenius, Kurtosis, Discriminator) from the initial six-gate evaluator. Distinguish these from the CrisisBench decision gates above.">
        {legacyGates.length ? legacyGates.map(([k, v]) => (<details key={k} style={{ marginBottom: 8 }}><summary style={{ cursor: "pointer", fontWeight: 600 }}>{k} <Verdict v={gateVerdict(v)} /></summary><pre style={{ whiteSpace: "pre-wrap", fontSize: 12, color: "var(--ink-2)" }}>{v}</pre></details>)) : <Empty what="Legacy gate evaluation" />}
      </Card>
    </>
  );
}
