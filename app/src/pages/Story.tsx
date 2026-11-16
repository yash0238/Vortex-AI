import { G } from "../glossary";
import type { Bench, Core, Market, Samples } from "../lib/data";
import { FAMILY, famColor, useData } from "../lib/data";
import { IntervalPlot, LineChart, Scatter, fmt } from "../ui/charts";
import { Card, Head, Loading, Stat, Term, Verdict } from "../ui/kit";

const go = (h: string) => { window.location.hash = h; };

/* ------------------------------------------------------------- Overview */
export function Overview() {
  const core = useData<Core>("core"), mk = useData<Record<string, Market>>("markets"), sm = useData<Samples>("samples");
  if (!core.data || !mk.data) return <Loading error={core.error ?? mk.error} />;
  const b: Bench | undefined = core.data.bench["nse/full"];
  const nse = mk.data.nse;
  const ref = b?.reference.metrics;
  const rows = b?.rows ?? [];
  const statics = rows.filter((r) => r.family_tag === "static");
  const ours = rows.find((r) => r.generator === "cgnsde2");
  const tau = core.data.summary["nse/full"]?.tau_B_vs_C;
  const real = sm.data?.["nse/full"]?.real;
  return (
    <>
      <section className="hero">
        <div>
          <div className="eyebrow">VORTEX-AI · B.E. major project · Fr. CRCE Mumbai</div>
          <h1 style={{ marginTop: 10 }}>A flight simulator for stock-market crashes.</h1>
          <p className="lede" style={{ marginTop: 14 }}>
            Pilots train on simulated storms because real ones are rare and dangerous. Banks and trading desks face the same problem with crashes:
            India's market has had only a few dozen stress episodes since 2010. VORTEX-AI generates realistic artificial market histories for 47 large NSE
            stocks, and then asks the question most research skips: <b>when the simulated market crashes, do the stocks fall together the way real stocks do?</b>
          </p>
          <div style={{ display: "flex", gap: 10, marginTop: 18, flexWrap: "wrap" }}>
            <button className="btn primary" onClick={() => go("#/benchmark")}>See the results</button>
            <button className="btn" onClick={() => go("#/how")}>How it works</button>
            <button className="btn" onClick={() => go("#/lab")}>Open the scenario lab</button>
          </div>
        </div>
        <Card title="What a crisis looks like in real data" sub="Equal-weight portfolio of the 47 stocks, 60-day paths. Red = windows the rule labels as crisis.">
          {real?.normal && real?.crisis ? (
            <LineChart height={250} yFmt={(v) => v.toFixed(0) + "%"} legend={false}
              series={[...real.normal.paths.slice(0, 10).map((y: number[], i: number) => ({ name: "calm " + i, color: "var(--s-static)", y, faint: true })),
                ...real.crisis.paths.slice(0, 10).map((y: number[], i: number) => ({ name: "crisis " + i, color: "var(--crisis)", y, faint: true, width: 1.4 }))]} />
          ) : <Loading />}
        </Card>
      </section>

      <div className="grid g4">
        <Card flat><Stat k="Real stocks, daily data" v={`${nse.n_assets} × ${nse.n_days.toLocaleString()}`} s={`${nse.first} to ${nse.last}, NSE India`} /></Card>
        <Card flat><Stat k="Crisis episodes found" v={nse.episodes.length} s={`${nse.windows.n_crisis} of ${nse.windows.n} windows are crisis windows`} /></Card>
        <Card flat><Stat k={<Term k="boost_pct">Real crisis correlation boost</Term>} v={ref ? `+${fmt(ref.boost_pct.real, 0)}%` : "…"} s={ref ? `95% band ${fmt(ref.boost_pct.lo, 0)}% to ${fmt(ref.boost_pct.hi, 0)}%` : ""} /></Card>
        <Card flat><Stat k="Models compared" v={rows.length || "…"} s="same data, same rules, 5 random seeds each" /></Card>
      </div>

      <Card title="The idea in one picture" sub={<>Each dot is one model. Left-right: the classic fidelity score everyone reports (<Term k="corr_fro" />, left is better). Up-down: the <Term k="boost_pct" /> the model actually produces. The shaded bands are where real data lies. A model can sit perfectly inside the vertical band and still be nowhere near the horizontal one.</>}>
        {b && ref ? (
          <>
            <Scatter height={380} xLabel="Correlation error (lower = closer to real)" yLabel="Crisis correlation boost, %"
              xBand={[0, ref.corr_fro.ceiling as number]} yBand={[ref.boost_pct.lo as number, ref.boost_pct.hi as number]} yFmt={(v) => v.toFixed(0) + "%"}
              points={rows.map((r) => ({ x: r.corr_fro, y: r.boost_pct, label: r.name, color: famColor(r.family_tag), bold: r.generator === "cgnsde2",
                note: `crisis windows produced: ${fmt((r.crisis_freq ?? 0) * 100, 1)}%` }))} />
            <div className="legend">{Object.entries(FAMILY).filter(([k]) => rows.some((r) => r.family_tag === k)).map(([k, f]) => (<span key={k}><i style={{ background: f.color }} />{f.label}</span>))}
              <span><i style={{ background: "var(--band)" }} />real-data band</span></div>
          </>) : <Loading />}
      </Card>

      <div className="grid g3">
        <Card title="1. The problem">
          <p>Risk models are tested on history, but history contains very few crashes. Synthetic data can fill the gap only if it reproduces what makes crashes dangerous: <b>everything falls at once</b>.</p>
        </Card>
        <Card title="2. What we found out about the usual tests">
          <p>The standard quality score compares average correlations. {statics.length > 0 && ref ? (<>Simple static models pass it (correlation error {statics.map((s) => fmt(s.corr_fro)).join(" and ")} against a noise ceiling of {fmt(ref.corr_fro.ceiling)}), yet produce a crisis boost of only {statics.map((s) => fmt(s.boost_pct, 1) + "%").join(" and ")} where real data shows {fmt(ref.boost_pct.real, 0)}%.</>) : "Results are loading."}</p>
        </Card>
        <Card title="3. What we built">
          <p><b>CrisisBench</b>: a frozen, reproducible test with a crisis rule applied identically to real and synthetic data, noise bands on every number, held-out crashes and a bank-style risk backtest. And <b>CG-NSDE</b>, our graph-based neural model, tested under exactly the same rules as every baseline. Our first version failed that test; the second, redesigned after we understood why, now gives a crisis boost of {ours ? <>{fmt(ours.boost_pct, 0)}%</> : "n/a"}.</p>
        </Card>
      </div>

      <Card title="Did the two kinds of test agree?" sub={<>We rank all models by correlation fidelity (family B) and by crisis behaviour (family C) and measure agreement with <Term k="kendall" />. +1 = same ranking, 0 = unrelated.</>}
        right={tau ? <span className="pill info">tau = {fmt(tau.tau)} [{fmt(tau.lo)}, {fmt(tau.hi)}] over {tau.n_generators} models</span> : undefined}>
        {b ? (<div className="grid g2">
          <div><h4 style={{ fontSize: 13, color: "var(--muted)", marginBottom: 6 }}>Correlation fidelity score (lower is better)</h4>
            <IntervalPlot zero rows={[...rows].sort((x, y) => x.score_B - y.score_B).map((r) => ({ label: r.name, color: famColor(r.family_tag), value: r.score_B, sd: r.score_B_sd, bold: r.generator === "cgnsde2" }))} ceiling={1} /></div>
          <div><h4 style={{ fontSize: 13, color: "var(--muted)", marginBottom: 6 }}>Crisis behaviour score (lower is better)</h4>
            <IntervalPlot zero rows={[...rows].sort((x, y) => x.score_C - y.score_C).map((r) => ({ label: r.name, color: famColor(r.family_tag), value: r.score_C, sd: r.score_C_sd, bold: r.generator === "cgnsde2" }))} ceiling={1} /></div>
        </div>) : <Loading />}
        <p style={{ fontSize: 13 }}>Scores are average errors in units of real sampling noise: below 1 means the model is as close to real data as a second sample of real data would be. See <a href="#/honest">what we can and cannot claim</a>.</p>
      </Card>

      <Card title="How this differs from what already exists">
        <div className="tbl-wrap"><table className="tbl"><thead><tr><th>Approach</th><th>Makes new scenarios?</th><th>Stocks linked to each other?</th><th>Knows crises are different?</th><th>Tested on unseen crashes?</th></tr></thead><tbody>
          <tr><td>Historical simulation (what most risk desks use)</td><td>No, replays the past</td><td>Yes (it is real data)</td><td>Only the crises that happened</td><td>Not possible</td></tr>
          <tr><td>GARCH / DCC models (classical finance)</td><td>Yes</td><td>Partly</td><td>Volatility yes, structure weakly</td><td>Here: yes</td></tr>
          <tr><td>TimeGAN, QuantGAN, TimeVAE (deep learning)</td><td>Yes</td><td>Implicitly</td><td>No explicit notion</td><td>Usually not; here: yes</td></tr>
          <tr className="ours"><td>VORTEX-AI: CG-NSDE + CrisisBench</td><td>Yes</td><td>Yes, a learned stock graph</td><td>Yes, by design, and measured</td><td>Yes: COVID-2020, 2022, 2024</td></tr>
        </tbody></table></div>
        <p style={{ fontSize: 13 }}>The last column is the point of the project: every row is put through the same held-out crash test, and the numbers on the following pages decide who actually delivers.</p>
      </Card>
      <p style={{ fontSize: 12, color: "var(--muted)" }}>Results built {core.data.built}. Protocol v{core.data.protocol.PROTOCOL_VERSION}, frozen {core.data.protocol.FROZEN_ON}. Research prototype: not investment advice.</p>
    </>
  );
}

/* --------------------------------------------------------- How it works */
export function How() {
  return (
    <>
      <Head eyebrow="How it works" title="From real prices to artificial crashes, step by step">
        No finance background needed. Hover any <Term k="window">dotted word</Term> for a one-line explanation.
      </Head>
      <Card title="The pipeline">
        <div className="flow">
          <div><small>Step 1</small><b>Real prices</b><span>15 years of daily prices for 47 NSE stocks, cleaned by fixed rules.</span></div>
          <div><small>Step 2</small><b>60-day windows</b><span>History is cut into overlapping 3-month windows of daily returns.</span></div>
          <div><small>Step 3</small><b>Crisis rule</b><span>One frozen rule marks each window calm or crisis from its own data.</span></div>
          <div><small>Step 4</small><b>Train generators</b><span>12 models learn from the same windows: 7 classical, 3 neural, and the two versions of ours.</span></div>
          <div><small>Step 5</small><b>Generate</b><span>Each model invents 2,000 new windows with no help from real data.</span></div>
          <div><small>Step 6</small><b>Judge</b><span>18 metrics in 4 families, each with a real-data noise band.</span></div>
          <div><small>Step 7</small><b>Stress it</b><span>Held-out crashes and a bank-style VaR backtest.</span></div>
        </div>
      </Card>
      <div className="grid g2">
        <Card title="Our model: CG-NSDE" sub="Contrastive Graph-Neural Stochastic Differential Equation">
          <div className="steps">
            <div className="step"><div><h3><Term k="gat">Graph attention</Term>: who moves with whom</h3><p>Each stock is a node, linked to its five most related stocks in that window. Attention layers learn which links matter and compress the whole market's wiring into one vector.</p></div></div>
            <div className="step"><div><h3><Term k="sde">Neural SDE</Term>: a hidden market mood that evolves</h3><p>A 64-number hidden state drifts and gets shaken by noise for 60 steps, one per day. The graph vector steers how it moves.</p></div></div>
            <div className="step"><div><h3><Term k="studentt">Fat-tailed emission</Term>: turning mood into returns</h3><p>Each day's hidden state sets the average, the stock-specific noise and a handful of common factors; returns are drawn from a fat-tailed distribution with one shared shock size per day, plus a volatility memory so wild days cluster.</p></div></div>
            <div className="step"><div><h3><Term k="supcon">Contrastive training</Term>: crisis and calm kept apart</h3><p>While training, crisis windows are pulled together and away from calm ones in the hidden space, and generated correlations are matched to the real ones separately for each regime.</p></div></div>
            <div className="step"><div><h3>Sampling with no real data</h3><p>After training, the hidden starting points of calm and crisis windows are summarised by two simple distributions. To generate, the model draws a regime, a starting point and noise. No real window is involved, so nothing can leak.</p></div></div>
          </div>
        </Card>
        <Card title="The judge: CrisisBench" sub="What makes the comparison fair">
          <div className="steps">
            <div className="step"><div><h3>One rule for real and fake</h3><p>The <Term k="crisis">crisis label</Term> is computed from each window's own returns with thresholds frozen on real history. Earlier versions of this project handed the label to the model; that inflated results and was removed.</p></div></div>
            <div className="step"><div><h3>Every number has a <Term k="noise_ceiling">noise band</Term></h3><p>Real history is re-sampled 500 times. If a model lands inside the band, it is as good as real data can tell.</p></div></div>
            <div className="step"><div><h3>Four families of metrics</h3><p>A: single-stock behaviour (fat tails, volatility clustering). B: average correlation fidelity. C: crisis behaviour. D: can a classifier tell real from fake?</p></div></div>
            <div className="step"><div><h3>Held-out crashes</h3><p>Models trained only on data up to 2019 are judged on COVID-2020, the 2022 shock and mid-2024.</p></div></div>
            <div className="step"><div><h3>Rules fixed before the runs</h3><p>Hypotheses, thresholds and pass rules were written down first. Failures stay in the tables.</p></div></div>
          </div>
        </Card>
      </div>
      <Card title="The crisis rule, exactly" sub="Applied to the last days of each 60-day window">
        <p className="mono" style={{ color: "var(--ink)", fontSize: 13.5 }}>score = 0.5 × (20-day average cross-stock dispersion ÷ its real 99th percentile) + 0.3 × (share of stocks down more than 5% in 5 days) + 0.2 × (share of stocks moving more than 9.5% today)</p>
        <p>A window is a crisis window if its score is at or above the 85th percentile of real fit-period windows. A second rule (20-day volatility of the equal-weight portfolio) is reported as a robustness check.</p>
      </Card>
    </>
  );
}

/* -------------------------------------------------------------- Glossary */
export function Glossary() {
  const fams: Record<string, string> = { A: "A · single-stock behaviour", B: "B · correlation fidelity", C: "C · crisis behaviour", D: "D · detectability" };
  const entries = Object.entries(G);
  return (
    <>
      <Head eyebrow="Reference" title="Glossary">Every term used in this dashboard, in plain language.</Head>
      {(["C", "B", "A", "D"] as const).map((f) => (
        <Card key={f} title={fams[f]}>
          <div className="grid g2">{entries.filter(([, e]) => e.family === f).map(([k, e]) => (<div key={k}><h3 style={{ fontSize: 14 }}>{e.name} <span className="pill">{e.better}</span></h3><p style={{ fontSize: 13.5 }}>{e.long}</p></div>))}</div>
        </Card>))}
      <Card title="Concepts">
        <div className="grid g2">{entries.filter(([, e]) => !e.family).map(([k, e]) => (<div key={k}><h3 style={{ fontSize: 14 }}>{e.name}</h3><p style={{ fontSize: 13.5 }}>{e.long}</p></div>))}</div>
      </Card>
    </>
  );
}

/* ---------------------------------------------------- Claims and limits */
export function Honest() {
  const core = useData<Core>("core");
  if (!core.data) return <Loading error={core.error} />;
  const hyp = core.data.hypotheses, P = core.data.protocol;
  return (
    <>
      <Head eyebrow="Claims and limits" title="What we can and cannot claim">
        The hypotheses and pass rules below were frozen on {P.FROZEN_ON}, before any model was scored. Verdicts are filled in by the analysis code, not by hand.
      </Head>
      <Card title="Preregistered hypotheses">
        <div className="tbl-wrap"><table className="tbl"><thead><tr><th>Hypothesis</th><th style={{ textAlign: "left" }}>Statement</th><th>Verdict</th><th style={{ textAlign: "left" }}>Evidence</th></tr></thead><tbody>
          {Object.entries(P.HYPOTHESES as Record<string, string>).map(([k, s]) => (<tr key={k}><td>{k}</td><td style={{ textAlign: "left", whiteSpace: "normal", minWidth: 280 }}>{s}</td><td><Verdict v={hyp?.[k]?.verdict} /></td><td style={{ textAlign: "left", whiteSpace: "normal", minWidth: 260 }}>{hyp?.[k]?.evidence ?? "not run yet"}</td></tr>))}
        </tbody></table></div>
      </Card>
      <Card title="Decision gates">
        <div className="tbl-wrap"><table className="tbl"><thead><tr><th>Gate</th><th style={{ textAlign: "left" }}>Rule</th><th>Result</th></tr></thead><tbody>
          {Object.entries(P.GATES as Record<string, string>).map(([k, s]) => (<tr key={k}><td>{k}</td><td style={{ textAlign: "left", whiteSpace: "normal" }}>{s}</td><td><Verdict v={hyp?.[k]?.verdict} /></td></tr>))}
        </tbody></table></div>
      </Card>
      <div className="grid g2">
        <Card title="Known limitations">
          <ul style={{ margin: 0, paddingLeft: 18, color: "var(--ink-2)", fontSize: 14 }}>
            <li><b>Survivorship bias.</b> The 47 stocks are companies that still exist today with a full history; failed or late-listed names are excluded.</li>
            <li><b>Few crises.</b> A few dozen episodes since 2010, and overlapping windows are not independent. That is why the real-data bands are wide and why small differences between models are not meaningful.</li>
            <li><b>No 2008.</b> Data starts in November 2010.</li>
            <li><b>Data source.</b> Prices come from Yahoo Finance via yfinance, an unofficial source; raw prices are not redistributed, only the rebuild recipe and checksums.</li>
            <li><b>CPU-scale neural baselines.</b> TimeGAN, QuantGAN and TimeVAE are our own PyTorch reimplementations trained with reduced budgets on a laptop CPU; their original authors' settings would train far longer.</li>
            <li><b>Daily data only.</b> No intraday microstructure, auctions or order books.</li>
            <li><b>Not investment advice.</b> This is a research prototype.</li>
          </ul>
        </Card>
        <Card title="What went wrong along the way (and is reported)">
          <ul style={{ margin: 0, paddingLeft: 18, color: "var(--ink-2)", fontSize: 14 }}>
            <li><b>The smoothing cheat.</b> An early neural-SDE version scored well on reconstruction by outputting smooth, predictable paths (day-to-day autocorrelation 0.99 against about 0 in real data).</li>
            <li><b>One bad tick.</b> A single wrong price (a 76% one-day "fall") inflated measured fat-tailedness by two orders of magnitude and invalidated early baseline numbers. Cleaning rules are now explicit and logged.</li>
            <li><b>Labels handed to the model.</b> Earlier evaluations labelled a generated window as "crisis" because a real crisis window was used to condition it. The reported crisis boost was therefore partly built in. Labels now come only from the generated returns.</li>
            <li><b>A name that outran the code.</b> The contrastive and graph losses were switched off in the previously reported model. They are now active, and ablations show what each part contributes.</li>
            <li><b>Lost artefacts.</b> The earlier cleaned dataset and trained checkpoints were not preserved, so every number here was regenerated from scratch under one protocol.</li>
          </ul>
        </Card>
      </div>
    </>
  );
}
