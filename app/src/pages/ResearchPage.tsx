import { Activity, ArrowUpRight, BookOpen, CircleAlert, FlaskConical, GitBranch, ShieldAlert } from "lucide-react";

const evidence = [
  { metric: "Median excess kurtosis", result: "4.70", target: "> 3", status: "Pass" },
  { metric: "Return lag-1 ACF", result: "0.0024", target: "Near zero", status: "Pass" },
  { metric: "Correlation Frobenius error", result: "4.82", target: "< 2.5", status: "Fail" },
  { metric: "Discriminative score", result: "0.766", target: "< 0.60", status: "Fail" },
  { metric: "Crisis correlation boost", result: "65.0%", target: "> 50%", status: "Pass" },
  { metric: "Crisis / normal CVaR ratio", result: "2.16x", target: "> 3x", status: "Fail" },
  { metric: "ACF of squared returns", result: "-0.014", target: "Positive", status: "Fail" },
];

const literature = [
  {
    title: "Beyond Visual Realism: Toward Reliable Financial Time Series Generation",
    meta: "Zhang et al. · ICASSP 2026",
    takeaway: "Shows that visual/stylized-fact similarity can still fail downstream backtests; a practical generator needs task-level validation.",
    href: "https://arxiv.org/abs/2601.12990",
  },
  {
    title: "Generative Diffusion Models of Stochastic Graph Signals",
    meta: "Uslu et al. · 2026 preprint, under review",
    takeaway: "Directly overlaps graph-conditioned generation and stock forecasting. VORTEX must distinguish itself through its NSE risk/stress-test setting and validated results, not by claiming graph generation is new.",
    href: "https://arxiv.org/abs/2607.06833",
  },
  {
    title: "Deep-MKV-TS: Path-Dependent McKean–Vlasov Control for Financial Time Series Generation",
    meta: "El Boustany et al. · 2026 preprint",
    takeaway: "A recent scenario-generation alternative that evaluates held-out intraday index futures and downstream risk-controlled exposure.",
    href: "https://arxiv.org/abs/2608.19394",
  },
  {
    title: "HGAN-SDEs: Hermite-Guided Adversarial Training for Neural SDEs",
    meta: "Xu et al. · 2025/2026 preprint",
    takeaway: "Overlaps Neural-SDE generation and focuses on an efficient path-level discriminator; it is a relevant baseline, not a domain-complete NSE stress-testing system.",
    href: "https://arxiv.org/abs/2512.20272",
  },
  {
    title: "Time-Causal VAE: Robust Financial Time Series Generator",
    meta: "Acciaio, Eckstein, Hou · 2024 preprint",
    takeaway: "Uses causal transport and evaluates downstream financial optimization, a key comparison for scenario usefulness.",
    href: "https://arxiv.org/abs/2411.02947",
  },
];

export default function ResearchPage() {
  return (
    <article className="research-page space-y-8">
      <header className="research-heading">
        <p className="text-xs uppercase tracking-widest text-gray-500">Research brief / VORTEX-AI</p>
        <h1 className="mt-2 text-3xl font-semibold">Stress scenarios for a connected market</h1>
        <p className="mt-3 max-w-4xl text-sm leading-6 text-gray-400">
          The project hypothesis is that learning cross-stock contagion, regime structure, and NSE-specific constraints together can produce more useful synthetic crash scenarios than fitting each return series independently. It is a research claim to test, not a result already established.
        </p>
      </header>

      <section className="research-status" aria-label="Research readiness">
        <div className="flex items-start gap-3">
          <CircleAlert className="mt-0.5 shrink-0 text-amber-300" size={20} />
          <div>
            <h2 className="font-semibold text-amber-100">Research prototype; not industry-ready</h2>
            <p className="mt-1 text-sm leading-6 text-amber-100/75">
              Saved experiments support a promising contagion signal, but several fidelity gates fail. No compatible trained CG-NSDE generator checkpoint is currently loaded; the live data tools work, while scenario inference is deliberately gated.
            </p>
          </div>
        </div>
      </section>

      <section aria-labelledby="method-title">
        <div className="research-section-heading">
          <div>
            <p className="text-xs uppercase tracking-widest text-gray-500">Candidate contribution</p>
            <h2 id="method-title" className="mt-1 text-xl font-semibold">What VORTEX is trying to combine</h2>
          </div>
          <FlaskConical size={20} className="text-sky-300" />
        </div>
        <div className="research-method-grid">
          <div className="research-method-item">
            <GitBranch size={19} className="text-sky-300" />
            <h3>Dynamic asset graph</h3>
            <p>GAT representations condition cross-asset dynamics, with crisis contagion evaluated separately from normal periods.</p>
          </div>
          <div className="research-method-item">
            <Activity size={19} className="text-lime-300" />
            <h3>Regime-aware latent SDE</h3>
            <p>A stochastic path generator is trained alongside a contrastive regime objective, then should be judged on held-out paths.</p>
          </div>
          <div className="research-method-item">
            <ShieldAlert size={19} className="text-orange-300" />
            <h3>NSE risk constraints</h3>
            <p>Circuit-filter behavior and crisis-sensitive risk measures target Indian-market stress testing rather than order execution.</p>
          </div>
        </div>
        <p className="mt-3 text-xs text-gray-500">Novelty is not proven by this dashboard. It needs a systematic related-work audit and controlled, reproducible comparisons.</p>
      </section>

      <section aria-labelledby="comparison-title">
        <div className="research-section-heading">
          <div>
            <p className="text-xs uppercase tracking-widest text-gray-500">Scope comparison</p>
            <h2 id="comparison-title" className="mt-1 text-xl font-semibold">Research workbench vs market platforms</h2>
          </div>
          <BookOpen size={20} className="text-sky-300" />
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] border-collapse text-left text-sm">
            <thead className="text-xs uppercase text-gray-500">
              <tr className="border-b border-border">
                <th className="px-3 py-3">Capability</th>
                <th className="px-3 py-3">Broker / charting apps</th>
                <th className="px-3 py-3">Generative research methods</th>
                <th className="px-3 py-3">VORTEX-AI</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-border/70">
                <th className="px-3 py-3 font-medium">Instrument search, live quotes, charts</th>
                <td className="px-3 py-3 text-gray-400">Core workflow; often exchange/broker connected</td>
                <td className="px-3 py-3 text-gray-400">Usually outside the model paper</td>
                <td className="px-3 py-3 text-gray-300">Prototype search/watchlist; unofficial Yahoo data, not licensed streaming</td>
              </tr>
              <tr className="border-b border-border/70">
                <th className="px-3 py-3 font-medium">Synthetic path generation</th>
                <td className="px-3 py-3 text-gray-400">Not the main broker workflow</td>
                <td className="px-3 py-3 text-gray-400">TimeGAN, TC-VAE, HGAN-SDE, graph diffusion and others</td>
                <td className="px-3 py-3 text-gray-300">Intended CG-NSDE generator; compatible trained weights absent</td>
              </tr>
              <tr className="border-b border-border/70">
                <th className="px-3 py-3 font-medium">Graph + regime + NSE stress testing</th>
                <td className="px-3 py-3 text-gray-400">Charting and execution tools, not this research claim</td>
                <td className="px-3 py-3 text-gray-400">Recent graph and SDE work overlaps components</td>
                <td className="px-3 py-3 text-gray-300">Candidate contribution; current saved gates are mixed</td>
              </tr>
              <tr>
                <th className="px-3 py-3 font-medium">Portfolio/orders/alerts</th>
                <td className="px-3 py-3 text-gray-400">Integrated, regulated workflows</td>
                <td className="px-3 py-3 text-gray-400">Normally not provided</td>
                <td className="px-3 py-3 text-gray-300">Not implemented; no broker integration</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section aria-labelledby="evidence-title">
        <div className="research-section-heading">
          <div>
            <p className="text-xs uppercase tracking-widest text-gray-500">Saved experiments / Stage 7.3b</p>
            <h2 id="evidence-title" className="mt-1 text-xl font-semibold">Evidence against the current gates</h2>
          </div>
          <a className="inline-flex items-center gap-1 text-xs text-sky-300" href="https://github.com/yash0238/Vortex-AI/blob/main/results/ablation_table.csv" target="_blank" rel="noreferrer">
            Source table <ArrowUpRight size={14} />
          </a>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[620px] border-collapse text-left text-sm">
            <thead className="text-xs uppercase text-gray-500">
              <tr className="border-b border-border">
                <th className="px-3 py-3">Metric</th><th className="px-3 py-3">Three-seed mean</th><th className="px-3 py-3">Target</th><th className="px-3 py-3">Gate</th>
              </tr>
            </thead>
            <tbody>
              {evidence.map((item) => (
                <tr key={item.metric} className="border-b border-border/70 last:border-0">
                  <th className="px-3 py-3 font-medium">{item.metric}</th>
                  <td className="px-3 py-3 tabular-nums">{item.result}</td>
                  <td className="px-3 py-3 text-gray-400">{item.target}</td>
                  <td className={`px-3 py-3 font-medium ${item.status === "Pass" ? "text-lime-300" : "text-rose-300"}`}>{item.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-gray-500">Saved three-seed artifacts, not results generated by the current untrained API process. Correlation, discriminative, tail-risk, and volatility-clustering gates still need work.</p>
      </section>

      <section aria-labelledby="literature-title">
        <div className="research-section-heading">
          <div>
            <p className="text-xs uppercase tracking-widest text-gray-500">Research scan · checked 2026-10-05</p>
            <h2 id="literature-title" className="mt-1 text-xl font-semibold">Relevant work to compare against</h2>
          </div>
          <a className="inline-flex items-center gap-1 text-xs text-sky-300" href="https://export.arxiv.org/api/query?search_query=all:%22financial%20time%20series%20generation%22&start=0&max_results=10" target="_blank" rel="noreferrer">
            arXiv query <ArrowUpRight size={14} />
          </a>
        </div>
        <div className="research-literature-list">
          {literature.map((paper) => (
            <a className="research-literature-row" href={paper.href} target="_blank" rel="noreferrer" key={paper.title}>
              <span className="research-literature-copy">
                <span className="research-literature-title">{paper.title}</span>
                <span className="research-literature-meta">{paper.meta}</span>
                <span className="research-literature-takeaway">{paper.takeaway}</span>
              </span>
              <ArrowUpRight size={17} aria-hidden="true" />
            </a>
          ))}
        </div>
      </section>

      <section className="research-source-band" aria-labelledby="source-title">
        <div>
          <p className="text-xs uppercase tracking-widest text-gray-500">Data and product boundary</p>
          <h2 id="source-title" className="mt-1 text-lg font-semibold">Live availability is not exchange-grade streaming</h2>
        </div>
        <ul>
          <li><strong>Stock quotes/history:</strong> Yahoo Finance through yfinance, suitable for research/demo use; timestamps are shown and data may be delayed. yfinance itself describes its interface as intended for personal use.</li>
          <li><strong>Mutual funds:</strong> MFAPI NAV history backed by AMFI-style scheme records; NAVs update daily, not tick-by-tick.</li>
          <li><strong>Exchange streaming:</strong> NSE describes a streaming market-data experience. Production redistribution/real-time use needs an authorized licensed feed; that is not configured here.</li>
        </ul>
      </section>
    </article>
  );
}