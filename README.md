# VORTEX-AI

A crisis-aware synthetic stock-market generator for Indian equities (CG-NSDE), a preregistered benchmark
(CrisisBench) that tests whether artificial market data behaves like real data **during crashes**, and a consumer app
built on it (**VORTEX Money**).

BE Major Project, Department of Computer Engineering, Fr. Conceicao Rodrigues College of Engineering, Mumbai.
Team: Yashovardhan Thopte, Alok Shukla, Aaryan Singh.

## Headline results (final, 18 November 2026)

Crisis score = mean error on crisis-dependence metrics in units of real sampling noise (lower is better), 20 generators,
including three strong classical risk models (GJR-EVT t-copula, asymmetric DCC, regime-switching t-copula) and the
official Diffusion-TS code (ICLR 2024), which generates almost no crisis windows.

| | India, all history | India, trained to 2019, judged on 2020-24 | Korea, held out | VaR/ES backtest (FZ0, generators) |
| --- | --- | --- | --- | --- |
| CG-NSDE v3, graph contagion (ours, post hoc) | 0.16 (rank 2) | **0.68 (rank 1)** | 0.44 (tie, rank 2) | **rank 1 of 18** |
| CG-NSDE v2, factor volatility (ours, post hoc) | **0.15 (rank 1)** | 0.74 (rank 2) | 0.50 | rank 2 |
| Strongest classical baseline | 0.19 (stationary bootstrap) | 0.87 (DCC-GARCH) | **0.43 (regime-switching t-copula)** | rank 3 (regime-switching t-copula) |
| CG-NSDE v1 (preregistered) | 1.22 (rank 12) | 1.21 (rank 13) | 0.79 | |

What failed, honestly: most preregistered hypotheses (correlation and crisis metrics rank generators similarly; the
metric-validity test failed narrowly; the preregistered v1 model failed), a confirmatory test of a label-free v3
(failed in Korea), and no model reaches the strength of the March 2020 crash (real boost about 101%, v3 25%). The v3
ablation shows its gain comes from contagion between stocks. Verdicts: `results/tables/hypotheses.json`; tables:
`results/tables/`; full story: `paper/main.tex` and `docs/STATE.md`.

## VORTEX Money (the consumer app)

"See your money's real future, crashes included." Goal planner with realistic crashes, crash rehearsal in rupees,
"diversification that disappears" meter, Fund X-ray for any Indian mutual fund (live AMFI data), "Will my money
last?" retirement simulator, behaviour-gap replay, tip reality check, plus streaks, missions, badges and a shareable
score card. Accounts and cross-device sync through its own backend.

```bash
cd app && npm install && npx vite build && cd ..     # first time
start_money.bat                                       # Windows: backend + app on http://127.0.0.1:8100/
python -m uvicorn money_api.app:app --port 8100       # macOS/Linux
docker build -f Dockerfile.money -t vortex-money .    # one container for app + API
```

Scenarios come from CG-NSDE v3: build `app/public/money/engine.json` with `python -m crisisbench.export_money --gen cgnsde3` (not in Git: it contains vendor-derived real returns). Backend tests:
`python -m pytest tests/test_money_api.py`. Brand guide: `docs/brand.md`. Educational tool, not investment advice.

## Research dashboard (for the project presentation)

```bash
cd app && npm install && npm run build && cd ..
start_dashboard.bat    # live API + dashboard on http://127.0.0.1:4173/
```

The dashboard reads `app/public/data/*.json` exported from the results; the API is only needed for live generation.

## Reproduce the experiments

```bash
python -m crisisbench.data --market nse         # download and clean a panel (also: us, kr, nse_ext)
python -m crisisbench.queue --list              # what is done and what is left
python -m crisisbench.queue --stage all --workers 5 --threads 2   # or run_experiments.bat / run_v3.bat
python -m crisisbench.analysis                  # tables, rank agreement, gates, hypotheses
python -m crisisbench.run_risk                  # VaR/ES backtest
python -m crisisbench.export_web                # refresh the dashboard data
python -m crisisbench.figures_extra             # fan chart, contagion graph, strategy sandbox
python -m crisisbench.paper_build               # figures, LaTeX tables, numbers, Overleaf zip
python -m crisisbench.step0                     # tail-cascade study on the French 49 industries (motivates v3)
```

The queue is resumable. Full-scale neural baselines on a free GPU: `cloud/vortex_fullscale.ipynb` and
`cloud/vortex_diffusion_rerun.ipynb` with `cloud/vortex_bundle.zip` (`python cloud/make_bundle.py`).

## Repository layout

| Path | What it is |
| --- | --- |
| `crisisbench/` | Frozen protocol, data, crisis labels, metrics, runner, queue, analysis, risk backtest, exports, paper build |
| `crisisbench/generators/` | 13 generators; ours: `cgnsde.py` (v1, preregistered), `cgnsde2.py` (v2), `cgnsde3.py` (v3) |
| `money_api/` | VORTEX Money backend (FastAPI: accounts, sync, fund-data cache; serves the app) |
| `app/` | React front ends: `index.html` research dashboard, `money.html` VORTEX Money |
| `api/server.py` | Research API: `/api/generate` (live scenarios) and the older market routes |
| `results/` | Per-run metrics, tables, gates, references, Step 0 study |
| `paper/` | IEEE paper source, references, generated figures, tables and numbers; Overleaf zip |
| `docs/` | `STATE.md` (working log), `step0_prereg.md`, `brand.md`, `archive/` (earlier plans) |
| `tests/` | `python -m pytest tests` |
| `models/`, `training/`, `eval/`, `baselines/`, `sandbox/`, `scripts/`, `src/`, root `*.py` | Earlier (v0) pipeline, still used by `api/main.py`; see `docs/archive/README_legacy.md` |

## Honesty notes

CG-NSDE v2 and v3 were designed after v1 failed; their design used only 2010-2019 data, they are excluded from every
preregistered verdict, and the paper reports them as post hoc. The panel has survivorship bias and uses free Yahoo
Finance data (raw prices are not redistributed). Research prototype, not investment advice.

## License

See `LICENSE`.
