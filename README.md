# VORTEX-AI

A crisis-aware synthetic stock-market generator for NIFTY-50, and a benchmark (CrisisBench) that tests whether
artificial market data behaves like real data **during crashes**, not only on average.

BE Major Project, Department of Computer Engineering, Fr. Conceicao Rodrigues College of Engineering, Mumbai.
Team: Yashovardhan Thopte, Alok Shukla, Aaryan Singh.

## Headline result (NSE, 47 stocks, 2010-2024)

| | Crisis score (lower is better), all history | Crisis score, trained to 2019, judged on 2020-24 |
| --- | --- | --- |
| CG-NSDE v2 (ours) | 0.15, rank 1 of 12 | 0.74, rank 1 of 12 |
| Best classical baseline | 0.19 (stationary bootstrap) | 0.87 (DCC-GARCH) |
| CG-NSDE v1 (preregistered) | 1.22, rank 10 | 1.21, rank 10 |

Real stocks move together 29.3% more in crisis windows than in calm ones; CG-NSDE v2 produces 29.2%, simple models
about 2-3%. No model, ours included, reaches the strength of the March 2020 crash (real about 101%, ours 22%).
Full tables: `results/tables/T1_nse_full.md`, `results/tables/T1_nse_pre2020.md`; verdicts on the preregistered
hypotheses: `results/tables/hypotheses.json`. Some experiments are still running (see Status).

## Show the dashboard

Requirements: Python 3.11+ with the packages in `requirements.txt`, Node.js 20+.

```
cd app
npm install          # first time only
npm run build        # first time, and after code changes
cd ..
start_dashboard.bat  # Windows: starts the live API and the dashboard, opens http://127.0.0.1:4173/
```

On macOS/Linux run the two servers by hand:

```
python -m uvicorn api.server:app --host 127.0.0.1 --port 8000
cd app && npx vite preview --port 4173 --host 127.0.0.1
```

The dashboard reads only `app/public/data/*.json`, exported from the result files, so it works without the API.
The API is needed only for the "Generate fresh scenarios (live)" button, which also needs a trained checkpoint in
`artifacts/models/nse/cgnsde2/` (created by the experiment queue; not stored in Git because of size).

## Reproduce the experiments

```
python -m crisisbench.data --market nse         # download and clean a panel (also: us, kr, nse_ext)
python -m crisisbench.queue --list              # what is done and what is left
python -m crisisbench.queue --stage all --workers 5 --threads 2   # or double-click run_experiments.bat
python -m crisisbench.analysis                  # tables, rank agreement, gates, hypotheses
python -m crisisbench.run_risk                  # VaR/ES backtest (needs the risk pools)
python -m crisisbench.export_web                # refresh the dashboard data
python -m crisisbench.paper_build               # figures, LaTeX tables, Overleaf zip
```

The queue is resumable: a finished run is skipped, so it can be stopped and restarted at any time. Full-scale neural
baselines on a free GPU: `cloud/vortex_fullscale.ipynb` with `cloud/vortex_bundle.zip` (build it with
`python cloud/make_bundle.py`).

## Repository layout

| Path | What it is |
| --- | --- |
| `crisisbench/` | Current system: frozen protocol, data, crisis labels, metrics, runner, queue, analysis, risk backtest, exports |
| `crisisbench/generators/` | The 12 generators; `cgnsde.py` (v1, preregistered) and `cgnsde2.py` (v2, post hoc) are ours |
| `api/server.py` | FastAPI: `/api/generate` (live scenarios), mounts the older market routes from `api/main.py` |
| `app/` | React dashboard |
| `results/` | Per-run metrics (`runs/`), tables, gates, real-data references |
| `data/panels/` | Cleaned return panels with audit logs (raw vendor files are not stored) |
| `paper/` | IEEE paper source, references, generated figures and tables |
| `tests/` | Unit tests (`python -m pytest tests`) |
| `docs/STATE.md` | Working state and next steps |
| `docs/archive/` | Earlier plans, handoffs and the previous README |
| `models/`, `training/`, `eval/`, `baselines/`, `sandbox/`, `scripts/`, `src/`, root `*.py` | Earlier (v0) pipeline, kept because `api/main.py` and the market explorer and mutual-fund pages still use parts of it. Its instructions are in `docs/archive/README_legacy.md` |

## Status (17 November 2026)

Done: data, benchmark, all 12 generators on NSE (5 seeds, two training periods), v2 risk refits, contamination and
simulation checks, dashboard, paper draft. Running: v2 ablations, v2 on the US, Korean and 2025-26 panels, remaining
baselines. To do: risk backtest and final analysis, full-scale neural baselines on GPU, final paper text.

## Honesty notes

CG-NSDE v2 was designed after v1 failed the benchmark; its design used only 2010-2019 data, and the paper reports it
as a post hoc model next to the preregistered v1. The panel has survivorship bias (current constituents only) and
uses free Yahoo Finance data. Research prototype, not investment advice.

## License

See `LICENSE`.
