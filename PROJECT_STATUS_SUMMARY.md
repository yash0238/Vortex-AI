# VORTEX-AI Project Status and Handoff

**Status date:** 2026-10-05

**Purpose:** Record the branch integration, verified run state, dashboard behavior, research artifacts, and remaining work.

## Executive Summary

The local `main` now contains the history from local `main`, fetched `origin/main`, `complete_implementation`, and `fix/stabilize`. The existing `feature/gat-and-modules` work was already an ancestor of local `main`. Safety refs `main_backup` and `local_main_backup` remain at the pre-merge local snapshot. The original local and remote main histories had no common ancestor, so they were joined with a merge commit; neither history was reset or force-pushed.

The React dashboard builds and runs, FastAPI starts against the checked-in data, and the new Mutual Fund Watchlist loads five live NAV series. The backend now derives missing final-day window labels and the small node-feature slice it actually needs, without creating a multi-hundred-megabyte cache. Training data alignment was corrected to use all 50 stocks and the intended approximately 15% crisis rate.

Important limit: there is **no trained CG-NSDE generator checkpoint** in this workspace. The two checked-in `.pt` files are older classifier checkpoints, not compatible latent-SDE generator weights. The UI and API now report/gate model inference rather than presenting random initialization as valid scenario or evaluation results. Stored research results are available, but a full generator training run was not performed here.

## Branch Integration

- `origin/main` was fetched and merged into local `main`. The local and remote roots were unrelated; both commit histories remain parents of the merge.
- `origin/complete_implementation` was merged. Its unique `sandbox/mf_case_study.py` remains in the tree; overlapping model/training files use the newer stabilized main implementations.
- `origin/fix/stabilize` was merged cleanly, bringing the Stage 7.3b results, three-seed tables, ablations, and convergence notes.
- `origin/feature/gat-and-modules` was already contained in the original local main history.
- Current main retains all branch histories. Conflict resolution selected one coherent working version for same-path files; alternate versions remain accessible from their original commits and backup refs.
- `.gitattributes` now applies Git LFS only to the generated `data/raw/window_node_features.npy`. The existing checked-in CSV/NPY files are regular Git blobs; applying LFS filters to them caused merge/checkout warnings.

## Project Layout

The repository is broad because it contains the app, model experiments, benchmarks, diagnostics, and generated research results. Paths were not mass-renamed because scripts and imports depend on them; these are the main ownership areas:

| Path | Role |
| --- | --- |
| `api/` | FastAPI data and model endpoints |
| `app/` | React + TypeScript + Vite end-user dashboard |
| `data/` | NSE data download, preprocessing, graph construction, node features, raw inputs |
| `models/` | GAT encoder, latent SDE, contrastive head, decoder |
| `training/` | VORTEX model, losses, trainer, configuration |
| `eval/` | Statistical, discriminative, and contagion evaluation |
| `sandbox/` | Circuit filtering and mutual-fund case-study utilities |
| `baselines/` | TimeGAN/QuantGAN runners, baseline tables and outputs |
| `scripts/` | Diagnostics, preprocessing helpers, staged experiment runners |
| `results/`, `runs/` | Saved evaluation JSON/CSV, figures, experiment logs |

## Data and Model Flow

1. `data/raw/nifty50_close.csv` and `nifty50_volume.csv` feed the return and feature pipeline.
2. `returns.npy` contains 3,700 daily rows across 50 stocks; `windows.npy` contains 3,641 windows of 60 days; `adj_matrices.npy` is 3,641 x 50 x 50.
3. Daily regime labels contain 555 crisis days of 3,700 (15%). If `window_regimes_v2.npy` is absent, the API and trainer derive each window's label from its final day. This yields 534 crisis windows of 3,641 (14.67%). The legacy `window_regimes.npy` is max-pooled and labels 73.4% of windows as crisis, so it is only a last-resort legacy fallback.
4. When `window_node_features.npy` is absent, the API computes compact features in memory and retains the final-day `(windows, stocks, features)` tensor needed for its endpoints. It does not persist the large four-dimensional feature array.
5. The intended research architecture is a dynamic GAT, graph-conditioned latent Neural SDE, supervised contrastive regime head, decoder, and NSE circuit constraints.

The live backend was verified with the real files: 3,641 windows, 50 stocks, 60 days per window, and a 14.67% crisis-window ratio. `GET /api/node-features?window_idx=0` returns 50 stock rows.

## Dashboard

The app contains Overview, Data Pipeline, GAT Encoder, Neural SDE, Scenario Generation, Mutual Funds, Evaluation, and Training views. A typed Axios API client now connects those pages to FastAPI. Desktop navigation is visible with the correct main-content offset; mobile navigation uses a drawer. The 390px mobile and 1440px desktop checks show no page-width overflow.

The Mutual Fund Watchlist retrieves NAV histories from the public MFAPI service at page load and refresh; it needs an internet connection but no API key. It supports 1Y/3Y/5Y/MAX chart ranges and scheme visibility toggles. The chart rebases each available series to 100. Table calculations are:

- 1Y return: latest NAV divided by the closest NAV at or before one year earlier, minus one.
- 1Y volatility: sample standard deviation of daily NAV returns over up to 252 observations, annualized by `sqrt(252)`.
- Since-launch CAGR: annualized change from the oldest available NAV observation to the latest.

Latest values returned by the source (all dated 2026-10-01):

| Requested fund | MFAPI scheme code | Latest NAV | 1Y return | 1Y annualized volatility | CAGR from earliest available NAV |
| --- | ---: | ---: | ---: | ---: | ---: |
| Edelweiss Mid Cap Fund - Direct Growth | 140228 | 121.837 | 4.47% | 15.16% | 19.53% |
| Invesco India Mid Cap Fund - Direct Growth | 120403 | 229.390 | 6.63% | 17.08% | 20.49% |
| Mirae Asset Midcap Fund - Direct Growth | 147445 | 41.387 | 4.73% | 15.51% | 22.03% |
| HSBC Small Cap Fund | 151130 | 99.559 | 12.41% | 17.52% | 18.55% |
| Bandhan Small Cap Fund - Direct Growth | 147946 | 56.085 | 10.49% | 15.87% | 29.88% |

The MFAPI/AMFI catalog reports the HSBC active series simply as `HSBC Small Cap Fund` and omits its plan label; the dashboard does not claim a plan for that row. NAV histories can be revised or delayed by the upstream provider. These are historical NAV calculations, not investment advice or forecasts.

## Saved Research Results

`results/ablation_table.csv` contains the three-seed Stage 7.3b summary. The `final-mean` row reports:

| Metric | Saved mean | Masterplan target | Status |
| --- | ---: | ---: | --- |
| Return lag-1 ACF | 0.0024 | Near zero | Pass |
| Median excess kurtosis | 4.70 | > 3 | Pass |
| Correlation Frobenius error | 4.82 | < 2.5 | Fail |
| Correlation MAE | 0.085 | Not specified as a masterplan gate | Diagnostic only |
| Discriminative score | 0.766 | < 0.60 | Fail |
| Crisis correlation boost | 65.0% | > 50% | Pass |
| Crisis/normal CVaR ratio | 2.16x | > 3x | Fail |
| ACF of squared returns | -0.014 | Positive volatility clustering | Fail |

The run therefore clears intermediate convergence gates (correlation error below 10, discriminative score below 0.8, and crisis boost above 15%), but it does **not** meet all masterplan publication targets. Results are saved experiment artifacts, not newly generated in this session.

`results/baseline_table.csv` records the fallback iid-Gaussian and static-t-PPCA comparisons. The same table marks TimeGAN and QuantGAN as blocked because their external submodules/runners were empty or unavailable in the Python 3.13 environment. Do not present them as reproduced baselines.

Research notes and interpretation are in `CONVERGENCE_PLAN.md`, `README.md`, `PRESENTATION.md`, `IMPLEMENTATION_PLAN.md`, and the stage evaluation artifacts. They discuss the GAT, neural-SDE, TimeGAN/QuantGAN, contrastive learning, NSE constraints, ablations, and why unconstrained diffusion, endpoint-only reconstruction, and early contrastive loss can undermine fidelity. Any paper claims still need primary-source citation verification before publication.

## Run Locally

The current workspace environment is `.venv` (Python 3.13.7), and the frontend dependencies are installed under `app/`.

Start the API in one PowerShell terminal:

```powershell
& '.\.venv\Scripts\python.exe' -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Start the dashboard in a second terminal:

```powershell
Push-Location app
npm ci
npm run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173`. API health is at `http://127.0.0.1:8000/api/health`; interactive API docs are at `http://127.0.0.1:8000/api/docs`.

`npm ci` must run from `app/`, not the repository root. The model runtime depends on compatible PyTorch/PyG wheels; the `.venv` used for this run has `torch`, `torch-geometric`, `torchsde`, `pytorch-lightning`, and the API/evaluation dependencies installed.

## Verification Performed

- `tests/test_smoke.py`: all smoke checks passed (imports, losses, GAT/SDE shapes, evaluation helpers, circuit bands).
- `npm run build` from `app/`: passed. Vite reports a non-blocking JavaScript chunk-size advisory (~516 kB minified).
- FastAPI health, stats, returns, and node-feature endpoints: passed against checked-in data.
- Real-data VORTEX forward pass: finite `(1, 60, 50)`, `(1, 50, 50)`, `(1, 32)` outputs.
- Training simulation endpoint: one epoch, batch size 2, finite loss `0.6264`; this is an in-memory simulation and saves no checkpoint.
- Browser: desktop and mobile layout, five live NAV rows, MAX range toggle, backend status, and no browser page errors verified.

## Remaining Work

- Train and save a compatible full `VORTEXModel` checkpoint. The available `best_model.pt` and `best_gat_model.pt` are classifier checkpoints, not a generator checkpoint. The API reports this and blocks model inference, scenario generation, and live model evaluation rather than reporting random-weight output.
- The training script warns that `data/raw/scaler.npz` is absent and falls back to raw returns. Produce a train-fitted scaler before a research-quality run, verify the train/validation/test split, and confirm the configured device/batch size.
- Rerun full validation on the trained checkpoint, including three seeds, ACF of squared returns, correlation fidelity, discriminative score, contagion boost, and CVaR. Current saved results fail some masterplan gates as detailed above.
- Repair/fetch the TimeGAN and QuantGAN submodules and run comparable baselines before claiming a full baseline comparison.
- The old `sandbox/mf_case_study.py` uses mock fund portfolios; the end-user dashboard instead uses live NAV series and does not infer current portfolio holdings.