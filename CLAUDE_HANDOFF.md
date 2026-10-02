# CLAUDE HANDOFF — VORTEX-AI (CG-NSDE) Full Context + Plan Request

> Purpose: give Claude **entire context** about this project so Claude can generate a **step-by-step implementation plan and methodology** that instructs the downstream coding agent (Muse Spark via OpenCode, win32/PowerShell) to produce the BEST output and bring the project to maximum potential.
> Downstream agent rules: short/concise factual output, `file_path:line_number` references, verify by execution, prefer editing existing files, never create docs unless requested, use specialized tools (Read/Edit/Glob/Grep) not bash for file ops, use `C:\Users\myind\Documents\GitHub\vortex-ai\Vortex-AI` as repo root.

## 1. Verdict: NOT at maximum potential

- `IMPLEMENTATION_SUMMARY.md:7` claims "13/13 Tasks Complete", but `PRESENTATION.md:187,337` correctly says "20% / Phases 0-2 complete".
- Verified audit (2026-09-25) finds **fatal runtime blockers + placeholders + missing deps + zero tests + large uncommitted state**. End-to-end `train_vortex.py` and `evaluate_vortex.py` **cannot run as-is**.
- Do NOT trust summary claims. Trust the file:line evidence in §7.

## 2. Project identity

- **Name:** VORTEX-AI — Contrastive Graph-Neural Stochastic Differential Equations (CG-NSDE) for NSE Stress Testing.
- **Problem:** Generate realistic synthetic financial scenarios capturing temporal dynamics + structural relationships between NIFTY-50 assets during crisis regimes, given limited historical crisis data (~3-4 crashes in 15 years).
- **Analogy:** Flight simulator for markets. Learn physics of normal/crisis, generate millions of scenarios for stress-testing.
- **Core novelty claim:** First to combine Dynamic GAT + graph-conditioned Neural SDE + Supervised Contrastive regime head for Indian markets.
- **Masterplan refs:** `VORTEX-AI_CG-NSDE_Masterplan.pdf` Part C.3/D.3/D.6/D.7/D.8/E/H/K, `gemini major project.pdf` (cited in docs, not in repo — treat as missing).
- **Targets (masterplan Part F):** Kurtosis >3.0, ACF(r²) >0.05, Corr error <2.5 Frobenius, Discriminative <0.60 (closer to 0.5 better), Crisis boost >50%, CVaR ratio >3x.
- **Measured (README, CUDA run, unverified):** Baseline LSTM val-loss 0.8463 / MSE 0.0694 / ROC-AUC 0.5621 / F1 0.8465; GAT val-loss 0.6121 / MSE 0.0829 / ROC-AUC 0.8326 / F1 0.8465 via `evaluate.py` + `models/checkpoints/best_gat_model.pt`.
- **Roadmap:** ICAIF 2026 (Aug 2026 deadline, 8-page), AAAI 2027, ICLR 2027 workshop, SSRN, Journal of Financial Data Science, Quantitative Finance.
- **SDG:** 8 (resilience), 9 (explainable AI infra), 16 (transparent regulation).
- **Ethics:** Research prototype, NOT trading advice. Survivorship bias acknowledged.

## 3. Repo state (2026-09-25)

- **Root:** `C:\Users\myind\Documents\GitHub\vortex-ai\Vortex-AI` is the git repo. Parent `C:\Users\myind\Documents\GitHub\vortex-ai` is NOT a repo.
- **Remote:** `https://github.com/yash0238/Vortex-AI.git`, HEAD `main`.
- **Branches (all tracked locally after 2026-09-25 work):**
  - `main` (586e68c) ↔ `origin/main`, 0 ahead/behind, but dirty worktree (see below).
  - `feature/gat-and-modules` diverged: ahead 5 / behind 5 vs origin (same messages, different SHAs — history rewritten).
  - `complete_implementation` (024f251) newly tracked locally from `origin/complete_implementation`. Histories are **unrelated**: `main` root `f39ff21`, `complete` root `352f190`, `git merge-base` empty. Union-add done (not merge): 7 files checked out from complete into main worktree.
  - Stale locals with deleted upstreams: `aaryxnblondead-implementation-summary`, `feature-gat-trainer` ([gone]).
- **Dirty worktree on `main`:**
  - Staged (A): `IMPLEMENTATION_SUMMARY.md`, `QUICKSTART.md`, `eval/discriminative.py`, `evaluate_vortex.py`, `train_vortex.py`, `training/vortex_model.py`, `visualize_results.py` (union-add from complete).
  - Modified (M): `.gitignore`, `models/contrastive.py`, `models/gat_encoder.py`, `models/neural_sde.py`, `sandbox/circuit_filter.py`, `src/model.py`, `training/config.yaml`, `training/losses.py`, `training/trainer.py`.
  - Untracked (??): `IMPLEMENTATION_PLAN.md`, `PRESENTATION.md`, `api/`, `app.py`, `app/` (~1000+ lines, loss risk — commit or stash).
- **Data on disk (generated, mostly git-ignored):** `data/raw/windows.npy` 43MB, `volume_windows.npy` 43MB, `window_node_features.npy` 262MB, `adj_matrices.npy` 36MB, `returns.npy`, `regimes.npy`, `window_regimes.npy`, `window_regimes_v2.npy`, `nifty50_close.csv` 3.1MB, `nifty50_volume.csv` 1.7MB. `.gitignore:223-224` ignores `data/raw/*.npy,*.csv` by design ("Exclude generated data artifacts").
- **Checkpoints:** only `models/checkpoints/best_model.pt` 255KB + `best_gat_model.pt` 135KB (Aug-28 baseline era). **No `*.ckpt`** for `VORTEXModel.load_from_checkpoint()` used in `evaluate_vortex.py:92`, `visualize_results.py:344`.
- **Tests:** none. No `tests/`, `test_*.py`.
- **Env:** win32, PowerShell 5.1, Python 3.13 via `.venv`, torch 2.13+cu126 claimed but CUDA unavailable in current env (use CPU). Node frontend in `app/`.

## 4. Full file inventory (42 .py + configs + docs + data)

```
Vortex-AI/
  app.py (Streamlit 965 lines, 8 pages)
  train.py → training/trainer.py
  train_vortex.py (309 lines, VortexDataset + CLI)
  evaluate.py (held-out GAT), evaluate_vortex.py (282 lines, BROKEN imports)
  check_pipeline.py, inspect_data.py, eda_dashboard.py, eda_visualizations.py, visualize_results.py (loss-curve stub)
  api/main.py (FastAPI 657 lines)
  app/ (Vite React: App.tsx, pages/Overview,DataPipeline,Models,Training,Evaluation,ScenarioGeneration, lib/api.ts)
  baselines/prepare_baseline_data.py, run_timegan_smoke.py, run_timegan_full.py + configs + results + external/TimeGAN,QuantGANs-replication (submodules?)
  data/download.py (yfinance NIFTY 2010-2024), preprocess.py, graph_builder.py, node_features.py (real 6-dim + 23-dim)
  eval/statistical.py (30 lines, only jarque_bera/adf/summary/correlation_summary)
  eval/discriminative.py (new), eval/descriminative.py (typo duplicate, dead), eval/contagion.py (32 lines, only spillover/directional/matrix), eval/statistical_timegan.py, eval/timegan_temporal.py
  models/gat_encoder.py (DynamicGATEncoder + legacy SpatialTemporalGNN), neural_sde.py (200 lines, GraphConditionedSDE+LatentSDEModel), decoder.py, contrastive.py (SupCon + legacy)
  src/__init__.py, data_processor.py (WINDOW=60 Pearson), model.py (11-line re-export shim)
  training/__init__.py, config.yaml (38 lines), trainer.py (514 lines, TWO trainers), gat_trainer.py, losses.py (178 lines), vortex_model.py (236 lines, Lightning)
  notebooks/demo.ipynb 55KB, vortexai.ipynb 1.6MB
  IMPLEMENTATION_PLAN.md (475 lines, Phase 3 spec), IMPLEMENTATION_SUMMARY.md (344 lines), QUICKSTART.md (354 lines), PRESENTATION.md (458 lines, 18 slides), README.md (212 lines)
  requirements.txt (33 lines), training/config.yaml, baselines/*.json, app/package.json etc.
```

Per-file classes/functions (from codebase scan — verify by reading heads):
- `train_vortex.py`: `VortexDataset(__init__,__len__,__getitem__,_build_node_features)`, `create_dataloaders,train,parse_args`
- `training/vortex_model.py:19`: `VORTEXModel(pl.LightningModule): forward,training_step,validation_step,configure_optimizers,generate_scenarios`
- `training/trainer.py:371`: duplicate `VORTEXModel` (tuple batch, `train_loss` logs) vs `vortex_model.py:19` (dict batch, `train/*` logs)
- `models/gat_encoder.py:73-169`: `DynamicGATEncoder`, `SpatialTemporalGNN` (legacy), `pool_graph_embedding,graph_consistency_loss`
- `models/neural_sde.py:22-153`: `GraphConditionedSDE(SDEIto): set_graph_context,_get_input,f,g`, `LatentSDEModel(GRU+sdeint euler+Linear)`
- `models/contrastive.py:17-93`: `SupConLoss,ProjectionHead` + legacy `ContrastiveLoss,info_nce_loss_simple`
- `training/losses.py:16-168`: `joint_loss,stylized_facts_loss,reconstruction_loss,graph_consistency_loss,circuit_filter_loss,total_loss→Tensor`
- `sandbox/circuit_filter.py:20-73`: `detect_circuit_breakers,circuit_breaker_stats,apply_circuit_filter,circuit_filter_loss`
- `api/main.py`: `ModelConfig`, 12 endpoints `/health,/stats,/returns,/window,/adjacency,/node-features,/regime-dist,/gat-forward,/sde-forward,/generate-scenario,/evaluation,/training-sim`

## 5. Architecture + tensor shapes (canonical)

- Pipeline: Yahoo → log-returns (T×N, N=50) → crisis labels → 60-day windows (N_win≈3440-3641,60,50) → Pearson/Granger adj (N_win,50,50) → node feats (N_win,N,6) → GAT → SDE → contrastive → circuit filter → eval/viz.
- `DynamicGATEncoder`: (B,N,F=6) → GATConv(6→64,heads=4)+LN+ELU → GATConv(256→64,heads=1) → node_embs (B,N,64) + learned_adj (B,N,N) from attention. `pool_graph_embedding` mean → graph_emb (B,64).
- `LatentSDEModel.forward(x(B,T,N),graph_emb(B,64))`: GRU(x)→z0(B,64) → `torchsde.sdeint(sde,z0,ts linspace(0,1,T),method=euler,dt=1/T)` → zs(B,T,64) → Linear→r_hat(B,T,N).
- SDE: `dX=mu([X,G,t])dt+sigma([X,G,t])dW`, drift MLP 129→128→128→64 Tanh, diffusion 129→128→64 Softplus, diagonal noise, Itô.
- Contrastive: temporal pool zs → ProjectionHead 64→32 L2-norm, SupCon τ=0.07.
- Loss: `L=1.0*L_rec(MSE+0.1*stylized)+0.1*L_graph(Frob)+0.1*L_con+0.0*L_circuit`. `lam_na=0` everywhere.
- Node feats (masterplan D.2, 6-dim): [mean_ret, abs_mean, vol_proxy, sector, dist_upper, dist_lower]. Real impl in `data/node_features.py:21-35,100-134`; train path uses placeholders (see §7).
- Circuit bands: A ±5% (SBI,Tata Motors,Tata Steel), B ±10% (most), C ±20%. Hard clip post-gen + soft penalty train.

## 6. Configs + commands

- `training/config.yaml`: epochs 50, bs 32, lr 1e-3, hidden 64, lstm_layers 2, dropout 0.3 + `cgnsde:{n_stocks:50,T:60,latent64,proj32,in6,heads4,tau0.07,lr1e-3,sde_hidden128,max200,bs32,grad_clip1.0,crisis85,lam_rec1.0,lam_graph0.1,lam_con0.1,lam_na0.0,lambda_styl0.1}`.
- `requirements.txt:1-33`: numpy,pandas,sklearn,statsmodels,scipy,yfinance,torch,torch-geometric,torchsde,lightning,wandb,sdmetrics,matplotlib,seaborn,plotly,streamlit,dash,backtesting. MISSING: `pyyaml,fastapi,uvicorn,pydantic,networkx,timegan`; has `lightning` but code imports `pytorch_lightning` (needs alias/`pytorch-lightning`).
- Canonical commands (README/QUICKSTART, PowerShell `py -3`):
  `py -3 -m pip install -r requirements.txt; py -3 data/preprocess.py; py -3 -m src.data_processor; py -3 check_pipeline.py; py -3 -m training.trainer --epochs 50; py -3 -m training.gat_trainer; py -3 evaluate.py; py -3 train_vortex.py --config training/config.yaml; py -3 evaluate_vortex.py --checkpoint <ckpt>; py -3 visualize_results.py --checkpoint <ckpt>`

## 7. Verified gaps blocking maximum potential (file:line evidence)

**Fatal (blocks e2e):**
1. `training/vortex_model.py:138-154,170-182` does `loss, components = total_loss(...)` but `training/losses.py:128-164` returns single `Tensor` → `ValueError` on any train run. Fix: return `(loss, dict)` or update callers.
2. `evaluate_vortex.py:27` imports `evaluate_stylized_facts,compare_distributions,test_no_autocorrelation_returns` — `eval/statistical.py:9-30` only has `jarque_bera_test,adf_test,summary_statistics,correlation_summary` → `ImportError`.
3. `evaluate_vortex.py:30` imports `evaluate_contagion,cvar_regime_ratio,granger_causality_test` — `eval/contagion.py:8-32` only has `compute_spillover_index,directional_spillovers,contagion_matrix` → `ImportError`.
4. Deps: `train_vortex.py:13` `yaml` missing; `api/main.py:18-20` `fastapi/uvicorn/pydantic` missing; `app.py,api/main.py` `networkx` missing; `baselines/run_timegan_smoke.py:28` `timegan` missing; `train_vortex.py:17,training/trainer.py:12` import `pytorch_lightning` but requirements has `lightning`.

**Correctness/placeholders:**
5. `train_vortex.py:98,102` `volume_proxy=returns.std`, `sector=torch.zeros(N)`; `train_vortex.py:74` `adj_empirical=adj` (trivializes graph loss). Real mapping in `data/node_features.py:21-35,100-134` + `nifty50_volume.csv` + `data/raw/window_node_features.npy` 262MB ignored.
6. `train_vortex.py:39` loads `window_regimes.npy` (old, ~73% crisis by `.max()`) vs corrected `window_regimes_v2.npy` used in `training/trainer.py:291`; `train_vortex.py:136-143` `random_split` non-stratified vs `trainer.py:324-336` stratified.
7. `training/losses.py:139,training/vortex_model.py:59,training/trainer.py:200,270,training/config.yaml:33-37,train_vortex.py:219` all `lam_na=0.0` — circuit term computed then zeroed.
8. `models/contrastive.py:135` returns `embeddings.sum()*0.0` when batch lacks positives — masks single-regime batches instead of skip/error.
9. `models/gat_encoder.py:131-149` per-sample `for b in range(B)` + `dense_to_sparse` per step (slow); `134-135` unconditional `+eye`; legacy `SpatialTemporalGNN:19-70` still imported by baseline.
10. `models/neural_sde.py:59,69-80` `graph_context=None` deref if `f/g` before `set_graph_context`; `122-126` `graph_emb_dim=latent_dim` hardcoded; Euler-only static `G_t`.
11. `sandbox/circuit_filter.py:14` `BAND_A_STOCKS` 3 tickers hardcoded; `17` `BAND_C_LIMIT` dead; `20,26` detect 0.095 vs train/infer 0.10; `api/main.py:25-36` tickers lack `.NS` so `in BAND_A` never matches.
12. `training/trainer.py:371` vs `training/vortex_model.py:19` duplicate `VORTEXModel` divergent batch APIs + log names + checkpoint collision `best_gat_model` (`trainer.py:486`).
13. `training/vortex_model.py:204-235` `generate_scenarios` loops SDE solves, keeps only `r_hat[0]`.
14. `visualize_results.py:327-329` `generate_loss_curves` prints placeholder instead of parsing wandb/tensorboard.
15. `eval/descriminative.py` vs `eval/discriminative.py` typo-duplicate; only latter staged.
16. `src/model.py:1-11` pure re-export shim, no API/I-O.
17. Granger: `data/graph_builder.py:88-99` O(N²) double loop, `103-109` fixed `alpha=0.5` mix; eval gates behind `--test_granger`.
18. No ablations (`ablate_*.py`, λ-sweep, `in_feats 6 vs 21` from `node_features.py:70-72`), `baselines/baseline_comparison.csv` 269B only.
19. Serving untracked/unpinned: `api/,app.py,app/` untracked, no uvicorn/fastapi pins.

## 8. What Claude must produce

Generate a **step-by-step implementation plan + methodology** for the downstream agent that:

1. **Fixes fatal blockers first** (total_loss contract, eval imports, requirements/pytorch_lightning alias) with minimal edits + executable verification (`py -3 -c "import ..."`, shape/finite asserts, 1-epoch smoke).
2. **Removes placeholders without scope creep:** wire real `window_node_features.npy`/`nifty50_volume.csv`/sector map into `VortexDataset`, fix `adj_empirical` target, use `window_regimes_v2.npy` + stratified split, enable `lam_na` path with test, fix SupCon single-regime handling, unify duplicate `VORTEXModel`, fix circuit-band consistency + `.NS` suffix, fix `generate_scenarios` batching.
3. **Performance:** batch GAT edge_index (no per-sample loop), optional Granger subsampling/caching, keep Euler `dt=1/T`, `clip_grad_norm_(1.0)`.
4. **Quality gates:** NaN prevention (`torch.isfinite`, `zs.std`, grad norms), per-phase acceptance (single-stock GBM sanity → 50-stock <30s CPU batch=8 → 10-epoch decreasing loss → 3/4 eval pass), no-merge unrelated histories (keep union-add; do NOT `merge --allow-unrelated` unless asked).
5. **Commits/checkpoints:** commit staged 7 files + modified 9 + untracked docs separately with clear messages; never discard `data/raw` ignored artifacts; produce `best_vortex.ckpt` compatible with eval/viz.
6. **Tests/docs:** add minimal `tests/test_smoke.py` (shapes, loss finite, no-import-error), update `requirements.txt`, keep README/QUICKSTART accurate, no new docs unless requested.
7. **Ablations/baselines (phase 2):** No-GAT, No-SupCon, static-graph, no-circuit; TimeGAN/QuantGAN comparison table; attention + fan-chart + P&L viz at 300 DPI.
8. **Methodology for downstream agent:** exact file edit order, PowerShell-safe commands (`; if ($?) {}` not `&&`), use `workdir` param not `cd`, Read-before-Edit, small diffs, run-after-each-fix, `git status/diff/log` before commit, never `push/commit` unless explicitly asked, preserve uncommitted work via stash.
9. **Output format:** ordered checklist with file:line edits, commands to run, expected outputs, rollback notes, and DONE criteria mapping to §2 targets + QUICKSTART success criteria (200 epochs, val<1.0, 4/4 eval).

Constraints: Windows PowerShell 5.1, CPU-first (CUDA optional), repo root above, do not overwrite `main` overlapping files without asking, do not track `data/raw/*.npy/*.csv` (ignored by design) unless user opts in.
