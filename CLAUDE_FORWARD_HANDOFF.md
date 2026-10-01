# CLAUDE FORWARD HANDOFF — VORTEX-AI (CG-NSDE) — 2026-10-01

> Goal: give Claude full context to take this project forward to trained + evaluated + publishable state. Downstream runner is OpenCode/Muse Spark on win32 PowerShell, repo root `C:\Users\myind\Documents\GitHub\vortex-ai\Vortex-AI`.

## 1. What this is

**VORTEX-AI / CG-NSDE**: Contrastive Graph-Neural SDE for NSE NIFTY-50 stress testing. Flight-simulator for markets: learn normal/crisis physics, generate synthetic crash scenarios on demand for strategy stress-testing.

Vision stack (proposal + masterplan + slides):
```
Block0 Preprocess (T=60, composite crisis top-15%, Pearson+Granger A_emp)
 -> Block1 Dynamic GAT (heads=4) -> G_t, h_i,t
 -> Block2 GRU Encoder -> z0
 -> Block3 torchsde Neural SDE dX=mu(X,G,t)dt+sigma(X,G,t)dW Euler dt=1/60
 -> Block4 SupCon (tau=0.07) regime separation
 -> Block5 Decoder + NSE circuit filter -> r_hat
Loss L = 1.0*L_rec(MSE+0.1*styl) + 0.1*L_graph + 0.1*L_con + lam_na*L_circuit
```
Targets: kurtosis>3, ACF(r²)>0.05, corr err<2.5 Fro, discriminative<0.60, crisis boost>50%, CVaR ratio>3x.

Team: Yashovardhan Thopte, Alok Shukla, Aaryan Singh, FRCRCE Bandra. SDG 8/9.

## 2. Repo state (verified 2026-10-01)

Root: `Vortex-AI/` is git repo (`origin https://github.com/yash0238/Vortex-AI.git`), parent `vortex-ai/` is not. See old audit `CLAUDE_HANDOFF.md:25-42` — much of that is now stale (it flagged fatal blockers that were since fixed, see §3).

Data on disk (`data/raw/`, gitignored by design):
`windows.npy` 43MB, `volume_windows.npy` 43MB, `window_node_features.npy` 262MB, `adj_matrices.npy` 36MB, `returns.npy`, `regimes.npy`, `window_regimes.npy`, `window_regimes_v2.npy` (corrected ~15% crisis), `nifty50_close.csv`, `nifty50_volume.csv`.

Checkpoints: only `models/checkpoints/best_model.pt`, `best_gat_model.pt` (Aug baseline era). No `best_vortex.ckpt` yet.

Env: win32 PowerShell 5.1, Python 3.13 `.venv`, torch+torchsde+lightning. CPU-first (CUDA often unavailable).

## 3. What is DONE (do not rebuild)

- Data: `data/download.py:8` NIFTY50 list, `data/preprocess.py:29` composite crisis, `data/graph_builder.py:15,73` Pearson thr 0.3 + OLS Granger alpha 0.5, `data/node_features.py:100,137` compact 6-dim `[r,|r|,vol,sector,band,dist]` + `window_node_features.npy`.
- Models: `models/gat_encoder.py:73` `DynamicGATEncoder` + `pool_graph_embedding:161`, `models/neural_sde.py:22,93` `GraphConditionedSDE(SDEIto)` + `LatentSDEModel` GRU+sdeint Euler, `models/contrastive.py:17,67` `SupConLoss` + `ProjectionHead`, `models/decoder.py:9` unused — ignore.
- Losses: `training/losses.py:134` `total_loss` now returns `(total, components)` — old fatal bug fixed. Kurtosis is relative SE `training/losses.py:60`, `lam_na` default 0.05.
- Lightning: `training/vortex_model.py:19` canonical `VORTEXModel` (dict batch, `train/*` logs, `use_gat/static_graph` ablations, warmup ` _eff_lam_con:108`, fixed `generate_scenarios:266` cycles batch). Duplicate in `training/trainer.py:371` is legacy — do not use.
- Dataset: `train_vortex.py:25` `VortexDataset` uses `window_regimes_v2.npy:41`, real `window_node_features.npy:69` + volume + sector map, stratified split + `StratifiedBatchSampler:222` (>=4 crisis+4 normal per batch, masterplan requirement).
- Eval: `evaluate_vortex.py:1` + `eval/statistical.py`, `eval/discriminative.py`, `eval/contagion.py` now exist (old ImportError fixed).
- Viz/API/UI: `visualize_results.py`, `api/main.py:346` scenario gen, `app.py` Streamlit, `app/` React.

Measured (unverified CUDA, README): baseline MSE 0.0694 / AUC 0.5621, GAT MSE 0.0829 / AUC 0.8326. 1-epoch CPU smoke passes 0/4 quality gates (expected).

## 4. Novelty verdict (from 2026-10-01 SOTA survey — use in paper Related Work)

- GAN: TimeGAN, QuantGAN = shallow multivariate, no graph. GAT-GAN (2023), Sig-Graph GAN (May 2026), CoMeTS-GAN+critic-diffusion (2026), CGAN-in-ABIDES = nearest graph-generative but static graph / no SDE / no SupCon.
- Diffusion: DDPM-wavelet, FTS-Diffusion ICLR24, CoFinDiff IJCAI25, GBM-diffusion, DigMA, DARL crash-intensity DDPM, Hybrid GAN+Diffusion = conditional crash gen exists, none graph-conditioned SDE.
- Neural SDE: torchsde, SDE-GAN Kidger, NANSDE-Net, NSF, arbitrage-free SDE = single-asset/fixed-dim. DGNSDE (Delay-Aware Graph Neural SDE) is closest to claim 1 but forecasting, not generative.
- Graph-finance: Temporal GAT, TAGN, DySTAGE, FinAdaptGAT = discriminative.
- Regime: ProteuS (ARMA-GARCH per regime), MC-TE-GAN macro crashes, Conditional Market-GAN = conditional gen exists; SupCon-inside-SDE is novel application.
- Industry: ABIDES, LOBGAN, DSLOB, JPM synthetic, FinRL = US LOB mechanics, no NSE bands.
- NSE: index halts 10/15/20% + pre-open (`nseindia.com`), T+1 (`nseclearing.in`) — no generator encodes these.

Defensible claim: **first to combine dynamic G_t→mu,sigma + SupCon regime + NSE rules for NSE daily returns**. Drop "first for any market". Scope T+1/auction as future work — current `sandbox/circuit_filter.py:31` is clip-only.

Cite per row: GAT-GAN, Sig-Graph GAN, CoMeTS-GAN, FTS-Diffusion, CoFinDiff, DARL, DGNSDE, TSGDiff, MC-TE-GAN.

## 5. Gaps to close (priority order for Claude)

1. **Train to convergence**: run `train_vortex.py` 10-epoch smoke → 200-epoch. Fix stylized-scale dominance, verify decreasing loss, produce `best_vortex.ckpt`.
2. **Conditional generation**: replace `sandbox/generate.py:6` Gaussian toy with SDE sampling conditioned on regime + G_t; hard-clip post-process; add CVaR to `sandbox/metrics.py`.
3. **Eval suite Tier1-4**: kurtosis, ACF(r²), no-ACF(r), leverage, Frobenius, Granger preservation, boost, discriminative GBC, CVaR, breach clustering, P&L width.
4. **Baselines+ablations**: TimeGAN/QuantGAN table + No-GAT/No-SupCon/Static/No-Filter (`--ablation` flag already in `train_vortex.py:461`).
5. **NSE honesty**: keep circuit clip+penalty (`lam_na>0` test); move T+1/auction to limitations/future.
6. **Cleanup**: pin `pyyaml,fastapi,uvicorn,networkx` in `requirements.txt:1`, dedupe `eval/descriminative.py` typo, `training/trainer.py` legacy model.

## 6. How to run (PowerShell, CPU-safe)

```powershell
py -3 data/preprocess.py
py -3 -m src.data_processor
py -3 check_pipeline.py
py -3 train_vortex.py --config training/config.yaml --max_epochs 10 --batch_size 16 --accelerator cpu --num_workers 0
py -3 evaluate_vortex.py --checkpoint models/checkpoints/best_vortex.ckpt --n_scenarios 100 --device cpu
py -3 visualize_results.py --checkpoint models/checkpoints/best_vortex.ckpt --output_dir results/figures
py -3 -m training.gat_trainer --epochs 50 --device cpu
py -3 evaluate.py --device cpu
```

Config: `training/config.yaml` (T60, latent64, proj32, heads4, tau0.07, lr1e-3, grad_clip1.0, lam_rec1.0/graph0.1/con0.1/na0.05, warmup20).

## 7. Rules for Claude / downstream agent

- Verify by execution, `file_path:line_number` refs, small diffs, Read-before-Edit, prefer editing over creating.
- PowerShell: use `;` not `&&`, use `workdir` param not `cd`.
- Do NOT `merge --allow-unrelated` (main vs complete_implementation histories unrelated). Do NOT track `data/raw/*.npy/*.csv`. Do NOT push/commit unless asked.
- NaN protocol: Euler only, dt=1/T, clip 1.0, check `torch.isfinite`, `zs.std`, grad norms.
- Acceptance: single-stock GBM sanity → 50-stock <30s CPU b=8 → 10-epoch decreasing → 3/4 eval pass.

## 8. Ask Claude to produce

1. Ordered fix→train→eval→ablate checklist with file:line edits + commands + expected outputs + rollback.
2. Gap-matrix Related Work draft (cite §4 papers).
3. Scoped NSE formulation (circuit in-scope, T+1/auction future).
4. DONE criteria mapping to §1 targets + `QUICKSTART.md:338` success block.

Constraints: keep `IMPLEMENTATION_SUMMARY.md:8-16` honest status (Phases 0-4 smoke, 0/4 gates, no 200-epoch yet), do not resurrect "13/13 complete".

## 9. Last evaluations (2026-09-26 — latest runs, include in Claude context)

All paths relative to repo root. Timestamps IST local dir listing.

### 9.1 CG-NSDE — `results/eval_fixed.json` (2026-09-26 15:31, LATEST) vs `results/eval_500.json` (14:35)

Both from `evaluate_vortex.py` on vortex ckpts (`models/checkpoints/last-v19.ckpt` 1.41MB latest, `resume_500.ckpt` lineage). Neither passes quality gates — 0-1/4 effectively:

| Metric | Target | eval_fixed (latest) | eval_500 | Read |
|---|---|---|---|---|
| kurt real / gen | gen>3 | 527.89 / 8.71 PASS | 527.89 / 20.68 PASS | real O(500) = outlier windows dominate; gen fat-tailed but far from real magnitude |
| ACF(r²) real / gen | >0.05 | 0.00048 / 0.030 FAIL | 0.00048 / 0.017 FAIL | gen>real but below threshold; no vol clustering |
| corr err (Fro) | <2.5 | 21.66 FAIL | 20.07 FAIL | ~10x over budget — graph not learned |
| discriminative | <0.60 | 1.00 FAIL | 1.00 FAIL | perfectly distinguishable, indistinguishability 0.0 |
| returns ACF lag1 | ~0 | 0.071 uncorrelated PASS | 0.022 PASS | only passing sanity |
| crisis boost | >50% | 3.47% FAIL (0.828→0.857) | 0.54% FAIL | no contagion lift |
| CVaR ratio | >3x | -38.07x FAIL (sign bug) | 0.78x FAIL | fixed: crisis CVaR -0.0014 vs normal +0.00003 — sign/magnitude broken, check loss sign convention |
| mean/std real vs gen | match | 0.00079/0.019 vs 0.0018/0.025 OK | 0.00079/0.019 vs 0.0051/0.346 FAIL | eval_500 variance exploded 18x — SDE diffusion unchecked |

Takeaway for Claude: reconstruction fires (means roughly match in fixed) but (a) stylized kurt scale mismatch, (b) graph loss not pulling corr, (c) SDE variance unstable across runs, (d) SupCon not separating (disc=1.0), (e) CVaR metric sign needs fix before trusting ratio.

### 9.2 TimeGAN baseline — `baselines/results/timegan_full/` (2026-08-28, 2000 iters, 1163s)

`statistical_metrics.csv:1-8`, `statistics.json:1-28`: mean err 7.8e-05, std err 0.001 (good), min/max collapse (-0.05/0.07 vs real -1.44/0.36), avg kurt -1.72 vs real 37.82 (misses tails), corr MAE 0.736. Classic mode-collapse / tail-smoothing — exactly the gap CG-NSDE claims to fix, but CG-NSDE currently scores worse on corr/disc. Use as baseline row, do not regress.

### 9.3 Training trace — `lightning_logs/version_33/` (latest)

`metrics.csv:1-12`, `hparams.yaml:1-17`: lam_rec 1.0 / graph 0.3 / con 0.2 / na 0.05, styl 0.5, lr 1e-3. Train total ~0.51-0.75, val ~0.51 flat — reconstruction ~0.47, contrastive ~1.9-2.3 (dominant), graph ~0.06, circuit ~0.02-0.08. Contrastive fighting rescaling → keep warmup `training/vortex_model.py:108` (20 epochs). ~34 versions (v0-v33) = many restarts, no convergence yet.

Checkpoints `models/checkpoints/`: full `vortex-epoch=00-45-val` + 4 ablations (`vortex-abl-no-gat/no-supcon/static-graph/no-circuit-epoch=00-08-val`) each ~1.41MB `.ckpt`, plus `last*.ckpt`, `resume_500.ckpt`. Ablation eval table still missing — run `evaluate_vortex.py` per ablation ckpt next.

### 9.4 Figures — `results/figures/` (2026-09-26 14:36)

`fan_chart.png` 175KB, `adjacency_heatmap.png` 141KB, `pnl_distribution.png` 199KB exist at 300 DPI but generated from pre-convergence ckpt — regenerate after 200-epoch run. `visualize_results.py` loss-curve is stub (prints placeholder).

Instruct Claude: lead with this §9 table, then §5 gaps. Do not claim targets met.
