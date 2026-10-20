# VORTEX-AI Epic Plan — CG-NSDE Framework Completion

**Epic:** Train, validate, and publish a production-quality CG-NSDE generator checkpoint for NIFTY-50 stress testing that meets all masterplan Part F quality gates.

**Context:** Local `main` has been merged with `origin/main`, `origin/complete_implementation`, and `origin/fix/stabilize`. All core modules (GAT encoder, Latent SDE, SupCon contrastive, losses, circuit filter, training/eval scripts) are implemented and smoke-verified. Data artifacts exist (3,641 windows × 60 days × 50 stocks, ~15% crisis). **No trained generator checkpoint exists** — only older `.pt` classifier checkpoints incompatible with `VORTEXModel.load_from_checkpoint`.

**Masterplan Part F targets:**
| Metric | Target | Current | Status |
|---|---|---|---|
| Kurtosis | > 3.0 | 4.70 | PASS |
| ACF(r²) | > 0.05 | -0.014 | FAIL |
| Corr error (Frob) | < 2.5 | 4.82 | FAIL |
| Discriminative score | < 0.60 | 0.766 | FAIL |
| Crisis corr boost | > 50% | 65.0% | PASS |
| CVaR ratio | > 3x | 2.16x | FAIL |

**Constraint:** Do NOT push to git unless pushing from the `aaryxnblondead` account. All work stays local/committed only.

---

## Stories

### Story 1: Data & Scaling Readiness — COMPLETED
**Priority:** P0 (blocker for training)  
**Status:** DONE — committed  
**Summary:** Fixed NIFTY50_SECTORS (added HDFCLIFE.NS, LTI.NS, SBILIFE.NS, TATAMOTORS.NS; removed TMPV.NS), fixed BAND_A_STOCKS, regenerated scaler.npz (50 tickers), split.npz, window_regimes_v2.npy (3640 entries, 12.1% crisis), window_node_features.npy (3640×60×50×6), volume_windows.npy (3640×60×50), regime_corr.npz (50×50). Removed date filter from build_scaler_split.py. Dataset loads cleanly: 3640 samples, T=60, N=50.

---

### Story 2: Single-Stock SDE Sanity Test — COMPLETED
**Priority:** P0  
**Status:** DONE  
**Summary:** Both sanity tests pass: single-stock (B=4, T=60, N=1, 0.25s) and full 50-stock (B=8, T=60, N=50, 0.09s). All outputs finite.

---

### Story 3: End-to-End Training Run — COMPLETED
**Priority:** P0 (primary deliverable)  
**Status:** DONE — committed  
**Summary:** Trained VORTEXModel through Stage 2 (correlation loss + vol dynamics + Student-t emission). Best checkpoint: `models/checkpoints/vortex-epoch=48-val/total=64.9007.ckpt` (val/total=64.90 NLL). Checkpoint loads and forward pass produces finite (B,T,N), (B,N,N), (B,proj_dim) outputs.

---

### Story 4: Full Evaluation Against Masterplan Gates — COMPLETED
**Priority:** P0  
**Status:** DONE — results committed  
**Summary:** Evaluation ran successfully. Results:
| Metric | Target | Result | Status |
|---|---|---|---|
| Kurtosis | > 3.0 | 3.21 | PASS |
| ACF(r²) | > 0.05 | -0.059 | FAIL |
| Corr error (Frob) | < 2.5 | 13.81 | FAIL |
| Discriminative score | < 0.60 | 1.00 | FAIL |
| Crisis corr boost | > 50% | -7.7% | FAIL |
| CVaR ratio | > 3x | 0.93x | FAIL |

Overall: 0/4 gates pass. Key blocker: discriminative score = 1.0 (synthetic data trivially distinguishable from real). GPU training recommended to close gaps.

---

### Story 5: Ablation Studies — PENDING
**Priority:** P1  
**Status:** NOT STARTED  
**Dependencies:** Story 3

---

### Story 6: Baseline Comparisons — PENDING
**Priority:** P1  
**Status:** NOT STARTED  
**Dependencies:** Story 3

---

### Story 7: Visualizations at 300 DPI — COMPLETED
**Priority:** P1  
**Status:** DONE — committed  
**Summary:** All three figures generated: fan_chart.png (377KB), adjacency_heatmap.png (141KB), pnl_distribution.png (213KB). Fixed DataLoader num_workers=0 for Windows in visualize_results.py.

---

### Story 8: Add Tests & Harden Imports — COMPLETED
**Priority:** P1  
**Status:** DONE  
**Summary:** All smoke tests pass. requirements.txt includes pyyaml, fastapi, uvicorn, pydantic, networkx. All imports resolve correctly.

---

### Story 9: Commit All Work (No Push) — COMPLETED
**Priority:** P0  
**Status:** DONE  
**Summary:** Committed as `983dbbd` on branch `fix/stabilize`. 13 files changed. Working tree clean. No push performed (account restriction: aaryxnblondead only).

---

### Story 10: Push from `aaryxnblondead` Account — PENDING
**Priority:** P2  
**Status:** NOT STARTED  
**Dependencies:** Story 9  
**Note:** Only push if user switches git config to `aaryxnblondead` and explicitly requests it.

---

## Milestones

| Milestone | Stories | Exit Criteria |
|---|---|---|
| M1: Data readiness | 1 | scaler.npz + split.npz exist; dataset loads z-space |
| M2: Model sanity | 2 | Single-stock + 50-stock forward pass pass |
| M3: Trained checkpoint | 3 | `best_vortex*.ckpt` saved; loss decreases |
| M4: Evaluation gates | 4 | `eval_final.json` with all 6 Part F metrics |
| M5: Ablations + baselines | 5, 6 | `ablation_table.csv` + `baseline_table.csv` |
| M6: Visuals + tests | 7, 8 | Figures at 300 DPI; smoke tests pass |
| M7: Commit | 9 | Clean tree, commits grouped, no push |

---

## Dependency Order

```
Story 1 (data/scaler) → Story 2 (SDE sanity) → Story 3 (training)
   Story 3 → Story 4 (evaluation) → Story 5 (ablations)
   Story 3 → Story 6 (baselines)  → Story 7 (visuals)
   Story 4 → Story 8 (tests/harden)
   Stories 1–8 → Story 9 (commit, no push)
   Story 9 → Story 10 (push, only from aaryxnblondead)
```

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| CPU training too slow (200 epochs) | High | Delay | Reduce to 50–100 epochs with early stopping; batch 8; num_workers=0 on Windows |
| No NaN-free convergence | Medium | Block | Gradient clip 1.0, Euler dt=1/60, sigma_max ≤ 1.0, monitor `zs.std()` |
| Corr error stays > 2.5 | Medium | Miss gate | Ramp lam_corr (Lane 1A), vol_dyn (Lane 1B), hetero emission (7.3) |
| Discriminative > 0.60 | Medium | Miss gate | Contrastive warmup, stratified batches (min 4 per regime), SupCon tau tuning |
| CVaR ratio < 3x | Medium | Miss gate | Circuit filter loss weight lam_na, crisis-conditioned prior |
| No GPU available | High | Minor | CPU-first per CONVERGENCE_PLAN §3; GPU run deferred |

---

## DONE Criteria (Epic Complete)

- [ ] `models/checkpoints/best_vortex*.ckpt` exists and loads via `VORTEXModel.load_from_checkpoint`
- [ ] `results/eval_final.json` records all six Part F metrics with pass/fail
- [ ] At least 3/6 masterplan gates pass (minimum viable per CONVERGENCE_PLAN §4)
- [ ] `results/ablation_table.csv` and `results/baseline_table.csv` populated
- [ ] `results/figures/*.png` at 300 DPI
- [ ] `python3 tests/test_smoke.py` passes
- [ ] `git status` clean, all work committed locally
- [ ] Nothing pushed unless from `aaryxnblondead` account