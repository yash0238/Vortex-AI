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

### Story 1: Data & Scaling Readiness
**Priority:** P0 (blocker for training)  
**Estimate:** 1 hour  
**Dependencies:** None  
**Goal:** Ensure `data/raw/scaler.npz` and `data/raw/split.npz` exist and are correct so training uses standardized z-space returns and a clean train/val/test split.

**Acceptance criteria:**
- `data/raw/scaler.npz` exists with mean/std arrays shape (50,)
- `data/raw/split.npz` exists with train/val/test indices
- `VortexDataset` loads scaler and standardizes windows (prints "Standardized windows")
- Train/val/test crisis rates are sane (~15% crisis overall, non-degenerate)

**Files to touch:** `scripts/build_scaler_split.py`, `data/raw/scaler.npz`, `data/raw/split.npz`

**Verification:** `python3 scripts/build_scaler_split.py` runs cleanly; `python3 -c "import numpy as np; d=np.load('data/raw/scaler.npz'); print(d['mean'].shape, d['std'].shape)"` outputs `(50,) (50,)`.

---

### Story 2: Single-Stock SDE Sanity Test
**Priority:** P0  
**Estimate:** 1 hour  
**Dependencies:** Story 1  
**Goal:** Verify `LatentSDEModel` produces finite, Gaussian-like output on a single stock before scaling to 50.

**Acceptance criteria:**
- `python3 models/neural_sde.py` single-stock test passes (shape (4,60,1), finite, mean≈0, std≈0.01)
- Full 50-stock forward pass passes (shape (8,60,50), finite, <30s CPU)

**Files to touch:** `models/neural_sde.py` (if fixes needed)

**Verification:** Run the `__main__` block in `models/neural_sde.py`; both sanity tests print "PASSED".

---

### Story 3: End-to-End Training Run (200 epochs)
**Priority:** P0 (primary deliverable)  
**Estimate:** 6–10 hours (CPU, batch 8, early stopping)  
**Dependencies:** Stories 1, 2  
**Goal:** Train `VORTEXModel` and save `models/checkpoints/best_vortex-{epoch}-{val/total:.4f}.ckpt` compatible with `evaluate_vortex.py` and `visualize_results.py`.

**Acceptance criteria:**
- `python3 train_vortex.py --config training/config.yaml --accelerator cpu --batch_size 8 --max_epochs 200 --num_workers 0` completes
- Best checkpoint saved under `models/checkpoints/` with filename prefix `best_vortex`
- `VORTEXModel.load_from_checkpoint(best_vortex.ckpt, strict=False)` loads without error
- No NaN/Inf in `train/total` or `val/total` across epochs
- Loss decreases over epochs (val/total trend downward)

**Files to touch:** `training/config.yaml` (CPU-safe overrides), `train_vortex.py` (if CLI tweaks needed)

**Verification:** `ls models/checkpoints/best_vortex*` shows at least one `.ckpt`; loading it works.

---

### Story 4: Full Evaluation Against Masterplan Gates
**Priority:** P0  
**Estimate:** 2 hours  
**Dependencies:** Story 3  
**Goal:** Run `evaluate_vortex.py` against the trained checkpoint and record all six Part F metrics.

**Acceptance criteria:**
- `python3 evaluate_vortex.py --checkpoint models/checkpoints/best_vortex.ckpt --device cpu --n_scenarios 200 --batch_size 8 --num_workers 0` runs without ImportError
- All four eval test groups (statistical fidelity, discriminative, contagion, CVaR) produce pass/fail flags
- Results saved to `results/eval_final.json` (n=200+)
- Kurtosis > 3, ACF(r²) > 0.05, Corr error < 2.5, Discriminative < 0.60, Crisis boost > 50%, CVaR > 3x — or documented gap with remediation plan

**Files to touch:** `evaluate_vortex.py` (if import/runtime fixes needed), `results/eval_final.json`

**Verification:** Evaluation summary prints `Overall: X/4 tests passed`; JSON file exists.

---

### Story 5: Ablation Studies (No-GAT, No-SupCon, Static-Graph, No-Circuit)
**Priority:** P1  
**Estimate:** 4–6 hours  
**Dependencies:** Story 3  
**Goal:** Quantify each component's contribution via short ablations (10–20 epochs each).

**Acceptance criteria:**
- Four ablation checkpoints saved: `best_vortex-abl-no-gat`, `best_vortex-abl-no-supcon`, `best_vortex-abl-static-graph`, `best_vortex-abl-no-circuit`
- `results/ablation_table.csv` updated with per-variant metrics
- Each ablation shows expected degradation vs full model (validates component necessity)

**Files to touch:** `train_vortex.py` (ablation flag already supported), `results/ablation_table.csv`

**Verification:** `python3 train_vortex.py --ablation no-gat ...` runs and saves checkpoint; CSV has rows for all four variants.

---

### Story 6: Baseline Comparisons (iid-Gaussian, static-t-PPCA)
**Priority:** P1  
**Estimate:** 2 hours  
**Dependencies:** Story 3  
**Goal:** Produce fallback baseline numbers for the paper's baseline comparison table (TimeGAN/QuantGAN remain BLOCKED per convergence plan — no py3.13 wheel).

**Acceptance criteria:**
- `results/baseline_table.csv` has rows for iid-Gaussian and static-t-PPCA with corr_fro, corr_mae, disc, boost_pct, cvar_x
- TimeGAN and QuantGAN marked BLOCKED with reason
- CG-NSDE row matches ablation/final results

**Files to touch:** `scripts/oracle_eval.py` or new baseline script, `results/baseline_table.csv`

**Verification:** CSV readable; values finite; BLOCKED entries documented.

---

### Story 7: Visualizations at 300 DPI
**Priority:** P1  
**Estimate:** 2 hours  
**Dependencies:** Story 3  
**Goal:** Generate fan charts, adjacency heatmaps, and P&L distributions from the trained checkpoint.

**Acceptance criteria:**
- `python3 visualize_results.py --checkpoint models/checkpoints/best_vortex.ckpt --device cpu` runs without error
- `results/figures/fan_chart.png`, `adjacency_heatmap.png`, `pnl_distribution.png` exist at 300 DPI
- `generate_loss_curves` reads actual wandb/CSV logs (not placeholder print)

**Files to touch:** `visualize_results.py` (if loss-curve stub needs fixing)

**Verification:** PNG files exist under `results/figures/`; file sizes > 50KB.

---

### Story 8: Add Tests & Harden Imports
**Priority:** P1  
**Estimate:** 2 hours  
**Dependencies:** Stories 1–4  
**Goal:** Add `tests/test_smoke.py` coverage (already present, verify it passes) and ensure all imports resolve without `pyyaml`/`fastapi`/`pytorch_lightning` alias issues.

**Acceptance criteria:**
- `python3 tests/test_smoke.py` prints "ALL SMOKE TESTS PASSED"
- `requirements.txt` includes `pyyaml`, `fastapi`, `uvicorn`, `pydantic`, `networkx`
- `import pytorch_lightning` works (requirements has both `lightning` and `pytorch-lightning`)

**Files to touch:** `requirements.txt`, `tests/test_smoke.py`

**Verification:** Smoke tests pass; `python3 -c "import pytorch_lightning; import fastapi; import networkx"` succeeds.

---

### Story 9: Commit All Work (No Push)
**Priority:** P0  
**Estimate:** 1 hour  
**Dependencies:** Stories 1–8  
**Goal:** Commit staged + modified + untracked files with clear messages. Do NOT push (account restriction).

**Acceptance criteria:**
- `git status` clean after commit
- Commits grouped: data/scaler/split, model fixes, training, eval, results, tests, docs
- No `data/raw/*.npy/*.csv` tracked (ignored by design)
- No push attempted

**Files to touch:** All modified files; `.gitignore` (confirm data/raw ignored)

**Verification:** `git log --oneline -5` shows new commits; `git status` clean; `git remote -v` unchanged.

---

### Story 10: Push from `aaryxnblondead` Account (Optional / User-Triggered)
**Priority:** P2  
**Estimate:** 0.5 hours  
**Dependencies:** Story 9  
**Goal:** Push local commits to `origin/main` only if the user switches git config to `aaryxnblondead` and explicitly requests it.

**Acceptance criteria:**
- `git config user.name` == `aaryxnblondead` (or user confirms)
- `git push origin main` succeeds
- No force-push; no history rewrite on unrelated branches

**Files to touch:** None (git config only)

**Verification:** `git log origin/main --oneline -1` matches local HEAD.

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