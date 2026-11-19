# CrisisBench & VORTEX-AI: The Honest Summary

**Date:** 2026-11-18  
**Project:** VORTEX-AI / CrisisBench Benchmark  
**Authoritative Reference:** [research/notes/final_report_vortex-ai-merged-publishability.md](file:///c:/Users/myind/Documents/GitHub/vortex-ai/research/notes/final_report_vortex-ai-merged-publishability.md)

---

## 1. What This Project Is (and Is Not)

- **What it IS:** A rigorous, preregistered **evaluation and benchmark paper** for synthetic multi-asset financial returns with mixed and negative results (**CrisisBench**), accompanied by a post-hoc reference architecture (**CG-NSDE v2**).
- **What it is NOT:** A "novel new architecture" paper claiming unconditional superiority. The model is a case study inside the benchmark, not the primary asset.
- **The Core Problem:** Generative models for multi-asset returns (GANs, VAEs, Diffusion, Neural SDEs) often achieve low average Wasserstein or correlation distances, yet fail catastrophically during market stress (contagion surges, volatility clustering, tail risk). CrisisBench is designed to reveal these failures.

---

## 2. What Can and Cannot Be Claimed

### What CAN Be Claimed (Defensible With Evidence)
1. **Preregistered Evaluation Protocol:** A formal protocol defining four metric families:
   - **Family A:** Unconditional statistical fidelity (kurtosis, autocorrelation of returns, ACF of squared returns).
   - **Family B:** Correlation structure (Frobenius error, MAE, leading eigenvector alignment).
   - **Family C:** Crisis dependence & contagion (crisis correlation boost, CVaR ratio, tail co-exceedance, connection gap).
   - **Family D:** Discriminative indistinguishability (post-hoc classifier accuracy).
2. **Noise Calibration:** Generator errors are calibrated against real-data sampling noise derived via block bootstrap on real historical windows.
3. **VaR/ES Risk Backtest Point Correlation (Gate G3 PASS Over 14 Generators):** Out-of-sample VaR/ES backtesting across all 14 valid generators (1,238 test days on equal-weight and Dirichlet portfolios; `timevae` excluded and disclosed due to degenerate non-positive ES `min_es = -0.00088`) shows Family C has a higher point correlation with FZ0 risk loss ($\tau_C = 0.67$ [0.54, 0.78]) than Family B ($\tau_B = 0.27$ [0.10, 0.52]), with full-scale `quantgan_fs` (#1, $\text{FZ0} = -3.357$ at 97.5%, $-2.984$ at 99%) and post-hoc **CG-NSDE v2** (#2, $\text{FZ0} = -3.307$ at 97.5%, $-2.863$ at 99%) leading all 14 generators. Because the 95% bootstrap intervals no longer overlap (`c.lo > b.hi` is `True`; 0.54 > 0.52), **H3 is supported and Gate G3 PASSES** over the 14-generator panel (flipped from FAIL after adding `timegan_fs`).
4. **CG-NSDE v2 Relative Performance & Complete 39/39 Ablations:** Across the expanded 14-generator panel (`T1_nse_full.md`, `T1_nse_pre2020.md`), CG-NSDE v2 still ranks #1 in Family C in both the full fit (score $0.15 \pm 0.01$) and pre-2020 fit (score $0.74 \pm 0.02$), ahead of full-scale GPU `quantgan_fs` (full $0.56$ rank 5, pre-2020 $0.94$ rank 5) and `diffusion_fs` (full $8.62$ rank 13, pre-2020 $8.35$ rank 14). Across all 13 ablation variants × 3 seeds (39/39 complete, `T2_ablation_nse_full.csv`), removing FHS innovations (`gauss` $0.33 \pm 0.02$, `param_innov` $0.31 \pm 0.03$), factor volatility (`scalar_vol` $0.31 \pm 0.05$), KL scaling (`beta01` $0.28 \pm 0.07$), contrastive loss (`no_supcon` $0.22 \pm 0.05$), leverage (`no_leverage` $0.19 \pm 0.12$), volatility dynamics (`no_voldyn` $0.18 \pm 0.03$), or the posterior GAT (`no_gat` $0.18 \pm 0.04$, $>3$ full-model SDs worse than full $0.145 \pm 0.010$) degrades Family C performance by $>1$ full-model SD (`0.010`), while `no_sde` ($0.15 \pm 0.02$), `no_graph_loss` ($0.14 \pm 0.02$), and `K16` ($0.15 \pm 0.03$) lie within $\le 1$ full-model SD.
5. **Code Defect Remediation:**
   - **Defect 1 (Architectural Leakage):** Resolved in v17 by removing target-window return/volatility injection into the encoder.
   - **Defect 2 (Sampler Mismatch):** Resolved by aligning the stylized-facts loss with `emission_sample_seq`.
   - **Defect 3 (Likelihood Dominance):** Resolved via per-asset likelihood normalization (`_nll_per_asset`), scaled KL ceiling (`beta_max: 0.05`), and a volatility coupling floor ($\kappa \ge 0.60$).

### What CANNOT Be Claimed (Must Be Explicitly Stated or Demoted)
1. ❌ **"Correlation Is Not Crisis" as a Universal In-Sample Finding (H1, H4/G4, KILL):**
   - **H1** ($\tau(B,C) < 0.3$ on NSE) is **NOT supported** in-sample ($\tau = 0.45$ [0.35, 0.56]) or held-out ($\tau = 0.35$ [0.05, 0.42]).
   - **H4 / G4** and **KILL**: When all 11 preregistered generators (including collapsed CPU-scale neural baselines that fail both B and C together) are scored on in-sample `full`, $\tau(B,C) \ge 0.3$ on US ($0.60$ [0.56, 0.64]) and South Korea ($0.49$ [0.45, 0.53]), so **H4 is not supported, Gate G4 FAILS, and the formal KILL condition (`tau(B,C) >= 0.3` on every market in-sample) is triggered**. However, on held-out `pre2020` fits, $\tau(B,C) < 0.3$ holds on both US ($0.17$ [0.00, 0.28]) and Korea ($0.00$ [-0.11, 0.39]), and on the 8-model classical/regime subset Korea in-sample is $0.29 < 0.3$. Thus, decoupling holds out-of-sample rather than in-sample across all 11 generators, while **Gate G3 now PASSES** over the 14-generator risk panel (`tau_C = 0.67 [0.54, 0.78]` vs `tau_B = 0.27 [0.10, 0.52]`, CIs do not overlap: 0.54 > 0.52).
2. ❌ **CG-NSDE v1 Succeeded:** The preregistered model (v1) **FAILED Gate G1** (crisis score 1.22 ± 0.28 vs 0.54 for simple Regime Gaussian).
3. ❌ **Held-Out Crisis Reproduction (Gate G2):** **FAILS for all models**, including v1 (0/2) and v2 (0/2). During the March 2020 COVID crash, real correlation surged +101.1% [65.1%, 140.1%], whereas v2 produced only +22.15% and v1 +2.91%. Real crises exhibit 2–5× stronger dependence than calm pre-crisis training data.
4. ❌ **CG-NSDE v2 Is Preregistered:** v2 was designed post-hoc after v1 results were inspected. It is disclosed as a post-hoc reference implementation throughout.
5. ❌ **CG-NSDE v2 Generalizes Perfectly Out-of-Sample:** v2 experiences a **~5× performance degradation** out-of-sample (Family C score worsens from 0.145 in-sample to 0.742 on held-out 2020–2024).
6. ❌ **"First to Combine" Claims:** Claims of being "first to combine GAT + Neural SDE + Heavy Tails" are retracted. TimeGAN, QuantGAN, Tail-GAN, MarketGANs, CoMeTS-GAN, MARCD, SFAG, Deep-MKV-TS, HGAN-SDEs, and CoFinDiff exist.
7. ❌ **Graph-Conditioned Generation at Inference:** The graph embedding enters only the posterior encoder in v2 (`no_gat` hurts Family C from $0.145 \pm 0.010$ to $0.18 \pm 0.04$, while `no_graph_loss` is $0.14 \pm 0.02$); generation uses learned per-regime priors.
8. ❌ **ELBO Objective:** The training loss is a weighted multi-term objective, not a true Evidence Lower Bound.

---

## 3. Preregistered Hypotheses & Gate Verdicts (Exact Numbers)

All verdicts below match [results/tables/hypotheses.json](file:///c:/Users/myind/Documents/GitHub/vortex-ai/Vortex-AI/results/tables/hypotheses.json) and corresponding gate files:

| ID | Formulation / Criterion | Verdict | Measured Evidence |
|---|---|---|---|
| **H1** | Correlation and crisis fidelity decouple ($\tau(B,C) < 0.3$) | **NOT SUPPORTED** | NSE in-sample $\tau(B,C) = 0.45$ [0.35, 0.56]; held-out $\tau(B,C) = 0.35$ [0.05, 0.42] over 11 models |
| **G1 (v1)** | Regime-aware beats static; CG-NSDE v1 beats simple baselines | **FAIL** | CG-NSDE v1 crisis score 1.22 ± 0.28 vs Regime Gaussian 0.54 and Gaussian HMM 0.70 |
| **G1 (v2)** | CG-NSDE v2 beats simple baselines (post-hoc) | **PASS** | CG-NSDE v2 crisis score **0.15 ± 0.01** (full) and **0.74 ± 0.02** (pre2020), ranking #1 of 14 |
| **G2 (v1)** | Held-out boost inside real 95% band on $\ge 2$ of 3 episodes | **FAIL** | 0 of 2 evaluable episodes (COVID 2020: 2.91% vs 101.1%; 2022 Shock: 2.91% vs 49.42%) |
| **G2 (v2)** | Held-out boost inside real 95% band on $\ge 2$ of 3 episodes | **FAIL** | 0 of 2 evaluable episodes (COVID 2020: 22.15% vs 101.1%; 2022 Shock: 22.15% vs 49.42%) |
| **H3 / G3** | Family C rank correlates with VaR/ES backtest rank ($\tau > 0.5$, non-overlapping $\tau_C > \tau_B$ CIs) | **PASS** | $\tau_C = 0.67$ [0.54, 0.78] vs $\tau_B = 0.27$ [0.10, 0.52] across 14 valid generators (`quantgan_fs` #1, `cgnsde2` #2 on FZ0; `timegan_fs` included; CIs do not overlap: 0.54 > 0.52) |
| **H4 / G4** | Decoupling ($\tau(B,C) < 0.3$) replicates on $\ge 1$ of US / KR (in-sample `full`) | **FAIL** | In-sample 11-model `full`: US $\tau(B,C) = 0.60$ [0.56, 0.64], KR $\tau(B,C) = 0.49$ [0.45, 0.53] (though held-out `pre2020` is $<0.3$ on both: US $0.17$, KR $0.00$) |
| **KILL** | $\tau(B,C) \ge 0.3$ on every market in-sample `full` | **TRIGGERED** | Triggered on in-sample 11-model `full` ($\tau = 0.45, 0.60, 0.49 \ge 0.3$ due to joint neural baseline failures; not triggered held-out on US/KR or on classical subset) |
| **H5** | Contamination sensitivity (ranking flips under 1 bad tick) | **SUPPORTED** | Pair-order rank flips observed under 1 bad tick (documented in `T7_contamination.md`) |

*Note on Gate G2 episodes:* `mid_2024` has only 1 crisis window ($< 5$ window minimum requirement per protocol) and was excluded; G2 was evaluated honestly on the 2 evaluable episodes.

---

## 4. Model Evolution: v1 vs v2

| Feature | CG-NSDE v1 (Preregistered) | CG-NSDE v2 (Post-Hoc Reference) |
|---|---|---|
| **Status** | Preregistered in protocol | Post-hoc reference implementation (disclosed) |
| **Family C Score (NSE Full)** | 1.22 ± 0.28 (Rank 11 of 14) | **0.15 ± 0.01 (Rank 1 of 14)** |
| **Family C Score (NSE Pre-2020)** | 1.21 ± 0.03 (Rank 11 of 14) | **0.74 ± 0.02 (Rank 1 of 14)** |
| **VaR/ES Backtest Rank (FZ0)** | Rank 6 of 13 ($\text{FZ0} = -3.067$ at 97.5%) | **Rank 2 of 13 ($\text{FZ0} = -3.307$ at 97.5%, $-2.863$ at 99%, behind `quantgan_fs` $-3.357$ / $-2.984$)** |
| **Parameters** | 152,889 (~152 K) | 41,396 (~41 K, ~73% reduction) |
| **Latent Space** | 64-dim SDE paths | Low-rank factor EGARCH SDE (`rank_K=8` default; `K1, K4, K16` ablations) |
| **Innovations** | Parametric SMT / Student-t | Filtered Historical Simulation (FHS) of standardized residuals |
| **Graph Dependency** | Window-specific graph fed to generator (leakage) | Dynamic GAT in posterior encoder only (`no_gat` hurts $S_C$ to $0.18 \pm 0.04$); generation uses regime priors |
| **Tail Parameter $\nu$** | Learned scalar bounded [4.5, 8.0] | Single global scalar $\nu$ |

---

## 5. Summary of Diagnostic Investigations

### Task 4 (Sampler Alignment A/B Test)
- **Objective:** Evaluate `stylized_sampler="seq"` (step-by-step EGARCH recursion) vs `"single"` (one-shot draw).
- **Finding:** Aligning the sampler reduces kurtosis error from 16.54 to 11.08 (vs 9.43 real) and $\text{ACF}(r^2)$ error from 0.0908 to 0.0733 (vs 0.0358 real). However, both arms pass 0/4 gate tests after 2 epochs due to persistent return dispersion.

### Task 5 (Likelihood Dominance Verification)
- **Objective:** Verify v18 architectural mechanisms preventing likelihood from overpowering fidelity terms.
- **Verification:**
  1. `_nll_per_asset`: Raw NLL (~65 nats) divided by $N=50$ stocks $\to$ ~1.3 nats per asset.
  2. `beta_max: 0.05`: Scaled ceiling prevents KL from overwhelming rescaled likelihood.
  3. $\kappa \ge 0.60$ floor: `.clamp(min=0.6)` guarantees EGARCH volatility feedback cannot be eliminated by likelihood gradients.
- **Test suite:** 47/47 unit and regression tests passing.

---

## 6. Computing & Experiment Execution Infrastructure

To prevent machine crashes and ensure reproducible benchmarks:
- **Policy:** **ZERO local training on development laptops.**
- **Cloud Environment:** Training runs on Kaggle GPU / CPU clusters via `KAGGLE_CONFIG_DIR=c:\Users\myind\Downloads`.
- **Completed & Synced Kaggle Shards (6/6 Complete; 100% of Queue Complete, `546/546` valid `metrics.json`):**
  1. ✅ `vortex-crisisbench-cpu-shard-1` (COMPLETE): 70 queue jobs (classical, TimeVAE, CG-NSDE v1, `nse_ext`, 4 `v2_markets`).
  2. ✅ `vortex-crisisbench-cpu-shard-2` (COMPLETE): 18 CG-NSDE v2 jobs (`api/server.py` `full/seed0.pt` checkpoint backfill, `v2_main` 10/10, `v2_risk` 4/4, `v2_markets` 19/19).
  3. ✅ `vortex-crisisbench-cpu-shard-3` (COMPLETE): 20 `v2_ablation` jobs (Part 1).
  4. ✅ `vortex-crisisbench-cpu-shard-4` (COMPLETE): 19 `v2_ablation` jobs (Part 2 — all 39/39 `v2_ablation` jobs complete).
  5. ✅ `vortex-crisisbench-gpu-shard-1` (COMPLETE): Remaining queue GAN jobs (incl. all 5 `kr/timegan/full` seeds -> **0 remaining queue jobs across all 10 stages**) + `diffusion_fs` (**14/14 complete**) + `quantgan_fs` (**14/14 complete**).
  6. ✅ `vortex-ai-fullscale-baselines` (T4 GPU, COMPLETE): `timegan_fs` (**14/14 complete**) — all full-scale GPU baselines landed; `timevae` excluded/disclosed for degenerate ES.
- **Colab:** `vortex-timegan` L4 session COMPLETE — all 14 `timegan_fs` jobs finished, downloaded, extracted, verified locally (546/546 valid `metrics.json`, `bad=0`). Colab session stopped cleanly.
- **No remaining in-flight compute jobs.** All 10 stages complete, all 3 full-scale GPU baselines (diffusion_fs, quantgan_fs, timegan_fs) at 14/14.
