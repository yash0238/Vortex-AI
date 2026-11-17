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
3. **Downstream Risk Ranking Validation (Gate G3 PASS):** Out-of-sample VaR/ES backtesting on equal-weight and Dirichlet portfolios demonstrates that Family C rank correlates strongly with backtest accuracy ($\tau = 0.80$ [0.60, 1.00]), whereas correlation fidelity (Family B) fails to predict risk utility ($\tau = -0.20$ [-0.60, 0.20]).
4. **Cross-Market Robustness (Gate G4 PASS via South Korea):** H4 tests whether $\tau(B,C) < 0.3$ replicates on at least 1 of the 2 international markets. G4 passes via South Korea alone ($\tau(B,C) = 0.29 < 0.3$ [0.21, 0.29]), whereas US ($\tau(B,C) = 0.60 > 0.3$ [0.56, 0.64]) does not satisfy $\tau < 0.3$ (partial replication).
5. **CG-NSDE v2 Relative Performance:** CG-NSDE v2 ranks #1 in Family C on the NSE-47 survivor panel in both the full fit (score 0.15 ± 0.01) and pre-2020 fit (score 0.74 ± 0.02), outperforming classical and neural baselines.
6. **Code Defect Remediation:**
   - **Defect 1 (Architectural Leakage):** Resolved in v17 by removing target-window return/volatility injection into the encoder.
   - **Defect 2 (Sampler Mismatch):** Resolved by aligning the stylized-facts loss with `emission_sample_seq`.
   - **Defect 3 (Likelihood Dominance):** Resolved via per-asset likelihood normalization (`_nll_per_asset`), scaled KL ceiling (`beta_max: 0.05`), and a volatility coupling floor ($\kappa \ge 0.60$).

### What CANNOT Be Claimed (Must Be Explicitly Stated or Demoted)
1. ❌ **"Correlation Is Not Crisis" as a Supported Finding:** Hypothesis H1 ($\tau(B,C) < 0.3$) is **NOT supported** on the NSE panel ($\tau = 0.45$ [0.35, 0.56] in-sample; $\tau = 0.35$ [0.05, 0.42] held-out). Correlation and crisis metrics agree moderately in-sample.
2. ❌ **CG-NSDE v1 Succeeded:** The preregistered model (v1) **FAILED Gate G1** (crisis score 1.22 ± 0.28 vs 0.54 for simple Regime Gaussian).
3. ❌ **Held-Out Crisis Reproduction (Gate G2):** **FAILS for all models**, including v1 (0/2) and v2 (0/2). During the March 2020 COVID crash, real correlation surged +101.1% [65.1%, 140.1%], whereas v2 produced only +22.15% and v1 +2.91%. Real crises exhibit 2–5× stronger dependence than calm pre-crisis training data.
4. ❌ **CG-NSDE v2 Is Preregistered:** v2 was designed post-hoc after v1 results were inspected. It is disclosed as a post-hoc reference implementation throughout.
5. ❌ **CG-NSDE v2 Generalizes Perfectly Out-of-Sample:** v2 experiences a **~5× performance degradation** out-of-sample (Family C score worsens from 0.145 in-sample to 0.742 on held-out 2020–2024).
6. ❌ **"First to Combine" Claims:** Claims of being "first to combine GAT + Neural SDE + Heavy Tails" are retracted. TimeGAN, QuantGAN, Tail-GAN, MarketGANs, CoMeTS-GAN, MARCD, SFAG, Deep-MKV-TS, HGAN-SDEs, and CoFinDiff exist.
7. ❌ **Graph-Conditioned Generation:** The graph embedding enters only the posterior encoder in v2; generation uses learned per-regime priors.
8. ❌ **ELBO Objective:** The training loss is a weighted multi-term objective, not a true Evidence Lower Bound.

---

## 3. Preregistered Hypotheses & Gate Verdicts (Exact Numbers)

All verdicts below match [results/tables/hypotheses.json](file:///c:/Users/myind/Documents/GitHub/vortex-ai/Vortex-AI/results/tables/hypotheses.json) and corresponding gate files:

| ID | Formulation / Criterion | Verdict | Measured Evidence |
|---|---|---|---|
| **H1** | Correlation and crisis fidelity decouple ($\tau(B,C) < 0.3$) | **NOT SUPPORTED** | NSE in-sample $\tau(B,C) = 0.45$ [0.35, 0.56]; held-out $\tau(B,C) = 0.35$ [0.05, 0.42] over 11 models |
| **G1 (v1)** | Regime-aware beats static; CG-NSDE v1 beats simple baselines | **FAIL** | CG-NSDE v1 crisis score 1.22 ± 0.28 vs Regime Gaussian 0.54 and Gaussian HMM 0.70 |
| **G1 (v2)** | CG-NSDE v2 beats simple baselines (post-hoc) | **PASS** | CG-NSDE v2 crisis score **0.15 ± 0.01** (full) and **0.74 ± 0.02** (pre2020), ranking #1 of 12 |
| **G2 (v1)** | Held-out boost inside real 95% band on $\ge 2$ of 3 episodes | **FAIL** | 0 of 2 evaluable episodes (COVID 2020: 2.91% vs 101.1%; 2022 Shock: 2.91% vs 49.42%) |
| **G2 (v2)** | Held-out boost inside real 95% band on $\ge 2$ of 3 episodes | **FAIL** | 0 of 2 evaluable episodes (COVID 2020: 22.15% vs 101.1%; 2022 Shock: 22.15% vs 49.42%) |
| **H3 / G3** | Family C rank correlates with VaR/ES backtest rank ($\tau > 0.5$, $\tau_C > \tau_B$) | **PASS** | $\tau_C = 0.80$ [0.60, 1.00] > 0.5; $\tau_B = -0.20$ [-0.60, 0.20] on 1238 test windows |
| **H4 / G4** | Decoupling ($\tau(B,C) < 0.3$) replicates on $\ge 1$ of US / KR | **PASS (via KR)** | KR $\tau(B,C) = 0.29 < 0.3$ [0.21, 0.29] passes; US $\tau(B,C) = 0.60 > 0.3$ [0.56, 0.64] does not (partial replication) |
| **KILL** | Static models rank well on Family C across all markets | **NOT TRIGGERED** | Static models consistently fail Family C (i.i.d. Gaussian rank 8, Static t-PPCA rank 7 on NSE) |
| **H5** | Contamination sensitivity (ranking flips under 1 bad tick) | **SUPPORTED** | Pair-order rank flips observed under 1 bad tick (documented in `T7_contamination.md`) |

*Note on Gate G2 episodes:* `mid_2024` has only 1 crisis window ($< 5$ window minimum requirement per protocol) and was excluded; G2 was evaluated honestly on the 2 evaluable episodes.

---

## 4. Model Evolution: v1 vs v2

| Feature | CG-NSDE v1 (Preregistered) | CG-NSDE v2 (Post-Hoc Reference) |
|---|---|---|
| **Status** | Preregistered in protocol | Post-hoc reference implementation (disclosed) |
| **Family C Score (NSE Full)** | 1.22 ± 0.28 (Rank 10) | **0.15 ± 0.01 (Rank 1)** |
| **Family C Score (NSE Pre-2020)** | 1.21 ± 0.03 (Rank 10) | **0.74 ± 0.02 (Rank 1)** |
| **Parameters** | 152,889 (~152 K) | 41,396 (~41 K, ~73% reduction) |
| **Latent Space** | 64-dim SDE paths | Low-rank factor EGARCH SDE (`rank_K=8` default; `K1, K4, K16` ablations) |
| **Innovations** | Parametric SMT / Student-t | Filtered Historical Simulation (FHS) of standardized residuals |
| **Graph Dependency** | Window-specific graph fed to generator (leakage) | Dynamic GAT in posterior encoder only; generation uses regime priors |
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
- **Kaggle Kernel Shards (In-Flight / Managed by `kaggle-runner`):**
  1. `vortex-crisisbench-gpu-shard-1` (2x T4 GPU): 22 GAN queue jobs + 28 fullscale jobs (`diffusion_fs`, `quantgan_fs`).
  2. `vortex-ai-fullscale-baselines` (T4 GPU): 42 fullscale baseline jobs (`diffusion_fs`, `quantgan_fs`, `timegan_fs`).
  3. `vortex-crisisbench-cpu-shard-1` (CPU 4 workers): 70 queue jobs (classical, TimeVAE, CG-NSDE v1, `nse_ext`, 4 `v2_markets`).
  4. `vortex-crisisbench-cpu-shard-2` (CPU 4 workers): 18 CG-NSDE v2 jobs (`api/server.py` weights backfill, `v2_main`, `v2_risk`, 12 `v2_markets`).
  5. `vortex-crisisbench-cpu-shard-3` (CPU 4 workers): 20 `v2_ablation` jobs (Part 1).
  6. `vortex-crisisbench-cpu-shard-4` (CPU 4 workers): 19 `v2_ablation` jobs (Part 2).

Outputs from completed shards are synchronized back into `results/runs/` and `artifacts/` for multi-seed aggregation and LaTeX table generation.
