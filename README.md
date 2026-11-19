# CrisisBench & VORTEX-AI

**Evaluating Whether Correlation Fidelity Predicts Crisis Utility in Synthetic Multi-Asset Returns**

BE Major Project, Department of Computer Engineering, Fr. Conceicao Rodrigues College of Engineering, Mumbai.  
**Team:** Yashovardhan Thopte, Alok Shukla, Aaryan Singh  
**Project Guide:** Dr. Ashok Kanthe  
**Paper Format:** IEEEtran Journal Format

For the detailed evidence-backed assessment, read [docs/HONEST_SUMMARY.md](file:///c:/Users/myind/Documents/GitHub/vortex-ai/Vortex-AI/docs/HONEST_SUMMARY.md).

---

## 1. Project Overview

VORTEX-AI is an investigation into generative models for multi-asset stock returns. The primary contribution is **CrisisBench**, a preregistered evaluation protocol that separates:
1. **Unconditional statistical fidelity** (Family A: kurtosis, squared-return autocorrelation)
2. **Correlation structure** (Family B: Frobenius error, MAE, leading eigenvector alignment)
3. **Crisis dependence and contagion** (Family C: crisis correlation boost, CVaR regime ratio, tail co-exceedance)
4. **Discriminative indistinguishability** (Family D: post-hoc classifier scoring)

Alongside the benchmark, the project develops **CG-NSDE v2** (Conditional Graph Neural Stochastic Differential Equation v2), a reference model with low-rank factor volatility dynamics and Filtered Historical Simulation (FHS), explicitly disclosed as a **post-hoc reference implementation**.

---

## 2. Benchmark Gate Verdicts & Preregistered Hypotheses

All verdicts are frozen against the NSE-47 survivor panel (2010–2024) and held-out crisis episodes (see [results/tables/hypotheses.json](file:///c:/Users/myind/Documents/GitHub/vortex-ai/Vortex-AI/results/tables/hypotheses.json)):

| Gate / Hypothesis | Formulation / Acceptance Rule | Verdict | Measured Empirical Evidence |
|---|---|---|---|
| **H1** | Correlation and crisis fidelity decouple ($\tau(B,C) < 0.3$) | **NOT SUPPORTED** | NSE in-sample $\tau(B,C) = 0.45$ [0.35, 0.56]; held-out $\tau(B,C) = 0.35$ [0.05, 0.42] across 11 models |
| **G1 (v1)** | CG-NSDE v1 (preregistered) beats simple baselines | **FAIL** | CG-NSDE v1 crisis score 1.22 ± 0.28 vs Regime Gaussian 0.54 and Gaussian HMM 0.70 |
| **G1 (v2)** | CG-NSDE v2 beats simple baselines (post-hoc) | **PASS** | CG-NSDE v2 crisis score **0.15 ± 0.01** (full fit, rank 1 of 14) and **0.74 ± 0.02** (pre-2020 fit, rank 1 of 14) |
| **G2 (v1 & v2)** | Held-out crisis boost inside real 95% band on $\ge 2$ of 3 episodes | **FAIL (Both)** | **0 of 2 evaluable episodes** (COVID-2020 real +101.1% vs v2 +22.15%, v1 +2.91%; 2022 Shock real +49.42% vs v2 +22.15%, v1 +2.91%) |
| **H3 / G3** | Family C rank correlates with VaR/ES backtest rank ($\tau > 0.5$, non-overlapping $\tau_C > \tau_B$ CIs) | **PASS** | $\tau(\text{Family C}, \text{risk}) = 0.67$ [0.54, 0.78] vs $\tau(\text{Family B}, \text{risk}) = 0.27$ [0.10, 0.52] across 14 valid generators (`quantgan_fs` #1, `cgnsde2` #2 on FZ0; `timegan_fs` included; CIs do not overlap: 0.54 > 0.52) |
| **H4 / G4** | Decoupling ($\tau(B,C) < 0.3$) replicates on $\ge 1$ of US / KR (in-sample `full`) | **FAIL** | In-sample 11-model `full`: US $\tau(B,C) = 0.60$ [0.56, 0.64], South Korea $\tau(B,C) = 0.49$ [0.45, 0.53] (though held-out `pre2020` is $<0.3$ on both: US $0.17$, KR $0.00$) |
| **KILL** | $\tau(B,C) \ge 0.3$ on every market in-sample `full` | **TRIGGERED** | Triggered on in-sample 11-model `full` ($\tau = 0.45, 0.60, 0.49 \ge 0.3$ due to joint neural baseline failures; not triggered held-out on US/KR or on classical subset) |
| **H5** | Contamination sensitivity (1 bad tick flips ranking) | **SUPPORTED** | Metric pair-order rank flips observed under single-tick corruption (`T7_contamination.md`) |

### Key Findings
1. **Family C Shows Stronger Point Alignment With Risk Than Family B, Though Formal Gates H1 / G3 / G4 / KILL Fail:** Across 13 valid generators (`timevae` excluded due to negative ES), Family C correlates higher with out-of-sample FZ0 risk loss ($\tau_C = 0.64$ [0.46, 0.77]) than Family B ($\tau_B = 0.18$ [-0.03, 0.46]), with full-scale `quantgan_fs` (#1, `-3.357` at 97.5%, `-2.984` at 99%) and **CG-NSDE v2** (#2, `-3.307` at 97.5%, `-2.863` at 99%) leading all generators. However, because the 95% CIs no longer overlap (`0.54 > 0.52`), **H3 is supported and Gate G3 PASSES** under the frozen non-overlapping CI criterion. Similarly, in-sample decoupling fails across all 11 preregistered models ($\tau = 0.45$ NSE, $0.60$ US, $0.49$ KR, triggering KILL), though held-out `pre2020` decoupling holds on US ($0.17$) and Korea ($0.00$).
2. **Held-Out Crisis Gap:** No generative model—classical or neural—reproduced the magnitude of out-of-sample crisis correlation surges. Real crisis dependence in March 2020 was 2–5× stronger than pre-crisis training data.
3. **v2 Post-Hoc Leadership Across 14 Generators & 39/39 Ablations:** Even after adding full-scale GPU neural baselines (`quantgan_fs` $S_C = 0.56$ full / $0.94$ pre-2020, rank 5; `diffusion_fs` $S_C = 8.62$ full / $8.35$ pre-2020), CG-NSDE v2 retains rank 1 on Family C in both in-sample ($0.15 \pm 0.01$) and held-out ($0.74 \pm 0.02$) fits (with a **~5× out-of-sample degradation**). Complete 13-variant × 3-seed ablations (`T2_ablation_nse_full.csv`) confirm FHS innovations (`gauss` $0.33 \pm 0.02$, `param_innov` $0.31 \pm 0.03$), factor volatility (`scalar_vol` $0.31 \pm 0.05$), KL scaling (`beta01` $0.28 \pm 0.07$), contrastive loss (`no_supcon` $0.22 \pm 0.05$), leverage (`no_leverage` $0.19 \pm 0.12$), volatility dynamics (`no_voldyn` $0.18 \pm 0.03$), and posterior GAT (`no_gat` $0.18 \pm 0.04$) each contribute $>1$ full-model SD (`0.010`) to Family C fidelity.

---

## 3. Computing Policy: Cloud GPU/CPU Workflow (Kaggle)

To guarantee machine stability and reproducible compute:
- **Strict Policy:** **ZERO local model training.** All generator training runs exclusively in the cloud.
- **Kaggle Configuration:** Managed via `KAGGLE_CONFIG_DIR=c:\Users\myind\Downloads` (using private bundle `brxkeaaryan/vortex-crisisbench-bundle`).

### Completed Experiment Stages (100% of Queue + 42/42 Full-Scale GPU Complete; `546/546` `metrics.json`)
- **NSE Main Experiments:** ✅ **110/110 completed** (11 models × 5 seeds × 2 fits).
- **v2 Main Experiments:** ✅ **10/10 completed** (full and pre-2020 fits, 5 seeds; both `full/seed0.pt` and `pre2020/seed0.pt` checkpoints on disk for `api/server.py`).
- **Full-Scale GPU Neural Baselines (`crisisbench.fullscale`):** ✅ **`diffusion_fs` 14/14 + `quantgan_fs` 14/14 + `timegan_fs` 14/14 completed** on CUDA (`vortex-crisisbench-gpu-shard-1` COMPLETE + Colab `vortex-timegan` COMPLETE). All 3 full-scale GPU baselines landed at 14/14.
- **NSE & v2 Risk Refits:** ✅ **44/44 (`nse_risk`) + 4/4 (`v2_risk`) + 10 full-scale risk pools completed** (14 valid generators evaluated in `T4_risk.md` / `T5_metric_validity.md`, `timevae` excluded).
- **v2 Ablations:** ✅ **39/39 completed** (13 variants × 3 seeds in `T2_ablation_nse_full.csv`).
- **Cross-Market (`v2_markets` & `markets`):** ✅ **19/19 `v2_markets` completed**; **164/176 `markets` completed** (12 skipped, **0 remaining** — all 5 `kr/timegan/full` seeds done).
- **Extended NSE (`nse_ext`), Contamination, & DGP Sim:** ✅ **27/33 `nse_ext` (6 skipped, 0 remaining)**, **63/63 `contamination`**, and **18/18 `sim` completed**.
- **Sampler Alignment (Task 4) & Likelihood Dominance (Task 5):** ✅ **Completed** (47/47 unit & regression tests passing).
- **Colab `vortex-timegan`:** ✅ **COMPLETE** — all 14 `timegan_fs` jobs finished on L4, downloaded, extracted, verified locally. Session stopped cleanly.

### No Remaining In-Flight Compute Jobs
All 10 pipeline stages complete, all 3 full-scale GPU baselines at 14/14, 546/546 valid `metrics.json` (`bad=0`).

---

## 4. Repository Layout

| Directory / File | Description |
|---|---|
| `crisisbench/` | Core benchmark: protocol definition, data loaders, metric computation, analysis, risk backtesting, LaTeX export |
| `crisisbench/generators/` | Model implementations: classical baselines, neural baselines, `cgnsde.py` (v1), `cgnsde2.py` (v2) |
| `results/` | Output tables (`results/tables/`), gate verdicts (`results/gates/`), and run outputs (`results/runs/`) |
| `reports/` | Audit reports: [task-4-report.md](file:///c:/Users/myind/Documents/GitHub/vortex-ai/Vortex-AI/reports/task-4-report.md) (Sampler Alignment), [task-5-report.md](file:///c:/Users/myind/Documents/GitHub/vortex-ai/Vortex-AI/reports/task-5-report.md) (Likelihood Dominance) |
| `tests/` | Pytest verification suite: `test_sampler_alignment.py`, `test_likelihood_dominance.py`, `test_regressions.py`, `test_smoke.py` |
| `app/` | Interactive React + Vite dashboard displaying benchmark comparisons, gate evidence, and risk analysis |
| `api/server.py` | FastAPI endpoint for scenario generation and model health monitoring |
| `paper/` | IEEEtran journal manuscript (`main.tex`), bibliography (`references.bib`), and auto-generated macros (`numbers.tex`) |
| `docs/` | Project status ([docs/STATE.md](file:///c:/Users/myind/Documents/GitHub/vortex-ai/Vortex-AI/docs/STATE.md)) and honest summary ([docs/HONEST_SUMMARY.md](file:///c:/Users/myind/Documents/GitHub/vortex-ai/Vortex-AI/docs/HONEST_SUMMARY.md)) |

---

## 5. Quickstart & Local Verification

### Running Automated Unit Tests
```bash
python -m pytest tests/test_sampler_alignment.py tests/test_likelihood_dominance.py tests/test_regressions.py tests/test_smoke.py -q
```
*Expected: 47 passed.*

### Launching the Dashboard
The dashboard reads static pre-exported JSON files (`app/public/data/*.json`) and runs independently of model training:
```bash
# Windows:
start_dashboard.bat

# Linux / macOS:
cd app && npm install && npm run build
npx vite preview --port 4173 --host 127.0.0.1
```
Open `http://127.0.0.1:4173/` in your browser.

---

## 6. Disclosures & Limitations

1. **Survivorship Bias:** The primary panel consists of 47 current NSE constituents with complete histories from 2010 to 2024 ("NSE-47 survivor panel"). Delisted companies are not represented.
2. **Post-Hoc Model:** CG-NSDE v2 was formulated after diagnosing v1 failures on the benchmark. It is disclosed as a post-hoc case study.
3. **Data Source:** Raw daily return data obtained via Yahoo Finance; intended as an academic research prototype, not financial or investment advice.

---

## 7. License
Distributed under the MIT License. See `LICENSE` for details.
