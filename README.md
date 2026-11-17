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
| **G1 (v2)** | CG-NSDE v2 beats simple baselines (post-hoc) | **PASS** | CG-NSDE v2 crisis score **0.15 ± 0.01** (full fit, rank 1 of 12) and **0.74 ± 0.02** (pre-2020 fit, rank 1 of 12) |
| **G2 (v1 & v2)** | Held-out crisis boost inside real 95% band on $\ge 2$ of 3 episodes | **FAIL (Both)** | **0 of 2 evaluable episodes** (COVID-2020 real +101.1% vs v2 +22.15%, v1 +2.91%; 2022 Shock real +49.42% vs v2 +22.15%, v1 +2.91%) |
| **H3 / G3** | Family C rank correlates with VaR/ES backtest rank ($\tau > 0.5$) | **PASS** | $\tau(\text{Family C}, \text{risk}) = \mathbf{0.80}$ [0.60, 1.00] > 0.5; $\tau(\text{Family B}, \text{risk}) = -0.20$ [-0.60, 0.20] |
| **H4 / G4** | Decoupling ($\tau(B,C) < 0.3$) replicates on $\ge 1$ of US / KR | **PASS (via KR)** | South Korea $\tau(B,C) = 0.29 < 0.3$ [0.21, 0.29] passes; US $\tau(B,C) = 0.60 > 0.3$ [0.56, 0.64] does not (partial replication) |
| **KILL** | Static generators achieve top Family C rank across markets | **NOT TRIGGERED** | Static models fail Family C (i.i.d. Gaussian rank 8, Static t-PPCA rank 7 on NSE) |
| **H5** | Contamination sensitivity (1 bad tick flips ranking) | **SUPPORTED** | Metric pair-order rank flips observed under single-tick corruption (`T7_contamination.md`) |

### Key Findings
1. **Crisis Metrics Predict Downstream Risk (G3 PASS), Though H1 Is Not Supported:** While Family B and C show moderate positive rank correlation on NSE ($\tau = 0.45$, so H1 $\tau < 0.3$ is not supported), standard correlation metrics (Frobenius norm, MAE) fail to predict downstream risk backtest performance ($\tau = -0.20$). Out-of-sample VaR/ES backtesting validates Family C ranking ($\tau = 0.80$), confirming that crisis metrics capture essential risk utility.
2. **Held-Out Crisis Gap:** No generative model—classical or neural—reproduced the magnitude of out-of-sample crisis correlation surges. Real crisis dependence in March 2020 was 2–5× stronger than pre-crisis training data.
3. **v2 Post-Hoc Leadership & OOS Degradation:** CG-NSDE v2 achieves rank 1 on Family C in both in-sample (0.15) and held-out (0.74) fits, but suffers a **~5× out-of-sample performance degradation**.

---

## 3. Computing Policy: Cloud GPU/CPU Workflow (Kaggle)

To guarantee machine stability and reproducible compute:
- **Strict Policy:** **ZERO local model training.** All generator training runs exclusively in the cloud.
- **Kaggle Configuration:** Managed via `KAGGLE_CONFIG_DIR=c:\Users\myind\Downloads` (using private bundle `brxkeaaryan/vortex-crisisbench-bundle`).

### Completed Experiment Stages
- **NSE Main Experiments:** ✅ **110/110 completed** (11 models × 5 seeds × 2 fits).
- **v2 Main Experiments:** ✅ **10/10 completed** (full and pre-2020 fits, 5 seeds).
- **v2 Risk Refits:** ✅ **4/4 completed** (yearly refits 2020–2023 for risk pools).
- **Risk Backtesting:** ✅ **Completed** (1,238 out-of-sample days, VaR/ES FZ0 scoring).
- **Sampler Alignment (Task 4):** ✅ **Completed** (Arm A vs Arm B A/B test evaluated on validation split).
- **Likelihood Dominance Verification (Task 5):** ✅ **Completed** (Unit & regression test suite passing).

### Active Cloud Kernel Shards (In-Flight)
1. `brxkeaaryan/vortex-crisisbench-gpu-shard-1` (2x T4 GPU): 22 queue GAN runs on CUDA + 28 fullscale jobs (`diffusion_fs`, `quantgan_fs`).
2. `brxkeaaryan/vortex-ai-fullscale-baselines` (T4 GPU): 42 fullscale baseline jobs (`diffusion_fs`, `quantgan_fs`, `timegan_fs`).
3. `brxkeaaryan/vortex-crisisbench-cpu-shard-1` (CPU 4 workers): 70 queue jobs (classical, TimeVAE, CG-NSDE v1, `nse_ext`, 4 `v2_markets`).
4. `brxkeaaryan/vortex-crisisbench-cpu-shard-2` (CPU 4 workers): 18 CG-NSDE v2 jobs (`api/server.py` weights backfill, `v2_main`, `v2_risk`, 12 `v2_markets`).
5. `brxkeaaryan/vortex-crisisbench-cpu-shard-3` (CPU 4 workers): 20 `v2_ablation` jobs (Part 1).
6. `brxkeaaryan/vortex-crisisbench-cpu-shard-4` (CPU 4 workers): 19 `v2_ablation` jobs (Part 2).

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
