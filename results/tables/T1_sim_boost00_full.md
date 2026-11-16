### Main comparison, market sim_boost00, fit 'full' (3424 real windows, 514 crisis)

| Model | corr_fro | corr_mae | pc1_share | crisis_freq | boost_pct | cvar_ratio | tail_coexc | kurt_med | acf1_sq | disc_feat | A score (rank) | B score (rank) | C score (rank) | D score (rank) | seeds | impl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Real data** (95% band) | ≤ 1.11 | ≤ 0.0192 | 0.271 [0.258, 0.285] | 0.15 [0.0743, 0.242] | -4.3 [-14, 7.81] | 1.77 [1.56, 1.97] | 0.18 [0.149, 0.204] | 3.33 [2.87, 3.58] | 0.0857 [0.0719, 0.0952] | 0.64 [0.5, 0.649] | – | – | – | – | – | – |
| i.i.d. Gaussian | **0.18** | **0.00308** | **0.271** | 0.0095 | n/p | n/p | 0.123 | -0.00102 | 0.000332 | 1 | 3.97 (6) | 0.09 (1) | 7.29 (6) | 2.58 (5) | 1 | own |
| Static t-PPCA | 1.34 | 0.0205 | **0.283** | 0.0105 | **7.39** | 1.56 | **0.172** | 1.3 | 0.000776 | 0.943 | 3.35 (5) | 1.14 (6) | 1.04 (4) | 2.63 (6) | 1 | own |
| GARCH-t FHS | **0.429** | **0.00752** | **0.266** | 0.0175 | 20.9 | **1.71** | 0.145 | **3.11** | **0.0854** | 0.977 | 0.19 (1) | 0.38 (4) | 1.73 (5) | 1.60 (2) | 1 | own on arch |
| DCC-GARCH-t | **0.438** | **0.00775** | **0.266** | 0.03 | **6.37** | **1.61** | **0.18** | 3.67 | **0.0853** | 0.957 | 0.26 (2) | 0.40 (5) | 0.91 (3) | 1.72 (4) | 1 | own |
| Regime Gaussian | **0.201** | **0.00349** | **0.273** | **0.131** | **0.461** | **1.69** | **0.174** | 1.08 | 0.11 | 0.985 | 1.88 (3) | 0.14 (2) | 0.36 (2) | 1.63 (3) | 1 | own |
| Gaussian HMM | **0.206** | **0.00358** | **0.273** | **0.138** | **-2.31** | **1.73** | **0.186** | 1.56 | 0.137 | 0.967 | 2.27 (4) | 0.15 (3) | 0.18 (1) | 1.57 (1) | 1 | hmmlearn |

Bold = the seed-mean lies inside the real-data 95% band (stat metrics) or under the real-vs-real noise ceiling (distance metrics). n/p = the generator produced too few crisis-labelled windows for the metric to exist. Scores are mean errors in units of real sampling noise (lower is better).