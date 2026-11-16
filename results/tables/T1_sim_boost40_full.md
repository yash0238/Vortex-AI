### Main comparison, market sim_boost40, fit 'full' (3424 real windows, 514 crisis)

| Model | corr_fro | corr_mae | pc1_share | crisis_freq | boost_pct | cvar_ratio | tail_coexc | kurt_med | acf1_sq | disc_feat | A score (rank) | B score (rank) | C score (rank) | D score (rank) | seeds | impl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Real data** (95% band) | ≤ 1.25 | ≤ 0.0221 | 0.314 [0.293, 0.331] | 0.15 [0.0749, 0.241] | 17 [6.59, 30.9] | 2.14 [1.9, 2.4] | 0.218 [0.173, 0.254] | 3.8 [3.19, 3.95] | 0.105 [0.0895, 0.117] | 0.637 [0.5, 0.647] | – | – | – | – | – | – |
| i.i.d. Gaussian | **0.187** | **0.00324** | **0.315** | 0.0155 | 5.8 | 1.24 | 0.14 | -0.00298 | 0.000292 | 1 | 4.25 (6) | 0.11 (1) | 1.65 (6) | 4.78 (6) | 1 | own |
| Static t-PPCA | 1.3 | **0.02** | **0.326** | 0.017 | **6.86** | 1.46 | **0.19** | 1.3 | 0.000826 | 0.947 | 3.68 (5) | 0.90 (4) | 1.22 (5) | 4.13 (5) | 1 | own |
| GARCH-t FHS | **1.18** | 0.0228 | 0.291 | 0.029 | **27.3** | **1.96** | 0.163 | **3.69** | **0.106** | 0.975 | 0.12 (1) | 1.10 (5) | 1.05 (4) | 2.05 (2) | 1 | own on arch |
| DCC-GARCH-t | **1.18** | 0.0228 | 0.291 | 0.0455 | **12.2** | 1.69 | **0.192** | 4.18 | **0.111** | 0.958 | 0.36 (2) | 1.10 (6) | 0.77 (3) | 2.48 (4) | 1 | own |
| Regime Gaussian | **0.318** | **0.0058** | **0.319** | **0.134** | **18.1** | **2.01** | **0.218** | 1.43 | 0.133 | 0.976 | 1.90 (3) | 0.27 (3) | 0.19 (2) | 2.05 (3) | 1 | own |
| Gaussian HMM | **0.315** | **0.00584** | **0.319** | **0.143** | **17.7** | **2.07** | **0.235** | 2.16 | 0.159 | 0.924 | 2.03 (4) | 0.26 (2) | 0.18 (1) | 1.44 (1) | 1 | hmmlearn |

Bold = the seed-mean lies inside the real-data 95% band (stat metrics) or under the real-vs-real noise ceiling (distance metrics). n/p = the generator produced too few crisis-labelled windows for the metric to exist. Scores are mean errors in units of real sampling noise (lower is better).