### Main comparison, market sim_ccc, fit 'full' (3424 real windows, 514 crisis)

| Model | corr_fro | corr_mae | pc1_share | crisis_freq | boost_pct | cvar_ratio | tail_coexc | kurt_med | acf1_sq | disc_feat | A score (rank) | B score (rank) | C score (rank) | D score (rank) | seeds | impl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Real data** (95% band) | ≤ 0.906 | ≤ 0.0155 | 0.263 [0.255, 0.272] | 0.15 [0.0781, 0.225] | 13.6 [5.7, 20.6] | 1.62 [1.35, 1.84] | 0.133 [0.113, 0.154] | 2.42 [2.1, 2.63] | 0.116 [0.104, 0.121] | 0.625 [0.5, 0.654] | – | – | – | – | – | – |
| i.i.d. Gaussian | **0.163** | **0.00275** | **0.261** | 0.0275 | 4.26 | 1.23 | **0.119** | -0.000735 | 7.11e-05 | 0.996 | 5.12 (6) | 0.17 (3) | 1.33 (5) | 1.60 (3) | 1 | own |
| Static t-PPCA | 1.15 | 0.0168 | 0.273 | 0.0365 | 1.97 | 1.23 | **0.14** | 0.317 | -0.000492 | 0.985 | 4.91 (3) | 1.31 (6) | 1.38 (6) | 1.77 (5) | 1 | own |
| GARCH-t FHS | **0.516** | **0.00834** | **0.26** | **0.127** | **19.6** | **1.58** | **0.134** | 3.66 | 0.144 | **0.48** | 2.01 (1) | 0.44 (5) | 0.53 (1) | 0.00 (1) | 1 | own on arch |
| DCC-GARCH-t | **0.486** | **0.0079** | **0.264** | **0.146** | **8.39** | **1.43** | 0.163 | 3.45 | 0.146 | 0.973 | 2.10 (2) | 0.32 (4) | 0.68 (2) | 1.53 (2) | 1 | own |
| Regime Gaussian | **0.156** | **0.00264** | **0.263** | **0.101** | 2.69 | 1.26 | **0.127** | 0.0613 | 0.0174 | 0.978 | 5.09 (5) | 0.10 (1) | 1.13 (3) | 1.75 (4) | 1 | own |
| Gaussian HMM | **0.159** | **0.0027** | **0.263** | 0.062 | 4.74 | 1.24 | **0.127** | 0.172 | 0.0228 | 0.986 | 5.05 (4) | 0.10 (2) | 1.13 (4) | 1.96 (6) | 1 | hmmlearn |

Bold = the seed-mean lies inside the real-data 95% band (stat metrics) or under the real-vs-real noise ceiling (distance metrics). n/p = the generator produced too few crisis-labelled windows for the metric to exist. Scores are mean errors in units of real sampling noise (lower is better).