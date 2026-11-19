### CG-NSDE e3 ablations, market kr, fit 'pre2020'

| Model | corr_fro | corr_mae | pc1_share | crisis_freq | boost_pct | cvar_ratio | tail_coexc | kurt_med | acf1_sq | disc_feat | A score (rank) | B score (rank) | C score (rank) | D score (rank) | seeds | impl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Real data** (95% band) | ≤ 3.8 | ≤ 0.0705 | 0.336 [0.269, 0.406] | 0.203 [0.0859, 0.335] | 16.4 [0.653, 29.9] | 1.86 [1.2, 2.34] | 0.252 [0.15, 0.34] | 6.65 [2.83, 7.68] | 0.207 [0.0749, 0.242] | 0.582 [0.5, 0.642] | – | – | – | – | – | – |
| CG-NSDE (ours, post hoc) | 7.48 ± 0.14 | 0.14 ± 0.0033 | 0.223 ± 0.0045 | **0.143 ± 0.0095** | **19.2 ± 1.5** | **1.8 ± 0.035** | 0.131 ± 0.0042 | 2.82 ± 0.37 | 0.0649 ± 0.012 | 0.793 ± 0.013 | 1.33 (1) | 1.82 (2) | 0.44 (1) | 3.11 (2) | 3 | own |
| CG-NSDE / no_regime | 7.43 ± 0.023 | 0.139 ± 0.00057 | 0.224 ± 0.0011 | **0.131 ± 0.011** | **17.6 ± 3.1** | **1.65 ± 0.1** | 0.134 ± 0.00091 | 2.59 ± 0.099 | 0.0584 ± 0.0046 | 0.793 ± 0.016 | 1.37 (2) | 1.80 (1) | 0.52 (2) | 2.70 (1) | 3 | own |

Bold = the seed-mean lies inside the real-data 95% band (stat metrics) or under the real-vs-real noise ceiling (distance metrics). n/p = the generator produced too few crisis-labelled windows for the metric to exist. Scores are mean errors in units of real sampling noise (lower is better).