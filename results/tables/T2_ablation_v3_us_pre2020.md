### CG-NSDE e3 ablations, market us, fit 'pre2020'

| Model | corr_fro | corr_mae | pc1_share | crisis_freq | boost_pct | cvar_ratio | tail_coexc | kurt_med | acf1_sq | disc_feat | A score (rank) | B score (rank) | C score (rank) | D score (rank) | seeds | impl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Real data** (95% band) | ≤ 6.77 | ≤ 0.135 | 0.438 [0.289, 0.535] | 0.477 [0.302, 0.665] | 33.7 [8.79, 57.6] | 2.03 [1.33, 2.9] | 0.311 [0.174, 0.409] | 8.38 [2.66, 9.93] | 0.263 [0.0511, 0.331] | 0.551 [0.5, 0.612] | – | – | – | – | – | – |
| CG-NSDE (ours, post hoc) | **5.11 ± 0.086** | **0.0856 ± 0.0016** | **0.417 ± 0.0084** | 0.136 ± 0.0051 | **38 ± 2.2** | **2.65 ± 0.29** | **0.311 ± 0.0032** | 19.1 ± 4.8 | **0.142 ± 0.034** | 0.713 ± 0.015 | 1.47 (2) | 0.49 (2) | 0.65 (2) | 2.58 (2) | 3 | own |
| CG-NSDE / no_regime | **5.07 ± 0.07** | **0.0848 ± 0.0012** | **0.422 ± 0.0074** | 0.143 ± 0.0023 | **36.1 ± 2.1** | **2.59 ± 0.19** | **0.321 ± 0.0091** | 11.2 ± 2.1 | **0.141 ± 0.031** | 0.731 ± 0.014 | 0.90 (1) | 0.48 (1) | 0.60 (1) | 2.54 (1) | 3 | own |

Bold = the seed-mean lies inside the real-data 95% band (stat metrics) or under the real-vs-real noise ceiling (distance metrics). n/p = the generator produced too few crisis-labelled windows for the metric to exist. Scores are mean errors in units of real sampling noise (lower is better).