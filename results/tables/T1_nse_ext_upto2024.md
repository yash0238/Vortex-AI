### Main comparison, market nse_ext, fit 'upto2024' (395 real windows, 5 crisis)

| Model | corr_fro | corr_mae | pc1_share | crisis_freq | boost_pct | cvar_ratio | tail_coexc | kurt_med | acf1_sq | disc_feat | A score (rank) | B score (rank) | C score (rank) | D score (rank) | seeds | impl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Real data** (95% band) | ≤ 4.47 | ≤ 0.077 | 0.299 [0.251, 0.349] | 0.0127 [0, 0.0228] | nan [nan, nan] | nan [nan, nan] | 0.214 [0.13, 0.262] | 2.81 [1.77, 3.2] | 0.0868 [0.067, 0.0987] | 0.77 [0.5, 0.878] | – | – | – | – | – | – |
| CG-NSDE v1 (preregistered) | 4.55 ± 0.28 | 0.0758 ± 0.0043 | **0.328 ± 0.012** | 0.335 ± 0.13 | 4.13 ± 1.7 | 1.28 ± 0.0062 | **0.2 ± 0.0068** | 1.53 ± 0.052 | 0.00996 ± 0.0012 | 0.924 ± 0.036 | 2.52 (1) | 0.89 (1) | n/p | 2.05 (1) | 3 | own |

Bold = the seed-mean lies inside the real-data 95% band (stat metrics) or under the real-vs-real noise ceiling (distance metrics). n/p = the generator produced too few crisis-labelled windows for the metric to exist. Scores are mean errors in units of real sampling noise (lower is better).