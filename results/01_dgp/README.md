# Does boost_pct measure what it claims? (controlled ground truth)

Produced by `python -m crisisbench.dgp --validate` from the simulated panels written by
`python -m crisisbench.dgp --build`. Each panel has the NSE panel's dates and shape. The data come from a
model whose crisis correlation boost is known (see the docstring of `crisisbench/dgp.py`).

| panel | true boost % | pure-state window boost % | oracle-label window boost % | benchmark boost % [95% CI] | precision | recall |
|---|---|---|---|---|---|---|
| sim_boost00 | 0.0 | 0.2 | -4.1 | -4.3 [-13.4, 7.6] | 0.95 | 0.88 |
| sim_boost20 | 20.0 | 18.3 | 6.7 | 6.3 [-3.1, 20.3] | 0.95 | 0.87 |
| sim_boost40 | 40.0 | 36.8 | 17.9 | 17.0 [7.3, 32.0] | 0.95 | 0.87 |
| sim_boost80 | 80.0 | 74.8 | 40.1 | 38.3 [27.1, 53.6] | 0.94 | 0.87 |
| sim_ccc | 0.0 | n/a | n/a | 13.6 [5.8, 20.6] | 0.00 | n/a |

Columns
- **true boost**: from the model's population correlation matrices, 100 * (mean|rho| crisis / mean|rho| normal - 1).
- **pure-state window boost**: the benchmark's window statistic (mean |off-diagonal| of a 60-day sample
  correlation matrix) averaged over 2000 simulated windows lying entirely in one state. No labelling error,
  no mixed windows: only the effect of estimating correlations from 60 days.
- **oracle-label window boost**: the same statistic on the panel's own overlapping windows, a window being
  crisis when at least 10 of its last 20 days are in the true crisis state.
- **benchmark boost**: `boost_pct` exactly as the runner computes it for the `full` fit (frozen composite
  rule fitted on all windows), with the circular block bootstrap interval (B = 200).
- **precision / recall**: rule labels against oracle labels (CSV: label_precision, label_recall).

CSV columns: true_boost_pct (population), pure_window_boost_pct, oracle_boost_pct, rule_boost_pct with
rule_lo / rule_hi (the benchmark's boost_pct and its bootstrap interval), mac_* (mean |correlation|),
crisis_day_share, oracle_crisis_windows, rule_crisis_windows, label_precision, label_recall.

What the table shows
- Finite-sample floor. For independent Gaussian assets (population correlation 0) the mean |sample
  correlation| in a 60-day window of 47 assets is 0.1044, not 0. The window statistic of the normal
  state is 0.284 on the panels against a population value of 0.250. Because this
  upward bias is larger when the true correlation is lower, relative boosts are compressed: a true boost of
  80% is 74.8% even with perfect labels and pure windows.
- With oracle labels on the actual panel the 80% panel gives 40.1%, and the frozen rule gives
  38.3% (interval 27.1 to 53.6).
- Null case. `sim_boost00` (volatility doubles, correlation unchanged) is scored -4.3%
  (interval -13.4 to 7.6); `sim_ccc` (volatility clustering, constant correlation, no regimes) is scored
  13.6% (interval 5.8 to 20.6).
- The benchmark value is therefore an ordinal, biased-towards-zero measure of the population boost: compare
  it between a generator and real data under the same rule, do not read it as the population number.

Limits: one realisation per panel (the bootstrap interval is the only uncertainty shown); a one-factor model
with independent Student-t shocks; the crisis state doubles all volatilities, which makes labelling easier
than on real data.
