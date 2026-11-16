# Assignment: adversarial review of three agent-written modules

You are the REVIEWER, not the author. The authors were interrupted before they could self-review. Assume there
are bugs. Files in scope (you may edit only these):
  crisisbench/generators/cgnsde.py, tests/test_cgnsde.py
  crisisbench/risk.py, tests/test_risk.py
  crisisbench/dgp.py, tests/test_dgp.py

Context you need (read these, do not edit them): docs/briefs/common.md (project rules),
crisisbench/generators/base.py, crisisbench/protocol.py, crisisbench/bench.py (see risk_pool and how generators are
called), crisisbench/labels.py. The lead is already running tests/test_cgnsde.py and a 2-epoch smoke run; you do
not need to run the full CG-NSDE test file (it is slow) unless you change cgnsde.py, and then only the affected
test functions.

What each module must be (the specification the authors had):

CG-NSDE (crisisbench/generators/cgnsde.py): GAT encoder over a per-window k-NN |correlation| graph -> pooled graph
embedding g; GRU posterior q(z0|x) with regime-conditional Gaussian prior (KL ramped); latent SDE
dz = f(z,g,t)dt + sigma(z,g,t)dW (torchsde Euler, 60 steps); per-day multivariate Student-t emission with
covariance B B' + diag(d) (rank_K factors), one shared chi-square mixing variable per day, and a volatility-state
recursion (teacher-forced in training, sequential in sampling). Loss = NLL + beta*KL + lam_corr*L_corr
(per-regime pooled-correlation matching vs TRAIN targets) + lam_con*SupCon (Khosla 2020) + lam_graph*L_graph.
Scaler from training days only; early stopping on validation NLL; best weights restored. Sampling must be fully
parametric and unconditional: [z0, g] drawn from per-regime Gaussian mixtures fitted on TRAINING windows only,
regime drawn with the training crisis frequency; no real window at sampling time. save/load round trip.
Check especially: (1) is the Student-t density correct (Woodbury, log-det, the (nu-2)/nu variance scaling
consistent between log_prob and sampler)? (2) is the vol-state recursion identical in the training likelihood and
in sequential sampling? (3) any leakage: validation or later data in the scaler, correlation targets, mixtures,
graph construction; (4) does validation use the same emission as training; (5) are returns inverse-scaled to raw
units correctly; (6) determinism of sample() given the rng; (7) do the ablation VARIANTS do what their names say
(no_sde really freezes z_t = z0; no_gat really has no message passing; K0 really diagonal); (8) numerical guards
(Cholesky failures, nu near 2, exploding vol state).

Risk (crisisbench/risk.py): VaR/ES are positive loss numbers at tail probability alpha = 1 - level; a violation is
r < -VaR. Check every statistic against its source: Kupiec POF LR; Christoffersen independence and conditional
coverage (empty-cell handling); Basel traffic light from Binomial(n, alpha) cumulative (yellow from 95%, red from
99.99%; n=250 must give green 0-4, yellow 5-9, red 10+); Acerbi-Szekely (2014) Z2 = sum(r_t * I_t / (alpha * ES_t))
/ n + 1 with ES positive (check the sign and the one-sided p-value direction under simulation); FZ0 loss of
Patton-Ziegel-Chen (2019) converted to the positive-loss convention (derive it: in their notation with v = VaR
quantile < 0 and e = ES < 0, L = -1{y<=v}(v - y)/(alpha e) + v/e + log(-e) - 1); empirical ES definition and
tail-count consistency; kNN window selection (really the k nearest states; boundaries); no look-ahead in
historical_simulation and garch_fhs_var_es.

DGP (crisisbench/dgp.py): simulated panels sim_boost00/20/40/80 (2-state Markov one-factor model, population mean
|corr| in crisis = (1 + XX/100) x normal; sim_boost00 has identical correlation in both states) and sim_ccc
(constant-conditional-correlation GARCH, no regimes); contaminated NSE copies nse_c_tick / nse_c_1pct /
nse_c_split. Check true_boost() is the population quantity it claims, the calibration solves what it says, the
panels on disk (data/panels/sim_*, nse_c_*) match the code, and validate() uses the benchmark's own functions.
Then RUN "python -m crisisbench.dgp --validate" (it writes results/01_dgp/metric_validation.csv and README.md;
this was never run) and make sure the CSV has clear column names, including: panel, true_boost_pct,
oracle_boost_pct, rule_boost_pct, rule_lo, rule_hi, label_precision, label_recall.

Method: read every line of the three modules; derive, do not trust; fix real defects in place; strengthen tests
that cannot fail; run tests/test_risk.py and tests/test_dgp.py. Do not refactor for style. Be frugal with tool
calls.

FINAL REPORT (your return value, at most 450 words): defects found (file:line, what, how fixed); what you verified
as correct; the metric_validation table; residual risks; anything the lead must change in files you may not edit.
