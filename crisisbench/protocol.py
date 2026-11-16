"""Frozen evaluation protocol for CrisisBench (VORTEX-AI).

Everything a reviewer could accuse us of tuning after seeing results lives
here, in one file, and is dumped verbatim to results/00_protocol/protocol.json
by `python -m crisisbench.run_protocol` BEFORE any generator is scored.

Do not edit a value in this file after results exist without adding an entry
to results/CHANGELOG.md.
"""
from __future__ import annotations

PROTOCOL_VERSION = "1.0.0"
FROZEN_ON = "2026-11-12"

# ----------------------------------------------------------------- windows
T = 60                      # trading days per window
STRIDE_REAL = 1             # real windows are every-day overlapping windows

# ------------------------------------------------------------------- splits
# Two fits per generator and seed.
#   "full"    : estimated on the whole sample, scored against the whole sample
#               (in-sample fidelity; the usual synthetic-data setting).
#   "pre2020" : estimated on data up to TRAIN_END only, scored against the
#               post-TRAIN_END sample (held-out crises: COVID-2020, 2022, 2024).
# Neural generators use the VAL block only for early stopping / checkpoint
# selection. Classical generators have no early stopping and are estimated on
# every day of the fit period.
SAMPLE_START = "2010-11-08"
SAMPLE_END = "2024-12-31"
TRAIN_END = "2019-12-31"
VAL_START = "2018-01-01"
VAL_END = "2019-12-31"
PURGE = T                   # windows within PURGE days of a split edge are dropped
FITS = ("full", "pre2020")

# Held-out crisis episodes (window END dates), used by the held-out table.
HELDOUT_EPISODES = {
    "covid_2020": ("2020-03-01", "2020-07-31"),
    "shock_2022": ("2022-02-01", "2022-06-30"),
    "mid_2024": ("2024-06-01", "2024-07-31"),
}

# -------------------------------------------------------------------- seeds
SEEDS = (0, 1, 2, 3, 4)
N_GEN = 2000                # generated windows per (generator, seed, fit)
N_GEN_POOL_RISK = 20000     # generated windows per fit for the VaR/ES backtest

# ------------------------------------------------------- crisis labelling rule
# The rule is applied IDENTICALLY to real and to generated windows, using only
# the window's own returns, so a generator can never be handed a label.
# For a (T, N) window of log returns:
#   disp_t   = cross-sectional std of returns on day t
#   rv       = mean of disp_t over the last VOL_LOOKBACK days of the window
#   dd       = fraction of assets whose last-DD_LOOKBACK-day summed return < DD_THRESH
#   cp       = fraction of assets with |r| > CIRCUIT_PROX on the last day
#   score    = W_VOL * rv / RV_REF + W_DD * dd + W_CP * cp
#   crisis   = score >= SCORE_THR
# RV_REF (99th percentile of rv) and SCORE_THR (CRISIS_PCT-th percentile of
# score) are estimated on the REAL windows of the fit period only and then
# frozen: no look-ahead, and no re-estimation on generated data.
LABEL_RULE = dict(
    name="composite_v1",
    VOL_LOOKBACK=20,
    DD_LOOKBACK=5,
    DD_THRESH=-0.05,
    CIRCUIT_PROX=0.095,
    W_VOL=0.5, W_DD=0.3, W_CP=0.2,
    RV_REF_Q=0.99,
    CRISIS_PCT=85.0,
)
# Robustness rule (reported in the label-sensitivity table, never for ranking):
# crisis = 20-day realised volatility of the equal-weight portfolio at or above
# its CRISIS_PCT-th percentile on the fit-period real windows.
LABEL_RULE_ALT = dict(name="ewvol_v1", VOL_LOOKBACK=20, CRISIS_PCT=85.0)

MIN_CRISIS_WINDOWS = 20     # fewer generated crisis windows => crisis metrics are "not produced"

# ------------------------------------------------------------------ metrics
# Family A: marginal and temporal.  B: static dependence.  C: crisis
# dependence.  D: distinguishability.  Names must match crisisbench.metrics.
METRIC_FAMILIES = {
    "A": ["std_ratio", "kurt_med", "acf1_ret", "acf1_sq", "acf5_sq_mean", "skew_ew"],
    "B": ["corr_fro", "corr_mae", "eig_dist", "pc1_share"],
    "C": ["crisis_freq", "boost_pct", "tail_coexc", "cvar_ratio", "pc1_share_gap", "connect_gap"],
    "D": ["disc_feat", "disc_raw"],
}
TAIL_Q = 0.05               # tail probability for co-exceedance and CVaR
CONNECT_RHO = 0.5           # |rho| threshold for the correlation-network density

# Uncertainty on REAL statistics: circular block bootstrap over window indices.
BOOT_B = 500
BOOT_BLOCK = 2 * T
CI_LEVEL = 0.95
# Normalised error of a generator on a metric = |gen - real| / half-width of the
# real CI (so every metric is in units of "real sampling noise"); capped so one
# metric cannot dominate a family score.  Family score = mean of capped errors.
NORM_ERR_CAP = 10.0

# Discriminator (family D): post-hoc classifier, real (label 1) vs generated.
DISC = dict(model="HistGradientBoosting", max_iter=200, max_depth=4, lr=0.1,
            n_per_class=1000, folds=5, purge=T, raw_pca=64)

# ------------------------------------------------------ downstream risk test
RISK = dict(
    portfolios=("equal_weight", "dirichlet_seed7"),
    levels=(0.95, 0.975, 0.99),
    oos_start="2020-01-01",
    refit_dates=("2019-12-31", "2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31"),
    state="rv20_ew",        # conditioning state: 20-day realised vol of the portfolio
    knn_frac=0.05,          # neighbours = this fraction of the generated pool
    hs_window=500,          # historical-simulation lookback (days)
)

# ----------------------------------------------- preregistered hypotheses
# Thresholds are judgment calls made before the runs; they are not evidence.
HYPOTHESES = {
    "H1": "Kendall tau between the family-B ranking and the family-C ranking of generators is below 0.3.",
    "H2": "At least one non-neural regime model (regime-conditional Gaussian or HMM) has a family-C score "
          "within one seed standard deviation of CG-NSDE (or better).",
    "H3": "Kendall tau between family-C rank and VaR/ES backtest rank exceeds 0.5 and exceeds the tau of "
          "family B, with non-overlapping bootstrap intervals.",
    "H4": "The sign of (tau_B_vs_C < 0.3) replicates on at least one of the two other markets.",
    "H5": "Injecting a single bad tick flips the order of at least one generator pair on at least one metric.",
}
GATES = {
    "G1": "PASS if every regime-aware generator beats every static generator on the family-C score AND no "
          "regime-conditional Gaussian/HMM matches CG-NSDE (H2 false). FAIL => benchmark leads the paper.",
    "G2": "PASS if CG-NSDE's held-out boost_pct lies inside the real 95% interval on at least 2 of 3 episodes.",
    "G3": "PASS if H3 holds.",
    "G4": "PASS if H4 holds.",
    "KILL": "If static generators also rank well on family C across seeds (H1 fails on every market), "
            "drop the benchmark framing.",
}

# Generator registry: name -> (family tag, implementation source).
STATIC = ("iid_gaussian", "t_ppca")
GENERATORS = {
    "iid_gaussian": ("static", "own"),
    "t_ppca": ("static", "own"),
    "stationary_bootstrap": ("resampling", "own (arch bootstrap)"),
    "garch_fhs": ("dynamic-classical", "own on arch"),
    "dcc_garch": ("dynamic-classical", "own"),
    "regime_gaussian": ("regime", "own"),
    "hmm_gaussian": ("regime", "hmmlearn"),
    "timegan": ("neural", "own PyTorch reimplementation"),
    "quantgan": ("neural", "own PyTorch reimplementation"),
    "timevae": ("neural", "own PyTorch reimplementation"),
    "cgnsde": ("neural-ours", "own"),
    "cgnsde2": ("neural-ours", "own"),     # amendment 2026-11-16 (post hoc): CG-NSDE v2, added after v1 results
}


def as_dict() -> dict:
    """Protocol as a plain JSON-serialisable dict."""
    g = globals()
    keys = [k for k in g if k.isupper()]
    return {k: g[k] for k in keys}
