"""Crisis labelling rule, applied identically to real and generated windows.

A window is labelled from its own returns only (see protocol.LABEL_RULE).
Thresholds are estimated once on the real windows of the fit period
(`fit_rule`) and then frozen; `label_windows` never re-estimates anything.
"""
from __future__ import annotations

import numpy as np

from . import protocol as P


def score_parts(W: np.ndarray) -> dict[str, np.ndarray]:
    """W: (n, T, N) log returns -> per-window rv, dd, cp, ewvol."""
    r = P.LABEL_RULE
    L, D = r["VOL_LOOKBACK"], r["DD_LOOKBACK"]
    disp = W[:, -L:, :].std(axis=2, ddof=1)             # (n, L) cross-sectional dispersion
    rv = disp.mean(axis=1)
    dd = (W[:, -D:, :].sum(axis=1) < r["DD_THRESH"]).mean(axis=1)
    cp = (np.abs(W[:, -1, :]) > r["CIRCUIT_PROX"]).mean(axis=1)
    ew = W[:, -P.LABEL_RULE_ALT["VOL_LOOKBACK"]:, :].mean(axis=2)
    return dict(rv=rv, dd=dd, cp=cp, ewvol=ew.std(axis=1, ddof=1))


def fit_rule(W_fit: np.ndarray) -> dict:
    """Estimate the frozen thresholds on real fit-period windows."""
    r = P.LABEL_RULE
    p = score_parts(W_fit)
    rv_ref = float(np.quantile(p["rv"], r["RV_REF_Q"]))
    score = r["W_VOL"] * p["rv"] / rv_ref + r["W_DD"] * p["dd"] + r["W_CP"] * p["cp"]
    thr = float(np.percentile(score, r["CRISIS_PCT"]))
    alt_thr = float(np.percentile(p["ewvol"], P.LABEL_RULE_ALT["CRISIS_PCT"]))
    return dict(rv_ref=rv_ref, score_thr=thr, alt_thr=alt_thr, n_fit_windows=int(len(W_fit)))


def label_windows(W: np.ndarray, rule: dict, alt: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels int8 (n,), score (n,)). alt=True uses the robustness rule."""
    p = score_parts(W)
    if alt:
        return (p["ewvol"] >= rule["alt_thr"]).astype(np.int8), p["ewvol"]
    r = P.LABEL_RULE
    score = r["W_VOL"] * p["rv"] / rule["rv_ref"] + r["W_DD"] * p["dd"] + r["W_CP"] * p["cp"]
    return (score >= rule["score_thr"]).astype(np.int8), score
