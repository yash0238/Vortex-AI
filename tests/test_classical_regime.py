"""Tests for crisisbench.generators.classical_regime.   Run: python tests/test_classical_regime.py

Ground truth: a simulated 2-regime Markov-switching Gaussian panel (N=8) with known
per-regime correlation (normal: 0.2 at 1% vol, crisis: 0.7 at 3% vol).
Tolerances (mean absolute off-diagonal error of the recovered regime correlation):
  HMMGaussian     0.03  (states are inferred day by day)
  RegimeGaussian  0.08  (looser on purpose: day labels come from the frozen rule, a
                         trailing 20-day statistic, so regime edges are contaminated)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crisisbench import protocol as P                                    # noqa: E402
from crisisbench.generators.base import FitData                          # noqa: E402
from crisisbench.generators.classical_regime import (HMMGaussian, RegimeGaussian,  # noqa: E402
                                                     simulate_chain, stationary)
from crisisbench.labels import fit_rule                                  # noqa: E402

N, D = 8, 5000
RHO = (0.2, 0.7)
VOL = (0.01, 0.03)
TRANS = np.array([[1 - 1 / 680, 1 / 680], [1 / 120, 1 - 1 / 120]])       # about 15% crisis days
OFF = ~np.eye(N, dtype=bool)


def equi(rho):
    return rho * np.ones((N, N)) + (1 - rho) * np.eye(N)


def make_panel(seed=7):  # seed chosen so the realised crisis share (0.15) matches the 85th-percentile labelling rule
    rng = np.random.default_rng(seed)
    s = simulate_chain(TRANS, D, rng)
    X = np.empty((D, N))
    for k in (0, 1):
        m = s == k
        X[m] = rng.standard_normal((m.sum(), N)) @ np.linalg.cholesky(VOL[k] ** 2 * equi(RHO[k])).T
    fd = FitData("sim", X, pd.bdate_range("2000-01-03", periods=D), [f"A{i}" for i in range(N)],
                 np.zeros(D, bool))
    fd.rule = fit_rule(fd.windows())
    return fd, s


def corr_err(G, rho):
    c = np.corrcoef(G.reshape(-1, N), rowvar=False)
    return float(np.abs(c - equi(rho))[OFF].mean())


def check_samples(g):
    a = g.sample(50, np.random.default_rng(5))
    b = g.sample(50, np.random.default_rng(5))
    c = g.sample(50, np.random.default_rng(6))
    assert a.shape == (50, P.T, N) and np.isfinite(a).all()
    assert np.array_equal(a, b), "sample() is not deterministic given the rng"
    assert not np.array_equal(a, c), "sample() ignores the rng"
    r1 = g.sample_regime(20, 1, np.random.default_rng(5))
    assert r1.shape == (20, P.T, N) and np.isfinite(r1).all()
    assert np.array_equal(r1, g.sample_regime(20, 1, np.random.default_rng(5)))
    try:
        g.sample_regime(5, 2, np.random.default_rng(0))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid regime accepted")


def check_recovery(g, tol, true_crisis_share):
    Gc = g.sample_regime(400, 1, np.random.default_rng(1))
    Gn = g.sample_regime(400, 0, np.random.default_rng(2))
    ec, en = corr_err(Gc, RHO[1]), corr_err(Gn, RHO[0])
    print(f"  {g.__class__.__name__}: corr MAE crisis {ec:.4f} normal {en:.4f} (tol {tol})")
    assert ec < tol and en < tol, (ec, en)
    # the two regimes must not be interchangeable (a swapped handle fails here)
    assert corr_err(Gc, RHO[0]) > 0.3 and corr_err(Gn, RHO[1]) > 0.3
    vc, vn = Gc.reshape(-1, N).std(0).mean(), Gn.reshape(-1, N).std(0).mean()
    assert abs(vc / VOL[1] - 1) < 0.1 and abs(vn / VOL[0] - 1) < 0.1, (vc, vn)
    # unconditional path: pooled variance is the stationary mixture of the two regimes
    G = g.sample(3000, np.random.default_rng(3))
    lab = np.abs(G.reshape(-1, N)).mean(1) > 0.014   # crisis-day proxy: mean |r| is ~0.024 in crisis, ~0.008 in calm
    assert abs(lab.mean() - true_crisis_share) < 0.08, (lab.mean(), true_crisis_share)
    assert G.reshape(-1, N).std(0).mean() > 1.2 * VOL[0], "unconditional path never visits the crisis state"


@pytest.fixture(scope="module")
def _panel():
    return make_panel()


@pytest.fixture(scope="module")
def fd(_panel):
    return _panel[0]


@pytest.fixture(scope="module")
def s(_panel):
    return _panel[1]


def test_chain():
    s = simulate_chain(TRANS, 400_000, np.random.default_rng(0))
    C = np.zeros((2, 2))
    np.add.at(C, (s[:-1], s[1:]), 1)
    assert np.abs(C / C.sum(1, keepdims=True) - TRANS).max() < 2e-3
    assert np.allclose(stationary(TRANS), [0.85, 0.15])


def test_regime_gaussian(fd, s):
    g = RegimeGaussian()
    info = g.fit(fd, 0)
    print("  regime_gaussian info:", {k: info[k] for k in ("days_normal", "days_crisis", "shrunk", "stationary_crisis")})
    assert info["days_normal"] + info["days_crisis"] == D and info["shrunk"] == [False, False]
    assert np.allclose(g.trans.sum(1), 1)
    check_samples(g)
    check_recovery(g, 0.08, s.mean())


def test_regime_shrinkage(fd):
    """A regime with fewer than 3N days is shrunk toward the pooled covariance and says so."""
    X = fd.returns[:400]
    small = FitData("sim", X, fd.dates[:400], fd.tickers, np.zeros(400, bool))
    small.rule = fit_rule(small.windows())
    lab = small.window_labels()
    day = np.concatenate([np.full(P.T - 1, lab[0]), lab])
    n1 = int(day.sum())
    assert 2 <= n1 < 3 * 47, n1
    wide = FitData("sim", np.tile(X, (1, 6))[:, :47] + 1e-4 * np.random.default_rng(0).standard_normal((400, 47)),
                   fd.dates[:400], [f"A{i}" for i in range(47)], np.zeros(400, bool))
    wide.rule = fit_rule(wide.windows())
    g = RegimeGaussian()
    info = g.fit(wide, 0)
    assert info["shrunk"][1] and 0 < info["shrink_weight"][1] < 1, info
    assert abs(info["shrink_weight"][1] - info["days_crisis"] / (3 * 47)) < 1e-12
    assert np.isfinite(g.sample(10, np.random.default_rng(0))).all()


def test_hmm_gaussian(fd, s):
    g = HMMGaussian(restarts=2)
    info = g.fit(fd, 0)
    print("  hmm_gaussian K:", info["K"], "bic:", [round(c["bic"]) for c in info["candidates"]])
    assert info["K"] == 2, "BIC should pick the true number of states"
    assert g.crisis_state != g.normal_state
    assert np.abs(np.sort(np.diag(g.trans)) - np.sort(np.diag(TRANS))).max() < 0.01
    check_samples(g)
    check_recovery(g, 0.03, s.mean())
    # Use the SAME restart budget: a different number of restarts can legitimately
    # select a different best fit, which is not a determinism failure.
    g2 = HMMGaussian(restarts=g.restarts)
    g2.fit(fd, 0)
    # hmmlearn's EM runs through multithreaded BLAS: repeated fits differ by ~1e-15.
    # That is numerical noise, not a determinism failure; compare with a tight tolerance.
    assert np.allclose(g.sample(5, np.random.default_rng(0)), g2.sample(5, np.random.default_rng(0)),
                       rtol=1e-10, atol=1e-12), "fit is not deterministic given the seed"


def test_hmm_degenerate_guard(fd):
    """With too few days for 3N per state at any K, fit must raise instead of returning a model."""
    tiny = FitData("sim", fd.returns[:40], fd.dates[:40], fd.tickers, np.zeros(40, bool))
    try:
        HMMGaussian(restarts=1, n_iter=20).fit(tiny, 0)
    except RuntimeError:
        return
    raise AssertionError("degenerate HMM accepted")


if __name__ == "__main__":
    fd, s = make_panel()
    print(f"simulated panel: {D} days, true crisis share {s.mean():.3f}")
    test_chain()
    test_regime_gaussian(fd, s)
    test_regime_shrinkage(fd)
    test_hmm_gaussian(fd, s)
    test_hmm_degenerate_guard(fd)
    print("OK")
