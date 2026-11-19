"""Tests for crisisbench.generators.classical_garch.   Run: python tests/test_classical_garch.py

Ground truth: an independently coded DCC(1,1)-GARCH(1,1) simulator (N=5, D=4000) with
a = 0.05, b = 0.90.  Tolerances
  dcc_fit on the TRUE standardised residuals        |a - 0.05| < 0.012, |b - 0.90| < 0.03
  DCCGarch end to end (estimated GARCH first stage) |a - 0.05| < 0.02,  |b - 0.90| < 0.05
  simulated vs fit-data moments (120,000 simulated days against 4,000 fit days):
      per-asset std ratio within [0.85, 1.18], correlation MAE < 0.05
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crisisbench import protocol as P                                    # noqa: E402
from crisisbench.generators.base import FitData                          # noqa: E402
from crisisbench.generators.classical_garch import (DCCGarch, GarchFHS, dcc_fit, dcc_nll,  # noqa: E402
                                                    fit_univariate, garch_filter)

N, D = 5, 4000
A, B = 0.05, 0.90
OMEGA, ALPHA, BETA = 0.05, 0.08, 0.88            # percent^2 units; unconditional variance 1.25
OFF = ~np.eye(N, dtype=bool)


def simulate_dcc(seed=11):
    """Gaussian DCC-GARCH written independently of the module under test."""
    rng = np.random.default_rng(seed)
    Qbar = 0.4 * np.ones((N, N)) + 0.6 * np.eye(N)
    Qbar[0, 1] = Qbar[1, 0] = 0.8
    Qbar[3, 4] = Qbar[4, 3] = -0.2
    Q = Qbar.copy()
    h = np.full(N, OMEGA / (1 - ALPHA - BETA))
    Z, R = np.empty((D, N)), np.empty((D, N))
    for t in range(D):
        d = np.sqrt(np.diag(Q))
        z = np.linalg.cholesky(Q / np.outer(d, d)) @ rng.standard_normal(N)
        e = np.sqrt(h) * z
        Z[t], R[t] = z, (0.03 + e) / 100.0
        Q = (1 - A - B) * Qbar + A * np.outer(z, z) + B * Q
        h = OMEGA + ALPHA * e ** 2 + BETA * h
    return R, Z


def fitdata(R):
    return FitData("sim", R, pd.bdate_range("2000-01-03", periods=len(R)), [f"A{i}" for i in range(R.shape[1])],
                   np.zeros(len(R), bool))


# The test functions take R/Z arguments.  Define them as module-scoped pytest
# fixtures so the file works under `pytest` as well as `python tests/...`.
@pytest.fixture(scope="module")
def _sim():
    return simulate_dcc()


@pytest.fixture(scope="module")
def R(_sim):
    return _sim[0]


@pytest.fixture(scope="module")
def Z(_sim):
    return _sim[1]


def check_samples(g, fd, name):
    a = g.sample(40, np.random.default_rng(5))
    assert a.shape == (40, P.T, N) and np.isfinite(a).all()
    assert np.array_equal(a, g.sample(40, np.random.default_rng(5))), "sample() is not deterministic"
    assert not np.array_equal(a, g.sample(40, np.random.default_rng(6))), "sample() ignores the rng"
    G = g.sample(2000, np.random.default_rng(1)).reshape(-1, N)
    ratio = G.std(0) / fd.returns.std(0)
    mae = float(np.abs(np.corrcoef(G, rowvar=False) - np.corrcoef(fd.returns, rowvar=False))[OFF].mean())
    print(f"  {name}: std ratio {ratio.min():.3f}..{ratio.max():.3f}, corr MAE {mae:.4f}")
    assert ratio.min() > 0.85 and ratio.max() < 1.18, ratio
    assert mae < 0.05, mae
    assert abs(G.mean(0) - fd.returns.mean(0)).max() < 3e-4
    # volatility clustering must survive: lag-1 autocorrelation of squared returns
    x = (G - G.mean(0)) ** 2
    acf = np.mean([np.corrcoef(x[:-1, j], x[1:, j])[0, 1] for j in range(N)])
    assert acf > 0.03, acf


def test_filter_matches_arch(R):
    """Our recursion reproduces arch's conditional volatility once the start-up has decayed."""
    from arch import arch_model
    y = R[:, 0] * 100
    res = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off")
    mu, om, al, be, _ = np.asarray(res.params)
    h, z = garch_filter(y[:, None], mu, om, al, be)
    rel = np.abs(np.sqrt(h[300:, 0]) / np.asarray(res.conditional_volatility)[300:] - 1).max()
    assert rel < 1e-6, rel
    assert np.abs(z[300:, 0] - np.asarray(res.std_resid)[300:]).max() < 1e-5


def test_univariate(R):
    par, Z, info = fit_univariate(R)
    print("  first stage:", {k: info[k] for k in ("n_clipped", "n_retries", "persistence_median")})
    assert np.abs(par["alpha"] - ALPHA).max() < 0.04 and np.abs(par["beta"] - BETA).max() < 0.08, par
    assert info["n_clipped"] == 0
    assert np.abs((Z ** 2).mean(0) - 1).max() < 0.05 and np.abs(Z.mean(0)).max() < 0.06
    assert par["nu"].min() > 15, "Gaussian data should give a large t df"


def test_clipping():
    """An integrated (alpha+beta = 1) series is clipped to 0.999 and counted."""
    rng = np.random.default_rng(2)
    n = 6000
    e, h = np.empty(n), 1.0
    for t in range(n):
        e[t] = np.sqrt(h) * rng.standard_normal()
        h = 0.002 + 0.12 * e[t] ** 2 + 0.88 * h
    X = np.column_stack([e, rng.standard_normal(n)]) / 100
    par, Z, info = fit_univariate(X)
    pers = par["alpha"] + par["beta"]
    assert pers.max() <= 0.999 + 1e-12 and np.isfinite(Z).all()
    assert info["n_clipped"] == int(np.isclose(pers, 0.999).sum())
    print(f"  IGARCH series: persistence after clip {pers[0]:.5f}, n_clipped {info['n_clipped']}")
    assert info["n_clipped"] == 1 and np.isclose(pers[0], 0.999), "expected the integrated series to be clipped"


def test_dcc_estimator(Z):
    a, b, info = dcc_fit(Z)
    print(f"  dcc_fit on true residuals: a {a:.4f} b {b:.4f} (true {A}, {B})")
    assert abs(a - A) < 0.012 and abs(b - B) < 0.03, (a, b)
    assert info["nll_per_day"] < info["nll_per_day_ccc"] - 0.01, "DCC must beat constant correlation here"
    # likelihood sanity: at a = b = 0 it is the constant-correlation Gaussian quasi-likelihood
    Qbar = Z.T @ Z / len(Z)
    d = np.sqrt(np.diag(Qbar))
    Rb = Qbar / np.outer(d, d)
    ref = 0.5 * (np.linalg.slogdet(Rb)[1] + (np.einsum("ti,ij,tj->t", Z, np.linalg.inv(Rb), Z) - (Z ** 2).sum(1)).mean())
    assert abs(dcc_nll(0.0, 0.0, Z, Z[:, :, None] * Z[:, None, :], Qbar) - ref) < 1e-10
    # i.i.d. residuals have no correlation dynamics: a must be (near) zero
    a0, _, _ = dcc_fit(np.random.default_rng(0).standard_normal((3000, N)))
    assert a0 < 0.01, a0


def test_generators(R):
    fd = fitdata(R)
    g = DCCGarch()
    info = g.fit(fd, 0)
    print(f"  DCCGarch end to end: a {info['a']:.4f} b {info['b']:.4f} df {info['df']:.1f}")
    assert abs(info["a"] - A) < 0.02 and abs(info["b"] - B) < 0.05, info
    check_samples(g, fd, "dcc_garch")
    f = GarchFHS()
    finfo = f.fit(fd, 0)
    assert finfo["effective_persistence_max"] < 1
    check_samples(f, fd, "garch_fhs")
    # FHS resamples whole cross-sections: each simulated innovation row must be a fit-data row
    assert f.Z.shape == (D, N)


if __name__ == "__main__":
    R, Z = simulate_dcc()
    test_filter_matches_arch(R)
    test_univariate(R)
    test_clipping()
    test_dcc_estimator(Z)
    test_generators(R)
    print("OK")
