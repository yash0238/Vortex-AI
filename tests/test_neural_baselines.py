"""Tests for the three neural baselines (tiny budgets, synthetic data).  Run: python tests/test_neural_baselines.py"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crisisbench import protocol as P                                    # noqa: E402
from crisisbench.generators import get                                   # noqa: E402
from crisisbench.generators.base import FitData                          # noqa: E402
from crisisbench.generators import neural_quantgan as Q                  # noqa: E402
from crisisbench.generators.neural_timegan import MinMaxScaler           # noqa: E402

N, D = 6, 700


def make_data(seed: int = 0) -> FitData:
    """One-factor returns with persistent stochastic volatility and asset-specific scales."""
    rng = np.random.default_rng(seed)
    logv = np.zeros(D)
    for t in range(1, D):
        logv[t] = 0.97 * logv[t - 1] + 0.25 * rng.standard_normal()
    vol = np.exp(logv - logv.mean())[:, None]
    beta = rng.uniform(0.5, 1.5, N)
    scale = np.linspace(0.005, 0.03, N)                                   # stds differ by a factor 6
    r = (beta * rng.standard_normal((D, 1)) + 0.7 * rng.standard_t(4, (D, N))) * vol * scale
    val = np.zeros(D, bool)
    val[450:600] = True
    return FitData("synthetic", r, pd.bdate_range("2015-01-01", periods=D), [f"A{i}" for i in range(N)], val)


@pytest.fixture(scope="module")
def data():
    return make_data()


TINY = {
    "timevae": dict(epochs=3),
    "quantgan": dict(iters=3, batch=16, hidden=16),
    "timegan": dict(iters=4, batch=32, hidden_dim=12),
}


def test_shapes_finite_deterministic(data):
    for name, hp in TINY.items():
        outs = []
        for _ in range(2):
            gen = get(name, **hp)
            assert gen.family == "neural" and gen.impl_source == "own PyTorch reimplementation"
            info = gen.fit(data, seed=3)
            assert "history" in info and len(info["history"]) > 0
            outs.append(gen.sample(7, np.random.default_rng(11)))
        G = outs[0]
        assert G.shape == (7, P.T, N), (name, G.shape)
        assert np.isfinite(G).all(), name
        assert G.std() > 0, name
        assert np.array_equal(outs[0], outs[1]), f"{name}: not deterministic given seed"
        other = gen.sample(7, np.random.default_rng(12))
        assert not np.array_equal(G, other), f"{name}: sample ignores the rng"
        gen2 = get(name, **hp)
        gen2.fit(data, seed=4)
        assert not np.array_equal(G, gen2.sample(7, np.random.default_rng(11))), f"{name}: fit ignores the seed"
        with tempfile.TemporaryDirectory() as tmp:
            gen.save(Path(tmp) / "sub" / "seed0")
            assert (Path(tmp) / "sub" / "seed0.pt").stat().st_size > 0, name
        # sampling must not depend on fit-time validation data being supplied again (unconditional)
        assert G.dtype == np.float64


def test_lambert(data):
    rng = np.random.default_rng(0)
    u = rng.standard_normal(5000) * 2.0
    for delta in (0.0, 0.05, 0.3, 1.0):
        z = Q.lambert_forward(u, delta)
        assert np.abs(Q.lambert_inverse(z, delta) - u).max() < 1e-6, delta
        assert np.abs(Q.lambert_forward(Q.lambert_inverse(z, delta), delta) - z).max() < 1e-6 * (1 + np.abs(z).max())
    # IGMM recovers the parameters of a simulated Lambert W x Gaussian sample
    uu = rng.standard_normal(200_000)
    y = 0.3 + 1.7 * Q.lambert_forward(uu, 0.2)
    mu, sigma, delta = Q.igmm(y)
    assert abs(delta - 0.2) < 0.02 and abs(mu - 0.3) < 0.02 and abs(sigma - 1.7) < 0.03, (mu, sigma, delta)
    assert Q.delta_gmm(rng.uniform(-1, 1, 5000)) == 0.0                   # platykurtic: no transform
    # scaler on heavy-tailed data: exact round trip, Gaussianised kurtosis, unit variance
    X = data.returns[~data.val_mask]
    sc = Q.LambertScaler().fit(X)
    assert (sc.delta > 0).all(), sc.delta
    Y = sc.transform(X)
    assert np.abs(sc.inverse(Y) - X).max() < 1e-6 * np.abs(X).max()
    assert np.abs(sc.inverse(Y) - X).max() < 1e-6
    kurt = ((Y - Y.mean(0)) ** 4).mean(0) / Y.var(0) ** 2
    assert np.abs(kurt - 3).max() < 1e-3, kurt
    assert np.abs(Y.std(0) - 1).max() < 1e-8 and np.abs(Y.mean(0)).max() < 1e-8
    assert Q.RF == 127
    import torch
    g = Q.TCN(3, N, 8, pad=False)
    assert g(torch.zeros(2, 3, P.T + Q.RF - 1)).shape == (2, N, P.T)
    # receptive field is exactly 127: output 0 depends on inputs 0..126 and on nothing later
    x = torch.randn(1, 3, P.T + Q.RF - 1, requires_grad=True)
    grad = torch.autograd.grad(g(x)[0, :, 0].sum(), x)[0].abs().sum((0, 1))
    assert (grad[:Q.RF] > 0).all() and (grad[Q.RF:] == 0).all()
    d = Q.TCN(N, 1, 8, pad=True)
    x = torch.randn(1, N, P.T, requires_grad=True)
    assert d(x).shape == (1, 1, P.T)
    grad = torch.autograd.grad(d(x)[0, 0, -1], x)[0].abs().sum((0, 1))
    assert (grad > 0).all()                                               # the score sees the whole window
    grad = torch.autograd.grad(d(x)[0, 0, 10], x)[0].abs().sum((0, 1))
    assert (grad[11:] == 0).all()                                         # causal


def test_timegan(data):
    W = data.windows()[data.split_windows()[0]]
    sc = MinMaxScaler().fit(W)
    Y = sc.transform(W)
    assert Y.min() == 0.0 and Y.max() == 1.0 and np.allclose(Y.reshape(-1, N).min(0), 0) \
        and np.allclose(Y.reshape(-1, N).max(0), 1)
    assert np.abs(sc.inverse(Y) - W).max() < 1e-12
    gen = get("timegan", iters=150, batch=32, hidden_dim=16)
    info = gen.fit(data, seed=0)
    h1 = [h["e_loss"] for h in info["history"]["phase1"]]
    assert np.mean(h1[-5:]) < 0.9 * np.mean(h1[:5]), (h1[:5], h1[-5:])
    assert len(info["history"]["phase2"]) > 0 and len(info["history"]["phase3"]) > 0
    G = gen.sample(50, np.random.default_rng(0))
    lo, hi = sc.lo, sc.hi                                                 # sigmoid + min-max: bounded output
    assert (G >= lo - 1e-12).all() and (G <= hi + 1e-12).all()


def test_timevae_std(data):
    gen = get("timevae", epochs=40, patience=40)
    info = gen.fit(data, seed=0)
    hist = info["history"]
    assert hist[info["best_epoch"]]["val"] == min(h["val"] for h in hist)  # best checkpoint is the restored one
    assert hist[-1]["train"] < hist[0]["train"]
    G = gen.sample(400, np.random.default_rng(0))
    ratio = G.reshape(-1, N).std(0) / data.returns[~data.val_mask].std(0)
    print("  timevae generated/real std per asset:", np.round(ratio, 2), "best epoch", info["best_epoch"])
    assert (ratio > 1 / 3).all() and (ratio < 3).all(), ratio
    # scaler comes from training days only
    assert np.allclose(gen.mu, data.returns[~data.val_mask].mean(0))


if __name__ == "__main__":
    import torch
    torch.set_num_threads(4)
    data = make_data()
    for fn in (test_lambert, test_shapes_finite_deterministic, test_timegan, test_timevae_std):
        t0 = time.time()
        fn(data)
        print(fn.__name__, f"passed ({time.time() - t0:.0f} s)", flush=True)
    print("OK")
