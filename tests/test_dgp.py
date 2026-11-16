"""Tests of crisisbench.dgp.  Run: .venv/Scripts/python.exe tests/test_dgp.py

Requires the panels on disk (python -m crisisbench.dgp --build) for the on-disk checks.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crisisbench import dgp as G  # noqa: E402
from crisisbench.data import PANELS, load_panel  # noqa: E402

N = load_panel("nse").n_assets


def test_calibration():
    m = G.model("sim_boost40")
    mac_n = G.mean_offdiag(m["corr_normal"])
    assert abs(mac_n - 0.25) < 1e-12, mac_n
    assert (m["corr_normal"] > 0).all()                         # so mean |rho| = mean rho
    assert abs(np.sqrt(np.diag(m["cov_normal"])).mean() - 0.013) < 1e-12
    for name, xx in G.SIM_BOOSTS.items():
        m = G.model(name)
        tb = G.true_boost(name)
        assert abs(tb - xx) < 1e-8, (name, tb)
        # true_boost is computed from the covariance matrices, not copied from the target:
        d = lambda S: S / np.sqrt(np.outer(np.diag(S), np.diag(S)))
        direct = 100 * (G.mean_offdiag(d(m["cov_crisis"])) / G.mean_offdiag(d(m["cov_normal"])) - 1)
        assert abs(direct - tb) < 1e-10
        # crisis volatility is clearly higher for every asset
        assert (np.diag(m["cov_crisis"]) >= 3.99 * np.diag(m["cov_normal"])).all()
    assert G.true_boost("sim_ccc") == 0.0


def test_boost00_same_correlation():
    m = G.model("sim_boost00")
    assert m["k"] == 1.0
    assert np.allclose(m["corr_normal"], m["corr_crisis"], rtol=0, atol=1e-14)
    assert np.allclose(m["cov_crisis"], G.CRISIS_SCALE ** 2 * m["cov_normal"], rtol=1e-14)
    assert not np.allclose(G.model("sim_boost20")["corr_normal"], G.model("sim_boost20")["corr_crisis"], atol=1e-3)


def test_population_matches_simulation():
    """Monte Carlo: the simulator's moments are the ones true_boost reports."""
    n = 300_000
    for name in ("sim_boost00", "sim_boost40", "sim_boost80"):
        m = G.model(name)
        rng = np.random.default_rng(123)
        mac = {}
        for s, key in ((0, "normal"), (1, "crisis")):
            R = G.simulate_boost(m, np.full(n, s), rng)
            C = np.corrcoef(R, rowvar=False)
            assert np.abs(C - m[f"corr_{key}"]).max() < 0.012, (name, key, np.abs(C - m[f"corr_{key}"]).max())
            assert np.allclose(R.std(axis=0), np.sqrt(np.diag(m[f"cov_{key}"])), rtol=0.02)
            assert abs(R.mean()) < 1e-4
            mac[key] = G.mean_offdiag(C)
        mc_boost = 100 * (mac["crisis"] / mac["normal"] - 1)
        assert abs(mc_boost - G.true_boost(name)) < 1.5, (name, mc_boost, G.true_boost(name))
        print(f"  {name}: Monte-Carlo boost {mc_boost:6.2f}%  population {G.true_boost(name):6.2f}%")
    # a mixed-state path is a different object: pooling the two states does NOT give the normal matrix
    m = G.model("sim_boost80")
    st = G.simulate_chain(n, np.random.default_rng(1))
    R = G.simulate_boost(m, st, np.random.default_rng(2))
    assert G.mean_offdiag(np.corrcoef(R[st == 1], rowvar=False)) > 1.7 * G.mean_offdiag(np.corrcoef(R[st == 0], rowvar=False))


def test_chain():
    s = G.simulate_chain(1_000_000, np.random.default_rng(5))
    assert abs(s.mean() - 0.15) < 0.01, s.mean()
    starts = np.flatnonzero(np.diff(np.r_[0, s]) == 1)
    ends = np.flatnonzero(np.diff(np.r_[s, 0]) == -1)
    dur = (ends - starts + 1).mean()
    assert abs(dur - 30.0) < 1.0, dur
    print(f"  chain: crisis share {s.mean():.4f} (0.15), mean crisis duration {dur:.2f} days (30)")


def test_ccc():
    m = G.model("sim_ccc")
    R = G.simulate_ccc(m, 300_000, np.random.default_rng(8))
    assert np.allclose(R.std(axis=0), m["vol"], rtol=0.05)
    # conditional correlation is constant: standardising by the TRUE conditional volatility recovers it
    w = m["vol"] ** 2 * (1 - m["a"] - m["b"])
    h = np.empty_like(R)
    h[0] = R[:2000].var(axis=0)
    for t in range(1, len(R)):
        h[t] = w + m["a"] * R[t - 1] ** 2 + m["b"] * h[t - 1]
    Z = (R / np.sqrt(h))[2000:]
    assert np.abs(np.corrcoef(Z, rowvar=False) - m["corr"]).max() < 0.012
    # ... in calm and in turbulent periods alike (split on cross-sectional mean variance ratio)
    lvl = (h[2000:] / m["vol"] ** 2).mean(axis=1)
    hi, lo = lvl > np.quantile(lvl, 0.85), lvl < np.quantile(lvl, 0.5)
    mh, ml = (G.mean_offdiag(np.corrcoef(Z[k], rowvar=False)) for k in (hi, lo))
    assert abs(mh - ml) < 0.01, (mh, ml)
    # volatility does cluster
    x = R[:, 0] ** 2
    assert np.corrcoef(x[:-1], x[1:])[0, 1] > 0.05


def test_oracle_labels():
    T = 60
    s = np.zeros(200, dtype=np.int8)
    s[100:110] = 1                                               # ten crisis days: days 100..109
    lab = G.oracle_labels(s, T)
    assert lab.shape == (200 - T + 1,)
    # window i ends on day i + 59; its last 20 days are i+40 .. i+59; all ten crisis days inside
    # <=> i + 40 <= 100 and i + 59 >= 109  <=>  50 <= i <= 60
    assert np.array_equal(np.flatnonzero(lab), np.arange(50, 61)), np.flatnonzero(lab)
    s[100] = 0                                                   # nine days are not enough
    assert G.oracle_labels(s, T).sum() == 0


def test_panels_on_disk():
    nse = load_panel("nse")
    for name in G.SIM_PANELS:
        p = load_panel(name)
        assert p.returns.shape == nse.returns.shape and p.returns.dtype == np.float64
        assert np.isfinite(p.returns).all() and p.volume.shape == nse.returns.shape
        assert p.dates.equals(nse.dates) and len(p.tickers) == N
        R, st = G.build_sim(name, *nse.returns.shape)            # deterministic rebuild
        assert np.array_equal(R, p.returns) and np.array_equal(st, G.load_state(name))
        audit = json.loads((PANELS / name / "audit.json").read_text())
        assert abs(audit["true_boost_pct"] - G.true_boost(name)) < 1e-9
    st = G.load_state("sim_boost40")
    assert st.shape == (len(nse.dates),) and set(np.unique(st)) == {0, 1} and 0.05 < st.mean() < 0.30
    for a, b in (("sim_boost00", "sim_boost80"), ("sim_boost20", "sim_boost40")):
        assert np.array_equal(G.load_state(a), G.load_state(b))            # common state path
        Ra, Rb = load_panel(a).returns, load_panel(b).returns
        assert np.array_equal(Ra[st == 0], Rb[st == 0]) and not np.allclose(Ra[st == 1], Rb[st == 1])
    assert G.load_state("sim_ccc").sum() == 0
    # the crisis state is visible to a volatility-based rule: dispersion about doubles
    R = load_panel("sim_boost00").returns
    ratio = R[st == 1].std(axis=1).mean() / R[st == 0].std(axis=1).mean()
    assert 1.8 < ratio < 2.2, ratio


def test_contaminated():
    nse = load_panel("nse")
    D, n = nse.returns.shape
    for name in G.CONTAMINATED:
        p = load_panel(name)
        audit = json.loads((PANELS / name / "audit.json").read_text())
        rec = audit["injected"]
        assert p.dates.equals(nse.dates) and p.tickers == nse.tickers and np.array_equal(p.volume, nse.volume)
        mask = np.zeros((D, n), bool)
        for e in rec:
            assert not mask[e["day"], e["asset"]], "cell recorded twice"
            mask[e["day"], e["asset"]] = True
            assert p.returns[e["day"], e["asset"]] == e["injected"]
            assert nse.returns[e["day"], e["asset"]] == e["clean"]
            assert str(nse.dates[e["day"]].date()) == e["date"] and nse.tickers[e["asset"]] == e["ticker"]
        assert np.array_equal(p.returns != nse.returns, mask), name          # exactly the recorded cells
        assert audit["n_injected"] == len(rec) == int(mask.sum())
        R2, rec2 = G.build_contaminated(name)                                # deterministic
        assert np.array_equal(R2, p.returns) and rec2 == rec
        v = np.array([e["injected"] for e in rec])
        if name == "nse_c_tick":
            assert len(rec) == 1 and v[0] == -0.76 and rec[0]["date"].startswith("2015")
        elif name == "nse_c_split":
            assert len(rec) == 1 and v[0] == -np.log(2.0) and rec[0]["date"].startswith("2016")
        else:
            assert len(rec) == round(0.01 * D * n)
            assert (np.abs(v) >= 0.30).all() and (np.abs(v) <= 0.60).all()
            assert 0.4 < (v > 0).mean() < 0.6 and abs(np.abs(v).mean() - 0.45) < 0.01


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name, flush=True)
            fn()
    print("OK")
