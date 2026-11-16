"""Tests of crisisbench.risk.  Run: .venv/Scripts/python.exe tests/test_risk.py

Every check is written so that a wrong formula fails it: statistics are compared with
numbers worked out by hand, with an independent scipy implementation (G-test), with
closed forms verified by numerical integration, or with a misspecified forecast that
the test must reject.
"""
import math
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crisisbench import risk as K  # noqa: E402

LEVELS = (0.95, 0.975, 0.99)


def t_var_es(nu: float, alpha: float) -> tuple[float, float]:
    """True VaR and ES (positive) of a standard Student-t.

    With q = t.ppf(alpha) and density f: int_{-inf}^{q} x f(x) dx = -f(q) (nu + q^2) / (nu - 1),
    so ES = f(q) (nu + q^2) / ((nu - 1) alpha).  Checked by quadrature in test_t_closed_form.
    """
    q = stats.t.ppf(alpha, nu)
    return -q, stats.t.pdf(q, nu) * (nu + q * q) / ((nu - 1) * alpha)


def test_t_closed_form():
    for nu in (4.0, 6.0):
        for a in (0.05, 0.01):
            v, e = t_var_es(nu, a)
            num = -integrate.quad(lambda x: x * stats.t.pdf(x, nu), -np.inf, -v)[0] / a
            assert abs(e - num) < 1e-7 * e, (nu, a, e, num)
            assert e > v > 0


def test_empirical_var_es():
    x = np.arange(1, 101, dtype=float) - 50.5          # -49.5 ... 49.5
    rng = np.random.default_rng(0)
    var, es = K.empirical_var_es(rng.permutation(x), LEVELS)
    # m = ceil(alpha * 100) = 5, 3, 1 (0.025 * 100 = 2.5 -> 3); 1 - 0.95 must not round up to 6
    assert np.allclose(var, [45.5, 47.5, 49.5]), var
    assert np.allclose(es, [47.5, 48.5, 49.5]), es
    var, es = K.empirical_var_es(np.linspace(-1, 1, 1000), (0.95,))
    assert np.isclose(var[0], -np.linspace(-1, 1, 1000)[49])      # 50th order statistic, not the 51st


def test_kupiec_by_hand():
    # n = 20, x = 3, alpha = 0.10, p_hat = 0.15:
    # LR = 2 [3 ln(0.15/0.10) + 17 ln(0.85/0.90)] = 2 [1.2163953 - 0.9716930] = 0.4894046
    h = np.zeros(20, int)
    h[[2, 9, 15]] = 1
    lr, p = K.kupiec_pof(h, 0.10)
    assert abs(lr - 0.4894046) < 1e-6, lr
    assert abs(p - stats.chi2.sf(0.4894046, 1)) < 1e-6
    g = stats.power_divergence([3, 17], f_exp=[2, 18], lambda_="log-likelihood")[0]   # independent code path
    assert abs(lr - g) < 1e-10
    # no violations: LR = -2 n ln(1 - alpha); all violations: LR = -2 n ln(alpha)
    lr0, p0 = K.kupiec_pof(np.zeros(250, int), 0.01)
    assert abs(lr0 - (-500 * math.log(0.99))) < 1e-10 and np.isfinite(p0)
    lr1, p1 = K.kupiec_pof(np.ones(50, int), 0.01)
    assert abs(lr1 - (-100 * math.log(0.01))) < 1e-9 and np.isfinite(p1) and p1 < 1e-10
    # p_hat == alpha exactly -> LR = 0
    h = np.zeros(100, int)
    h[:5] = 1
    assert K.kupiec_pof(h, 0.05) == (0.0, 1.0)


def test_christoffersen_by_hand():
    h = np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0])
    # 19 transitions counted by hand: n00 = 10, n01 = 3, n10 = 3, n11 = 3
    assert K._transitions(h) == (10, 3, 3, 3)
    # pi01 = 3/13, pi11 = 3/6, pi = 6/19
    l1 = 10 * math.log(10 / 13) + 3 * math.log(3 / 13) + 3 * math.log(0.5) + 3 * math.log(0.5)
    l0 = 13 * math.log(13 / 19) + 6 * math.log(6 / 19)
    lr, p = K.christoffersen_independence(h)
    assert abs(lr - 2 * (l1 - l0)) < 1e-10 and abs(lr - 1.3358104) < 1e-6, lr
    assert abs(p - stats.chi2.sf(lr, 1)) < 1e-12
    # the independence LR is the G-test of the 2x2 transition table (independent implementation)
    g = stats.chi2_contingency([[10, 3], [3, 3]], correction=False, lambda_="log-likelihood")[0]
    assert abs(lr - g) < 1e-9
    # conditional coverage at alpha = 0.10: null log-likelihood 13 ln 0.9 + 6 ln 0.1
    lc, pc = K.christoffersen_cc(h, 0.10)
    assert abs(lc - 2 * (l1 - (13 * math.log(0.9) + 6 * math.log(0.1)))) < 1e-10 and abs(lc - 8.0073207) < 1e-6
    assert abs(pc - stats.chi2.sf(lc, 2)) < 1e-12
    # exact decomposition on the transitions: LR_cc = LR_uc(hits[1:]) + LR_ind
    assert abs(lc - (K.kupiec_pof(h[1:], 0.10)[0] + lr)) < 1e-10


def test_christoffersen_empty_cells():
    # n11 = 0 (the usual case): reduced likelihood (1 - pi01)^n00 pi01^n01
    h = np.zeros(30, int)
    h[[4, 11, 20]] = 1
    n00, n01, n10, n11 = K._transitions(h)
    assert (n00, n01, n10, n11) == (23, 3, 3, 0)
    l1 = 23 * math.log(23 / 26) + 3 * math.log(3 / 26)
    l0 = 26 * math.log(26 / 29) + 3 * math.log(3 / 29)
    lr, p = K.christoffersen_independence(h)
    assert abs(lr - 2 * (l1 - l0)) < 1e-10 and 0 < p < 1
    # no hits, only hits, a single hit on the last day (empty "after a hit" row): all finite
    for hh in (np.zeros(40, int), np.ones(40, int), np.r_[np.zeros(39, int), 1]):
        for out in (K.christoffersen_independence(hh), K.christoffersen_cc(hh, 0.05)):
            assert all(np.isfinite(out)), (hh.sum(), out)
    assert K.christoffersen_independence(np.zeros(40, int)) == (0.0, 1.0)
    # perfectly clustered hits must be rejected, alternating-free independent-looking ones not
    assert K.christoffersen_independence(np.r_[np.zeros(80, int), np.ones(20, int)])[1] < 1e-6


def test_traffic_light():
    zones = [K.basel_traffic_light(x, 250) for x in range(0, 15)]
    assert zones == ["green"] * 5 + ["yellow"] * 5 + ["red"] * 5, zones
    # the boundaries come from the binomial cdf, for any sample size
    for n in (100, 500, 1240):
        c = stats.binom.cdf(np.arange(0, 60), n, 0.01)
        first_yellow, first_red = int(np.argmax(c >= 0.95)), int(np.argmax(c >= 0.9999))
        assert K.basel_traffic_light(first_yellow - 1, n) == "green"
        assert K.basel_traffic_light(first_yellow, n) == "yellow"
        assert K.basel_traffic_light(first_red - 1, n) == "yellow"
        assert K.basel_traffic_light(first_red, n) == "red"


def test_losses_by_hand():
    # one violation day and one quiet day, alpha = 0.05, VaR = 0.02, ES = 0.03
    r = np.array([-0.05, 0.01])
    v, e = np.full(2, 0.02), np.full(2, 0.03)
    fz = K.fz0_loss(r, v, e, 0.05)
    base = 0.02 / 0.03 + math.log(0.03) - 1
    assert np.allclose(fz, [0.03 / (0.05 * 0.03) + base, base])
    # the same numbers through the paper's formula in its own sign convention (v, e < 0)
    vq, eq = -0.02, -0.03
    paper = [-float(y <= vq) * (vq - y) / (0.05 * eq) + vq / eq + math.log(-eq) - 1 for y in r]
    assert np.allclose(fz, paper)
    ql = K.quantile_loss(r, v, 0.05)
    assert np.allclose(ql, [(0.05 - 1) * (-0.05 + 0.02), 0.05 * (0.01 + 0.02)]) and (ql >= 0).all()
    # Z2 by hand: sum r I / (T alpha ES) + 1 = (-0.05 / 0.03) / (2 * 0.05) + 1
    z2, p = K.acerbi_szekely_z2(r, v, e, 0.05)
    assert abs(z2 - ((-0.05 / 0.03) / 0.1 + 1)) < 1e-12 and math.isnan(p)


def test_fz0_minimised_by_truth():
    nu, a, n = 5.0, 0.025, 400_000
    rng = np.random.default_rng(1)
    r = 0.01 * rng.standard_t(nu, n)
    v0, e0 = (0.01 * x for x in t_var_es(nu, a))
    true = K.fz0_loss(r, np.full(n, v0), np.full(n, e0), a)
    for fv, fe in ((1.25, 1.25), (0.8, 0.8), (1.0, 1.25), (1.0, 0.85), (1.2, 1.0), (0.8, 1.0)):
        d = K.fz0_loss(r, np.full(n, fv * v0), np.full(n, fe * e0), a) - true
        assert d.mean() > 3 * d.std(ddof=1) / math.sqrt(n), (fv, fe, d.mean(), d.std() / math.sqrt(n))
    for fv in (1.25, 0.8):
        d = K.quantile_loss(r, np.full(n, fv * v0), a) - K.quantile_loss(r, np.full(n, v0), a)
        assert d.mean() > 3 * d.std(ddof=1) / math.sqrt(n)


def _reject_rates(scale_forecast: float, cluster: bool, reps: int = 300, T: int = 1000, a: float = 0.05):
    """Share of replications in which each coverage test rejects at 5%."""
    nu = 5.0
    v0, e0 = t_var_es(nu, a)
    rng = np.random.default_rng(42)
    rej = np.zeros(3)
    for _ in range(reps):
        sig = np.exp(0.5 * np.sin(np.arange(T) / 40.0))            # forecast knows the true scale
        r = sig * rng.standard_t(nu, T)
        h = r < -sig * v0 * scale_forecast
        if cluster:                    # first-order Markov hits with the right unconditional rate:
            u = rng.random(T)          # pi01 = 0.03, pi11 = 0.40 -> stationary 0.03 / 0.63 = 0.0476
            h = np.zeros(T, bool)
            for t in range(1, T):
                h[t] = u[t] < (0.40 if h[t - 1] else 0.03)
        rej += [K.kupiec_pof(h, a)[1] < 0.05, K.christoffersen_independence(h)[1] < 0.05,
                K.christoffersen_cc(h, a)[1] < 0.05]
    return rej / reps


def test_coverage_tests_size_and_power():
    size = _reject_rates(1.0, False)
    assert ((size > 0.015) & (size < 0.10)).all(), size          # nominal 5%, 300 replications
    under = _reject_rates(0.75, False)                            # VaR understated by 25%
    assert under[0] > 0.9 and under[2] > 0.8, under
    clus = _reject_rates(1.0, True)                               # clustered violations
    assert clus[1] > 0.9 and clus[2] > 0.9 and clus[0] < 0.5, clus
    print(f"  size {size.round(3)}  power(VaR x0.75) {under.round(3)}  power(clustered) {clus.round(3)}")
    return size


def test_z2_size_and_power():
    nu, a, T, reps, k = 5.0, 0.05, 500, 300, 4000
    z = stats.t.ppf((np.arange(k) + 0.5) / k, nu)                 # discretised true predictive law
    zv, ze = K.empirical_var_es(z, (1 - a,))
    v_true, e_true = t_var_es(nu, a)
    assert abs(zv[0] - v_true) < 0.01 * v_true and abs(ze[0] - e_true) < 0.02 * e_true
    rng = np.random.default_rng(7)
    sig = np.exp(0.5 * np.sin(np.arange(T) / 40.0))
    nb = sig[:, None] * z[None, :]
    rej, rej_bad, zs = 0, 0, []
    for _ in range(reps):
        r = sig * rng.standard_t(nu, T)
        z2, p = K.acerbi_szekely_z2(r, sig * zv[0], sig * ze[0], a, nb, n_sim=300, rng=rng)
        zs.append(z2)
        rej += p < 0.05
        # a model whose predictive law is 30% too narrow, tested against its own law
        rej_bad += K.acerbi_szekely_z2(r, 0.7 * sig * zv[0], 0.7 * sig * ze[0], a, 0.7 * nb, n_sim=300, rng=rng)[1] < 0.05
    zs = np.array(zs)
    assert abs(zs.mean()) < 4 * zs.std(ddof=1) / math.sqrt(reps), zs.mean()     # E[Z2] = 0 under H0
    assert 0.015 < rej / reps < 0.10, rej / reps
    assert rej_bad / reps > 0.9, rej_bad / reps
    print(f"  Z2 mean {zs.mean():+.4f} (sd {zs.std():.3f})  size {rej / reps:.3f}  power {rej_bad / reps:.3f}")
    return rej / reps


def test_knn_matches_brute_force():
    rng = np.random.default_rng(3)
    s = np.r_[rng.gamma(2.0, 1.0, 500), [1.0, 1.0, 1.0]]         # with ties
    y = rng.standard_normal(s.size)
    x = np.r_[rng.gamma(2.0, 1.0, 60), -5.0, 100.0, s.min(), s.max(), 1.0]
    for frac in (0.05, 0.013, 1.0, 1e-9):
        k = min(s.size, max(1, math.ceil(frac * s.size - 1e-9)))
        var, es, nb = K.knn_var_es(s, y, x, LEVELS, frac, return_neighbours=True)
        assert nb.shape == (x.size, k) and var.shape == es.shape == (x.size, 3)
        for d, xd in enumerate(x):
            dist = np.abs(s - xd)
            kth = np.sort(dist)[k - 1]
            sure = dist < kth                                      # must be in; ties at kth are free
            got = np.sort(nb[d])
            assert np.isin(y[sure], got).all()
            assert np.isin(got, y[dist <= kth]).all()
            m = [K._tail_count(1 - c, k) for c in LEVELS]
            assert np.allclose(var[d], [-got[j - 1] for j in m])
            assert np.allclose(es[d], [-got[:j].mean() for j in m])


def test_knn_recovers_conditional_gaussian():
    rng = np.random.default_rng(5)
    n = 400_000
    s = rng.uniform(0.005, 0.02, n)
    y = (s * rng.standard_normal(n)).astype(np.float32)           # the pool is stored as float32
    q = np.array([0.0075, 0.010, 0.015])
    var, es = K.knn_var_es(s.astype(np.float32), y, q, LEVELS, 0.02)
    for j, c in enumerate(LEVELS):
        a = 1 - c
        zq = -stats.norm.ppf(a)
        ze = stats.norm.pdf(zq) / a                               # Gaussian ES
        assert np.allclose(var[:, j], q * zq, rtol=0.05), (c, var[:, j], q * zq)
        assert np.allclose(es[:, j], q * ze, rtol=0.05), (c, es[:, j], q * ze)
    assert (es >= var).all() and (np.diff(var, axis=1) > 0).all() and (np.diff(var, axis=0) > 0).all()
    # an unconditional pool quantile would be the same for every state: the 2x spread must show
    assert var[2, 0] / var[0, 0] > 1.8


def test_historical_simulation():
    rng = np.random.default_rng(11)
    r = 0.01 * rng.standard_t(4, 900)
    var, es = K.historical_simulation(r, 250, LEVELS)
    assert np.isnan(var[:250]).all() and np.isfinite(var[250:]).all() and np.isfinite(es[250:]).all()
    for t in (250, 517, 899):                                     # direct recomputation from the past only
        past = np.sort(r[t - 250:t])
        assert np.allclose(var[t], [-past[12], -past[6], -past[2]])          # m = 13, 7, 3
        assert np.allclose(es[t], [-past[:13].mean(), -past[:7].mean(), -past[:3].mean()])
    # no look-ahead: changing day s and later leaves forecasts up to and including day s unchanged
    s = 600
    r2 = r.copy()
    r2[s:] = -0.5
    v2, e2 = K.historical_simulation(r2, 250, LEVELS)
    assert np.array_equal(var[:s + 1], v2[:s + 1], equal_nan=True) and np.array_equal(es[:s + 1], e2[:s + 1], equal_nan=True)
    assert not np.allclose(var[s + 1], v2[s + 1])                 # ... and the day after does react


def _garch_path(n, rng, om=0.02, a=0.08, b=0.90, nu=6.0):
    z = rng.standard_t(nu, n) * math.sqrt((nu - 2) / nu)
    r, h = np.empty(n), om / (1 - a - b)
    for t in range(n):
        r[t] = math.sqrt(h) * z[t]
        h = om + a * r[t] ** 2 + b * h
    return r / 100.0


def test_garch_fhs():
    from arch import arch_model
    rng = np.random.default_rng(21)
    r = _garch_path(2600, rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        var, es = K.garch_fhs_var_es(r, LEVELS, refit_every=400, min_obs=1000)
        assert np.isnan(var[:1000]).all() and np.isfinite(var[1000:]).all() and (es[1000:] >= var[1000:]).all()
        # independent recomputation with arch's own forecast: first day after a fit, and a day
        # late in a refit block (parameters of the block's fit, data up to the day before)
        for t0, t in ((1000, 1000), (1000, 1399), (1400, 1650)):
            fit = arch_model(100 * r[:t0], mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off")
            z = np.sort(np.asarray(fit.resid) / np.asarray(fit.conditional_volatility))
            fixed = arch_model(100 * r[:t], mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False).fix(fit.params)
            s = math.sqrt(float(np.asarray(fixed.forecast(horizon=1).variance)[-1, 0]))
            mu = float(fit.params["mu"])
            m = [K._tail_count(1 - c, t0) for c in LEVELS]
            assert np.allclose(var[t], [-(mu + s * z[j - 1]) / 100 for j in m], rtol=1e-6), (t, var[t])
            assert np.allclose(es[t], [-(mu + s * z[:j].mean()) / 100 for j in m], rtol=1e-6), (t, es[t])
        # calibration on data from the model class
        rate = (r[1000:, None] < -var[1000:]).mean(0)
        assert np.all(np.abs(rate - [0.05, 0.025, 0.01]) < [0.02, 0.012, 0.008]), rate
        # the forecast tracks volatility: violations are not clustered
        assert K.christoffersen_independence(r[1000:] < -var[1000:, 0])[1] > 0.01
        # no look-ahead (also across the refits at 1400, 1800, 2200)
        s0 = 1500
        r2 = r.copy()
        r2[s0:] = r[s0:] * 3.0 + 0.01
        v2, e2 = K.garch_fhs_var_es(r2, LEVELS, refit_every=400, min_obs=1000)
    assert np.array_equal(var[:s0 + 1], v2[:s0 + 1], equal_nan=True) and np.array_equal(es[:s0 + 1], e2[:s0 + 1], equal_nan=True)
    assert not np.allclose(var[s0 + 1:], v2[s0 + 1:])
    print(f"  GARCH-FHS violation rates {rate.round(4)} (nominal 0.05, 0.025, 0.01)")


def test_summarise():
    rng = np.random.default_rng(9)
    nu, a, T = 5.0, 0.01, 1500
    r = 0.01 * rng.standard_t(nu, T)
    v0, e0 = (0.01 * x for x in t_var_es(nu, a))
    var, es = np.full(T, v0), np.full(T, e0)
    var[:100], es[:100] = np.nan, np.nan                           # warm-up days are dropped
    out = K.summarise(r, var, es, a)
    h = r[100:] < -v0
    assert out["n"] == 1400 and out["violations"] == int(h.sum()) and abs(out["expected"] - 14.0) < 1e-9
    assert abs(out["rate"] - h.mean()) < 1e-15
    assert out["kupiec_p"] == K.kupiec_pof(h, a)[1] and out["cc_p"] == K.christoffersen_cc(h, a)[1]
    assert out["christoffersen_ind_p"] == K.christoffersen_independence(h)[1]
    assert out["traffic_light"] == K.basel_traffic_light(int(h.sum()), 1400, a)
    assert abs(out["fz0"] - K.fz0_loss(r[100:], var[100:], es[100:], a).mean()) < 1e-12
    assert abs(out["quantile_loss"] - K.quantile_loss(r[100:], var[100:], a).mean()) < 1e-15
    assert abs(out["z2"] - K.acerbi_szekely_z2(r[100:], var[100:], es[100:], a)[0]) < 1e-12 and math.isnan(out["z2_p"])
    nb = np.tile(0.01 * stats.t.ppf((np.arange(2000) + 0.5) / 2000, nu), (T, 1))
    assert 0 < K.summarise(r, var, es, a, neighbours=nb, n_sim=200, rng=np.random.default_rng(0))["z2_p"] <= 1
    for bad in (lambda: K.fz0_loss(r, var, es, a), lambda: K.kupiec_pof([0, 2, 1], 0.05),
                lambda: K.kupiec_pof([0, 1], 0.95), lambda: K.fz0_loss(r[:2], np.ones(2), -np.ones(2), a)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("invalid input was accepted")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name, flush=True)
            fn()
    print("OK")
