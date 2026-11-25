"""Strong classical baselines (added after the protocol freeze; reported as post hoc rows).

These are the models a bank's market-risk team would actually use for multi-asset stress scenarios, chosen to be
the hardest classical competitors for a claim about crisis dependence.

Shared first stage: per asset, constant mean + GJR-GARCH(1,1,1) with Student-t innovations (Glosten, Jagannathan &
Runkle 1993), i.e. h_t = omega + (alpha + gamma 1[e_{t-1}<0]) e_{t-1}^2 + beta h_{t-1}: falls raise volatility more
than rises (leverage).  Persistence alpha + gamma/2 + beta is clipped to 0.999.

GjrEvtTCopula   McNeil & Frey (2000) margins + Student-t copula (Demarta & McNeil 2005):
  each asset's standardised residuals get a semi-parametric distribution (empirical body, generalised Pareto tails
  beyond the 10% / 90% quantiles); dependence is a t copula whose correlation comes from Kendall's tau
  (R = sin(pi tau / 2), projected to the nearest correlation matrix) and whose degrees of freedom maximise the
  copula likelihood on a grid.  Tail dependence is symmetric and constant over time.

AdccGjr         Asymmetric DCC (Cappiello, Engle & Sheppard 2006):
  Q_t = (1-a-b) Qbar - g Nbar + a z z' + g n n' + b Q_{t-1},  n = min(z, 0),
  so correlations rise after JOINT NEGATIVE shocks, the classical model of crisis correlation.  (a, b, g) maximise
  the Gaussian QML second stage subject to a + b + delta g < 0.999 (delta = largest eigenvalue of
  Qbar^-1/2 Nbar Qbar^-1/2).  Innovations by filtered historical simulation: the fitted model's whitened residuals
  R_t^-1/2 z_t are resampled as whole cross-sections and re-correlated by the simulated R_t.

RsTCopula       Regime-switching t copula (in the spirit of Rodriguez 2007; Okimoto 2008):
  a two-state Gaussian HMM on the equal-weight portfolio return gives a calm and a turbulent state with a Markov
  chain; each state has its own t copula (Kendall correlation + df) AND its own semi-parametric residual margins,
  both estimated on the GJR residuals of that state's days.  The common state therefore moves the size of every
  asset's shocks together (common volatility) as well as their correlation and tail dependence.

All three are sampled unconditionally: one long path, 500-day burn-in, cut into non-overlapping windows.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .. import protocol as P
from .base import FitData, Generator, chop

BURN_IN = 500
SCALE = 100.0
MAX_PERS = 0.999
TAIL = 0.10
DF_GRID = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0, 20.0, 30.0, 50.0)
_STARTS = (None, (0.03, 0.08, 0.88, 8.0), (0.05, 0.10, 0.80, 6.0), (0.02, 0.05, 0.92, 10.0))


# ------------------------------------------------------------------ margins
def gjr_fit(returns: np.ndarray) -> tuple[dict, np.ndarray, dict]:
    from arch import arch_model
    y = np.asarray(returns, dtype=np.float64) * SCALE
    D, N = y.shape
    par = np.empty((N, 6))                                  # mu omega alpha gamma beta nu
    for j in range(N):
        am = arch_model(y[:, j], mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t", rescale=False)
        v, ok = y[:, j].var(), False
        for st in _STARTS:
            sv = None if st is None else np.array([y[:, j].mean(), v * (1 - st[0] - st[1] / 2 - st[2]), st[0], st[1], st[2], st[3]])
            r = am.fit(disp="off", show_warning=False, starting_values=sv, options=dict(maxiter=1000))
            p = np.asarray(r.params, dtype=np.float64)
            if r.convergence_flag == 0 and np.isfinite(p).all() and p[1] > 0 and p[5] > 2:
                par[j], ok = p, True
                break
        if not ok:
            raise RuntimeError(f"GJR-GARCH-t did not converge for asset {j}")
    mu, om, al, ga, be, nu = par.T.copy()
    ga = np.maximum(ga, -al)                                # keep alpha + gamma >= 0
    pers = al + ga / 2 + be
    f = np.where(pers >= MAX_PERS, MAX_PERS / np.maximum(pers, 1e-12), 1.0)
    al, ga, be = al * f, ga * f, be * f
    p = dict(mu=mu, omega=om, alpha=al, gamma=ga, beta=be, nu=nu)
    h, Z = gjr_filter(y, p)
    if not (np.isfinite(Z).all() and (h > 0).all()):
        raise RuntimeError("non-finite GJR residuals")
    return p, Z, dict(n_clipped=int((pers >= MAX_PERS).sum()), gamma_median=float(np.median(ga)),
                      persistence_median=float(np.median(al + ga / 2 + be)), nu_median=float(np.median(nu)))


def gjr_filter(y, p):
    e = y - p["mu"]
    h = np.empty_like(e)
    h[0] = e.var(0)
    for t in range(1, len(e)):
        h[t] = p["omega"] + (p["alpha"] + p["gamma"] * (e[t - 1] < 0)) * e[t - 1] ** 2 + p["beta"] * h[t - 1]
    return h, e / np.sqrt(h)


def gjr_simulate(p: dict, Z: np.ndarray) -> np.ndarray:
    h = p["omega"] / (1 - p["alpha"] - p["gamma"] / 2 - p["beta"])
    e = np.empty_like(Z)
    for t in range(len(Z)):
        e[t] = np.sqrt(h) * Z[t]
        h = p["omega"] + (p["alpha"] + p["gamma"] * (e[t] < 0)) * e[t] ** 2 + p["beta"] * h
    out = (p["mu"] + e) / SCALE
    if not np.isfinite(out).all():
        raise FloatingPointError("simulated GJR path is not finite")
    return out


class SemiParam:
    """Per-asset residual distribution: empirical body, GPD tails beyond the TAIL / 1-TAIL quantiles."""

    def __init__(self, Z: np.ndarray):
        self.sorted = np.sort(Z, 0)
        self.lo, self.hi = np.quantile(Z, TAIL, 0), np.quantile(Z, 1 - TAIL, 0)
        self.gl, self.gu = [], []
        for j in range(Z.shape[1]):
            xl = self.lo[j] - Z[Z[:, j] < self.lo[j], j]
            xu = Z[Z[:, j] > self.hi[j], j] - self.hi[j]
            self.gl.append(stats.genpareto.fit(xl, floc=0.0))
            self.gu.append(stats.genpareto.fit(xu, floc=0.0))

    def ppf(self, U: np.ndarray) -> np.ndarray:
        U = np.clip(U, 1e-6, 1 - 1e-6)
        n = len(self.sorted)
        out = np.empty_like(U)
        for j in range(U.shape[1]):
            u = U[:, j]
            body = np.interp(u, (np.arange(n) + 0.5) / n, self.sorted[:, j])
            cl, _, sl = self.gl[j]
            cu, _, su = self.gu[j]
            lo = self.lo[j] - stats.genpareto.ppf(np.clip(1 - u / TAIL, 0, 1 - 1e-9), cl, scale=sl)
            hi = self.hi[j] + stats.genpareto.ppf(np.clip(1 - (1 - u) / TAIL, 0, 1 - 1e-9), cu, scale=su)
            out[:, j] = np.where(u < TAIL, lo, np.where(u > 1 - TAIL, hi, body))
        return out


# ------------------------------------------------------------------ t copula
def nearest_corr(R: np.ndarray) -> np.ndarray:
    R = (R + R.T) / 2
    w, V = np.linalg.eigh(R)
    R = (V * np.maximum(w, 1e-4)) @ V.T
    d = np.sqrt(np.diag(R))
    return R / np.outer(d, d)


def kendall_corr(Z: np.ndarray) -> np.ndarray:
    N = Z.shape[1]
    R = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            R[i, j] = R[j, i] = np.sin(np.pi * stats.kendalltau(Z[:, i], Z[:, j])[0] / 2)
    return nearest_corr(R)


def tcop_loglik(U: np.ndarray, R: np.ndarray, nu: float) -> float:
    X = stats.t.ppf(np.clip(U, 1e-6, 1 - 1e-6), nu)
    L = np.linalg.cholesky(R)
    q = (np.linalg.solve(L, X.T) ** 2).sum(0)
    N = R.shape[0]
    from scipy.special import gammaln
    joint = (gammaln((nu + N) / 2) - gammaln(nu / 2) - N / 2 * np.log(nu * np.pi) - np.log(np.diag(L)).sum()
             - (nu + N) / 2 * np.log1p(q / nu))
    return float(joint.sum() - stats.t.logpdf(X, nu).sum())


def fit_tcopula(Z: np.ndarray) -> tuple[np.ndarray, float, dict]:
    U = (stats.rankdata(Z, axis=0) - 0.5) / len(Z)
    R = kendall_corr(Z)
    ll = {nu: tcop_loglik(U, R, nu) for nu in DF_GRID}
    nu = max(ll, key=ll.get)
    return R, nu, dict(copula_df=nu, copula_ll_per_day=ll[nu] / len(Z))


def tcop_draw(R: np.ndarray, nu: float, n: int, rng) -> np.ndarray:
    g = rng.standard_normal((n, R.shape[0])) @ np.linalg.cholesky(R).T
    w = np.sqrt(nu / rng.chisquare(nu, n))[:, None]
    return stats.t.cdf(g * w, nu)


# ------------------------------------------------------------------ generators
class GjrEvtTCopula(Generator):
    family = "dynamic-classical"
    impl_source = "own on arch + scipy"

    def fit(self, data: FitData, seed: int) -> dict:
        self.p, Z, info = gjr_fit(data.returns)
        self.marg = SemiParam(Z)
        self.R, self.nu, cinfo = fit_tcopula(Z)
        return dict(info, **cinfo)

    def sample(self, n, rng):
        L = BURN_IN + n * P.T
        Z = self.marg.ppf(tcop_draw(self.R, self.nu, L, rng))
        Z = (Z - Z.mean(0)) / Z.std(0)                       # unit-variance innovations for the GJR recursion
        return chop(gjr_simulate(self.p, Z)[BURN_IN:], n)


def adcc_nll(a, b, g, Z, outer, nouter, Qbar, Nbar) -> float:
    D, N = Z.shape
    c = (1 - a - b) * Qbar - g * Nbar
    Q = Qbar.copy()
    tot = 0.0
    for t in range(D):
        if t:
            Q = c + a * outer[t - 1] + g * nouter[t - 1] + b * Q
        d = np.sqrt(np.diag(Q))
        R = Q / np.outer(d, d)
        try:
            Lc = np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            return 1e9
        x = np.linalg.solve(Lc, Z[t])
        tot += 2 * np.log(np.diag(Lc)).sum() + x @ x - Z[t] @ Z[t]
    return 0.5 * tot / D


class AdccGjr(Generator):
    family = "dynamic-classical"
    impl_source = "own"

    def fit(self, data: FitData, seed: int) -> dict:
        from scipy.optimize import minimize
        self.p, Z, info = gjr_fit(data.returns)
        Zs = Z[-1500:] if len(Z) > 1500 else Z               # second stage on the last 6 years for speed
        Nm = np.minimum(Zs, 0.0)
        self.Qbar, self.Nbar = Zs.T @ Zs / len(Zs), Nm.T @ Nm / len(Zs)
        w, V = np.linalg.eigh(self.Qbar)
        Qmh = (V / np.sqrt(w)) @ V.T
        self.delta = float(np.linalg.eigvalsh(Qmh @ self.Nbar @ Qmh).max())
        outer, nouter = Zs[:, :, None] * Zs[:, None, :], Nm[:, :, None] * Nm[:, None, :]
        sig = lambda x: 1 / (1 + np.exp(-x))

        def par(th):
            s = MAX_PERS * sig(th[0])                       # a + b + delta g = s
            fa, fg = sig(th[1]), sig(th[2])
            a = s * fa * (1 - fg)
            g = s * fg * 0.5 / self.delta
            b = s - a - self.delta * g
            return a, max(b, 0.0), g

        best = None
        for x0 in ([3.0, -3.0, -2.0], [2.0, -2.5, -1.0], [4.0, -3.5, -3.0]):
            r = minimize(lambda th: adcc_nll(*par(th), Zs, outer, nouter, self.Qbar, self.Nbar), x0, method="Nelder-Mead",
                         options=dict(maxfev=250, xatol=1e-3, fatol=1e-7))
            if best is None or r.fun < best.fun:
                best = r
        self.a, self.b, self.g = par(best.x)
        # whitened residuals of the fitted model on the full sample, for filtered historical simulation
        Nf = np.minimum(Z, 0.0)
        c = (1 - self.a - self.b) * self.Qbar - self.g * self.Nbar
        Q, E = self.Qbar.copy(), np.empty_like(Z)
        for t in range(len(Z)):
            if t:
                Q = c + self.a * np.outer(Z[t - 1], Z[t - 1]) + self.g * np.outer(Nf[t - 1], Nf[t - 1]) + self.b * Q
            d = np.sqrt(np.diag(Q))
            E[t] = np.linalg.solve(np.linalg.cholesky(Q / np.outer(d, d)), Z[t])
        self.E = E
        return dict(info, a=self.a, b=self.b, g=self.g, delta=self.delta, nll_per_day=float(best.fun))

    def sample(self, n, rng):
        L, N = BURN_IN + n * P.T, self.Qbar.shape[0]
        idx = rng.integers(0, len(self.E), L)
        c = (1 - self.a - self.b) * self.Qbar - self.g * self.Nbar
        Q, Z = self.Qbar.copy(), np.empty((L, N))
        for t in range(L):
            d = np.sqrt(np.diag(Q))
            z = np.linalg.cholesky(Q / np.outer(d, d)) @ self.E[idx[t]]
            Z[t] = z
            nz = np.minimum(z, 0.0)
            Q = c + self.a * np.outer(z, z) + self.g * np.outer(nz, nz) + self.b * Q
        return chop(gjr_simulate(self.p, Z)[BURN_IN:], n)


class RsTCopula(Generator):
    family = "regime"
    impl_source = "own on arch + hmmlearn + scipy"

    def fit(self, data: FitData, seed: int) -> dict:
        from hmmlearn.hmm import GaussianHMM
        self.p, Z, info = gjr_fit(data.returns)
        ew = np.asarray(data.returns).mean(1, keepdims=True) * SCALE
        hmm = GaussianHMM(2, covariance_type="full", n_iter=200, random_state=seed).fit(ew)
        st = hmm.predict(ew)
        turb = int(np.argmax(hmm.covars_.reshape(2, -1)[:, 0]))        # the high-variance state
        self.state = (st == turb).astype(int)
        self.T = hmm.transmat_[[1 - turb, turb]][:, [1 - turb, turb]]  # rows/cols: calm, turbulent
        self.cop, self.marg = [], []
        for k in (0, 1):
            Zk = Z[self.state == k]
            if len(Zk) < 60:
                raise ValueError(f"regime {k} has only {len(Zk)} days")
            R, nu, _ = fit_tcopula(Zk)
            self.cop.append((R, nu))
            self.marg.append(SemiParam(Zk))
        sd = [float(Z[self.state == k].std()) for k in (0, 1)]
        off = ~np.eye(Z.shape[1], dtype=bool)
        return dict(info, turbulent_share=float(self.state.mean()), p_stay=[float(self.T[0, 0]), float(self.T[1, 1])],
                    mean_corr=[float(self.cop[k][0][off].mean()) for k in (0, 1)], copula_df=[self.cop[k][1] for k in (0, 1)],
                    resid_sd=sd)

    def sample(self, n, rng):
        L = BURN_IN + n * P.T
        s = np.empty(L, dtype=int)
        s[0] = int(rng.random() < self.state.mean())
        for t in range(1, L):
            s[t] = int(rng.random() < self.T[s[t - 1], 1])
        Z = np.empty((L, self.cop[0][0].shape[0]))
        for k in (0, 1):
            m = s == k
            if m.any():
                Z[m] = self.marg[k].ppf(tcop_draw(*self.cop[k], int(m.sum()), rng))
        # one common rescaling (not per state) keeps the state-to-state difference in shock size
        Z = (Z - Z.mean(0)) / Z.std(0)
        return chop(gjr_simulate(self.p, Z)[BURN_IN:], n)
