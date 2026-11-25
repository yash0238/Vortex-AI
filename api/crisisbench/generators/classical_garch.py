"""Dynamic-volatility baselines: GARCH filtered historical simulation and DCC-GARCH.

First stage (shared, `fit_univariate`): per asset, constant mean + GARCH(1,1) with
Student-t innovations, estimated by ML with the `arch` package on returns x100.  A fit
that does not converge is retried from other starting values; if none converges, fit
raises.  Persistence alpha+beta >= 0.999 is clipped to 0.999 (alpha and beta scaled by
the same factor, omega kept); the number of clipped assets is reported.

GarchFHS  Filtered historical simulation (Barone-Adesi, Giannopoulos & Vosper 1999):
  whole cross-sections of standardised residuals are resampled with replacement
  (i.i.d. days, so cross-sectional dependence is kept and temporal dependence comes
  only from each asset's variance recursion).

DCCGarch  Engle (2002) DCC(1,1) with correlation targeting,
      Q_t = (1-a-b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1},   R_t = diag(Q_t)^-1/2 Q_t diag(Q_t)^-1/2,
  Qbar = (1/D) sum z_t z_t', Q_1 = Qbar; (a, b) maximise the second-stage Gaussian
  quasi-likelihood  -1/2 sum_t (log|R_t| + z_t' R_t^-1 z_t - z_t' z_t).

Deviations from the references (also reported by the lead)
  * Residuals are recomputed with our own variance recursion started at the sample
    variance (arch starts from an exponentially weighted backcast); needed so that
    clipped parameters and residuals are consistent.  Differences vanish after the
    first few dozen days.
  * FHS residuals are used as filtered (not re-centred or re-scaled), as in the
    reference.  Their second moment m2 is not exactly 1, so the persistence that
    matters in simulation is alpha*m2 + beta; fit raises if that is >= 1 for any asset.
  * DCC (a, b) are found by Nelder-Mead on an unconstrained reparametrisation
    (a+b = 0.999*sigmoid(u), a = (a+b)*sigmoid(v)), so a, b >= 0 and a+b < 0.999 hold
    by construction, from several starts (best likelihood kept).
  * DCC simulation innovations are multivariate Student-t with ONE common df (median
    of the univariate dfs), scaled to unit variance and correlated by R_t.  This is
    not the estimated model (Gaussian QML second stage, asset-specific t marginals):
    all assets share one mixing variable per day, which adds joint tail dependence.

Sampling (both): one long path, 500-day burn-in, cut into non-overlapping windows.
"""
from __future__ import annotations

import numpy as np

from .. import protocol as P
from .base import FitData, Generator, chop

BURN_IN = 500
SCALE = 100.0              # returns are fitted in percent for the optimiser
MAX_PERSISTENCE = 0.999
_STARTS = (None, (0.05, 0.90, 8.0), (0.10, 0.80, 6.0), (0.03, 0.95, 10.0), (0.15, 0.70, 5.0), (0.20, 0.50, 4.0))


def garch_filter(y: np.ndarray, mu, omega, alpha, beta) -> tuple[np.ndarray, np.ndarray]:
    """GARCH(1,1) variance recursion for all assets at once: (variance h, residual z)."""
    e = y - mu
    h = np.empty_like(e)
    h[0] = e.var(axis=0)
    for t in range(1, len(e)):
        h[t] = omega + alpha * e[t - 1] ** 2 + beta * h[t - 1]
    return h, e / np.sqrt(h)


def fit_univariate(returns: np.ndarray) -> tuple[dict, np.ndarray, dict]:
    """Per-asset GARCH(1,1)-t. Returns (params in percent units, residuals (D, N), info)."""
    from arch import arch_model
    y = np.asarray(returns, dtype=np.float64) * SCALE
    N = y.shape[1]
    par = np.empty((N, 5))                  # mu, omega, alpha, beta, nu
    retries = 0
    for j in range(N):
        am = arch_model(y[:, j], mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False)
        v, ok = y[:, j].var(), False
        for k, st in enumerate(_STARTS):
            sv = None if st is None else np.array([y[:, j].mean(), v * (1 - st[0] - st[1]), st[0], st[1], st[2]])
            res = am.fit(disp="off", show_warning=False, starting_values=sv, options=dict(maxiter=1000))
            p = np.asarray(res.params, dtype=np.float64)
            if res.convergence_flag == 0 and np.isfinite(p).all() and p[1] > 0 and p[4] > 2:
                par[j], ok = p, True
                retries += k
                break
        if not ok:
            raise RuntimeError(f"GARCH(1,1)-t did not converge for asset {j} from {len(_STARTS)} starting values")
    mu, omega, alpha, beta, nu = par.T.copy()
    pers = alpha + beta
    clip = pers >= MAX_PERSISTENCE
    f = np.where(clip, MAX_PERSISTENCE / np.where(clip, pers, 1.0), 1.0)
    alpha, beta = alpha * f, beta * f
    h, Z = garch_filter(y, mu, omega, alpha, beta)
    if not (np.isfinite(Z).all() and (h > 0).all()):
        raise RuntimeError("non-finite GARCH residuals")
    uncond = omega / (1 - alpha - beta)
    ratio = np.sqrt(uncond / y.var(axis=0))        # implied / sample unconditional volatility
    info = dict(n_clipped=int(clip.sum()), n_retries=int(retries),
                persistence_median=float(np.median(alpha + beta)), persistence_max=float((alpha + beta).max()),
                nu_median=float(np.median(nu)), nu_min=float(nu.min()),
                uncond_vol_ratio_min_med_max=[float(ratio.min()), float(np.median(ratio)), float(ratio.max())],
                resid_m2_min_max=[float((Z ** 2).mean(0).min()), float((Z ** 2).mean(0).max())])
    return dict(mu=mu, omega=omega, alpha=alpha, beta=beta, nu=nu), Z, info


def _simulate(par: dict, Z: np.ndarray, h0: np.ndarray) -> np.ndarray:
    """Push innovations Z (L, N) through each asset's recursion; raw-unit returns."""
    om, al, be = par["omega"], par["alpha"], par["beta"]
    e = np.empty_like(Z)
    h = h0.copy()
    for t in range(len(Z)):
        e[t] = np.sqrt(h) * Z[t]
        h = om + al * e[t] ** 2 + be * h
    out = (par["mu"] + e) / SCALE
    if not np.isfinite(out).all():
        raise FloatingPointError("simulated GARCH path is not finite")
    return out


class GarchFHS(Generator):
    """Per-asset GARCH(1,1)-t with cross-sectionally resampled standardised residuals."""
    family = "dynamic-classical"
    impl_source = "own on arch"

    def fit(self, data: FitData, seed: int) -> dict:
        self.par, self.Z, info = fit_univariate(data.returns)
        m2 = (self.Z ** 2).mean(0)
        eff = self.par["alpha"] * m2 + self.par["beta"]
        info["effective_persistence_max"] = float(eff.max())
        bad = eff >= 0.999
        if bad.any():           # resampled residuals have E[z^2] slightly above 1: clip effective persistence
            shrink = 0.999 / eff[bad]
            self.par["alpha"] = self.par["alpha"].copy()
            self.par["beta"] = self.par["beta"].copy()
            self.par["alpha"][bad] *= shrink
            self.par["beta"][bad] *= shrink
            eff = self.par["alpha"] * m2 + self.par["beta"]
        info["n_effective_persistence_clipped"] = int(bad.sum())
        self.h0 = self.par["omega"] / (1 - eff)
        return info

    def sample(self, n, rng):
        idx = rng.integers(0, len(self.Z), BURN_IN + n * P.T)
        return chop(_simulate(self.par, self.Z[idx], self.h0)[BURN_IN:], n)


# ------------------------------------------------------------------------- DCC
def dcc_nll(a: float, b: float, Z: np.ndarray, outer: np.ndarray, Qbar: np.ndarray) -> float:
    """Negative second-stage log-likelihood (Engle 2002, eq. 30) divided by D."""
    D, N = Z.shape
    Q = np.empty((D, N, N))
    Q[0] = Qbar
    c = (1 - a - b) * Qbar
    for t in range(1, D):
        Q[t] = c + a * outer[t - 1] + b * Q[t - 1]
    d = np.sqrt(np.einsum("tii->ti", Q))
    Lc = np.linalg.cholesky(Q)                      # R = diag(1/d) Q diag(1/d)
    logdet = 2 * (np.log(np.einsum("tii->ti", Lc)).sum(1) - np.log(d).sum(1))
    x = np.linalg.solve(Lc, (Z * d)[:, :, None])[:, :, 0]
    return float(0.5 * (logdet + (x ** 2).sum(1) - (Z ** 2).sum(1)).mean())


def _ab(theta) -> tuple[float, float]:
    s = MAX_PERSISTENCE / (1 + np.exp(-theta[0]))
    a = s / (1 + np.exp(-theta[1]))
    return float(a), float(s - a)


def dcc_fit(Z: np.ndarray, starts=((0.02, 0.95), (0.05, 0.85), (0.02, 0.50))) -> tuple[float, float, dict]:
    """Estimate DCC(1,1) (a, b) on standardised residuals Z (D, N)."""
    from scipy.optimize import minimize
    Qbar = Z.T @ Z / len(Z)
    outer = Z[:, :, None] * Z[:, None, :]
    logit = lambda p: np.log(p / (1 - p))
    runs = []
    for a0, b0 in starts:
        x0 = [logit((a0 + b0) / MAX_PERSISTENCE), logit(a0 / (a0 + b0))]
        r = minimize(lambda th: dcc_nll(*_ab(th), Z, outer, Qbar), x0, method="Nelder-Mead",
                     options=dict(xatol=1e-3, fatol=1e-8, maxfev=400))
        a, b = _ab(r.x)
        runs.append(dict(start=[a0, b0], a=a, b=b, nll_per_day=float(r.fun), success=bool(r.success), nfev=int(r.nfev)))
    good = [r for r in runs if r["success"] and np.isfinite(r["nll_per_day"])]
    if not good:
        raise RuntimeError(f"DCC optimisation did not converge from any start: {runs}")
    best = min(good, key=lambda r: r["nll_per_day"])
    nll0 = dcc_nll(0.0, 0.0, Z, outer, Qbar)       # constant-correlation benchmark
    return best["a"], best["b"], dict(dcc_runs=runs, nll_per_day=best["nll_per_day"], nll_per_day_ccc=nll0)


class DCCGarch(Generator):
    """GARCH(1,1) marginals + Engle DCC(1,1) correlations, multivariate-t innovations."""
    family = "dynamic-classical"
    impl_source = "own"

    def fit(self, data: FitData, seed: int) -> dict:
        self.par, Z, info = fit_univariate(data.returns)
        self.a, self.b, dinfo = dcc_fit(Z)
        self.Qbar = Z.T @ Z / len(Z)
        self.nu = float(np.median(self.par["nu"]))
        self.h0 = self.par["omega"] / (1 - self.par["alpha"] - self.par["beta"])
        return dict(info, a=self.a, b=self.b, df=self.nu, **dinfo)

    def sample(self, n, rng):
        L, N = BURN_IN + n * P.T, len(self.h0)
        g = rng.standard_normal((L, N))
        mix = np.sqrt((self.nu - 2.0) / rng.chisquare(self.nu, L))    # unit-variance multivariate t
        Z = np.empty((L, N))
        Q, c = self.Qbar.copy(), (1 - self.a - self.b) * self.Qbar
        for t in range(L):
            z = (np.linalg.cholesky(Q) @ g[t]) / np.sqrt(np.diag(Q)) * mix[t]
            Z[t] = z
            Q = c + self.a * np.outer(z, z) + self.b * Q
        return chop(_simulate(self.par, Z, self.h0)[BURN_IN:], n)
