"""Value-at-Risk / Expected-Shortfall forecasts and backtest statistics.

Conventions (used by every function here)
  r_t      portfolio log return of day t
  c        coverage level in protocol.RISK["levels"]; alpha = 1 - c is the tail probability
  VaR, ES  POSITIVE loss numbers; a violation ("hit") is r_t < -VaR_t
  Empirical VaR/ES of a sample x of size k at tail probability alpha, with
  m = ceil(alpha * k) and x_(1) <= ... <= x_(k):
      VaR = -x_(m)                  (lower empirical alpha-quantile, inf{x : F_k(x) >= alpha})
      ES  = -mean(x_(1), ..., x_(m))  (mean of the tail at or beyond the quantile)
  The same estimator is used for the k-NN pool, historical simulation and the
  GARCH-FHS standardised residuals, so the three forecasters differ only in the
  sample they look at.

Forecasters (all return arrays of shape (days, len(levels)); column l <-> levels[l])
  knn_var_es            conditional on a scalar state, from a generated (state, next) pool
  historical_simulation rolling window of past returns
  garch_fhs_var_es      GARCH(1,1)-t volatility + filtered historical simulation

Tests
  kupiec_pof                    Kupiec (1995) proportion-of-failures LR, chi2(1)
  christoffersen_independence   Christoffersen (1998) first-order Markov LR, chi2(1)
  christoffersen_cc             Christoffersen (1998) conditional coverage LR, chi2(2)
  basel_traffic_light           Basel (1996) zones from the binomial cumulative probability
  acerbi_szekely_z2             Acerbi and Szekely (2014) test 2, Monte-Carlo p-value
  fz0_loss, quantile_loss       consistent scoring functions (Patton, Ziegel and Chen 2019)

Choices that a reader should know about (none changes a published statistic)
  * christoffersen_cc is the likelihood ratio on the n-1 transitions (Christoffersen's
    definition, conditional on the first observation).  It equals
    kupiec_pof(hits[1:]) + christoffersen_independence(hits) exactly; it is NOT the
    often-used approximation kupiec_pof(hits) + independence, which differs by the
    first observation.
  * Empty transition cells: 0 * log(0) = 0.  A row with no observations (no hit is
    ever followed by another day) contributes likelihood 1, which reproduces
    Christoffersen's reduced likelihood (1 - pi01)^n00 * pi01^n01 when n11 = 0.
    With no hits at all (or only hits) the independence statistic is 0, p = 1.
  * The Z2 p-value is simulated from per-day predictive SAMPLES.  In the simulation a
    draw equal to the VaR atom counts as a violation, because the empirical ES above
    includes that atom; then E[Z2] = 1 - m / (k * alpha), which is exactly 0 when
    alpha * k is an integer and at most 1 / (alpha * k) away from 0 otherwise.
  * garch_fhs_var_es freezes parameters AND the standardised-residual sample at each
    refit; only the variance recursion is updated daily.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from scipy.special import xlogy

__all__ = [
    "empirical_var_es", "knn_var_es", "historical_simulation", "garch_fhs_var_es", "kupiec_pof",
    "christoffersen_independence", "christoffersen_cc", "basel_traffic_light", "acerbi_szekely_z2",
    "fz0_loss", "quantile_loss", "summarise",
]


# ------------------------------------------------------------------ helpers
def _alphas(levels) -> np.ndarray:
    lv = np.atleast_1d(np.asarray(levels, dtype=np.float64))
    if not ((lv > 0.5) & (lv < 1)).all():
        raise ValueError(f"coverage levels must lie in (0.5, 1), got {lv}")
    return 1.0 - lv


def _tail_count(alpha: float, k: int) -> int:
    # 1 - 0.95 is 0.05000000000000004 in floating point: without the guard
    # ceil(alpha * 1000) would be 51, not 50.
    return max(1, int(np.ceil(alpha * k - 1e-9)))


def _check_alpha(alpha: float) -> float:
    if not 0 < alpha < 0.5:
        raise ValueError(f"alpha is the TAIL probability and must lie in (0, 0.5), got {alpha}")
    return float(alpha)


def _hits(hits) -> np.ndarray:
    h = np.asarray(hits)
    if h.ndim != 1 or h.size == 0:
        raise ValueError("hits must be a non-empty 1-d sequence")
    if not np.isin(h, (0, 1)).all():
        raise ValueError("hits must be boolean / 0-1")
    return h.astype(np.int64)


def _forecast_args(r, var, es=None):
    r, var = np.asarray(r, dtype=np.float64), np.asarray(var, dtype=np.float64)
    arrs = [r, var] + ([] if es is None else [np.asarray(es, dtype=np.float64)])
    if any(a.ndim != 1 or a.shape != r.shape for a in arrs):
        raise ValueError("r, var and es must be 1-d arrays of equal length")
    if not all(np.isfinite(a).all() for a in arrs):
        raise ValueError("non-finite value in r / var / es (drop warm-up days first; summarise() does)")
    return arrs


def empirical_var_es(x: np.ndarray, levels, presorted: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Empirical VaR and ES along the last axis: x (..., k) -> two arrays (..., len(levels))."""
    xs = np.asarray(x, dtype=np.float64)
    if not presorted:
        xs = np.sort(xs, axis=-1)
    k = xs.shape[-1]
    ms = [_tail_count(a, k) for a in _alphas(levels)]
    cs = np.cumsum(xs[..., :max(ms)], axis=-1)
    var = np.stack([-xs[..., m - 1] for m in ms], axis=-1)
    es = np.stack([-cs[..., m - 1] / m for m in ms], axis=-1)
    return var, es


# -------------------------------------------------------------- forecasters
def knn_var_es(pool_state, pool_next, state_t, levels, knn_frac: float, return_neighbours: bool = False):
    """VaR/ES of the next-day return conditional on the state, from a generated pool.

    For each day the k = ceil(knn_frac * len(pool)) pool entries whose state is closest
    to state_t[d] (absolute distance) are the predictive sample.  In one dimension the
    k nearest points are contiguous in sorted order, so the window start is found by a
    bisection on the sorted states, vectorised over days.  Distance ties are resolved
    towards the lower state.

    Returns (var, es), each (days, len(levels)); with return_neighbours also the
    (days, k) neighbour returns, sorted ascending within each day.
    """
    s = np.asarray(pool_state, dtype=np.float64).ravel()
    y = np.asarray(pool_next, dtype=np.float64).ravel()
    x = np.atleast_1d(np.asarray(state_t, dtype=np.float64))
    if s.shape != y.shape or s.size == 0:
        raise ValueError("pool_state and pool_next must be non-empty and of equal length")
    if x.ndim != 1:
        raise ValueError("state_t must be 1-d")
    if not (np.isfinite(s).all() and np.isfinite(y).all() and np.isfinite(x).all()):
        raise ValueError("non-finite value in the pool or in state_t")
    if not 0 < knn_frac <= 1:
        raise ValueError("knn_frac must lie in (0, 1]")
    n = s.size
    k = min(n, int(np.ceil(knn_frac * n - 1e-9)))
    order = np.argsort(s, kind="stable")
    s, y = s[order], y[order]
    pos = np.searchsorted(s, x, side="left")
    lo, hi = np.clip(pos - k, 0, n - k), np.clip(pos, 0, n - k)
    while (lo < hi).any():                      # smallest start whose window cannot improve by moving right
        mid = (lo + hi) // 2
        nxt = np.minimum(mid + k, n - 1)        # only read where mid < hi <= n - k, i.e. mid + k <= n - 1
        right = (lo < hi) & (x - s[mid] > s[nxt] - x)
        lo = np.where(right, mid + 1, lo)
        hi = np.where((lo < hi) & ~right, mid, hi)
    nb = np.sort(y[lo[:, None] + np.arange(k)[None, :]], axis=1)
    var, es = empirical_var_es(nb, levels, presorted=True)
    return (var, es, nb) if return_neighbours else (var, es)


def historical_simulation(r, window: int, levels) -> tuple[np.ndarray, np.ndarray]:
    """Rolling empirical VaR/ES: the forecast for day t uses r[t-window : t] only.

    Returns (var, es), each (len(r), len(levels)); rows t < window are NaN.
    """
    r = np.asarray(r, dtype=np.float64)
    if r.ndim != 1 or not np.isfinite(r).all():
        raise ValueError("r must be a finite 1-d array")
    if window < 2:
        raise ValueError("window must be at least 2")
    L = len(_alphas(levels))
    var, es = np.full((len(r), L), np.nan), np.full((len(r), L), np.nan)
    if len(r) > window:
        past = np.lib.stride_tricks.sliding_window_view(r, window)[:-1]     # row i = r[i : i+window]
        var[window:], es[window:] = empirical_var_es(past, levels)
    return var, es


def garch_fhs_var_es(r, levels, refit_every: int = 250, min_obs: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """GARCH(1,1)-t filtered historical simulation, strictly out of sample.

    At t0 = min_obs, min_obs + refit_every, ... a constant-mean GARCH(1,1) with
    Student-t innovations is estimated by ML (arch) on r[:t0].  For the days
    t0 <= t < t0 + refit_every the parameters and the standardised residuals
    z = (r - mu) / sigma of that fit are frozen and only
        sigma2_t = omega + a * (r_{t-1} - mu)^2 + b * sigma2_{t-1}
    is rolled forward, so the forecast for day t uses r[:t] only.  Then
        VaR_t = -(mu + sigma_t * q_alpha(z)),   ES_t = -(mu + sigma_t * mean(z | z <= q_alpha(z))).
    Rows t < min_obs are NaN.  A non-converged fit is reported with a warning; a
    non-finite or non-positive parameter raises.
    """
    from arch import arch_model
    r = np.asarray(r, dtype=np.float64)
    if r.ndim != 1 or not np.isfinite(r).all():
        raise ValueError("r must be a finite 1-d array")
    if refit_every < 1 or min_obs < 100:
        raise ValueError("refit_every must be >= 1 and min_obs >= 100")
    L = len(_alphas(levels))
    var, es = np.full((len(r), L), np.nan), np.full((len(r), L), np.nan)
    y = 100.0 * r                                # percent returns: well-scaled likelihood
    for t0 in range(min_obs, len(r), refit_every):
        res = arch_model(y[:t0], mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(
            disp="off", show_warning=False)
        mu, om, a, b = (float(res.params[k]) for k in ("mu", "omega", "alpha[1]", "beta[1]"))
        sig = np.asarray(res.conditional_volatility, dtype=np.float64)
        eps = np.asarray(res.resid, dtype=np.float64)
        if not (np.isfinite([mu, om, a, b]).all() and om > 0 and np.isfinite(sig).all() and (sig > 0).all()):
            raise RuntimeError(f"GARCH fit on r[:{t0}] gave unusable parameters {dict(res.params)}")
        if res.convergence_flag != 0:
            warnings.warn(f"GARCH fit on r[:{t0}] did not converge (flag {res.convergence_flag})", RuntimeWarning)
        zv, ze = empirical_var_es(eps / sig, levels)             # minus quantile, minus tail mean of z
        h = om + a * eps[-1] ** 2 + b * sig[-1] ** 2             # variance forecast for day t0
        for t in range(t0, min(t0 + refit_every, len(r))):
            s = np.sqrt(h)
            var[t], es[t] = (s * zv - mu) / 100.0, (s * ze - mu) / 100.0
            h = om + a * (y[t] - mu) ** 2 + b * h
    return var, es


# ----------------------------------------------------------- coverage tests
def _bern_loglik(n0: float, n1: float, p: float) -> float:
    """log[(1-p)^n0 * p^n1] with 0 * log(0) = 0."""
    return float(xlogy(n0, 1.0 - p) + xlogy(n1, p))


def kupiec_pof(hits, alpha: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures test.  Returns (LR, p-value), LR ~ chi2(1) under H0.

    With n days, x hits and p_hat = x / n:
        LR = -2 * { log[(1-alpha)^(n-x) alpha^x] - log[(1-p_hat)^(n-x) p_hat^x] }.
    x = 0 gives LR = -2 n log(1-alpha) and x = n gives LR = -2 n log(alpha) (no NaN).
    """
    h, alpha = _hits(hits), _check_alpha(alpha)
    n, x = h.size, int(h.sum())
    lr = max(0.0, -2.0 * (_bern_loglik(n - x, x, alpha) - _bern_loglik(n - x, x, x / n)))
    return lr, float(stats.chi2.sf(lr, 1))


def _transitions(h: np.ndarray) -> tuple[int, int, int, int]:
    a, b = h[:-1], h[1:]
    return (int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum()),
            int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum()))


def _markov_loglik(n00: int, n01: int, n10: int, n11: int) -> float:
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0     # an empty row has likelihood 1 whatever pi is
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    return _bern_loglik(n00, n01, pi01) + _bern_loglik(n10, n11, pi11)


def christoffersen_independence(hits) -> tuple[float, float]:
    """Christoffersen independence test.  Returns (LR, p-value), LR ~ chi2(1) under H0.

    n_ij = number of days with hit state i followed by state j (n - 1 transitions),
    pi01 = n01 / (n00 + n01), pi11 = n11 / (n10 + n11), pi = (n01 + n11) / (n - 1):
        LR = -2 * { log[(1-pi)^(n00+n10) pi^(n01+n11)]
                    - log[(1-pi01)^n00 pi01^n01 (1-pi11)^n10 pi11^n11] }.
    """
    h = _hits(hits)
    if h.size < 2:
        raise ValueError("need at least two observations")
    n00, n01, n10, n11 = _transitions(h)
    pi = (n01 + n11) / (h.size - 1)
    lr = max(0.0, -2.0 * (_bern_loglik(n00 + n10, n01 + n11, pi) - _markov_loglik(n00, n01, n10, n11)))
    return lr, float(stats.chi2.sf(lr, 1))


def christoffersen_cc(hits, alpha: float) -> tuple[float, float]:
    """Christoffersen conditional-coverage test.  Returns (LR, p-value), LR ~ chi2(2) under H0.

        LR = -2 * { log[(1-alpha)^(n00+n10) alpha^(n01+n11)] - log L(pi01, pi11) }
    on the n - 1 transitions (see the module docstring for the relation to Kupiec).
    """
    h, alpha = _hits(hits), _check_alpha(alpha)
    if h.size < 2:
        raise ValueError("need at least two observations")
    n00, n01, n10, n11 = _transitions(h)
    lr = max(0.0, -2.0 * (_bern_loglik(n00 + n10, n01 + n11, alpha) - _markov_loglik(n00, n01, n10, n11)))
    return lr, float(stats.chi2.sf(lr, 2))


def basel_traffic_light(n_violations: int, n_obs: int, alpha: float = 0.01) -> str:
    """Basel zone from P(X <= n_violations), X ~ Binomial(n_obs, alpha).

    green: below 95%; yellow: from 95% and below 99.99%; red: from 99.99%.
    For n_obs = 250, alpha = 0.01 this is the Basel table (green 0-4, yellow 5-9, red 10+).
    """
    alpha = _check_alpha(alpha)
    if not 0 <= n_violations <= n_obs or n_obs < 1:
        raise ValueError("need 0 <= n_violations <= n_obs and n_obs >= 1")
    cum = float(stats.binom.cdf(n_violations, n_obs, alpha))
    return "green" if cum < 0.95 else ("yellow" if cum < 0.9999 else "red")


# ------------------------------------------------------------------ ES tests
def acerbi_szekely_z2(r, var, es, alpha: float, neighbours=None, n_sim: int = 2000,
                      rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Acerbi and Szekely (2014) second test.  Returns (Z2, p-value).

        Z2 = sum_t r_t * 1{r_t < -VaR_t} / (T * alpha * ES_t) + 1.
    E[Z2] = 0 when VaR and ES are correct (E[r 1{r < -VaR}] = -alpha * ES); Z2 < 0 means
    the tail losses were larger or more frequent than forecast.

    p-value (only when `neighbours`, a (T, k) array holding a predictive sample for each
    day, is given; NaN otherwise): n_sim paths are drawn day by day from those samples,
    Z2 is recomputed with the SAME var and es, and
        p = (1 + #{Z2_sim <= Z2}) / (n_sim + 1)       (one-sided: reject for small Z2).
    rng defaults to np.random.default_rng(0).
    """
    r, var, es = _forecast_args(r, var, es)
    alpha = _check_alpha(alpha)
    if not (es > 0).all():
        raise ValueError("ES must be strictly positive")
    T = r.size
    z2 = float((r * (r < -var) / es).sum() / (T * alpha) + 1.0)
    if neighbours is None:
        return z2, float("nan")
    nb = np.asarray(neighbours, dtype=np.float64)
    if nb.ndim != 2 or nb.shape[0] != T or not np.isfinite(nb).all():
        raise ValueError("neighbours must be a finite (len(r), k) array")
    rng = np.random.default_rng(0) if rng is None else rng
    rows, sims, done = np.arange(T), np.empty(n_sim), 0
    chunk = max(1, int(2_000_000 // T))
    while done < n_sim:
        b = min(chunk, n_sim - done)
        X = nb[rows[None, :], rng.integers(0, nb.shape[1], size=(b, T))]
        sims[done:done + b] = (X * (X <= -var[None, :]) / es[None, :]).sum(axis=1) / (T * alpha) + 1.0
        done += b
    return z2, float((1 + (sims <= z2).sum()) / (n_sim + 1))


def fz0_loss(r, var, es, alpha: float) -> np.ndarray:
    """Fissler-Ziegel FZ0 loss per day (Patton, Ziegel and Chen 2019, eq. 6).

    The paper writes it for the quantile v < 0 and the tail mean e <= v < 0:
        L(Y, v, e) = -1{Y <= v} (v - Y) / (alpha * e) + v / e + log(-e) - 1.
    Sign conversion to this module's positive loss numbers, v = -VaR and e = -ES:
        L = 1{r < -VaR} * (-VaR - r) / (alpha * ES) + VaR / ES + log(ES) - 1.
    (<= or < is immaterial: the term is zero at r = -VaR.)  Its expectation is
    minimised by the true (VaR, ES); lower is better.  Requires ES > 0.
    """
    r, var, es = _forecast_args(r, var, es)
    alpha = _check_alpha(alpha)
    if not (es > 0).all():
        raise ValueError("ES must be strictly positive")
    return (r < -var) * (-var - r) / (alpha * es) + var / es + np.log(es) - 1.0


def quantile_loss(r, var, alpha: float) -> np.ndarray:
    """Pinball loss per day of the alpha-quantile forecast q = -VaR: (alpha - 1{r < q}) * (r - q)."""
    r, var = _forecast_args(r, var)
    alpha = _check_alpha(alpha)
    q = -var
    return (alpha - (r < q)) * (r - q)


def summarise(r, var, es, alpha: float, neighbours=None, n_sim: int = 2000,
              rng: np.random.Generator | None = None) -> dict:
    """All backtest numbers of one forecast series at one tail probability.

    Days on which r, var or es is not finite (forecaster warm-up) are dropped; `n` is
    the number of days actually tested.  The traffic light uses Binomial(n, alpha).
    z2_p is NaN unless per-day predictive samples (`neighbours`) are supplied.
    """
    r, var, es = (np.asarray(a, dtype=np.float64) for a in (r, var, es))
    if not (r.ndim == 1 and r.shape == var.shape == es.shape):
        raise ValueError("r, var and es must be 1-d arrays of equal length")
    ok = np.isfinite(r) & np.isfinite(var) & np.isfinite(es)
    if ok.sum() < 2:
        raise ValueError("fewer than two days with a finite forecast")
    r, var, es = r[ok], var[ok], es[ok]
    if neighbours is not None:
        neighbours = np.asarray(neighbours)[ok]
    hits = r < -var
    n, x = int(hits.size), int(hits.sum())
    n_bad_es = int((es <= 0).sum())
    if n_bad_es:        # ES forecast is a gain: Z2 and FZ0 are undefined; VaR-only tests still apply
        return dict(
            n=n, violations=x, expected=float(alpha * n), rate=float(x / n),
            kupiec_p=kupiec_pof(hits, alpha)[1],
            christoffersen_ind_p=christoffersen_independence(hits)[1],
            cc_p=christoffersen_cc(hits, alpha)[1],
            traffic_light=basel_traffic_light(x, n, alpha),
            z2=float("nan"), z2_p=float("nan"), fz0=float("nan"),
            quantile_loss=float(quantile_loss(r, var, alpha).mean()), es_nonpositive_days=n_bad_es,
        )
    z2, z2_p = acerbi_szekely_z2(r, var, es, alpha, neighbours, n_sim, rng)
    return dict(
        n=n, violations=x, expected=float(alpha * n), rate=float(x / n),
        kupiec_p=kupiec_pof(hits, alpha)[1],
        christoffersen_ind_p=christoffersen_independence(hits)[1],
        cc_p=christoffersen_cc(hits, alpha)[1],
        traffic_light=basel_traffic_light(x, n, alpha),
        z2=z2, z2_p=z2_p,
        fz0=float(fz0_loss(r, var, es, alpha).mean()),
        quantile_loss=float(quantile_loss(r, var, alpha).mean()),
    )
