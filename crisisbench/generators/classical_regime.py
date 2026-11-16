"""Regime baselines: a supervised two-regime Gaussian and an unsupervised Gaussian HMM.

Both are Markov-switching multivariate Gaussians: a hidden state follows a first-order
Markov chain and each day is drawn N(mu_k, Sigma_k) with a FULL covariance per state.
They differ only in how the states are obtained.

RegimeGaussian (supervised)
  Each fit-period day takes the frozen crisis label (labels.label_windows with
  data.rule) of the 60-day window ENDING on that day; the first T-1 days take the label
  of the first window.  Per-regime mean and covariance are sample moments; the 2x2
  transition matrix is the row-normalised count of daily label transitions.
  If a regime has n_k < 3N days its covariance is shrunk linearly toward the pooled
  (all-days) covariance with weight w = n_k / (3N) on the regime estimate; reported in
  info["shrink_weight"].
  Caveat (by construction of the brief, not a bug): the label of a day is a trailing
  20-day statistic, so day labels lag the true regime and regime moments are mildly
  contaminated at regime edges.

HMMGaussian (unsupervised, hmmlearn GaussianHMM, Baum-Welch)
  Fitted on per-asset standardised returns.  K is chosen in `ks` by BIC, each K fitted
  from `restarts` seeds keeping the highest log-likelihood.
  Deviations / implementation notes
  * hmmlearn applies `min_covar` to full covariances only at initialisation, not in the
    M-step.  The ridge is therefore ALSO imposed in the M-step through a diagonal
    `covars_prior = ridge * I` (a MAP estimate: `ridge` pseudo-days of identity scatter
    per state, in standardised units).  The log-likelihood used for BIC is the plain
    data log-likelihood of the fitted model.
  * Degenerate-state guard: a K whose best fit has a state with fewer than 3N Viterbi
    days is rejected; the chosen K is the best BIC among the remaining (smaller or
    larger) candidates.  If every K is rejected, fit raises.
  * "Crisis" for sample_regime is the state with the highest average (standardised)
    variance, "normal" the lowest; with K > 2 the middle states have no handle.

Sampling (both): one long daily path from the chain (500-day burn-in), cut into
non-overlapping windows.  sample_regime: every day i.i.d. from the requested state.
"""
from __future__ import annotations

import numpy as np

from .. import protocol as P
from .base import FitData, Generator, chop

BURN_IN = 500


def simulate_chain(trans: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """State path of a first-order Markov chain, started in state 0 (callers burn in)."""
    cum = np.cumsum(trans, axis=1)
    cum[:, -1] = 1.0                       # guard against rounding in the last column
    u = rng.random(length)
    s = np.empty(length, dtype=np.int64)
    cur = 0
    for t in range(length):
        cur = int(np.searchsorted(cum[cur], u[t], side="right"))
        s[t] = cur
    return s


def stationary(trans: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eig(trans.T)
    p = np.real(v[:, np.argmax(np.real(w))])
    return p / p.sum()


class _MarkovGaussian(Generator):
    """Shared sampler. Subclasses set trans (K,K), mu (K,N), L (K,N,N) Cholesky factors,
    loc/scale (N,) to undo standardisation, and crisis_state / normal_state."""
    family = "regime"
    supports_regime = True
    loc: np.ndarray | float = 0.0
    scale: np.ndarray | float = 1.0

    def _emit(self, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        eps = rng.standard_normal((len(states), self.mu.shape[1]))
        X = np.empty_like(eps)
        for k in range(len(self.mu)):
            m = states == k
            X[m] = self.mu[k] + eps[m] @ self.L[k].T
        return self.loc + self.scale * X

    def sample(self, n, rng):
        s = simulate_chain(self.trans, BURN_IN + n * P.T, rng)[BURN_IN:]
        return chop(self._emit(s, rng), n)

    def sample_regime(self, n, regime, rng):
        if regime not in (0, 1):
            raise ValueError(f"regime must be 0 (normal) or 1 (crisis), got {regime!r}")
        k = self.crisis_state if regime == 1 else self.normal_state
        return chop(self._emit(np.full(n * P.T, k, dtype=np.int64), rng), n)


class RegimeGaussian(_MarkovGaussian):
    """Supervised two-regime Markov-switching Gaussian (state 1 = crisis, 0 = normal)."""
    crisis_state, normal_state = 1, 0

    def fit(self, data: FitData, seed: int) -> dict:
        X = data.returns
        D, N = X.shape
        wl = np.asarray(data.window_labels(), dtype=np.int64)
        day = np.concatenate([np.full(P.T - 1, wl[0]), wl])
        assert len(day) == D
        counts = np.bincount(day, minlength=2)
        if counts.min() < 2:
            raise ValueError(f"a regime has fewer than 2 days (normal/crisis days = {counts.tolist()})")
        pooled = np.cov(X, rowvar=False)
        self.mu = np.stack([X[day == k].mean(0) for k in (0, 1)])
        w = np.minimum(1.0, counts / (3.0 * N))
        cov = [w[k] * np.cov(X[day == k], rowvar=False) + (1 - w[k]) * pooled for k in (0, 1)]
        self.L = np.stack([np.linalg.cholesky(c + 1e-12 * np.eye(N)) for c in cov])
        C = np.zeros((2, 2))
        np.add.at(C, (day[:-1], day[1:]), 1.0)
        if (C.sum(1) == 0).any():
            raise ValueError(f"a regime is never left or never observed before the last day: counts {C.tolist()}")
        self.trans = C / C.sum(1, keepdims=True)
        pi = stationary(self.trans)
        return dict(days_normal=int(counts[0]), days_crisis=int(counts[1]),
                    shrink_weight=[float(x) for x in w], shrunk=[bool(x < 1) for x in w],
                    trans=self.trans.tolist(), stationary_crisis=float(pi[1]),
                    n_crisis_spells=int(C[0, 1] + (day[0] == 1)))


class HMMGaussian(_MarkovGaussian):
    """Gaussian HMM with full covariances on standardised returns; K by BIC."""
    impl_source = "hmmlearn"

    def __init__(self, ks=(2, 3, 4), restarts: int = 3, n_iter: int = 200, tol: float = 1e-3,
                 min_covar: float = 1e-3, ridge: float = 1.0, **kw):
        super().__init__(ks=list(ks), restarts=restarts, n_iter=n_iter, tol=tol, min_covar=min_covar,
                         ridge=ridge, **kw)
        self.ks, self.restarts, self.n_iter, self.tol = tuple(ks), restarts, n_iter, tol
        self.min_covar, self.ridge = min_covar, ridge

    def _fit_k(self, Z: np.ndarray, K: int, seed: int):
        """Best-of-restarts fit for K states: (model, loglik, record)."""
        from hmmlearn.hmm import GaussianHMM
        N = Z.shape[1]
        best, lls, errors = None, [], []
        for r in range(self.restarts):
            m = GaussianHMM(n_components=K, covariance_type="full", min_covar=self.min_covar,
                            covars_prior=self.ridge * np.eye(N), covars_weight=1.0,
                            n_iter=self.n_iter, tol=self.tol, random_state=seed * 1000 + 10 * K + r)
            try:
                m.fit(Z)
                ll = float(m.score(Z))
            except (ValueError, np.linalg.LinAlgError) as e:    # recorded in info, never hidden
                errors.append(repr(e))
                continue
            if not np.isfinite(ll):
                errors.append("non-finite log-likelihood")
                continue
            lls.append(ll)
            if best is None or ll > best[1]:
                best = (m, ll)
        rec = dict(K=K, restart_logliks=lls, restart_errors=errors)
        if best is None:
            return None, None, rec
        m, ll = best
        D = len(Z)
        n_par = (K - 1) + K * (K - 1) + K * N + K * N * (N + 1) // 2
        occ = np.bincount(m.predict(Z), minlength=K)
        rec.update(loglik=ll, n_params=n_par, bic=float(-2 * ll + n_par * np.log(D)),
                   converged=bool(m.monitor_.converged), em_iters=int(m.monitor_.iter),
                   state_days=occ.tolist(), degenerate=bool(occ.min() < 3 * N))
        return m, ll, rec

    def fit(self, data: FitData, seed: int) -> dict:
        X = data.returns
        N = X.shape[1]
        self.loc, self.scale = X.mean(0), X.std(0, ddof=1)
        Z = (X - self.loc) / self.scale
        fits, records = {}, []
        for K in self.ks:
            m, _, rec = self._fit_k(Z, K, seed)
            records.append(rec)
            if m is not None and not rec["degenerate"]:
                fits[K] = (m, rec["bic"])
        if not fits:
            raise RuntimeError(f"no HMM with K in {self.ks} has all states >= 3N = {3 * N} days: {records}")
        K = min(fits, key=lambda k: fits[k][1])
        m = fits[K][0]
        self.trans = np.asarray(m.transmat_, dtype=np.float64)
        self.mu = np.asarray(m.means_, dtype=np.float64)
        cov = np.asarray(m.covars_, dtype=np.float64)
        self.L = np.linalg.cholesky(cov)
        avg_var = np.einsum("kii->k", cov) / N
        self.crisis_state, self.normal_state = int(np.argmax(avg_var)), int(np.argmin(avg_var))
        return dict(K=K, candidates=records, trans=self.trans.tolist(), state_avg_var=avg_var.tolist(),
                    crisis_state=self.crisis_state, normal_state=self.normal_state,
                    stationary=stationary(self.trans).tolist())
