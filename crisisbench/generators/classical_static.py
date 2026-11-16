"""Static and resampling baselines: no regime structure by construction."""
from __future__ import annotations

import numpy as np

from .. import protocol as P
from .base import FitData, Generator, chop


class IIDGaussian(Generator):
    """Days are i.i.d. N(mu, Sigma) with the full sample covariance."""
    family = "static"

    def fit(self, data: FitData, seed: int) -> dict:
        self.mu = data.returns.mean(0)
        self.L = np.linalg.cholesky(np.cov(data.returns, rowvar=False) + 1e-12 * np.eye(data.n_assets))
        return {}

    def sample(self, n, rng):
        z = rng.standard_normal((n, P.T, len(self.mu)))
        return self.mu + z @ self.L.T


class TPPCA(Generator):
    """Static multivariate Student-t with a rank-k factor covariance (probabilistic PCA).

    Sigma = B B' + diag(psi) from the top-k eigenpairs of the sample covariance; one
    chi-square mixing variable per day gives joint fat tails; nu by profile likelihood.
    Days are i.i.d., so there is no regime or volatility clustering by construction.
    """
    family = "static"

    def __init__(self, k: int = 8, **kw):
        super().__init__(k=k, **kw)
        self.k = k

    def fit(self, data: FitData, seed: int) -> dict:
        from scipy import special
        X = data.returns
        self.mu = X.mean(0)
        S = np.cov(X, rowvar=False)
        w, V = np.linalg.eigh(S)
        w, V = w[::-1], V[:, ::-1]
        B = V[:, :self.k] * np.sqrt(np.clip(w[:self.k], 0, None))
        psi = np.clip(np.diag(S) - (B ** 2).sum(1), 1e-10, None)
        Sig = B @ B.T + np.diag(psi)
        N = X.shape[1]
        Xc = X - self.mu
        sol = np.linalg.solve(Sig, Xc.T).T
        maha = (Xc * sol).sum(1)                    # Mahalanobis w.r.t. the covariance
        _, logdet = np.linalg.slogdet(Sig)
        best = (-np.inf, 8.0)
        for nu in np.arange(2.5, 40.01, 0.25):      # scale matrix = Sig * (nu-2)/nu keeps Cov = Sig
            s = (nu - 2.0) / nu
            ll = (special.gammaln((nu + N) / 2) - special.gammaln(nu / 2) - 0.5 * N * np.log(nu * np.pi)
                  - 0.5 * (logdet + N * np.log(s)) - 0.5 * (nu + N) * np.log1p(maha / (s * nu))).sum()
            if ll > best[0]:
                best = (ll, float(nu))
        self.nu = best[1]
        self.L = np.linalg.cholesky(Sig * (self.nu - 2.0) / self.nu)
        return dict(nu=self.nu, k=self.k)

    def sample(self, n, rng):
        N = len(self.mu)
        z = rng.standard_normal((n, P.T, N)) @ self.L.T
        w = rng.chisquare(self.nu, size=(n, P.T, 1)) / self.nu
        return self.mu + z / np.sqrt(w)


class StationaryBootstrap(Generator):
    """Politis-Romano stationary bootstrap of whole cross-sections (mean block = `block` days).

    A resampling reference, not a model: it replays real days in real order within
    blocks, so it inherits real regimes inside a block but cannot create new ones.
    """
    family = "resampling"

    def __init__(self, block: int = 20, **kw):
        super().__init__(block=block, **kw)
        self.block = block

    def fit(self, data: FitData, seed: int) -> dict:
        self.X = data.returns.copy()
        return {}

    def sample(self, n, rng):
        D = len(self.X)
        L = n * P.T
        idx = np.empty(L, dtype=np.int64)
        idx[0] = rng.integers(D)
        jump = rng.random(L) < 1.0 / self.block
        fresh = rng.integers(0, D, L)
        for t in range(1, L):
            idx[t] = fresh[t] if jump[t] else (idx[t - 1] + 1) % D
        return chop(self.X[idx], n)
