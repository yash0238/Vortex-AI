"""CG-NSDE v3: v2 plus a graph-Hawkes cascade in the volatility states (post hoc, disclosed).

WHY (motivation recorded before any v3 run; see docs/STATE.md and docs/step0_prereg.md)
  v2 reproduces in-sample crisis dependence but (i) under-produces volatility clustering (ACF of squared
  returns 0.075 vs 0.142 real) and (ii) trained to 2019 it produces a crisis boost of 22% where COVID-2020
  had 101%: once a stress episode starts, v2 cannot amplify it.  Step 0 (French 49 industries 1926-2026)
  found that daily counts of 1%-tail exceedances follow a near-critical self-exciting process (branching
  ratio 0.93-0.95) that forecasts the size of record crashes better than an extreme-value baseline, given
  their onset.  v3 builds that mechanism into the generator: tail events raise the volatility of the
  stocks connected to them, and of the market factor, so stress can cascade through the asset graph.

Mechanism (on top of v2's EGARCH-type factor / idiosyncratic log-variance recursion)
  e_{t,i}     = 1[x_{t,i} < q_i], q_i the 1% quantile of asset i's z-scored TRAINING returns (unconditional,
                so the feedback is self-exciting, not self-normalising)
  A           = row-stochastic contagion graph: softmax over the k nearest neighbours (|correlation| of the
                training days) of learnable logits initialised at |corr|; diagonal excluded
  v^i_{t+1,i} += kappa_s e_{t,i} + kappa_g sum_j A_ij e_{t,j}           (own and neighbour contagion)
  v^f_{t+1,k} += kappa_m,k mean_j e_{t,j}                                (market-wide cascade, per factor)
  kappa >= 0 (softplus).  With the persistence rho of v2 every excitation decays geometrically, i.e. an
  exponential Hawkes kernel in log-variance.  Teacher-forced on real returns in training (the indicator is
  data, no gradient needed), fed with the model's own draws in sampling, where it can cascade.
Everything else (graph encoder, regime prior, latent SDE, SupCon, graph loss, FHS innovations) is v2.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .cgnsde2 import CGNSDE2, DEFAULTS as DEFAULTS2, Net2, V_MAX, V_MIN

DEFAULTS = dict(DEFAULTS2, cascade_self=True, cascade_graph=True, cascade_mkt=True, contagion_k=10,
                contagion_graph="knn", tail_q=0.01, kappa_init=-3.0, use_regime=True)


class _NoLabels:
    """FitData proxy whose window labels are all 'normal': the model never sees the crisis rule."""
    def __init__(self, data):
        self._d = data

    def __getattr__(self, k):
        return getattr(self._d, k)

    def window_labels(self):
        return np.zeros(len(self._d.windows()), dtype=np.int8)


class Net3(Net2):
    def __init__(self, n_assets: int, hp: dict, T=None, init=None):
        super().__init__(n_assets, hp, init=init) if T is None else super().__init__(n_assets, hp, T, init)
        N, K = self.N, self.K
        self.c_self, self.c_graph, self.c_mkt = bool(hp["cascade_self"]), bool(hp["cascade_graph"]), bool(hp["cascade_mkt"])
        self.graph_kind = hp["contagion_graph"]
        if self.graph_kind not in ("knn", "uniform"):
            raise ValueError(f"unknown contagion_graph {self.graph_kind!r}")
        self.register_buffer("q", torch.full((N,), -2.33))
        self.register_buffer("mask", torch.ones(N, N) - torch.eye(N))
        self.logit = nn.Parameter(torch.zeros(N, N))
        k0 = float(hp["kappa_init"])
        self.kappa_s_raw = nn.Parameter(torch.tensor(k0))
        self.kappa_g_raw = nn.Parameter(torch.tensor(k0))
        self.kappa_m_raw = nn.Parameter(torch.full((K,), k0))
        self._k = int(hp["contagion_k"])
        self._tq = float(hp["tail_q"])

    def set_cascade(self, Ztr: np.ndarray) -> None:
        """Thresholds and contagion graph from the z-scored TRAINING days only."""
        N = self.N
        self.q.copy_(torch.from_numpy(np.quantile(Ztr, self._tq, axis=0)).float())
        C = np.abs(np.corrcoef(Ztr.T))
        np.fill_diagonal(C, 0.0)
        if self.graph_kind == "uniform":
            M = np.ones((N, N)) - np.eye(N)
            L = np.zeros((N, N))
        else:
            M = np.zeros((N, N))
            nb = np.argsort(-C, axis=1)[:, :self._k]
            M[np.arange(N)[:, None], nb] = 1.0
            L = np.log(np.clip(C, 1e-3, None))
        self.mask.copy_(torch.from_numpy(M).float())
        with torch.no_grad():
            self.logit.copy_(torch.from_numpy(L).float())
        if self.graph_kind == "uniform":
            self.logit.requires_grad_(False)

    def contagion(self) -> torch.Tensor:
        z = self.logit.masked_fill(self.mask == 0, float("-inf"))
        return torch.softmax(z, dim=1)

    def kappas(self):
        f = lambda r, on: F.softplus(r) if on else torch.zeros_like(r)
        return f(self.kappa_s_raw, self.c_self), f(self.kappa_g_raw, self.c_graph), f(self.kappa_m_raw, self.c_mkt)

    def step(self, x, a, v, nu):
        logp, vn, Bs, ds = super().step(x, a, v, nu)
        if self.vol_mode != "factor" or not (self.c_self or self.c_graph or self.c_mkt):
            return logp, vn, Bs, ds
        K = self.K
        e = (x < self.q).to(x.dtype)                                         # (b, N) tail events
        ks, kg, km = self.kappas()
        add_i = ks * e + kg * (e @ self.contagion().T)
        add_f = km * e.mean(1, keepdim=True)
        vn = torch.cat([vn[:, :K] + add_f, vn[:, K:] + add_i], 1).clamp(V_MIN, V_MAX)
        return logp, vn, Bs, ds

    @torch.no_grad()
    def cascade_info(self) -> dict:
        ks, kg, km = self.kappas()
        rho = torch.sigmoid(self.rho_raw)
        A = self.contagion()
        ent = float(-(A.clamp(min=1e-12).log() * A).sum(1).mean())
        return dict(kappa_self=round(float(ks), 4), kappa_graph=round(float(kg), 4),
                    kappa_mkt=[round(float(v), 4) for v in km], rho_idio=round(float(rho[-1]), 4),
                    contagion_entropy=round(ent, 4),
                    # log-variance added in total by one tail event of one asset, summed over the decay
                    excitation_idio=round(float((ks + kg) / (1 - rho[-1])), 4))


class CGNSDE3(CGNSDE2):
    NET = Net3
    DYN_PARAMS = CGNSDE2.DYN_PARAMS + ("kappa_s_raw", "kappa_g_raw", "kappa_m_raw")
    name = "cgnsde3"
    VARIANTS = {
        **CGNSDE2.VARIANTS,
        "no_cascade": dict(cascade_self=False, cascade_graph=False, cascade_mkt=False),
        "no_graph_cascade": dict(cascade_graph=False),
        "no_mkt_cascade": dict(cascade_mkt=False),
        "uniform_graph": dict(contagion_graph="uniform"),
        # no crisis labels anywhere in training: one shared prior, no SupCon, unstratified batches
        "no_regime": dict(use_regime=False, lam_con=0.0, min_crisis=0),
    }

    def __init__(self, **hparams):
        if "epochs" in hparams:
            hparams["max_epochs"] = hparams.pop("epochs")
        unknown = set(hparams) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown CG-NSDE v3 hyperparameters: {sorted(unknown)}")
        hp = {**DEFAULTS, **hparams}
        hp["gmm_components"] = [int(k) for k in hp["gmm_components"]]
        CGNSDE2.__bases__[0].__init__(self, **hp)
        self.net = None

    def fit(self, data, seed: int) -> dict:
        info = super().fit(data if self.hparams["use_regime"] else _NoLabels(data), seed)
        info["use_regime"] = bool(self.hparams["use_regime"])
        return info

    @torch.no_grad()
    def _fit_mixtures(self, A, y, seed, gen, chunk: int = 512) -> None:
        if self.hparams["use_regime"]:
            return super()._fit_mixtures(A, y, seed, gen, chunk)
        y1 = y.copy()
        y1[: max(2 * max(self.hparams["gmm_components"]), 10)] = 1     # placeholder so the parent can run
        super()._fit_mixtures(A, y1, seed, gen, chunk)
        from sklearn.mixture import GaussianMixture   # refit the single shared mixture on ALL windows
        lat = []
        for s in range(0, len(A["x"]), chunk):
            g, _ = self.net.graph(A["feats"][s:s + chunk], A["adj"][s:s + chunk])
            mu, logvar = self.net.encode(A["x"][s:s + chunk], g)
            lat.append(mu + torch.randn(mu.shape, generator=gen) * torch.exp(0.5 * logvar))
        X = torch.cat(lat).double().numpy()
        fits = [GaussianMixture(c, covariance_type="diag", random_state=seed, reg_covar=1e-4).fit(X)
                for c in self.hparams["gmm_components"]]
        gm = min(fits, key=lambda f: f.bic(X))
        m = dict(w=gm.weights_.copy(), mean=gm.means_.copy(), var=gm.covariances_.copy(), bic=float(gm.bic(X)))
        self.mix = {0: m, 1: m}

    @torch.no_grad()
    def contagion_matrix(self) -> np.ndarray:
        self._require_fit()
        return self.net.contagion().numpy()
