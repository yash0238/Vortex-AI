"""CG-NSDE v2: Contrastive Graph-Neural SDE with factor-volatility states.

WHY A SECOND VERSION (post hoc, disclosed in the paper)
  The preregistered v1 (`cgnsde.py`) was run first and failed: its validation likelihood was best after
  2 epochs (it memorises the heavily overlapping training windows through the weakly regularised latent
  z0 and the unregularised graph embedding g that both feed the decoder), and its volatility state scales
  the WHOLE covariance by one number, which cannot change correlations.  v2 was designed after seeing
  those v1 results.  Its hyperparameters were chosen on the 2010-2019 fit period only; the benchmark
  protocol, metrics and label rule are unchanged, and v1 results stay in the tables.

Model (per-asset z-score space; scaler from TRAINING days only)
  graph      as v1: k-NN |correlation| graph per window, 2-layer GAT -> mean-pooled g.
  posterior  q(z0 | x) = N(mu, diag): MLP on [GRU(x), g].  The graph embedding enters ONLY here, so every
             bit of window information the decoder gets passes through the KL term.
  prior      p(z0 | regime) learned diagonal Gaussian per regime.  KL is SUMMED over latent dimensions and
             divided by T (per-day ELBO scale); beta = 1 after the ramp.
  latent     dz = f(z, t) dt + sigma(z, t) dW on [0, 1], Euler, T steps (prior dynamics, as v1).
  emission   r_t | z_t, v_t: mean m_t, covariance  B_t diag(exp v^f_t) B_t' + diag(d_t exp v^i_t),
             multivariate Student-t (one chi-square mixing variable per day) or Gaussian.
  vol state  v_t = (v^f_t in R^K, v^i_t in R): log-variance of each factor and of idiosyncratic noise.
             EGARCH-type recursion on STANDARDISED shocks (stable by construction):
               fhat_t  = posterior mean of the standardised factors given r_t under the day-t covariance
               c_k     = Var(fhat_k) = 1 - [(I + Bs' Ds^-1 Bs)^-1]_kk
               v^f_{t+1,k} = clamp(rho_k v^f_{t,k} + alpha_k (fhat_k^2 / c_k - 1) + theta_k fhat_k / sqrt c_k)
               v^i_{t+1}   = clamp(rho_i v^i_t   + alpha_i (mean_i resid_i^2 / ds_i / c_i - 1))
             v_1 = 2 tanh(W z0).  Teacher-forced on real returns in training, self-fed in sampling.
             When factor variances rise relative to idiosyncratic variance, correlations rise: the crisis
             correlation boost can be produced by the dynamics instead of being memorised.
  loss       NLL + beta KL / T + lam_con SupCon + lam_graph L_graph (+ optional lam_corr L_corr).
  sampling   unconditional: regime ~ Bernoulli(training crisis frequency), z0 from a per-regime diagonal
             Gaussian mixture fitted to posterior samples of z0 (components by BIC), SDE path, sequential
             emission.  No real window, graph or label is used.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchsde
from torch import nn

from .. import protocol as P
from .base import FitData, Generator
from .cgnsde import (N_FEATS, _Increments, _SDEFunc, corr_loss, emission_draw, graph_loss, node_features,
                     pooled_corr, ramp, sector_feature, stratified_batches, supcon_loss, window_graphs)

SHOCK_CAP = 20.0
V_MIN, V_MAX = -3.0, 3.5

DEFAULTS = dict(
    latent_dim=16, graph_dim=32, gat_hidden=32, gat_heads=4, knn_k=5, enc_hidden=64, sde_hidden=64, proj_dim=16,
    rank_K=8, emission="mt", sde_sigma_max=0.5, vol_mode="factor", leverage=True, use_gat=True, use_sde=True,
    beta_max=1.0, beta_ramp=5, lam_corr=0.0, corr_ramp=5, lam_con=0.1, con_warmup=5, tau=0.07, lam_graph=0.1,
    batch=32, min_crisis=6, lr=1e-3, lr_dyn=2e-2, weight_decay=1e-5, max_epochs=40, patience=8, grad_clip=1.0,
    gmm_components=(1, 2, 3, 5), innovations="fhs", verbose=False,
)
# innovations = "mt": parametric multivariate Student-t draws.  "fhs": filtered historical simulation, i.e.
# whole cross-sections of the model's own standardised training residuals eta = Sigma_t^(-1/2) r_t (symmetric
# root, last day of each training window) are resampled and re-coloured with the simulated Sigma_t^(1/2).


def sym_root(Bs, ds, inverse: bool = False):
    """Symmetric square root (or inverse root) of Bs Bs' + diag(ds), batched."""
    S = Bs @ Bs.transpose(-1, -2) + torch.diag_embed(ds)
    ev, U = torch.linalg.eigh(S.double())
    p = ev.clamp(min=1e-10).pow(-0.5 if inverse else 0.5)
    return ((U * p.unsqueeze(-2)) @ U.transpose(-1, -2)).to(Bs.dtype)


class Net2(nn.Module):
    def __init__(self, n_assets: int, hp: dict, T: int = P.T, init=None):
        super().__init__()
        from torch_geometric.nn import GATConv
        L, G, H, He = hp["latent_dim"], hp["graph_dim"], hp["gat_hidden"], hp["enc_hidden"]
        self.N, self.T, self.L, self.G, self.K = n_assets, T, L, G, int(hp["rank_K"])
        if self.K < 1:
            raise ValueError("v2 needs rank_K >= 1")
        if hp["vol_mode"] not in ("factor", "scalar", "none"):
            raise ValueError(f"unknown vol_mode {hp['vol_mode']!r}")
        if hp["emission"] not in ("mt", "gauss"):
            raise ValueError(f"unknown emission {hp['emission']!r}")
        self.sigma_max = float(hp["sde_sigma_max"])
        self.use_gat, self.use_sde = bool(hp["use_gat"]), bool(hp["use_sde"])
        self.vol_mode, self.leverage, self.student = hp["vol_mode"], bool(hp["leverage"]), hp["emission"] == "mt"
        if self.use_gat:
            self.gat1 = GATConv(N_FEATS, H, heads=hp["gat_heads"], concat=True, add_self_loops=False)
            self.gat2 = GATConv(H * hp["gat_heads"], G, heads=1, concat=False, add_self_loops=False)
            self.norm1, self.norm2 = nn.LayerNorm(H * hp["gat_heads"]), nn.LayerNorm(G)
        else:
            self.node_mlp = nn.Sequential(nn.Linear(N_FEATS, H), nn.ELU(), nn.LayerNorm(H), nn.Linear(H, G))
        self.enc = nn.GRU(n_assets, He, batch_first=True)
        self.post = nn.Sequential(nn.Linear(He + G, He), nn.ELU(), nn.Linear(He, 2 * L))
        self.prior_mu = nn.Parameter(torch.zeros(2, L))
        self.prior_logvar = nn.Parameter(torch.zeros(2, L))
        if self.use_sde:
            Hs = hp["sde_hidden"]
            self.drift = nn.Sequential(nn.Linear(L + 1, Hs), nn.Tanh(), nn.Linear(Hs, L))
            self.diff = nn.Sequential(nn.Linear(L + 1, Hs), nn.Tanh(), nn.Linear(Hs, L))
        self.proj = nn.Sequential(nn.Linear(L, L), nn.ReLU(), nn.Linear(L, hp["proj_dim"]))
        # Static factor loadings and idiosyncratic variances (PCA start, see CGNSDE2.fit); the latent path
        # only shifts the K factor log-variances and the common idiosyncratic log-variance.
        if init is None:
            B0 = 0.15 * torch.randn(n_assets, self.K)
            B0[:, 0] = 0.5
            d0 = torch.full((n_assets,), 0.7)
        else:
            B0, d0 = init
        self.B = nn.Parameter(B0.clone().float())
        self.d_raw = nn.Parameter(torch.log(torch.expm1(d0.clone().float().clamp(min=0.05))))
        self.mod = nn.Linear(L, self.K + 1)
        nn.init.normal_(self.mod.weight, std=0.02)
        nn.init.zeros_(self.mod.bias)
        self.v0 = nn.Linear(L, self.K + 1)
        nn.init.normal_(self.v0.weight, std=0.02)
        nn.init.zeros_(self.v0.bias)
        self.rho_raw = nn.Parameter(torch.full((self.K + 1,), 2.944))      # 0.95
        self.alpha_raw = nn.Parameter(torch.full((self.K + 1,), -3.49))    # softplus -> 0.03
        self.theta = nn.Parameter(torch.zeros(self.K))
        if self.student:
            self.raw_nu = nn.Parameter(torch.tensor(3.47))                  # nu = 4.5 + softplus ~ 8

    # graph (same convention as v1) ---------------------------------------------
    def graph(self, feats, adj):
        Bn, N, _ = feats.shape
        if not self.use_gat:
            return self.node_mlp(feats).mean(1), None
        b, i, j = adj.nonzero(as_tuple=True)
        ei = torch.stack([b * N + j, b * N + i])
        h = F.elu(self.norm1(self.gat1(feats.reshape(Bn * N, -1), ei)))
        h, (ei2, alpha) = self.gat2(h, ei, return_attention_weights=True)
        h = F.elu(self.norm2(h))
        att = feats.new_zeros(Bn, N, N)
        att[ei2[1] // N, ei2[1] % N, ei2[0] % N] = alpha.squeeze(-1)
        return h.view(Bn, N, -1).mean(1), att

    def encode(self, x, g):
        h = self.enc(x)[1].squeeze(0)
        mu, logvar = self.post(torch.cat([h, g], 1)).chunk(2, dim=1)
        return mu, logvar.clamp(-10.0, 3.0)

    def path(self, z0, dW):
        if not self.use_sde:
            return z0.unsqueeze(1).expand(-1, self.T, -1)
        ts = torch.arange(self.T + 1, dtype=z0.dtype, device=z0.device)
        ctx = z0.new_zeros(len(z0), 0)
        zs = torchsde.sdeint(_SDEFunc(self, ctx), z0, ts, bm=_Increments(dW), method="euler", dt=1.0)
        return zs[1:].permute(1, 0, 2)

    def nu(self):
        return 4.5 + F.softplus(self.raw_nu) if self.student else None

    def heads(self, zs):
        """Latent shifts of the log-variances, (..., K+1): K factors, then the common idiosyncratic shift."""
        return 1.5 * torch.tanh(self.mod(zs))

    def v_init(self, z0):
        """(b, K+N): factor states, then one idiosyncratic state per asset (common start value)."""
        v = 2.0 * torch.tanh(self.v0(z0))
        if self.vol_mode == "scalar":
            return v[:, :1].expand(-1, self.K + self.N)
        return torch.cat([v[:, :self.K], v[:, self.K:].expand(-1, self.N)], 1)

    def scales(self, a, v):
        K = self.K
        Bs = self.B.unsqueeze(0) * torch.exp(0.5 * (v[:, None, :K] + a[:, None, :K]))
        ds = (F.softplus(self.d_raw) + 0.01).unsqueeze(0) * torch.exp(v[:, K:] + a[:, K:])
        return Bs, ds

    def step(self, x, a, v, nu):
        """One day.  x (b, N); a (b, K+1); v (b, K+N).  Returns (log p(x), next v, Bs, ds)."""
        N, K = self.N, self.K
        Bs, ds = self.scales(a, v)
        Bd = Bs / ds.unsqueeze(-1)
        M = torch.eye(K, dtype=x.dtype, device=x.device) + Bs.transpose(1, 2) @ Bd
        Lc = torch.linalg.cholesky(M)
        e = x
        u = Bd.transpose(1, 2) @ e.unsqueeze(-1)
        Gu = torch.cholesky_solve(u, Lc)
        quad = ((e * e / ds).sum(-1) - (u * Gu).sum((-1, -2))).clamp(min=0.0)
        logdet = torch.log(ds).sum(-1) + 2.0 * torch.log(torch.diagonal(Lc, dim1=-2, dim2=-1)).sum(-1)
        if nu is None:
            logp = -0.5 * (N * math.log(2 * math.pi) + logdet + quad)
        else:
            c0 = (nu - 2.0) / nu
            logp = (torch.lgamma((nu + N) / 2) - torch.lgamma(nu / 2) - 0.5 * N * torch.log(nu * math.pi)
                    - 0.5 * (logdet + N * torch.log(c0)) - (nu + N) / 2 * torch.log1p(quad / (c0 * nu)))
        if self.vol_mode == "none":
            return logp, v, Bs, ds
        rho, alpha = torch.sigmoid(self.rho_raw), F.softplus(self.alpha_raw)
        if self.vol_mode == "scalar":
            s = (quad / N).clamp(max=SHOCK_CAP) - 1.0
            vn = (rho[0] * v[:, 0] + alpha[0] * s).clamp(V_MIN, V_MAX)
            return logp, vn.unsqueeze(1).expand(-1, K + N), Bs, ds
        fh = Gu.squeeze(-1)
        Ginv = torch.cholesky_inverse(Lc)                                   # (I + Bs' Ds^-1 Bs)^-1
        c = (1.0 - Ginv.diagonal(dim1=-2, dim2=-1)).clamp(min=1e-4)
        res = e - (Bs @ Gu).squeeze(-1)                                     # Cov(res) = Ds - Bs Ginv Bs'
        ci = (1.0 - ((Bs @ Ginv) * Bs).sum(-1) / ds).clamp(min=1e-3)
        sf = (fh * fh / c).clamp(max=SHOCK_CAP) - 1.0
        si = (res * res / (ds * ci)).clamp(max=SHOCK_CAP) - 1.0
        vf = rho[:K] * v[:, :K] + alpha[:K] * sf
        if self.leverage:
            vf = vf + self.theta * (fh / c.sqrt()).clamp(-5.0, 5.0)
        vi = rho[K] * v[:, K:] + alpha[K] * si
        return logp, torch.cat([vf, vi], 1).clamp(V_MIN, V_MAX), Bs, ds

    def teacher(self, x, zs, z0):
        """Teacher-forced pass: (log p (b, T), Bs (b, T, N, K), ds (b, T, N), mean = 0)."""
        a = self.heads(zs)
        v, nu = self.v_init(z0), self.nu()
        lp, Bl, dl = [], [], []
        for t in range(self.T):
            logp, v, Bs, ds = self.step(x[:, t], a[:, t], v, nu)
            lp.append(logp), Bl.append(Bs), dl.append(ds)
        return torch.stack(lp, 1), torch.stack(Bl, 1), torch.stack(dl, 1), torch.zeros_like(x)

    @torch.no_grad()
    def sample_seq(self, zs, z0, eN, eK, w, eta=None):
        """eta (n, T, N): resampled standardised residuals (FHS); None -> parametric draws."""
        a = self.heads(zs)
        v, nu = self.v_init(z0), self.nu()
        one, zero = torch.ones(len(z0)), torch.zeros(len(z0), self.N)
        out = []
        for t in range(self.T):
            Bs, ds = self.scales(a[:, t], v)
            if eta is not None:
                r = (sym_root(Bs, ds) @ eta[:, t].unsqueeze(-1)).squeeze(-1)
            else:
                r = emission_draw(zero, ds, Bs, one, nu, eN[:, t], eK[:, t], w[:, t])
            out.append(r)
            _, v, _, _ = self.step(r, a[:, t], v, nu)
        return torch.stack(out, 1)


def kl_sum(mu_q, lv_q, mu_p, lv_p):
    """KL(q || p) of diagonal Gaussians, summed over dimensions, mean over the batch."""
    return 0.5 * (lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p) ** 2) / lv_p.exp() - 1.0).sum(1).mean()


def loss_terms(net: Net2, batch: dict, corr_target, tau: float, gen: torch.Generator, want=(), sample_z0=True) -> dict:
    x, y = batch["x"], batch["y"]
    g, att = net.graph(batch["feats"], batch["adj"])
    mu, logvar = net.encode(x, g)
    z0 = mu + torch.randn(mu.shape, generator=gen) * torch.exp(0.5 * logvar) if sample_z0 else mu
    zs = net.path(z0, torch.randn(net.T, len(x), net.L, generator=gen))
    lp, Bs, ds, m = net.teacher(x, zs, z0)
    out = dict(nll=-lp.mean(), kl=kl_sum(mu, logvar, net.prior_mu[y], net.prior_logvar[y]) / net.T)
    if "corr" in want:
        nu = net.nu()
        n = len(x)
        w = torch.distributions.Chi2(nu.detach()).sample((n, net.T, 1)) if nu is not None else None
        r = emission_draw(m, ds, Bs, torch.ones(n, net.T), None if nu is None else nu.detach(),
                          torch.randn((n, net.T, net.N), generator=gen), torch.randn((n, net.T, net.K), generator=gen), w)
        out["corr"] = corr_loss(r, y, corr_target)
    if "supcon" in want:
        out["supcon"] = supcon_loss(net.proj(zs.mean(1)), y, tau)
    if "graph" in want and att is not None:
        out["graph"] = graph_loss(att, batch["tgt"])
    return out


class CGNSDE2(Generator):
    NET = Net2
    DYN_PARAMS = ("rho_raw", "alpha_raw", "theta", "raw_nu")
    name = "cgnsde2"
    family = "neural-ours"
    impl_source = "own"
    supports_regime = True
    VARIANTS = {
        "no_gat": dict(use_gat=False), "no_supcon": dict(lam_con=0.0), "no_graph_loss": dict(lam_graph=0.0),
        "no_sde": dict(use_sde=False), "scalar_vol": dict(vol_mode="scalar"), "no_voldyn": dict(vol_mode="none"),
        "no_leverage": dict(leverage=False), "gauss": dict(emission="gauss"), "param_innov": dict(innovations="mt"), "beta01": dict(beta_max=0.1),
        "K1": dict(rank_K=1), "K4": dict(rank_K=4), "K16": dict(rank_K=16),
    }

    def __init__(self, **hparams):
        if "epochs" in hparams:
            hparams["max_epochs"] = hparams.pop("epochs")
        unknown = set(hparams) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown CG-NSDE v2 hyperparameters: {sorted(unknown)}")
        hp = {**DEFAULTS, **hparams}
        hp["gmm_components"] = [int(k) for k in hp["gmm_components"]]
        super().__init__(**hp)
        self.net: Net2 | None = None

    def _arrays(self, Z, W_raw, vol_mean) -> dict:
        adj, tgt = window_graphs(Z, self.hparams["knn_k"])
        return dict(x=torch.from_numpy(Z.astype(np.float32)), adj=torch.from_numpy(adj), tgt=torch.from_numpy(tgt),
                    feats=torch.from_numpy(node_features(Z, W_raw, vol_mean, self.sector)))

    def fit(self, data: FitData, seed: int) -> dict:
        hp = self.hparams
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        gen = torch.Generator().manual_seed(seed)
        T, N = P.T, data.n_assets
        self.tickers, self.sector = list(data.tickers), sector_feature(list(data.tickers))
        train_days = ~data.val_mask
        self.mu, self.sd = data.returns[train_days].mean(0), data.returns[train_days].std(0)
        if not (self.sd > 0).all():
            raise ValueError("an asset has zero variance on the training days")
        W = data.windows()
        tr, va = data.split_windows()
        y_all = data.window_labels().astype(np.int64)
        if len(va) == 0:
            raise ValueError("no validation window: early stopping is impossible")
        vol_mean = None
        self.vol_mu = self.vol_sd = None
        if data.volume is not None and np.any(np.nan_to_num(data.volume) > 0):
            lv = np.log1p(np.clip(np.nan_to_num(data.volume), 0.0, None))
            self.vol_mu, self.vol_sd = lv[train_days].mean(0), lv[train_days].std(0)
            lvz = np.where(self.vol_sd > 0, (lv - self.vol_mu) / np.where(self.vol_sd > 0, self.vol_sd, 1.0), 0.0)
            c = np.concatenate([np.zeros((1, N)), np.cumsum(lvz, 0)])
            vol_mean = (c[T:] - c[:-T]) / T
        Z = (W - self.mu) / self.sd
        sub = lambda idx: self._arrays(Z[idx], W[idx], None if vol_mean is None else vol_mean[idx])
        A_tr, A_va = sub(tr), sub(va)
        y_tr = y_all[tr]
        A_tr["y"], A_va["y"] = torch.from_numpy(y_tr), torch.from_numpy(y_all[va])
        n_cr = int(y_tr.sum())
        if n_cr < hp["min_crisis"] or len(y_tr) - n_cr < hp["batch"] - hp["min_crisis"]:
            raise ValueError(f"cannot build stratified batches: {n_cr} crisis / {len(y_tr) - n_cr} normal training windows")
        self.p_crisis = float(y_tr.mean())
        corr_target = torch.stack([pooled_corr(A_tr["x"][A_tr["y"] == k].double()).float() for k in (0, 1)])

        # PCA start for loadings / idiosyncratic variances (z-scored TRAINING days only).
        Ztr = (data.returns[train_days] - self.mu) / self.sd
        ev, evec = np.linalg.eigh(np.corrcoef(Ztr.T))
        K = int(hp["rank_K"])
        B0 = evec[:, ::-1][:, :K] * np.sqrt(ev[::-1][:K])
        B0 = B0 * np.sign(B0.sum(0, keepdims=True))
        init = (torch.from_numpy(B0.copy()), torch.from_numpy(1.0 - (B0 ** 2).sum(1)))
        self.net = net = self.NET(N, hp, init=init)
        if hasattr(net, "set_cascade"):
            net.set_cascade(Ztr)
        # The few scalar dynamics parameters (persistence, shock impact, leverage, tail index) sit behind
        # sigmoid / softplus maps with small slopes; at the network learning rate they hardly move before the
        # static parts overfit, so they get their own, larger rate and no weight decay.
        dyn = [p for n_, p in net.named_parameters() if n_ in self.DYN_PARAMS]
        rest = [p for n_, p in net.named_parameters() if n_ not in self.DYN_PARAMS]
        opt = torch.optim.Adam([dict(params=rest, lr=hp["lr"], weight_decay=hp["weight_decay"]),
                                dict(params=dyn, lr=hp["lr_dyn"], weight_decay=0.0)])
        take = lambda A, idx: {k: v[idx] for k, v in A.items()}
        active = [k for k, lam in (("corr", hp["lam_corr"]), ("supcon", hp["lam_con"]), ("graph", hp["lam_graph"]))
                  if lam > 0]
        history, best, best_state, bad = [], (math.inf, -1), None, 0
        for epoch in range(int(hp["max_epochs"])):
            t0 = time.time()
            w = dict(kl=hp["beta_max"] * ramp(epoch, hp["beta_ramp"]), corr=hp["lam_corr"] * ramp(epoch, hp["corr_ramp"]),
                     supcon=hp["lam_con"] * ramp(epoch, hp["con_warmup"]), graph=hp["lam_graph"])
            net.train()
            acc, nb = {}, 0
            for idx in stratified_batches(y_tr, hp["batch"], hp["min_crisis"], rng):
                terms = loss_terms(net, take(A_tr, torch.from_numpy(idx)), corr_target, hp["tau"], gen, want=active)
                loss = terms["nll"] + sum(w[k] * v for k, v in terms.items() if k != "nll")
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at epoch {epoch}: "
                                             f"{ {k: float(v) for k, v in terms.items()} }")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), hp["grad_clip"])
                opt.step()
                for k, v in {**terms, "total": loss}.items():
                    acc[k] = acc.get(k, 0.0) + float(v)
                nb += 1
            val_nll, val_kl = self._val(A_va, seed)
            val = val_nll + hp["beta_max"] * val_kl          # negative per-day ELBO at the final beta
            if not math.isfinite(val):
                raise FloatingPointError(f"non-finite validation loss at epoch {epoch}")
            rec = dict(epoch=epoch + 1, **{f"train_{k}": v / nb for k, v in acc.items()}, val_nll=val_nll,
                       val_kl=val_kl, val=val, beta=w["kl"], seconds=time.time() - t0)
            history.append(rec)
            if hp["verbose"]:
                print("  ".join(f"{k}={v:.4g}" for k, v in rec.items()), flush=True)
            if epoch + 1 >= hp["beta_ramp"] and val < best[0]:      # compare only at the final beta
                best, bad = (val, epoch + 1), 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            elif epoch + 1 >= hp["beta_ramp"]:
                bad += 1
                if bad >= hp["patience"]:
                    break
        if best_state is None:
            best, best_state = (val, len(history)), {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state)
        net.eval()
        self._fit_mixtures(A_tr, y_tr, seed, gen)
        K = net.K
        self.info = dict(
            model_version=2, epochs_run=len(history), best_epoch=best[1], best_val=best[0], history=history,
            nu=float(net.nu()) if net.student else None,
            rho=[round(float(v), 4) for v in torch.sigmoid(net.rho_raw)],
            alpha=[round(float(v), 4) for v in F.softplus(net.alpha_raw)],
            theta=[round(float(v), 4) for v in net.theta[:K]],
            mixture_components={"normal": len(self.mix[0]["w"]), "crisis": len(self.mix[1]["w"])},
            n_train_windows=int(len(tr)), n_val_windows=int(len(va)), n_train_crisis=n_cr,
            train_crisis_freq=self.p_crisis, batches_per_epoch=nb, n_params=sum(p.numel() for p in net.parameters()),
            sec_per_epoch=float(np.mean([h["seconds"] for h in history])), seed=int(seed))
        if hasattr(net, "cascade_info"):
            self.info.update(net.cascade_info())
        return self.info

    @torch.no_grad()
    def _val(self, A: dict, seed: int, chunk: int = 256) -> tuple[float, float]:
        net = self.net
        net.eval()
        gen = torch.Generator().manual_seed(1_000_003 + seed)
        nll = kl = 0.0
        for s in range(0, len(A["x"]), chunk):
            b = {k: v[s:s + chunk] for k, v in A.items()}
            t = loss_terms(net, b, None, 0.0, gen, want=(), sample_z0=False)
            nll += float(t["nll"]) * len(b["x"])
            kl += float(t["kl"]) * len(b["x"])
        return nll / len(A["x"]), kl / len(A["x"])

    @torch.no_grad()
    def _fit_mixtures(self, A: dict, y: np.ndarray, seed: int, gen: torch.Generator, chunk: int = 512) -> None:
        from sklearn.mixture import GaussianMixture
        lat = []
        for s in range(0, len(A["x"]), chunk):
            g, _ = self.net.graph(A["feats"][s:s + chunk], A["adj"][s:s + chunk])
            mu, logvar = self.net.encode(A["x"][s:s + chunk], g)
            lat.append(mu + torch.randn(mu.shape, generator=gen) * torch.exp(0.5 * logvar))
        # standardised residual of the LAST day of every training window (teacher-forced states)
        res = []
        net = self.net
        for s in range(0, len(A["x"]), chunk):
            x = A["x"][s:s + chunk]
            z0 = torch.cat(lat)[s:s + chunk]
            a = net.heads(net.path(z0, torch.randn(net.T, len(x), net.L, generator=gen)))
            v, nu = net.v_init(z0), net.nu()
            for t in range(net.T - 1):
                _, v, _, _ = net.step(x[:, t], a[:, t], v, nu)
            Bs, ds = net.scales(a[:, -1], v)
            res.append((sym_root(Bs, ds, inverse=True) @ x[:, -1].unsqueeze(-1)).squeeze(-1))
        self.eta = torch.cat(res).numpy().astype(np.float32)
        lat = torch.cat(lat).double().numpy()
        self.mix = {}
        for k in (0, 1):
            X = lat[y == k]
            fits = [GaussianMixture(c, covariance_type="diag", random_state=seed, reg_covar=1e-4).fit(X)
                    for c in self.hparams["gmm_components"] if len(X) >= 2 * c]
            if not fits:
                raise ValueError(f"regime {k}: {len(X)} training windows are too few for any mixture")
            gm = min(fits, key=lambda f: f.bic(X))
            self.mix[k] = dict(w=gm.weights_.copy(), mean=gm.means_.copy(), var=gm.covariances_.copy(),
                               bic=float(gm.bic(X)))

    # --------------------------------------------------------------- sampling
    def _require_fit(self):
        if self.net is None or not hasattr(self, "mix"):
            raise RuntimeError("CG-NSDE v2 is not fitted")

    @torch.no_grad()
    def _generate(self, regimes: np.ndarray, rng: np.random.Generator, chunk: int = 1000) -> np.ndarray:
        self._require_fit()
        net = self.net
        net.eval()
        f32 = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))
        nu = float(net.nu()) if net.student else None
        out = []
        for s in range(0, len(regimes), chunk):
            reg = regimes[s:s + chunk]
            n = len(reg)
            lat = np.empty((n, net.L))
            for k in (0, 1):
                sel = np.where(reg == k)[0]
                mx = self.mix[k]
                comp = rng.choice(len(mx["w"]), size=len(sel), p=mx["w"] / mx["w"].sum())
                lat[sel] = mx["mean"][comp] + np.sqrt(mx["var"][comp]) * rng.standard_normal((len(sel), net.L))
            z0 = f32(lat)
            zs = net.path(z0, f32(rng.standard_normal((net.T, n, net.L))))
            w = f32(rng.chisquare(nu, (n, net.T, 1))) if nu is not None else torch.ones(n, net.T, 1)
            eta = None
            if self.hparams["innovations"] == "fhs":
                eta = torch.from_numpy(self.eta[rng.integers(0, len(self.eta), (n, net.T))])
            z = net.sample_seq(zs, z0, f32(rng.standard_normal((n, net.T, net.N))),
                               f32(rng.standard_normal((n, net.T, net.K))), w, eta)
            out.append(z.double().numpy() * self.sd + self.mu)
        G = np.concatenate(out)
        if not np.isfinite(G).all():
            raise FloatingPointError("CG-NSDE v2 produced non-finite returns")
        return G

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        self._require_fit()
        return self._generate((rng.random(n) < self.p_crisis).astype(np.int64), rng)

    def sample_regime(self, n: int, regime: int, rng: np.random.Generator) -> np.ndarray:
        if regime not in (0, 1):
            raise ValueError("regime must be 0 (normal) or 1 (crisis)")
        return self._generate(np.full(n, regime, dtype=np.int64), rng)

    # --------------------------------------------------------- introspection
    def _prep(self, windows_raw, volume_windows=None) -> dict:
        W = np.asarray(windows_raw, dtype=np.float64)
        if W.ndim != 3 or W.shape[1:] != (P.T, len(self.mu)):
            raise ValueError(f"expected windows of shape (n, {P.T}, {len(self.mu)}), got {W.shape}")
        vol_mean = None
        if volume_windows is not None and self.vol_mu is not None:
            lv = np.log1p(np.clip(np.nan_to_num(np.asarray(volume_windows, dtype=np.float64)), 0.0, None))
            vol_mean = np.where(self.vol_sd > 0, (lv - self.vol_mu) / np.where(self.vol_sd > 0, self.vol_sd, 1.0),
                                0.0).mean(1)
        return self._arrays((W - self.mu) / self.sd, W, vol_mean)

    @torch.no_grad()
    def encode(self, windows_raw, volume_windows=None) -> dict:
        """z0 = posterior mean, g = graph embedding, proj = normalised contrastive projection,
        vol = teacher-forced log-variance states (n, T, K+1) of the given RAW windows."""
        self._require_fit()
        net = self.net
        A = self._prep(windows_raw, volume_windows)
        g, _ = net.graph(A["feats"], A["adj"])
        z0, _ = net.encode(A["x"], g)
        zs = net.path(z0, torch.zeros(net.T, len(z0), net.L))
        a = net.heads(zs)
        v, nu, vs = net.v_init(z0), net.nu(), []
        for t in range(net.T):
            vs.append(torch.cat([v[:, :net.K] + a[:, t, :net.K], v[:, net.K:].mean(1, keepdim=True) + a[:, t, net.K:]], 1))
            _, v, _, _ = net.step(A["x"][:, t], a[:, t], v, nu)
        proj = F.normalize(net.proj(zs.mean(1)), dim=1)
        return dict(z0=z0.numpy(), g=g.numpy(), proj=proj.numpy(), vol=torch.stack(vs, 1).numpy())

    @torch.no_grad()
    def attention(self, window_raw, volume_window=None) -> np.ndarray:
        self._require_fit()
        if not self.net.use_gat:
            raise NotImplementedError("this variant has no GAT, hence no attention matrix")
        A = self._prep(np.asarray(window_raw)[None], None if volume_window is None else np.asarray(volume_window)[None])
        return self.net.graph(A["feats"], A["adj"])[1][0].numpy()

    # ------------------------------------------------------------ persistence
    @staticmethod
    def _file(path) -> Path:
        path = Path(path)
        return path if path.suffix == ".pt" else path.with_name(path.name + ".pt")

    def save(self, path) -> None:
        self._require_fit()
        t = lambda a: None if a is None else torch.as_tensor(np.asarray(a))
        f = self._file(path)
        f.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(
            format=2, hparams=dict(self.hparams), tickers=list(self.tickers), name=self.name,
            state_dict=self.net.state_dict(), mu=t(self.mu), sd=t(self.sd), vol_mu=t(self.vol_mu),
            vol_sd=t(self.vol_sd), sector=t(self.sector), p_crisis=self.p_crisis, eta=t(self.eta),
            mix={k: {q: (t(v) if q != "bic" else v) for q, v in m.items()} for k, m in self.mix.items()},
            info=getattr(self, "info", {})), f)

    @classmethod
    def load(cls, path) -> "CGNSDE2":
        ck = torch.load(cls._file(path), map_location="cpu", weights_only=True)
        if ck.get("format") != 2:
            raise ValueError(f"unsupported CG-NSDE v2 checkpoint format: {ck.get('format')}")
        n = lambda a: None if a is None else a.numpy()
        self = cls(**ck["hparams"])
        self.name, self.tickers, self.p_crisis, self.info = ck["name"], ck["tickers"], ck["p_crisis"], ck["info"]
        self.mu, self.sd, self.vol_mu, self.vol_sd = n(ck["mu"]), n(ck["sd"]), n(ck["vol_mu"]), n(ck["vol_sd"])
        self.sector, self.eta = n(ck["sector"]), n(ck["eta"])
        self.mix = {int(k): {q: (n(v) if q != "bic" else v) for q, v in m.items()} for k, m in ck["mix"].items()}
        self.net = cls.NET(len(self.tickers), self.hparams)
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        return self
