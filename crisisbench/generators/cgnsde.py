"""CG-NSDE: Contrastive Graph-Neural SDE generator (the project's own model).

Model (all in per-asset z-score space; scaler from TRAINING days only)
  graph      per window: symmetric k-NN graph on |Pearson correlation| (+ self loops),
             6 node features; 2-layer GAT (4 heads, then 1 head) -> mean-pooled g.
  posterior  GRU over the window -> q(z0 | x) = N(mu, diag exp(logvar)).
  prior      p(z0 | regime) = learned diagonal Gaussian per regime (KL weight beta).
  latent     dz = f(z, g, t) dt + sigma(z, g, t) dW on [0, 1], torchsde Euler, T = 60
             fixed steps; day t (1..T) is emitted from the state after t steps.
  emission   r_t | z_t: mean m_t, covariance mult_t * (B_t B_t' + diag d_t); multivariate
             Student-t with ONE chi-square mixing variable per day ("mt") or Gaussian.
             rank_K = 0 gives a diagonal covariance (independent Student-t per asset).
  vol state  v_1 = 0, v_t = clamp(rho v_{t-1} + (1 - rho)(mean_i e_{t-1,i}^2 - 1), -1, 2),
             mult_t = exp(2 kappa v_t), e = (r - m) / base marginal std.  Teacher-forced on
             real returns in training/validation, fed with its own samples in sampling.
  loss       NLL + beta KL + lam_corr L_corr + lam_con L_supcon + lam_graph L_graph.
  sampling   FULLY PARAMETRIC AND UNCONDITIONAL.  After training every training window is
             encoded to (posterior sample z0, g) and a diagonal Gaussian mixture is fitted
             to the joint [z0, g] per regime (components by BIC; ex-post density estimation,
             Ghosh et al. 2020).  sample() draws a regime with the training crisis frequency,
             draws [z0, g] from that regime's mixture, solves the SDE and samples the
             emission sequentially.  No real window, graph or label is used, and no training
             window is stored on the fitted object.

Fixes relative to the audited legacy code (models/, training/vortex_model.py)
  (a) SupCon and graph losses are part of the loss in NLL mode (they were dead).
  (b) sampling needs no real adjacency / node features (see above).
  (c) diffusion = sigma_max * sigmoid(net) with a LINEAR last layer (legacy applied
      sigmoid twice, so diffusion could not go below 0.5 sigma_max).
  (d) the GAT runs once per batch on offset edge indices, with no Python loop.
  (e) rank 0 works for the Student-t emission.
  (f) legacy multivariate-t log_prob treated Sigma as the SCALE matrix while its sampler
      used scale Sigma (nu-2)/nu; here both use scale = Sigma (nu-2)/nu, so Cov = Sigma.
  (g) legacy sampled via an N x N Cholesky; here x = B eps_K + sqrt(d) eps_N (same law).

Deviations / choices not fixed by the brief (also listed in the agent report)
  - This is not the path-space latent SDE of Li et al. (2020): only z0 has a posterior;
    the Brownian path is sampled from the prior dynamics during training, so the NLL term
    is a single-sample Jensen upper bound on -log p(x | z0, g).
  - KL is averaged over latent dimensions (legacy convention); with beta_max = 0.01 the
    model is a regularised autoencoder, which is why the ex-post mixture is used to sample.
  - Vol-state shock standardises by the base marginal std sqrt(d + sum_k B^2) (legacy used
    sqrt(d) only, which biases the shock upwards when rank_K > 0).
  - L_graph = mean over nodes of the squared L2 distance between the node's layer-2
    attention row and the row-normalised |correlation| over its graph neighbours
    (legacy: mean over all N^2 entries).  No GAT dropout (legacy 0.1): PyG returns
    attention after dropout, which would corrupt L_graph.
  - L_corr samples use the teacher-forced vol multiplier and a detached nu.
  - Batches hold exactly `min_crisis` crisis windows (crisis is oversampled relative to
    its ~15% frequency); an epoch is one pass over the normal training windows.
  - Validation NLL uses z0 = posterior mean and one FIXED Brownian draw (common random
    numbers across epochs) so the early-stopping criterion is deterministic.
  - use_gat=False: per-node MLP + mean pool; there is no attention, so L_graph is absent.
    use_sde=False: z_t = z0, so g does not reach the emission at all.
  - Outputs are not clipped: a Student-t draw can produce an implausibly large return.
"""
from __future__ import annotations

import importlib.util
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

ROOT = Path(__file__).resolve().parents[2]
N_FEATS = 6
BAND = 0.10

DEFAULTS = dict(
    latent_dim=64, graph_dim=64, gat_hidden=64, gat_heads=4, knn_k=5, sde_hidden=128, proj_dim=32,
    rank_K=8, emission="mt", sde_sigma_max=0.75, vol_dyn=True, use_gat=True, use_sde=True,
    beta_max=0.01, beta_ramp=10, lam_corr=100.0, corr_ramp=5, lam_con=0.1, con_warmup=5, tau=0.07,
    lam_graph=0.1, batch=32, min_crisis=6, lr=1e-3, max_epochs=40, patience=8, grad_clip=1.0,
    gmm_components=(1, 3, 5, 10), verbose=False,
)


# ------------------------------------------------------------------ preprocessing
def sector_feature(tickers: list[str]) -> np.ndarray:
    """sector id / number of sectors from the NSE map; 0 for tickers not in the map."""
    spec = importlib.util.spec_from_file_location("_vortex_node_features", ROOT / "data" / "node_features.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return np.array([mod.NIFTY50_SECTORS.get(t, 0) / mod.NUM_SECTORS for t in tickers], dtype=np.float32)


def window_graphs(Z: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Z (n, T, N) z-scored windows -> (adj bool (n, N, N) with self loops, row-normalised |corr| on adj)."""
    n, T, N = Z.shape
    Zc = Z - Z.mean(1, keepdims=True)
    cov = np.einsum("ntj,ntk->njk", Zc, Zc, optimize=True) / T
    sd = np.sqrt(np.einsum("njj->nj", cov))
    ok = sd > 1e-12                                     # a flat asset has no correlation: treated as 0
    sd = np.where(ok, sd, 1.0)
    A = np.abs(cov / sd[:, :, None] / sd[:, None, :]) * (ok[:, :, None] & ok[:, None, :])
    eye = np.eye(N, dtype=bool)
    A[:, eye] = -1.0
    kk = min(k, N - 1)
    nb = np.argpartition(-A, kk - 1, axis=2)[:, :, :kk]
    adj = np.zeros((n, N, N), dtype=bool)
    np.put_along_axis(adj, nb, True, axis=2)
    adj |= adj.transpose(0, 2, 1)
    adj[:, eye] = True
    A[:, eye] = 1.0
    tgt = A * adj
    return adj, (tgt / tgt.sum(2, keepdims=True)).astype(np.float32)


def node_features(Z: np.ndarray, W_raw: np.ndarray, vol_mean: np.ndarray | None, sector: np.ndarray) -> np.ndarray:
    """(n, N, 6): mean z, mean |z|, std z, mean z-scored log-volume, sector, max |raw r| / 10% band."""
    n, _, N = Z.shape
    f = np.zeros((n, N, N_FEATS), dtype=np.float32)
    f[..., 0], f[..., 1], f[..., 2] = Z.mean(1), np.abs(Z).mean(1), Z.std(1)
    if vol_mean is not None:
        f[..., 3] = vol_mean
    f[..., 4] = sector[None, :]
    f[..., 5] = np.abs(W_raw).max(1) / BAND
    return f


# ----------------------------------------------------------------------- emission
def emission_log_prob(x, m, d, B, mult, nu):
    """log density of x (..., N): mean m, covariance mult * (B B' + diag d).

    nu None -> Gaussian.  nu and B -> multivariate t with scale matrix Cov * (nu-2)/nu.
    nu and B None -> independent Student-t per asset with variance mult * d.
    """
    N = x.shape[-1]
    v = x - m
    if nu is not None and B is None:
        s2 = d * mult.unsqueeze(-1) * (nu - 2.0) / nu
        return (torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - 0.5 * torch.log(nu * math.pi * s2)
                - (nu + 1) / 2 * torch.log1p(v * v / (nu * s2))).sum(-1)
    quad = (v * v / d).sum(-1)
    logdet = torch.log(d).sum(-1)
    if B is not None:                                   # Woodbury on the K x K capacitance matrix
        Bd = B / d.unsqueeze(-1)
        M = torch.eye(B.shape[-1], dtype=x.dtype, device=x.device) + B.transpose(-1, -2) @ Bd
        Lc = torch.linalg.cholesky(M)
        u = (Bd.transpose(-1, -2) @ v.unsqueeze(-1))
        quad = quad - (u * torch.cholesky_solve(u, Lc)).sum((-1, -2))
        logdet = logdet + 2.0 * torch.log(torch.diagonal(Lc, dim1=-2, dim2=-1)).sum(-1)
    quad = quad.clamp(min=0.0) / mult
    logdet = logdet + N * torch.log(mult)
    if nu is None:
        return -0.5 * (N * math.log(2 * math.pi) + logdet + quad)
    c = (nu - 2.0) / nu
    return (torch.lgamma((nu + N) / 2) - torch.lgamma(nu / 2) - 0.5 * N * torch.log(nu * math.pi)
            - 0.5 * (logdet + N * torch.log(c)) - (nu + N) / 2 * torch.log1p(quad / (c * nu)))


def emission_draw(m, d, B, mult, nu, eN, eK, w):
    """Reparameterised draw.  eN (..., N), eK (..., K) standard normal; w chi-square(nu),
    shape (..., 1) (shared, multivariate t) or (..., N) (independent t); ignored if nu is None."""
    x = torch.sqrt(d) * eN
    if B is not None:
        x = x + (B @ eK.unsqueeze(-1)).squeeze(-1)
    x = x * torch.sqrt(mult).unsqueeze(-1)
    if nu is None:
        return m + x
    return m + torch.sqrt((nu - 2.0) / nu) * x / torch.sqrt(w / nu)


# ---------------------------------------------------------------------------- SDE
class _Increments(torchsde.BaseBrownian):
    """Brownian motion on the integer grid 0..T given by pre-drawn unit-variance increments (T, B, L)."""

    def __init__(self, dW):
        self.dW = dW

    def __call__(self, ta, tb=None, return_U=False, return_A=False):
        i = int(round(float(ta)))
        if tb is None or abs(float(tb) - float(ta) - 1.0) > 1e-6 or return_U or return_A:
            raise ValueError("increments are defined for unit steps on the integer grid only")
        return self.dW[i]

    def __repr__(self):
        return f"_Increments(shape={tuple(self.dW.shape)})"

    dtype = property(lambda self: self.dW.dtype)
    device = property(lambda self: self.dW.device)
    shape = property(lambda self: self.dW.shape[1:])
    levy_area_approximation = property(lambda self: "none")


class _SDEFunc(nn.Module):
    """Ito SDE in integer time s = t * T: drift f / T and diffusion sigma / sqrt(T) per unit step."""
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, net: "CGNSDENet", g: torch.Tensor):
        super().__init__()
        self.net, self.ctx, self.T = net, g, net.T

    def _inp(self, t, y):
        return torch.cat([y, self.ctx, (t / self.T).expand(y.shape[0], 1)], dim=-1)

    def f(self, t, y):
        return torch.tanh(self.net.drift(self._inp(t, y))) / self.T

    def g(self, t, y):
        return self.net.sigma_max * torch.sigmoid(self.net.diff(self._inp(t, y))) / math.sqrt(self.T)


# -------------------------------------------------------------------------- network
class CGNSDENet(nn.Module):
    def __init__(self, n_assets: int, hp: dict, T: int = P.T):
        super().__init__()
        from torch_geometric.nn import GATConv
        L, G, H = hp["latent_dim"], hp["graph_dim"], hp["gat_hidden"]
        self.N, self.T, self.L, self.G, self.K = n_assets, T, L, G, int(hp["rank_K"])
        self.sigma_max = float(hp["sde_sigma_max"])
        self.use_gat, self.use_sde, self.vol_dyn = bool(hp["use_gat"]), bool(hp["use_sde"]), bool(hp["vol_dyn"])
        if hp["emission"] not in ("mt", "gauss"):
            raise ValueError(f"unknown emission {hp['emission']!r}")
        self.student = hp["emission"] == "mt"
        if self.use_gat:
            self.gat1 = GATConv(N_FEATS, H, heads=hp["gat_heads"], concat=True, add_self_loops=False)
            self.gat2 = GATConv(H * hp["gat_heads"], G, heads=1, concat=False, add_self_loops=False)
            self.norm1, self.norm2 = nn.LayerNorm(H * hp["gat_heads"]), nn.LayerNorm(G)
        else:
            self.node_mlp = nn.Sequential(nn.Linear(N_FEATS, H), nn.ELU(), nn.LayerNorm(H), nn.Linear(H, G))
        self.enc = nn.GRU(n_assets, L, batch_first=True)
        self.to_mu, self.to_logvar = nn.Linear(L, L), nn.Linear(L, L)
        self.prior_mu = nn.Parameter(torch.zeros(2, L))
        self.prior_logvar = nn.Parameter(torch.zeros(2, L))
        if self.use_sde:
            Hs = hp["sde_hidden"]
            self.drift = nn.Sequential(nn.Linear(L + G + 1, Hs), nn.Tanh(), nn.Linear(Hs, Hs), nn.Tanh(),
                                       nn.Linear(Hs, L))
            self.diff = nn.Sequential(nn.Linear(L + G + 1, Hs), nn.Tanh(), nn.Linear(Hs, L))
        self.proj = nn.Sequential(nn.Linear(L, L), nn.ReLU(), nn.Linear(L, hp["proj_dim"]))
        self.emi_m, self.emi_d = nn.Linear(L, n_assets), nn.Linear(L, n_assets)
        nn.init.constant_(self.emi_d.bias, 0.54)        # softplus(0.54) ~ 1: unit variance in z-space
        if self.K > 0:
            self.emi_B = nn.Linear(L, n_assets * self.K)
            nn.init.normal_(self.emi_B.weight, std=0.05)
            nn.init.zeros_(self.emi_B.bias)
        if self.student:
            self.raw_nu = nn.Parameter(torch.tensor(3.47))      # nu = 4.5 + softplus(.) ~ 8 at init
        if self.vol_dyn:
            self.rho_raw = nn.Parameter(torch.tensor(2.197))    # rho 0.9
            self.kappa_raw = nn.Parameter(torch.tensor(-2.197))  # kappa 0.1

    # graph -------------------------------------------------------------------
    def graph(self, feats, adj):
        """feats (B, N, 6), adj bool (B, N, N) incl. self loops -> g (B, G), attention (B, N, N) or None.

        attention[b, i, j] = layer-2 weight of neighbour j in node i's update (rows sum to 1).
        """
        Bn, N, _ = feats.shape
        if not self.use_gat:
            return self.node_mlp(feats).mean(1), None
        b, i, j = adj.nonzero(as_tuple=True)
        ei = torch.stack([b * N + j, b * N + i])             # source j -> target i, offset per graph
        h = F.elu(self.norm1(self.gat1(feats.reshape(Bn * N, -1), ei)))
        h, (ei2, alpha) = self.gat2(h, ei, return_attention_weights=True)
        h = F.elu(self.norm2(h))
        att = feats.new_zeros(Bn, N, N)
        att[ei2[1] // N, ei2[1] % N, ei2[0] % N] = alpha.squeeze(-1)
        return h.view(Bn, N, -1).mean(1), att

    # latent ------------------------------------------------------------------
    def encode(self, x):
        h = self.enc(x)[1].squeeze(0)
        return self.to_mu(h), self.to_logvar(h).clamp(-12.0, 4.0)

    def path(self, z0, g, dW):
        """Latent states for days 1..T, (B, T, L).  dW: unit-variance increments (T, B, L)."""
        if not self.use_sde:
            return z0.unsqueeze(1).expand(-1, self.T, -1)
        ts = torch.arange(self.T + 1, dtype=z0.dtype, device=z0.device)
        zs = torchsde.sdeint(_SDEFunc(self, g), z0, ts, bm=_Increments(dW), method="euler", dt=1.0)
        return zs[1:].permute(1, 0, 2)

    # emission ----------------------------------------------------------------
    def nu(self):
        return 4.5 + F.softplus(self.raw_nu) if self.student else None

    def heads(self, zs):
        m = 0.5 * torch.tanh(self.emi_m(zs))
        d = F.softplus(self.emi_d(zs)) + 0.01
        B = 0.5 * torch.tanh(self.emi_B(zs)).view(*zs.shape[:-1], self.N, self.K) if self.K > 0 else None
        return m, d, B

    @staticmethod
    def base_var(d, B):
        return d if B is None else d + B.pow(2).sum(-1)

    def vol_step(self, v, r, m, s2):
        """Next vol state from today's return r, base mean m and base marginal variance s2."""
        rho = torch.sigmoid(self.rho_raw)
        return (rho * v + (1 - rho) * (((r - m) ** 2 / s2).mean(-1) - 1.0)).clamp(-1.0, 2.0)

    def vol_mult(self, v):
        return torch.exp(2.0 * torch.sigmoid(self.kappa_raw) * v)

    def teacher_mult(self, x, m, d, B):
        """Covariance multipliers (B, T) driven by the REAL returns x up to the previous day."""
        if not self.vol_dyn:
            return torch.ones_like(m[..., 0])
        s2 = self.base_var(d, B)
        v = torch.zeros_like(m[:, 0, 0])
        out = [self.vol_mult(v)]
        for t in range(1, m.shape[1]):
            v = self.vol_step(v, x[:, t - 1], m[:, t - 1], s2[:, t - 1])
            out.append(self.vol_mult(v))
        return torch.stack(out, 1)

    def noise_shapes(self, n):
        """Shapes of (eN, eK, w) for n paths."""
        wshape = (n, self.T, 1 if self.K > 0 else self.N)
        return (n, self.T, self.N), (n, self.T, max(self.K, 1)), wshape

    @torch.no_grad()
    def sample_seq(self, zs, eN, eK, w):
        """Sequential emission with the vol state fed by the model's own samples."""
        m, d, B = self.heads(zs)
        s2, nu = self.base_var(d, B), self.nu()
        v = torch.zeros_like(m[:, 0, 0])
        out = []
        for t in range(self.T):
            mult = self.vol_mult(v) if self.vol_dyn else torch.ones_like(v)
            r = emission_draw(m[:, t], d[:, t], None if B is None else B[:, t], mult, nu, eN[:, t], eK[:, t], w[:, t])
            out.append(r)
            if self.vol_dyn:
                v = self.vol_step(v, r, m[:, t], s2[:, t])
        return torch.stack(out, 1)


# ------------------------------------------------------------------------- losses
def gaussian_kl(mu_q, lv_q, mu_p, lv_p):
    """KL(N(mu_q, e^lv_q) || N(mu_p, e^lv_p)), mean over batch and latent dimension."""
    return 0.5 * (lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p) ** 2) / lv_p.exp() - 1.0).mean()


def supcon_loss(feats, y, tau):
    """Supervised contrastive loss L_out of Khosla et al. (2020), eq. 2; anchors need >= 1 positive."""
    u = F.normalize(feats, dim=1)
    eye = torch.eye(len(u), dtype=torch.bool, device=u.device)
    sim = (u @ u.T / tau).masked_fill(eye, float("-inf"))
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (y[:, None] == y[None, :]) & ~eye
    npos = pos.sum(1)
    if not (npos > 0).any():
        raise RuntimeError("SupCon: no anchor has a positive in this batch")
    keep = npos > 0
    return -(logp.masked_fill(~pos, 0.0).sum(1)[keep] / npos[keep]).mean()


def pooled_corr(x):
    f = x.reshape(-1, x.shape[-1])
    f = f - f.mean(0, keepdim=True)
    cov = f.T @ f / (len(f) - 1)
    sd = torch.sqrt(torch.diagonal(cov)).clamp(min=1e-8)
    return cov / sd[:, None] / sd[None, :]


def corr_loss(r, y, target, min_windows=4):
    """Mean over regimes of the mean squared off-diagonal gap between the pooled correlation of the
    sampled windows of that regime and the training target (regimes with < min_windows are skipped)."""
    N = r.shape[-1]
    off = ~torch.eye(N, dtype=torch.bool, device=r.device)
    terms = [((pooled_corr(r[y == k]) - target[k])[off] ** 2).mean() for k in (0, 1) if (y == k).sum() >= min_windows]
    if not terms:
        raise RuntimeError("correlation loss: no regime has enough windows in this batch")
    return torch.stack(terms).mean()


def graph_loss(att, tgt):
    return ((att - tgt) ** 2).sum(-1).mean()


def loss_terms(net: CGNSDENet, batch: dict, corr_target, tau: float, gen: torch.Generator,
               want=("corr", "supcon", "graph"), sample_z0: bool = True) -> dict:
    """Unweighted loss terms for one batch (keys: nll, kl and those of `want` that apply)."""
    x, y = batch["x"], batch["y"]
    g, att = net.graph(batch["feats"], batch["adj"])
    mu, logvar = net.encode(x)
    z0 = mu + torch.randn(mu.shape, generator=gen) * torch.exp(0.5 * logvar) if sample_z0 else mu
    zs = net.path(z0, g, torch.randn(net.T, len(x), net.L, generator=gen))
    m, d, B = net.heads(zs)
    mult = net.teacher_mult(x, m, d, B)
    nu = net.nu()
    out = dict(nll=-emission_log_prob(x, m, d, B, mult, nu).mean(),
               kl=gaussian_kl(mu, logvar, net.prior_mu[y], net.prior_logvar[y]))
    if "corr" in want:
        sN, sK, sw = net.noise_shapes(len(x))
        w = None
        if nu is not None:
            w = torch.distributions.Chi2(nu.detach()).sample(sw)
        r = emission_draw(m, d, B, mult, None if nu is None else nu.detach(),
                          torch.randn(sN, generator=gen), torch.randn(sK, generator=gen), w)
        out["corr"] = corr_loss(r, y, corr_target)
    if "supcon" in want:
        out["supcon"] = supcon_loss(net.proj(zs.mean(1)), y, tau)
    if "graph" in want and att is not None:
        out["graph"] = graph_loss(att, batch["tgt"])
    return out


def ramp(epoch: int, n: int) -> float:
    return min(1.0, (epoch + 1) / max(1, n))


def epoch_weights(hp: dict, epoch: int) -> dict:
    return dict(kl=hp["beta_max"] * ramp(epoch, hp["beta_ramp"]), corr=hp["lam_corr"] * ramp(epoch, hp["corr_ramp"]),
                supcon=hp["lam_con"] * ramp(epoch, hp["con_warmup"]), graph=hp["lam_graph"])


def total_loss(terms: dict, w: dict):
    return terms["nll"] + sum(w[k] * v for k, v in terms.items() if k != "nll")


def stratified_batches(y: np.ndarray, batch: int, n_crisis: int, rng: np.random.Generator):
    """One pass over the normal windows; every batch holds exactly n_crisis crisis windows
    (crisis indices are cycled through reshuffled permutations).  The last short batch is dropped."""
    cr, no = np.where(y == 1)[0], rng.permutation(np.where(y == 0)[0])
    n_no = batch - n_crisis
    pool = np.empty(0, dtype=np.int64)
    for s in range(0, len(no) - n_no + 1, n_no):
        while len(pool) < n_crisis:
            perm = rng.permutation(cr)
            held = np.isin(perm, pool)                  # leftovers go last so a batch never repeats a window
            pool = np.concatenate([pool, perm[~held], perm[held]])
        yield np.concatenate([pool[:n_crisis], no[s:s + n_no]])
        pool = pool[n_crisis:]


# ---------------------------------------------------------------------- generator
class CGNSDE(Generator):
    name = "cgnsde"
    family = "neural-ours"
    impl_source = "own"
    supports_regime = True
    VARIANTS = {
        "no_gat": dict(use_gat=False), "no_supcon": dict(lam_con=0.0), "no_graph_loss": dict(lam_graph=0.0),
        "no_corr": dict(lam_corr=0.0), "no_voldyn": dict(vol_dyn=False), "no_sde": dict(use_sde=False),
        "gauss": dict(emission="gauss"), "K0": dict(rank_K=0), "K2": dict(rank_K=2), "K4": dict(rank_K=4),
        "K16": dict(rank_K=16), "lam_graph03": dict(lam_graph=0.3),
    }

    def __init__(self, **hparams):
        if "epochs" in hparams:                           # alias used on the command line
            hparams["max_epochs"] = hparams.pop("epochs")
        unknown = set(hparams) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown CG-NSDE hyperparameters: {sorted(unknown)}")
        hp = {**DEFAULTS, **hparams}
        hp["gmm_components"] = [int(k) for k in hp["gmm_components"]]
        super().__init__(**hp)
        self.net: CGNSDENet | None = None

    # ---------------------------------------------------------------- fitting
    def _arrays(self, Z: np.ndarray, W_raw: np.ndarray, vol_mean) -> dict:
        adj, tgt = window_graphs(Z, self.hparams["knn_k"])
        return dict(x=torch.from_numpy(Z.astype(np.float32)), adj=torch.from_numpy(adj), tgt=torch.from_numpy(tgt),
                    feats=torch.from_numpy(node_features(Z, W_raw, vol_mean, self.sector)))

    def fit(self, data: FitData, seed: int) -> dict:
        hp = self.hparams
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        gen = torch.Generator().manual_seed(seed)
        T, N = P.T, data.n_assets
        self.tickers, self.sector = list(data.tickers), sector_feature(list(data.tickers))
        train_days = ~data.val_mask
        self.mu = data.returns[train_days].mean(0)
        self.sd = data.returns[train_days].std(0)
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
            vol_mean = (c[T:] - c[:-T]) / T                # (n_windows, N) window mean of z-scored log-volume
        Z = (W - self.mu) / self.sd
        sub = lambda idx: self._arrays(Z[idx], W[idx], None if vol_mean is None else vol_mean[idx])
        A_tr, A_va = sub(tr), sub(va)
        y_tr = y_all[tr]
        A_tr["y"] = torch.from_numpy(y_tr)
        n_cr = int(y_tr.sum())
        if n_cr < hp["min_crisis"] or len(y_tr) - n_cr < hp["batch"] - hp["min_crisis"]:
            raise ValueError(f"cannot build stratified batches: {n_cr} crisis / {len(y_tr) - n_cr} normal training windows")
        self.p_crisis = float(y_tr.mean())
        # Per-regime pooled correlation of the z-scored TRAINING windows (targets of L_corr).
        corr_target = torch.stack([pooled_corr(A_tr["x"][A_tr["y"] == k].double()).float() for k in (0, 1)])

        self.net = net = CGNSDENet(N, hp)
        opt = torch.optim.Adam(net.parameters(), lr=hp["lr"])
        take = lambda A, idx: {k: v[idx] for k, v in A.items()}
        active = [k for k, lam in (("corr", hp["lam_corr"]), ("supcon", hp["lam_con"]), ("graph", hp["lam_graph"]))
                  if lam > 0]
        history, best, best_state, bad = [], (math.inf, -1), None, 0
        for epoch in range(int(hp["max_epochs"])):
            t0 = time.time()
            w = epoch_weights(hp, epoch)
            net.train()
            acc, nb = {}, 0
            for idx in stratified_batches(y_tr, hp["batch"], hp["min_crisis"], rng):
                terms = loss_terms(net, take(A_tr, torch.from_numpy(idx)), corr_target, hp["tau"], gen, want=active)
                loss = total_loss(terms, w)
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
            val = self._val_nll(A_va, seed)
            if not math.isfinite(val):
                raise FloatingPointError(f"non-finite validation NLL at epoch {epoch}")
            rec = dict(epoch=epoch + 1, **{f"train_{k}": v / nb for k, v in acc.items()}, val_nll=val,
                       beta=w["kl"], lam_corr=w["corr"], lam_con=w["supcon"], lam_graph=w["graph"],
                       seconds=time.time() - t0)
            history.append(rec)
            if hp["verbose"]:
                print("  ".join(f"{k}={v:.4g}" for k, v in rec.items()), flush=True)
            if val < best[0]:
                best, bad = (val, epoch + 1), 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= hp["patience"]:
                    break
        net.load_state_dict(best_state)
        net.eval()
        self._fit_mixtures(A_tr, y_tr, seed, gen)
        self.info = dict(
            epochs_run=len(history), best_epoch=best[1], best_val_nll=best[0], history=history,
            nu=float(net.nu()) if net.student else None,
            rho=float(torch.sigmoid(net.rho_raw)) if net.vol_dyn else None,
            kappa=float(torch.sigmoid(net.kappa_raw)) if net.vol_dyn else None,
            mixture_components={"normal": len(self.mix[0]["w"]), "crisis": len(self.mix[1]["w"])},
            n_train_windows=int(len(tr)), n_val_windows=int(len(va)), n_train_crisis=n_cr,
            train_crisis_freq=self.p_crisis, batches_per_epoch=nb, n_params=sum(p.numel() for p in net.parameters()),
            sec_per_epoch=float(np.mean([h["seconds"] for h in history])), seed=int(seed))
        return self.info

    @torch.no_grad()
    def _val_nll(self, A: dict, seed: int, chunk: int = 256) -> float:
        """Teacher-forced NLL with z0 = posterior mean and a fixed Brownian draw."""
        net = self.net
        net.eval()
        gen = torch.Generator().manual_seed(1_000_003 + seed)
        tot = 0.0
        for s in range(0, len(A["x"]), chunk):
            b = {k: v[s:s + chunk] for k, v in A.items()}
            b["y"] = torch.zeros(len(b["x"]), dtype=torch.long)     # labels only enter the KL, unused here
            tot += float(loss_terms(net, b, None, 0.0, gen, want=(), sample_z0=False)["nll"]) * len(b["x"])
        return tot / len(A["x"])

    @torch.no_grad()
    def _fit_mixtures(self, A: dict, y: np.ndarray, seed: int, gen: torch.Generator, chunk: int = 512) -> None:
        """Ex-post density of [posterior sample z0, g] per regime: diagonal GMM, components by BIC."""
        from sklearn.mixture import GaussianMixture
        lat = []
        for s in range(0, len(A["x"]), chunk):
            g, _ = self.net.graph(A["feats"][s:s + chunk], A["adj"][s:s + chunk])
            mu, logvar = self.net.encode(A["x"][s:s + chunk])
            z0 = mu + torch.randn(mu.shape, generator=gen) * torch.exp(0.5 * logvar)
            lat.append(torch.cat([z0, g], 1))
        lat = torch.cat(lat).double().numpy()
        self.mix = {}
        for k in (0, 1):
            X = lat[y == k]
            fits = [GaussianMixture(c, covariance_type="diag", random_state=seed).fit(X)
                    for c in self.hparams["gmm_components"] if len(X) >= 2 * c]
            if not fits:
                raise ValueError(f"regime {k}: {len(X)} training windows are too few for any mixture")
            gm = min(fits, key=lambda f: f.bic(X))
            self.mix[k] = dict(w=gm.weights_.copy(), mean=gm.means_.copy(), var=gm.covariances_.copy(),
                               bic=float(gm.bic(X)))

    # --------------------------------------------------------------- sampling
    def _require_fit(self):
        if self.net is None or not hasattr(self, "mix"):
            raise RuntimeError("CG-NSDE is not fitted")

    @torch.no_grad()
    def _generate(self, regimes: np.ndarray, rng: np.random.Generator, chunk: int = 1000) -> np.ndarray:
        self._require_fit()
        torch.set_num_threads(4)
        net = self.net
        net.eval()
        f32 = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))
        nu = float(net.nu()) if net.student else None
        out = []
        for s in range(0, len(regimes), chunk):
            reg = regimes[s:s + chunk]
            n = len(reg)
            lat = np.empty((n, net.L + net.G))
            for k in (0, 1):
                sel = np.where(reg == k)[0]
                mx = self.mix[k]
                comp = rng.choice(len(mx["w"]), size=len(sel), p=mx["w"] / mx["w"].sum())
                lat[sel] = mx["mean"][comp] + np.sqrt(mx["var"][comp]) * rng.standard_normal((len(sel), lat.shape[1]))
            lat = f32(lat)
            zs = net.path(lat[:, :net.L], lat[:, net.L:], f32(rng.standard_normal((net.T, n, net.L))))
            sN, sK, sw = net.noise_shapes(n)
            w = f32(rng.chisquare(nu, sw)) if nu is not None else torch.ones(sw)
            z = net.sample_seq(zs, f32(rng.standard_normal(sN)), f32(rng.standard_normal(sK)), w)
            out.append(z.double().numpy() * self.sd + self.mu)
        G = np.concatenate(out)
        if not np.isfinite(G).all():
            raise FloatingPointError("CG-NSDE produced non-finite returns")
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
        """Latent view of RAW windows (n, T, N): z0 = posterior mean, g = graph embedding, proj = L2-normalised
        contrastive projection of the time-pooled drift-only (zero-noise) latent path.  Deterministic.
        Without volume_windows the volume node feature is 0 (it was the window mean in training)."""
        self._require_fit()
        self.net.eval()
        A = self._prep(windows_raw, volume_windows)
        g, _ = self.net.graph(A["feats"], A["adj"])
        z0, _ = self.net.encode(A["x"])
        zs = self.net.path(z0, g, torch.zeros(self.net.T, len(z0), self.net.L))
        proj = F.normalize(self.net.proj(zs.mean(1)), dim=1)
        return dict(z0=z0.numpy(), g=g.numpy(), proj=proj.numpy())

    @torch.no_grad()
    def attention(self, window_raw, volume_window=None) -> np.ndarray:
        """(N, N) layer-2 GAT attention of one RAW (T, N) window; row i = weights over i's neighbours."""
        self._require_fit()
        if not self.net.use_gat:
            raise NotImplementedError("this variant has no GAT, hence no attention matrix")
        self.net.eval()
        A = self._prep(np.asarray(window_raw)[None], None if volume_window is None else np.asarray(volume_window)[None])
        return self.net.graph(A["feats"], A["adj"])[1][0].numpy()

    # ------------------------------------------------------------ persistence
    @staticmethod
    def _file(path) -> Path:
        path = Path(path)
        return path if path.suffix == ".pt" else path.with_name(path.name + ".pt")

    def save(self, path) -> None:
        """One file (<path>.pt) holding plain tensors and Python types only (loads with weights_only=True)."""
        self._require_fit()
        t = lambda a: None if a is None else torch.as_tensor(np.asarray(a))
        f = self._file(path)
        f.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(
            format=1, hparams=dict(self.hparams), tickers=list(self.tickers), name=self.name,
            state_dict=self.net.state_dict(), mu=t(self.mu), sd=t(self.sd), vol_mu=t(self.vol_mu),
            vol_sd=t(self.vol_sd), sector=t(self.sector), p_crisis=self.p_crisis,
            mix={k: {q: (t(v) if q != "bic" else v) for q, v in m.items()} for k, m in self.mix.items()},
            info=getattr(self, "info", {})), f)

    @classmethod
    def load(cls, path) -> "CGNSDE":
        ck = torch.load(cls._file(path), map_location="cpu", weights_only=True)
        if ck.get("format") != 1:
            raise ValueError(f"unsupported CG-NSDE checkpoint format: {ck.get('format')}")
        torch.set_num_threads(4)
        n = lambda a: None if a is None else a.numpy()
        self = cls(**ck["hparams"])
        self.name, self.tickers, self.p_crisis, self.info = ck["name"], ck["tickers"], ck["p_crisis"], ck["info"]
        self.mu, self.sd, self.vol_mu, self.vol_sd = n(ck["mu"]), n(ck["sd"]), n(ck["vol_mu"]), n(ck["vol_sd"])
        self.sector = n(ck["sector"])
        self.mix = {int(k): {q: (n(v) if q != "bic" else v) for q, v in m.items()} for k, m in ck["mix"].items()}
        self.net = CGNSDENet(len(self.tickers), self.hparams)
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        return self
