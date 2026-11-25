"""Quant GAN baseline (Wiese, Knobloch, Korn, Kretschmer 2020, "Quant GANs: Deep Generation of
Financial Time Series"), own PyTorch reimplementation with a multi-asset extension.

Preprocessing (per asset, estimated on the training days = days with val_mask False)
  1. standardise;  2. inverse Lambert-W transform (heavy-tail Lambert W x Gaussian, Goerg 2015)
  with (mu, sigma, delta) estimated by IGMM (delta makes the kurtosis of the back-transformed
  series equal to 3);  3. standardise again.  Generation applies the EXACT inverse:
  u -> mu + sigma * u * exp(delta / 2 * u^2).

Networks: temporal convolutional networks.  7 temporal blocks (kernel 1, then kernel 2 with
  dilations 1, 2, 4, 8, 16, 32), two dilated causal convolutions + PReLU per block, residual
  connection per block, skip connections of all blocks summed into a 1x1 output convolution,
  80 hidden channels.  Receptive field 1 + 2 * (1 + 2 + ... + 32) = 127.
  generator      noise path (T + 126, noise_dim), unpadded ("valid") convolutions -> (T, N)
  discriminator  (T, N) window, causal zero left-padding, score = output at the LAST day
                 (the only position whose receptive field covers the whole window)

Training: WGAN-GP (Gulrajani et al. 2017: lambda 10, 5 critic steps per generator step, Adam
  lr 1e-4, betas (0, 0.9)) rather than the weight-clipping WGAN that the benchmark brief
  attributes to the paper.  Reason: with the small fixed iteration budget available on CPU,
  hard clipping to +-0.01 of an 80-channel, 14-convolution critic drives it towards very simple
  functions and makes training slow and sensitive to the clip value; the gradient penalty
  enforces the Lipschitz constraint without capacity loss.  Cost: one double-backward pass per
  critic step.  Fixed iteration budget, no early stopping, the validation block is not used.

Deviations from the reference
  * Multi-asset: one generator emits N series per day from a shared noise path and the critic
    sees N channels (the paper is univariate).  Cross-asset dependence is therefore learned
    only through the adversarial loss.
  * Critic windows are T = 60 days (the benchmark window), shorter than the receptive field
    127, hence the causal zero padding in the critic.  The generator is exactly as in the
    paper: it never sees padding.
  * WGAN-GP instead of weight clipping (above).  noise_dim 3; PyTorch default initialisation.
  * `iters` counts generator updates; the default is far below what a GPU run would use.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import brentq
from scipy.special import lambertw
from torch import nn

from .. import protocol as P
from .base import FitData, Generator

DILATIONS = (1, 2, 4, 8, 16, 32)
RF = 1 + 2 * sum(DILATIONS)          # 127


# ------------------------------------------------------------------ Lambert W
def lambert_inverse(z: np.ndarray, delta: float) -> np.ndarray:
    """Gaussianise: u with u * exp(delta/2 u^2) = z."""
    z = np.asarray(z, dtype=np.float64)
    if delta < 0:
        raise ValueError("delta must be >= 0")
    if delta == 0:
        return z.copy()
    return np.sign(z) * np.sqrt(np.real(lambertw(delta * z * z)) / delta)


def lambert_forward(u: np.ndarray, delta: float) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    return u * np.exp(0.5 * delta * u * u)


def _kurt(x: np.ndarray) -> float:
    c = x - x.mean()
    return float((c ** 4).mean() / (c ** 2).mean() ** 2)


def delta_gmm(z: np.ndarray) -> float:
    """delta >= 0 such that the kurtosis of W_delta(z) is 3 (0 if z is not leptokurtic)."""
    f = lambda d: _kurt(lambert_inverse(z, d)) - 3.0
    if f(0.0) <= 0:
        return 0.0
    hi = 0.5
    while f(hi) > 0:
        hi *= 2
        if hi > 64:
            raise RuntimeError("delta_gmm: kurtosis cannot be brought down to 3")
    return float(brentq(f, 0.0, hi, xtol=1e-10))


def igmm(y: np.ndarray, tol: float = 1e-6, max_iter: int = 200) -> tuple[float, float, float]:
    """Iterative generalised method of moments (Goerg 2015) for (mu, sigma, delta), gamma = 0."""
    y = np.asarray(y, dtype=np.float64)
    mu, sigma, delta = float(np.median(y)), float(y.std()), 0.0
    for _ in range(max_iter):
        delta_new = delta_gmm((y - mu) / sigma)
        x = mu + sigma * lambert_inverse((y - mu) / sigma, delta_new)
        mu_new, sigma_new = float(x.mean()), float(x.std())
        step = max(abs(mu_new - mu), abs(sigma_new - sigma), abs(delta_new - delta))
        mu, sigma, delta = mu_new, sigma_new, delta_new
        if step < tol:
            return mu, sigma, delta
    raise RuntimeError("IGMM did not converge")


import os
DEV = torch.device(os.environ.get("CRISISBENCH_DEVICE", "cpu"))   # "cuda" for full-scale GPU runs

class LambertScaler:
    """standardise -> inverse Lambert W (IGMM) -> standardise; exact inverse."""

    def fit(self, X: np.ndarray) -> "LambertScaler":
        X = np.asarray(X, dtype=np.float64)
        self.m1, self.s1 = X.mean(0), X.std(0)
        if not (np.isfinite(self.s1).all() and (self.s1 > 0).all()):
            raise ValueError("LambertScaler: an asset has zero or non-finite variance")
        Z = (X - self.m1) / self.s1
        par = np.array([igmm(Z[:, j]) for j in range(X.shape[1])])
        self.mu, self.sigma, self.delta = par[:, 0], par[:, 1], par[:, 2]
        U = self._gauss(Z)
        self.m2, self.s2 = U.mean(0), U.std(0)
        return self

    def _gauss(self, Z):
        V = (Z - self.mu) / self.sigma
        return np.stack([lambert_inverse(V[..., j], self.delta[j]) for j in range(V.shape[-1])], axis=-1)

    def transform(self, X):
        return (self._gauss((np.asarray(X, dtype=np.float64) - self.m1) / self.s1) - self.m2) / self.s2

    def inverse(self, Y):
        U = np.asarray(Y, dtype=np.float64) * self.s2 + self.m2
        V = np.stack([lambert_forward(U[..., j], self.delta[j]) for j in range(U.shape[-1])], axis=-1)
        return (V * self.sigma + self.mu) * self.s1 + self.m1

    def state(self) -> dict:
        return {k: getattr(self, k) for k in ("m1", "s1", "mu", "sigma", "delta", "m2", "s2")}


# ------------------------------------------------------------------------ TCN
class _Block(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, d: int, pad: bool):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, dilation=d)
        self.c2 = nn.Conv1d(cout, cout, k, dilation=d)
        self.a1, self.a2 = nn.PReLU(), nn.PReLU()
        self.down = nn.Conv1d(cin, cout, 1) if cin != cout else None
        self.p = (k - 1) * d if pad else 0          # causal: pad on the left only

    def forward(self, x):
        h = self.a1(self.c1(F.pad(x, (self.p, 0))))
        h = self.a2(self.c2(F.pad(h, (self.p, 0))))
        res = x if self.down is None else self.down(x)
        return h + res[..., -h.shape[-1]:], h       # (block output, skip)


class TCN(nn.Module):
    def __init__(self, cin: int, cout: int, hidden: int, pad: bool):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(cin, hidden, 1, 1, pad)]
                                    + [_Block(hidden, hidden, 2, d, pad) for d in DILATIONS])
        self.out = nn.Conv1d(hidden, cout, 1)

    def forward(self, x):                            # (B, C, L) -> (B, cout, L - RF + 1) or (B, cout, L)
        skips = []
        for b in self.blocks:
            x, s = b(x)
            skips.append(s)
        L = x.shape[-1]
        return self.out(sum(s[..., -L:] for s in skips))


class QuantGAN(Generator):
    family = "neural"
    impl_source = "own PyTorch reimplementation"

    def __init__(self, iters: int = 500, epochs: int | None = None, batch: int = 64, hidden: int = 80,
                 noise_dim: int = 3, n_critic: int = 5, gp_lambda: float = 10.0, lr: float = 1e-4, **kw):
        super().__init__(iters=iters, epochs=epochs, batch=batch, hidden=hidden, noise_dim=noise_dim,
                         n_critic=n_critic, gp_lambda=gp_lambda, lr=lr, **kw)
        self.iters, self.epochs, self.batch, self.hidden = int(iters), epochs, int(batch), int(hidden)
        self.noise_dim, self.n_critic, self.gp_lambda, self.lr = int(noise_dim), int(n_critic), float(gp_lambda), float(lr)
        self.G = None

    def fit(self, data: FitData, seed: int) -> dict:
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        g = torch.Generator().manual_seed(seed)
        self.scaler = LambertScaler().fit(data.returns[~data.val_mask])
        tr, _ = data.split_windows()
        if len(tr) < self.batch:
            raise ValueError(f"QuantGAN: {len(tr)} training windows < batch {self.batch}")
        X = torch.as_tensor(self.scaler.transform(data.windows()[tr]), dtype=torch.float32).transpose(1, 2).to(DEV)
        N, Lz, B = data.n_assets, P.T + RF - 1, self.batch
        # "epochs" (smoke tests): one epoch = one pass of generator updates over the training windows
        iters = self.iters if self.epochs is None else int(self.epochs) * math.ceil(len(tr) / B)
        self.G = TCN(self.noise_dim, N, self.hidden, pad=False).to(DEV)
        self.D = TCN(N, 1, self.hidden, pad=True).to(DEV)
        oG = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.0, 0.9))
        oD = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.0, 0.9))
        critic = lambda x: self.D(x)[:, 0, -1]
        every = max(1, iters // 200)
        hist, t0 = [], time.time()
        for it in range(iters):
            for _ in range(self.n_critic):
                real = X[torch.from_numpy(rng.integers(0, len(X), B)).to(DEV)]
                with torch.no_grad():
                    fake = self.G(torch.randn(B, self.noise_dim, Lz, generator=g).to(DEV))
                a = torch.rand(B, 1, 1, generator=g).to(DEV)
                xi = (a * real + (1 - a) * fake).requires_grad_(True)
                grad = torch.autograd.grad(critic(xi).sum(), xi, create_graph=True)[0]
                gp = ((grad.flatten(1).norm(dim=1) - 1) ** 2).mean()
                wdist = critic(real).mean() - critic(fake).mean()
                lossD = -wdist + self.gp_lambda * gp
                oD.zero_grad()
                lossD.backward()
                oD.step()
            lossG = -critic(self.G(torch.randn(B, self.noise_dim, Lz, generator=g).to(DEV))).mean()
            oG.zero_grad()
            lossG.backward()
            oG.step()
            if not (torch.isfinite(lossD) and torch.isfinite(lossG)):
                raise FloatingPointError(f"QuantGAN: non-finite loss at iteration {it}")
            if it % every == 0 or it == iters - 1:
                hist.append(dict(iter=it, d_loss=float(lossD), g_loss=float(lossG), wdist=float(wdist), gp=float(gp)))
        self.G.eval()
        return dict(n_train=int(len(tr)), iters=iters, critic_steps=iters * self.n_critic, loss="WGAN-GP",
                    receptive_field=RF, sec_per_iter=(time.time() - t0) / max(iters, 1),
                    delta_min=float(self.scaler.delta.min()), delta_median=float(np.median(self.scaler.delta)),
                    delta_max=float(self.scaler.delta.max()),
                    n_params_G=sum(p.numel() for p in self.G.parameters()),
                    n_params_D=sum(p.numel() for p in self.D.parameters()), history=hist)

    @torch.no_grad()
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.G is None:
            raise RuntimeError("QuantGAN.sample called before fit")
        out = []
        for i in range(0, n, 256):
            z = rng.standard_normal((min(256, n - i), self.noise_dim, P.T + RF - 1))
            out.append(self.G(torch.as_tensor(z, dtype=torch.float32).to(DEV)).transpose(1, 2).cpu().numpy())
        G = self.scaler.inverse(np.concatenate(out))
        if not np.isfinite(G).all():
            raise FloatingPointError("QuantGAN produced non-finite returns (Lambert-W tail overflow)")
        return G

    def save(self, path) -> None:
        path = Path(str(path) + ".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(G=self.G.state_dict(), D=self.D.state_dict(), scaler=self.scaler.state(),
                        hparams=self.hparams), path)
