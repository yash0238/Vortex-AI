"""Denoising diffusion baseline for return windows (own PyTorch implementation).

Model     DDPM (Ho, Jain & Abbeel 2020) on per-asset z-scored windows x0 (T, N), trained to predict the clean
          window x0 directly (MSE), as in Diffusion-TS (Yuan & Qiao 2024); cosine noise schedule (Nichol &
          Dhariwal 2021), 1000 steps.
Denoiser  Transformer encoder over the T days: each day (N returns) is one token, linear in/out projections,
          learned positional embedding, sinusoidal diffusion-step embedding added to every token
          (d_model 128, 4 layers, 4 heads, feed-forward 256, GELU, pre-norm).
Sampling  deterministic DDIM (Song, Meng & Ermon 2021), eta = 0, `sample_steps` uniformly spaced steps.
          Unconditional: no real window, date or label is used.
Training  Adam, lr 2e-4, batch 128, `iters` iterations; an exponential moving average of the weights
          (decay 0.999) is used for sampling.  No early stopping (the validation block is unused).

Deviations / remarks
  * This is a generic unconditional DDPM, not a reimplementation of a specific published financial diffusion
    model (e.g. Diffusion-TS, CSDI, TimeGrad); it is labelled "DDPM (own)" in the tables.
  * Default `iters` is a CPU smoke budget; the full-scale runs use the budget in crisisbench/fullscale.py.
  * Fix of 2026-11-18, after the first full-scale run: that run used the epsilon parametrisation.  The cosine
    schedule ends at alpha_bar = 2.4e-9, so the first DDIM steps divide the denoiser's residual error by
    sqrt(alpha_bar) ~ 5e-5 and the samples diverged (generated std 780x real; clipping x0 alone still gave 11x
    in a CPU test).  The x0 parametrisation has no such division at high noise.  The predicted x0 is also
    clipped to +-c, c = the largest |z-score| in the training windows (data-derived), and eps is recomputed
    from it; the first run's results were discarded.
"""
from __future__ import annotations

import copy
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .. import protocol as P
from .base import FitData, Generator

DEV = torch.device(os.environ.get("CRISISBENCH_DEVICE", "cpu"))


def cosine_alpha_bar(n: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, n, n + 1, dtype=torch.float64) / n
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, 0).float()          # alpha_bar_t for t = 1..n (index t-1)


class _Denoiser(nn.Module):
    def __init__(self, N: int, T: int, d: int, layers: int, heads: int, ff: int):
        super().__init__()
        self.d = d
        self.inp, self.out = nn.Linear(N, d), nn.Linear(d, N)
        self.pos = nn.Parameter(torch.zeros(1, T, d))
        nn.init.normal_(self.pos, std=0.02)
        self.temb = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, t):
        half = self.d // 2
        freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device) / half)
        a = t.float()[:, None] * freq[None]
        emb = self.temb(torch.cat([a.sin(), a.cos()], 1))
        h = self.inp(x) + self.pos + emb[:, None, :]
        return self.out(self.norm(self.enc(h)))


class Diffusion(Generator):
    name = "diffusion"
    family = "neural"
    impl_source = "own PyTorch implementation"

    def __init__(self, iters: int = 300, batch: int = 128, lr: float = 2e-4, d_model: int = 128, layers: int = 4,
                 heads: int = 4, ff: int = 256, n_steps: int = 1000, sample_steps: int = 100, ema: float = 0.999, **kw):
        super().__init__(iters=iters, batch=batch, lr=lr, d_model=d_model, layers=layers, heads=heads, ff=ff,
                         n_steps=n_steps, sample_steps=sample_steps, ema=ema, **kw)
        self.iters, self.batch, self.lr = int(iters), int(batch), float(lr)
        self.arch = (int(d_model), int(layers), int(heads), int(ff))
        self.n_steps, self.sample_steps, self.ema = int(n_steps), int(sample_steps), float(ema)
        self.net = None

    def fit(self, data: FitData, seed: int) -> dict:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        g = torch.Generator().manual_seed(seed)
        train_days = data.returns[~data.val_mask]
        self.mu, self.sd = train_days.mean(0), train_days.std(0, ddof=1)
        if not (np.isfinite(self.sd).all() and (self.sd > 0).all()):
            raise ValueError("Diffusion: an asset has zero or non-finite training variance")
        tr, _ = data.split_windows()
        if len(tr) < self.batch:
            raise ValueError(f"Diffusion: {len(tr)} training windows < batch {self.batch}")
        X = torch.as_tensor((data.windows()[tr] - self.mu) / self.sd, dtype=torch.float32).to(DEV)
        self.n_assets = data.n_assets
        self.clip = float(X.abs().max())
        net = _Denoiser(self.n_assets, P.T, *self.arch).to(DEV)
        ema = copy.deepcopy(net)
        for p in ema.parameters():
            p.requires_grad_(False)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        ab = cosine_alpha_bar(self.n_steps).to(DEV)
        B, every, hist, t0 = self.batch, max(1, self.iters // 200), [], time.time()
        for it in range(self.iters):
            x0 = X[torch.from_numpy(rng.integers(0, len(X), B)).to(DEV)]
            t = torch.randint(0, self.n_steps, (B,), generator=g).to(DEV)
            eps = torch.randn(x0.shape, generator=g).to(DEV)
            a = ab[t][:, None, None]
            loss = F.mse_loss(net(a.sqrt() * x0 + (1 - a).sqrt() * eps, t), x0)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Diffusion: non-finite loss at iteration {it}")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.mul_(self.ema).add_(pn, alpha=1 - self.ema)
            if it % every == 0 or it == self.iters - 1:
                hist.append(dict(iter=it, loss=float(loss.detach())))
        self.net = ema.eval()
        self.ab = ab
        return dict(n_train=int(len(tr)), iters=self.iters, sec_per_iter=(time.time() - t0) / max(self.iters, 1),
                    n_params=sum(p.numel() for p in net.parameters()), device=str(DEV), history=hist)

    @torch.no_grad()
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Diffusion.sample called before fit")
        steps = np.unique(np.linspace(0, self.n_steps - 1, self.sample_steps).round().astype(int))[::-1]
        out = []
        for i in range(0, n, 500):
            b = min(500, n - i)
            x = torch.as_tensor(rng.standard_normal((b, P.T, self.n_assets)), dtype=torch.float32).to(DEV)
            for j, s in enumerate(steps):
                a = self.ab[s]
                x0 = self.net(x, torch.full((b,), int(s), device=DEV)).clamp(-self.clip, self.clip)
                eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
                if j + 1 < len(steps):
                    a2 = self.ab[steps[j + 1]]
                    x = a2.sqrt() * x0 + (1 - a2).sqrt() * eps
                else:
                    x = x0
            out.append(x.cpu().numpy().astype(np.float64))
        G = np.concatenate(out) * self.sd + self.mu
        if not np.isfinite(G).all():
            raise FloatingPointError("Diffusion produced non-finite returns")
        return G

    def save(self, path) -> None:
        path = Path(str(path) + ".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state_dict=self.net.state_dict(), mu=self.mu, sd=self.sd, clip=self.clip, hparams=self.hparams), path)
