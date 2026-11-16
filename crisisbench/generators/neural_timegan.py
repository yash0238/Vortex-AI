"""TimeGAN baseline (Yoon, Jarrett, van der Schaar, NeurIPS 2019, "Time-series Generative
Adversarial Networks"), own PyTorch reimplementation.

Networks (GRU, hidden_dim 64, 3 layers; the supervisor has layers - 1; sigmoid outputs, the
discriminator returns per-step logits):  embedder X -> H, recovery H -> X~, generator Z -> E^,
supervisor H -> next-step H, discriminator H -> logit.  Noise Z ~ U[0, 1], dim = N per step.

Training, three phases of `iters` iterations each (Algorithm of the paper / released code):
  1 autoencoding   E, R:  eta * sqrt(mse(X, R(E(X))))                               (eta = 10)
  2 supervised     S:     mse(H[1:], S(H)[:-1]),  H = E(X)
  3 joint          per iteration, TWICE:
                     G, S:  bce(D(S(G(Z))), 1) + gamma * bce(D(G(Z)), 1)
                            + w_sup * sqrt(mse(H[1:], S(H)[:-1])) + w_mom * (V1 + V2)
                            V1 = mean |std_batch(X^) - std_batch(X)|, V2 = mean |mean_batch(X^) - mean_batch(X)|
                     E, R:  eta * sqrt(mse(X, X~)) + 0.1 * mse(H[1:], S(H)[:-1])
                   then ONCE:
                     D:     bce(D(H), 1) + bce(D(S(G(Z))), 0) + gamma * bce(D(G(Z)), 0),
                            the update is applied only when this loss > 0.15
  gamma = 1, eta = 10; w_sup = w_mom = 100 are the weights of the authors' released code as
  recalled (not re-checked against the source in this project).  Adam lr 1e-3, batch 128.

Scaling: min-max to [0, 1] per asset from the training windows; exact inverse at generation.
sample(): Z -> G -> S -> R -> inverse scaling.  Generated returns are therefore bounded by the
per-asset training minimum and maximum.

Deviations from the reference
  * Iterations: the paper's 50,000 per phase are impossible on this CPU; the default `iters`
    is the largest per-phase count that fits the 20-minute budget (see the report).
  * Min-max is per asset over all days of the training windows (the reference scales each
    feature over the whole data set in the same way; no per-window scaling).
  * Fixed budget, no early stopping; the validation block is not used at all.
  * All windows have the full length T, so there is no sequence-length masking.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .. import protocol as P
from .base import FitData, Generator


import os
DEV = torch.device(os.environ.get("CRISISBENCH_DEVICE", "cpu"))   # "cuda" for full-scale GPU runs

class MinMaxScaler:
    def fit(self, W: np.ndarray) -> "MinMaxScaler":
        flat = np.asarray(W, dtype=np.float64).reshape(-1, W.shape[-1])
        self.lo, self.hi = flat.min(0), flat.max(0)
        if not (np.isfinite(self.lo).all() and np.isfinite(self.hi).all() and (self.hi > self.lo).all()):
            raise ValueError("MinMaxScaler: an asset has zero or non-finite range")
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.lo) / (self.hi - self.lo)

    def inverse(self, Y):
        return np.asarray(Y, dtype=np.float64) * (self.hi - self.lo) + self.lo


class _RNN(nn.Module):
    def __init__(self, n_in: int, hidden: int, n_out: int, layers: int, sigmoid: bool = True):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, layers, batch_first=True)
        self.lin = nn.Linear(hidden, n_out)
        self.sigmoid = sigmoid

    def forward(self, x):
        y = self.lin(self.gru(x)[0])
        return torch.sigmoid(y) if self.sigmoid else y


def _moment_loss(xh, x):
    v1 = (torch.sqrt(xh.var(0, unbiased=False) + 1e-6) - torch.sqrt(x.var(0, unbiased=False) + 1e-6)).abs().mean()
    v2 = (xh.mean(0) - x.mean(0)).abs().mean()
    return v1 + v2


class TimeGAN(Generator):
    family = "neural"
    impl_source = "own PyTorch reimplementation"

    def __init__(self, iters: int = 400, epochs: int | None = None, batch: int = 128, hidden_dim: int = 64,
                 layers: int = 3, lr: float = 1e-3, gamma: float = 1.0, eta: float = 10.0, w_sup: float = 100.0,
                 w_mom: float = 100.0, d_thresh: float = 0.15, **kw):
        super().__init__(iters=iters, epochs=epochs, batch=batch, hidden_dim=hidden_dim, layers=layers, lr=lr,
                         gamma=gamma, eta=eta, w_sup=w_sup, w_mom=w_mom, d_thresh=d_thresh, **kw)
        self.iters, self.epochs, self.batch = int(iters), epochs, int(batch)
        self.hidden_dim, self.layers, self.lr = int(hidden_dim), int(layers), float(lr)
        self.gamma, self.eta, self.w_sup, self.w_mom, self.d_thresh = (float(gamma), float(eta), float(w_sup),
                                                                       float(w_mom), float(d_thresh))
        if self.layers < 2:
            raise ValueError("TimeGAN needs layers >= 2 (the supervisor has layers - 1)")
        self.nets = None

    def fit(self, data: FitData, seed: int) -> dict:
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        g = torch.Generator().manual_seed(seed)
        tr, _ = data.split_windows()
        if len(tr) < self.batch:
            raise ValueError(f"TimeGAN: {len(tr)} training windows < batch {self.batch}")
        Wtr = data.windows()[tr]
        self.scaler = MinMaxScaler().fit(Wtr)
        X = torch.as_tensor(self.scaler.transform(Wtr), dtype=torch.float32).to(DEV)
        N, H, L, B = data.n_assets, self.hidden_dim, self.layers, self.batch
        self.n_assets = N
        # "epochs" (smoke tests): iterations per phase = epochs * batches per pass over the training windows
        iters = self.iters if self.epochs is None else int(self.epochs) * math.ceil(len(tr) / B)
        E, R = _RNN(N, H, H, L), _RNN(H, H, N, L)
        G, S, D = _RNN(N, H, H, L), _RNN(H, H, H, L - 1), _RNN(H, H, 1, L, sigmoid=False)
        self.nets = nn.ModuleDict(dict(E=E, R=R, G=G, S=S, D=D)).to(DEV)
        params = lambda *ms: [p for m in ms for p in m.parameters()]
        oER = torch.optim.Adam(params(E, R), lr=self.lr)
        oS = torch.optim.Adam(params(S), lr=self.lr)       # phase 2 (the generator has no gradient there)
        oGS = torch.optim.Adam(params(G, S), lr=self.lr)
        oD = torch.optim.Adam(params(D), lr=self.lr)
        bce = F.binary_cross_entropy_with_logits
        ones, zeros = torch.ones(B, P.T, 1, device=DEV), torch.zeros(B, P.T, 1, device=DEV)
        batch_x = lambda: X[torch.from_numpy(rng.integers(0, len(X), B)).to(DEV)]
        noise = lambda: torch.rand(B, P.T, N, generator=g).to(DEV)
        sup = lambda h: F.mse_loss(S(h)[:, :-1], h[:, 1:])
        every = max(1, iters // 200)
        hist = dict(phase1=[], phase2=[], phase3=[])
        times = {}

        def check(loss, phase, it):
            if not torch.isfinite(loss):
                raise FloatingPointError(f"TimeGAN: non-finite loss in phase {phase}, iteration {it}")

        def step(opt, loss):
            opt.zero_grad()
            loss.backward()
            opt.step()

        t0 = time.time()
        for it in range(iters):                                   # phase 1: autoencoding
            x = batch_x()
            loss = self.eta * torch.sqrt(F.mse_loss(R(E(x)), x))
            check(loss, 1, it)
            step(oER, loss)
            if it % every == 0 or it == iters - 1:
                hist["phase1"].append(dict(iter=it, e_loss=float(loss)))
        times["phase1"] = time.time() - t0

        t0 = time.time()
        for it in range(iters):                                   # phase 2: supervised loss only
            with torch.no_grad():
                h = E(batch_x())
            loss = sup(h)
            check(loss, 2, it)
            step(oS, loss)
            if it % every == 0 or it == iters - 1:
                hist["phase2"].append(dict(iter=it, s_loss=float(loss)))
        times["phase2"] = time.time() - t0

        t0 = time.time()
        d_updates = 0
        for it in range(iters):                                   # phase 3: joint training
            for _ in range(2):
                x, z = batch_x(), noise()
                h = E(x).detach()                                 # only G and S are updated here
                e_hat = G(z)
                h_hat = S(e_hat)
                g_u = bce(D(h_hat), ones)
                g_ue = bce(D(e_hat), ones)
                g_s = sup(h)
                g_v = _moment_loss(R(h_hat), x)
                g_loss = g_u + self.gamma * g_ue + self.w_sup * torch.sqrt(g_s) + self.w_mom * g_v
                check(g_loss, 3, it)
                step(oGS, g_loss)
                h = E(x)
                e_rec = F.mse_loss(R(h), x)
                e_loss = self.eta * torch.sqrt(e_rec) + 0.1 * sup(h)
                check(e_loss, 3, it)
                step(oER, e_loss)
            x, z = batch_x(), noise()
            with torch.no_grad():
                h, e_hat = E(x), G(z)
                h_hat = S(e_hat)
            d_loss = bce(D(h), ones) + bce(D(h_hat), zeros) + self.gamma * bce(D(e_hat), zeros)
            check(d_loss, 3, it)
            if float(d_loss) > self.d_thresh:
                step(oD, d_loss)
                d_updates += 1
            if it % every == 0 or it == iters - 1:
                hist["phase3"].append(dict(iter=it, d_loss=float(d_loss), g_adv=float(g_u), g_adv_e=float(g_ue),
                                           g_sup=float(g_s), g_mom=float(g_v), e_rec=float(e_rec)))
        times["phase3"] = time.time() - t0
        self.nets.eval()
        return dict(n_train=int(len(tr)), iters_per_phase=iters, d_updates=d_updates,
                    sec_per_iter={k: v / max(iters, 1) for k, v in times.items()},
                    n_params=sum(p.numel() for p in self.nets.parameters()), history=hist)

    @torch.no_grad()
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.nets is None:
            raise RuntimeError("TimeGAN.sample called before fit")
        out = []
        for i in range(0, n, 512):
            z = torch.as_tensor(rng.random((min(512, n - i), P.T, self.n_assets)), dtype=torch.float32).to(DEV)
            out.append(self.nets["R"](self.nets["S"](self.nets["G"](z))).cpu().numpy())
        G = self.scaler.inverse(np.concatenate(out))
        if not np.isfinite(G).all():
            raise FloatingPointError("TimeGAN produced non-finite returns")
        return G

    def save(self, path) -> None:
        path = Path(str(path) + ".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state_dict=self.nets.state_dict(), lo=self.scaler.lo, hi=self.scaler.hi,
                        hparams=self.hparams), path)
