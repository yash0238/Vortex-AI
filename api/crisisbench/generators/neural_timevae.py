"""TimeVAE baseline (Desai, Freeman, Wang, Beaver 2021, "TimeVAE: A Variational Auto-Encoder
for Multivariate Time Series Generation"), BASE architecture, own PyTorch reimplementation.

Model (base TimeVAE: no trend / seasonality blocks)
  encoder   Conv1D(k=3, stride 2, ReLU) x len(hidden) -> flatten -> dense z_mean, z_log_var
  decoder   dense(ReLU) -> reshape -> ConvTranspose1D(k=3, stride 2, ReLU) x len(hidden)
            (the last one maps to N channels) -> flatten -> dense(T * N) -> reshape (T, N)
  loss      recon_wt * [ SSE(x, x_hat) + SSE(mean_over_assets(x), mean_over_assets(x_hat)) ] + KL,
            summed over the batch.  recon_wt = 3, latent_dim = 8, hidden = (50, 100, 200) are the
            defaults of the authors' released code as recalled (not re-checked against the source
            in this project); the second SSE term is the extra term of that code's reconstruction
            loss and can be switched off with axis_term=False.
  sample    z ~ N(0, I) -> decoder mean -> inverse scaling.  As in the paper NO observation noise
            is added to the decoder mean, so generated windows are smoother than real ones.

Deviations from the reference
  * Inputs are per-asset z-scored (scaler from the training days, i.e. days with val_mask False);
    the paper min-max scales.  Required by the benchmark brief.
  * Early stopping on the validation ELBO of the purged validation windows (data.split_windows),
    best checkpoint restored; the reference code stops on the training loss.
  * "same" padding: PyTorch pads symmetrically (padding=1) where TensorFlow pads on the right
    only for even lengths.  Sequence lengths are identical (60 -> 30 -> 15 -> 8 -> 16 -> 32 -> 64).
  * Adam lr 1e-3, batch 32, no learning-rate schedule.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .. import protocol as P
from .base import FitData, Generator


class _Net(nn.Module):
    def __init__(self, T: int, N: int, hidden: tuple, latent: int):
        super().__init__()
        self.T, self.N, self.latent = T, N, latent
        enc, c, L = [], N, T
        for h in hidden:
            enc += [nn.Conv1d(c, h, 3, stride=2, padding=1), nn.ReLU()]
            c, L = h, (L - 1) // 2 + 1
        self.enc = nn.Sequential(*enc)
        self.c_last, self.L_last = c, L
        self.z_mean = nn.Linear(L * c, latent)
        self.z_logvar = nn.Linear(L * c, latent)
        self.dec_in = nn.Linear(latent, L * c)
        dec = []
        for h in list(reversed(hidden[:-1])) + [N]:
            dec += [nn.ConvTranspose1d(c, h, 3, stride=2, padding=1, output_padding=1), nn.ReLU()]
            c, L = h, 2 * L
        self.dec = nn.Sequential(*dec)
        self.dec_out = nn.Linear(L * N, T * N)

    def encode(self, x):                       # x (B, T, N)
        h = self.enc(x.transpose(1, 2)).flatten(1)
        return self.z_mean(h), self.z_logvar(h)

    def decode(self, z):
        h = torch.relu(self.dec_in(z)).view(-1, self.c_last, self.L_last)
        return self.dec_out(self.dec(h).flatten(1)).view(-1, self.T, self.N)


def _loss(net: _Net, x, eps, recon_wt: float, axis_term: bool):
    """Negative ELBO (up to the constant of the Gaussian likelihood), SUMMED over the batch."""
    mu, logvar = net.encode(x)
    xh = net.decode(mu + torch.exp(0.5 * logvar) * eps)
    rec = ((x - xh) ** 2).sum()
    if axis_term:
        rec = rec + ((x.mean(2) - xh.mean(2)) ** 2).sum()
    kl = -0.5 * (1 + logvar - mu ** 2 - torch.exp(logvar)).sum()
    return recon_wt * rec + kl, rec, kl


class TimeVAE(Generator):
    family = "neural"
    impl_source = "own PyTorch reimplementation"

    def __init__(self, epochs: int = 100, patience: int = 15, batch: int = 32, lr: float = 1e-3,
                 latent_dim: int = 8, hidden: tuple = (50, 100, 200), recon_wt: float = 3.0,
                 axis_term: bool = True, **kw):
        hidden = tuple(int(h) for h in hidden)
        super().__init__(epochs=epochs, patience=patience, batch=batch, lr=lr, latent_dim=latent_dim,
                         hidden=list(hidden), recon_wt=recon_wt, axis_term=axis_term, **kw)
        self.epochs, self.patience, self.batch, self.lr = int(epochs), int(patience), int(batch), float(lr)
        self.latent_dim, self.hidden = int(latent_dim), hidden
        self.recon_wt, self.axis_term = float(recon_wt), bool(axis_term)
        self.net = None

    def fit(self, data: FitData, seed: int) -> dict:
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        train_days = data.returns[~data.val_mask]
        self.mu = train_days.mean(0)
        self.sd = train_days.std(0, ddof=1)
        if not (np.isfinite(self.sd).all() and (self.sd > 0).all()):
            raise ValueError("TimeVAE: an asset has zero or non-finite training variance")
        W = data.windows()
        tr, va = data.split_windows()
        if len(tr) == 0 or len(va) == 0:
            raise ValueError(f"TimeVAE needs training and validation windows (got {len(tr)}, {len(va)})")
        Z = torch.as_tensor((W - self.mu) / self.sd, dtype=torch.float32)
        Xtr, Xva = Z[tr], Z[va]
        self.net = _Net(P.T, data.n_assets, self.hidden, self.latent_dim)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        g = torch.Generator().manual_seed(seed)
        # fixed validation noise: the same eps every epoch, so the stopping curve is comparable
        eps_va = torch.randn(len(Xva), self.latent_dim, generator=torch.Generator().manual_seed(seed + 1))
        hist, best, best_state, best_ep, bad = [], np.inf, None, -1, 0
        t0 = time.time()
        for ep in range(self.epochs):
            self.net.train()
            perm = rng.permutation(len(Xtr))
            tot = 0.0
            for i in range(0, len(perm), self.batch):
                x = Xtr[perm[i:i + self.batch]]
                loss, _, _ = _loss(self.net, x, torch.randn(len(x), self.latent_dim, generator=g),
                                   self.recon_wt, self.axis_term)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"TimeVAE: non-finite training loss at epoch {ep}")
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss)
            self.net.eval()
            with torch.no_grad():
                lv, rv, kv = _loss(self.net, Xva, eps_va, self.recon_wt, self.axis_term)
            val = float(lv) / len(Xva)
            if not np.isfinite(val):
                raise FloatingPointError(f"TimeVAE: non-finite validation loss at epoch {ep}")
            hist.append(dict(epoch=ep, train=tot / len(Xtr), val=val, val_rec=float(rv) / len(Xva),
                             val_kl=float(kv) / len(Xva)))
            if val < best:
                best, best_state, best_ep, bad = val, copy.deepcopy(self.net.state_dict()), ep, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        return dict(n_train=int(len(tr)), n_val=int(len(va)), epochs_run=len(hist), best_epoch=best_ep,
                    best_val_neg_elbo=best, sec_per_epoch=(time.time() - t0) / len(hist),
                    n_params=sum(p.numel() for p in self.net.parameters()), history=hist)

    @torch.no_grad()
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("TimeVAE.sample called before fit")
        z = torch.as_tensor(rng.standard_normal((n, self.latent_dim)), dtype=torch.float32)
        out = [self.net.decode(z[i:i + 512]).numpy().astype(np.float64) for i in range(0, n, 512)]
        G = np.concatenate(out) * self.sd + self.mu
        if not np.isfinite(G).all():
            raise FloatingPointError("TimeVAE produced non-finite returns")
        return G

    def save(self, path) -> None:
        path = Path(str(path) + ".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state_dict=self.net.state_dict(), mu=self.mu, sd=self.sd, hparams=self.hparams), path)
