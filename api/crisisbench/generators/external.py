"""Generators whose samples were produced outside this repository by OFFICIAL code (no reimplementation).

DiffusionTS   Diffusion-TS (Yuan & Qiao, ICLR 2024), official repository github.com/Y-debug-sys/Diffusion-TS, trained
              on a GPU by cloud/dts_driver.py with the authors' energy.yaml configuration (seq_length 60, 47 features).
              Samples live in artifacts/external/diffusion_ts/<market>/<fit>/seed<k>.npy; `fit` only checks that the
              file for this (market, fit, seed) exists, `sample` returns windows from it without replacement.
The benchmark scores them with the unchanged metric code, like any other generator.
"""
from __future__ import annotations

import json

import numpy as np

from .base import FitData, Generator


class Precomputed(Generator):
    family = "neural"
    impl_source = "official code"
    folder = ""

    def fit(self, data: FitData, seed: int) -> dict:
        from ..bench import ARTIFACTS
        d = ARTIFACTS / "external" / self.folder / data.market / data.fit_name
        f = d / f"seed{seed}.npy"
        if not f.exists():
            raise FileNotFoundError(f"no official-code samples at {f}")
        self.G = np.load(f).astype(np.float64)
        if self.G.shape[1:] != (60, data.n_assets) or not np.isfinite(self.G).all():
            raise ValueError(f"bad sample array {self.G.shape}")
        meta = d / f"seed{seed}.json"
        return dict(n_available=int(len(self.G)), **(json.loads(meta.read_text()) if meta.exists() else {}))

    def sample(self, n, rng):
        # one shuffled order per run, served in sequence, so repeated calls (the risk pool asks in batches of 2000)
        # never return the same window twice
        if not hasattr(self, "_order"):
            self._order, self._pos = rng.permutation(len(self.G)), 0
        if self._pos + n > len(self.G):
            raise ValueError(f"asked for {self._pos + n} windows in total, only {len(self.G)} were generated")
        idx = self._order[self._pos:self._pos + n]
        self._pos += n
        return self.G[idx]


class DiffusionTS(Precomputed):
    folder = "diffusion_ts"
