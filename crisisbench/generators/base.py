"""Generator interface. Every model in the benchmark implements this and nothing else.

Contract
  fit(data, seed)   estimate on data.returns (a contiguous (D, N) block of daily log
                    returns in RAW units).  Neural models must hold out the days where
                    data.val_mask is True for early stopping only (see data.split_windows).
                    Returns a dict of free-form training info (JSON-serialisable).
  sample(n, rng)    return (n, T, N) float array of RAW log returns, T = protocol.T.
                    UNCONDITIONAL: no real window, label or date may be passed in at
                    sampling time.  Anything learned from the fit data (including stored
                    training windows) is fair; test data is never visible.
  sample_regime(n, regime, rng)   optional; regime 1 = crisis, 0 = normal.  Only for
                    generators with an explicit regime handle.  Default: not supported.

Windows are scored as exchangeable samples, so a generator that simulates one long
path should cut it into NON-overlapping windows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import protocol as P


@dataclass
class FitData:
    market: str
    returns: np.ndarray               # (D, N) raw daily log returns of the fit period
    dates: pd.DatetimeIndex           # (D,)
    tickers: list[str]
    val_mask: np.ndarray              # (D,) bool: validation days (early stopping only)
    volume: np.ndarray | None = None  # (D, N)
    rule: dict = field(default_factory=dict)   # frozen crisis rule (labels.fit_rule on this period)
    fit_name: str = "full"

    @property
    def n_assets(self) -> int:
        return self.returns.shape[1]

    def windows(self) -> np.ndarray:
        """All overlapping (T, N) windows of the fit period; window i ends on day i + T - 1."""
        v = np.lib.stride_tricks.sliding_window_view(self.returns, P.T, axis=0)
        return np.ascontiguousarray(v.transpose(0, 2, 1))

    def split_windows(self) -> tuple[np.ndarray, np.ndarray]:
        """(train_idx, val_idx) into self.windows().

        A window is validation if it lies entirely inside the validation block, and
        training if it contains no validation day (this purges every window that
        straddles a block edge).
        """
        v = self.val_mask.astype(int)
        c = np.concatenate([[0], np.cumsum(v)])
        inwin = c[P.T:] - c[:-P.T]          # validation days inside each window
        return np.where(inwin == 0)[0], np.where(inwin == P.T)[0]

    def window_labels(self) -> np.ndarray:
        from ..labels import label_windows
        return label_windows(self.windows(), self.rule)[0]


class Generator:
    name = "base"
    family = "static"
    impl_source = "own"
    supports_regime = False

    def __init__(self, **hparams):
        self.hparams = hparams

    def fit(self, data: FitData, seed: int) -> dict:
        raise NotImplementedError

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def sample_regime(self, n: int, regime: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError(f"{self.name} has no regime handle")

    # Optional persistence (neural models override to save weights).
    def save(self, path) -> None:
        pass


def chop(path: np.ndarray, n: int) -> np.ndarray:
    """Cut a (n*T, N) path into n non-overlapping (T, N) windows."""
    return np.ascontiguousarray(path[: n * P.T].reshape(n, P.T, path.shape[1]))
