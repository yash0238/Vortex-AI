"""Generator registry. `get(name)` returns a fresh, unfitted generator."""
from __future__ import annotations

import importlib

# name -> (module, class).  Modules are imported lazily so a missing optional
# dependency only breaks the generator that needs it.
REGISTRY = {
    "iid_gaussian": ("classical_static", "IIDGaussian"),
    "t_ppca": ("classical_static", "TPPCA"),
    "stationary_bootstrap": ("classical_static", "StationaryBootstrap"),
    "regime_gaussian": ("classical_regime", "RegimeGaussian"),
    "hmm_gaussian": ("classical_regime", "HMMGaussian"),
    "garch_fhs": ("classical_garch", "GarchFHS"),
    "dcc_garch": ("classical_garch", "DCCGarch"),
    "timevae": ("neural_timevae", "TimeVAE"),
    "quantgan": ("neural_quantgan", "QuantGAN"),
    "timegan": ("neural_timegan", "TimeGAN"),
    "diffusion": ("neural_diffusion", "Diffusion"),
    "cgnsde": ("cgnsde", "CGNSDE"),
    "cgnsde2": ("cgnsde2", "CGNSDE2"),
    "cgnsde3": ("cgnsde3", "CGNSDE3"),
    "gjr_evt_tcopula": ("classical_strong", "GjrEvtTCopula"),
    "adcc_gjr": ("classical_strong", "AdccGjr"),
    "rs_tcopula": ("classical_strong", "RsTCopula"),
    "diffusion_ts": ("external", "DiffusionTS"),
}


def get(name: str, **hparams):
    base = name.split("@")[0]            # "cgnsde@no_gat" -> variant of cgnsde
    mod, cls = REGISTRY[base]
    klass = getattr(importlib.import_module(f"crisisbench.generators.{mod}"), cls)
    if "@" in name:
        hparams = {**klass.VARIANTS[name.split("@")[1]], **hparams}
    g = klass(**hparams)
    g.name = name
    return g
