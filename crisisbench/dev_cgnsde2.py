"""Development check for CG-NSDE v2.  Uses ONLY the 2010-2019 fit period ('pre2020' fit): the model is
fitted on its training windows and its samples are compared with the real windows of the same period.
The 2020-2024 held-out period is never touched here.

    python -m crisisbench.dev_cgnsde2 --tag base --epochs 12 [--hparams JSON] [--variant NAME] [--seed 0]
Appends one line to results/dev_cgnsde2.jsonl.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from . import metrics as M
from .bench import RESULTS, make_fit_data
from .generators import get

KEYS = ["crisis_freq", "boost_pct", "cvar_ratio", "tail_coexc", "pc1_share", "pc1_share_gap", "connect_gap",
        "kurt_med", "acf1_sq", "acf1_ret", "vol_ann", "corr_fro", "corr_mae", "eig_dist"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gen", default="cgnsde2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--hparams", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    fd, _, _ = make_fit_data("nse", "pre2020")
    W = fd.windows()
    real = M.aggregate(M.window_stats(W, fd.rule))
    hp = {}                                   # "key=value,key=value" (values parsed as JSON, else kept as text)
    for kv in (a.hparams.split(",") if a.hparams else []):
        k, v = kv.split("=", 1)
        try:
            hp[k] = json.loads(v)
        except json.JSONDecodeError:
            hp[k] = v
    g = get(a.gen, verbose=True, **hp) if a.gen.startswith("cgnsde") else get(a.gen, **hp)
    t0 = time.time()
    info = g.fit(fd, a.seed)
    minutes = (time.time() - t0) / 60
    modes = ["mt", "fhs"] if a.gen.startswith("cgnsde2") else [None]
    for mode in modes:
        if mode:
            g.hparams["innovations"] = mode
            print(f"--- innovations = {mode}", flush=True)
        report(a, g, fd, W, real, info, minutes, hp, mode)


def report(a, g, fd, W, real, info, minutes, hp, mode):
    G = g.sample(a.n, np.random.default_rng(1000 + a.seed))
    ga = M.aggregate(M.window_stats(G, fd.rule))
    cmp_ = M.compare(ga, real)
    rows = {}
    for k in KEYS:
        r, v = real.get(k), cmp_.get(k, ga.get(k))
        if v is None and r is None:
            continue
        f = lambda q: None if q is None or isinstance(q, (list, dict, np.ndarray)) else round(float(q), 4)
        rows[k] = dict(real=f(r), gen=f(v))
        print(f"{k:14s} real={rows[k]['real']}  gen={rows[k]['gen']}", flush=True)
    sd_ratio = float(G.std() / W.std())
    xs_ratio = float(G.std(2).mean() / W.std(2).mean())
    print(f"overall std gen/real = {sd_ratio:.3f}   cross-sectional dispersion gen/real = {xs_ratio:.3f}", flush=True)
    rows["std_ratio"], rows["xs_ratio"] = round(sd_ratio, 3), round(xs_ratio, 3)
    keep = {k: v for k, v in (info or {}).items() if k != "history"}
    print("info:", keep, flush=True)
    with open(RESULTS / "dev_cgnsde2.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(tag=a.tag, innovations=mode, gen=a.gen, seed=a.seed, hparams=hp, minutes=round(minutes, 1),
                                 metrics=rows, info=keep), default=str) + "\n")


if __name__ == "__main__":
    main()
