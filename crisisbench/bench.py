"""Benchmark runner: fit one generator, sample, score, save.

    python -m crisisbench.bench --market nse --gen iid_gaussian --fit full --seed 0

Outputs (resumable: an existing metrics.json is skipped unless --force)
    results/runs/<market>/<gen>/<fit>/seed<k>/metrics.json
    artifacts/samples/<market>/<gen>/<fit>/seed<k>.npy        (float32, seed 0 only by default)
    artifacts/risk_pool/<market>/<gen>/<fit>/seed<k>.npz      (with --risk-pool)
    results/reference/<market>/<fit>.json                     (real values, CIs, ceilings)

Fit names
    full        fit and score on the whole sample
    pre2020     fit up to protocol.TRAIN_END, score on the purged post-TRAIN_END sample
    upto<YYYY>  fit up to <YYYY>-12-31 (risk-backtest refits; scored like pre2020 on later data)
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import protocol as P
from .data import load_panel, ROOT
from .generators import get
from .generators.base import FitData
from .labels import fit_rule
from . import metrics as M

RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"


def fit_end(fit: str) -> pd.Timestamp | None:
    if fit == "full":
        return None
    if fit == "pre2020":
        return pd.Timestamp(P.TRAIN_END)
    if fit.startswith("upto"):
        return pd.Timestamp(f"{fit[4:]}-12-31")
    raise ValueError(fit)


def make_fit_data(market: str, fit: str) -> tuple[FitData, np.ndarray, pd.DatetimeIndex]:
    """Returns (fit data, reference windows, reference window end dates)."""
    panel = load_panel(market)
    end = fit_end(fit)
    days = np.ones(len(panel.dates), bool) if end is None else np.asarray(panel.dates <= end)
    d = panel.dates[days]
    val = np.asarray((d >= P.VAL_START) & (d <= P.VAL_END))
    if end is not None and end < pd.Timestamp(P.VAL_END):
        raise ValueError("fit period must contain the validation block")
    fd = FitData(market, panel.returns[days], d, panel.tickers, val, panel.volume[days], fit_name=fit)
    fd.rule = fit_rule(fd.windows())
    W, wend = panel.windows(P.T)
    if end is None:
        ref_mask = np.ones(len(W), bool)
    else:                      # windows that lie entirely after the fit period
        first = np.searchsorted(panel.dates, end, side="right") + P.T - 1
        ref_mask = np.arange(len(W)) + P.T - 1 >= first
    return fd, W[ref_mask], wend[ref_mask]


def reference(market: str, fit: str, fd: FitData, Wref: np.ndarray, force: bool = False) -> dict:
    """Real statistics, bootstrap CIs and noise ceilings; cached on disk."""
    path = ARTIFACTS / "reference" / market / f"{fit}.pkl"
    if path.exists() and not force:
        return pickle.loads(path.read_bytes())
    ws = M.window_stats(Wref, fd.rule)
    ref = M.real_reference(ws)
    ref["alt"] = M.real_reference(ws, alt_labels=True)["metrics"]
    for mode in ("feat", "raw"):
        ref["metrics"][f"disc_{mode}"] = M.real_vs_real_disc(Wref, mode)
    ref["tail_q"] = ws.tail_q
    ref["n_windows"] = int(ws.n)
    ref["n_crisis"] = int(ws.labels.sum())
    ref["rule"] = fd.rule
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(ref))
    out = RESULTS / "reference" / market / f"{fit}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(market=market, fit=fit, n_windows=ref["n_windows"], n_crisis=ref["n_crisis"],
                                   rule=fd.rule, metrics=ref["metrics"], metrics_alt_rule=ref["alt"],
                                   mac_crisis=ref["full"]["mac_crisis"], mac_normal=ref["full"]["mac_normal"]),
                              indent=2, default=float))
    return ref


def score(G: np.ndarray, ref: dict, Wref: np.ndarray, rule: dict, seed: int, disc: bool = True) -> dict:
    """All metrics of generated windows G against the cached real reference."""
    assert G.ndim == 3 and G.shape[1] == P.T and G.shape[2] == Wref.shape[2], G.shape
    assert np.isfinite(G).all(), "generator produced non-finite returns"
    gs = M.window_stats(G, rule)
    out = {}
    for tag, alt in (("main", False), ("alt_rule", True)):
        vals = M.compare(M.aggregate(gs, alt_labels=alt), ref["full"] if not alt else M.aggregate(
            M.window_stats(Wref, rule), alt_labels=True))
        refm = dict(ref["metrics"] if not alt else ref["alt"])
        if tag == "main" and disc:
            for mode in ("feat", "raw"):
                vals[f"disc_{mode}"] = M.discriminator_accuracy(Wref, G, mode, seed=seed)
                refm[f"disc_{mode}"] = ref["metrics"][f"disc_{mode}"]
        errs = M.normalised_errors(vals, refm)
        out[tag] = dict(metrics=errs, family=M.family_scores(errs))
    out["n_gen"] = int(len(G))
    out["n_gen_crisis"] = int(gs.labels.sum())
    return out


def risk_pool(gen, n: int, rng: np.random.Generator, n_assets: int, batch: int = 2000) -> dict:
    """(state, next-day return) pairs per portfolio from generated windows.

    state = realised vol of the portfolio over days T-21..T-2 (20 days), outcome = day T-1.
    """
    w_d = np.random.default_rng(7).dirichlet(np.ones(n_assets))
    weights = {"equal_weight": np.full(n_assets, 1.0 / n_assets), "dirichlet_seed7": w_d}
    acc = {k: ([], []) for k in weights}
    done = 0
    while done < n:
        G = gen.sample(min(batch, n - done), rng)
        for k, w in weights.items():
            pr = G @ w                                        # (b, T)
            acc[k][0].append(pr[:, -21:-1].std(axis=1, ddof=1))
            acc[k][1].append(pr[:, -1])
        done += len(G)
    out = {}
    for k in weights:
        out[f"{k}__state"] = np.concatenate(acc[k][0]).astype(np.float32)
        out[f"{k}__next"] = np.concatenate(acc[k][1]).astype(np.float32)
        out[f"{k}__w"] = weights[k]
    return out


def _pid_alive(pid: int) -> bool:
    import os
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, pid)           # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    k.GetExitCodeProcess(h, ctypes.byref(code))
    k.CloseHandle(h)
    return code.value == 259                        # STILL_ACTIVE


def _claim(out_dir) -> bool:
    """Take the job so that two queue processes never run it at once. A claim whose
    process is gone, or that is older than 8 hours, is stale and is taken over."""
    import atexit, os, time
    out_dir.mkdir(parents=True, exist_ok=True)
    c = out_dir / ".claim"
    for _ in range(2):
        try:
            fd = os.open(c, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            atexit.register(lambda: c.unlink(missing_ok=True))
            return True
        except FileExistsError:
            try:
                pid = int(c.read_text().strip() or 0)
                fresh = time.time() - c.stat().st_mtime < 8 * 3600
            except (OSError, ValueError):
                pid, fresh = 0, False
            if pid and fresh and _pid_alive(pid):
                return False
            c.unlink(missing_ok=True)
    return False


def run(market: str, gen_name: str, fit: str, seed: int, force: bool = False, save_samples: bool | None = None,
        want_risk_pool: bool = False, n_gen: int = P.N_GEN, hparams: dict | None = None, tag: str | None = None) -> dict:
    label = tag or gen_name
    out_dir = RESULTS / "runs" / market / label / fit / f"seed{seed}"
    mpath = out_dir / "metrics.json"
    ppath = ARTIFACTS / "risk_pool" / market / label / fit / f"seed{seed}.npz"
    if mpath.exists() and not force and (not want_risk_pool or ppath.exists()):
        return json.loads(mpath.read_text())
    skip = RESULTS / "skip_gens.txt"        # one generator-name prefix per line: jobs that are no longer wanted
    key = f"{market}/{gen_name}/{fit}"      # a line matches as a prefix of the generator name or of market/gen/fit
    if skip.exists() and any(p.strip() and (gen_name.startswith(p.strip()) or key.startswith(p.strip()))
                             for p in skip.read_text().splitlines()):
        print(f"dropped by results/skip_gens.txt: {market}/{label}/{fit}/seed{seed}\nskipped", flush=True)
        raise SystemExit(0)
    if not _claim(out_dir):               # another queue process is running this job right now
        print(f"claimed elsewhere: {market}/{label}/{fit}/seed{seed}\nskipped", flush=True)
        raise SystemExit(0)
    fd, Wref, wend = make_fit_data(market, fit)
    ref = reference(market, fit, fd, Wref)
    gen = get(gen_name, **(hparams or {}))
    np.random.seed(seed)
    t0 = time.time()
    info = gen.fit(fd, seed) or {}
    fit_min = (time.time() - t0) / 60
    rng = np.random.default_rng(10_000 + seed)
    t0 = time.time()
    G = np.asarray(gen.sample(n_gen, rng), dtype=np.float64)
    samp_s = time.time() - t0
    res = score(G, ref, Wref, fd.rule, seed)
    res.update(market=market, generator=label, base_generator=gen_name, fit=fit, seed=seed,
               family_tag=gen.family, impl_source=gen.impl_source, train_minutes=fit_min, sample_seconds=samp_s,
               fit_days=int(len(fd.dates)), fit_first=str(fd.dates[0].date()), fit_last=str(fd.dates[-1].date()),
               ref_windows=int(len(Wref)), hparams=gen.hparams, info=info, protocol_version=P.PROTOCOL_VERSION)
    if gen.supports_regime:
        try:
            Gc = np.asarray(gen.sample_regime(500, 1, rng), dtype=np.float64)
            Gn = np.asarray(gen.sample_regime(500, 0, rng), dtype=np.float64)
            off = ~np.eye(Gc.shape[2], dtype=bool)
            mac = lambda A: float(np.mean([np.abs(np.corrcoef(a, rowvar=False)[off]).mean() for a in A]))
            mc, mn = mac(Gc), mac(Gn)
            res["conditional"] = dict(mac_crisis=mc, mac_normal=mn, boost_pct=100 * (mc - mn) / mn,
                                      note="labels REQUESTED from the generator, not derived from its output")
        except NotImplementedError:
            pass
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_samples if save_samples is not None else seed == 0:
        sp = ARTIFACTS / "samples" / market / label / fit / f"seed{seed}.npy"
        sp.parent.mkdir(parents=True, exist_ok=True)
        np.save(sp, G.astype(np.float32))
    try:
        gen.save(ARTIFACTS / "models" / market / label / fit / f"seed{seed}")
    except Exception as e:                       # persistence must never lose a finished run
        res["save_error"] = repr(e)
    if want_risk_pool:
        ppath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ppath, **risk_pool(gen, P.N_GEN_POOL_RISK, np.random.default_rng(20_000 + seed), fd.n_assets))
    mpath.write_text(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="nse")
    ap.add_argument("--gen", required=True)
    ap.add_argument("--fit", default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--risk-pool", action="store_true")
    ap.add_argument("--n-gen", type=int, default=P.N_GEN)
    ap.add_argument("--hparams", type=str, default=None, help="JSON dict of generator hyperparameters")
    ap.add_argument("--tag", type=str, default=None, help="output label (default: generator name)")
    a = ap.parse_args()
    r = run(a.market, a.gen, a.fit, a.seed, a.force, None, a.risk_pool, a.n_gen,
            json.loads(a.hparams) if a.hparams else None, a.tag)
    m = r["main"]
    print(f"{r['generator']} {a.market}/{a.fit}/seed{a.seed}  fit {r['train_minutes']:.1f} min  "
          f"crisis windows {r['n_gen_crisis']}/{r['n_gen']}")
    print("  family scores:", {k: round(v, 2) for k, v in m["family"].items()})
    print("  " + "  ".join(f"{k}={v['value']:.3g}" if v["value"] is not None else f"{k}=n/a"
                           for k, v in m["metrics"].items()))
