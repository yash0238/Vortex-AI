"""Resumable experiment queue. Each job is one bench run in its own process.

    python -m crisisbench.queue --stage nse_main --workers 2
    python -m crisisbench.queue --stage all --workers 2
    python -m crisisbench.queue --list            (show stages and how many jobs are still to do)

A finished job (metrics.json present) is skipped, so the queue can be stopped and
restarted at any time. Failures are logged to results/queue_log.csv and do not stop
the queue; rerun the stage to retry them.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import protocol as P
from .data import ROOT

RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
CLASSICAL = ["iid_gaussian", "t_ppca", "stationary_bootstrap", "garch_fhs", "dcc_garch", "regime_gaussian",
             "hmm_gaussian"]
NEURAL = ["timevae", "quantgan", "timegan", "cgnsde"]
ALL = CLASSICAL + NEURAL
ABLATIONS = ["cgnsde@no_gat", "cgnsde@no_supcon", "cgnsde@no_graph_loss", "cgnsde@no_corr", "cgnsde@no_voldyn",
             "cgnsde@no_sde", "cgnsde@gauss", "cgnsde@K0", "cgnsde@K2", "cgnsde@K4", "cgnsde@K16",
             "cgnsde@lam_graph03"]
RISK_REFITS = ["upto2020", "upto2021", "upto2022", "upto2023"]


def J(market, gen, fit, seed, risk=False):
    return dict(market=market, gen=gen, fit=fit, seed=seed, risk=risk)


def stages() -> dict[str, list[dict]]:
    s: dict[str, list[dict]] = {}
    # cheap models first inside each stage so tables fill up early
    s["nse_main"] = [J("nse", g, f, k, risk=(f == "pre2020" and k == 0))
                     for g in ALL for k in P.SEEDS for f in P.FITS]
    s["nse_risk"] = [J("nse", g, f, 0, risk=True) for g in ALL for f in RISK_REFITS]
    s["nse_ablation"] = [J("nse", g, "full", k) for g in ABLATIONS for k in (0, 1, 2)]   # in-sample fit only (CPU budget)
    s["markets"] = ([J(m, g, "full", k) for m in ("us", "kr") for g in ALL for k in P.SEEDS]
                    + [J(m, g, "pre2020", k) for m in ("us", "kr") for g in ALL for k in (0, 1, 2)])
    s["nse_ext"] = [J("nse_ext", g, "upto2024", k) for g in ALL for k in (0, 1, 2)]
    s["contamination"] = [J(m, g, "full", k) for m in ("nse_c_tick", "nse_c_1pct", "nse_c_split")
                          for g in CLASSICAL for k in (0, 1, 2)]
    s["sim"] = [J(m, g, "full", 0) for m in ("sim_boost00", "sim_boost40", "sim_ccc")
                for g in ("iid_gaussian", "t_ppca", "garch_fhs", "dcc_garch", "regime_gaussian", "hmm_gaussian")]
    # CG-NSDE v2 (post hoc model, see generators/cgnsde2.py): same grid as v1; ablations 3 seeds, in-sample fit
    v2 = "cgnsde2"
    s["v2_main"] = [J("nse", v2, f, k, risk=(f == "pre2020" and k == 0)) for k in P.SEEDS for f in P.FITS]
    s["v2_risk"] = [J("nse", v2, f, 0, risk=True) for f in RISK_REFITS]
    s["v2_markets"] = ([J(m, v2, "full", k) for m in ("us", "kr") for k in P.SEEDS]
                       + [J(m, v2, "pre2020", k) for m in ("us", "kr") for k in (0, 1, 2)]
                       + [J("nse_ext", v2, "upto2024", k) for k in (0, 1, 2)])
    s["v2_ablation"] = [J("nse", g, "full", k) for g in ABLATIONS_V2 for k in (0, 1, 2)]
    return s


ABLATIONS_V2 = ["cgnsde2@" + v for v in ("no_gat", "no_supcon", "no_graph_loss", "no_sde", "scalar_vol", "no_voldyn",
                                         "no_leverage", "gauss", "param_innov", "beta01", "K1", "K4", "K16")]
# v1 ablations are dropped (v1 stopped at epoch 2, its ablations are uninformative); `results/skip_gens.txt`
# makes already-running queue processes skip them too.
ORDER = ["v2_main", "v2_risk", "v2_markets", "v2_ablation", "nse_main", "nse_risk", "markets", "nse_ext",
         "contamination", "sim"]


def done(j: dict) -> bool:
    m = RESULTS / "runs" / j["market"] / j["gen"] / j["fit"] / f"seed{j['seed']}" / "metrics.json"
    p = ARTIFACTS / "risk_pool" / j["market"] / j["gen"] / j["fit"] / f"seed{j['seed']}.npz"
    return m.exists() and (not j["risk"] or p.exists())


def run_job(j: dict, threads: int) -> tuple[dict, int, float, str]:
    cmd = [sys.executable, "-m", "crisisbench.bench", "--market", j["market"], "--gen", j["gen"],
           "--fit", j["fit"], "--seed", str(j["seed"])]
    if j["risk"]:
        cmd.append("--risk-pool")
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               PYTHONWARNINGS="ignore", PYTHONIOENCODING="utf-8")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = (r.stdout + "\n" + r.stderr).strip().splitlines()[-12:]
    return j, r.returncode, time.time() - t0, "\n".join(tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="run the job list backwards (second queue process)")
    ap.add_argument("--gens", default=None, help="comma list: restrict to these generators")
    a = ap.parse_args()
    S = stages()
    names = ORDER if a.stage == "all" else a.stage.split(",")
    if a.list:
        for n in ORDER:
            todo = sum(not done(j) for j in S[n])
            print(f"{n:14s} {len(S[n]) - todo:4d} done / {len(S[n]):4d}")
        return
    jobs = [j for n in names for j in S[n] if not done(j)]
    if a.reverse:
        jobs = jobs[::-1]
    if a.gens:
        keep = set(a.gens.split(","))
        jobs = [j for j in jobs if j["gen"] in keep]
    print(f"{len(jobs)} jobs to run with {a.workers} workers x {a.threads} threads", flush=True)
    if os.name == "nt":                     # keep Windows from sleeping while the queue runs (no setting is changed)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    log = RESULTS / "queue_log.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    new = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as fh, ThreadPoolExecutor(a.workers) as ex:
        w = csv.writer(fh)
        if new:
            w.writerow(["finished", "market", "gen", "fit", "seed", "risk", "returncode", "minutes", "tail"])
        n_ok = n_bad = 0
        for j, rc, sec, tail in ex.map(lambda j: run_job(j, a.threads), jobs):
            n_ok, n_bad = n_ok + (rc == 0), n_bad + (rc != 0)
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), j["market"], j["gen"], j["fit"], j["seed"], j["risk"],
                        rc, f"{sec / 60:.2f}", tail if rc else tail.splitlines()[-2] if tail else ""])
            fh.flush()
            print(f"[{n_ok + n_bad}/{len(jobs)}] {'ok ' if rc == 0 else 'FAIL'} {j['market']}/{j['gen']}/{j['fit']}/seed{j['seed']}"
                  f"  {sec / 60:.1f} min", flush=True)
            if rc:
                print("    " + tail.replace("\n", "\n    "), flush=True)
    print(f"finished: {n_ok} ok, {n_bad} failed", flush=True)


if __name__ == "__main__":
    main()
