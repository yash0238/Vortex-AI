"""Full-scale neural baselines on a GPU (Kaggle / Colab).  Runs the UNCHANGED benchmark runner with larger
training budgets and writes ordinary run folders under a separate generator label (`<gen>_fs`), so the
laptop-scale rows stay in the tables next to them.

    CRISISBENCH_DEVICE=cuda python -m crisisbench.fullscale --hours 11

Resumable: finished runs are skipped.  After every run the results are zipped to `fullscale_results.zip`
(next to the project folder, or --out), so an interrupted session still leaves usable results.

Budgets (iterations are optimiser steps of the generator)
  timegan    50,000 iterations per phase (the default of the authors' released code as recalled), batch 128
  quantgan   10,000 generator updates x 5 critic updates, batch 64  (no canonical budget is published)
  diffusion  30,000 iterations, batch 128, DDIM 100 sampling steps
Order: main table first (seed-major, so a partial session covers every model), then the yearly refits that
the VaR/ES backtest needs (seed 0).
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
import zipfile
from pathlib import Path

from . import protocol as P
from .bench import ARTIFACTS, RESULTS, run
from .data import ROOT

BUDGET = {
    "timegan": dict(iters=50_000),
    "quantgan": dict(iters=10_000),
    "diffusion": dict(iters=30_000),
}


def jobs(gens):
    out = [("nse", g, f, k, f == "pre2020" and k == 0) for k in P.SEEDS for g in gens for f in ("full", "pre2020")]
    out += [("nse", g, f, 0, True) for f in ("upto2020", "upto2021", "upto2022", "upto2023") for g in gens]
    return out


def pack(out: Path) -> None:
    tmp = out.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for base in (RESULTS / "runs", ARTIFACTS / "risk_pool", ARTIFACTS / "samples"):
            for p in base.glob("*/*_fs/**/*") if base.exists() else []:
                if p.is_file() and p.name != ".claim":
                    z.write(p, p.relative_to(ROOT))
    shutil.move(tmp, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=11.0, help="do not START a run after this many hours")
    ap.add_argument("--gens", default="diffusion,quantgan,timegan")
    ap.add_argument("--out", default=str(ROOT.parent / "fullscale_results.zip"))
    ap.add_argument("--smoke", action="store_true", help="tiny budgets, one job per generator (pipeline test)")
    a = ap.parse_args()
    gens = a.gens.split(",")
    t0 = time.time()
    todo = jobs(gens)
    if a.smoke:
        todo = [(m, g, "full", 0, False) for m, g, f, k, r in todo if f == "full" and k == 0]
    print(f"device = {os.environ.get('CRISISBENCH_DEVICE', 'cpu')}; {len(todo)} jobs", flush=True)
    for i, (market, g, fit, seed, risk) in enumerate(todo):
        if (time.time() - t0) / 3600 > a.hours:
            print("time budget reached; stopping before the next run", flush=True)
            break
        hp = dict(iters=20) if a.smoke else BUDGET[g]
        t1 = time.time()
        try:
            r = run(market, g, fit, seed, hparams=hp, tag=f"{g}_fs" + ("_smoke" if a.smoke else ""),
                    want_risk_pool=risk, n_gen=200 if a.smoke else P.N_GEN)
            fam = r["main"]["family"]
            print(f"[{i + 1}/{len(todo)}] {g} {fit} seed{seed}: {(time.time() - t1) / 60:.1f} min  "
                  f"scores {({k: round(v, 2) for k, v in fam.items()})}", flush=True)
        except SystemExit:
            print(f"[{i + 1}/{len(todo)}] {g} {fit} seed{seed}: skipped", flush=True)
        except Exception as e:                      # keep going: one failed run must not lose the session
            print(f"[{i + 1}/{len(todo)}] {g} {fit} seed{seed}: FAILED {type(e).__name__}: {e}", flush=True)
        if not a.smoke:
            pack(Path(a.out))
    print(f"done in {(time.time() - t0) / 3600:.2f} h; results in {a.out}", flush=True)


if __name__ == "__main__":
    main()
