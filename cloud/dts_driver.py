"""Run the OFFICIAL Diffusion-TS code (Yuan & Qiao, ICLR 2024; github.com/Y-debug-sys/Diffusion-TS) on CrisisBench.

    python cloud/dts_driver.py --repo /path/to/Diffusion-TS [--steps 25000] [--smoke]

Uses the authors' own dataset, model and trainer classes and one of their published configurations, changing only
seq_length (60) and feature_size (47) to our windows: Config/stocks.yaml (their setting for stock data: 2+2 layers,
d_model 64, 500 diffusion steps, 10,000 training steps; the default, chosen for the available GPU time) or
Config/energy.yaml (their largest real dataset: 4+3 layers, d_model 96, 1000 steps, 25,000 training steps).
Input: the per-asset z-scored daily returns of each fit period (all fit-period days, unconditional generation, as
the authors do with training_ratio 1).  The authors' MinMax scaling to [-1, 1] and its inverse are applied by their
dataset class.  Output: artifacts/external/diffusion_ts/nse/<fit>/seed<k>.npy, (n, 60, 47) daily log returns,
2000 windows per main run and 22000 (2000 main + 20000 pool) for the runs whose risk pool the VaR backtest needs.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

if not hasattr(np, "row_stack"):            # the official sampler uses np.row_stack, removed in newer NumPy
    np.row_stack = np.vstack

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
OUT = HERE / "artifacts" / "external" / "diffusion_ts" / "nse"

JOBS = ([("full", k, 2000) for k in range(5)] + [("pre2020", 0, 22000)] + [("pre2020", k, 2000) for k in range(1, 5)]
        + [(f, 0, 22000) for f in ("upto2020", "upto2021", "upto2022", "upto2023")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", default="stocks", choices=["stocks", "energy"],
                    help="authors' config: stocks (their stock-data setting, 10k steps) or energy (largest, 25k steps)")
    ap.add_argument("--steps", type=int, default=None, help="override the config's training steps")
    ap.add_argument("--sampling-steps", type=int, default=None, help="override the config's sampling steps")
    ap.add_argument("--only", default="", help="comma list fit:seed, e.g. full:0,pre2020:0")
    ap.add_argument("--no-pool", action="store_true", help="2000 windows only (no 20000-window risk pool)")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU pipeline test, one job")
    ap.add_argument("--hours", type=float, default=11.0)
    ap.add_argument("--zip", default=str(HERE.parent / "diffusion_ts_results.zip"))
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import torch
    from Utils.io_utils import load_yaml_config, instantiate_from_config
    from Data.build_dataloader import build_dataloader
    from engine.solver import Trainer
    from crisisbench.bench import make_fit_data

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    jobs = JOBS[:1] if a.smoke else JOBS
    if a.only:
        want = {(x.split(":")[0], int(x.split(":")[1])) for x in a.only.split(",")}
        jobs = [j for j in jobs if (j[0], j[1]) in want]
    if a.no_pool:
        jobs = [(f_, k, 2000) for f_, k, _ in jobs]
    print(f"device {dev}; config {a.config}; jobs {jobs}", flush=True)
    for fit, seed, n in jobs:
        # the smoke test writes to its own folder so it can never be mistaken for (or block) a real run
        f = (OUT.parent / "_smoke" if a.smoke else OUT) / fit / f"seed{seed}.npy"
        if f.exists() and not a.smoke:
            print("skip", fit, seed, flush=True)
            continue
        if (time.time() - t0) / 3600 > a.hours:
            print("time budget reached", flush=True)
            break
        t1 = time.time()
        fd, _, _ = make_fit_data("nse", fit)
        R = np.asarray(fd.returns, dtype=np.float64)
        mu, sd = R.mean(0), R.std(0)
        csv = repo / "Data" / "datasets" / f"crisisbench_{fit}.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(csv, (R - mu) / sd, delimiter=",", header=",".join(f"a{i}" for i in range(R.shape[1])), comments="")
        cfg = load_yaml_config(str(repo / "Config" / f"{a.config}.yaml"))
        m = cfg["model"]["params"]
        m["seq_length"], m["feature_size"] = 60, R.shape[1]
        if a.smoke:
            m["timesteps"] = m["sampling_timesteps"] = 20
        elif a.sampling_steps:
            m["sampling_timesteps"] = min(a.sampling_steps, m["timesteps"])
        s = cfg["solver"]
        s["max_epochs"] = 30 if a.smoke else (a.steps or s["max_epochs"])
        s["save_cycle"] = max(1, s["max_epochs"] // 2)
        s["results_folder"] = str(repo / f"Checkpoints_cb_{fit}_s{seed}")
        for part in ("train_dataset", "test_dataset"):
            p = cfg["dataloader"][part]["params"]
            p["data_root"], p["window"], p["name"] = str(csv), 60, f"cb_{fit}"
        cfg["dataloader"]["train_dataset"]["params"]["proportion"] = 1.0
        torch.manual_seed(seed); np.random.seed(seed)
        args = argparse.Namespace(save_dir=str(repo / "OUTPUT" / f"cb_{fit}_s{seed}"), name=f"cb_{fit}_s{seed}")
        os.makedirs(args.save_dir, exist_ok=True)
        model = instantiate_from_config(cfg["model"]).to(dev)
        dl = build_dataloader(cfg, args)
        trainer = Trainer(config=cfg, args=args, model=model, dataloader=dl, logger=None)
        trainer.train()
        t2 = time.time()
        nn = 64 if a.smoke else n
        samp = trainer.sample(num=nn, size_every=min(nn, 1000), shape=[60, R.shape[1]])[:nn]
        z = dl["dataset"].unnormalize(samp)                       # authors' inverse: [-1,1] -> [0,1] -> MinMax inverse
        G = z * sd + mu
        if not np.isfinite(G).all():
            raise FloatingPointError("non-finite samples")
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, G.astype(np.float32))
        (f.parent / f"seed{seed}.json").write_text(
            f'{{"train_minutes": {(t2 - t1) / 60:.2f}, "sample_minutes": {(time.time() - t2) / 60:.2f}, "steps": {s["max_epochs"]}, '
            f'"sampling_timesteps": {m["sampling_timesteps"]}, "config": "{a.config}", "device": "{dev}", "std_ratio": {float(G.std() / R.std()):.4f}}}')
        print(f"{fit} seed{seed}: train {(t2 - t1) / 60:.1f} min, sample {(time.time() - t2) / 60:.1f} min, "
              f"std ratio {G.std() / R.std():.3f}", flush=True)
        if not a.smoke:
            with zipfile.ZipFile(a.zip, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in OUT.rglob("*"):
                    if p.is_file():
                        zf.write(p, p.relative_to(HERE))
    print("done", flush=True)


if __name__ == "__main__":
    main()
