"""Unified Kaggle GPU/CPU runner for CrisisBench full-scale neural baselines and remaining queue jobs.

Supports sharded execution across multiple Kaggle GPU (2x T4) and CPU kernels so the entire
remaining workload (42 full-scale GPU jobs + 143 non-skipped queue jobs + 6 nse/full/seed0
artifact backfills) completes well within Kaggle session limits.

Usage inside Kaggle kernel:
    python -m cloud.kaggle_runner --mode gpu_shard_1 --workers 4 --hours 11.0 --out /kaggle/working/output_bundle.zip
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from crisisbench import fullscale as FS
from crisisbench import protocol as P
from crisisbench import queue as Q
from crisisbench.bench import ARTIFACTS, RESULTS, make_fit_data, reference, run as bench_run
from crisisbench.data import ROOT

GPU_CAPABLE_GENS = {"diffusion", "quantgan", "timegan"}


def load_skip_prefixes() -> list[str]:
    skip_file = RESULTS / "skip_gens.txt"
    if not skip_file.exists():
        return []
    return [line.strip() for line in skip_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_skipped(j: dict, skip_prefixes: list[str]) -> bool:
    gen = j["gen"]
    key = f"{j['market']}/{gen}/{j['fit']}"
    return any(gen.startswith(p) or key.startswith(p) for p in skip_prefixes)


def want_samples_for(market: str, label: str, fit: str, seed: int) -> bool:
    """Only nse full/pre2020 seed0 non-ablation runs are read by export_web.py."""
    return market == "nse" and fit in ("full", "pre2020") and int(seed) == 0 and "@" not in label


def job_is_done(spec: dict) -> bool:
    market = spec["market"]
    label = spec.get("tag") or spec["gen"]
    fit = spec["fit"]
    seed = int(spec["seed"])
    risk = bool(spec.get("risk", False))
    need_samples = bool(spec.get("save_samples", False))
    need_model = bool(spec.get("need_model", False))

    mpath = RESULTS / "runs" / market / label / fit / f"seed{seed}" / "metrics.json"
    ppath = ARTIFACTS / "risk_pool" / market / label / fit / f"seed{seed}.npz"
    spath = ARTIFACTS / "samples" / market / label / fit / f"seed{seed}.npy"
    mod_dir = ARTIFACTS / "models" / market / label / fit
    has_model = any(mod_dir.glob(f"seed{seed}*")) if mod_dir.exists() else False

    if not mpath.exists():
        return False
    if risk and not ppath.exists():
        return False
    if need_samples and not spath.exists():
        return False
    if need_model and not has_model:
        return False
    return True


def build_all_specs() -> dict[str, list[dict]]:
    """ Partition all remaining work into clean shards."""
    skip_prefixes = load_skip_prefixes()
    S = Q.stages()

    # 1. Fullscale GPU jobs (42 total: 14 diffusion, 14 quantgan, 14 timegan)
    fs_diff_quant = []
    fs_timegan = []
    for market, g, fit, seed, risk in FS.jobs(["diffusion", "quantgan", "timegan"]):
        tag = f"{g}_fs"
        spec = dict(
            kind="fullscale",
            stage="fullscale",
            market=market,
            gen=g,
            tag=tag,
            fit=fit,
            seed=int(seed),
            risk=bool(risk),
            hparams=FS.BUDGET[g],
            n_gen=P.N_GEN,
            save_samples=want_samples_for(market, tag, fit, seed),
            need_model=False,
            preserve_existing_metrics=True,
            gpu=True,
        )
        if g in ("diffusion", "quantgan"):
            fs_diff_quant.append(spec)
        else:
            fs_timegan.append(spec)

    # 2. Queue jobs (143 non-skipped) + 6 nse/full/seed0 sample/checkpoint backfills
    q_gan_gpu = []
    q_cpu_1 = []      # classical + timevae + cgnsde (v1)
    q_cpu_2 = []      # cgnsde2 main + risk + markets + nse/cgnsde2/full/seed0 checkpoint backfill
    q_ablation = []   # 39 v2_ablation jobs

    # Backfill nse/<gen>/full/seed0 samples (and cgnsde2 model checkpoint for api/server.py)
    for g in ("hmm_gaussian", "timevae", "quantgan", "timegan", "cgnsde", "cgnsde2"):
        base = g.split("@")[0]
        spec = dict(
            kind="backfill",
            stage="backfill_nse_full",
            market="nse",
            gen=g,
            tag=None,
            fit="full",
            seed=0,
            risk=False,
            hparams=None,
            n_gen=P.N_GEN,
            save_samples=True,
            need_model=(g == "cgnsde2"),
            preserve_existing_metrics=True,
            gpu=(base in GPU_CAPABLE_GENS),
        )
        if base in GPU_CAPABLE_GENS:
            q_gan_gpu.append(spec)
        elif base == "cgnsde2":
            q_cpu_2.append(spec)
        else:
            q_cpu_1.append(spec)

    for stage_name in Q.ORDER:
        for j in S[stage_name]:
            if is_skipped(j, skip_prefixes):
                continue
            if Q.done(j):
                continue
            gen = j["gen"]
            base = gen.split("@")[0]
            market = j["market"]
            fit = j["fit"]
            seed = int(j["seed"])
            risk = bool(j["risk"])
            spec = dict(
                kind="queue",
                stage=stage_name,
                market=market,
                gen=gen,
                tag=None,
                fit=fit,
                seed=seed,
                risk=risk,
                hparams=None,
                n_gen=P.N_GEN,
                save_samples=want_samples_for(market, gen, fit, seed),
                need_model=(gen == "cgnsde2" and market == "nse" and fit in ("full", "pre2020") and seed == 0),
                preserve_existing_metrics=True,
                gpu=(base in GPU_CAPABLE_GENS),
            )
            if stage_name == "v2_ablation":
                q_ablation.append(spec)
            elif base in GPU_CAPABLE_GENS:
                q_gan_gpu.append(spec)
            elif base == "cgnsde2":
                q_cpu_2.append(spec)
            else:
                q_cpu_1.append(spec)

    # Balance wall-clock time across cpu_shard_1..4: move 4 US/KR v2_markets jobs to the tail of cpu_shard_1
    if len(q_cpu_2) > 18:
        moved = q_cpu_2[-4:]
        q_cpu_2 = q_cpu_2[:-4]
        q_cpu_1 = q_cpu_1 + moved

    mid = (len(q_ablation) + 1) // 2  # 20 and 19
    q_cpu_3 = q_ablation[:mid]
    q_cpu_4 = q_ablation[mid:]

    return {
        # Primary shards + fallback tails so GPU kernels also help finish CPU shards once GPU work is done
        "gpu_shard_1": q_gan_gpu + fs_diff_quant + q_cpu_1 + q_cpu_2,
        "gpu_shard_2": fs_timegan + q_cpu_3 + q_cpu_4,
        "gpu_only_1": q_gan_gpu + fs_diff_quant,
        "gpu_only_2": fs_timegan,
        "cpu_shard_1": q_cpu_1,
        "cpu_shard_2": q_cpu_2,
        "cpu_shard_3": q_cpu_3,
        "cpu_shard_4": q_cpu_4,
        "all": q_gan_gpu + fs_diff_quant + fs_timegan + q_cpu_1 + q_cpu_2 + q_cpu_3 + q_cpu_4,
    }


def prewarm_references() -> None:
    """Ensure artifacts/reference/<market>/<fit>.pkl exists before spawning parallel workers."""
    pairs = [
        ("nse", "full"), ("nse", "pre2020"),
        ("nse", "upto2020"), ("nse", "upto2021"), ("nse", "upto2022"), ("nse", "upto2023"),
        ("us", "full"), ("us", "pre2020"),
        ("kr", "full"), ("kr", "pre2020"),
        ("nse_ext", "upto2024"),
    ]
    for market, fit in pairs:
        pkl_path = ARTIFACTS / "reference" / market / f"{fit}.pkl"
        if pkl_path.exists():
            continue
        json_path = RESULTS / "reference" / market / f"{fit}.json"
        orig_json = json_path.read_bytes() if json_path.exists() else None
        print(f"[prewarm] computing cached reference for {market}/{fit} ...", flush=True)
        fd, Wref, _ = make_fit_data(market, fit)
        reference(market, fit, fd, Wref)
        if orig_json is not None:
            json_path.write_bytes(orig_json)


def execute_single_job(spec: dict) -> None:
    import torch
    n_thr = max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))
    torch.set_num_threads(n_thr)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.set_num_threads = lambda *_: None

    market = spec["market"]
    gen = spec["gen"]
    fit = spec["fit"]
    seed = int(spec["seed"])
    risk = bool(spec.get("risk", False))
    hparams = spec.get("hparams")
    tag = spec.get("tag")
    label = tag or gen
    save_samples = bool(spec.get("save_samples", False))

    mpath = RESULTS / "runs" / market / label / fit / f"seed{seed}" / "metrics.json"
    orig_bytes = mpath.read_bytes() if (mpath.exists() and spec.get("preserve_existing_metrics", True)) else None

    bench_run(
        market=market,
        gen_name=gen,
        fit=fit,
        seed=seed,
        force=(orig_bytes is not None),
        save_samples=save_samples,
        want_risk_pool=risk,
        n_gen=int(spec.get("n_gen", P.N_GEN)),
        hparams=hparams,
        tag=tag,
    )
    if orig_bytes is not None:
        mpath.write_bytes(orig_bytes)


def collect_job_outputs(spec: dict) -> list[Path]:
    """Return existing output files produced for this job spec."""
    market = spec["market"]
    label = spec.get("tag") or spec["gen"]
    fit = spec["fit"]
    seed = int(spec["seed"])
    files = []
    mpath = RESULTS / "runs" / market / label / fit / f"seed{seed}" / "metrics.json"
    if mpath.exists():
        files.append(mpath)
    ppath = ARTIFACTS / "risk_pool" / market / label / fit / f"seed{seed}.npz"
    if ppath.exists():
        files.append(ppath)
    spath = ARTIFACTS / "samples" / market / label / fit / f"seed{seed}.npy"
    if spath.exists() and want_samples_for(market, label, fit, seed):
        files.append(spath)
    mod_dir = ARTIFACTS / "models" / market / label / fit
    if mod_dir.exists() and seed == 0 and (label in ("cgnsde2", "cgnsde") or label.endswith("_fs")):
        for p in mod_dir.glob(f"seed{seed}*"):
            if p.is_file():
                files.append(p)
    return files


def pack_outputs(out_zip: Path, tracked: set[Path], extra_files: list[Path]) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_zip.with_suffix(".tmp.zip")
    all_files = sorted(set(tracked) | {p for p in extra_files if p.exists()})
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in all_files:
            if p.exists() and p.is_file() and p.name != ".claim":
                z.write(p, p.relative_to(ROOT))
    shutil.move(tmp, out_zip)


def run_worker_job(
    idx: int,
    total: int,
    spec: dict,
    worker_slot: int,
    n_gpus: int,
    threads_per_worker: int,
    t0: float,
    hours_limit: float,
) -> tuple[dict, int, float, str]:
    if (time.time() - t0) / 3600.0 > hours_limit:
        return spec, -2, 0.0, "time_limit_reached"
    if job_is_done(spec):
        return spec, 0, 0.0, "already_done"

    env = dict(
        os.environ,
        OMP_NUM_THREADS=str(threads_per_worker),
        MKL_NUM_THREADS=str(threads_per_worker),
        OPENBLAS_NUM_THREADS=str(threads_per_worker),
        PYTHONWARNINGS="ignore",
        PYTHONIOENCODING="utf-8",
    )
    if spec.get("gpu") and n_gpus > 0:
        gpu_id = worker_slot % n_gpus
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["CRISISBENCH_DEVICE"] = "cuda"
    else:
        env["CRISISBENCH_DEVICE"] = "cpu"

    cmd = [sys.executable, "-m", "cloud.kaggle_runner", "--single-job", json.dumps(spec)]
    t1 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    elapsed = time.time() - t1
    tail = "\n".join((r.stdout + "\n" + r.stderr).strip().splitlines()[-10:])
    return spec, r.returncode, elapsed, tail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", help="Shard mode to execute")
    ap.add_argument("--workers", type=int, default=4, help="Parallel worker processes")
    ap.add_argument("--threads", type=int, default=1, help="OMP/MKL threads per worker process")
    ap.add_argument("--hours", type=float, default=11.0, help="Stop starting new jobs after this many hours")
    ap.add_argument("--out", default=str(ROOT.parent / "output_bundle.zip"), help="Output zip path")
    ap.add_argument("--single-job", default=None, help="Internal: JSON spec for a single job subprocess")
    ap.add_argument("--list", action="store_true", help="List job counts per mode and exit")
    args = ap.parse_args()

    if args.single_job is not None:
        execute_single_job(json.loads(args.single_job))
        return

    shards = build_all_specs()
    if args.list:
        for k, specs in shards.items():
            todo = [s for s in specs if not job_is_done(s)]
            print(f"{k:14s}: {len(todo):3d} todo / {len(specs):3d} total")
        return

    if args.mode not in shards:
        raise ValueError(f"Unknown mode {args.mode!r}; choose from {sorted(shards)}")

    import torch
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
    print(f"=== CrisisBench Kaggle Runner ===", flush=True)
    print(f"mode={args.mode} | workers={args.workers} | threads={args.threads} | gpus={n_gpus} {gpu_names}", flush=True)

    prewarm_references()

    specs = [s for s in shards[args.mode] if not job_is_done(s)]
    print(f"Jobs to run in mode '{args.mode}': {len(specs)}", flush=True)

    out_zip = Path(args.out)
    log_path = RESULTS / f"kaggle_log_{args.mode}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tracked_files: set[Path] = set()
    lock = threading.Lock()
    t0 = time.time()

    n_ok = n_fail = n_skip = 0
    with open(log_path, "w", newline="", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=args.workers) as ex:
        w = csv.writer(fh)
        w.writerow(["finished", "stage", "market", "gen", "fit", "seed", "risk", "rc", "minutes", "tail"])
        fh.flush()

        futures = {}
        for i, spec in enumerate(specs):
            fut = ex.submit(
                run_worker_job,
                i + 1,
                len(specs),
                spec,
                i % args.workers,
                n_gpus,
                args.threads,
                t0,
                args.hours,
            )
            futures[fut] = (i + 1, spec)

        for fut in as_completed(futures):
            idx, spec = futures[fut]
            _, rc, elapsed, tail = fut.result()
            label = spec.get("tag") or spec["gen"]
            with lock:
                if rc == 0:
                    n_ok += 1
                    for p in collect_job_outputs(spec):
                        tracked_files.add(p)
                    status_str = "OK  "
                elif rc == -2:
                    n_skip += 1
                    status_str = "TIME"
                else:
                    n_fail += 1
                    status_str = "FAIL"

                w.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    spec["stage"],
                    spec["market"],
                    label,
                    spec["fit"],
                    spec["seed"],
                    spec["risk"],
                    rc,
                    f"{elapsed / 60.0:.2f}",
                    tail.splitlines()[-1] if tail else "",
                ])
                fh.flush()
                print(
                    f"[{n_ok + n_fail + n_skip}/{len(specs)}] {status_str} "
                    f"{spec['stage']}:{spec['market']}/{label}/{spec['fit']}/seed{spec['seed']} "
                    f"({elapsed / 60.0:.1f} min)",
                    flush=True,
                )
                if rc not in (0, -2) and tail:
                    print("    " + tail.replace("\n", "\n    "), flush=True)
                pack_outputs(out_zip, tracked_files, [log_path])

    pack_outputs(out_zip, tracked_files, [log_path])
    size_mb = out_zip.stat().st_size / 1e6 if out_zip.exists() else 0.0
    print(
        f"=== Finished mode={args.mode} in {(time.time() - t0) / 3600.0:.2f} h: "
        f"{n_ok} ok, {n_fail} failed, {n_skip} time-skipped | {out_zip} ({size_mb:.2f} MB) ===",
        flush=True,
    )


if __name__ == "__main__":
    main()
