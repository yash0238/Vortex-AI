"""Build cloud bundles for remote GPU/CPU training.

Usage:
    python cloud/make_bundle.py          # Full bundle (cloud/vortex_bundle.zip): all panels, refs, and finished runs
    python cloud/make_bundle.py --colab  # Lean Colab bundle (cloud/vortex_colab_bundle.zip): code + NSE panel/refs + *_fs state
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def collect_common_code() -> list[Path]:
    files: list[Path] = []
    files += sorted(p for p in (ROOT / "crisisbench").rglob("*.py") if p.is_file())
    files += [ROOT / "cloud" / "__init__.py", ROOT / "cloud" / "kaggle_runner.py"]
    files += [ROOT / "data" / "node_features.py"]
    if (ROOT / "results" / "skip_gens.txt").exists():
        files.append(ROOT / "results" / "skip_gens.txt")
    return files


def build_zip(out: Path, files: list[Path]) -> None:
    seen: set[Path] = set()
    unique_files: list[Path] = []
    for p in files:
        rp = p.resolve()
        if rp not in seen and rp.exists() and rp.is_file():
            seen.add(rp)
            unique_files.append(p)

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in unique_files:
            z.write(p, Path("vortex") / p.relative_to(ROOT))
    print(f"{out}  {out.stat().st_size / 1e6:.2f} MB, {len(unique_files)} files")


def build_full() -> None:
    files = collect_common_code()
    panels_dir = ROOT / "data" / "panels"
    if panels_dir.exists():
        for panel_dir in sorted(panels_dir.iterdir()):
            if panel_dir.is_dir():
                for p in sorted(panel_dir.iterdir()):
                    if p.is_file():
                        files.append(p)
    if (ROOT / "artifacts" / "reference").exists():
        files += sorted(p for p in (ROOT / "artifacts" / "reference").rglob("*.pkl") if p.is_file())
    if (ROOT / "results" / "reference").exists():
        files += sorted(p for p in (ROOT / "results" / "reference").rglob("*.json") if p.is_file())
    if (ROOT / "results" / "runs").exists():
        files += sorted(p for p in (ROOT / "results" / "runs").rglob("metrics.json") if p.is_file())
    if (ROOT / "artifacts" / "risk_pool").exists():
        files += sorted(p for p in (ROOT / "artifacts" / "risk_pool").rglob("*.npz") if p.is_file())
    build_zip(ROOT / "cloud" / "vortex_bundle.zip", files)


def build_colab() -> None:
    """Lean bundle for Google Colab full-scale neural baseline runs (NSE only + finished *_fs markers)."""
    files = collect_common_code()
    nse_panel = ROOT / "data" / "panels" / "nse"
    if nse_panel.exists():
        files += sorted(p for p in nse_panel.iterdir() if p.is_file())
    ref_art = ROOT / "artifacts" / "reference" / "nse"
    if ref_art.exists():
        files += sorted(p for p in ref_art.glob("*.pkl") if p.is_file())
    ref_res = ROOT / "results" / "reference" / "nse"
    if ref_res.exists():
        files += sorted(p for p in ref_res.glob("*.json") if p.is_file())
    runs_nse = ROOT / "results" / "runs" / "nse"
    if runs_nse.exists():
        files += sorted(p for p in runs_nse.glob("*_fs/**/metrics.json") if p.is_file())
    pools_nse = ROOT / "artifacts" / "risk_pool" / "nse"
    if pools_nse.exists():
        files += sorted(p for p in pools_nse.glob("*_fs/**/*.npz") if p.is_file())
    build_zip(ROOT / "cloud" / "vortex_colab_bundle.zip", files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--colab", action="store_true", help="Build lean Colab bundle (cloud/vortex_colab_bundle.zip)")
    ap.add_argument("--all", action="store_true", help="Build both full and Colab bundles")
    args = ap.parse_args()
    if args.all:
        build_full()
        build_colab()
    elif args.colab:
        build_colab()
    else:
        build_full()


if __name__ == "__main__":
    main()
