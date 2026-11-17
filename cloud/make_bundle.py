"""Build cloud/vortex_bundle.zip: the code, all market panels, cached references, and existing
finished runs/risk_pools so Kaggle GPU/CPU kernels skip completed work and waste nothing.

    python cloud/make_bundle.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "cloud" / "vortex_bundle.zip"

files: list[Path] = []
files += sorted(p for p in (ROOT / "crisisbench").rglob("*.py") if p.is_file())
files += [ROOT / "cloud" / "__init__.py", ROOT / "cloud" / "kaggle_runner.py"]
files += [ROOT / "data" / "node_features.py"]
if (ROOT / "results" / "skip_gens.txt").exists():
    files.append(ROOT / "results" / "skip_gens.txt")

# All market panels (nse, us, kr, nse_ext, contamination panels, sim panels), excluding raw vendor caches
if (ROOT / "data" / "panels").exists():
    for panel_dir in sorted((ROOT / "data" / "panels").iterdir()):
        if panel_dir.is_dir():
            for p in sorted(panel_dir.iterdir()):
                if p.is_file():
                    files.append(p)

# Cached references (pkl + json)
if (ROOT / "artifacts" / "reference").exists():
    files += sorted(p for p in (ROOT / "artifacts" / "reference").rglob("*.pkl") if p.is_file())
if (ROOT / "results" / "reference").exists():
    files += sorted(p for p in (ROOT / "results" / "reference").rglob("*.json") if p.is_file())

# Existing finished run metrics and risk pools so completed jobs are skipped
if (ROOT / "results" / "runs").exists():
    files += sorted(p for p in (ROOT / "results" / "runs").rglob("metrics.json") if p.is_file())
if (ROOT / "artifacts" / "risk_pool").exists():
    files += sorted(p for p in (ROOT / "artifacts" / "risk_pool").rglob("*.npz") if p.is_file())

seen: set[Path] = set()
unique_files: list[Path] = []
for p in files:
    rp = p.resolve()
    if rp not in seen and rp.exists():
        seen.add(rp)
        unique_files.append(p)

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in unique_files:
        z.write(p, Path("vortex") / p.relative_to(ROOT))

print(f"{out}  {out.stat().st_size / 1e6:.2f} MB, {len(unique_files)} files")
