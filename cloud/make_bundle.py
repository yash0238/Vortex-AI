"""Build cloud/vortex_bundle.zip: the code, the NSE panel and the cached real-data reference that the
full-scale GPU runs need (no raw vendor files, no results).   python cloud/make_bundle.py"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "cloud" / "vortex_bundle.zip"
files = [p for p in (ROOT / "crisisbench").rglob("*.py")]
files += [p for p in (ROOT / "data" / "panels" / "nse").iterdir() if p.is_file()]
files += list((ROOT / "artifacts" / "reference" / "nse").glob("*.pkl"))
files += [ROOT / "data" / "node_features.py"]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, Path("vortex") / p.relative_to(ROOT))
print(f"{out}  {out.stat().st_size / 1e6:.1f} MB, {len(files)} files")
