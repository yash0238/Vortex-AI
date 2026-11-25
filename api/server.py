"""Small API for the dashboard's Scenario Lab: live generation from the trained CG-NSDE.

    .venv\\Scripts\\python.exe -m uvicorn api.server:app --port 8000

POST /api/generate {"n": 400, "seed": 1, "fit": "full"}  ->  the same structure the dashboard reads from
app/public/data/samples.json for one generator (example paths, fan quantiles, per-regime statistics), computed on
freshly sampled windows.  Crisis / calm is decided by the frozen rule on the generated returns, exactly as in the
benchmark.  The dashboard works without this server; only the "generate live" button needs it.

The legacy market-data endpoints used by the Market Explorer page are mounted from api/main.py when it imports.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from crisisbench.bench import ARTIFACTS, make_fit_data          # noqa: E402
from crisisbench.export_web import summarise_windows            # noqa: E402
from crisisbench.generators.cgnsde3 import CGNSDE3 as CGNSDE               # noqa: E402  (final model)

app = FastAPI(title="VORTEX-AI scenario API", docs_url="/api/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class GenerateRequest(BaseModel):
    n: int = Field(400, ge=50, le=2000)
    seed: int = 0
    fit: str = Field("full", pattern="^(full|pre2020)$")


@lru_cache(maxsize=2)
def _model(fit: str):
    path = ARTIFACTS / "models" / "nse" / "cgnsde3" / fit / "seed0"
    try:
        gen = CGNSDE.load(path)
    except FileNotFoundError as e:
        raise HTTPException(503, f"No trained CG-NSDE checkpoint at {path}. Run the benchmark queue first.") from e
    return gen, _rule(fit)


def _rule(fit: str) -> dict:
    """Frozen crisis-rule thresholds (two numbers). Saved once to api/rules/ so a deployed server needs only the
    checkpoint and these thresholds, never the raw price panel."""
    f = ROOT / "api" / "rules" / f"nse_{fit}.json"
    if f.exists():
        return json.loads(f.read_text())
    fd, _, _ = make_fit_data("nse", fit)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(fd.rule, indent=1))
    return fd.rule


@app.on_event("startup")
def _warm() -> None:
    """Load the in-sample model at start-up so the first click on "generate live" does not wait for it."""
    try:
        _model("full")
    except HTTPException:
        pass


@app.get("/api/health")
def health() -> dict:
    have = {f: any((ARTIFACTS / "models" / "nse" / "cgnsde3" / f).glob("seed0*")) for f in ("full", "pre2020")}
    return dict(status="ok", checkpoints=have)


@app.post("/api/generate")
def generate(req: GenerateRequest) -> dict:
    gen, rule = _model(req.fit)
    W = gen.sample(req.n, np.random.default_rng(req.seed))
    out = summarise_windows(np.asarray(W, dtype=np.float64), rule)
    out["source"] = f"CG-NSDE, fit '{req.fit}', seed {req.seed}"
    return out


try:                                    # legacy market-data routes for the Market Explorer side tool
    from api.main import app as _legacy
    for r in _legacy.router.routes:
        if getattr(r, "path", "").startswith("/api/market"):
            app.router.routes.append(r)
except Exception:                       # the research API must work even if the legacy module cannot import
    pass
