"""Export everything the dashboard shows into static JSON (app/public/data/).

    python -m crisisbench.export_web

The dashboard reads only these files, so it works without a backend and can never
show a number that is not in results/.  Sections whose inputs do not exist yet are
exported as null and the page says "not run yet".
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import protocol as P
from .analysis import MAIN, PRETTY, RESULTS, TABLES, pretty
from .bench import ARTIFACTS, make_fit_data
from .data import MARKETS, PANELS, ROOT, load_panel
from .labels import label_windows
from . import metrics as M

OUT = ROOT / "app" / "public" / "data"
SECTORS = {
    "Financials": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS", "BAJFINANCE.NS",
                   "BAJAJFINSV.NS", "INDUSINDBK.NS"],
    "IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "COALINDIA.NS", "NTPC.NS", "POWERGRID.NS"],
    "Consumer": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS", "ASIANPAINT.NS",
                 "TITAN.NS"],
    "Autos": ["MARUTI.NS", "M&M.NS", "TMPV.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS"],
    "Pharma": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "APOLLOHOSP.NS"],
    "Materials": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "ULTRACEMCO.NS", "SHREECEM.NS", "GRASIM.NS",
                  "UPL.NS"],
    "Infra & Telecom": ["LT.NS", "ADANIPORTS.NS", "BHARTIARTL.NS"],
}


def _r(a, nd=4):
    """Round arrays/scalars to plain JSON."""
    if a is None:
        return None
    if isinstance(a, (float, np.floating)):
        return None if not np.isfinite(a) else round(float(a), nd)
    a = np.asarray(a, dtype=float)
    return np.where(np.isfinite(a), np.round(a, nd), None).tolist()


def _csv(name: str):
    p = TABLES / f"{name}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return json.loads(df.to_json(orient="records", double_precision=5))


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def sector_of(tk: str) -> str:
    for s, ts in SECTORS.items():
        if tk in ts:
            return s
    return "Other"


def corr_of(W: np.ndarray) -> np.ndarray:
    return np.corrcoef(W.reshape(-1, W.shape[2]), rowvar=False)


def episodes(dates: pd.DatetimeIndex, lab: np.ndarray) -> list[dict]:
    out, start = [], None
    for i, v in enumerate(lab):
        if v and start is None:
            start = i
        if (not v or i == len(lab) - 1) and start is not None:
            end = i if v else i - 1
            out.append(dict(start=str(dates[start].date()), end=str(dates[end].date()), windows=int(end - start + 1)))
            start = None
    merged: list[dict] = []                      # runs separated by fewer than 10 trading days are one episode
    pos = {str(d.date()): i for i, d in enumerate(dates)}
    for e in out:
        if merged and pos[e["start"]] - pos[merged[-1]["end"]] < 10:
            merged[-1]["end"] = e["end"]
            merged[-1]["windows"] += e["windows"]
        else:
            merged.append(dict(e))
    return merged


def market_block(market: str) -> dict | None:
    if not (PANELS / market / "returns.npy").exists():
        return None
    panel = load_panel(market)
    audit = _json(PANELS / market / "audit.json") or {}
    fd, W, wend = make_fit_data(market, "full")
    lab, score = label_windows(W, fd.rule)
    ew = panel.returns.mean(1)
    idx = 100 * np.exp(np.cumsum(ew))
    rv = pd.Series(ew).rolling(20).std().to_numpy() * np.sqrt(252) * 100
    ws = M.window_stats(W, fd.rule)
    step = 2                                           # thin the daily series for the browser
    order = np.arange(panel.n_assets)
    sectors = [sector_of(t) for t in panel.tickers] if market.startswith("nse") else ["" for _ in panel.tickers]
    if market.startswith("nse"):
        order = np.array(sorted(range(panel.n_assets), key=lambda i: (sectors[i], panel.tickers[i])))
    Cn, Cc = corr_of(W[lab == 0]), corr_of(W[lab == 1])
    return dict(
        market=market, label=MARKETS.get(market, {}).get("label", market), n_assets=panel.n_assets,
        n_days=len(panel.dates), first=str(panel.dates[0].date()), last=str(panel.dates[-1].date()),
        tickers=[panel.tickers[i] for i in order], sectors=[sectors[i] for i in order],
        sha256=audit.get("sha256_returns"), cleaning_log=audit.get("log", []),
        kurtosis_median=_r(audit.get("kurtosis_median")),
        series=dict(date=[str(d.date()) for d in panel.dates[::step]], index=_r(idx[::step], 2), vol=_r(rv[::step], 2)),
        windows=dict(n=int(len(W)), n_crisis=int(lab.sum()),
                     end=[str(d.date()) for d in wend[::step]], label=lab[::step].tolist(),
                     score=_r(score[::step], 3), mac=_r(ws.mac[::step], 3), threshold=_r(fd.rule["score_thr"], 4)),
        episodes=episodes(wend, lab),
        corr_normal=_r(Cn[np.ix_(order, order)], 3), corr_crisis=_r(Cc[np.ix_(order, order)], 3),
        mac=dict(normal=_r(float(ws.mac[lab == 0].mean())), crisis=_r(float(ws.mac[lab == 1].mean()))),
        ann_vol=_r(panel.returns.std(0)[order] * np.sqrt(252) * 100, 1),
    )


def summarise_windows(W: np.ndarray, rule: dict) -> dict:
    """Example paths, fan quantiles and per-regime statistics of a set of windows (frozen rule labels)."""
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    lab, _ = label_windows(W, rule)
    ew = W.mean(2)
    cum = 100 * (np.exp(np.cumsum(ew, 1)) - 1)
    rng = np.random.default_rng(0)
    d = dict(n=int(len(W)), n_crisis=int(lab.sum()))
    for name, sel in (("normal", lab == 0), ("crisis", lab == 1)):
        if sel.sum() < 5:
            d[name] = None
            continue
        c = cum[sel]
        pick = rng.choice(len(c), min(12, len(c)), replace=False)
        ws_mac = [float(np.abs(np.corrcoef(w, rowvar=False)[~np.eye(W.shape[2], dtype=bool)]).mean())
                  for w in W[sel][:200]]
        d[name] = dict(paths=_r(c[pick], 2), fan=_r(np.quantile(c, q, axis=0), 2),
                       vol=_r(float(ew[sel].std() * np.sqrt(252) * 100), 2), mac=_r(float(np.mean(ws_mac)), 3),
                       worst_day=_r(float(np.quantile(ew[sel], 0.01) * 100), 2),
                       hist=np.histogram(ew[sel].ravel() * 100, bins=41, range=(-8, 8))[0].tolist())
    return d


def sample_block(market: str, fit: str, gens: list[str]) -> dict:
    """A few example paths, fan quantiles and regime correlations per generator (seed 0)."""
    fd, Wref, _ = make_fit_data(market, fit)
    one = lambda W: summarise_windows(W, fd.rule)
    out = {"real": one(Wref)}
    for g in gens:
        p = ARTIFACTS / "samples" / market / g / fit / "seed0.npy"
        if p.exists():
            out[g] = one(np.load(p).astype(np.float64))
    return out


def bench_block(market: str, fit: str) -> dict | None:
    t = _csv(f"T1_{market}_{fit}")
    if t is None:
        return None
    ref = _json(RESULTS / "reference" / market / f"{fit}.json")
    tau = None
    p = TABLES / f"F2_metric_tau_{market}_{fit}.csv"
    if p.exists():
        m = pd.read_csv(p, index_col=0)
        tau = dict(metrics=list(m.index), matrix=_r(m.to_numpy(), 3))
    return dict(rows=t, reference=ref, metric_tau=tau, alt_rule=_csv(f"T1alt_{market}_{fit}"),
                ablation=_csv(f"T2_ablation_{market}_{fit}"), ablation_v3=_csv(f"T2_ablation_v3_{market}_{fit}"))


def _plain(x: str) -> str:
    """Display text without internal version labels (initial design / without contagion / final model)."""
    for a, b in (("(with v2, post hoc)", "(with post hoc models)"), ("(v2, post hoc)", "(without contagion, post hoc)"),
                 ("CG-NSDE v1", "CG-NSDE initial design"), ("CG-NSDE v2", "CG-NSDE without contagion"),
                 ("CG-NSDE v3", "CG-NSDE (final model)")):
        x = x.replace(a, b)
    return x


def _plain_obj(o):
    if isinstance(o, dict):
        return {_plain(k): _plain_obj(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_plain_obj(v) for v in o]
    return _plain(o) if isinstance(o, str) else o


def cgnsde_block() -> dict | None:
    p = RESULTS / "runs" / "nse" / "cgnsde3" / "full" / "seed0" / "metrics.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    p1 = RESULTS / "runs" / "nse" / "cgnsde" / "full" / "seed0" / "metrics.json"
    v1 = json.loads(p1.read_text()).get("info") if p1.exists() else None
    if v1:
        v1 = {k: v for k, v in v1.items() if k != "history"}
    extra = _json(RESULTS / "07_model" / "cgnsde_explain.json")
    return dict(info=d.get("info"), hparams=d.get("hparams"), train_minutes=d.get("train_minutes"),
                conditional=d.get("conditional"), explain=extra, v1_info=v1)


def _dump(obj) -> str:
    """Strict JSON for the browser: NaN / inf (metrics that were not produced) become null."""
    import math

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        if isinstance(o, (np.floating, np.integer)):
            o = o.item()
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return o
    return json.dumps(clean(obj), default=float, allow_nan=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = _json(TABLES / "summary.json") or {}
    markets = {m: market_block(m) for m in ("nse", "us", "kr", "nse_ext")}
    bench = {}
    for m in ("nse", "us", "kr", "nse_ext", "nse_c_tick", "nse_c_1pct", "nse_c_split"):
        for f in ("full", "pre2020", "upto2024"):
            b = bench_block(m, f)
            if b:
                bench[f"{m}/{f}"] = b
    samples = {f"nse/{f}": sample_block("nse", f, MAIN) for f in ("full", "pre2020")}
    gates = {p.stem: p.read_text(encoding="utf-8") for p in sorted((RESULTS / "gates").glob("*.md"))} \
        if (RESULTS / "gates").exists() else {}
    core = dict(
        built=time.strftime("%Y-%m-%d %H:%M"), protocol=json.loads(json.dumps(P.as_dict(), default=list)),
        pretty=PRETTY, summary=summary, bench=bench, gates=gates,
        heldout=_csv("T3_heldout"), risk=_csv("T4_risk"), risk_series=_json(RESULTS / "06_risk" / "series.json"),
        metric_validity=_csv("T5_metric_validity"), contamination=_csv("T7_contamination"),
        dgp=_csv_any(RESULTS / "01_dgp" / "metric_validation.csv"), cgnsde=cgnsde_block(),
        hypotheses=_plain_obj(_json(TABLES / "hypotheses.json")),
    )
    core["gates"] = {k: _plain(v) for k, v in core["gates"].items()}
    (OUT / "core.json").write_text(_dump(core), encoding="utf-8")
    (OUT / "markets.json").write_text(_dump(markets), encoding="utf-8")
    (OUT / "samples.json").write_text(_dump(samples), encoding="utf-8")
    for f in ("core", "markets", "samples"):
        print(f, round((OUT / f"{f}.json").stat().st_size / 1e6, 2), "MB")


def _csv_any(p: Path):
    if not p.exists():
        return None
    return json.loads(pd.read_csv(p).to_json(orient="records", double_precision=5))


if __name__ == "__main__":
    main()
