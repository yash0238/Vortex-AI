"""Turn run outputs into the paper's tables, hypothesis tests and gate files.

    python -m crisisbench.analysis            (everything that has data)

Reads   results/runs/**/metrics.json, results/reference/**, results/06_risk/*
Writes  results/tables/T*.csv + .md, results/gates/GATE_*.md, results/tables/summary.json

Nothing here fits a model or draws a random number that affects a reported value
except bootstrap intervals, which use fixed seeds.  Gate files are generated; do not
edit them by hand.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import protocol as P
from .data import ROOT
from .bench import make_fit_data
from . import metrics as M

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
GATES = RESULTS / "gates"
FAMILIES = list(P.METRIC_FAMILIES)
ALL_METRICS = [m for f in FAMILIES for m in P.METRIC_FAMILIES[f]]

# ------------------------------------------------------------------- bootstrap cache
_REAL_BOOT_CACHE: dict[tuple[str, str], tuple[dict, dict[str, np.ndarray]]] = {}


def _real_boot_reps(market: str, fit: str, B: int = 200, seed: int = 0
                    ) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Real-date circular block-bootstrap replicates for every metric.

    Uses the cached real block-bootstrap summary in results/reference/<market>/<fit>.json
    when available (falling back to window_stats otherwise) so analysis runs in seconds.
    """
    key = (market, fit)
    if key in _REAL_BOOT_CACHE:
        return _REAL_BOOT_CACHE[key]
    ref_path = RESULTS / "reference" / market / f"{fit}.json"
    rng = np.random.default_rng(seed)
    if ref_path.exists():
        ref_doc = json.loads(ref_path.read_text())
        rm = ref_doc.get("metrics", {})
        point: dict[str, float] = {}
        reps: dict[str, np.ndarray] = {}
        for m in ALL_METRICS:
            if m not in rm:
                continue
            info = rm[m]
            kind = info.get("kind", "stat")
            r_val = float(info.get("real", 0.0))
            point[m] = r_val
            if kind in ("stat", "disc") and np.isfinite(info.get("half", np.nan)):
                se = float(info["half"]) / 1.959963984540054
                reps[m] = r_val + rng.normal(0.0, se, size=B)
            elif kind == "dist" and np.isfinite(info.get("ceiling", np.nan)):
                ceil = float(info["ceiling"])
                med = float(info.get("median", 0.5 * ceil))
                se = max(ceil - med, 0.0) / 1.6448536269514722
                reps[m] = np.clip(med + rng.normal(0.0, se, size=B), 0.0, None)
            else:
                reps[m] = np.full(B, r_val, dtype=float)
        _REAL_BOOT_CACHE[key] = (point, reps)
        return point, reps
    fd, W, _ = make_fit_data(market, fit)
    ws = M.window_stats(W, fd.rule)
    full = M.aggregate(ws)
    point = M.compare(full, full)
    n = ws.n
    block = P.BOOT_BLOCK
    reps = {k: np.empty(B, dtype=float) for k in ALL_METRICS}
    for b in range(B):
        idx = M.block_bootstrap_idx(n, rng, block)
        p = M.compare(M.aggregate(ws, idx), full)
        for m in ALL_METRICS:
            if m in p:
                reps[m][b] = p[m]
    _REAL_BOOT_CACHE[key] = (point, reps)
    return point, reps


def _seed_date_ci(vals, real_point: float, real_reps: np.ndarray, seed: int = 0,
                  B: int = 200, level: float = 0.95) -> tuple[float, float]:
    """Seed x date CI: resample generator seeds and shift by real-date bootstrap deviation."""
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0 or real_reps is None or real_reps.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    if x.size > 1:
        seed_means = x[rng.integers(0, x.size, size=(B, x.size))].mean(axis=1)
    else:
        seed_means = np.full(B, x[0])
    reps = np.asarray(real_reps, dtype=float)
    reps = reps[np.isfinite(reps)]
    if reps.size == 0:
        draws = seed_means
    else:
        take = rng.integers(0, reps.size, size=B)
        draws = seed_means + (reps[take] - real_point)
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(draws, [a, 1.0 - a])
    return (float(lo), float(hi))
PREREGISTERED = (
    "iid_gaussian", "t_ppca", "stationary_bootstrap", "garch_fhs", "dcc_garch",
    "regime_gaussian", "hmm_gaussian", "timevae", "quantgan", "timegan", "cgnsde",
)
PRETTY = {
    "iid_gaussian": "i.i.d. Gaussian", "t_ppca": "Static t-PPCA", "stationary_bootstrap": "Stationary bootstrap",
    "garch_fhs": "GARCH-t FHS", "dcc_garch": "DCC-GARCH-t", "regime_gaussian": "Regime Gaussian",
    "hmm_gaussian": "Gaussian HMM", "timevae": "TimeVAE", "quantgan": "QuantGAN", "timegan": "TimeGAN",
    "diffusion": "Diffusion (DDPM)",
    "quantgan_fs": "QuantGAN (full-scale)", "timegan_fs": "TimeGAN (full-scale)",
    "diffusion_fs": "Diffusion (full-scale)",
    "cgnsde": "CG-NSDE v1 (preregistered)", "cgnsde2": "CG-NSDE v2 (ours, post hoc)",
}
MAIN = list(PRETTY)
REGIME_AWARE = ("regime_gaussian", "hmm_gaussian", "cgnsde")


def pretty(g: str) -> str:
    if g in PRETTY:
        return PRETTY[g]
    if g.startswith("cgnsde2@"):
        return "CG-NSDE v2 / " + g.split("@")[1]
    if g.startswith("cgnsde@"):
        return "CG-NSDE v1 / " + g.split("@")[1]
    return g


def _binom_ci(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact Clopper-Pearson binomial confidence interval for k successes in n trials."""
    if n <= 0:
        return (np.nan, np.nan)
    ci = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=level, method="exact")
    return (float(ci.low), float(ci.high))


def _seed_boot_ci(vals, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Nonparametric bootstrap 95% CI of the seed mean."""
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    if x.size == 1:
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [a, 1.0 - a])
    return (float(lo), float(hi))


def _block_boot_ci(vals, ref_metric: dict | None, n_real: int = 3424, n_boot: int = 2000,
                   seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Joint seed x circular-block-bootstrap 95% CI combining seed resampling with block-bootstrap variance."""
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    seed_means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1) if x.size > 1 else np.full(n_boot, x[0])
    block_se = 0.0
    if ref_metric:
        scale = float(np.sqrt(max(n_real, P.T) / max(x.size * P.N_GEN, P.T)))
        if ref_metric.get("kind") in ("stat", "disc") and np.isfinite(ref_metric.get("half", np.nan)):
            block_se = float(ref_metric["half"]) / 1.959963984540054 * scale
        elif ref_metric.get("kind") == "dist" and np.isfinite(ref_metric.get("ceiling", np.nan)):
            med = float(ref_metric.get("median", 0.5 * ref_metric["ceiling"]))
            block_se = max(float(ref_metric["ceiling"]) - med, 0.0) / 1.6448536269514722 * scale
    draws = seed_means + rng.normal(0.0, block_se, size=n_boot) if block_se > 0 else seed_means
    if ref_metric and ref_metric.get("kind") == "dist":
        draws = np.clip(draws, 0.0, None)
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(draws, [a, 1.0 - a])
    return (float(lo), float(hi))


# ------------------------------------------------------------------ loading
def load_runs(market: str | None = None) -> pd.DataFrame:
    """Long frame: one row per (market, generator, fit, seed, rule, metric)."""
    rows = []
    base = RESULTS / "runs"
    for p in base.glob("*/*/*/seed*/metrics.json"):
        mk, gen, fit = p.parts[-5], p.parts[-4], p.parts[-3]
        if gen.startswith("_") or gen.endswith("_smoke") or (market and mk != market):
            continue
        d = json.loads(p.read_text())
        for rule in ("main", "alt_rule"):
            for m, v in d[rule]["metrics"].items():
                rows.append(dict(market=mk, generator=gen, fit=fit, seed=d["seed"], rule=rule, metric=m,
                                 value=v["value"], err=v["err"], inside=v["inside"], produced=v["produced"]))
            for f, s in d[rule]["family"].items():
                rows.append(dict(market=mk, generator=gen, fit=fit, seed=d["seed"], rule=rule, metric=f"family_{f}",
                                 value=s, err=s, inside=None, produced=True))
        rows.append(dict(market=mk, generator=gen, fit=fit, seed=d["seed"], rule="meta", metric="train_minutes",
                         value=d["train_minutes"], err=np.nan, inside=None, produced=True))
        if "conditional" in d:
            rows.append(dict(market=mk, generator=gen, fit=fit, seed=d["seed"], rule="meta", metric="cond_boost_pct",
                             value=d["conditional"]["boost_pct"], err=np.nan, inside=None, produced=True))
    return pd.DataFrame(rows)


def load_meta() -> dict:
    out = {}
    for p in (RESULTS / "runs").glob("*/*/*/seed*/metrics.json"):
        d = json.loads(p.read_text())
        out.setdefault(p.parts[-4], dict(impl_source=d["impl_source"], family_tag=d["family_tag"]))
    return out


def reference(market: str, fit: str) -> dict:
    return json.loads((RESULTS / "reference" / market / f"{fit}.json").read_text())


# ------------------------------------------------------------------- tables
def _fmt(m, s, nd=3):
    if m is None or not np.isfinite(m):
        return "n/p"
    return f"{m:.{nd}g}" if (s is None or not np.isfinite(s) or s == 0) else f"{m:.{nd}g} ± {s:.2g}"


def main_table(df: pd.DataFrame, market: str, fit: str, gens: list[str], rule: str = "main") -> pd.DataFrame:
    """Per generator: metric mean, sd, seed/block-bootstrap CIs, pass counts, family scores and ranks."""
    d = df[(df.market == market) & (df.fit == fit) & (df.rule == rule) & df.generator.isin(gens)]
    if d.empty:
        return pd.DataFrame()
    ref_doc = reference(market, fit) if (RESULTS / "reference" / market / f"{fit}.json").exists() else {}
    ref_metrics = ref_doc.get("metrics" if rule == "main" else "metrics_alt_rule", {})
    n_real = int(ref_doc.get("n_windows", 3424))
    g = d.groupby(["generator", "metric"])
    mean = g["value"].mean().unstack()
    sd = g["value"].std(ddof=1).unstack()
    vmin = g["value"].min().unstack()
    vmax = g["value"].max().unstack()
    err = g["err"].mean().unstack()
    err_sd = g["err"].std(ddof=1).unstack()
    d_in = d[d.inside.notna()].assign(inside=lambda x: x.inside.astype(float))
    inside_grp = d_in.groupby(["generator", "metric"])["inside"]
    inside = inside_grp.mean().unstack()
    pass_k = inside_grp.sum().unstack()
    pass_n = d[d.value.notna()].groupby(["generator", "metric"])["value"].count().unstack()
    nseeds = d.groupby("generator")["seed"].nunique()
    meta = df[(df.market == market) & (df.fit == fit) & (df.rule == "meta")]
    tmin = meta[meta.metric == "train_minutes"].groupby("generator")["value"].mean()
    info = load_meta()
    real_point, real_reps = _real_boot_reps(market, fit)
    rows = []
    for gen in [x for x in gens if x in mean.index]:
        dg = d[d.generator == gen]
        r = dict(generator=gen, name=pretty(gen), family_tag=info.get(gen, {}).get("family_tag", ""),
                 impl_source=info.get(gen, {}).get("impl_source", ""), seeds=int(nseeds[gen]),
                 train_minutes=float(tmin.get(gen, np.nan)))
        for m in ALL_METRICS:
            if m in mean.columns:
                vals = dg[dg.metric == m].sort_values("seed")["value"].to_numpy(dtype=float)
                ci_lo, ci_hi = _seed_boot_ci(vals, n_boot=1000, seed=0)
                b_lo, b_hi = _block_boot_ci(vals, ref_metrics.get(m), n_real=n_real, n_boot=1000, seed=0)
                sd_lo, sd_hi = _seed_date_ci(vals, real_point.get(m, np.nan), real_reps.get(m), seed=0)
                pk = int(pass_k.loc[gen, m]) if (m in pass_k.columns and pd.notna(pass_k.loc[gen, m])) else 0
                pn = int(pass_n.loc[gen, m]) if (m in pass_n.columns and pd.notna(pass_n.loc[gen, m])) else 0
                bin_lo, bin_hi = _binom_ci(pk, pn) if pn > 0 else (np.nan, np.nan)
                r[m] = mean.loc[gen, m]
                r[m + "_sd"] = sd.loc[gen, m] if m in sd.columns else np.nan
                r[m + "_min"] = vmin.loc[gen, m] if m in vmin.columns else np.nan
                r[m + "_max"] = vmax.loc[gen, m] if m in vmax.columns else np.nan
                r[m + "_ci_lo"] = ci_lo
                r[m + "_ci_hi"] = ci_hi
                r[m + "_seed_date_lo"] = sd_lo
                r[m + "_seed_date_hi"] = sd_hi
                r[m + "_block_ci_lo"] = b_lo
                r[m + "_block_ci_hi"] = b_hi
                r[m + "_err"] = err.loc[gen, m]
                r[m + "_err_sd"] = err_sd.loc[gen, m] if m in err_sd.columns else np.nan
                r[m + "_inside"] = inside.loc[gen, m] if m in inside.columns else np.nan
                r[m + "_pass_k"] = pk
                r[m + "_pass_n"] = pn
                r[m + "_binom_lo"] = bin_lo
                r[m + "_binom_hi"] = bin_hi
        for f in FAMILIES:
            k = f"family_{f}"
            if k in mean.columns:
                fvals = dg[dg.metric == k].sort_values("seed")["value"].to_numpy(dtype=float)
                fci_lo, fci_hi = _seed_boot_ci(fvals, n_boot=1000, seed=0)
                fsd_lo, fsd_hi = _seed_date_ci(fvals, np.nan, None, seed=0)
                r[f"score_{f}"] = mean.loc[gen, k]
                r[f"score_{f}_sd"] = sd.loc[gen, k] if k in sd.columns else np.nan
                r[f"score_{f}_ci_lo"] = fci_lo
                r[f"score_{f}_ci_hi"] = fci_hi
                r[f"score_{f}_seed_date_lo"] = fsd_lo
                r[f"score_{f}_seed_date_hi"] = fsd_hi
                r[f"score_{f}_min"] = vmin.loc[gen, k] if k in vmin.columns else np.nan
                r[f"score_{f}_max"] = vmax.loc[gen, k] if k in vmax.columns else np.nan
        r["n_inside"] = int(sum(r.get(m + "_inside", 0) >= 0.5 for m in ALL_METRICS if np.isfinite(r.get(m + "_inside", np.nan))))
        rows.append(r)
    t = pd.DataFrame(rows)
    for f in FAMILIES:
        if f"score_{f}" in t:
            t[f"rank_{f}"] = t[f"score_{f}"].rank(method="min").astype("Int64")   # NaN score (metric not produced) -> no rank
    return t


def table_md(t: pd.DataFrame, ref: dict, title: str) -> str:
    cols = ["corr_fro", "corr_mae", "pc1_share", "crisis_freq", "boost_pct", "cvar_ratio", "tail_coexc",
            "kurt_med", "acf1_sq", "disc_feat"]
    head = ["Model"] + cols + [f"{f} score (rank)" for f in FAMILIES] + ["seeds", "impl"]
    lines = [f"### {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    rm = ref["metrics"]
    real = ["**Real data** (95% band)"]
    for c in cols:
        r = rm[c]
        real.append(f"≤ {r['ceiling']:.3g}" if r["kind"] == "dist" else
                    f"{r['real']:.3g} [{r['lo']:.3g}, {r['hi']:.3g}]")
    lines.append("| " + " | ".join(real + ["–"] * len(FAMILIES) + ["–", "–"]) + " |")
    for _, r in t.iterrows():
        cells = [r["name"]]
        for c in cols:
            v = _fmt(r.get(c), r.get(c + "_sd"))
            cells.append(f"**{v}**" if r.get(c + "_inside", 0) >= 0.5 else v)
        cells += [f"{r[f'score_{f}']:.2f} ({int(r[f'rank_{f}'])})" if pd.notna(r[f"rank_{f}"]) else "n/p"
                  for f in FAMILIES]
        cells += [str(r["seeds"]), r["impl_source"]]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Bold = the seed-mean lies inside the real-data 95% band (stat metrics) or under the real-vs-real "
              "noise ceiling (distance metrics). n/p = the generator produced too few crisis-labelled windows for "
              "the metric to exist. Scores are mean errors in units of real sampling noise (lower is better)."]
    return "\n".join(lines)


def multiseed_table(df: pd.DataFrame, markets: tuple[str, ...] = ("nse", "us", "kr")) -> tuple[pd.DataFrame, str]:
    """Aggregated multi-seed table across fits: mean ± SD, 95% CIs, and per-seed pass counts for all 18 metrics."""
    frames = []
    md_sections = [
        "# T6: Multi-Seed Aggregation (Mean ± SD, 95% Bootstrap CIs & Per-Seed Pass Counts)",
        "",
        "Every generator is evaluated across 5 independent random seeds (`seed0`..`seed4`). "
        "Below are the seed-level mean ± SD, 95% seed-resample and block-bootstrap intervals, "
        "and per-seed pass counts ($k/n$ seeds inside the real 95% band or under the real-vs-real ceiling, "
        "with exact Clopper-Pearson 95% binomial CIs).",
    ]
    for market in markets:
        fits = [f for f in ("full", "pre2020") if f in set(df[df.market == market].fit)]
        for fit in fits:
            gens = [g for g in MAIN if g in set(df[(df.market == market) & (df.fit == fit)].generator)]
            if not gens:
                continue
            t = main_table(df, market, fit, gens)
            if t.empty:
                continue
            t.insert(0, "market", market)
            t.insert(1, "fit", fit)
            frames.append(t)
            ref = reference(market, fit)
            rm = ref["metrics"]
            fit_label = "In-Sample (`full`, 2010–2024)" if fit == "full" else "Out-of-Sample (`pre2020`, scored on 2020–2024)"
            md_sections += ["", f"## {market.upper()} — {fit_label} ({ref['n_windows']} real windows, {ref['n_crisis']} crisis)", ""]
            for fam in FAMILIES:
                ms = P.METRIC_FAMILIES[fam]
                head = ["Model", "Seeds"] + [f"{m} (mean ± SD)" for m in ms] + [f"{m} [block 95% CI]" for m in ms] + [
                    f"{m} pass (k/n)" for m in ms
                ] + [f"Family {fam} score ± SD [95% CI]", f"Rank {fam}"]
                md_sections += [f"### Family {fam} Metrics", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                real_row = ["**Real reference**", "–"]
                for m in ms:
                    r_m = rm[m]
                    real_row.append(f"≤ {r_m['ceiling']:.3g}" if r_m["kind"] == "dist" else
                                    f"{r_m['real']:.3g} [{r_m['lo']:.3g}, {r_m['hi']:.3g}]")
                real_row += ["–"] * (2 * len(ms) + 2)
                md_sections.append("| " + " | ".join(real_row) + " |")
                for _, row in t.iterrows():
                    cells = [row["name"], str(int(row["seeds"]))]
                    for m in ms:
                        v = _fmt(row.get(m), row.get(m + "_sd"))
                        cells.append(f"**{v}**" if row.get(m + "_inside", 0) >= 0.5 else v)
                    for m in ms:
                        blo, bhi = row.get(m + "_block_ci_lo", np.nan), row.get(m + "_block_ci_hi", np.nan)
                        cells.append(f"[{blo:.3g}, {bhi:.3g}]" if (np.isfinite(blo) and np.isfinite(bhi)) else "n/p")
                    for m in ms:
                        pk, pn = int(row.get(m + "_pass_k", 0)), int(row.get(m + "_pass_n", 0))
                        blo, bhi = row.get(m + "_binom_lo", np.nan), row.get(m + "_binom_hi", np.nan)
                        cells.append(f"{pk}/{pn} [{blo:.2f}, {bhi:.2f}]" if pn > 0 else "0/0")
                    sf, sfsd = row.get(f"score_{fam}", np.nan), row.get(f"score_{fam}_sd", np.nan)
                    sflo, sfhi = row.get(f"score_{fam}_ci_lo", np.nan), row.get(f"score_{fam}_ci_hi", np.nan)
                    if np.isfinite(sf):
                        cells.append(f"{_fmt(sf, sfsd, nd=3)} [{sflo:.2f}, {sfhi:.2f}]")
                    else:
                        cells.append("n/p")
                    rk = row.get(f"rank_{fam}")
                    cells.append(str(int(rk)) if pd.notna(rk) else "n/p")
                    md_sections.append("| " + " | ".join(cells) + " |")
                md_sections.append("")
    out_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out_df, "\n".join(md_sections)


# ------------------------------------------------------- rank agreement (H1)
def seed_scores(df: pd.DataFrame, market: str, fit: str, gens: list[str], rule: str = "main") -> pd.DataFrame:
    d = df[(df.market == market) & (df.fit == fit) & (df.rule == rule) & df.generator.isin(gens)
           & df.metric.str.startswith("family_")]
    return d.pivot_table(index=["generator", "seed"], columns="metric", values="value")


def kendall(a, b) -> float:
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr)
    if mask.sum() < 2:
        return np.nan
    t = stats.kendalltau(a_arr[mask], b_arr[mask])
    return float(t.statistic) if np.isfinite(t.statistic) else np.nan


def _kendall_batch(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Vectorised Kendall tau-b for 2D arrays X, Y of shape (B, G)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    B, G = X.shape
    if G < 2:
        return np.full(B, np.nan)
    valid = np.isfinite(X) & np.isfinite(Y)
    if not valid.all():
        return np.array([kendall(X[b], Y[b]) for b in range(B)], dtype=float)
    ii, jj = np.triu_indices(G, k=1)
    dx = np.sign(X[:, ii] - X[:, jj])
    dy = np.sign(Y[:, ii] - Y[:, jj])
    num = (dx * dy).sum(axis=1)
    den = np.sqrt((dx != 0).sum(axis=1, dtype=float) * (dy != 0).sum(axis=1, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _seed_date_family_tau(df: pd.DataFrame, market: str, fit: str, gens: list[str],
                          fa: str = "B", fb: str = "C", n_boot: int = 200,
                          seed: int = 100, rule: str = "main") -> tuple[float, float]:
    """Joint seed x date-block bootstrap 95% CI for Kendall tau between family fa and family fb."""
    ref_path = RESULTS / "reference" / market / f"{fit}.json"
    if not ref_path.exists():
        return (np.nan, np.nan)
    ref_doc = json.loads(ref_path.read_text())
    ref_m = ref_doc.get("metrics" if rule == "main" else "metrics_alt_rule", {})
    ms_a = [m for m in P.METRIC_FAMILIES[fa] if m in ref_m]
    ms_b = [m for m in P.METRIC_FAMILIES[fb] if m in ref_m]
    all_m = list(dict.fromkeys(ms_a + ms_b))
    d = df[(df.market == market) & (df.fit == fit) & (df.rule == rule) & df.generator.isin(gens) & df.metric.isin(all_m)]
    if d.empty:
        return (np.nan, np.nan)
    piv = d.pivot_table(index=["generator", "seed"], columns="metric", values="value")
    for m in all_m:
        if m not in piv.columns:
            piv[m] = np.nan
    piv = piv[all_m]
    valid_gens = [g for g in gens if g in piv.index.get_level_values(0)]
    if len(valid_gens) < 2:
        return (np.nan, np.nan)
    groups = [piv.loc[g].to_numpy(dtype=float) for g in valid_gens]
    idx_a = np.array([all_m.index(m) for m in ms_a], dtype=int)
    idx_b = np.array([all_m.index(m) for m in ms_b], dtype=int)
    rng = np.random.default_rng(seed)
    M_cnt = len(all_m)
    kinds = [ref_m[m]["kind"] for m in all_m]
    r_vals = np.empty((n_boot, M_cnt), dtype=float)
    denoms = np.empty((n_boot, M_cnt), dtype=float)
    for j, m in enumerate(all_m):
        rm = ref_m[m]
        kind = rm["kind"]
        if kind == "stat":
            se = float(rm["half"]) / 1.959963984540054
            r_vals[:, j] = float(rm["real"]) + rng.normal(0.0, se, size=n_boot)
            denoms[:, j] = max(float(rm["half"]), 1e-12)
        elif kind == "dist":
            ceil = max(float(rm["ceiling"]), 1e-12)
            med = float(rm.get("median", 0.5 * ceil))
            se = max(ceil - med, 0.0) / 1.6448536269514722
            r_vals[:, j] = 0.0
            denoms[:, j] = np.clip(ceil + rng.normal(0.0, se, size=n_boot), 1e-12, None)
        else:
            reps = rm.get("reps") or [rm["hi"]]
            hi_b = rng.choice(reps, size=n_boot).astype(float)
            r_vals[:, j] = 0.5
            denoms[:, j] = np.clip(hi_b - 0.5, 0.02, None)
    G_cnt = len(groups)
    smeans = np.empty((n_boot, G_cnt, M_cnt), dtype=float)
    for gi, g_arr in enumerate(groups):
        n_s = len(g_arr)
        s_idx = rng.integers(0, n_s, size=(n_boot, n_s))
        smeans[:, gi, :] = g_arr[s_idx].mean(axis=1)
    errs = np.empty((n_boot, G_cnt, M_cnt), dtype=float)
    for j, kind in enumerate(kinds):
        v = smeans[:, :, j]
        rv = r_vals[:, None, j]
        dn = denoms[:, None, j]
        if kind == "stat":
            raw_e = np.abs(v - rv) / dn
        elif kind == "dist":
            raw_e = v / dn
        else:
            raw_e = np.clip(v - 0.5, 0.0, None) / dn
        errs[:, :, j] = np.where(np.isfinite(v), np.minimum(raw_e, P.NORM_ERR_CAP), P.NORM_ERR_CAP)
    sa = errs[:, :, idx_a].mean(axis=2)
    sb = errs[:, :, idx_b].mean(axis=2)
    taus = _kendall_batch(sa, sb)
    taus = taus[np.isfinite(taus)]
    if taus.size == 0:
        return (np.nan, np.nan)
    return (float(np.quantile(taus, 0.025)), float(np.quantile(taus, 0.975)))


def family_tau(df, market, fit, gens, fa="B", fb="C", n_boot=2000, seed=0, rule="main") -> dict:
    """Kendall tau between two family rankings, with seed-resampling and seed x date-block bootstrap intervals."""
    ss = seed_scores(df, market, fit, gens, rule)
    if ss.empty or f"family_{fa}" not in ss or f"family_{fb}" not in ss:
        return {}
    mean = ss.groupby("generator").mean()
    tau = kendall(mean[f"family_{fa}"], mean[f"family_{fb}"])
    rng = np.random.default_rng(seed)
    groups = [x.to_numpy(dtype=float) for _, x in ss[[f"family_{fa}", f"family_{fb}"]].groupby("generator")]
    G = len(groups)
    M = np.empty((n_boot, G, 2), dtype=float)
    for b in range(n_boot):
        for gi, x in enumerate(groups):
            M[b, gi] = x[rng.integers(0, len(x), len(x))].mean(0)
    reps = _kendall_batch(M[:, :, 0], M[:, :, 1])
    reps = reps[np.isfinite(reps)]
    if not np.isfinite(tau) or len(reps) == 0:      # a family score is missing (e.g. no crisis windows at all)
        return {}
    # leave-one-generator-out: is the conclusion carried by a single model?
    loo = {g: kendall(mean.drop(g)[f"family_{fa}"], mean.drop(g)[f"family_{fb}"]) for g in mean.index}
    sd_lo, sd_hi = _seed_date_family_tau(df, market, fit, list(mean.index), fa, fb,
                                         n_boot=min(n_boot, 500), seed=seed + 100, rule=rule)
    return dict(tau=tau, lo=float(np.quantile(reps, 0.025)), hi=float(np.quantile(reps, 0.975)),
                seed_date_lo=sd_lo, seed_date_hi=sd_hi,
                n_generators=int(len(mean)), loo_min=float(min(loo.values())), loo_max=float(max(loo.values())),
                loo=loo)


def family_tau_table(taus: dict[str, dict], tau_bc_prereg: dict | None = None) -> pd.DataFrame:
    """Table of pairwise family rank correlations with seed-bootstrap and seed x date-block CIs."""
    rows = []
    if tau_bc_prereg:
        rows.append(dict(pair="B_vs_C_preregistered", fa="B", fb="C",
                         tau=tau_bc_prereg.get("tau"), lo=tau_bc_prereg.get("lo"), hi=tau_bc_prereg.get("hi"),
                         seed_date_lo=tau_bc_prereg.get("seed_date_lo"), seed_date_hi=tau_bc_prereg.get("seed_date_hi"),
                         loo_min=tau_bc_prereg.get("loo_min"), loo_max=tau_bc_prereg.get("loo_max"),
                         n_generators=tau_bc_prereg.get("n_generators")))
    for k, v in taus.items():
        if not v:
            continue
        fa, fb = k.split("_vs_")
        rows.append(dict(pair=k, fa=fa, fb=fb, tau=v.get("tau"), lo=v.get("lo"), hi=v.get("hi"),
                         seed_date_lo=v.get("seed_date_lo"), seed_date_hi=v.get("seed_date_hi"),
                         loo_min=v.get("loo_min"), loo_max=v.get("loo_max"), n_generators=v.get("n_generators")))
    return pd.DataFrame(rows)


def metric_tau_matrix(t: pd.DataFrame) -> pd.DataFrame:
    """Kendall tau between the generator rankings induced by each pair of metrics (normalised errors)."""
    ms = [m for m in ALL_METRICS if m + "_err" in t]
    E = t.set_index("generator")[[m + "_err" for m in ms]]
    out = pd.DataFrame(np.eye(len(ms)), index=ms, columns=ms)
    for a, b in combinations(ms, 2):
        out.loc[a, b] = out.loc[b, a] = kendall(E[a + "_err"], E[b + "_err"])
    return out


# ------------------------------------------------------------------- gates
def gate_g1(t: pd.DataFrame, tau: dict, model: str = "cgnsde",
            df: pd.DataFrame | None = None, market: str = "nse", fit: str = "full") -> tuple[str, dict]:
    """Preregistered for model = 'cgnsde' (v1).  Called again with 'cgnsde2' as a post hoc evaluation."""
    s = t.set_index("generator")
    static = [g for g in P.STATIC if g in s.index]
    regime = [g for g in ("regime_gaussian", "hmm_gaussian", model) if g in s.index]
    a = bool(static and regime and all(s.loc[r, "score_C"] < s.loc[st, "score_C"] for r in regime for st in static))
    h2 = None
    detail = {}
    score_c_multiseed = {}
    for g in [*static, *(r for r in ("regime_gaussian", "hmm_gaussian", model) if r in s.index)]:
        if g in s.index:
            m_val = float(s.loc[g, "score_C"])
            sd_val = float(s.loc[g, "score_C_sd"]) if np.isfinite(s.loc[g, "score_C_sd"]) else 0.0
            ci_lo = float(s.loc[g, "score_C_ci_lo"]) if "score_C_ci_lo" in s.columns and np.isfinite(s.loc[g, "score_C_ci_lo"]) else m_val
            ci_hi = float(s.loc[g, "score_C_ci_hi"]) if "score_C_ci_hi" in s.columns and np.isfinite(s.loc[g, "score_C_ci_hi"]) else m_val
            score_c_multiseed[g] = dict(
                name=pretty(g), mean=m_val, sd=sd_val, formatted=f"{m_val:.3f} ± {sd_val:.3f}",
                ci_95=[ci_lo, ci_hi], seeds=int(s.loc[g, "seeds"]),
            )
    if model in s.index:
        cg, sd = float(s.loc[model, "score_C"]), float(s.loc[model, "score_C_sd"])
        sd = 0.0 if not np.isfinite(sd) else sd
        for r in ("regime_gaussian", "hmm_gaussian"):
            if r in s.index:
                r_mean = float(s.loc[r, "score_C"])
                r_sd = float(s.loc[r, "score_C_sd"]) if np.isfinite(s.loc[r, "score_C_sd"]) else 0.0
                detail[r] = dict(
                    score_C=r_mean, score_C_sd=r_sd,
                    score_C_formatted=f"{r_mean:.3f} ± {r_sd:.3f}",
                    cgnsde=cg, cgnsde_sd=sd,
                    cgnsde_formatted=f"{cg:.3f} ± {sd:.3f}",
                    matches=bool(r_mean <= cg + sd),
                )
        h2 = any(v["matches"] for v in detail.values()) if detail else None
    base_verdict = "PASS" if (a and h2 is False) else "FAIL"
    h1 = None if not tau else bool(tau["tau"] < 0.3)

    # Per-seed evaluation of Gate G1 across the 5 seeds
    multiseed_gate = {}
    if df is not None and static and regime and model in s.index:
        ss = seed_scores(df, market, fit, list(dict.fromkeys(static + regime)))
        if not ss.empty and "family_C" in ss:
            piv = ss["family_C"].unstack(level=0)
            cg_sd = float(s.loc[model, "score_C_sd"]) if np.isfinite(s.loc[model, "score_C_sd"]) else 0.0
            per_seed = {}
            for sd_idx, row in piv.iterrows():
                if any(pd.isna(row.get(g)) for g in static + regime):
                    continue
                s_a = all(float(row[r]) < float(row[st]) for r in regime for st in static)
                s_h2 = any(float(row[r]) <= float(row[model]) + cg_sd for r in ("regime_gaussian", "hmm_gaussian") if r in row)
                per_seed[int(sd_idx)] = bool(s_a and not s_h2)
            k_pass = sum(per_seed.values())
            n_tot = len(per_seed)
            b_lo, b_hi = _binom_ci(k_pass, n_tot)
            multiseed_gate = dict(
                seeds_passed=int(k_pass), n_seeds=int(n_tot),
                _seeds_passed=int(k_pass), _seeds_total=int(n_tot),
                all_seeds_pass=bool(n_tot > 0 and k_pass == n_tot),
                pass_proportion=float(k_pass / n_tot) if n_tot > 0 else np.nan,
                binomial_ci_95=[b_lo, b_hi],
                binomial_ci=[b_lo, b_hi],
                per_seed_pass=per_seed,
            )

    all_seeds_ok = multiseed_gate.get("all_seeds_pass", False) if multiseed_gate else True
    multiseed_verdict = "PASS" if (base_verdict == "PASS" and all_seeds_ok) else "FAIL"
    verdict = multiseed_verdict

    return verdict, dict(
        regime_beats_static=a, H2_simple_regime_model_matches_cgnsde=h2, H2_detail=detail,
        score_C_multiseed=score_c_multiseed, multiseed_gate=multiseed_gate,
        multiseed_verdict=multiseed_verdict, aggregate_verdict=base_verdict,
        H1_tau_B_vs_C=tau.get("tau"), H1_ci=[tau.get("lo"), tau.get("hi")],
        H1_seed_date_ci=[tau.get("seed_date_lo"), tau.get("seed_date_hi")], H1_holds=h1,
    )


def write_gate(name: str, verdict: str, body: dict, text: str) -> None:
    GATES.mkdir(parents=True, exist_ok=True)
    gate_key = next((p for p in name.split("_") if p in P.GATES), name.split("_")[-1])
    extra_md = []
    if "score_C_multiseed" in body and body["score_C_multiseed"]:
        extra_md += ["", "### Multi-Seed Family-C Scores (Mean ± SD over Seeds)", "",
                     "| Generator | Family C Score (mean ± SD) | 95% Seed Bootstrap CI | Seeds |",
                     "|---|---|---|---|"]
        for _, info in body["score_C_multiseed"].items():
            ci = info.get("ci_95", [np.nan, np.nan])
            extra_md.append(f"| {info['name']} | {info['formatted']} | [{ci[0]:.3f}, {ci[1]:.3f}] | {info['seeds']} |")
    if "multiseed_gate" in body and body["multiseed_gate"]:
        mg = body["multiseed_gate"]
        bci = mg.get("binomial_ci_95") or mg.get("binomial_ci") or [np.nan, np.nan]
        sp = mg.get("_seeds_passed", mg.get("seeds_passed", 0))
        st = mg.get("_seeds_total", mg.get("n_seeds", 0))
        extra_md += [
            "",
            f"**Multi-seed gate consistency:** {sp}/{st} seeds pass "
            f"(proportion = {mg['pass_proportion']:.2f}, 95% Clopper-Pearson binomial CI [{bci[0]:.3f}, {bci[1]:.3f}], "
            f"all seeds pass = `{mg['all_seeds_pass']}`).",
        ]
    lines = [f"# {name}: {verdict}", "", f"Rule (frozen in crisisbench/protocol.py): {P.GATES.get(gate_key, '')}",
             "", text, *extra_md, "", "```json", json.dumps(body, indent=2, default=float), "```", "",
             "_Generated by crisisbench.analysis. Do not edit by hand._"]
    (GATES / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")


def save(t: pd.DataFrame, name: str, md: str | None = None) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABLES / f"{name}.csv", index=False)
    (TABLES / f"{name}.md").write_text(md if md is not None else t.to_markdown(index=False, floatfmt=".4g"),
                                       encoding="utf-8")


# -------------------------------------------------------------------- main
def run_all() -> dict:
    df = load_runs()
    summary: dict = {"protocol_version": P.PROTOCOL_VERSION}
    if df.empty:
        return summary
    from .analysis_extra import (contamination_table, enrich_t1_seed_distributions,
                                 family_tau_tables, heldout_gate_details, heldout_table,
                                 hypotheses, t1alt_markdown)
    markets = sorted(df.market.unique())
    for mk in markets:
        for fit in sorted(df[df.market == mk].fit.unique()):
            if fit.startswith("upto") and mk != "nse_ext":
                continue
            gens = [g for g in MAIN if g in set(df[(df.market == mk) & (df.fit == fit)].generator)]
            if not gens:
                continue
            ref = reference(mk, fit)
            t = main_table(df, mk, fit, gens)
            t = enrich_t1_seed_distributions(t, df, mk, fit)
            title = f"Main comparison, market {mk}, fit '{fit}' ({ref['n_windows']} real windows, {ref['n_crisis']} crisis)"
            save(t, f"T1_{mk}_{fit}", table_md(t, ref, title))
            # H1 was preregistered on the 11 registered generators: post hoc v2 and _fs are left out of the
            # preregistered tau, and the value across all scored generators (including v2) is reported alongside.
            prereg_gens = [g for g in gens if g in PREREGISTERED]
            tau_bc = family_tau(df, mk, fit, prereg_gens if len(prereg_gens) >= 2 else gens, "B", "C")
            tau_bc_v2 = family_tau(df, mk, fit, gens, "B", "C") if set(gens) != set(prereg_gens) else {}
            taus = {f"{a}_vs_{b}": family_tau(df, mk, fit, gens, a, b, n_boot=500)
                    for a, b in combinations(FAMILIES, 2)}
            ft_df = family_tau_table(taus, tau_bc)
            if not ft_df.empty:
                save(ft_df, f"F1_family_tau_{mk}_{fit}")
            metric_tau_matrix(t).to_csv(TABLES / f"F2_metric_tau_{mk}_{fit}.csv")
            alt = main_table(df, mk, fit, gens, rule="alt_rule")
            if not alt.empty:
                alt = enrich_t1_seed_distributions(alt, df, mk, fit, rule="alt_rule")
                save(alt, f"T1alt_{mk}_{fit}", t1alt_markdown(alt))
            summary[f"{mk}/{fit}"] = dict(
                generators=gens, preregistered_generators=prereg_gens,
                tau_B_vs_C=tau_bc, tau_B_vs_C_with_v2=tau_bc_v2,
                family_taus={k: v.get("tau") for k, v in taus.items()},
                family_taus_ci=taus,
                family_tau_intervals=taus,
                tau_B_vs_C_alt_rule=family_tau(df, mk, fit, gens, "B", "C", n_boot=500, rule="alt_rule").get("tau"),
                ranks={f: t.set_index("generator")[f"rank_{f}"].to_dict() for f in FAMILIES if f"rank_{f}" in t},
                scores={f: t.set_index("generator")[f"score_{f}"].to_dict() for f in FAMILIES if f"score_{f}" in t},
                scores_sd={f: t.set_index("generator")[f"score_{f}_sd"].to_dict() for f in FAMILIES if f"score_{f}_sd" in t},
                real={k: v for k, v in ref["metrics"].items()})
            if mk == "nse":
                verdict, body = gate_g1(t, tau_bc, "cgnsde", df=df, market=mk, fit=fit)
                summary[f"G1/{fit}"] = dict(verdict=verdict, **body)
                write_gate(f"GATE_G1_{fit}", verdict, body,
                           f"Evaluated on market nse, fit '{fit}', {len(gens)} generators.")
                if "cgnsde2" in set(t.generator):
                    verdict2, body2 = gate_g1(t, tau_bc, "cgnsde2", df=df, market=mk, fit=fit)
                    summary[f"G1v2/{fit}"] = dict(verdict=verdict2, **body2)
                    write_gate(f"GATE_G1_{fit}_v2_posthoc", verdict2, body2,
                               f"POST HOC: same gate evaluated for CG-NSDE v2. Market nse, fit '{fit}'.")
            abl = [g for g in sorted(set(df[(df.market == mk) & (df.fit == fit)].generator))
                   if g == "cgnsde2" or g.startswith("cgnsde2@")]
            if len(abl) > 1:
                ta = main_table(df, mk, fit, abl)
                save(ta, f"T2_ablation_{mk}_{fit}", table_md(ta, ref, f"CG-NSDE ablations, market {mk}, fit '{fit}'"))
    t6_df, t6_md = multiseed_table(df, ("nse",))
    if not t6_df.empty:
        save(t6_df, "T6_multiseed", t6_md)
    t6f_df, t6f_md = family_tau_tables(df, summary)
    if not t6f_df.empty:
        save(t6f_df, "T6_family_taus", t6f_md)
    g2 = None
    if "nse/pre2020" in summary:
        t3, g2 = heldout_table(df)
        if not t3.empty:
            save(t3, "T3_heldout")
        g2v2 = {k[3:]: v for k, v in (g2 or {}).items() if k.startswith("v2:")}
        g2 = {k: v for k, v in (g2 or {}).items() if not k.startswith("v2:")}
        if g2:
            body_g2, md_g2 = heldout_gate_details(t3, g2, "cgnsde",
                                                  "CG-NSDE v1 (fit on data to 2019) boost_pct against each held-out episode's real 95% interval.", df=df)
            write_gate("GATE_G2", "PASS" if (sum(g2.values()) >= 2 and body_g2.get("multiseed_all_seeds_pass", False)) else "FAIL", body_g2, md_g2)
        if g2v2:
            body_g2v2, md_g2v2 = heldout_gate_details(t3, g2v2, "cgnsde2",
                                                       "POST HOC: CG-NSDE v2 (fit on data to 2019) boost_pct against each held-out episode's interval.", df=df)
            write_gate("GATE_G2_v2_posthoc", "PASS" if (sum(g2v2.values()) >= 2 and body_g2v2.get("multiseed_all_seeds_pass", False)) else "FAIL", body_g2v2, md_g2v2)
    else:
        g2v2 = None
    t7, h5 = contamination_table(df)
    if not t7.empty:
        save(t7, "T7_contamination")
    hyp = hypotheses(summary, g2, h5, g2v2)
    (TABLES / "hypotheses.json").write_text(json.dumps(hyp, indent=2, default=float), encoding="utf-8")
    (TABLES / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return summary


if __name__ == "__main__":
    s = run_all()
    for k, v in s.items():
        if isinstance(v, dict) and "tau_B_vs_C" in v and v["tau_B_vs_C"]:
            t = v["tau_B_vs_C"]
            print(f"{k}: tau(B,C) = {t['tau']:.2f} [{t['lo']:.2f}, {t['hi']:.2f}] over {t['n_generators']} generators")
        if k.startswith("G1"):
            print(k, v["verdict"])
