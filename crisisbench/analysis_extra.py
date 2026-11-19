"""Held-out episodes (T3, gate G2), contamination (T7, H5) and hypothesis verdicts."""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd

from . import protocol as P

EPISODE_LABEL = {"covid_2020": "COVID crash, 2020", "shock_2022": "2022 rate and war shock",
                 "mid_2024": "Mid-2024 election shock"}


def heldout_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-episode real crisis boost (block-bootstrap interval) against each generator's boost."""
    from .analysis import MAIN, load_meta, pretty
    from .bench import make_fit_data
    from . import metrics as M
    fd, W, wend = make_fit_data("nse", "pre2020")
    ws = M.window_stats(W, fd.rule)
    lab, mac = ws.labels, ws.mac
    normal = mac[lab == 0]
    rng = np.random.default_rng(2024)
    d = df[(df.market == "nse") & (df.fit == "pre2020") & (df.rule == "main") & (df.metric == "boost_pct")]
    gens = [g for g in MAIN if g in set(d.generator)]
    info = load_meta()
    rows, verdict = [], {}
    for ep, (a, b) in P.HELDOUT_EPISODES.items():
        sel = (lab == 1) & np.asarray((wend >= a) & (wend <= b))
        if sel.sum() < 5:
            continue
        cr = mac[sel]
        real = 100 * (cr.mean() / normal.mean() - 1)
        reps = [100 * (cr[M.block_bootstrap_idx(len(cr), rng, 20)].mean()
                       / normal[M.block_bootstrap_idx(len(normal), rng, P.BOOT_BLOCK)].mean() - 1)
                for _ in range(2000)]
        lo, hi = np.quantile(reps, [0.025, 0.975])
        for g in gens:
            v = d[d.generator == g]["value"].astype(float)
            gm, gs = (v.mean(), v.std(ddof=1)) if v.notna().any() else (np.nan, np.nan)
            inside = bool(np.isfinite(gm) and lo <= gm <= hi)
            rows.append(dict(episode=ep, episode_label=EPISODE_LABEL.get(ep, ep), n_real_crisis=int(sel.sum()),
                             real=real, lo=lo, hi=hi, generator=g, name=pretty(g),
                             family_tag=info.get(g, {}).get("family_tag", ""), gen=gm, gen_sd=gs, inside=inside))
            if g == "cgnsde":
                verdict[ep] = inside
            if g == "cgnsde2":
                verdict["v2:" + ep] = inside
    return pd.DataFrame(rows), verdict


def contamination_table(df: pd.DataFrame) -> tuple[pd.DataFrame, bool | None]:
    """How injected data errors move the real reference and flip generator pair orderings."""
    from .analysis import ALL_METRICS, RESULTS, pretty, reference
    rows, tick_flip = [], None
    clean = df[(df.market == "nse") & (df.fit == "full") & (df.rule == "main")]
    panels = (("nse_c_tick", "one bad tick (-76%)"), ("nse_c_1pct", "1% bad ticks"),
              ("nse_c_split", "one unadjusted split"))
    for mk, label in panels:
        dirty = df[(df.market == mk) & (df.fit == "full") & (df.rule == "main")]
        if dirty.empty or not (RESULTS / "reference" / mk / "full.json").exists():
            continue
        rc, rd = reference("nse", "full")["metrics"], reference(mk, "full")["metrics"]
        gens = sorted(set(dirty.generator) & set(clean.generator))
        seeds = sorted(set(dirty.seed))
        n_before = len(rows)
        for m in ALL_METRICS:
            a = clean[(clean.metric == m) & clean.generator.isin(gens) & clean.seed.isin(seeds)].groupby("generator")["err"].mean()
            b = dirty[(dirty.metric == m) & dirty.generator.isin(gens)].groupby("generator")["err"].mean()
            if len(a) < 2 or len(b) < 2:
                continue
            flips = [(x, y) for x, y in combinations(gens, 2)
                     if (a[x] - a[y]) * (b[x] - b[y]) < 0 and abs(a[x] - a[y]) > 1e-9]
            key = "ceiling" if rc[m]["kind"] == "dist" else "real"
            rows.append(dict(panel=label, metric=m, real_clean=rc[m].get(key), real_contaminated=rd[m].get(key),
                             pair_flips=len(flips), pairs=len(gens) * (len(gens) - 1) // 2,
                             example=" vs ".join(pretty(g) for g in flips[0]) if flips else ""))
        if mk == "nse_c_tick":
            tick_flip = any(r["pair_flips"] > 0 for r in rows[n_before:])
    return pd.DataFrame(rows), tick_flip


def hypotheses(summary: dict, g2: dict | None, h5: bool | None, g2v2: dict | None = None) -> dict:
    from .analysis import RESULTS, pretty

    def tau(mk, fit):
        return (summary.get(f"{mk}/{fit}") or {}).get("tau_B_vs_C") or {}

    def show(t):
        return f"{t['tau']:.2f} [{t['lo']:.2f}, {t['hi']:.2f}]"

    H: dict = {}
    t_full, t_pre = tau("nse", "full"), tau("nse", "pre2020")
    if t_full:
        ts = [t for t in (t_full, t_pre) if t]
        ev = f"NSE in-sample tau(B,C) = {show(t_full)} over {t_full['n_generators']} models"
        if t_pre:
            ev += f"; held-out tau(B,C) = {show(t_pre)}"
        if all(t["hi"] < 0.3 for t in ts):
            v = "supported"
        elif all(t["tau"] < 0.3 for t in ts):
            v = "inconclusive (point estimate below 0.3, interval crosses it)"
        else:
            v = "not supported"
        H["H1"] = dict(verdict=v, evidence=ev)
    g1 = summary.get("G1/full")
    if g1 and g1.get("H2_simple_regime_model_matches_cgnsde") is not None:
        H["H2"] = dict(verdict="supported" if g1["H2_simple_regime_model_matches_cgnsde"] else "not supported",
                       evidence="; ".join(f"{pretty(k)} crisis score {v['score_C']:.2f} vs CG-NSDE "
                                          f"{v['cgnsde']:.2f} ± {v['cgnsde_sd']:.2f}" for k, v in g1["H2_detail"].items()))
        H["G1"] = dict(verdict=g1["verdict"])
    p = RESULTS / "06_risk" / "G3.json"
    if p.exists():
        b = json.loads(p.read_text())
        H["H3"] = dict(verdict="supported" if b["H3_holds"] else "not supported",
                       evidence=f"tau(crisis family, backtest rank) = {b['tau_C']:.2f} [{b['ci_C'][0]:.2f}, {b['ci_C'][1]:.2f}]; "
                                f"tau(correlation family, backtest rank) = {b['tau_B']:.2f} [{b['ci_B'][0]:.2f}, {b['ci_B'][1]:.2f}]")
        H["G3"] = dict(verdict="PASS" if b["H3_holds"] else "FAIL")
    other = {m: tau(m, "full") for m in ("us", "kr") if tau(m, "full")}
    if other and t_full:
        ok = any(t["tau"] < 0.3 for t in other.values())
        H["H4"] = dict(verdict="supported" if ok else "not supported",
                       evidence="; ".join(f"{m.upper()}: tau(B,C) = {show(t)}" for m, t in other.items()))
        H["G4"] = dict(verdict="PASS" if ok else "FAIL")
        H["KILL"] = dict(verdict="triggered" if all(t["tau"] >= 0.3 for t in [t_full, *other.values()]) else "not triggered")
    if h5 is not None:
        H["H5"] = dict(verdict="supported" if h5 else "not supported",
                       evidence="pair-order flips after one injected bad tick: see the contamination table")
    if g2:
        n = sum(g2.values())
        H["G2"] = dict(verdict="PASS" if n >= 2 else "FAIL",
                       evidence=f"CG-NSDE v1 boost inside the real episode interval on {n} of {len(g2)} episodes")
    p2 = RESULTS / "06_risk" / "G3_with_v2.json"
    if p2.exists():
        b = json.loads(p2.read_text())
        H["H3 (with v2, post hoc)"] = dict(
            verdict="supported" if b["H3_holds"] else "not supported",
            evidence=f"tau(crisis family, backtest rank) = {b['tau_C']:.2f} [{b['ci_C'][0]:.2f}, {b['ci_C'][1]:.2f}]; "
                     f"tau(correlation family, backtest rank) = {b['tau_B']:.2f} [{b['ci_B'][0]:.2f}, {b['ci_B'][1]:.2f}]; "
                     f"CG-NSDE v2 FZ0 rank {b['fz0_rank'].get('cgnsde2')} of {len(b['generators'])}")
    # Post hoc: the same checks for CG-NSDE v2 (designed after the v1 results; not preregistered).
    g1v2 = summary.get("G1v2/full")
    if g1v2 and g1v2.get("H2_simple_regime_model_matches_cgnsde") is not None:
        H["H2 (v2, post hoc)"] = dict(
            verdict="supported" if g1v2["H2_simple_regime_model_matches_cgnsde"] else "not supported",
            evidence="; ".join(f"{pretty(k)} crisis score {v['score_C']:.2f} vs CG-NSDE v2 "
                               f"{v['cgnsde']:.2f} ± {v['cgnsde_sd']:.2f}" for k, v in g1v2["H2_detail"].items()))
        H["G1 (v2, post hoc)"] = dict(verdict=g1v2["verdict"])
    if g2v2:
        n = sum(g2v2.values())
        H["G2 (v2, post hoc)"] = dict(verdict="PASS" if n >= 2 else "FAIL",
                                      evidence=f"CG-NSDE v2 boost inside the real episode interval on {n} of {len(g2v2)} episodes")
    return H
