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
    from .analysis import MAIN, _binom_ci, _seed_boot_ci, load_meta, pretty
    from .bench import make_fit_data
    from .labels import label_windows
    from . import metrics as M
    fd, W, wend = make_fit_data("nse", "pre2020")
    W = np.asarray(W, dtype=np.float64)
    n_w, T, N = W.shape
    mu = W.sum(1) / T
    cov = np.einsum("ntj,ntk->njk", W, W, optimize=True) / T - mu[:, :, None] * mu[:, None, :]
    sd = np.sqrt(np.clip(np.einsum("njj->nj", cov), 1e-20, None))
    C = np.nan_to_num(cov / sd[:, :, None] / sd[:, None, :], nan=0.0)
    off = ~np.eye(N, dtype=bool)
    mac = np.abs(C[:, off]).mean(1)
    lab, _ = label_windows(W, fd.rule)
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
            v = d[d.generator == g].sort_values("seed")["value"].astype(float)
            vc = v.dropna().to_numpy()
            gm, gs = (v.mean(), v.std(ddof=1)) if v.notna().any() else (np.nan, np.nan)
            ci_lo, ci_hi = _seed_boot_ci(vc, n_boot=1000, seed=0)
            seed_in = (vc >= lo) & (vc <= hi)
            k_in, n_in = int(seed_in.sum()), int(len(vc))
            bin_lo, bin_hi = _binom_ci(k_in, n_in) if n_in > 0 else (np.nan, np.nan)
            inside = bool(np.isfinite(gm) and lo <= gm <= hi)
            all_in = bool(n_in > 0 and k_in == n_in)
            gen_mean_sd = f"{gm:.2f} ± {gs:.2f}" if np.isfinite(gm) else "n/p"
            rows.append(dict(episode=ep, episode_label=EPISODE_LABEL.get(ep, ep), n_real_crisis=int(sel.sum()),
                             real=real, lo=lo, hi=hi, generator=g, name=pretty(g),
                             family_tag=info.get(g, {}).get("family_tag", ""), gen=gm, gen_sd=gs,
                             gen_mean_sd=gen_mean_sd,
                             gen_ci_lo=ci_lo, gen_ci_hi=ci_hi,
                             seeds=n_in, seed_pass_count=k_in, seed_pass_prop=float(k_in / n_in) if n_in > 0 else np.nan,
                             seeds_inside=k_in, n_seeds=n_in,
                             pass_rate=float(k_in / n_in) if n_in > 0 else np.nan,
                             binom_lo=bin_lo, binom_hi=bin_hi,
                             all_seeds_inside=all_in, inside=inside))
            if g == "cgnsde":
                verdict[ep] = inside
            if g == "cgnsde2":
                verdict["v2:" + ep] = inside
    return pd.DataFrame(rows), verdict


def heldout_gate_details(t3: pd.DataFrame, g2: dict, model: str, header_text: str,
                         df: pd.DataFrame | None = None) -> tuple[dict, str]:
    """Build multi-seed gate body and Markdown summary for GATE_G2 / GATE_G2_v2_posthoc."""
    from .analysis import pretty, _binom_ci
    ep_details = {}
    md_lines = [
        header_text,
        "",
        "### Held-Out Episode Multi-Seed Evaluation",
        "",
        "| Episode | Real Crisis Windows | Real Boost [95% Block-Boot CI] | "
        f"{pretty(model)} (mean ± SD) | 95% Seed CI | Seeds Passing [95% Binom CI] | All Seeds Pass | Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    sub = t3[t3.generator == model] if not t3.empty else pd.DataFrame()
    all_ep_all_seeds = []
    # Per-seed overall G2 pass across episodes: for each seed (nse/pre2020 boost_pct), count
    # episodes whose real [lo, hi] contains the seed's boost_pct; seed passes G2 iff >= 2 inside.
    seed_values: dict[int, float] = {}
    per_seed_inside: dict[int, int] = {}
    if df is not None:
        model_df = df[(df.market == "nse") & (df.fit == "pre2020") & (df.rule == "main")
                      & (df.metric == "boost_pct") & (df.generator == model)]
        for _, r in model_df.iterrows():
            seed_values[int(r["seed"])] = float(r["value"])
        for s in seed_values:
            per_seed_inside[s] = 0
    for ep, passed in g2.items():
        r_ep = sub[sub.episode == ep]
        if r_ep.empty:
            ep_details[ep] = dict(inside=bool(passed))
            continue
        r = r_ep.iloc[0]
        gm = float(r["gen"]) if np.isfinite(r["gen"]) else np.nan
        gs = float(r["gen_sd"]) if np.isfinite(r["gen_sd"]) else 0.0
        clo = float(r.get("gen_ci_lo", np.nan))
        chi = float(r.get("gen_ci_hi", np.nan))
        k_in = int(r.get("seeds_inside", 0))
        n_in = int(r.get("n_seeds", 0))
        blo = float(r.get("binom_lo", np.nan))
        bhi = float(r.get("binom_hi", np.nan))
        all_in = bool(r.get("all_seeds_inside", False))
        all_ep_all_seeds.append(all_in)
        lo = float(r["lo"]); hi = float(r["hi"])
        if seed_values:
            for s, val in seed_values.items():
                if lo <= val <= hi:
                    per_seed_inside[s] = per_seed_inside.get(s, 0) + 1
        ep_details[ep] = dict(
            episode_label=str(r["episode_label"]),
            inside=bool(passed),
            n_real_crisis=int(r["n_real_crisis"]),
            real_boost_pct=float(r["real"]),
            real_ci_95=[float(r["lo"]), float(r["hi"])],
            gen_mean=gm,
            gen_sd=gs,
            gen_formatted=f"{gm:.2f} ± {gs:.2f}" if np.isfinite(gm) else "n/p",
            gen_ci_95=[clo, chi],
            seeds_passed=k_in,
            n_seeds=n_in,
            all_seeds_pass=all_in,
            pass_proportion=float(r.get("pass_rate", np.nan)),
            binomial_ci_95=[blo, bhi],
        )
        md_lines.append(
            f"| {r['episode_label']} (`{ep}`) | {int(r['n_real_crisis'])} | "
            f"{r['real']:.1f}% [{r['lo']:.1f}, {r['hi']:.1f}] | "
            f"{gm:.2f}% ± {gs:.2f}% | [{clo:.2f}, {chi:.2f}] | "
            f"{k_in}/{n_in} [{blo:.2f}, {bhi:.2f}] | `{all_in}` | **{'PASS' if passed else 'FAIL'}** |"
        )
    excluded = {
        ep: dict(window_range=list(rng_ep), min_required_crisis_windows=5,
                 reason="Excluded because real panel contains fewer than 5 crisis-labelled windows in episode.")
        for ep, rng_ep in P.HELDOUT_EPISODES.items() if ep not in g2
    }
    if excluded:
        md_lines += [
            "",
            "**Excluded episodes (< 5 real crisis windows):** " +
            ", ".join(f"`{k}` ({v['window_range'][0]}..{v['window_range'][1]})" for k, v in excluded.items()),
        ]
    # Per-seed overall G2 pass summary across episodes.
    _seeds_passed = 0
    _seeds_total = len(seed_values)
    pass_proportion = np.nan
    binomial_ci_95 = [np.nan, np.nan]
    all_seeds_pass = False
    per_seed_pass: dict[int, bool] = {}
    multiseed_all_seeds_pass = False
    if seed_values:
        per_seed_pass = {s: (per_seed_inside.get(s, 0) >= 2) for s in seed_values}
        _seeds_passed = sum(per_seed_pass.values())
        pass_proportion = float(_seeds_passed / _seeds_total) if _seeds_total > 0 else np.nan
        b_lo, b_hi = _binom_ci(_seeds_passed, _seeds_total) if _seeds_total > 0 else (np.nan, np.nan)
        binomial_ci_95 = [b_lo, b_hi]
        all_seeds_pass = bool(_seeds_total > 0 and _seeds_passed == _seeds_total)
        multiseed_all_seeds_pass = all_seeds_pass
        md_lines += [
            "",
            f"**Multi-seed G2 consistency:** {_seeds_passed}/{_seeds_total} seeds pass G2 across episodes "
            f"(proportion = {pass_proportion:.2f}, 95% Clopper-Pearson binomial CI "
            f"[{binomial_ci_95[0]:.3f}, {binomial_ci_95[1]:.3f}], all seeds pass = `{all_seeds_pass}`).",
        ]
    body = {
        **g2,
        "model": model,
        "episodes_passed": int(sum(g2.values())),
        "episodes_evaluated": int(len(g2)),
        "multiseed_all_episodes_all_seeds_pass": bool(len(all_ep_all_seeds) >= 2 and sum(all_ep_all_seeds) >= 2),
        "multiseed_all_seeds_pass": multiseed_all_seeds_pass,
        "_seeds_passed": _seeds_passed,
        "_seeds_total": _seeds_total,
        "pass_proportion": pass_proportion,
        "binomial_ci_95": binomial_ci_95,
        "all_seeds_pass": all_seeds_pass,
        "per_seed_pass": per_seed_pass,
        "episodes": ep_details,
        "excluded_episodes": excluded,
    }
    return body, "\n".join(md_lines)


def enrich_t1_seed_distributions(t: pd.DataFrame, df: pd.DataFrame, market: str,
                                 fit: str, rule: str = "main") -> pd.DataFrame:
    """Add mean±SD strings and seed min/max columns to a main_table frame.

    Adds <m>_mean_sd, <m>_seed_min, <m>_seed_max for every metric and
    score_<f>_mean_sd for every family score, without removing or renaming
    any existing column.
    """
    from .analysis import ALL_METRICS, FAMILIES, _fmt
    for m in list(ALL_METRICS):
        if m in t.columns:
            t[m + "_mean_sd"] = [_fmt(r.get(m), r.get(m + "_sd")) for _, r in t.iterrows()]
            if m + "_min" in t.columns:
                t[m + "_seed_min"] = t[m + "_min"]
            if m + "_max" in t.columns:
                t[m + "_seed_max"] = t[m + "_max"]
    for f in FAMILIES:
        k = f"score_{f}"
        if k in t.columns:
            t[k + "_mean_sd"] = [_fmt(r.get(k), r.get(k + "_sd")) for _, r in t.iterrows()]
    return t


def t1alt_markdown(t: pd.DataFrame, title: str = "T1alt — robustness rule (mean ± SD)") -> str:
    """Clean markdown for a T1alt_* frame showing mean ± SD strings."""
    if t.empty:
        return f"### {title}\n\nn/p\n"
    metric_cols = [c for c in t.columns if c.endswith("_mean_sd")]
    head = ["Model"] + [c[: -len("_mean_sd")] for c in metric_cols] + ["seeds", "impl_source"]
    lines = [f"### {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in t.iterrows():
        cells = [r.get("name", r.get("generator", ""))]
        cells += [r.get(c, "n/p") for c in metric_cols]
        cells += [str(r.get("seeds", "")), r.get("impl_source", "")]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines)


def family_tau_tables(df: pd.DataFrame, summary: dict) -> tuple[pd.DataFrame, str]:
    """Build T6_family_taus.csv/.md with bootstrap intervals for all family pairs."""
    from .analysis import RESULTS
    rows = []
    for key, s in summary.items():
        if not isinstance(s, dict) or "/" not in str(key):
            continue
        mk, fit = str(key).split("/", 1)
        taus = s.get("family_tau_intervals") or s.get("family_taus_ci") or s.get("family_taus") or {}
        gens = s.get("generators", []) or []
        n_gen = len(gens)
        n_seeds = len(P.SEEDS)
        for pair, v in taus.items():
            if isinstance(v, dict) and v:
                fa, fb = str(pair).split("_vs_")
                rows.append(dict(
                    market=mk, fit=fit, pair=pair, fa=fa, fb=fb,
                    tau=v.get("tau"), ci_lo=v.get("lo"), ci_hi=v.get("hi"),
                    seed_date_lo=v.get("seed_date_lo"), seed_date_hi=v.get("seed_date_hi"),
                    loo_min=v.get("loo_min"), loo_max=v.get("loo_max"),
                    generators=n_gen, seeds=n_seeds,
                ))
            elif isinstance(v, (int, float)):
                fa, fb = str(pair).split("_vs_")
                rows.append(dict(
                    market=mk, fit=fit, pair=pair, fa=fa, fb=fb,
                    tau=float(v), ci_lo=np.nan, ci_hi=np.nan,
                    seed_date_lo=np.nan, seed_date_hi=np.nan,
                    loo_min=np.nan, loo_max=np.nan,
                    generators=n_gen, seeds=n_seeds,
                ))
    t = pd.DataFrame(rows)
    head = ["market", "fit", "pair", "tau", "ci_lo", "ci_hi", "seed_date_lo", "seed_date_hi",
            "loo_min", "loo_max", "generators", "seeds"]
    lines = ["# T6: Family Tau Tables (bootstrap intervals)", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    if not t.empty:
        for _, r in t.iterrows():
            cells = []
            for c in head:
                v = r.get(c)
                if v is None or (isinstance(v, (float, np.floating)) and pd.isna(v)):
                    cells.append("")
                elif isinstance(v, (int, np.integer)):
                    cells.append(str(int(v)))
                elif isinstance(v, (float, np.floating)):
                    cells.append(f"{float(v):.4g}")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    return t, md


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
