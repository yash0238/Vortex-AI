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

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
GATES = RESULTS / "gates"
FAMILIES = list(P.METRIC_FAMILIES)
ALL_METRICS = [m for f in FAMILIES for m in P.METRIC_FAMILIES[f]]
PRETTY = {
    "iid_gaussian": "i.i.d. Gaussian", "t_ppca": "Static t-PPCA", "stationary_bootstrap": "Stationary bootstrap",
    "garch_fhs": "GARCH-t FHS", "dcc_garch": "DCC-GARCH-t", "regime_gaussian": "Regime Gaussian",
    "hmm_gaussian": "Gaussian HMM", "timevae": "TimeVAE", "quantgan": "QuantGAN", "timegan": "TimeGAN",
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


# ------------------------------------------------------------------ loading
def load_runs(market: str | None = None) -> pd.DataFrame:
    """Long frame: one row per (market, generator, fit, seed, rule, metric)."""
    rows = []
    base = RESULTS / "runs"
    for p in base.glob("*/*/*/seed*/metrics.json"):
        mk, gen, fit = p.parts[-5], p.parts[-4], p.parts[-3]
        if gen.startswith("_") or (market and mk != market):
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
    """Per generator: metric mean and sd over seeds, family scores, family ranks."""
    d = df[(df.market == market) & (df.fit == fit) & (df.rule == rule) & df.generator.isin(gens)]
    if d.empty:
        return pd.DataFrame()
    g = d.groupby(["generator", "metric"])
    mean = g["value"].mean().unstack()
    sd = g["value"].std(ddof=1).unstack()
    err = g["err"].mean().unstack()
    inside = d[d.inside.notna()].assign(inside=lambda x: x.inside.astype(float)).groupby(
        ["generator", "metric"])["inside"].mean().unstack()
    nseeds = d.groupby("generator")["seed"].nunique()
    meta = df[(df.market == market) & (df.fit == fit) & (df.rule == "meta")]
    tmin = meta[meta.metric == "train_minutes"].groupby("generator")["value"].mean()
    info = load_meta()
    rows = []
    for gen in [x for x in gens if x in mean.index]:
        r = dict(generator=gen, name=pretty(gen), family_tag=info.get(gen, {}).get("family_tag", ""),
                 impl_source=info.get(gen, {}).get("impl_source", ""), seeds=int(nseeds[gen]),
                 train_minutes=float(tmin.get(gen, np.nan)))
        for m in ALL_METRICS:
            if m in mean.columns:
                r[m] = mean.loc[gen, m]
                r[m + "_sd"] = sd.loc[gen, m] if m in sd.columns else np.nan
                r[m + "_err"] = err.loc[gen, m]
                r[m + "_inside"] = inside.loc[gen, m] if m in inside.columns else np.nan
        for f in FAMILIES:
            k = f"family_{f}"
            if k in mean.columns:
                r[f"score_{f}"] = mean.loc[gen, k]
                r[f"score_{f}_sd"] = sd.loc[gen, k] if k in sd.columns else np.nan
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


# ------------------------------------------------------- rank agreement (H1)
def seed_scores(df: pd.DataFrame, market: str, fit: str, gens: list[str], rule: str = "main") -> pd.DataFrame:
    d = df[(df.market == market) & (df.fit == fit) & (df.rule == rule) & df.generator.isin(gens)
           & df.metric.str.startswith("family_")]
    return d.pivot_table(index=["generator", "seed"], columns="metric", values="value")


def kendall(a, b) -> float:
    t = stats.kendalltau(a, b)
    return float(t.statistic) if np.isfinite(t.statistic) else np.nan


def family_tau(df, market, fit, gens, fa="B", fb="C", n_boot=2000, seed=0, rule="main") -> dict:
    """Kendall tau between two family rankings, with a seed-resampling bootstrap interval."""
    ss = seed_scores(df, market, fit, gens, rule)
    if ss.empty or f"family_{fa}" not in ss or f"family_{fb}" not in ss:
        return {}
    mean = ss.groupby("generator").mean()
    tau = kendall(mean[f"family_{fa}"], mean[f"family_{fb}"])
    rng = np.random.default_rng(seed)
    groups = {g: x.to_numpy() for g, x in ss[[f"family_{fa}", f"family_{fb}"]].groupby("generator")}
    reps = []
    for _ in range(n_boot):
        m = np.array([x[rng.integers(0, len(x), len(x))].mean(0) for x in groups.values()])
        reps.append(kendall(m[:, 0], m[:, 1]))
    reps = np.array(reps)
    reps = reps[np.isfinite(reps)]
    if not np.isfinite(tau) or len(reps) == 0:      # a family score is missing (e.g. no crisis windows at all)
        return {}
    # leave-one-generator-out: is the conclusion carried by a single model?
    loo = {g: kendall(mean.drop(g)[f"family_{fa}"], mean.drop(g)[f"family_{fb}"]) for g in mean.index}
    return dict(tau=tau, lo=float(np.quantile(reps, 0.025)), hi=float(np.quantile(reps, 0.975)),
                n_generators=int(len(mean)), loo_min=float(min(loo.values())), loo_max=float(max(loo.values())),
                loo=loo)


def metric_tau_matrix(t: pd.DataFrame) -> pd.DataFrame:
    """Kendall tau between the generator rankings induced by each pair of metrics (normalised errors)."""
    ms = [m for m in ALL_METRICS if m + "_err" in t]
    E = t.set_index("generator")[[m + "_err" for m in ms]]
    out = pd.DataFrame(np.eye(len(ms)), index=ms, columns=ms)
    for a, b in combinations(ms, 2):
        out.loc[a, b] = out.loc[b, a] = kendall(E[a + "_err"], E[b + "_err"])
    return out


# ------------------------------------------------------------------- gates
def gate_g1(t: pd.DataFrame, tau: dict, model: str = "cgnsde") -> tuple[str, dict]:
    """Preregistered for model = 'cgnsde' (v1).  Called again with 'cgnsde2' as a post hoc evaluation."""
    s = t.set_index("generator")
    static = [g for g in P.STATIC if g in s.index]
    regime = [g for g in ("regime_gaussian", "hmm_gaussian", model) if g in s.index]
    a = bool(static and regime and all(s.loc[r, "score_C"] < s.loc[st, "score_C"] for r in regime for st in static))
    h2 = None
    detail = {}
    if model in s.index:
        cg, sd = s.loc[model, "score_C"], s.loc[model, "score_C_sd"]
        sd = 0.0 if not np.isfinite(sd) else sd
        for r in ("regime_gaussian", "hmm_gaussian"):
            if r in s.index:
                detail[r] = dict(score_C=float(s.loc[r, "score_C"]), cgnsde=float(cg), cgnsde_sd=float(sd),
                                 matches=bool(s.loc[r, "score_C"] <= cg + sd))
        h2 = any(v["matches"] for v in detail.values()) if detail else None
    verdict = "PASS" if (a and h2 is False) else "FAIL"
    h1 = None if not tau else bool(tau["tau"] < 0.3)
    return verdict, dict(regime_beats_static=a, H2_simple_regime_model_matches_cgnsde=h2, H2_detail=detail,
                         H1_tau_B_vs_C=tau.get("tau"), H1_ci=[tau.get("lo"), tau.get("hi")], H1_holds=h1)


def write_gate(name: str, verdict: str, body: dict, text: str) -> None:
    GATES.mkdir(parents=True, exist_ok=True)
    lines = [f"# {name}: {verdict}", "", f"Rule (frozen in crisisbench/protocol.py): {P.GATES.get(name.split('_')[-1], '')}",
             "", text, "", "```json", json.dumps(body, indent=2, default=float), "```", "",
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
            title = f"Main comparison, market {mk}, fit '{fit}' ({ref['n_windows']} real windows, {ref['n_crisis']} crisis)"
            save(t, f"T1_{mk}_{fit}", table_md(t, ref, title))
            # H1 was preregistered on the registered generators: the post hoc v2 is left out of the headline tau
            # and the value with v2 is reported next to it.
            tau_bc = family_tau(df, mk, fit, [g for g in gens if g != "cgnsde2"], "B", "C")
            tau_bc_v2 = family_tau(df, mk, fit, gens, "B", "C") if "cgnsde2" in gens else {}
            taus ={f"{a}_vs_{b}": family_tau(df, mk, fit, gens, a, b, n_boot=500)
                    for a, b in combinations(FAMILIES, 2)}
            metric_tau_matrix(t).to_csv(TABLES / f"F2_metric_tau_{mk}_{fit}.csv")
            alt = main_table(df, mk, fit, gens, rule="alt_rule")
            if not alt.empty:
                save(alt, f"T1alt_{mk}_{fit}")
            summary[f"{mk}/{fit}"] = dict(
                generators=gens, tau_B_vs_C=tau_bc, tau_B_vs_C_with_v2=tau_bc_v2,family_taus={k: v.get("tau") for k, v in taus.items()},
                tau_B_vs_C_alt_rule=family_tau(df, mk, fit, gens, "B", "C", n_boot=500, rule="alt_rule").get("tau"),
                ranks={f: t.set_index("generator")[f"rank_{f}"].to_dict() for f in FAMILIES if f"rank_{f}" in t},
                scores={f: t.set_index("generator")[f"score_{f}"].to_dict() for f in FAMILIES if f"score_{f}" in t},
                real={k: v for k, v in ref["metrics"].items()})
            if mk == "nse":
                verdict, body = gate_g1(t, tau_bc)
                summary[f"G1/{fit}"] = dict(verdict=verdict, **body)
                write_gate(f"GATE_G1_{fit}", verdict, body,
                           f"Evaluated on market nse, fit '{fit}', {len(gens)} generators.")
                if "cgnsde2" in set(t.generator):
                    verdict2, body2 = gate_g1(t, tau_bc, "cgnsde2")
                    summary[f"G1v2/{fit}"] = dict(verdict=verdict2, **body2)
                    write_gate(f"GATE_G1_{fit}_v2_posthoc", verdict2, body2,
                               f"POST HOC: same gate evaluated for CG-NSDE v2. Market nse, fit '{fit}'.")
            abl = [g for g in sorted(set(df[(df.market == mk) & (df.fit == fit)].generator))
                   if g == "cgnsde2" or g.startswith("cgnsde2@")]
            if len(abl) > 1:
                ta = main_table(df, mk, fit, abl)
                save(ta, f"T2_ablation_{mk}_{fit}", table_md(ta, ref, f"CG-NSDE ablations, market {mk}, fit '{fit}'"))
    from .analysis_extra import contamination_table, heldout_table, hypotheses
    g2 = None
    if "nse/pre2020" in summary:
        t3, g2 = heldout_table(df)
        if not t3.empty:
            save(t3, "T3_heldout")
        g2v2 = {k[3:]: v for k, v in (g2 or {}).items() if k.startswith("v2:")}
        g2 = {k: v for k, v in (g2 or {}).items() if not k.startswith("v2:")}
        if g2:
            write_gate("GATE_G2", "PASS" if sum(g2.values()) >= 2 else "FAIL", g2,
                       "CG-NSDE v1 (fit on data to 2019) boost_pct against each held-out episode's real 95% interval.")
        if g2v2:
            write_gate("GATE_G2_v2_posthoc", "PASS" if sum(g2v2.values()) >= 2 else "FAIL", g2v2,
                       "POST HOC: CG-NSDE v2 (fit on data to 2019) boost_pct against each held-out episode's interval.")
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
