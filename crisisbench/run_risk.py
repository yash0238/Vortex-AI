"""Downstream VaR / Expected-Shortfall backtest and metric-validity analysis (T4, T5, gate G3).

    python -m crisisbench.run_risk

For every generator, each out-of-sample day t (2020-01-01 onward) gets a VaR/ES forecast
from that generator's pool of synthetic (state, next-day return) pairs:
  state_t = realised volatility of the REAL portfolio over days t-20 .. t-1,
  forecast = empirical VaR/ES of the next-day returns of the 5% of pool entries whose
             synthetic 20-day volatility is closest to state_t.
The pool used in calendar year Y comes from the generator refitted on data up to
31 Dec of year Y-1 (protocol.RISK["refit_dates"]), so every forecast is out of sample.
Benchmarks: rolling historical simulation and GARCH(1,1)-t filtered historical simulation.

Ranking for metric validity (frozen): mean FZ0 loss at the 97.5% level, equal-weight
portfolio.  T5 = Kendall tau between that ranking and the ranking by each benchmark
family score / metric error from the 'pre2020' fit, with a joint bootstrap over
backtest days (blocks of 20) and seeds.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import protocol as P
from . import risk as R
from .analysis import MAIN, PRETTY, RESULTS, TABLES, FAMILIES, ALL_METRICS, kendall, load_meta, load_runs, write_gate
from .bench import ARTIFACTS
from .data import load_panel

OUT = RESULTS / "06_risk"
RANK_LEVEL, RANK_PF = 0.975, "equal_weight"
COLORS = {"hist_sim": "var(--s-static)", "garch_fhs_bench": "var(--s-dyn)", "cgnsde2": "var(--s-ours)",
          "regime_gaussian": "var(--s-regime)", "timegan": "var(--s-neural)", "iid_gaussian": "var(--s-resample)"}


def pool_for(gen: str, year: int):
    fit = "pre2020" if year == 2020 else f"upto{year - 1}"
    p = ARTIFACTS / "risk_pool" / "nse" / gen / fit / "seed0.npz"
    return (np.load(p), fit) if p.exists() else (None, fit)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel("nse")
    levels = np.array(P.RISK["levels"])
    oos = np.asarray(panel.dates >= P.RISK["oos_start"])
    idx = np.where(oos)[0]
    years = panel.dates[idx].year.to_numpy()
    w = {"equal_weight": np.full(panel.n_assets, 1 / panel.n_assets),
         "dirichlet_seed7": np.random.default_rng(7).dirichlet(np.ones(panel.n_assets))}
    info = load_meta()
    rows, daily, series = [], {}, None
    for pf, wt in w.items():
        r_all = panel.returns @ wt
        r = r_all[idx]
        state = np.array([r_all[t - 20:t].std(ddof=1) for t in idx])
        fc: dict[str, tuple] = {}
        v, e = R.historical_simulation(r_all, P.RISK["hs_window"], levels)
        fc["hist_sim"] = (v[idx], e[idx], None, "Historical simulation (500 days)", "benchmark", "")
        v, e = R.garch_fhs_var_es(r_all, levels)
        fc["garch_fhs_bench"] = (v[idx], e[idx], None, "GARCH-t FHS on the portfolio", "benchmark", "")
        for g in MAIN:
            V = np.full((len(idx), len(levels)), np.nan)
            E = V.copy()
            NB, used, ok = [None] * len(idx), [], True
            for y in np.unique(years):
                pool, fit = pool_for(g, int(y))
                if pool is None:
                    ok = False
                    break
                if not np.allclose(pool[f"{pf}__w"], wt):
                    raise RuntimeError(f"portfolio weights of pool {g}/{fit} do not match")
                m = years == y
                vv, ee, nb = R.knn_var_es(pool[f"{pf}__state"], pool[f"{pf}__next"], state[m], levels,
                                          P.RISK["knn_frac"], return_neighbours=True)
                V[m], E[m] = vv, ee
                for i, j in enumerate(np.where(m)[0]):
                    NB[j] = nb[i]
                used.append(fit)
            if ok:
                fc[g] = (V, E, np.array(NB), PRETTY[g], "generator", info.get(g, {}).get("family_tag", ""))
        for name, (V, E, NB, label, kind, fam) in fc.items():
            for k, lv in enumerate(levels):
                a = 1 - lv
                s = R.summarise(r, V[:, k], E[:, k], a, neighbours=NB, n_sim=1000, rng=np.random.default_rng(0))
                rows.append(dict(method=name, name=label, kind=kind, family_tag=fam, portfolio=pf, level=lv,
                                 n=s["n"], violations=s["violations"], expected=s["expected"], rate=s["rate"],
                                 kupiec_p=s["kupiec_p"], ind_p=s["christoffersen_ind_p"], cc_p=s["cc_p"],
                                 traffic=s["traffic_light"], z2=s["z2"], z2_p=s["z2_p"], fz0=s["fz0"],
                                 qloss=s["quantile_loss"], es_nonpositive_days=s.get("es_nonpositive_days", 0)))
                if s.get("es_nonpositive_days"):
                    print(f"{name}/{pf}/{lv}: ES forecast <= 0 on {s['es_nonpositive_days']} days; "
                          "Z2 and FZ0 undefined, method left out of the FZ0 ranking")
                if pf == RANK_PF and abs(lv - RANK_LEVEL) < 1e-9 and kind == "generator" and np.isfinite(s["fz0"]):
                    daily[name] = R.fz0_loss(r, V[:, k], E[:, k], a)
        if pf == "equal_weight":
            k99 = int(np.argmax(levels))
            show = [m for m in ("hist_sim", "garch_fhs_bench", "cgnsde2", "regime_gaussian", "timegan", "iid_gaussian") if m in fc]
            series = dict(dates=[str(d.date()) for d in panel.dates[idx]], ret=np.round(100 * r, 3).tolist(),
                          methods={m: dict(name=fc[m][3], color=COLORS.get(m), var99=np.round(-100 * fc[m][0][:, k99], 3).tolist())
                                   for m in show})
    t4 = pd.DataFrame(rows)
    t4["rank_fz0"] = t4.groupby(["portfolio", "level"])["fz0"].rank(method="min").astype("Int64")   # NaN: FZ0 undefined
    t4.to_csv(TABLES / "T4_risk.csv", index=False)
    (TABLES / "T4_risk.md").write_text(t4.to_markdown(index=False, floatfmt=".4g"), encoding="utf-8")
    (OUT / "series.json").write_text(json.dumps(series), encoding="utf-8")

    # ------------------------------------------------ T5 metric validity
    # H3 / G3 were preregistered on the registered generators: the post hoc CG-NSDE v2 is left out of the
    # verdict and a second, labelled evaluation includes it.
    allg = [g for g in MAIN if g in daily]
    from .analysis import POSTHOC
    validity(daily, [g for g in allg if g not in POSTHOC], "", "GATE_G3",
             "Kendall tau between benchmark family rank (pre2020 fit) and backtest rank; positive = lower benchmark "
             "error goes with lower FZ0 loss.")
    if any(g in allg for g in POSTHOC):
        validity(daily, allg, "_with_v2", "GATE_G3_v2_posthoc",
                 "POST HOC: same test with the rows added after the protocol freeze (CG-NSDE v2, full-scale GPU "
                 "baselines) included among the generators.")


def validity(daily: dict, gens: list, suffix: str, gate: str, note: str) -> None:
    if len(gens) < 4:
        print(f"only {len(gens)} generators have complete risk pools, T5{suffix} skipped")
        return
    L = np.column_stack([daily[g] for g in gens])                      # (days, gens)
    df = load_runs("nse")
    d = df[(df.fit == "pre2020") & (df.rule == "main") & df.generator.isin(gens)]
    keys = [f"family_{f}" for f in FAMILIES] + ALL_METRICS
    seedval = {k: {g: x.sort_values("seed")["err"].to_numpy() for g, x in d[d.metric == k].groupby("generator")} for k in keys}
    rng = np.random.default_rng(0)
    nb = int(np.ceil(len(L) / 20))
    out = []
    for k in keys:
        if any(g not in seedval[k] for g in gens):
            continue
        vals = [seedval[k][g] for g in gens]
        if any(not np.isfinite(v).any() for v in vals):            # metric not produced by some generator
            continue
        point = kendall([np.nanmean(v) for v in vals], L.mean(0))
        reps = []
        for _ in range(2000):
            st = rng.integers(0, len(L), nb)
            ii = ((st[:, None] + np.arange(20)[None, :]) % len(L)).ravel()[:len(L)]
            ms = [np.nanmean(v[rng.integers(0, len(v), len(v))]) for v in vals]
            reps.append(kendall(ms, L[ii].mean(0)))
        reps = np.array(reps, dtype=float)
        reps = reps[np.isfinite(reps)]
        if not np.isfinite(point) or len(reps) == 0:
            continue
        fam = k.startswith("family_")
        out.append(dict(key=k, label=("Family " + k[-1] + " score") if fam else k, kind="family" if fam else "metric",
                        tau=point, lo=float(np.quantile(reps, 0.025)), hi=float(np.quantile(reps, 0.975)),
                        n_generators=len(gens)))
    t5 = pd.DataFrame(out)
    t5.to_csv(TABLES / f"T5_metric_validity{suffix}.csv", index=False)
    (TABLES / f"T5_metric_validity{suffix}.md").write_text(t5.to_markdown(index=False, floatfmt=".3g"), encoding="utf-8")
    if not {"family_B", "family_C"} <= set(t5.key):
        print(f"family B or C missing from T5{suffix}; G3 not evaluated")
        return
    c, b = t5[t5.key == "family_C"].iloc[0], t5[t5.key == "family_B"].iloc[0]
    h3 = bool(c.tau > 0.5 and c.tau > b.tau and c.lo > b.hi)
    body = dict(tau_C=float(c.tau), ci_C=[float(c.lo), float(c.hi)], tau_B=float(b.tau), ci_B=[float(b.lo), float(b.hi)],
                rank_basis=f"mean FZ0, level {RANK_LEVEL}, {RANK_PF}", generators=gens, H3_holds=h3,
                fz0_rank={g: int(x) for g, x in zip(gens, pd.Series(L.mean(0)).rank(method="min"))})
    write_gate(gate, "PASS" if h3 else "FAIL", body, note)
    (OUT / f"G3{suffix}.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    print(f"--- T5{suffix} ({len(gens)} generators)")
    print(t5[t5.kind == "family"].to_string(index=False))
    print(f"G3{suffix}:", "PASS" if h3 else "FAIL")


if __name__ == "__main__":
    main()
