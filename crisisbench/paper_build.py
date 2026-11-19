"""Build every figure, table and number of the paper from results/ (nothing is typed by hand).

    python -m crisisbench.paper_build

Writes  paper/figures/*.pdf, paper/tables/*.tex, paper/numbers.tex (LaTeX macros), and
        paper/vortex_paper_overleaf.zip (main.tex, references.bib, figures, tables, numbers).
Sections whose inputs are missing are skipped and listed at the end.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import MAIN, PRETTY, RESULTS, TABLES, reference
from .data import ROOT

PAPER = ROOT / "paper"
FIG, TAB = PAPER / "figures", PAPER / "tables"
# categorical palette (validated order), light surface
C = {"neural-ours": "#2a78d6", "neural": "#6250d6", "regime": "#1baf7a", "dynamic-classical": "#eda100",
     "resampling": "#e87ba4", "static": "#898781"}
FAM_LABEL = {"neural-ours": "CG-NSDE (ours)", "neural": "Neural baselines", "regime": "Regime models",
             "dynamic-classical": "GARCH family", "resampling": "Resampling", "static": "Static"}
SHORT = {"iid_gaussian": "iid-G", "t_ppca": "t-PPCA", "stationary_bootstrap": "S-Boot", "garch_fhs": "GARCH-FHS",
         "dcc_garch": "DCC", "regime_gaussian": "Reg-G", "hmm_gaussian": "HMM", "timevae": "TimeVAE",
         "quantgan": "QuantGAN", "timegan": "TimeGAN", "cgnsde": "CG-NSDE v1", "cgnsde2": "CG-NSDE v2", "cgnsde3": "CG-NSDE v3",
         "timegan_fs": "TimeGAN-GPU", "quantgan_fs": "QuantGAN-GPU", "diffusion_fs": "DDPM-GPU",
         "gjr_evt_tcopula": "GJR-t-cop", "adcc_gjr": "ADCC", "rs_tcopula": "RS-t-cop"}
INK, MUTED, GRID, BAND = "#0b0b0b", "#898781", "#e1e0d9", "#cde2fb"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "pdf.fonttype": 42,
                     "legend.frameon": False, "figure.dpi": 150})
skipped: list[str] = []
macros: dict[str, str] = {}


def t1(market: str, fit: str) -> pd.DataFrame | None:
    p = TABLES / f"T1_{market}_{fit}.csv"
    return pd.read_csv(p) if p.exists() else None


def num(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/p"
    return f"{x:.{nd}f}"


def tex_escape(s: str) -> str:
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("±", r"$\pm$")


def write_table(name: str, header: list[str], rows: list[list[str]], align: str | None = None) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    align = align or ("l" + "r" * (len(header) - 1))
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" if r != ["\\midrule"] else r"\midrule" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


_DIGITS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
           "8": "Eight", "9": "Nine"}


def macro_name(name: str) -> str:
    """LaTeX command names may hold letters only: digits become words, everything else is dropped
    (cgnsde2 -> CgnsdeTwo, 'H2 (v2, post hoc)' -> HTwovTwoposthoc)."""
    return "".join(_DIGITS.get(c, c) for c in name if c.isalpha() or c.isdigit())


def macro(name: str, value) -> None:
    macros[macro_name(name)] = value if isinstance(value, str) else num(value)


def legend_handles(tags):
    return [plt.Line2D([], [], marker="o", ls="", color=C[t], label=FAM_LABEL[t], markersize=5) for t in C if t in tags]


# ------------------------------------------------------------------ figures
def fig_scatter(market="nse", fit="full") -> None:
    t = t1(market, fit)
    if t is None:
        return skipped.append("fig_scatter")
    ref = reference(market, fit)["metrics"]
    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    ax.axhspan(ref["boost_pct"]["lo"], ref["boost_pct"]["hi"], color=BAND, lw=0, zorder=0)
    ax.axvspan(0, ref["corr_fro"]["ceiling"], color=BAND, lw=0, zorder=0, alpha=0.6)
    ax.axhline(ref["boost_pct"]["real"], color="#1c5cab", lw=0.8, zorder=1)
    for _, r in t.iterrows():
        if not np.isfinite(r.get("boost_pct", np.nan)):
            continue
        ax.scatter(r.corr_fro, r.boost_pct, s=26 if r.generator != "cgnsde2" else 46, color=C.get(r.family_tag, MUTED),
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(SHORT.get(r.generator, r.generator), (r.corr_fro, r.boost_pct), xytext=(4, 3),
                    textcoords="offset points", fontsize=6.5, color=INK)
    ax.set_xlabel("Correlation error $\\|C_{gen}-C_{real}\\|_F$ (lower is better)")
    ax.set_ylabel("Crisis correlation boost (%)")
    ax.legend(handles=legend_handles(set(t.family_tag)), fontsize=6, loc="best", handletextpad=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / f"f1_scatter_{market}_{fit}.pdf")
    plt.close(fig)


def fig_scores(market="nse", fit="full") -> None:
    t = t1(market, fit)
    if t is None:
        return skipped.append("fig_scores")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5), sharey=False)
    for ax, f, title in zip(axes, ("B", "C"), ("Correlation fidelity (family B)", "Crisis dependence (family C)")):
        d = t.sort_values(f"score_{f}", ascending=False)
        y = np.arange(len(d))
        ax.axvspan(0, 1, color=BAND, lw=0, zorder=0)
        ax.errorbar(d[f"score_{f}"], y, xerr=d[f"score_{f}_sd"].fillna(0), fmt="none", ecolor=MUTED, elinewidth=1, zorder=2)
        ax.scatter(d[f"score_{f}"], y, color=[C.get(k, MUTED) for k in d.family_tag], s=24, zorder=3, edgecolor="white", linewidth=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT.get(g, g) for g in d.generator], color=INK)
        ax.set_xlabel("score: mean error in units of real sampling noise")
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xlim(left=0)
        ax.grid(axis="y", visible=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / f"f2_scores_{market}_{fit}.pdf")
    plt.close(fig)


def fig_heldout() -> None:
    p = TABLES / "T3_heldout.csv"
    if not p.exists():
        return skipped.append("fig_heldout")
    t = pd.read_csv(p)
    eps = list(dict.fromkeys(t.episode))
    fig, axes = plt.subplots(1, len(eps), figsize=(3.5 * len(eps), 2.6), squeeze=False)
    for ax, ep in zip(axes[0], eps):
        d = t[t.episode == ep].reset_index(drop=True)
        y = np.arange(len(d))
        ax.axvspan(d.lo[0], d.hi[0], color=BAND, lw=0, zorder=0)
        ax.axvline(d.real[0], color="#1c5cab", lw=0.8)
        ax.errorbar(d.gen, y, xerr=d.gen_sd.fillna(0), fmt="none", ecolor=MUTED, elinewidth=1)
        ax.scatter(d.gen, y, color=[C.get(k, MUTED) for k in d.family_tag], s=22, zorder=3, edgecolor="white", linewidth=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT.get(g, g) for g in d.generator], color=INK)
        ax.set_title(f"{d.episode_label[0]} (n={d.n_real_crisis[0]})", fontsize=8, loc="left")
        ax.set_xlabel("Crisis correlation boost (%)")
        ax.grid(axis="y", visible=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "f3_heldout.pdf")
    plt.close(fig)


def fig_var() -> None:
    p = RESULTS / "06_risk" / "series.json"
    if not p.exists():
        return skipped.append("fig_var")
    s = json.loads(p.read_text())
    x = pd.to_datetime(s["dates"])
    r = np.array(s["ret"])
    fig, ax = plt.subplots(figsize=(7.1, 2.4))
    ax.bar(x, r, width=1.0, color="#c3c2b7", lw=0)
    cols = {"hist_sim": "#898781", "garch_fhs_bench": "#eda100", "cgnsde2": "#2a78d6", "regime_gaussian": "#1baf7a",
            "timegan": "#6250d6", "iid_gaussian": "#e87ba4"}
    for k, m in s["methods"].items():
        ax.plot(x, m["var99"], lw=1.0 if k != "cgnsde2" else 1.4, color=cols.get(k, INK), label=m["name"])
    ax.set_ylabel("daily return / 99% VaR (%)")
    ax.legend(ncol=3, fontsize=6.5, loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "f4_var.pdf")
    plt.close(fig)


def fig_dgp() -> None:
    p = RESULTS / "01_dgp" / "metric_validation.csv"
    if not p.exists():
        return skipped.append("fig_dgp")
    t = pd.read_csv(p)
    d = t[t.panel.str.startswith("sim_boost")]
    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    ax.plot([0, 80], [0, 80], color="#c3c2b7", lw=0.8, ls="--", label="identity")
    ax.errorbar(d.true_boost_pct, d.rule_boost_pct, yerr=[d.rule_boost_pct - d.rule_lo, d.rule_hi - d.rule_boost_pct],
                fmt="o", color="#2a78d6", ms=4, lw=1, label="benchmark estimate (95% CI)")
    ax.scatter(d.true_boost_pct, d.oracle_boost_pct, marker="x", color=INK, s=18, label="oracle labels", zorder=3)
    c = t[t.panel == "sim_ccc"]
    if len(c):
        ax.errorbar([0], c.rule_boost_pct, yerr=[c.rule_boost_pct - c.rule_lo, c.rule_hi - c.rule_boost_pct], fmt="s",
                    color="#eb6834", ms=4, lw=1, label="no-regime GARCH null")
    ax.set_xlabel("True population boost (%)")
    ax.set_ylabel("Measured window boost (%)")
    ax.legend(fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "f5_dgp.pdf")
    plt.close(fig)


def fig_ablation() -> None:
    p = TABLES / "T2_ablation_nse_full.csv"
    if not p.exists():
        return skipped.append("fig_ablation")
    t = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6))
    for ax, f in zip(axes, ("C", "A")):
        d = t.sort_values(f"score_{f}", ascending=False)
        y = np.arange(len(d))
        ax.axvspan(0, 1, color=BAND, lw=0, zorder=0)
        ax.errorbar(d[f"score_{f}"], y, xerr=d[f"score_{f}_sd"].fillna(0), fmt="none", ecolor=MUTED, elinewidth=1)
        ax.scatter(d[f"score_{f}"], y, s=22, zorder=3, edgecolor="white", linewidth=0.7,
                   color=["#2a78d6" if g == "cgnsde2" else MUTED for g in d.generator])
        ax.set_yticks(y)
        ax.set_yticklabels([g.replace("cgnsde2@", "").replace("cgnsde2", "full model").replace("_", " ") for g in d.generator], color=INK)
        ax.set_xlabel(f"family {f} score")
        ax.set_xlim(left=0)
        ax.grid(axis="y", visible=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "f6_ablation.pdf")
    plt.close(fig)


# ------------------------------------------------------------------- tables
def tab_main(market="nse", fit="full", name=None) -> None:
    t = t1(market, fit)
    if t is None:
        return skipped.append(f"tab_main_{market}_{fit}")
    ref = reference(market, fit)["metrics"]
    cols = [("corr_fro", "Corr. err.", 2), ("boost_pct", "Boost \\%", 1), ("cvar_ratio", "CVaR ratio", 2),
            ("tail_coexc", "Tail co-ex.", 3), ("kurt_med", "Kurt.", 2), ("acf1_sq", "ACF$(r^2)$", 3), ("disc_feat", "Disc.", 2)]
    head = ["Model"] + [c[1] for c in cols] + ["$S_A$", "$S_B$", "$S_C$", "$S_D$"]
    real = ["Real (95\\% band)"]
    for k, _, nd in cols:
        r = ref[k]
        real.append(f"$\\le${r['ceiling']:.{nd}f}" if r["kind"] == "dist" else f"{r['real']:.{nd}f} [{r['lo']:.{nd}f}, {r['hi']:.{nd}f}]")
    rows = [real + ["--"] * 4, ["\\midrule"]]
    for _, r in t.iterrows():
        cells = [tex_escape(PRETTY.get(r.generator, r.generator))]
        for k, _, nd in cols:
            v = r.get(k, np.nan)
            s = num(v, nd)
            cells.append(f"\\textbf{{{s}}}" if r.get(k + "_inside", 0) >= 0.5 else s)
        for f in "ABCD":
            best = r[f"rank_{f}"] == 1
            s = f"{r[f'score_{f}']:.2f}"
            cells.append(f"\\underline{{{s}}}" if best else s)
        rows.append(cells)
    write_table(name or f"t1_{market}_{fit}", head, rows)


def tab_risk() -> None:
    p = TABLES / "T4_risk.csv"
    if not p.exists():
        return skipped.append("tab_risk")
    t = pd.read_csv(p)
    d = t[(t.portfolio == "equal_weight") & (np.isclose(t.level, 0.975))].sort_values("fz0")
    rows = [[tex_escape(r["name"]), str(int(r.violations)), num(r.expected, 1), num(r.kupiec_p, 3), num(r.ind_p, 3),
             num(r.cc_p, 3), r.traffic, num(r.z2, 2), num(r.fz0, 3)] for _, r in d.iterrows()]
    write_table("t4_risk", ["Method", "Viol.", "Exp.", "$p_{UC}$", "$p_{IND}$", "$p_{CC}$", "Light", "$Z_2$", "FZ0"], rows)


def tab_validity() -> None:
    p = TABLES / "T5_metric_validity.csv"
    if not p.exists():
        return skipped.append("tab_validity")
    t = pd.read_csv(p)
    pv = TABLES / "T5_metric_validity_with_v2.csv"
    tv = pd.read_csv(pv) if pv.exists() else None
    rows = []
    for _, r in t.iterrows():
        row = [tex_escape(r.label), num(r.tau), f"[{r.lo:.2f}, {r.hi:.2f}]"]
        if tv is not None:
            x = tv[tv.key == r.key]
            row += [num(x.tau.iloc[0]), f"[{x.lo.iloc[0]:.2f}, {x.hi.iloc[0]:.2f}]"] if len(x) else ["--", "--"]
        rows.append(row)
    head = ["Ranking by", f"$\\tau$ ({int(t.n_generators.iloc[0])} reg.)", "95\\% CI"]
    if tv is not None:
        head += [f"$\\tau$ ({int(tv.n_generators.iloc[0])}, +post hoc)", "95\\% CI"]
    write_table("t5_validity", head, rows)
    macro("nRiskGen", str(int(t.n_generators.iloc[0])))
    for f in "ABCD":
        x = t[t.key == f"family_{f}"]
        if len(x):
            macro(f"tauRisk{f}", float(x.tau.iloc[0])); macro(f"tauRisk{f}Lo", float(x.lo.iloc[0])); macro(f"tauRisk{f}Hi", float(x.hi.iloc[0]))
        if tv is not None:
            x = tv[tv.key == f"family_{f}"]
            if len(x):
                macro(f"tauRiskP{f}", float(x.tau.iloc[0])); macro(f"tauRiskP{f}Lo", float(x.lo.iloc[0])); macro(f"tauRiskP{f}Hi", float(x.hi.iloc[0]))
    if tv is not None:
        macro("nRiskGenP", str(int(tv.n_generators.iloc[0])))


def tab_taus(summary: dict) -> None:
    rows = []
    names = {"nse": "India (NSE)", "us": "United States", "kr": "Korea (KOSPI)", "nse_ext": "NSE 2025--26"}
    for mk in ("nse", "us", "kr"):
        for fit in ("full", "pre2020"):
            t = (summary.get(f"{mk}/{fit}") or {}).get("tau_B_vs_C")
            if t:
                rows.append([names[mk], "in-sample" if fit == "full" else "held-out", str(t["n_generators"]), num(t["tau"]),
                             f"[{t['lo']:.2f}, {t['hi']:.2f}]", f"[{t['loo_min']:.2f}, {t['loo_max']:.2f}]"])
                tg = f"tauBC{mk}{'Full' if fit == 'full' else 'Held'}"
                macro(tg, t["tau"]); macro(tg + "Lo", t["lo"]); macro(tg + "Hi", t["hi"])
    if rows:
        write_table("t6_taus", ["Market", "Fit", "Models", "$\\tau(B,C)$", "95\\% CI (seeds)", "Leave-one-out range"], rows, "llrrrr")
    else:
        skipped.append("tab_taus")


def tab_ablation() -> None:
    p = TABLES / "T2_ablation_nse_full.csv"
    if not p.exists():
        return skipped.append("tab_ablation")
    t = pd.read_csv(p).set_index("generator")
    ph = TABLES / "T2_ablation_nse_pre2020.csv"
    th = pd.read_csv(ph).set_index("generator") if ph.exists() else None
    rows = []
    for g, r in t.iterrows():
        name = tex_escape(g.replace("cgnsde2@", "-- ").replace("cgnsde2", "Full model (v2)"))
        row = [name, num(r.boost_pct, 1), num(r.score_A), num(r.score_C)]
        if th is not None:
            h = th.loc[g] if g in th.index else None
            row += [num(h.boost_pct, 1), num(h.score_A), num(h.score_C), num(h.score_D)] if h is not None else ["--"] * 4
        rows.append(row)
    head = ["Variant", "Boost \\%", "$S_A$", "$S_C$"] + (["Boost \\%", "$S_A$", "$S_C$", "$S_D$"] if th is not None else [])
    write_table("t2_ablation", head, rows, "l" + "r" * (len(head) - 1))
    if th is not None:
        for g, h in th.iterrows():
            k = "".join(w.capitalize() for w in g.replace("cgnsde2@", "").replace("cgnsde2", "full").split("_")).replace("0", "Zero").replace("1", "One").replace("4", "Four").replace("6", "Six")
            macro(f"ablHeldC{k}", h.score_C); macro(f"ablHeldA{k}", h.score_A); macro(f"ablHeldD{k}", h.score_D)
            macro(f"ablHeldCsd{k}", h.score_C_sd)


def contamination() -> None:
    p = TABLES / "T7_contamination.csv"
    if not p.exists():
        return skipped.append("contamination")
    t = pd.read_csv(p)
    for panel, tag in zip(t.panel.unique(), ("Tick", "Spikes", "Split")):
        x = t[t.panel == panel]
        macro(f"contam{tag}Metrics", str(int((x.pair_flips > 0).sum())))
        macro(f"contam{tag}MaxFlips", str(int(x.pair_flips.max())))
        macro(f"contam{tag}Pairs", str(int(x.pairs.iloc[0])))
        macro(f"contamNMetrics", str(len(x)))


def risk_numbers() -> None:
    p = TABLES / "T4_risk.csv"
    if not p.exists():
        return
    t = pd.read_csv(p)
    d = t[(t.portfolio == "equal_weight") & np.isclose(t.level, 0.975) & t.fz0.notna()].sort_values("fz0")
    gens = d[d.kind == "generator"].reset_index(drop=True)
    for g, tag in (("cgnsde2", "VTwo"), ("cgnsde3", "VThree"), ("quantgan_fs", "QuantganFs"), ("garch_fhs", "GarchFhs"), ("diffusion_fs", "DiffusionFs")):
        i = gens.index[gens.method == g]
        if len(i):
            macro(f"riskGenRank{tag}", str(int(i[0]) + 1))
    macro("riskNGen", str(len(gens)))
    d99 = t[(t.portfolio == "equal_weight") & np.isclose(t.level, 0.99)]
    for g, tag in (("cgnsde2", "VTwo"), ("cgnsde3", "VThree"), ("hist_sim", "Hs"), ("garch_fhs_bench", "GarchBench")):
        x = d99[d99.method == g]
        if len(x):
            macro(f"viol{tag}", str(int(x.violations.iloc[0])))
    macro("violExp", num(d99.expected.iloc[0], 1))
    macro("nGreenNinetyNine", str(int((d99[d99.kind == "generator"].traffic == "green").sum())))


def tab_ablation_v3() -> None:
    p, ph = TABLES / "T2_ablation_v3_nse_full.csv", TABLES / "T2_ablation_v3_nse_pre2020.csv"
    if not ph.exists():
        return skipped.append("tab_ablation_v3")
    th = pd.read_csv(ph).set_index("generator")
    tf = pd.read_csv(p).set_index("generator") if p.exists() else None
    order = ["cgnsde3"] + [g for g in th.index if g != "cgnsde3"]
    rows = []
    for g in order:
        h = th.loc[g]
        name = tex_escape(g.replace("cgnsde3@", "-- ").replace("cgnsde3", "Full model (v3)").replace("_", " "))
        row = [name]
        if tf is not None:
            f = tf.loc[g] if g in tf.index else None
            row += [num(f.score_A), num(f.score_C)] if f is not None else ["--", "--"]
        row += [num(h.boost_pct, 1), num(h.score_A), f"{h.score_C:.2f} $\\pm$ {h.score_C_sd:.2f}", num(h.score_D)]
        rows.append(row)
        k = "".join(w.capitalize() for w in g.replace("cgnsde3@", "").replace("cgnsde3", "full").split("_"))
        macro(f"vThreeHeldC{k}", h.score_C); macro(f"vThreeHeldA{k}", h.score_A); macro(f"vThreeHeldCsd{k}", h.score_C_sd)
    head = ["Variant"] + (["$S_A$ in", "$S_C$ in"] if tf is not None else []) + ["Boost \\% held", "$S_A$ held", "$S_C$ held", "$S_D$ held"]
    write_table("t2b_ablation_v3", head, rows, "l" + "r" * (len(head) - 1))
    for mk in ("us", "kr"):
        t = t1(mk, "pre2020")
        if t is not None:
            t = t.set_index("generator")
            for g, tag in (("cgnsde3", "Full"), ("cgnsde2", "VTwo")):
                if g in t.index:
                    macro(f"confirm{mk.capitalize()}{tag}", t.loc[g].score_C)
        pa = TABLES / f"T2_ablation_v3_{mk}_pre2020.csv"
        if pa.exists():
            a = pd.read_csv(pa).set_index("generator")
            for g, tag in (("cgnsde3", "Full"), ("cgnsde3@no_regime", "NoRegime")):
                if g in a.index:
                    macro(f"confirm{mk.capitalize()}{tag}", a.loc[g].score_C)
                    macro(f"confirm{mk.capitalize()}{tag}Sd", a.loc[g].score_C_sd)


def step0_numbers() -> None:
    p, q = RESULTS / "step0" / "summary.json", RESULTS / "step0" / "fingerprints.json"
    if not p.exists():
        return skipped.append("step0")
    s = json.loads(p.read_text())
    macro("sZeroBlocks", str(s["french_blocks"]))
    for k, tag in (("m1", "Cascade"), ("m0a", "Evt"), ("m0b", "NoCascade")):
        macro(f"sZeroLs{tag}", s["french_mean_ls"][k]); macro(f"sZeroNseLs{tag}", s["nse_mean_ls"][k])
    macro("sZeroRecWins", str(s["P2_wins"])); macro("sZeroRecTotal", str(s["P2_total"]))
    macro("sZeroFOne", "held" if s["fingerprints"]["F1"] else "failed"); macro("sZeroFTwo", "held" if s["fingerprints"]["F2"] else "failed")


def tab_hypotheses() -> None:
    p = TABLES / "hypotheses.json"
    if not p.exists():
        return skipped.append("tab_hypotheses")
    h = json.loads(p.read_text())
    rows = [[k, tex_escape(v.get("verdict", "")), tex_escape(v.get("evidence", ""))] for k, v in h.items()]
    write_table("t8_hypotheses", ["Item", "Verdict", "Evidence"], rows, "llp{0.62\\linewidth}")
    for k, v in h.items():
        macro("verdict" + k, tex_escape(v.get("verdict", "")))


def numbers(summary: dict) -> None:
    for mk in ("nse", "us", "kr"):
        p = RESULTS / "reference" / mk / "full.json"
        if p.exists():
            r = json.loads(p.read_text())
            tag = mk.capitalize()
            macro(f"real{tag}Boost", r["metrics"]["boost_pct"]["real"])
            macro(f"real{tag}BoostLo", r["metrics"]["boost_pct"]["lo"])
            macro(f"real{tag}BoostHi", r["metrics"]["boost_pct"]["hi"])
            macro(f"real{tag}Windows", str(r["n_windows"]))
            macro(f"real{tag}Crisis", str(r["n_crisis"]))
    t = t1("nse", "full")
    if t is not None:
        for _, r in t.iterrows():
            k = "".join(w.capitalize() for w in r.generator.split("_"))
            macro(f"boost{k}", r.get("boost_pct", np.nan))
            macro(f"corr{k}", r.get("corr_fro", np.nan))
            macro(f"scoreC{k}", r.score_C)
            macro(f"scoreB{k}", r.score_B)
            macro(f"scoreA{k}", r.score_A)
        macro("nGenerators", str(len(t)))
    th = t1("nse", "pre2020")
    if th is not None:
        for _, r in th.iterrows():
            k = "".join(w.capitalize() for w in r.generator.split("_"))
            macro(f"heldBoost{k}", r.get("boost_pct", np.nan))
            macro(f"heldScoreC{k}", r.score_C)
            macro(f"heldRankC{k}", str(int(r.rank_C)) if pd.notna(r.rank_C) else "n/p")
        if t is not None:
            for _, r in t.iterrows():
                k = "".join(w.capitalize() for w in r.generator.split("_"))
                macro(f"rankC{k}", str(int(r.rank_C)) if pd.notna(r.rank_C) else "n/p")
                macro(f"rankA{k}", str(int(r.rank_A)) if pd.notna(r.rank_A) else "n/p")
    p = RESULTS / "reference" / "nse" / "pre2020.json"
    if p.exists():
        r = json.loads(p.read_text())["metrics"]["boost_pct"]
        macro("realHeldBoost", r["real"]); macro("realHeldBoostLo", r["lo"]); macro("realHeldBoostHi", r["hi"])
    p = TABLES / "T3_heldout.csv"
    if p.exists():
        t3 = pd.read_csv(p)
        for ep, tag in (("covid_2020", "Covid"), ("shock_2022", "Shock"), ("mid_2024", "Mid")):
            e = t3[t3.episode == ep]
            if len(e):
                macro(f"real{tag}Boost", e.real.iloc[0]); macro(f"real{tag}BoostLo", e.lo.iloc[0])
                macro(f"real{tag}BoostHi", e.hi.iloc[0]); macro(f"n{tag}Crisis", str(int(e.n_real_crisis.iloc[0])))
                g = e[e.generator == "cgnsde2"]
                if len(g):
                    macro(f"vTwo{tag}Boost", g.gen.iloc[0])
                macro(f"nInside{tag}", str(int(e.inside.astype(str).str.lower().eq("true").sum())))
    p = RESULTS / "01_dgp" / "metric_validation.csv"
    if p.exists():
        dg = pd.read_csv(p).set_index("panel")
        for panel, tag in (("sim_boost00", "Zero"), ("sim_boost20", "Twenty"), ("sim_boost40", "Forty"),
                           ("sim_boost80", "Eighty"), ("sim_ccc", "Ccc")):
            if panel in dg.index:
                macro(f"dgp{tag}Rule", dg.loc[panel, "rule_boost_pct"])
                macro(f"dgp{tag}Lo", dg.loc[panel, "rule_lo"]); macro(f"dgp{tag}Hi", dg.loc[panel, "rule_hi"])
    for v, gen in (("vOne", "cgnsde"), ("vTwo", "cgnsde2")):
        f = RESULTS / "runs" / "nse" / gen / "full" / "seed0" / "metrics.json"
        if f.exists():
            info = json.loads(f.read_text()).get("info", {})
            macro(f"{v}BestEpoch", str(info.get("best_epoch", "")))
            macro(f"{v}Epochs", str(info.get("epochs_run", "")))
            if info.get("n_params"):
                macro(f"{v}Params", f"{int(info['n_params']):,}".replace(",", "{,}"))
    (PAPER / "numbers.tex").write_text("% Generated by crisisbench.paper_build. Do not edit.\n" + "".join(
        f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())), encoding="utf-8")


def make_zip() -> None:
    main = PAPER / "main.tex"
    if not main.exists():
        return skipped.append("zip (paper/main.tex missing)")
    z = PAPER / "vortex_paper_overleaf.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in [main, PAPER / "references.bib", PAPER / "numbers.tex", *sorted(FIG.glob("*.pdf")), *sorted(TAB.glob("*.tex"))]:
            if f.exists():
                zf.write(f, f.relative_to(PAPER))
    print("zip:", z, round(z.stat().st_size / 1e6, 2), "MB")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    summary = json.loads((TABLES / "summary.json").read_text()) if (TABLES / "summary.json").exists() else {}
    for mk, fit in (("nse", "full"), ("nse", "pre2020")):
        fig_scatter(mk, fit)
        fig_scores(mk, fit)
        tab_main(mk, fit)
    for mk in ("us", "kr"):
        tab_main(mk, "full")
    tab_main("nse_ext", "upto2024")
    fig_heldout(); fig_var(); fig_dgp(); fig_ablation()
    tab_risk(); tab_validity(); tab_taus(summary); tab_ablation(); tab_ablation_v3(); tab_hypotheses(); contamination(); risk_numbers(); step0_numbers()
    numbers(summary)
    make_zip()
    print("figures:", len(list(FIG.glob("*.pdf"))), "tables:", len(list(TAB.glob("*.tex"))), "macros:", len(macros))
    if skipped:
        print("skipped (inputs missing):", ", ".join(skipped))


if __name__ == "__main__":
    main()
