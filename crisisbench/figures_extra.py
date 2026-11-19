"""Figures and diagnostics from the project masterplan that are not part of the frozen protocol
(reported as supplementary, never used for a verdict):

  F7  fan chart: cumulative equal-weight return over a 60-day window, real vs generated (5-95% bands),
      separately for windows the frozen rule labels calm and crisis
  F8  learned contagion graph of CG-NSDE v3 next to the real |correlation| matrix (assets ordered by sector)
  F9  strategy sandbox: distribution of 40-day P&L of (a) equal-weight buy-and-hold and (b) cross-sectional
      momentum (long top-10 / short bottom-10 by the first 20 days' return), real vs generated windows
  T9  extra stylised facts per generator: leverage effect corr(r_t, |r_{t+1..t+5}|) of the equal-weight
      portfolio, lead-lag strength (mean |lag-1 cross-correlation|), KS distance of the strategy P&L

    python -m crisisbench.figures_extra [--gens cgnsde3,cgnsde2,garch_fhs,...]
Uses the stored samples artifacts/samples/nse/<gen>/full/seed0.npy (2000 unconditional windows).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .labels import label_windows
from .analysis import PRETTY, TABLES
from .bench import ARTIFACTS, make_fit_data
from .paper_build import FIG, INK, MUTED, plt

DEFAULT = ["cgnsde3", "cgnsde2", "garch_fhs", "dcc_garch", "regime_gaussian", "stationary_bootstrap", "quantgan_fs",
           "timegan_fs", "iid_gaussian"]


def labels(W, rule):
    return label_windows(W, rule)[0].astype(bool)


def ew(W):
    return W.mean(2)                                                     # (n, T) equal-weight daily returns


def strat_pnl(W):
    """40-day P&L of buy-and-hold and of a 10/10 momentum book formed on the first 20 days."""
    bh = W[:, 20:, :].mean(2).sum(1)
    past = W[:, :20, :].sum(1)
    order = np.argsort(past, axis=1)
    lo, hi = order[:, :10], order[:, -10:]
    fut = W[:, 20:, :].sum(1)
    mom = np.take_along_axis(fut, hi, 1).mean(1) - np.take_along_axis(fut, lo, 1).mean(1)
    return bh, mom


def leverage(W):
    p = ew(W)
    r, a = p[:, :-5], np.stack([np.abs(p[:, k:p.shape[1] - 5 + k]) for k in range(1, 6)], 0).mean(0)
    r, a = r.ravel(), a.ravel()
    return float(np.corrcoef(r, a)[0, 1])


def leadlag(W):
    Z = (W - W.mean(1, keepdims=True)) / (W.std(1, keepdims=True) + 1e-12)
    C = np.einsum("wti,wtj->wij", Z[:, :-1], Z[:, 1:]) / (W.shape[1] - 1)
    N = W.shape[2]
    off = ~np.eye(N, dtype=bool)
    return float(np.abs(C.mean(0))[off].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", default=",".join(DEFAULT))
    a = ap.parse_args()
    fd, Wr, _ = make_fit_data("nse", "full")
    rule = fd.rule
    gens = [g for g in a.gens.split(",") if (ARTIFACTS / "samples" / "nse" / g / "full" / "seed0.npy").exists()]
    S = {g: np.load(ARTIFACTS / "samples" / "nse" / g / "full" / "seed0.npy") for g in gens}
    FIG.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- F7 fan chart
    show = [g for g in ("cgnsde3", "cgnsde2", "garch_fhs", "regime_gaussian") if g in S][:3]
    fig, axes = plt.subplots(2, 1 + len(show), figsize=(7.0, 3.6), sharey="row", sharex=True)
    for row, crisis in enumerate((False, True)):
        for col, (name, W) in enumerate([("Real", Wr)] + [(PRETTY.get(g, g), S[g]) for g in show]):
            lab = labels(W, rule)
            sel = W[lab == crisis]
            ax = axes[row, col]
            if len(sel) < 20:
                ax.text(0.5, 0.5, "too few windows", transform=ax.transAxes, ha="center", color=MUTED)
                continue
            cum = np.cumsum(ew(sel), 1) * 100
            x = np.arange(1, cum.shape[1] + 1)
            q = np.percentile(cum, [5, 25, 50, 75, 95], axis=0)
            colr = MUTED if col == 0 else "#2a78d6"
            ax.fill_between(x, q[0], q[4], color=colr, alpha=0.18, lw=0)
            ax.fill_between(x, q[1], q[3], color=colr, alpha=0.35, lw=0)
            ax.plot(x, q[2], color=INK if col == 0 else colr, lw=1.0)
            if row == 0:
                ax.set_title(name.replace(" (ours, post hoc)", ""), fontsize=7)
            if col == 0:
                ax.set_ylabel(("Crisis" if crisis else "Calm") + " windows\ncum. return %")
    for ax in axes[1]:
        ax.set_xlabel("day")
    fig.tight_layout()
    fig.savefig(FIG / "f7_fan.pdf")
    plt.close(fig)

    # ---------------------------------------------------------------- F8 contagion graph
    try:
        from .generators.cgnsde3 import CGNSDE3
        g3 = CGNSDE3.load(ARTIFACTS / "models" / "nse" / "cgnsde3" / "full" / "seed0.pt")
        A = g3.contagion_matrix()
        order = np.argsort(g3.sector, kind="stable") if g3.sector is not None else np.arange(len(A))
        Rr = np.abs(np.corrcoef(fd.returns.T))
        np.fill_diagonal(Rr, np.nan)
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
        im0 = axes[0].imshow(Rr[np.ix_(order, order)], cmap="Blues", vmin=0, vmax=0.8)
        axes[0].set_title("Real |correlation|", fontsize=8)
        im1 = axes[1].imshow(A[np.ix_(order, order)], cmap="Blues", vmin=0, vmax=max(0.3, float(A.max())))
        axes[1].set_title("Learned contagion graph $A$ (v3)", fontsize=8)
        for ax, im in zip(axes, (im0, im1)):
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        fig.tight_layout()
        fig.savefig(FIG / "f8_contagion.pdf")
        plt.close(fig)
    except FileNotFoundError:
        print("F8 skipped: no cgnsde3 checkpoint yet")

    # ---------------------------------------------------------------- F9 + T9 strategy sandbox, stylised facts
    rb, rm = strat_pnl(Wr)
    rows = [dict(generator="real", name="Real data", leverage=leverage(Wr), leadlag=leadlag(Wr),
                 bh_sd=rb.std() * 100, mom_sd=rm.std() * 100, bh_q01=np.percentile(rb, 1) * 100, ks_bh=0.0, ks_mom=0.0)]
    for g, W in S.items():
        b, m = strat_pnl(W)
        rows.append(dict(generator=g, name=PRETTY.get(g, g), leverage=leverage(W), leadlag=leadlag(W),
                         bh_sd=b.std() * 100, mom_sd=m.std() * 100, bh_q01=np.percentile(b, 1) * 100,
                         ks_bh=ks_2samp(b, rb).statistic, ks_mom=ks_2samp(m, rm).statistic))
    t9 = pd.DataFrame(rows)
    t9.to_csv(TABLES / "T9_sandbox_facts.csv", index=False)
    (TABLES / "T9_sandbox_facts.md").write_text(t9.round(4).to_markdown(index=False), encoding="utf-8")
    print(t9.round(3).to_string(index=False))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, (k, real, title) in zip(axes, ((0, rb, "Buy-and-hold, 40-day P&L %"), (1, rm, "Momentum 10/10, 40-day P&L %"))):
        bins = np.linspace(np.percentile(real * 100, 0.5), np.percentile(real * 100, 99.5), 50)
        ax.hist(real * 100, bins=bins, density=True, color=MUTED, alpha=0.5, label="Real")
        for g, colr in (("cgnsde3", "#2a78d6"), ("garch_fhs", "#eda100"), ("timegan_fs", "#6250d6")):
            if g in S:
                ax.hist(strat_pnl(S[g])[k] * 100, bins=bins, density=True, histtype="step", lw=1.1, color=colr,
                        label=PRETTY.get(g, g).replace(" (ours, post hoc)", ""))
        ax.set_title(title, fontsize=8)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "f9_sandbox.pdf")
    plt.close(fig)
    (TABLES / "T9_meta.json").write_text(json.dumps(dict(generators=gens)), encoding="utf-8")


if __name__ == "__main__":
    main()
