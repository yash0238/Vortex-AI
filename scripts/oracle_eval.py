"""Stage 4 oracle v2: block-purged disc, tight kurtosis, full+train references (raw space)."""
import json
import numpy as np
import sys
sys.path.insert(0, ".")
from eval.statistical import evaluate_stylized_facts
from eval.discriminative import discriminative_score
from eval.contagion import evaluate_contagion, cvar_regime_ratio

rng = np.random.default_rng(11)
W = np.load("data/raw/windows.npy").astype(float)
lab = np.load("data/raw/window_regimes_v2.npy")
sp = np.load("data/raw/split.npz")
te = np.asarray(sp["test_idx"])
tr = np.asarray(sp["train_idx"])
Xt, lt = W[te], lab[te]
print("test windows:", len(Xt), "crisis:", int(lt.sum()))
out = {}

# A: two NON-ADJACENT chronological real blocks (block CV + purge inside disc fn)
n = min(500, len(Xt))
b1, b2 = Xt[:200], Xt[538:738]
print("block gap windows:", 538 - 200, "crisis b1/b2:", int(lab[te][:200].sum()), int(lab[te][538:738].sum()))
out["A_styl"] = evaluate_stylized_facts(b1, b2)
out["A_disc"] = discriminative_score(b1, b2, n_samples=200)
print("A styl:", {k: round(float(v), 4) if isinstance(v, float) else v for k, v in out["A_styl"].items()})
print("A disc:", {k: round(float(v), 4) if isinstance(v, float) else v for k, v in out["A_disc"].items()})

# B: iid Gaussian, same block protocol
mu, sd = Xt.mean(axis=(0, 1)), Xt.std(axis=(0, 1))
G = rng.normal(mu, sd, size=(n, Xt.shape[1], Xt.shape[2]))
out["B_styl"] = evaluate_stylized_facts(Xt[:n], G)
out["B_disc"] = discriminative_score(Xt[:n], G, n_samples=n)
print("B styl:", {k: round(float(v), 4) if isinstance(v, float) else v for k, v in out["B_styl"].items()})
print("B disc:", {k: round(float(v), 4) if isinstance(v, float) else v for k, v in out["B_disc"].items()})

# C: references on FULL dataset and TRAIN only; CVaR on 60-day equal-weight cumulative log return
def refs(X, l, tag):
    c = evaluate_contagion(X, l)
    v = cvar_regime_ratio(X, l)
    print(f"{tag}: n={len(X)} crisis={int(l.sum())} boost={c['crisis_corr_boost']:.4f} "
          f"cvar_n={v['cvar_normal']:.6f} cvar_c={v['cvar_crisis']:.6f} ratio={v['cvar_ratio']:.4f}")
    return {"contagion": c, "cvar": v, "n": len(X), "n_crisis": int(l.sum())}

out["C_full"] = refs(W, lab, "C full ")
out["C_train"] = refs(W[tr], lab[tr], "C train")

with open("results/oracle_stage4.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
print("saved results/oracle_stage4.json")
