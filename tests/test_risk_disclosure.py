"""Disclosure guard: a collapsed generator must be excluded from the backtest,
and that exclusion must be recorded rather than silently dropped (Phase 4 final).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "06_risk"
GATE = ROOT / "results" / "gates" / "GATE_G3.md"


def test_degenerate_generator_is_disclosed():
    p = OUT / "excluded_generators.json"
    assert p.exists(), "degenerate-generator disclosure file is missing"
    d = json.loads(p.read_text(encoding="utf-8"))
    assert "timevae" in d, "the collapsed TimeVAE generator must be disclosed"
    rec = d["timevae"]
    assert rec["reason"], "exclusion must carry a reason"
    assert rec["min_es"] <= 0, "the recorded ES must be the observed non-positive value"
    assert rec["fits"], "the affected refit years must be listed"


def test_g3_gate_carries_the_exclusion():
    assert GATE.exists(), "GATE_G3.md is missing"
    text = GATE.read_text(encoding="utf-8")
    body = json.loads(text.split("```json", 1)[1].split("```", 1)[0])
    assert "excluded_generators" in body, "G3 body must disclose excluded generators"
    assert "timevae" in body["excluded_generators"]
    # The verdict must be consistent with the frozen H3 rule (strict CI non-overlap),
    # whatever that rule currently implies for the observed numbers.
    expected = "PASS" if body["H3_holds"] else "FAIL"
    assert text.splitlines()[0] == f"# GATE_G3: {expected}", (
        f"gate title {text.splitlines()[0]!r} contradicts H3_holds={body['H3_holds']}"
    )


def test_es_sign_convention_explains_the_failure():
    """Empirical ES is negative when the whole k-NN tail is positive."""
    import numpy as np
    from crisisbench.risk import empirical_var_es

    var, es = empirical_var_es(np.full((1, 100), 0.00087), [0.95])
    assert var[0, 0] < 0, "VaR convention: -quantile"
    assert es[0, 0] < 0, "a collapsed all-positive pool yields a non-positive ES"
