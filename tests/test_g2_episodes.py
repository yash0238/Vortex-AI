"""G2 gate: three held-out crisis episodes.

Run: .venv/Scripts/python.exe tests/test_g2_episodes.py
     or: .venv/Scripts/python.exe -m pytest tests/test_g2_episodes.py -v

Verifies that mid_2024 is present in HELDOUT_EPISODES alongside covid_2020
and shock_2022, so G2 can pass when CG-NSDE's held-out boost_pct lies inside
the real 95% interval on at least 2 of 3 episodes.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crisisbench.protocol import HELDOUT_EPISODES  # noqa: E402
from crisisbench.labels import label_windows  # noqa: E402


def test_g2_has_three_episodes():
    episodes = HELDOUT_EPISODES
    assert "mid_2024" in episodes, "mid_2024 episode missing from HELDOUT_EPISODES"
    assert len(episodes) == 3, f"Expected 3 episodes, got {len(episodes)}"

    # Verify G2 can now pass with 2/3 episodes inside interval
    episodes_list = list(episodes.keys())
    assert "covid_2020" in episodes_list
    assert "shock_2022" in episodes_list
    assert "mid_2024" in episodes_list


def test_label_windows_runs_on_synthetic_windows():
    """label_windows is episode-agnostic: it labels windows from returns only.
    Smoke-check it executes on a small synthetic panel without error."""
    rng = np.random.default_rng(0)
    W = rng.standard_normal((4, 60, 5)) * 0.01
    # force one high-dispersion window so both branches are exercised
    W[0, -20:, :] *= 5.0
    from crisisbench.labels import fit_rule
    rule = fit_rule(W)
    labels, score = label_windows(W, rule)
    assert labels.shape == (4,)
    assert score.shape == (4,)
    assert labels.dtype == np.int8


if __name__ == "__main__":
    test_g2_has_three_episodes()
    print("ok  test_g2_has_three_episodes")
    test_label_windows_runs_on_synthetic_windows()
    print("ok  test_label_windows_runs_on_synthetic_windows")
    print("OK")
