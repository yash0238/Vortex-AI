"""Test for G3 gate (VaR/ES backtest).  Run: .venv/Scripts/python.exe -m pytest tests/test_vares_backtest.py -v"""
from pathlib import Path
import json


def parse_gate_result(content: str) -> dict:
    """Parse a gate markdown file and return the JSON body."""
    # Find the JSON block
    import re
    match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
    if not match:
        raise ValueError("No JSON block found in gate file")
    return json.loads(match.group(1))


def test_g3_gate_exists_and_passes():
    """G3 gate file exists and contains τ(family C, VaR/ES) > 0.5 check."""
    gate_file = Path("results/gates/GATE_G3.md")
    assert gate_file.exists(), "GATE_G3.md missing - G3 never run"

    content = gate_file.read_text()
    assert "tau_C" in content
    assert "tau_B" in content

    # Parse result
    result = parse_gate_result(content)
    assert result["tau_C"] > 0.5
    assert result["tau_C"] > result["tau_B"]


if __name__ == "__main__":
    test_g3_gate_exists_and_passes()
    print("Test passed!")