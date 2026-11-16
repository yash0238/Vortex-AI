# Common brief for sub-agents (read fully before your assignment)

Project root: c:/Users/DELL/Downloads/Major Project/Vortex-AI (Windows 11, bash available; Python =
.venv/Scripts/python.exe, CPU-only torch 2.14, 10 cores / 12 GB RAM, NO GPU). A benchmark job may be using the CPU.

This is "CrisisBench", the evaluation package of the VORTEX-AI project (synthetic multi-asset equity return
generators, judged on whether they reproduce crisis-time dependence). READ FIRST: crisisbench/protocol.py,
crisisbench/generators/base.py (the interface contract), crisisbench/bench.py (runner),
crisisbench/generators/classical_static.py and crisisbench/generators/__init__.py (worked example + registry:
module and class names you must match). Skim crisisbench/metrics.py only if you need it.

HARD RULES
- Create/edit ONLY the files assigned to you. Never edit protocol.py, metrics.py, labels.py, bench.py, data.py,
  base.py, generators/__init__.py. If one needs a change, describe it in your final report instead.
- Research integrity: never hardcode, invent or tune toward a result. No silent fallbacks (raise instead).
  Deviations from a published method go in the module docstring and in your report.
- sample() returns RAW daily log returns, shape (n, 60, N), finite, UNCONDITIONAL (no real window/label/date at
  sampling time). Deterministic: seed numpy/torch from fit(seed); sample() uses only the rng passed in.
- CPU manners: torch.set_num_threads(4); tests use tiny budgets. Do NOT run full-budget training or the benchmark
  queue; measure seconds per epoch/iteration at real data size and report the projected full-fit minutes.
- Never kill processes you did not start; no blanket kills (no taskkill /IM, no pkill).
- Smoke-test through the real runner with a throwaway tag, then delete only what you created:
    .venv/Scripts/python.exe -m crisisbench.bench --market nse --gen NAME --fit pre2020 --seed 0 --n-gen 400 --tag _test_NAME --force --hparams "{\"epochs\":1}"
  (outputs: results/runs/nse/_test_NAME, artifacts/samples/nse/_test_NAME, artifacts/models/nse/_test_NAME)
- No new dependencies. No pytest: tests are plain scripts under tests/ run as "python tests/FILE.py" that assert
  and print OK.
- Be frugal: do not re-read files you have already read; keep tool calls few and batched.
- Before finishing, re-read your code as a hostile reviewer (maths vs the cited method, leakage, NaN / non-PSD /
  exploding variance, tests that cannot fail) and fix what you find.

FINAL REPORT (your return value; markdown, at most 400 words): files written; default hyperparameters; measured
timings and projected full-fit minutes; the metric line printed by the runner in your smoke test; every deviation
from the reference method; limitations; any change needed in a file you may not edit.
