# Review brief — merged publishability report

## Task

You are a fresh reviewer. You did not write the document and have no prior conversation. Your job is to attack it, not to endorse it.

Target: `research/notes/final_report_vortex-ai-merged-publishability.md`

## Sources under test

- `research/notes/final_report_vortex-ai-publishability-70ddb2.md` (report A)
- `Vortex-AI/research/notes/final_report_vortex-cgnsde-publishability-2ad971.md` (report B)

The merged report claims to reconcile these two against the repository. Check that it does so honestly.

## What to check

1. **Misattribution.** Does any number appear under the wrong source, track, fit, or market? The legacy six-gate track, the dev-period numbers, and the CrisisBench numbers use different data and scalings. Flag every place a real reference value from one track is compared with a generator value from another.

2. **Overstatement.** Does the merged report assert anything the repo does not support? Specifically verify:
   - G1/G2/G3/G4 verdicts and their exact rules against `Vortex-AI/crisisbench/protocol.py` and `Vortex-AI/results/gates/*.md`
   - `Vortex-AI/results/tables/hypotheses.json`
   - `Vortex-AI/results/tables/T3_heldout.md` and `T1_nse_full.md`
   - `Vortex-AI/results/eval_current.json`
   - ν, bounds, NLL scaling, sampler mismatch in `models/neural_sde.py`, `training/vortex_model.py`, `training/config.yaml`
   - v2 architecture claims in `crisisbench/generators/cgnsde2.py` (one ν or per-factor ν; default innovations; number of variants; where the graph enters)

3. **Hidden disagreements.** Where the two source reports conflict, does the merged report resolve the conflict with evidence, or does it paper over it? Name any conflict that is still unresolved and should be left unresolved.

4. **Causal claims.** The merged report ranks three defects and says report A's "missing gates are the cause" is a causal error. Test that reasoning. Is the ranking supported, or is it a plausible story?

5. **Journal-review exposure.** What would a referee attack first that the report does not already concede?

## Rules

- Read-only. Do not edit the target or any other file.
- Quote the exact line or number you are disputing.
- Separate "wrong" from "unsupported" from "unclear".
- No praise, no summary of what the document does well unless it bears on a finding.
- If a claim checks out, say so in one line; do not pad.
