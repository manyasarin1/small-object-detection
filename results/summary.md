# Results Summary

## Confirmed numbers

### M3 (decode-bug fix, reference baseline for the plain-FPN ladder)
From `models/m3_1_loss_norm/train_m3_1.py`'s own docstring, which hardcodes this as "REFERENCE TO
BEAT" from an earlier completed run (M3 with fixed decoding, same evaluator and test chips):

- Best F1: **0.514** at threshold 0.075 (precision 0.628, recall 0.435)
- mAP50:95: **0.024**
- AP50: **0.059**
- AP75: **0.016**
- AP by size (small / medium / large): **0.012 / 0.036 / 0.105**

### M3.1 (+ per-positive loss normalization, + background prior p=0.99)
- mAP50:95: **0.091** (confirmed correct; supersedes an earlier internal figure of 0.065 which was wrong)
- Best plain-FPN result in the ladder so far.

### M2 (objectness head)
Re-evaluated with the fixed decoding as part of M3's and M3.1's `PREVIOUS_MODELS` pass (not from
M2's own original run, which used the pre-fix decoding). Exact re-evaluated figures were not
re-extracted into this repo snapshot — pull them from the M3 or M3.1 training logs if needed for
a precise number; qualitatively, M2 < M3 < M3.1 on mAP50:95.

### M1 (center-sampling assignment)
`models/m1_center_sampling/train_m1.py` still contains the original, pre-decode-fix evaluation
logic, so M1's own reported numbers are **not directly comparable** to M2 onward. M1 was also
not consistently found/re-evaluated by later scripts in every run (a checkpoint-filename lookup
issue was observed and never fully root-caused). Treat M1's isolated numbers as indicative only.

## M4 (CPNet backbone) — ⚠️ unresolved verification issue

M4 is reported elsewhere in this project's history as underperforming M3.1 (mAP50:95 ≈ 0.058)
despite roughly 5.5x the compute, which would make it a negative result for the CPNet backbone
at this scale. **This number has not been re-verified against a confirmed completed training
run in this repo pass.** When the M4 Kaggle notebook's (`manyasarin/set1-m4-cpnet`) version
history was checked, none of its saved versions corresponded to a genuine multi-hour completed
training run — the versions found were either "Quick Version" (seconds-long) runs or
"Save & Run All" attempts that were cancelled after a few minutes. The actual multi-hour M4
training run referenced throughout this project may live in a different notebook/version than
the one currently saved under that name, or may not have been preserved.

**Action needed:** before citing M4's numbers as verified in any paper or final write-up,
confirm which actual Kaggle run (version ID) produced them, and re-pull the real training log
from that run. Until that's resolved, M4's numbers in this repo should be treated as
provisional/historical, not verified.

## What has NOT been run

- **M5 / frequency (Gabor) branch** has not been executed. It exists only as a future step in
  the research plan (test only after the local object-background contrast module justifies it,
  as its own separate ablation). No results for it exist anywhere in this project.

## Research direction

Plain-FPN ladder (M2 → M3 → M3.1) shows clean, compounding gains from (a) fixing the decode bug
and (b) correctly normalizing the classification loss per positive location. CPNet (M4) has not
beaten M3.1 on the numbers available, pending the verification note above. Current plan does not
make future work depend on CPNet, and does not add a difficulty-weighted loss until a difficulty
score is independently defensible (see the 5-step experiment plan: freeze/audit M3.1 → establish
repeatability across seeds → build one local object-background contrast module → test a
Gabor/frequency branch only if justified, as a separate ablation → validate any "camouflage"/
difficulty claim against real annotations or a human-validated protocol).
