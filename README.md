# Small Object Detection — xView Ablation Ladder

Ablation study on the xView dataset, building up a small/difficult-object detector step by step
from a plain FPN baseline to a graph-based backbone (CPNet), then evaluating whether the backbone
swap is worth its cost.

## Ladder

| Model | Change on top of previous | Status |
|---|---|---|
| M0 | Plain FPN + cls/bbox heads, no center-sampling assignment | Never trained — conceptual baseline only, scripts already handle its absence |
| M1 | + FCOS-style center-sampling target assignment | Trained (old/unfixed decoding — see note below) |
| M2 | + objectness head (focal BCE, positive-normalized) | Trained |
| M3 | + centerness head, **decode-bug fix** (argmax restricted to the 60 real classes, excluding background) | Trained |
| M3.1 | + classification loss normalized per-positive-location (not per-level mean) + background-prior init (p=0.99) | Trained |
| M4 | Plain FPN replaced with **CPNet** backbone (Cascade Aggregation Pyramid + Ternary Cascade Perception Module) | Trained |

No model past M1 required retraining to produce this repo — each later script re-evaluates all
earlier checkpoints in-process with the current (fixed) decoding logic, so apples-to-apples
numbers come from each script's own `PREVIOUS_MODELS` re-evaluation pass, not from re-running
earlier notebooks.

## Decode bug

M0–M2's decoding used `probs.max(dim=0)` over all 61 softmax outputs (60 object classes +
background), which let the background class win the argmax and suppress real detections. M3
fixed this to `probs[1:].max(dim=0)`, restricting the argmax to the 60 real classes. From M3
onward, every earlier checkpoint is re-evaluated with the fixed decoding before comparison, so
the ladder's reported numbers (M2 onward) are apples-to-apples. M1's own training script
(`models/m1_center_sampling/train_m1.py`) still contains the original, pre-fix decoding, so M1's
numbers in isolation are not directly comparable to M2–M4 — see `results/` for details.

## Results (headline: mAP50:95)

| Model | mAP50:95 | Notes |
|---|---|---|
| M2 | see `results/` | Re-evaluated with fixed decoding |
| M3 | 0.024 | Fixed decoding; F1 0.514 @ thr 0.075 |
| M3.1 | 0.091 | Best plain-FPN result; corrected figure (an earlier internal note had this as 0.065, now confirmed wrong) |
| M4 (CPNet) | see `results/` | ~5.5x compute of M3.1, did not surpass it |

Full breakdown, per-class AP counts, and compute cost in `results/`.

## Conclusion so far

The plain-FPN ladder (M2 → M3 → M3.1) shows clean, compounding gains from fixing the decode bug
and correctly normalizing the classification loss. Swapping in the CPNet backbone (M4) did not
beat M3.1 despite substantially higher compute. Current research direction treats this as a
negative result for CPNet at this scale/budget and does **not** make the next steps depend on
CPNet. Next steps (in progress): freeze and audit M3.1 (per-class AP, AP-by-size, qualitative
examples, no retraining), establish repeatability across seeds, then explore one carefully
defined local object-background contrast module before considering any frequency/Gabor branch —
which has not yet been run; it exists only as a future hypothetical in the research plan, not as
a result in this repo.

## Repo layout

```
models/
  m1_center_sampling/train_m1.py
  m2_objectness/train_m2.py
  m3_decode_fix/train_m3.py
  m3_1_loss_norm/train_m3_1.py
  m4_cpnet/train_m4_cpnet.py
results/
  summary.md
```

Checkpoints (`.pt` files) are intentionally excluded — code and results only.
