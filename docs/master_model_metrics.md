# Model comparison: within-gene and pooled

| Model | Features | Within-gene win-rate [95% CI] | P@1 | Pooled AUROC | Pooled AUPRC | Source |
|---|---:|---|---:|---:|---:|---|
| Chance | 0 | **0.500** | 0.292 | 0.500 | 0.114 | analytic |
| Acceptor 3'ss alone | 1 | **0.620** [0.599, 0.639] | 0.398 | 0.611 | 0.149 | refit |
| Window GC alone [-500, 0) * | 1 | **0.630** [0.611, 0.649] | 0.426 | 0.611 | 0.164 | refit |
| GC dip depth alone | 1 | **0.630** [0.611, 0.648] | 0.405 | 0.613 | 0.154 | refit |
| Scalar splice model | 3 | **0.666** [0.648, 0.684] | 0.427 | 0.636 | 0.158 | refit |
| Positional splice (locked) | 51 | **0.696** [0.678, 0.714] | 0.459 | 0.663 | 0.188 | rescored |
| Full binned splice | 126 | **0.710** [0.694, 0.729] | 0.480 | 0.672 | 0.200 | refit |
| Positional splice + dip depth | 52 | **0.713** [0.697, 0.732] | 0.486 | 0.686 | 0.209 | refit |
| Positional splice + GC per bin | 76 | **0.740** [0.723, 0.758] | 0.518 | 0.705 | 0.219 | refit |
| Boosted tree on the 51 splice features * | 51 | **0.683** [0.665, 0.701] | 0.456 | 0.858 | 0.508 | refit |
| Sequence CNN | - | **0.821** [0.806, 0.836] | 0.648 | 0.818 | 0.449 | saved |

\* not a ladder step; see caption.

All models scored on the same 1,623 held-out test genes (14,198 peaks, 1,623 real 3' ends, base rate 0.114) over the window [-2000, +500]. Every metric is computed by the same code (binned_common.suite), so the pooled and within-gene columns are comparable across rows. **The within-gene win-rate is the headline**; pooled AUROC and AUPRC are shown for completeness. Two rows are marked as illustrations rather than ladder steps. Window GC scores higher within gene (0.630) than pooled (0.611), though the gap is modest; the boosted tree is the striking case, strong pooled (0.858) but below the ridge within gene (0.683). 0.696 is the splice-only ceiling; 0.740 is splice plus positional composition, which is a different axis.
