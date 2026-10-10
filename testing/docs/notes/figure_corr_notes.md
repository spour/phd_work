# `figure_corr` methods notes

Panel titles carry the claim; everything a reader needs to check the claim is here,
keyed by panel letter. Nothing in this document is recomputed for the figure: all
numbers come from `task_corr_panels.py`, `task_binning_test.py` and `task_rerank_corr.py`.

## The corrected negative set

Competitors are dropped if either flag is set. Canonical peaks are never dropped.

- `apa_annotated`: a PolyASite cluster or a PolyA_DB site within 500 nt of the pA.
  These are real alternative 3' ends, so they are not model errors.
- `in_reference_terminal_exon`: the 500 nt strand-oriented window ending at the pA
  overlaps the last exon of any GENCODE v49 transcript by at least 1 nt. Permissive
  by design; the argmax-overlap reading (`fine_compartment == 2`) would remove 3,758
  competitors instead of 6,370.

Counts: 64,140 competitors in, 18,119 APA-annotated, 6,370 terminal exon, 5,294 both,
19,195 removed in total, 44,945 retained. 1,660 genes lost every competitor and were
dropped with their canonical. Analysis set 49,868 peaks = 4,923 canonical + 44,945
competitor in 4,923 genes.

This removes the TERM-TERM stratum by construction, so the compartment-control panel
of the previous figure cannot be computed here. Panels g and h replace it.

## Correction to the feature-definition table

The featurisation window is **[pA-2000, pA+500] in 8 bins of 312.5 nt**, not the
1000/500 that `scripts/slurm_featurize_shortlist.sh` defaults implied. Recovered
empirically: 2000/500 reproduces `spl_upstream_max_me3`, `spl_pas_prox_u1`,
`spl_u1_suppression_up`, `spl_local_me5` and `spl_term_3ss` to 1e-4 on 200 random
peaks; 1000/500 reproduces none of them.

- `spl_upstream_max_me3`: max MaxEntScan `score3` over bins 1-6, i.e. 23-mers starting
  pA-2000 .. pA-126 (AG at pA-1982 .. pA-108). Strictly upstream. The earlier note
  that this feature leaks past the pA was wrong.
- `spl_pas_prox_u1`: mean U1 duplex energy as -dG over 9-mers starting pA-437 .. pA-126.
- `spl_u1_suppression_up`: unchanged, `_donor_density(seq, upstream_nt-500, upstream_nt,
  0.6)` = count per 100 nt of GT-cored 9-mers scoring at least 0.6 on the hand-built
  `_DONOR` PWM over [pA-500, pA).

## Model selection

Forward selection on the corrected set (28-feature pool, LOCO on 80% of genes, untouched
20% gene holdout at seed 42) reproduces the previous ordering exactly:
`spl_upstream_max_me3`, then `spl_pas_prox_u1`, then `spl_u1_suppression_up`. The
holdout plateau is 0.4302 at k=6 and every k from 2 up has a holdout CI covering it, so
the strict rule selects **k = 2** (coefficients +0.4266 and -0.1537, both CI-clean).
**k = 3 is used here**: the third feature is CI-clean with zero sign flips (-0.1562
[-0.2031,-0.1172]) and raises holdout P@1 from 0.4049 to 0.4180, excess over chance
+0.115 to +0.128. k=2 is the strict-rule alternative and is reported in
`rerank_corr_path.tsv` / `rerank_corr_coefs.tsv`.

## The binning test

Both binned blocks use one grid: 25 bins of 100 nt over [pA-2000, pA+500), the span the
current features use. Per bin, max and mean for MaxEnt 3'ss (23-mer, AG required at
0-based 18-19), MaxEnt 5'ss (9-mer, GT at 3-4) and U1 (-dG over every 9-mer); GC is the
G+C fraction of the same bin. A bin with no scoreable position is NaN and mean-imputed
inside each training fold, exactly as the existing features are; GC is never NaN, which
is part of what the comparison tests. MaxEnt `score3` was vectorised for this (the
maxentpy decomposition is nine fixed-width table lookups); it matches maxentpy on 300
random 23-mers and the rebuilt pipeline features match the parquet to 0.000000 on 300
peaks.

Holdout results, 988 genes, chance P@1 0.2902 (permutation 0.2900), chance MRR 0.5183
(permutation 0.5186), base rate 0.1005:

| variant | features | P@1 | MRR | AUROC | AUPRC |
| --- | --- | --- | --- | --- | --- |
| three features | 3 | 0.4180 | 0.6111 | 0.6144 | 0.1438 |
| + binned splicing | 153 | 0.4514 | 0.6343 | 0.6667 | 0.1907 |
| binned GC alone | 25 | 0.4393 | 0.6207 | 0.6638 | 0.1793 |
| + binned GC | 28 | 0.4666 | 0.6485 | 0.6942 | 0.2085 |
| + binned splicing + GC | 178 | 0.4919 | 0.6627 | 0.6976 | 0.2201 |

Paired increments (same holdout genes, same 200 gene resamples) over the three features:
binned splicing +0.0334 P@1 [+0.0040,+0.0608] and +0.0523 AUROC [+0.0414,+0.0644];
binned GC +0.0486 [+0.0192,+0.0730] and +0.0798 [+0.0646,+0.0932]; both +0.0739
[+0.0465,+0.1012] and +0.0833 [+0.0703,+0.0966]. Over three features + binned GC, adding
binned splicing still gains +0.0253 P@1 [+0.0030,+0.0476] but its AUROC gain is null
(+0.0034 [-0.0042,+0.0131]). Over binned GC alone, adding the three features gains
+0.0273 P@1 [+0.0010,+0.0527] and +0.0304 AUROC [+0.0189,+0.0420].

Binning helps both blocks. Per feature it helps composition more, but see the
matched-count section below before reading that as complementarity.

## Panels

**a.** Exclusion-flag composition by class, as a fraction of each class. Flags are
computed for canonical peaks too and are shown, not applied: 97% of canonical peaks
carry at least one flag, which is what makes them canonical. Bar labels are peak counts.

**b.** PolyA_DB nearest-site usage. RPM and PSE, median with a gene-free percentile
bootstrap CI, IQR as the light bar, log axis. Retained competitors have **no** PolyA_DB
site within 500 nt by construction (the fraction with a site within 500 nt is 0.952 for
canonical, 0.557 for removed, 0.000 for retained), so their plotted values belong to a
distant site and should be read as a floor, not as usage of the peak itself. The point
of the panel is the ordering: canonical 3.69 RPM / 47.8% PSE, removed 0.36 / 4.9%,
retained 0.30 / 2.5%.

**c, d.** ROC and precision-recall over all 49,868 analysis peaks, LOCO. The baseline is
`rf_prob`, the model whose false positives define the competitor set, which is why its
AUROC is below 0.5 here: within a contested gene it ranks the canonical last by
construction. Base rate 0.099.

**e.** P@1 per held-out chromosome with a gene bootstrap CI; the grey point is that
chromosome's own chance rate, which varies with candidate-set size. Chromosomes with
fewer than 20 genes are omitted.

**f.** Standardized coefficients of the three named features, in the three-feature fit
and inside the 178-feature model, with 95% CI over gene-clustered bootstrap and the
sign-flip fraction. `spl_upstream_max_me3` holds (+0.427 to +0.395, zero flips).
`spl_u1_suppression_up` holds (-0.156 to -0.111). `spl_pas_prox_u1` does not: it moves
from -0.126 [-0.190,-0.064] to +0.056 [-0.116,+0.224] with 30% sign flips, because the
binned GC block absorbs it. That feature correlated +0.66 with single-window GC in the
earlier check, so this is the expected outcome, and it means the U1-proximity term
should not be interpreted mechanistically once composition is in the model.

**g.** Held-out discriminant of the three-feature model against GC of the 500 nt window
ending at the pA. Within-class Pearson r: canonical +0.031, competitor -0.110 (the
earlier k=2 model gave -0.006 and -0.135). Line is a per-class linear fit. This panel
says only that the mechanistic model is not a readout of *one* GC number; panel h says
what happens when GC is given a positional profile instead.

**h.** Composition as a competing explanation. P@1 with gene bootstrap CI, dashed line
is chance (0.285); the annotation gives AUROC and the within-gene win-rate (fraction of
canonical-versus-competitor pairs the predictor gets right; 0.5 is chance). A single
500 nt GC window is worse than chance (win-rate 0.360, P@1 0.199, AUROC 0.452) because
competitor windows are GC-richer on average in a way that does not order candidates
within a gene. The same nucleotide content binned into 25 windows is a strong predictor
(win-rate 0.657, P@1 0.424, AUROC 0.663), so the earlier "GC is anti-predictive"
statement was an artifact of collapsing the profile to one number. Residualising the
three-feature discriminant on single-window GC costs almost nothing (P@1 0.4105 to
0.4018, AUROC 0.626 to 0.623).

**i.** What each block separates. Every block is fitted on the real task (canonical
versus competitor, LOCO) and its discriminant is then read on three populations:
canonical peaks, retained competitors, and the matched intronic decoys of those
competitors. Blue is the canonical - competitor gap, plum is the competitor - decoy gap,
both with a gene-clustered bootstrap CI.

The three named features separate canonical from competitor (+0.211 [+0.196,+0.226]) and
place model false positives **level with random intronic sequence** (+0.004
[-0.004,+0.011], CI covering zero). That is the behaviour a measure of terminal-exon
splicing architecture should have: a competitor lacks that architecture as completely as
background intron does, so the feature set reads compartment and not CPA-likeness.

Both binned blocks instead order canonical > competitor > background (splice-binned
+0.304 then +0.088; GC-binned +0.351 then +0.150). They partly encode CPA-likeness
itself, which is the quantity the baseline models were already selecting on when they
produced the competitors. This is the strongest single argument for reporting the
three-feature model as the claim and the 178-feature model as a ceiling.

## Compartment control: matched intronic decoys

The TERM-TERM probe cannot be run on the corrected set, so compartment is controlled by
construction instead. For each retained competitor, five positions are drawn from the
same gene that are inside a GENCODE intron, outside every terminal exon, at least 1 kb
from any PolyASite repSite, at least 1 kb from any shortlist peak of that gene, and
matched to the competitor on distance to the nearest annotated 3'ss and 5'ss (nearest
neighbour in log distance, greedy in a seed-42 order, without replacement inside a gene).
223,960 decoys for 44,799 competitors; 146 competitors had no usable intronic candidate.
Match quality: nearest 3'ss median 6,123 nt for competitors versus 6,095 nt for decoys
(IQR [1616,19236] versus [1629,19181]); nearest 5'ss 5,454 versus 5,501 nt.

Ranking one competitor against its five decoys, same protocol, 9,169 holdout sets in 977
genes, chance P@1 0.1667, chance MRR 0.4084, base rate 0.1667:

| block | P@1 | MRR | AUROC | AUPRC |
| --- | --- | --- | --- | --- |
| GC-binned (25) | 0.4065 [0.3918,0.4205] | 0.6090 | 0.7171 [0.7068,0.7264] | 0.3127 |
| splice-binned (150) | 0.3481 [0.3355,0.3632] | 0.5603 | 0.6648 [0.6561,0.6743] | 0.2673 |
| k3 splicing (3) | 0.2943 [0.2827,0.3054] | 0.4764 | 0.5058 [0.4983,0.5146] | 0.1704 |
| union (178) | 0.4249 [0.4083,0.4438] | 0.6257 | 0.7347 [0.7235,0.7454] | 0.3378 |

Neither binned block collapses, so neither is a readout of intron versus terminal exon.
Scoring decoys with the canonical-versus-competitor model gives a monotone ordering:
GC-binned canonical -2.031, competitor -2.382, decoy -2.532; splice-binned -2.070,
-2.374, -2.462. The three named features order canonical above competitor (+0.211) but
place competitors and intronic background together (+0.004), which is what a measure of
terminal-exon splicing architecture should do: a competitor lacks it as completely as
random intronic sequence does.

## Block overlap: additive but overlapping, not independent

The two binned blocks share a dominant direction. First canonical correlation **0.893**,
**20 of 25** canonical correlations above 0.5 (none above 0.9); every GC bin is about
half predictable from the splice block (**mean R2 0.528**, median 0.533), while the
splice block is mostly not GC (median R2 0.051, max 0.401). Co-located correlations run
as the mechanism predicts: U1 mean +0.63, U1 max +0.38, MaxEnt 3'ss mean -0.29 (the
pyrimidine tract), MaxEnt 5'ss mean +0.14.

At **matched feature count** the blocks do not complement each other. Holdout, 988 genes,
chance P@1 0.2902, feature subsets inside each block ranked on dev genes only:

| variant | features | P@1 | MRR | AUROC | AUPRC |
| --- | --- | --- | --- | --- | --- |
| splice-binned | 150 | 0.4312 | 0.6136 | 0.6400 | 0.1640 |
| GC-binned | 25 | 0.4393 | 0.6207 | 0.6638 | 0.1793 |
| three features | 3 | 0.4180 | 0.6111 | 0.6144 | 0.1438 |
| best-3 GC bins | 3 | 0.3927 | 0.5930 | 0.5915 | 0.1395 |
| best-25 splice bins | 25 | 0.3998 | 0.5911 | 0.6034 | 0.1377 |
| best-25 splice + 25 GC | 50 | 0.4393 | 0.6238 | 0.6736 | 0.1853 |
| union | 178 | 0.4919 | 0.6627 | 0.6976 | 0.2201 |

Adding the best 25 splice bins on top of the 25 GC bins is worth **+0.0000 P@1
[-0.018,+0.020]** and +0.0098 AUROC [+0.0019,+0.0188]. The union's advantage over
GC-binned (+0.0526 P@1 [+0.0243,+0.0850]) therefore comes from having **150 splice
features against 25 GC features**, not from complementarity between the blocks. At three
features the hand-built splice model beats the best three GC bins (0.4180 versus 0.3927,
difference -0.0253 [-0.0678,+0.0122], CI crossing zero).

The correct phrase for the two blocks is **additive but overlapping**. Caveat on the
subset rows: "best-k" is a univariate dev ranking by |AUROC - 0.5|, which serves the
splice block poorly (its best-25 subset scores below three hand-built features), so the
matched-count comparison is a floor for splicing, not a ceiling.

## What binned GC is reading: a positional contrast, not a level

Per-bin univariate AUROC (all 49,868 analysis peaks; `block_perbin.tsv`) has a sign flip
at **-350 nt**:

| bin centre | AUROC | mean GC canonical | mean GC competitor | gap |
| --- | --- | --- | --- | --- |
| -1950 | 0.559 | 0.430 | 0.406 | +0.024 |
| -1250 | 0.558 | 0.430 | 0.407 | +0.024 |
| -550 | 0.529 | 0.422 | 0.407 | +0.014 |
| -350 | 0.491 | 0.412 | 0.412 | +0.000 |
| -150 | 0.429 | 0.391 | 0.412 | -0.021 |
| -50 | 0.423 | 0.332 | 0.350 | -0.019 |
| +250 | 0.487 | 0.414 | 0.418 | -0.003 |
| +450 | 0.543 | 0.422 | 0.405 | +0.017 |

Canonical peaks are GC-richer than competitors across a broad distal plateau
(-2000 to -450, AUROC about 0.55) and again downstream (+350 to +500), and AT-richer in
[-350, +250], deepest at -150 and -50. The pA dip itself is shared by both classes
(0.332 versus 0.350). The block therefore reads a **contrast between distal and proximal
composition**, not a level, and it accumulates: best single bin AUROC 0.553, best 3
0.592, best 5 0.610, best 10 0.637, all 25 0.664. This also explains the single-window
result mechanically: the 500 nt window [-500, 0) straddles the sign flip, the halves
cancel, and what survives is AUROC 0.452.

Two corrections to the earlier GC-profile read. The largest class gap on the corrected
set is **0.021 at -150 nt** with per-position AUROC 0.429, not 0.052 at -165 nt with
AUROC 0.377; the earlier figures came from a 500-gene paired sample of the *uncorrected*
pool, whose APA-annotated and terminal-exon competitors carried most of the difference.
And the discriminating region is not only [-500,-100]: the opposite-signed distal
plateau contributes as much.

**This mechanism needs its own analysis and is deliberately not in the figure.** It is
not a coefficient story: per-bin ridge coefficients in a block with first canonical
correlation 0.893 are not interpretable one at a time, and the profile above is
univariate. What the figure claims about composition is limited to panels h and i.

## What this figure does not show

- The 178-feature model is not a mechanistic claim. It is the ceiling reachable with
  local sequence summaries under this protocol, and most of its advantage over the
  three features is composition, not splice-site strength. 175 of its 178 features are
  bins with no individual meaning, and `spl_pas_prox_u1` loses its sign inside it.
- The binned-GC signal is **not** compartment: the decoy control fixes compartment by
  construction and GC-binned still reaches AUROC 0.717 there. What it is instead, a
  distal-versus-proximal composition contrast that overlaps CPA-likeness, is described
  above and is left for a separate analysis.
- The decoy control tests competitor against intronic background. It does not, and
  cannot, control compartment for the canonical-versus-competitor contrast itself:
  canonical peaks are terminal-exon by definition, so no matched canonical decoy exists
  inside an intron.
