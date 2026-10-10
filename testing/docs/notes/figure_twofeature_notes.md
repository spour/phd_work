# `figure_twofeature` methods notes

Per-site classification on the upstream-only candidate set: **one splicing feature and one
composition feature**. Panel titles carry the claim; everything needed to check a claim is
here. No number in this document was computed under an earlier candidate definition.

**The headline metric is the gene-averaged within-gene win-rate**, because its permutation
null is 0.5 by construction and measured at 0.4999. Pooled AUROC is reported beside it with
its own permutation null, which is not 0.5. See "The pooled-AUROC null is not 0.5".

**The headline numbers are the untouched-holdout ones: win-rate 0.645, AUROC 0.649 against a
null of 0.527, excess +0.121.** Panel d of the figure is LOCO at full n and reads 0.655 and
0.658; the two protocols and why they differ are set out under "Which number is the
headline" in the figure caption.

## Candidate definition

Three filters on the 64,140 competitors of the contested-gene shortlist:

1. `apa_annotated` — a PolyASite cluster or PolyA_DB site within 500 nt (18,119 removed).
   These are real alternative 3' ends, not model errors.
2. `in_reference_terminal_exon` — the 500 nt strand-oriented window ending at the pA
   overlaps a GENCODE v49 last exon by at least 1 nt (1,076 removed beyond the APA set).
3. **downstream of the canonical pA in transcript orientation (1,398 removed).**
   Transcription terminates at the used poly(A) site, so sequence past it was never a
   candidate for premature cleavage; scoring it was an artifact of the scan window.

Kept: **43,547 competitors and 4,847 canonical sites in 4,847 genes**, median 4 and mean
10.0 candidates per gene. 76 genes lost every competitor at step 3 and were dropped.

**Why ranking metrics are gone.** After filter 3 the canonical site is by construction the
last candidate in every gene, so the most-downstream rule scores P@1 = 1.0000 over all
4,847 genes. Any within-gene top-1 metric now answers a question the candidate definition
already settled. P@1 and MRR are retired; the task is per-site classification and the
metrics are pooled AUROC, AUPRC against its base rate, and the gene-averaged within-gene
win-rate. Chance is win-rate 0.5 and AUPRC = base rate 0.1012, both definitional; for
pooled AUROC chance is the score's own within-gene permutation null, which sits near 0.53,
not 0.5.

## Model

**Two features, one coefficient each. One splicing, one composition.**

| feature | standardized beta | 95% CI | sign flips | marginal AUROC | its null | excess | marginal win-rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `spl_upstream_max_me3` | +0.4273 | [+0.3898,+0.4690] | 0.000 | 0.6194 | 0.5328 | +0.0866 | 0.6100 |
| `gc_contrast` | +0.4053 | [+0.3752,+0.4393] | 0.000 | 0.6097 | 0.5155 | +0.0942 | 0.6108 |

    gc_contrast = mean G+C over pA-2000..-438  minus  mean G+C over pA-437..-126

the same two windows the retired `u1_contrast` used, k-mer start offsets on the pipeline
grid. **Untouched holdout, 974 genes and 9,624 sites, the headline: win-rate 0.6450
[0.6196,0.6682], AUROC 0.6486 against its null of 0.5272, excess +0.1214**, AUPRC 0.1689
against a base rate of 0.1012. LOCO over all 48,394 analysis sites, which is what panel d
draws: win-rate 0.6548, AUROC 0.6576, null 0.5326, excess +0.1249. The marginal columns in
the table above are LOCO-free, computed on the feature itself over all analysis sites.

**Why `u1_contrast` was replaced.** The Almada protection panel was retired because a
composition-matched non-donor motif reproduced it. The same control was then put to the
model's own U1 feature (`task_control_contrast.py`), over the identical windows, with each
score judged against its own permutation null:

| contrast over pA-2000..-438 minus pA-437..-126 | AUROC | its null | excess | win-rate |
| --- | --- | --- | --- | --- |
| real U1 table (`u1_contrast`) | 0.5747 | 0.5101 | +0.0646 | 0.5853 |
| U1 table with its values permuted across 9-mers, seed 0 | 0.4966 | 0.4991 | -0.0025 | 0.4902 |
| the same, seed 1 | 0.4927 | 0.5004 | -0.0077 | 0.4903 |
| the same, seed 2 | 0.5134 | 0.5045 | +0.0089 | 0.5081 |
| real U1 table read on the reverse complement | 0.5766 | 0.5099 | +0.0667 | 0.5818 |
| MaxEnt 5'ss over the same windows | 0.5228 | 0.5035 | +0.0193 | 0.5291 |
| **G+C fraction over the same windows** | **0.6097** | 0.5155 | **+0.0942** | **0.6108** |

The permuted tables sit at their null, so the window geometry is not the artifact: there is
real sequence structure here. But the table read on the **reverse complement**, which cannot
be U1 recognition, reproduces the effect exactly, and it correlates with the real contrast
at only -0.149, so the two are reading a shared underlying gradient rather than U1 binding.
GC over the same windows beats the U1 version outright.

The partition settles it. Residualising each on the other (OLS over the analysis rows):

| | AUROC | its null | excess | win-rate |
| --- | --- | --- | --- | --- |
| `u1_contrast` with the GC gradient removed | 0.5086 | 0.4999 | **+0.0086** | 0.5113 |
| `gc_contrast` with `u1_contrast` removed | 0.5833 | 0.5125 | **+0.0708** | 0.5839 |

and at the model level, holdout win-rate: acceptor alone 0.6087, + `u1_contrast` 0.6400,
+ `u1_contrast` with GC removed **0.6082** (back to the acceptor alone), + `gc_contrast`
**0.6450**, + both 0.6479. `u1_contrast` was a lossy proxy for the GC gradient. The model
now carries the gradient directly and calls it composition. `figure_supp_u1` is the panel
version of this table; `r(u1_contrast, gc_contrast) = +0.594`.

## What gc_contrast measures, and what it does not

**The proposed reading was that real 3' ends have an extended AT-rich approach, so they
should be AT-richer than competitors in the distal window. They are not. The sign is the
other way round** (`task_gc_windows.py`, `gc_windows.tsv`):

| window | canonical | competitor | difference |
| --- | --- | --- | --- |
| distal, pA-2000..-438 | 0.4271 | 0.4068 | **+0.0203** |
| proximal, pA-437..-126 | 0.4058 | 0.4127 | -0.0068 |
| contrast | +0.0212 | -0.0059 | +0.0271 |

The distal window supplies **75%** of the class difference, and it supplies it by real 3'
ends being GC-**richer** upstream, not AT-richer. On the metagene the two classes cross at
**-344 nt**: real 3' ends are GC-richer than competitors everywhere upstream of that point
and AT-richer than them inside it, and both classes share the same sharp AT-rich trough at
the pA (canonical 0.331, competitor 0.350 at -50 nt, so the trough is deeper at real ends).
The fitted split point of -437 sits next to that crossover, which is why a two-window
contrast works at all and why the split-point sweep is flat between -250 and -1000.

**The reading the data support.** A real 3' end is a locally AT-rich site set inside
GC-richer sequence: the poly(A) signal architecture against terminal-exon and 3'UTR context.
A model false positive is a locally AT-rich site inside an intron that is AT-rich
throughout, so there is no contrast to measure. `gc_contrast` scores how far the site's
AT-richness stands out from its surroundings, not how far it extends. This is a compartment
statement, and it is consistent with the decoy contrast (`figure_supp_decoy`): on this model
competitors sit slightly **below**
matched intronic decoys (-0.056 [-0.073,-0.039]), because they were selected for local
CPA-like AT-richness inside sequence that is already AT-rich.

**The hexamer test, which the mechanism had to pass.** AATAAA or ATTAAA in [pA-60, pA),
the convention of `task_composition_test.py`. The inversion is reproduced: competitors carry
a strong hexamer more often than real 3' ends, **0.923 against 0.800**. If the gradient were
just the hexamer, it would vanish among peaks that all carry one. It does the opposite:

| stratum | genes | sites | AUROC | its null | excess | win-rate | 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all peaks | 4,847 | 48,394 | 0.6097 | 0.5155 | +0.0942 | 0.6108 | [0.5999,0.6234] |
| **hexamer present** | 3,874 | 31,395 | 0.6292 | 0.5179 | **+0.1114** | **0.6312** | [0.6186,0.6423] |
| hexamer absent | 448 | 3,349 | 0.5782 | 0.5217 | +0.0564 | 0.5532 | [0.5232,0.5898] |

Mean `gc_contrast` in the four cells: canonical with hexamer +0.0254 (n = 3,878), canonical
without +0.0046 (n = 969), competitor with hexamer -0.0053 (n = 40,186), competitor without
-0.0126 (n = 3,361). Among peaks that all carry the canonical hexamer the gradient separates
the classes better than it does overall. The hexamer and the context are different
quantities, and it is the context that tracks use.

## The pooled-AUROC null is not 0.5

Every within-gene label permutation gives each gene exactly one positive, but pooled AUROC
compares positives against negatives **across** genes. A gene with 40 sites contributes one
positive and 39 negatives, a gene with 3 contributes one and 2, so the positive class is an
unweighted average over genes while the negative class is size-weighted. Any score whose
gene-level mean correlates with peaks-per-gene therefore has a null that drifts off 0.5.
This is the same class of artifact as a P@1 chance rate that is not 1/n.

For a fixed score the ranks do not change under permutation, so the null is closed form:

    E[AUROC] = ( mean_g(mean rank in gene g) - (G+1)/2 ) / (N - G)
    E[AUROC] - 0.5 = - cov_g(n_g, mean rank in g) / ( n_bar (N - G) )

The drift is exactly the covariance between gene size and gene-level score. Here bigger
genes carry lower-scoring peaks (r between gene size and gene mean rank is -0.04 to -0.07
for every score in the figure), so every null sits **above** 0.5. Measured with 1000
permutations and by the closed form, agreeing to about 1e-3 (`task_null_bands.py`,
`null_bands.tsv`):

| score | AUROC | null band, 1000 permutations | exact null | excess |
| --- | --- | --- | --- | --- |
| acceptor + gc_contrast, holdout | 0.6486 | [0.5102,0.5443] | 0.5272 | +0.1214 |
| GC-binned (25), holdout | 0.6618 | [0.5217,0.5563] | 0.5391 | +0.1227 |
| splice-binned (150), holdout | 0.6386 | [0.5004,0.5354] | 0.5186 | +0.1201 |
| union (177), holdout | 0.6966 | [0.5121,0.5485] | 0.5296 | +0.1670 |
| `spl_upstream_max_me3` | 0.6194 | [0.5249,0.5403] | 0.5328 | +0.0866 |
| `gc_contrast` | 0.6097 | [0.5024,0.5180] | 0.5155 | +0.0942 |
| `u1_contrast` (retired) | 0.5747 | [0.5024,0.5180] | 0.5101 | +0.0646 |
| `spl_pas_prox_u1` | 0.4592 | - | 0.5358 | -0.0766 |
| `spl_upstream_mean_u1` | 0.5258 | - | 0.5661 | -0.0404 |
| anchor distance to pA | 0.4730 | [0.4844,0.5000] | 0.4920 | -0.0190 |
| baseline model `rf_prob` | 0.3400 | [0.5795,0.5923] | 0.5859 | -0.2459 |

Two numbers change meaning under the correct reference. `spl_upstream_mean_u1` at 0.5258
looked marginally predictive and is in fact **below** its null of 0.5661, which agrees with
its within-gene win-rate of 0.4293 and strengthens the suppression diagnosis. Single-window
GC at 0.4525 against a null of 0.5504 is further below chance than it appeared.

**A refit null does not rescue 0.5.** Refitting the model on permuted labels, 50 LOCO
refits, gives 0.5203 [0.5082,0.5293] against the score-fixed 0.5298, so the score-fixed null
used in the figure is the conservative of the two.

**The win-rate has no such problem.** Inside a gene the win-rate of site k is
(midrank_k - 1)/(n - 1), whose mean over a uniformly drawn k is exactly 0.5 for any score
and any tie pattern. Measured at 1000 permutations it is 0.4995 to 0.5002 for every score in
the figure. That is why it is the headline metric and why 0.500 is drawn as its reference.

**In the figure.** Panels d, i and the supplement print each quantity's own null beside its
AUROC and the excess over it; panel d plots the AUROC half as excess over each block's own
null beside the win-rate half, and panel g as excess over each chromosome's own null, which
ranges 0.5103 to 0.5640. No 0.5 AUROC reference line appears anywhere in the figure.

## The anchor acceptor: strength, not distance

`spl_dist_to_max_3ss` was dropped by selection and had never been measured on this
negative set, so it was measured directly (`task_anchor_sweep.py`, `task_anchor_answer.py`;
`anchor_distance.tsv`). The anchor is the argmax of the same scan the strength feature
takes its maximum over, so the two quantities describe one site: `anchor_dist` =
2000 - (argmax offset + 20) nt from that acceptor's AG to the pA. It reproduces the
pipeline's own `spl_dist_to_max_3ss` at r = +0.916 and picks the same anchor in 95.5% of
sites; the pipeline version searches all the way to the pA rather than to pA-126.

| quantity | AUROC | 95% CI | win-rate | 95% CI | permutation null win-rate |
| --- | --- | --- | --- | --- | --- |
| anchor acceptor score | 0.6194 | [0.6126,0.6272] | 0.6100 | [0.5986,0.6209] | 0.5002 [0.4887,0.5108] |
| anchor distance to pA | 0.4730 | [0.4647,0.4815] | 0.4789 | [0.4669,0.4907] | 0.4994 [0.4885,0.5100] |
| `spl_dist_to_max_3ss` | 0.4808 | [0.4725,0.4893] | 0.4909 | [0.4823,0.5025] | 0.4994 [0.4874,0.5108] |

The two are near-orthogonal (r = +0.007), so this is not a strength effect in disguise.
Distance carries a small competitor-direction signal: canonical sites sit **closer** to
their anchor acceptor (median 954 nt against 1,038 nt), the LOCO coefficient is negative in
all 23 folds (-0.104, range -0.109 to -0.098), and the marginal AUROC of 0.473 lies below
its own permutation band of [0.484,0.499]. It does not survive to a fitted model. On the deployed model (acceptor + `gc_contrast`,
holdout AUROC 0.6486, null 0.5272) adding the anchor distance gives 0.6499, a paired
gene-bootstrap delta of **+0.0013 [-0.0024,+0.0050]**, and the win-rate moves from 0.6450 to
0.6485. The same test on the retired U1-era model, kept because the marginal columns above
were computed alongside it:

| model, holdout | AUROC | 95% CI | win-rate | delta vs the model of the day |
| --- | --- | --- | --- | --- |
| strength alone | 0.6160 | [0.5960,0.6333] | 0.6087 | |
| distance alone | 0.5012 | [0.4779,0.5255] | 0.5218 | |
| strength + distance | 0.6169 | [0.5976,0.6352] | 0.6172 | |
| acceptor + `u1_contrast` (retired) | 0.6325 | [0.6135,0.6498] | 0.6400 | |
| + anchor distance | 0.6330 | [0.6142,0.6505] | 0.6394 | +0.0006 [-0.0029,+0.0039] |
| + `spl_dist_to_max_3ss` | 0.6330 | [0.6142,0.6502] | 0.6370 | +0.0005 [-0.0027,+0.0039] |

Distance adds a delta whose CI covers zero on both models, and adding it to strength alone
buys +0.0009. **The signal is the strength of the best upstream acceptor, not its distance**,
which is what panel a now says. Two reading notes. The pooled AUROC of the fitted
distance-alone model (0.5012) is not 1 - 0.4821 because LOCO predictions from 23 folds
carry 23 different centrings, and with a feature this weak the between-fold offsets
dominate the pooled ranking; the coefficient sign, not the pooled number, is the evidence.
And the permutation null for pooled AUROC is not exactly 0.5 under within-gene label
permutation (it is feature-specific: [0.526,0.539] for strength, [0.484,0.499] for
distance), so every AUROC above is compared to its own band and not to 0.5. The win-rate
null is 0.500 for all of them.

The earlier terminal-exon-length result (win-rate 0.470) was on a different feature set and
a different negative set; it agrees in direction and is superseded by the table above.

## Window sweep

Both windows were swept jointly, full grid, same protocol. Feature 1 is unchanged from the
earlier sweep, so its half carries over exactly (`window_sweep.tsv`); feature 2 was redone
for `gc_contrast` (`task_gc_sweep.py`, `gc_sweep.tsv`). The acceptor window is pA-W..pA-126.
The GC split point s divides pA-2000..pA-s from pA-s..pA-100, on the 100 nt bin grid; the
binned s = 400 version of the deployed -437/-126 contrast matches it at r = 0.967, so the
grid is a shape, not five-decimal truth. Cells are AUROC excess over each cell's own null,
holdout:

| acceptor W | s = 200 | s = 400 | s = 700 | s = 1000 | s = 1300 |
| --- | --- | --- | --- | --- | --- |
| 500 | +0.1004 | +0.0998 | +0.0773 | +0.0682 | +0.0582 |
| 1000 | +0.1126 | +0.1129 | +0.0958 | +0.0884 | +0.0809 |
| **2000** | +0.1250 | **+0.1255** | +0.1113 | +0.1058 | +0.1001 |
| 5000 | +0.1168 | +0.1183 | +0.1029 | +0.0955 | +0.0886 |

**The deployed cell is the best cell**, W = 2000 with the split at -437, and it was not
chosen by this grid: both windows were inherited from the pipeline's own featurisation and
then tested. The acceptor bound saturates at 2000 as before (5000 is worse once each cell is
judged against its own null, because the null itself rises with a wider window). Unlike the
retired U1 version, the split point now matters: a shallow split at 200 to 400 beats a deep
one at 1300 by 0.025 excess, which is what the -344 nt crossover predicts. Win-rate agrees
across the grid, 0.5738 to 0.6627, best at W = 2000 to 5000 with s = 200 to 400.

## Superseded selection

Forward selection was re-run under the per-site objective (dev AUROC, not P@1), same
28-feature pool, LOCO on the 80% dev genes, untouched 20% gene holdout at seed 42. The
holdout plateau is 0.6403 at k=6; every k from 3 up has a holdout CI covering it and every
k from 3 to 12 has all coefficient CIs excluding zero, so the rule gives **k = 3**:

| feature | standardized beta | 95% CI | sign flips |
| --- | --- | --- | --- |
| `spl_upstream_max_me3` | +0.4290 | [+0.3914,+0.4668] | 0.000 |
| `spl_pas_prox_u1` | -0.3236 | [-0.3618,-0.2762] | 0.000 |
| `spl_upstream_mean_u1` | +0.2730 | [+0.1944,+0.3334] | 0.000 |

`spl_u1_suppression_up` is out and `spl_upstream_mean_u1` is in; the acceptor coefficient
is unchanged from the earlier fit (+0.429 vs +0.427).

The selection that produced it is reported for completeness. Its U1 pair was replaced by a
single `u1_contrast`, and that contrast was then retired in favour of `gc_contrast` once the
control table showed it was a proxy for the GC gradient. Everything in this section is
history: no U1 term is in the model.

**The U1 pair was a suppression pair.** The two terms
are nested by construction: `spl_pas_prox_u1` is bin 6 of the six bins averaged in
`spl_upstream_mean_u1`, and they correlate at r = +0.610. `spl_upstream_mean_u1` has a
pooled univariate standardized coefficient of only +0.062 (marginal AUROC 0.526, below its
own permutation null of 0.5661) that
inflates 4.4-fold to +0.273 once its own sub-window is in the model. Its within-gene
marginal runs the other way (win-rate 0.429): inside a gene the canonical site has *lower*
upstream U1 than its own competitors. Both terms carry information — dropping
`spl_upstream_mean_u1` costs 0.0257 AUROC, dropping `spl_pas_prox_u1` costs 0.0134,
dropping the acceptor costs 0.0780 — but +0.273 cannot be read as "canonical sites have
more upstream U1".

**The U1 metagene, now `figure_supp_u1` panel a.** Gene-averaged, canonical lies below competitor at every
position from -2000 to +500. Both classes show the same sharp U1 minimum at the pA
(canonical 40.0/kb, competitor 48.4/kb at -12 nt; relative depth 56% and 54%), so the dip
is a property of poly(A) sites in general and does not separate used from unused ones. The
proximal-versus-distal split reconciles the earlier hit-density result: at threshold 5.25
over +/-500 nt canonical is U1-poorer (22.4 vs 26.1 per kb), while the 2 kb upstream mean
is higher at canonical sites pooled across genes (0.710 vs 0.645 for bins 1-5). It is a
conditional-versus-marginal and proximal-versus-distal difference at the same time.

## Figure caption

**One splicing feature and one composition feature separate real human 3' ends from
baseline-model false positives.** Eight panels, eleven plots.

**a**, Where the two features are measured, all on one axis relative to the pA. Top, the
gene model with the pA marked; middle, green, the acceptor search window pA-2000..pA-126
with the anchor 3'ss drawn at the median canonical anchor distance of 954 nt, feature 1
being `spl_upstream_max_me3`, the maximum MaxEntScan 3'ss score in that window and not the
anchor's position; bottom, blue and pink, the two composition windows, feature 2 being
`gc_contrast` = mean G+C over pA-2000..-438 minus mean G+C over pA-437..-126.
**b**, The three filters on the 64,140 competitors of the contested-gene shortlist, leaving
43,547 competitors and 4,847 canonical sites in 4,847 genes. Removed competitors are real
alternative 3' ends rather than model errors: among those with a matched PolyA_DB site
within 500 nt they have median RPM 0.43 and PSE 8.24% (n = 10,093), against 3.29 and 44.78%
for canonical sites under the same rule (n = 4,599).
**c**, ROC (left) and precision-recall (right) for the two-feature model, LOCO across all
48,394 analysis sites: win-rate 0.655, null 0.500; AUROC 0.658, null 0.533, excess +0.125;
AUPRC 0.169 against a base rate of 0.101 (dashed). The diagonal is the ROC reference. The
baseline model is not drawn: `rf_prob` scores AUROC 0.340 against a null of 0.586 here
because its own false positives define the competitor set.
**d**, Left, what each feature contributes on the untouched 20% gene holdout, 974 genes and
9,624 sites, as within-gene win-rate (null 0.500) and as AUROC excess over each model's own
permutation null (0.527 for both). The paired increment of the composition feature over the
acceptor alone is printed under each half: **+0.036 [+0.012,+0.060] in win-rate, two-sided
bootstrap p = 7.0e-3**, and **+0.033 [+0.020,+0.044] in AUROC excess, p < 5e-4**, from 2,000
gene-clustered replicates in which both models are re-evaluated on the same resampled genes
so the difference is taken within a replicate. Right, the standardized coefficients of the
deployed model, ridge 1.0, 95% CI over a gene-clustered bootstrap; the sign-flip fraction is
0.00 for both features. The uninterpretable ceilings are no longer drawn; their numbers are
below.
**e**, Mean GC per 100 nt bin by class with a gene-clustered 95% CI, both `gc_contrast`
windows shaded, the class crossover marked and the per-class n printed on each line. Every
100 nt bin has the same n, 4,847 canonical and 43,547 competitor, since no peak loses a bin
to a window edge. Real 3' ends are GC-richer than competitors in
the distal window, 0.427 against 0.407, and AT-richer within 344 nt of the pA; both classes
dip at the pA and the dip is deeper at real ends, 0.331 against 0.350 at -50 nt. The acceptor
bound and the GC split point were chosen from a grid of 4 upstream bounds x 5 split points;
the deployed combination was the best cell, and the -437 split falls beside the -344 nt
crossover the panel marks.
**f**, The distributions behind both features, same styling so they read side by side. Left,
per-peak values by class as mirrored histograms, 60 shared bins over the central 99% of each
range, canonical above and competitor below, class medians dotted. `gc_contrast`: AUROC
0.610, medians +0.021 and -0.005. `spl_upstream_max_me3`: AUROC 0.619, medians +10.350 and
+9.721. The classes overlap heavily, which is what an AUROC near 0.61 looks like at the level
of single peaks. Right, the paired form, which is the statistic the paper actually reports:
for each of the 4,847 analysis genes, the canonical peak's value minus the mean of its own
competitors', sorted. `gc_contrast` **0.635 of genes above zero** [0.622,0.648], median
difference +0.029; `spl_upstream_max_me3` **0.623 above zero** [0.609,0.636], median +0.577.
Both p < 1e-4 by within-gene label permutation, 10,000 shuffles, no shuffle reaching the
observed fraction; the permutation null of the fraction is 0.500 and 0.495. The win-rate on
these quantities is each feature's own, 0.6108 and 0.6100; the headline 0.6450 is the
two-feature model's on the holdout and is in panel d.

**g**, Per-chromosome AUROC excess over each chromosome's own permutation null, which ranges
0.510 to 0.564; gene-clustered bootstrap, chromosomes with fewer than 20 genes omitted. The
solid vertical line is the **pooled excess of the same fit, +0.125**, so the spread reads
against the genome-wide value and not only against zero. That line is the LOCO-at-full-n
number because that is the protocol the per-chromosome points come from; the holdout headline
is +0.121, and the two are compared under "Which number is the headline".
**h**, The model's held-out linear predictor against GC read at a single 500 nt window around
the pA, with the within-class correlations; single-window GC explains 9.4% of the predictor's
variance and is itself below chance as a classifier (AUROC 0.4525 against a null of 0.5504).
Colour means class in **e**, **f** and **h**, and model in **d**; the shared legend covers
both. **P values appear only where a p value is the claim**: on the paired differences in
**f** and on the increment in **d**. Panels c, e, g and h carry no p values, because a curve,
a profile, a per-chromosome spread and a scatter are descriptions rather than tests. Throughout: gene split, leave-one-chromosome-out, ridge logistic 1.0
standardized in fold, seed-42 20% gene holdout, gene-clustered bootstrap, and nulls from 1000
within-gene label permutations.

**Which number is the headline.** Two protocols appear in the figure and they differ by
about 0.01. Panels **c** and **g** are LOCO over **all** 48,394 sites: every peak is scored
by a fit that never saw its chromosome, but the gene holdout is not applied, so AUROC 0.6576
and win-rate 0.6548. Panel **d** uses the **untouched 20% gene holdout**: fits on the 80%
development genes only, still leave-one-chromosome-out, scored on 974 genes and 9,624 sites
that were never used for selection or for any other choice, giving AUROC 0.6486 and win-rate
0.6450. The stricter holdout figures are the headline and the ones quoted in the text:
**win-rate 0.645, AUROC 0.649, excess +0.121 over a null of 0.527**. Panel c is the curve
shape at full n; its scalars are reported in this caption and nowhere else.

**Panel d is a partition, not a bake-off.** An earlier version of this figure claimed that
removing GC cost nothing. That claim is dead: the model contains the composition gradient, so
the question is not whether GC is in it but how much each part carries. Splicing alone
reaches excess +0.089 and win-rate 0.6087; adding one composition number takes it to +0.121
and 0.6450, the increment tested above.

**The uninterpretable ceilings are not drawn.** On the same holdout: GC-binned (25 features)
reaches AUROC 0.6618, excess +0.1227, win-rate 0.6542; splice-binned (150) reaches 0.6386,
excess +0.1201, win-rate 0.6590; all blocks together (25 + 150 + the two named features =
177) reach 0.6966, excess +0.1670, win-rate 0.7115. Two readings follow and both belong in
the text rather than on the panel. The two interpretable features recover the whole of what
the 25-feature positional composition profile finds, +0.121 against +0.123, within noise.
Binning splice features adds nothing over the one named acceptor term. And 175
uninterpretable features add a further +0.046 on top of the two named ones, which is the
honest ceiling on this task. `gc_blocks.tsv`.

**Panel h is a partition, not a denial.** When feature 2 was `u1_contrast` this scatter was
evidence that the model was not reading composition, and it was moved to the supplement when
that claim collapsed. Feature 2 is now a GC contrast, so the question is different: how much
of the predictor is composition read at a **single** window, given that single-window GC is
itself below chance? It correlates with the predictor at r = -0.306 overall, -0.182 within
canonical and -0.322 within competitor, so 9.4% of the variance, up from 1.1% under the U1
version. The sign is negative, because real 3' ends are locally AT-richer (mean 500 nt GC
0.391 against 0.399 for competitors), and `gc_contrast` itself correlates with single-window
GC at -0.524. A single window and a two-window contrast are not the same measurement, which
is why one is below chance and the other carries +0.094 excess.

**Plot types.** Of the two options for reducing three dot-and-CI panels, the coefficients
were folded in beside the contribution ladder as the second half of panel **d**, since both
answer "what does each part contribute". The per-chromosome panel was **kept as an ordered
dot plot** rather than being turned into a histogram: with n = 23 a histogram has about five
bins, it hides the per-chromosome confidence intervals, and the ordered plot shows that the
weakest chromosome (chr19, +0.058) is still positive, which a histogram would only imply.
Panel **f** then adds the two distributional views that no interval plot can show.

## P@1 and MRR: descriptive only

Reported because they are asked for, not because they carry a claim. Holdout protocol, the
two-feature model, 974 genes and 9,624 peaks, ties credited fairly (1/k at the top for P@1,
the reciprocal rank averaged over the tie block for MRR), never by argmax.

| metric | observed | chance, analytic | chance, 10,000 permutations | ratio |
| --- | --- | --- | --- | --- |
| P@1 | **0.4230** | 0.2917 = mean 1/n_i | 0.2919 [0.2659,0.3183] | 1.45x |
| MRR | **0.6188** | 0.5201 = mean H_{n_i}/n_i | 0.5203 [0.5040,0.5364] | 1.19x |

The analytic and permutation chance rates agree to three decimals, which is the check that
the tie handling and the chance definition describe the same experiment.

**Why they are descriptive.** After the downstream-of-canonical filter the canonical site is
by construction the last candidate in every gene, so the most-downstream rule scores P@1 =
1.0000 over all 4,847 analysis genes. Any within-gene top-1 metric therefore answers a
question the candidate definition has already settled, and a model can only lose to a rule
that is correct by fiat. That is why the task is per-site classification, why the reported
within-gene metric is the win-rate, which is not degenerate under this candidate definition,
and why no P@1 or MRR appears in the figure. `task_p1_mrr.py`, `p1_mrr.tsv`.

## Cut panels, numbers retained

Three panels were cut for reading badly at 7 pt, not for their content. Their numbers stand.

**PolyA_DB usage of removed competitors** (was panel c). Under a single measurement rule, a
matched PolyA_DB site within 500 nt: canonical n = 4,599, median RPM 3.29, PSE 44.78%;
removed APA competitors n = 10,093, RPM 0.43, PSE 8.24%. Retained competitors cannot enter
that comparison at all, because having no PolyASite or PolyA_DB site within 500 nt is the
definition of the retained group (n = 43,547, median distance to the nearest PolyA_DB site
16.3 kb). The one sentence that survives is in the panel b caption; the filter justification
was always panel b's job. `up_panel_c_matched.tsv`.

`polyadb_peak_cache.npz` stores the nearest same-strand site at **any** distance with that
site's AvgRPM.3READS and PSE.3READS, which is why an earlier version of this panel was not
like-for-like: the fraction whose value came from a site inside 500 nt was 0.949 for
canonical (median distance 3 nt), 0.557 for removed APA (median 311 nt) and 0.000 for
retained (median 16,317 nt), so the retained bar was the usage of an unrelated site 16 kb
away plotted beside two groups measured at the peak.

**The hexamer stratification** (was panel h). Hexamer = AATAAA or ATTAAA in [pA-60, pA).
Competitors carry one more often than real 3' ends, **0.923 against 0.800**. `gc_contrast`
separates the classes *within* each stratum, and does so better among the peaks that carry a
hexamer than overall:

| stratum | genes | sites | AUROC | its null | excess | win-rate | 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all peaks | 4,847 | 48,394 | 0.6097 | 0.5155 | +0.0942 | 0.6108 | [0.5999,0.6234] |
| hexamer present | 3,874 | 31,395 | 0.6292 | 0.5179 | **+0.1114** | **0.6312** | [0.6186,0.6423] |
| hexamer absent | 448 | 3,349 | 0.5782 | 0.5217 | +0.0564 | 0.5532 | [0.5232,0.5898] |

Cell means: canonical with hexamer +0.0254 (n = 3,878), canonical without +0.0046 (n = 969),
competitor with hexamer -0.0053 (n = 40,186), competitor without -0.0126 (n = 3,361). Four
points on one axis did not carry that; the table does. `gc_hexamer.tsv`.

**The intronic-decoy contrast** (was panel j) is now `figure_supp_decoy`, where the reading
is written out. Canonical - competitor +0.359 [+0.342,+0.379]; competitor - intronic decoy
**-0.056 [-0.073,-0.039]** for the two-feature model against **+0.132 [+0.114,+0.148]** for
GC-binned (25), whose canonical - competitor gap is +0.332 [+0.314,+0.351]. A measure of
CPA-likeness orders canonical > competitor > intronic background, because the competitors are
exactly the positions a CPA model already picked out of that background; GC-binned does that
and the two-feature model does not. The model instead places competitors slightly **below**
unselected intronic sequence, which is the expected direction once feature 2 is the GC
gradient: a competitor is a locally AT-rich site inside an intron that is AT-rich throughout,
so its distal-minus-proximal contrast is lower than a random intronic position's. The panel
uses a 100 nt bin approximation of `gc_contrast` (distal = bins over pA-2000..-400, proximal
= bins over pA-400..-100), because the decoy scan stores 100 nt bins rather than the
pipeline's 312.5 nt grid; it correlates with the exact feature at r = 0.967 and is applied
identically to peaks and decoys. `gc_decoy.tsv`.

## The protection-window sweep: supplementary, and why

The Almada et al. 2013 rule was tested directly and then controlled. It **does not survive
its controls**, so it is in `figure_supp_almada` and is described as consistent with, but
not demonstrating, telescripting. Nothing from it enters the model or the main figure.

**What the raw sweep showed.** Defining protection as a strong scan-hit donor within W nt,
the upstream win-rate runs 0.4900, 0.4826, 0.4664, 0.4605, 0.4718 for W = 100, 200, 500,
1000, 2000 at cutoff 8.46 against a permutation null of 0.500, with downstream never
departing by more than 0.012; the 8.77 cutoff gives the same pattern. At W = 500, 25.7% of
competitors carry a strong upstream donor against 21.0% of real 3' ends.

**Control 1, annotated donors.** Only **9.1%** of the scan's strong-donor hits sit on a
GENCODE-annotated 5'ss (identical within 0, 2 or 10 nt), and the median distance to a real
donor is 9,186 nt against 1,169 nt for scan hits: the "U1 sites" are overwhelmingly
cryptic. Re-run on annotated 5'ss the asymmetry **inverts and grows downstream** (upstream
0.4959 to 0.4964 across W, downstream 0.4965 falling to 0.3530 at W = 2000). That
downstream pattern is compartment stated plainly: a competitor lies inside an intron and
has annotated donors on both sides, while a canonical site in a terminal exon has no
downstream intron.

**Control 2, composition-matched motif.** The hand-built donor PWM with its nine columns
permuted (seed 0), 9-mers carrying GT at the invariant positions excluded, threshold
calibrated to the same hit density, **reproduces the upstream deficit**: 0.4799 versus
0.4664 at W = 500 and 0.4703 versus 0.4605 at W = 1000. A motif that cannot bind U1 does
the same work.

**Control 3, GC.** The two GC controls disagree and the stratified one is the trustworthy
one. Residualising the binary protection indicator on upstream GC flips the win-rate from
0.4664 to 0.5484, but canonical sites are systematically AT-richer upstream, so that
regression removes class signal along with composition and over-corrects. Restricting to
canonical-competitor pairs in the same upstream-GC quintile, the competitor is the
protected one in 0.5959 of 4,449 discordant pairs at W = 500 and 0.5597 of 6,210 at
W = 1000, so the deficit does survive window-GC stratification. It does not survive
control 2, and local motif density is the relevant composition here, not window GC.

**Compartment component, measured.** Of peaks protected at W = 500, 99.7% of competitors
lie in an annotated intron against 20.6% of canonical peaks, while the protecting donor is
the host intron's own 5'ss in only 5.6% and 2.7% of cases respectively.

**Conclusion.** The sweep recovers a real, reproducible asymmetry, but the asymmetry is
carried by cryptic GT-rich sequence rather than by U1 sites, and it reverses direction when
real 5'ss are used. It is reported as a negative control on the mechanism, not as evidence
for it.

## Anticipated objections

1. **The negatives are model artifacts, not biology.** Panel b. 18,119 competitors
   carrying an APA annotation were removed, and among those with a matched PolyA_DB site
   within 500 nt the removed population is genuinely used: median RPM 0.43 and PSE 8.24%
   (n = 10,093), against 3.29 and 44.78% for canonical sites (n = 4,599) under the same
   rule. Retained competitors have no site within 500 nt and so cannot be placed on that
   axis at all; see "Cut panels, numbers retained".
2. **It just detects introns.** `figure_supp_decoy`. With compartment fixed by construction, one
   competitor against five intronic decoys from the same gene matched on distance to the
   nearest annotated 3'ss and 5'ss, the two features separate canonical from competitor by
   +0.359 [+0.342,+0.379] and place competitors slightly below random intronic sequence,
   -0.056 [-0.073,-0.039]. Reading compartment **is** the mechanism, and the paper says so:
   a terminal exon is an upstream acceptor plus GC-richer exonic context around an AT-rich
   poly(A) signal, which is exactly what the two terms measure. What the model does not do
   is order canonical > competitor > background, the signature of scoring CPA-likeness
   itself; GC-binned (25) does that (+0.332 and +0.132), which is why it is shown beside it.
3. **It is composition, not splicing.** Half of it is composition, and panel a says so. The
   partition in panel d is the answer: splicing alone reaches excess +0.089,
   splicing plus one composition number +0.121, the 25-feature composition block +0.123, the
   union +0.167. Neither feature is redundant with the other: dropping `gc_contrast` costs
   0.036 win-rate, dropping the acceptor costs more. The claim is not that composition is
   absent, it is that two interpretable numbers, one of each kind, recover what an
   uninterpretable positional composition profile finds.
4. **The acceptor term is circular with annotation.** 40.4% of canonical anchors fall
   within 2 nt of a GENCODE acceptor and 59.6% do not, with a median distance of 264 nt;
   the competitor figure is 9.0%. The feature is a MaxEnt scan of sequence, not a lookup,
   and most canonical anchors are not at an annotated site.
5. **A trivial positional rule does better.** It does — perfectly, and by construction:
   after filter 3 the canonical is the last candidate in every gene, so the rule scores
   1.000. That is why within-gene ranking was abandoned in favour of per-site
   classification, and why no P@1 appears anywhere in this figure.
6. **Feature selection is overfit.** The full selection path with dev and holdout AUROC at
   every k is in `upstream_path.tsv`; selection ran on the 80% dev genes under LOCO and the
   20% holdout was never used for any choice. The chosen k is the smallest attaining the
   holdout plateau with every coefficient sign stable.
7. **The chance rate is wrong.** It was. Pooled AUROC has no 0.5 null under within-gene
   label permutation, and an earlier draft of this figure drew chance lines at 0.500. Every
   AUROC in the figure is now stated against its own permutation null, measured at 1000
   permutations and matched by a closed form, and no 0.5 AUROC reference line remains. The
   headline metric is the win-rate, whose null is exactly 0.5 by construction and 0.4999
   measured. See "The pooled-AUROC null is not 0.5".
8. **Tie-breaking leaks.** Ranking, where it was still used, credited a tie as 1/k rather
   than taking an argmax; a constant score returns exactly the analytic chance rate, and
   permuting candidate order inside every gene leaves every number unchanged (checked:
   0.410522 -> 0.410522).
9. **The bootstrap unit is wrong.** Genes, never peaks, everywhere: 500 replicates for the
   block and per-chromosome panels, 300 for the decoy contrast, 200 for the curves.
10. **k was cherry-picked.** See objection 6 and `upstream_path.tsv`; the path is reported
    with CIs at every k from 1 to 12 and the rule is stated in advance.
11. **The effect size is small.** It is, and the correct reference makes it smaller than an
    earlier draft claimed. Holdout win-rate 0.645 against a null of 0.500, AUROC 0.649
    against a null of 0.527 (excess +0.121), AUPRC 0.169 against a base rate of 0.101. This
    is a weak per-site signal that is positive on all 23 held-out chromosomes after each is
    compared to its own null, not a classifier anyone should deploy.
12. **Why not SpliceAI.** SpliceAI's 10 kb context encodes the presence of a downstream
    donor directly, so the acceptor-by-donor interaction appears only within SpliceAI and
    not in MaxEnt; the comparison is in the Figure 3 analysis and is not re-derived here.
13. **The mechanism is asserted, not tested.** It is tested and it fails its controls.
    The Almada protection rule reproduces on this set, but a composition-matched non-donor
    motif reproduces it equally well, only 9.1% of the scan's "U1 sites" are annotated
    5'ss, and on annotated 5'ss the asymmetry inverts. See `figure_supp_almada`. The paper
    claims a per-site sequence signal, not a demonstrated telescripting mechanism.
14. **The acceptor term is really a distance to the upstream exon.** It is not. Strength
    and distance to the same anchor are near-orthogonal here (r = +0.007), distance alone
    has a holdout AUROC of 0.501, and adding it to the model buys +0.0006 [-0.0029,+0.0039].
    Distance does carry a weak competitor-direction signal (canonical anchors are closer,
    median 954 against 1,038 nt, negative coefficient in all 23 LOCO folds), and it is
    reported rather than hidden. See "The anchor acceptor: strength, not distance".
15. **The windows were tuned, and -437 is arbitrary.** They were inherited from the
    pipeline's featurisation and then swept, not chosen. On the deployed model a 4 x 5 grid
    over the acceptor bound and the GC split point puts the deployed cell at the top of the
    grid (excess +0.1255), which is a coincidence of the inheritance and not a selection: no
    cell was ever chosen on the holdout. Panel e marks the -344 nt crossover the split sits
    beside; the grid itself is in the caption. See "Window sweep".
16. **`gc_contrast` is just the poly(A) hexamer.** It is not; see the hexamer table under
    "Cut panels, numbers retained".
17. **The effect is a difference of medians, so it cannot separate anything.** Panel f says
    both halves of that out loud, for both features. Per peak the classes overlap heavily
    (medians +0.021 against -0.005 for `gc_contrast`, +10.350 against +9.721 for the
    acceptor), which is what an AUROC near 0.61 looks like. Per gene, in the paired form the
    paper reports, 0.635 and 0.623 of the 4,847 genes have their canonical peak above the
    mean of their own competitors, both p < 1e-4 over 10,000 within-gene label permutations
    with no shuffle reaching the observed fraction.
18. **No p values anywhere.** There are two, and only where a p value is the claim: the
    paired differences in panel f and the increment of the composition feature over the
    acceptor alone in panel d (+0.036 win-rate, p = 7.0e-3; +0.033 AUROC excess, p < 5e-4).
    Curves, profiles, per-chromosome spreads and scatters are descriptions and carry none.
    Competitors carry AATAAA or ATTAAA more often than real 3' ends (0.923 against 0.800),
    and among peaks that all carry one the gradient separates the classes better than it
    does overall (win-rate 0.6312 [0.6186,0.6423], excess +0.1114 against +0.0942 for all
    peaks). The hexamer and the context around it are different quantities.
17. **Feature 2 is a U1 result relabelled.** The opposite: the U1 version was retired
    because it was a composition result mislabelled. A permuted U1 table sits at its null, a
    U1 table read on the reverse complement reproduces the effect, GC over the same windows
    beats it, and `u1_contrast` residualised on the GC gradient has excess +0.009. See
    `figure_supp_u1` and the model section.

## Provenance

`task_upstream_blocks.py` (filter 3, counts, per-site blocks) ->
`task_upstream_select.py` (per-site forward selection) -> `task_u1_contrast.py` and
`task_u1_single.py` (the U1 lineage, now retired) -> `task_control_contrast.py` (the control
table that retired it; `control_contrast_scan.npz`, `control_contrast.tsv`) ->
`task_gc_model.py` (the deployed model, partition, decoy transfer, hexamer test; `gc_*.tsv`)
-> `make_figure_twofeature.R`.

Supporting: `task_null_bands.py` -> `null_bands.tsv`, `null_bychrom.tsv` (the permutation
nulls behind every reference line). `task_anchor_sweep.py` -> `task_anchor_answer.py` ->
`anchor_distance.tsv`, `window_sweep.tsv` (strength versus distance, acceptor bound).
`task_gc_sweep.py` -> `gc_sweep.tsv` (the sweep on the deployed model).
`task_gc_windows.py` -> `gc_windows.tsv` (the window decomposition and the crossover).
`task_dist_panels.py` -> `gcc_hist.tsv`, `gcc_paired.tsv`, `me3_hist.tsv`, `me3_paired.tsv`,
`gc_profile_n.tsv` (panel f for both features with its permutation p values, and the panel e
sample sizes). `task_d_increment.py` -> `d_increment.tsv` (the paired increment in panel d).
`task_panel_c_gc.py` -> `up_panel_c_matched.tsv` (the PolyA_DB comparison under a common
measurement rule) and `gc_scatter.tsv` (panel g).
`make_figure_supp_u1.R` -> `figure_supp_u1` (why feature 2 is not U1).
`make_figure_supp_decoy.R` -> `figure_supp_decoy` (the intronic-decoy contrast, cut from the
main figure).
`task_almada.py` and `task_almada_controls.py` -> `make_figure_supp_almada.R` (the
protection-window sweep and its controls). The decoy positions come from
`task_decoy_build.py` and `task_decoy_test.py`, unchanged.

Shareable table: `task_build_final_parquet.py` -> `cpa_peaks_final.parquet` with
`cpa_peaks_final.README.md`, all 70,723 peaks flagged rather than filtered, carrying both
features and the held-out discriminant. It supersedes `rerank_corr_peaks.parquet`, now
renamed `rerank_corr_peaks.SUPERSEDED.parquet`.

Caches: `binned_features178.npz` and `decoy_features.npz` (100 nt binned scan),
`control_contrast_scan.npz` (the two GC windows, exact), `anchor_sweep.npz` (acceptor scan
to pA-5000), `hexamer_cache.npz` (AATAAA/ATTAAA in [pA-60, pA)).
