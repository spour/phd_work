# figure_rerank_primary — methods and panel notes

Everything that used to sit as grey text under a panel title, plus the old caption block,
lives here. The figure itself carries titles, axes, data and effect-size annotations only.

## Model and data

Primary analysis: competitors whose 500 nt window overlaps no PolyASite cluster and no
PolyA_DB site (peak more than 500 nt from either), **46,021 competitor peaks over 5,008
genes**; the canonical set is unchanged, one used 3' end per gene. Genes are kept only if
one canonical and at least one competitor survive the restriction, so the training and the
ranking populations are the same set.

Model: logistic regression on three standardized features, ridge 1.0, features standardized
inside each training fit, IRLS implemented inline (no sklearn, no scipy).

```
spl_upstream_max_me3    strongest upstream 3' ss (MaxEnt)      +0.421 [+0.388, +0.457]
spl_pas_prox_u1         U1 signal near the polyA signal        -0.123 [-0.187, -0.065]
spl_u1_suppression_up   upstream U1 suppression                -0.163 [-0.211, -0.126]
```

Evaluation: leave-one-chromosome-out (LOCO) throughout, plus an untouched 20% gene holdout
drawn once at seed 42. Confidence intervals are bootstraps over genes.

**Headline (untouched 20% of genes): P@1 0.414 [0.383, 0.444] against chance 0.289, excess
+0.125 [+0.098, +0.149].** Chance is the random floor, mean 1 / candidates per gene; it
moves with contest size, so excess (or lift) is the quantity that compares across strata.

Colour vocabulary in the figure: blue = reranker, orange = RF baseline, grey = chance (the
permutation null is chance), black = a fitted coefficient.

## Panels

**a — ROC, reranker vs baseline.** LOCO over all genes in the primary set. The RF baseline
sits below chance by construction: a competitor reaches the shortlist only if it already
outscored the true end under the RF, so the RF cannot rank the used site first. Its curve is
shown to make that degeneracy explicit, not as a competitive comparison.

**b — Precision-recall, base rate 9.8%.** One canonical per gene, so the base rate is the
canonical share of candidate rows (0.098). The dashed grey line is that base rate, which is
the chance level for precision.

**c — P@1 by held-out chromosome.** Each row is the model fit on every other chromosome and
evaluated on the genes of that one. Diamonds are that chromosome's own chance rate, which
varies with how many candidates its genes carry. Pooled LOCO P@1 is 0.408. The headline
number quoted above is the gene-holdout protocol, not this one; the two differ in which
genes are held out, not in the model.

**d — P@1 by number of competitors.** Numbers above the bars are lift over that bucket's
chance; numbers below are genes in the bucket. Lift peaks in the 3-4 competitor bucket
(+0.16) and falls to +0.07 at 9 or more competitors: discrimination declines as the contest
grows, and raw P@1 falls faster than chance does.

**e — Permutation null.** The canonical label is shuffled within each gene and the whole
LOCO fit repeated, 200 times, so the null carries the same protocol and the same candidate
structure. Observed 0.408 against a null centred on 0.284 (the analytic chance rate),
p = 0.005 (the floor at 200 permutations).

**g — Rank of the used 3' end.** Rank of the canonical peak among its own gene's candidates.
Categories are mutually exclusive with "last" taking precedence, so a rank-2 of 2 counts as
last. Numbers above the bars are observed over chance.

| rank | genes | observed | chance | ratio |
|------|-------|----------|--------|-------|
| 1    | 2,041 | 40.8%    | 28.4%  | 1.44  |
| 2    | 631   | 12.6%    | 11.9%  | 1.06  |
| 3    | 356   | 7.1%     | 6.9%   | 1.04  |
| 4+   | 1,053 | 21.0%    | 24.4%  | 0.86  |
| last | 927   | 18.5%    | 28.4%  | 0.65  |

Rank ≤ 3 is 75.6% observed against 68.7% chance; median rank 2 of a median 4 candidates. The
signal is concentrated at the top and bottom of the ranking: rank 1 is 1.44x chance and last
is 0.65x. Conditional on a miss (2,967 genes) the picture is close to chance — 58.9% of
misses still rank the used end in the top 3, but 31.2% rank it last. This panel replaced an
earlier normalised-score-gap histogram, which piled ~800 genes at exactly 1.0 because a used
end ranked last has a normalised gap of 1 by construction.

**h — Standardized coefficients.** Fit on the dev genes (the 80% outside the holdout), 95%
CI from a bootstrap over genes; "flip" is the fraction of bootstrap resamples in which the
coefficient changed sign. All three are signed and CI-clean with zero or near-zero flips.
A positive coefficient points to the used end, a negative one to the competitor.

**i — Coefficients with compartment held fixed.** The same three coefficients refit on the
TERM-TERM stratum, where the canonical peak and at least one competitor are both in the
terminal exon (1,969 genes, all competitors). Compartment is constant inside every candidate
set there, so nothing about which compartment a peak occupies can contribute.

With compartment fixed the acceptor roughly halves but keeps its sign and its clean CI
(+0.421 → +0.221, zero flips) while both density terms hold (spl_u1_suppression_up
−0.163 → −0.145; spl_pas_prox_u1 −0.123 → −0.119). The acceptor therefore carries part
compartment; the density terms are the compartment-independent component. The model still
beats its own chance rate there, excess +0.083 [+0.039, +0.127].

Raw versus excess, because this is easy to misread: the TERM-TERM raw P@1 (0.511) is *higher*
than the primary (0.414) only because chance there is 0.428 against 0.289. The excess is
*lower* (+0.083 [+0.039, +0.127] against +0.125 [+0.097, +0.152]), and excess is the
comparable quantity. Restricting that stratum further to unannotated competitors leaves 271
genes, where the density terms roughly double and overtake the acceptor but the excess CI
spans zero (+0.104 [−0.020, +0.236]): a direction, not an estimable magnitude.

**j — P@1 vs model size.** Forward selection path on the untouched 20%, with a
bootstrap-over-genes ribbon; the dev LOCO path agrees throughout and is in the supplement.
Every k from 3 up has a holdout CI covering the k = 12 value, so nine extra features buy
+0.022 with overlapping intervals. The step at k = 7 (0.430 against 0.414) sits inside those
overlapping intervals and its added features include two with unstable signs
(spl_mean_top3_5ss 47.3% sign flips, spl_rel_3ss_dn 32.3%), so k = 3 stands: it is the
smallest set attaining the plateau and the only one in which every coefficient has a
statable sign.

## Feature definitions

Read from the generating code, not from the names. The chain is
`scripts/featurize_shortlist.py` (`_work`) -> `scripts/splice_common.py`
(`compute_splice_row`) -> `scripts/extract_splicing_features.py` (`extract_one_site`,
`scan_me3`, `scan_u1`, `aggregate_per_bin`) and `scripts/compute_gene_baselines.py`
(`derive_axes`), with the extra features in `scripts/splice_extra_features.py`.

Common frame: the sequence window is fetched in **transcript orientation** and spans
**pA-1000 .. pA+500** (`--upstream_nt 1000 --downstream_nt 500`, the defaults used by
`scripts/slurm_featurize_shortlist.sh`), so the pA sits at index 1000 of a 1,500 nt window.
Bin features use `--n_bins 8`, i.e. eight equal 187.5 nt bins over the whole 1,500 nt, so
the bin grid is not aligned to the pA. Every scan scores a k-mer at its **start position**.

| | spl_upstream_max_me3 | spl_pas_prox_u1 | spl_u1_suppression_up |
|---|---|---|---|
| window (k-mer starts, vs pA) | -1000 .. +125 (bins 1-6 of 8) | -62.5 .. +125 (bin 6 of 8) | -500 .. 0 (strictly upstream) |
| orientation | transcript | transcript | transcript |
| scorer | MaxEntScan `score3`, 23-mer | U1 duplex lookup, 9-mer (4^9 table) | hand-built donor PWM, 9-mer |
| register | AG at 23-mer positions 19-20 (1-based), i.e. intron[-20..-1] + exon[+1..+3] | donor 9-mer exon[-3..-1] + intron[+1..+6]; strongest entry CAGGTGAGT / CAGGTAAGT at dG -11.7 / -11.6 | same donor register, GT required at 0-based 9-mer positions 3-4 |
| reduction | max over valid AG 23-mers (per-bin max, then max of bins 1-6) | arithmetic mean over every valid 9-mer starting in the bin | count of 9-mers scoring >= threshold, expressed as density |
| threshold | none; 23-mers without AG are NaN and excluded | none; all 9-mers enter the mean | `>= 0.6` on the PWM mean, a hard-coded default in `u1_suppression_up(..., thresh=0.6)`; no calibration is recorded in the code |
| units | MaxEnt bits (log-odds) | -dG in kcal/mol (sign flipped by `scan_u1`, so higher = stronger U1 binding) | sites per 100 nt |
| observed range (contested set) | -0.004 to 15.917, median 9.825, 5-95% 7.505 to 12.324 | -2.440 to 4.763, median 0.616, 5-95% -0.264 to 1.718 | 0 to 1.426, median 0, 72.8% exactly zero, q90 0.204 |
| missing / imputed | 0 of 70,723 (0.000) | 0 of 70,723 (0.000) | 0 of 70,723 (0.000) |
| computed by | `derive_axes` -> `upstream_max_me3` = `max(bin1..bin6 me3_max)`, bins from `extract_one_site` / `scan_me3` | `derive_axes` -> `pas_prox_u1` = `feats["bin6_u1_mean"]`, from `extract_one_site` / `scan_u1` | `splice_extra_features.u1_suppression_up` -> `_donor_density(seq, 1000-500, 1000, 0.6)` |

Three things the names hide:

**"upstream" in `spl_upstream_max_me3` is only approximately upstream.** Bins 1-6 of an
eight-bin grid over a 1,500 nt window end at window coordinate 1,125, which is pA+125. Since
the score is indexed by the 23-mer start and the AG sits 18-19 nt into it, the acceptors
actually covered run from pA-982 to pA+143. The feature is the strongest MaxEnt acceptor
anywhere in that range, not in a window that stops at the pA.

**"pas_prox" is not anchored on a PAS hexamer.** No hexamer is detected anywhere in this
feature. It is bin 6 of the fixed grid, a 187.5 nt window straddling the pA (-62.5 to +125),
and the value is the mean U1 affinity of every 9-mer starting in it, with no threshold and
no requirement that the 9-mer look like a donor.

**The two U1 features share a name, not a definition.** They differ on every axis:

| axis | spl_pas_prox_u1 | spl_u1_suppression_up |
|---|---|---|
| scorer | U1 snRNA duplex free energy, from the precomputed 4^9 lookup `u1_lookup.npy` | a nine-position hand-specified donor weight table (`_DONOR` in `splice_extra_features.py`), not the U1 lookup and not MaxEnt |
| motif requirement | none; every 9-mer contributes | GT core required at 9-mer positions 3-4 |
| reduction | mean of a continuous score | count above a fixed cutoff, per 100 nt |
| window span | 187.5 nt straddling the pA (-62.5 .. +125) | 500 nt strictly upstream (-500 .. 0) |
| units | -dG kcal/mol | sites / 100 nt |
| distribution | continuous, symmetric-ish, median 0.616 | zero-inflated, 72.8% exactly zero |

So one is "how U1-friendly is the sequence immediately around the polyA site, on average",
and the other is "how many strong donor motifs sit in the 500 nt before it". They are
correlated at r = +0.207, and both survive forward selection with CI-clean coefficients
(-0.123 and -0.163) because they measure different things: local U1 affinity around the
site versus donor burden across the upstream half-kilobase. Note that
`spl_u1_suppression_up` is zero for 72.8% of candidates, so its coefficient is driven by the
27% of windows that carry at least one strong donor motif; the model treats it as continuous.


## Feature pool and exclusions

Pool: every `spl_*` feature except `spl_dist_dn_5ss` (70% missing) and `spl_dist_up_5ss`
(sign unstable, 26% imputed), plus `scan_acc_rank2k`, `scan_acc_margin2k`, `scan_in_intron`
and `acc_minus_max5`. The 5 kb count features (`scan_n3ss_dn5k`, `scan_n3ss_up5k`,
`scan_n3ss_diff`, `scan_n5ss_up5k`, `scan_n5ss_dn5k`) are excluded: a 5 kb window is outside
any plausible local mechanism, and the TERM-TERM probe showed the 3' ss version is a
compartment proxy rather than independent architecture (−0.289 → −0.001, 50.3% sign flips).

## Robustness and scope

On the unrestricted competitor set (64,140 peaks, 6,583 genes) the 4-feature model gives
holdout P@1 0.394 [0.368, 0.423] against chance 0.272, excess +0.122 [+0.098, +0.146].
Restricting to unannotated competitors raises the excess to +0.125 and tightens every
coefficient: against pure model artifacts the architecture signal is stronger, not weaker.

In that unrestricted analysis, of 2,167 confident misses 758 landed on a PolyASite cluster
and 281 were used at 10% or more — near-miss alternative 3' ends rather than noise. That
population is absent here by construction, which is why the atlas-overlap panel was dropped.

The anchor is not a relabelled annotation: 40.4% of canonical anchors sit within 2 nt of a
GENCODE acceptor and 59.6% do not (median distance 264 nt), against 9.0% for competitors.
The scan finds real acceptors preferentially, but most of the canonical signal comes from
acceptors GENCODE does not annotate at that position.

Two panels were dropped from earlier drafts as uninformative: the RF-override outcome (the
RF never picks the used end, by shortlist construction) and the compartment composition of
wrong picks (intronic at the same rate as the competitor pool they are drawn from).

## Coefficient bookkeeping

Every `spl_upstream_max_me3` value quoted here (+0.421) is the k = 3 model fit on the dev
genes, the model this figure is built on. The +0.455 reported during forward selection is
the same feature in the 12-feature plateau fit printed at the end of the selection run, not
the final model; the two differ because the extra nine features absorb part of the signal.
`rerank_pri_coefs.tsv` holds the k = 3 fit used in panels h and i; `rerank_primary_coefs.tsv`
holds the 12-feature fit.

## Files

| file | contents |
|------|----------|
| `task_rerank_primary.py` | forward selection on the primary set (path, chosen set) |
| `task_rerank_primary_panels.py` | metrics, curves, per-chromosome, contest, permutation null, coefficients, TERM-TERM probe |
| `task_rerank_rank.py` | rank distribution of the used 3' end (panel g) |
| `task_rerank_robust.py` | unannotated / TERM-TERM robustness runs and the anchor-annotation check |
| `make_figure_rerank_primary.R` | the figure; computes nothing |
