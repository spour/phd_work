# `cpa_peaks_final.parquet`

Per-peak table for the final analysis: real human 3' ends against the false positives of a
baseline CPA model, with the two-feature model's held-out score. One row per peak, **70,723
rows in 6,583 genes**. Nothing is deleted; every subset used in the paper is reconstructed
by filtering the flag columns, so the file is self-contained.

Built by `task_build_final_parquet.py`. **This file supersedes `rerank_corr_peaks.parquet`**,
which predates the downstream-of-canonical filter and carries `discriminant_k3` and
`discriminant_full` from the retired three-feature U1 model; it has been renamed
`rerank_corr_peaks.SUPERSEDED.parquet` and should not be used.

## Genome and coordinates

Assembly **hg38** (`chrom` is chr-prefixed).

`center` is the **poly(A) site** of the peak, not the acceptor and not the window start.

**`center` is 0-based.** Every sequence fetch in this project, in the pipeline and in the
analysis scripts, indexes the FASTA with `center` as a 0-based offset, and the feature
rebuild reproduced the pipeline's own features to a maximum absolute difference of 0.000000
only under that reading. That is the evidence for the convention. Agreement with the
databases is **not** evidence either way, for the reason in the next paragraph.

**Do not expect a base-exact match to PolyASite 2.0 or PolyA_DB v4, and do not read the
shortfall as an off-by-one.** `center` is where the shortlist's model scoring placed the
peak, not a called cleavage base, and cleavage positions are genuinely spread. For the 6,583
canonical peaks, which were pinned to atlas-derived dominant CPA sites and so should match
if anything does, the fraction with a same-strand site nearby is:

| window | PolyASite 2.0 | PolyA_DB v4 | either |
| --- | --- | --- | --- |
| exact | 0.0545 | 0.0632 | 0.0782 |
| +/-1 nt | 0.1961 | 0.2304 | 0.2771 |
| +/-25 nt | 0.8286 | 0.8577 | 0.9236 |
| +/-500 nt | 0.9692 | 0.9515 | **0.9945** |

The signed offset has no dominant spike: over the 4,533 canonical peaks within +/-5 nt of a
PolyA_DB site the mode is -1 nt and holds 706 of them, against 416 at 0 and 395 at +1, and
the median absolute offset is 3 nt with a 75th percentile of 7 nt. A true one-base error
would put most peaks at a single offset; this is positional spread. Consistently, shifting
`center` by one base changes `flag_apa_annotated` on **11 of 70,723 peaks**, so nothing in
this file is sensitive to the choice.

Practical advice for joining: use a +/-25 nt window, which recovers 92% of canonical peaks,
and treat a base-exact join as unavailable. If your annotation is 1-based, add 1 to `center`
first; it will change almost nothing, but it is the correct comparison.

The upstream window used by the features is `[center-2000, center-126]` in **transcript**
orientation, meaning decreasing genomic coordinate on the minus strand. In the fetch
convention the pipeline uses, the plus strand takes its first upstream base at genomic
`center-2000` and the minus strand at genomic `center+1999`, a one-base asymmetry between
strands that is inherited from the pipeline and preserved here so the features reproduce.

## Rows: what a peak is

Peaks come from the contested-gene shortlist (`scripts/build_shortlist.py`), built on 500 nt
windows; `center` is each window's midpoint.

- **`is_canonical`** marks, per gene, the peak pinned to the gene's dominant CPA site from
  the atlas-derived `Dominant_CPA_baseline_total.bed`: among the shortlist windows
  overlapping that annotated dominant-usage interval, the best-scoring one. Exactly one per
  gene, 6,583 in this file.
- **Competitors** (`is_canonical == False`) are baseline-model false positives: windows that
  **all three** baseline models independently score at least as high as the gene's own
  canonical site (`rf_prob >= rf_canonical` and `lr_prob >= lr_canonical` and
  `cnn_prob >= cnn_canonical`), then collapsed by non-maximum suppression to one peak per
  site. No single model does the selecting, so RF, LR and CNN are all spent on defining the
  negative set, and a competitor is by construction a position a CPA model prefers to the
  real 3' end.

## Columns

| column | type | meaning |
| --- | --- | --- |
| `gene` | str | gene symbol; every gene has exactly one canonical peak |
| `chrom` | str | hg38 chromosome, chr-prefixed |
| `center` | int64 | 0-based poly(A) site of the peak |
| `strand` | str | `+` or `-` |
| `is_canonical` | bool | the gene's dominant CPA site (see above) |
| `flag_apa_annotated` | bool | a PolyASite 2.0 cluster or a PolyA_DB v4 site within 500 nt |
| `flag_reference_terminal_exon` | bool | the 500 nt strand-oriented window ending at the pA overlaps a GENCODE v49 last exon by at least 1 nt |
| `flag_downstream_of_canonical` | bool | a competitor at or downstream of its gene's canonical pA in transcript orientation; always False for canonical peaks |
| `in_analysis_set` | bool | the 48,394 peaks the paper analyses |
| `spl_upstream_max_me3` | float | feature 1, see below |
| `gc_contrast` | float | feature 2, see below |
| `discriminant` | float | held-out linear predictor of the final model; NA outside the analysis set |
| `loco_chromosome` | str | the fold that scored the peak, identified by the chromosome it held out; NA outside the analysis set |
| `in_gene_holdout` | bool | the peak's gene is in the untouched 20% gene holdout |

The three flags are **raw properties of each peak**, not the sequential exclusions. A peak
can carry more than one, and canonical peaks carry them too (a real 3' end is usually
APA-annotated and in a terminal exon). Raw counts over all 70,723 peaks: 24,666 / 12,785 /
4,380, of which 6,547 / 6,415 / 0 are canonical.

The figure's filter counts are the **sequential** exclusions over the competitor pool of
64,140, and are reproduced from this file as:

```python
comp = ~T.is_canonical
apa   = comp &  T.flag_apa_annotated                                  # 18,119
term  = comp &  T.flag_reference_terminal_exon & ~T.flag_apa_annotated  # 1,076
down  = comp & ~T.flag_apa_annotated & ~T.flag_reference_terminal_exon \
             &  T.flag_downstream_of_canonical                        # 1,398
kept  = comp &  T.in_analysis_set                                     # 43,547
```

`in_analysis_set` is fully derivable from the flags, and this reproduces it row for row:

```python
keep_comp = comp & ~T.flag_apa_annotated & ~T.flag_reference_terminal_exon \
                 & ~T.flag_downstream_of_canonical
genes     = set(T.loc[keep_comp, "gene"])
derived   = keep_comp | (T.is_canonical & T.gene.isin(genes))
assert (derived == T.in_analysis_set).all()          # 48,394 peaks in 4,847 genes
```

A gene enters the analysis set only if it keeps at least one competitor, so genes whose
competitors were all removed leave with their canonical peak; that is why the analysis set
covers 4,847 of the 6,583 genes.

## Features

Both are measured in transcript orientation relative to `center`, on the pipeline's own
featurisation grid (upstream 2000 nt, downstream 500 nt, k-mer **start** offsets).

- **`spl_upstream_max_me3`** — the maximum MaxEntScan `score3` over 23-mers whose start lies
  in `[center-2000, center-126]`, requiring AG at 0-based positions 18-19 of the 23-mer.
  It is the **strength** of the best upstream acceptor, not its position: the distance to
  that same anchor adds +0.001 AUROC to the model, with a confidence interval covering zero.
- **`gc_contrast`** — mean G+C over `[center-2000, center-438]` minus mean G+C over
  `[center-437, center-126]`, N excluded from each denominator. Positive means the site is
  locally AT-rich against GC-richer upstream sequence. The classes cross at -344 nt, next to
  the -437 split; the split point and the acceptor bound were inherited from the pipeline and
  then swept over a 4 x 5 grid in which the deployed cell is the best cell.

Present for all 70,723 peaks. `gc_contrast` was rescanned for this file over every peak and
agrees with the analysis cache on the 48,394 analysis peaks to a maximum absolute difference
of 0.

## `discriminant`

The final model is one ridge logistic regression on the two features above (lambda = 1.0,
intercept unpenalised, features standardized within each training fold with train-fold mean
imputation).

For each chromosome, the model is fitted on the analysis-set peaks of the **80% development
genes on the other chromosomes**, and then scores every analysis-set peak on that
chromosome; `loco_chromosome` records which fold that was. A gene lies on one chromosome, so
no peak is ever scored by a fit that saw its own gene. The **20% gene holdout**
(`in_gene_holdout`, seed 42) was never used for feature selection, for the choice of k, or
for any window or cutoff decision, and it is the set the headline numbers come from:

| on the 974 holdout genes, 9,624 peaks | value |
| --- | --- |
| within-gene win-rate | **0.6450** (permutation null 0.500) |
| pooled AUROC | **0.6486** (its own permutation null 0.5272, excess +0.1214) |
| AUPRC | 0.1689 (base rate 0.1012) |

Reproducing those two numbers from this file is the build script's final check, and it
passes exactly.

**Pooled AUROC does not have a 0.5 null here.** Under the within-gene label permutation used
throughout, each gene contributes one positive but pooled AUROC compares across genes, so any
score whose gene-level mean correlates with peaks-per-gene has a null off 0.5. Every AUROC in
the paper is quoted against its own permutation null. The within-gene win-rate has a null of
exactly 0.5 and is the headline metric.

Peaks outside the analysis set have `discriminant` and `loco_chromosome` set to NA. They were
excluded by the filters, so no model was fitted to them and none scored them.
