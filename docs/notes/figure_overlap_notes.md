# Are model false positives ever used experimentally

A competitor is a **model false positive**: an intragenic window that LR, RF and CNN all
score at least as high as its gene's own true CPA site. It is a model's mistake, not an
atlas-measured site. This overlap asks whether a position a CPA model prefers to the real
3' end is ever used in an experiment. It does not ask whether measured sites recur between
datasets.

Figure `figure_overlap_filtered.pdf` / `.png`, built by `make_figure_overlap.R`, which does
no computation.

## The set

`cpa_peaks_final.parquet` with `in_analysis_set == True`. Checked before anything else, and
the accounting script exits without writing if any count disagrees:

| | value | expected |
| --- | --- | --- |
| peaks | 48,394 | 48,394 OK |
| canonical | 4,847 | 4,847 OK |
| model false positives | 43,547 | 43,547 OK |
| genes | 4,847 | 4,847 OK |

The earlier run on 44,945 predates the downstream-of-canonical filter and is superseded.
Its outputs are the `ovl_filt_*` files; the corrected run is `ovl_ana_*`.

## Accounting: the false-positive set

Sequential, over all 70,723 shortlist peaks. "Genes hit" is genes containing at least one
removed peak; "genes emptied" is genes that lose their last false positive at that step and
leave the set with their canonical peak.

| step | removed | genes hit | genes emptied | remaining | genes |
| --- | --- | --- | --- | --- | --- |
| model false positives, all | 0 | 0 | 0 | 64,140 | 6,583 |
| APA-annotated: a PolyASite or PolyA_DB site within 500 nt | 18,119 | 5,122 | 1,575 | 46,021 | 5,008 |
| in a reference terminal exon | 1,076 | 689 | 85 | 44,945 | 4,923 |
| at or downstream of the gene's canonical site | 1,398 | 266 | 76 | 43,547 | 4,847 |

Reproduces `in_analysis_set` row for row.

## Accounting: the external datasets

| dataset | raw | merged blocks | top-N cut | lost to liftOver | duplicates merged | strand duplicates | final | in a contested gene | in an analysis gene |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3'-seq atlas | 9,557 | - | - | 0 (0.00%) | 4,106 | - | **5,451** | 2,898 | 2,258 |
| Oh 2017 control AMO | 184,174 | 50,472 | 10,472 | 25 (0.06%) | - | 39,978 | **39,972** | 8,643 | 6,275 |
| Oh 2017 U1 AMO | 187,018 | 67,976 | 27,976 | 30 (0.07%) | - | 39,970 | **39,970** | 8,329 | 6,320 |

Raw is intervals in the downloaded bigWig for the Oh sets and rows in the two downloaded
IPA tables for the atlas. Contested genes are the 6,583 shortlist genes, analysis genes the
4,847; membership is same-strand inside the gene body from `pc_genes.bed`.

**Two denominator problems the accounting turned up.**

1. *The Oh tables carry every site twice.* The published bigWig is unstranded and the peak
   table emits each called position on both strands with the same value, so 80,000 rows are
   40,000 sites. A row-based denominator halves every reported fraction of those datasets,
   which is what the previous run did: 1.64% and 1.56% should have been 3.27% and 3.12%.
   Sites are now counted once, and an Oh site counts as a hit if a false positive on either
   strand is within the tolerance, which is the only reading an unstranded assay supports.
   Hit counts barely move (1,309 to 1,307 and 276 to 273), so the double counting was almost
   entirely in the denominator.
2. *The Oh peak tables are a top-40,000 cut of a merged peak call, and the cut lands inside
   a tie block at the signal floor.* Intervals are first merged when their gap is at most
   25 nt, 187,018 to 67,976 blocks for U1 AMO and 184,174 to 50,472 for control AMO, and the
   top-40,000 then discards 27,976 and 10,472 blocks. The value at rank 40,000 is the floor
   itself, 0.0181 and 0.0200, sitting on 48,532 and 32,364 blocks, so the tie is broken by
   genomic iteration order and chr1 and chr2 are over-represented: 6,603 of 19,158 chr1
   intervals against 914 of 8,598 on chr10. `task_oh_topn.py` reproduces the published call
   exactly, 40,000 of 40,000 for both conditions, and re-runs it at 20,000 and 60,000; see
   `figure_u1_hits_notes.md`.

Neither affects the competitor-side fold or p, which never use the dataset as a denominator.

## Panel a: fraction of each dataset landing on a model false positive

Tolerance 50 nt. Percentage and Wilson 95% CI are dataset-side on unique sites. Fold and p
are false-positive-side: the peaks hit against the same sites displaced by a random 10 to
100 kb offset, 200 rounds, so p floors at 0.005.

| dataset | sites | all 64,140 false positives | 43,547 after the three filters |
| --- | --- | --- | --- |
| 3'-seq atlas (circular) | 5,451 | 596, **10.93%** [10.13, 11.79], 40.4-fold, p < 0.005 | 19, **0.35%** [0.22, 0.54], 2.1-fold, p = 0.015 |
| Oh 2017 control AMO | 39,972 | 1,307, **3.27%** [3.10, 3.45], 10.5-fold, p < 0.005 | 83, **0.21%** [0.17, 0.26], 1.2-fold, p = 0.065 |
| Oh 2017 U1 AMO | 39,970 | 1,249, **3.12%** [2.96, 3.30], 10.3-fold, p < 0.005 | 273, **0.68%** [0.61, 0.77], 3.9-fold, p < 0.005 |

**The separation holds at 43,547.** U1 AMO 3.9-fold at the p floor against its own control at
1.2-fold, p = 0.065; the fold ratio is 3.25, against 3.2 at 44,945 (3.8 and 1.2). Removing
the 1,398 downstream peaks moved U1 AMO from 3.8 to 3.9 and left the control where it was.
Dataset-side the same contrast is 0.68% against 0.21%, a ratio of 3.3.

On the full 64,140 the two AMO conditions are indistinguishable, 10.5-fold against 10.3-fold,
because the unfiltered pool is dominated by annotated poly(A) sites that both conditions
detect. The control condition is the comparison that makes the filtered number readable: it
is the same assay, the same lab, the same peak caller, and the same top-N cut.

## Circularity

The first filter removes false positives within 500 nt of a PolyASite or PolyA_DB site. The
3'-seq atlas is an APA atlas of that same class, so its collapse from 10.93% to 0.35% is a
property of the filter and carries no information about the peaks. The row is labelled
circular on the panel. The Oh sets are perturbation data from a different assay and are not
touched by the filter.

## immune and cll are not independent. Decision: merge

Merged into one 3'-seq atlas set: 9,557 raw sites, 5,451 after collapsing sites within 25 nt
on the same strand, so 4,106 of the raw sites are the same site counted twice. Row-into-column
agreement is 0.836 and 0.890 within 50 nt and 0.412 and 0.438 at exact coordinates, the
signature of one experiment peak-called twice. The alternative, restricting to the 330
CLL-gained IPAs, was rejected because that list is not in the downloaded files:
`GSE111310_ipa_tpm.txt.gz` carries per-sample TPM only, so using it would mean re-deriving
the differential call rather than using the published one.

## Panel b: UpSet, analysis set only

370 of the 43,547 false positives carry a match within 50 nt. Set sizes: U1 AMO 274,
control AMO 87, 3'-seq atlas 19.

| combination | false positives |
| --- | --- |
| U1 AMO only | 265 |
| control AMO only | 79 |
| 3'-seq atlas only | 17 |
| control AMO + U1 AMO | 7 |
| all three | 1 |
| 3'-seq atlas + U1 AMO | 1 |

361 of the 370 are hit by one dataset alone, so the three sets corroborate different peaks
rather than the same ones.

## Legend sentence replacing the pairwise heatmap

Lee 2018 reused the Singh 2018 immune samples: their site lists agree at 0.84 and 0.89
within 50 nt (0.41 and 0.44 at exact coordinates), so they are merged into one 3'-seq atlas
set. The two conditions of the Oh 2017 experiment agree at 0.21, and agreement across labs
is 0.01 to 0.21.

## Two parsing traps in `external/lifted/`

Both corrupt coordinates without raising an error, and both are avoided by writing plain
chrom/pos/strand tables.

1. `cll_hg38.tsv` carries a bare CR before the two appended hg38 columns. A default
   `read_csv` splits every line in two, returns 9,261 rows for a 4,630-site file, and drops
   `hg38_chrom`/`hg38_pos` entirely.
2. The lifted tables keep the original hg19 `start` and `end` beside the lifted `hg38_pos`.
   `03_simple_overlap.py` replaces a position with the midpoint of start and end when a peak
   looks wide, and `end - hg38_pos` here is a liftover displacement, median 152,974 nt for
   immune, not a peak width.

Neither affects the earlier `fig_site_overlap` run, which passed the pre-liftover hg19
tables and let the script do the lift.

## Provenance

```
python task_overlap_accounting.py                       # verify, account, write the peak table
python scripts/03_simple_overlap.py --peaks ovl_peaks_analysis.tsv \
  --external atlas3seq=ovl_atlas3seq.tsv \
             oh_camo=external/peaks/oh2017_cAMO_hg38.tsv \
             oh_u1amo=external/peaks/oh2017_U1AMO_hg38.tsv \
  --tolerances 25,50,100 --shuffle 200 --out ovl_ana
Rscript make_figure_overlap.R
```

`ovl_unfilt_*` (all 64,140) and `ovl_pairwise_*` (agreement matrix) are unchanged from the
earlier session and are still valid; the competitor pool they use did not change. Console
output is in `ovl_runs.log` and `ovl_runs_analysis.log`. The filter check against PolyASite
2.0 and PolyA_DB v4, 0 of 44,945 within 500 nt with the nearest survivor at 501 nt, is in
`task_overlap_prep.py` and `ovl_filter_check.tsv`; the analysis set is a subset of that set,
so it inherits the result.
