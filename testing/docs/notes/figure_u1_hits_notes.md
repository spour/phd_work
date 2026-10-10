# The model false positives a U1 AMO experiment used

Set: `cpa_peaks_final.parquet`, `in_analysis_set` and not canonical, 43,547 model false
positives in 4,847 genes. Hit flags read from `ovl_ana_per_peak.tsv` and joined on
(chrom, center, strand, gene). Figure `figure_u1_hits.pdf` / `.png`.

**Count correction.** 274 false positives carry a U1 AMO site within 50 nt, not 273, and 87
carry a control AMO site, not 83. The 273 and 83 are dataset-side, the number of Oh sites
landing on a false positive; the two differ when two Oh sites fall near the same peak. So the
split is 274 against 43,273. Six false positives are hit by both conditions.

## 1. Discriminant

Cliff's delta with a gene-clustered 95% CI; Mann-Whitney by normal approximation with tie
correction. Delta is 2A - 1 for the common-language A, so it is the AUROC used elsewhere in
this project rescaled to [-1, 1].

| comparison | n | median | Cliff's delta | p |
| --- | --- | --- | --- | --- |
| U1 AMO hit vs not hit | 274 / 43,273 | -2.198 vs -2.364 | **+0.149** [+0.080, +0.214] | 2.1e-05 |
| control AMO hit vs not hit | 87 / 43,460 | -2.123 vs -2.364 | **+0.273** [+0.156, +0.389] | 1.0e-05 |
| canonical vs U1 AMO hit | 4,847 / 274 | -2.031 vs -2.198 | +0.168 [+0.097, +0.237] | 2.8e-06 |
| canonical vs not hit | 4,847 / 43,273 | -2.031 vs -2.364 | +0.317 [+0.301, +0.333] | 6e-288 |

The comparison is not null: the used false positives score above the unused ones, and they
sit between the unused ones and the real 3' ends.

**The specificity check fails.** Control AMO hits are shifted at least as far, +0.273 against
+0.149, and the two intervals overlap. Control AMO is the same lab, assay, peak caller and
top-N cut without the perturbation, so the discriminant shift is a property of which
positions this assay detects, not of U1 loss. On the discriminant alone the 274 cannot be
read as U1-dependent cryptic sites.

## 2. The two features

| quantity | comparison | Cliff's delta | p |
| --- | --- | --- | --- |
| acceptor strength | U1 AMO hit vs not hit | +0.022 [-0.047, +0.087] | 0.54 |
| acceptor strength | control AMO hit vs not hit | +0.165 [+0.034, +0.300] | 0.0077 |
| gc_contrast | U1 AMO hit vs not hit | **+0.180** [+0.112, +0.246] | 2.7e-07 |
| gc_contrast | control AMO hit vs not hit | +0.196 [+0.064, +0.313] | 0.0016 |

**gc_contrast carries all of it.** The acceptor is null for U1 AMO hits, delta +0.022 with an
interval spanning zero, so the whole of the discriminant shift is composition. Against the
canonicals, U1 AMO hits are indistinguishable on gc_contrast, delta +0.028 [-0.042, +0.101],
p = 0.43, while they remain below on the acceptor, +0.220 [+0.147, +0.283], p = 9e-10. A
used false positive has the composition of a real 3' end and not the acceptor.

## 3. Within gene

174 genes carry at least one hit and at least one non-hit false positive, holding 228 of the
274 hits and 4,030 non-hit false positives. Per gene, the mean discriminant of the hits minus
the mean of the non-hits:

```
mean paired difference   +0.1168
median                   +0.1433
fraction above zero       0.5747
within-gene permutation null -0.0001 [-0.0915, +0.0925]   p = 0.0066  (10,000 shuffles)
```

The shift survives holding the gene fixed, so it is not gene expression or gene length. The
effect is small relative to the spread, and the fraction above zero is 0.57 against a null of
0.5, so it is a shift in a minority of genes rather than a property of most.

## 4. Geometry

| quantity | comparison | median | Cliff's delta | p |
| --- | --- | --- | --- | --- |
| distance to TSS | U1 AMO hit vs not hit | 23,755 vs 85,761 nt | **-0.492** [-0.561, -0.417] | 7e-45 |
| distance to TSS | control AMO hit vs not hit | 42,543 vs 85,220 nt | -0.273 [-0.396, -0.141] | 1.1e-05 |
| distance to canonical | U1 AMO hit vs not hit | -54,455 vs -93,636 nt | +0.223 [+0.138, +0.307] | 1.7e-10 |
| distance to canonical | control AMO hit vs not hit | -33,391 vs -93,464 nt | +0.357 [+0.239, +0.457] | 8e-09 |
| host intron length | U1 AMO hit vs not hit | 16,943 vs 35,003 nt | -0.253 [-0.336, -0.165] | 7e-13 |
| host intron length | control AMO hit vs not hit | 8,798 vs 34,994 nt | -0.362 [-0.478, -0.244] | 2e-08 |
| host intron rank | U1 AMO hit vs not hit | 2 vs 3 | **-0.249** [-0.320, -0.180] | 8e-13 |
| host intron rank | control AMO hit vs not hit | 5 vs 3 | +0.109 [-0.041, +0.250] | 0.087 |

Host intron from the Ensembl canonical transcript of each gene, GENCODE v49, ranked from the
TSS; 270 of the 274 hits and 37,310 of the non-hits fall inside one.

Share in the first two introns, Wilson 95% CI:

| group | n | first two introns |
| --- | --- | --- |
| U1 AMO hit | 270 | **62.2%** [56.3, 67.8] |
| control AMO hit | 80 | 37.5% [27.7, 48.5] |
| not hit | 37,310 | 42.5% [42.0, 43.0] |

**Yes, the 274 are largely the TSS-proximal ones, and here the control separates.** U1 AMO
hits are strongly first-intron biased while control AMO hits are not: 62.2% against 37.5%,
and the intron-rank delta is -0.249 for U1 AMO against +0.109, interval spanning zero, for
the control. The Oh 2017 figure of 38% of PCPA in the first two introns is what the control
condition and the unused false positives both show; only the U1 AMO hits exceed it.

So the two readings separate. The discriminant shift is not specific to U1 loss. The
first-intron enrichment is.

## 5. Robustness to the top-N cut

The published peak call is recovered exactly by `task_oh_topn.py`: merge bigWig intervals
whose gap is at most 25 nt, keep each block's argmax position 1-based and its maximum value,
take the N largest. At N = 40,000 this reproduces both published tables, 40,000 of 40,000.

The value at rank 40,000 is the signal floor itself, 0.0181 for U1 AMO on 48,532 of 67,976
blocks and 0.0200 for control AMO on 32,364 of 50,472. Every cut tested lands inside that tie
block, so the tie is broken by genomic iteration order, which is why chr1 and chr2 are
over-represented in the published tables: 6,603 of 19,158 chr1 intervals against 914 of 8,598
on chr10.

| N | U1 AMO hits | fold | p | control AMO hits | fold | p | fold ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20,000 | 163 | **4.8** | < 0.005 | 27 | 0.8 | 0.91 | 6.0 |
| 40,000 | 273 | **3.8** | < 0.005 | 83 | 1.2 | 0.065 | 3.2 |
| 60,000 | 394 | **3.6** | < 0.005 | 112 | 1.3 | < 0.005 | 2.8 |

Counts are dataset-side. Control AMO has only 50,472 blocks, so its N = 60,000 row is the
whole set. The U1 AMO enrichment holds at every cut, 3.6 to 4.8, always at the p floor, and
the control stays at or near 1. The fold ratio falls monotonically as the cut widens, 6.0 to
2.8, which is what adding floor-value ties should do. The claim that survives is directional
and bounded: U1 AMO sites land on model false positives 3.6 to 4.8 times more often than
displaced sites do, and 3 to 6 times more often than the matched control condition, with the
exact number set by a peak-call cut that is arbitrary inside its tie block.

## Provenance

```
python task_u1_hit_character.py     # stats, within-gene permutation, geometry
python task_u1_hit_intron.py        # per-rank counts and the Wilson interval
python task_u1_hit_ecdf.py          # panel a curves
python task_oh_topn.py              # rebuild the Oh peak call at 20k, 40k, 60k
python scripts/03_simple_overlap.py --peaks ovl_peaks_analysis.tsv \
  --external camo20k=... u1amo20k=... camo40k=... u1amo40k=... camo60k=... u1amo60k=... \
  --hg19 camo20k u1amo20k camo40k u1amo40k camo60k u1amo60k \
  --tolerances 50 --shuffle 200 --out ovl_topn
Rscript make_figure_u1_hits.R
```

`gencode_canonical_exons.tsv` is a 348,001-line extract of the Ensembl canonical exons from
`../train/data/annotation/gencode.v49.annotation.gtf`, kept so the intron assignment does not
need the 3.3 GB file again. Console output is in `ovl_topn.log`.
