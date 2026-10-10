# PROVENANCE — pub/chap2_paradox_fig/

Built 2026-08-31. All inputs read-only; nothing outside this directory was written.

## Environment

```
python   3.10   /home/spour98/venv_2025/bin/python3   (pandas 2.2.3, numpy 1.26.4)
pyarrow  17.0.0 PYTHONPATH=/cvmfs/.../gcccore/arrow/17.0.0/lib/python3.10/site-packages
R        4.5.0  /cvmfs/.../gcccore/r/4.5.0/bin/Rscript
                ggplot2 4.0.3, patchwork 1.3.2, dplyr 1.2.1, tidyr 1.3.2,
                readr 2.2.0, scales 1.4.0, hexbin 1.28.6
```

`module load arrow/17.0.0` alone does not work: it sets `PYTHONPATH` to a directory
with no pyarrow for python3.10. `$EBROOTARROW/lib/python3.10/site-packages` must be
put on `PYTHONPATH` explicitly.

## Files

| Output | Built by | Renders |
|---|---|---|
| `panelA.tsv` | phase-1 python (inline, not saved to this dir) | panel A |
| `panelB.tsv` | reshape of an existing pipeline output | panel B |
| `panelC.tsv` | phase-1 python | panel C |
| `make_paradox_fig.R` | — | reads the three TSVs, writes the figure |
| `paradox_fig.pdf` / `.png` | `make_paradox_fig.R` | 17 x 12 in, cairo_pdf, dpi 400 |

## panelA.tsv — n = 4,332 genes

One row per gene. Source: `pub/shortlist_cpa.parquet` (50,130 rows, 28 cols).

| Column | Source column | Rule |
|---|---|---|
| `gene_id` | `gene` | — |
| `canonical_aparent2` | `aparent2_prob` | row with `is_canonical == True` |
| `bestcomp_aparent2` | `aparent2_prob` | best-competitor row |
| `bestcomp_consensus` | **DERIVED** — see below | best-competitor row |
| `dist_nt` | `dist_to_canon` | best-competitor row |

**Best competitor** = the row maximising `bestcomp_consensus` among rows with
`is_canonical == False AND in_analysis_set == True AND aparent2_prob NOT NULL`.
The APARENT2-non-null restriction is applied *before* the argmax, so the selected
competitor always has a plotted y-value; without it the argmax could pick a row
with no APARENT2 score and the gene would drop out.

**`bestcomp_consensus` is the one inferred column.** There is no consensus score in
any file. In `pub/scripts/build_shortlist.py` "consensus" is a boolean predicate,
`rf_prob >= rf_canonical AND lr_prob >= lr_canonical AND cnn_prob >= cnn_canonical`
(`:212`, `:229-231`), followed by per-model NMS; `--score_col` (`:122-124`) selects a
*mode*, not a column. Per instruction, `bestcomp_consensus` is defined here as
`mean(rf_prob, lr_prob, cnn_prob)` — the same scalar the pipeline itself ranks on at
`build_shortlist.py:202` (`mean_score = np.mean([S[m] for m in MODELS], axis=0)`,
used at `:203`, `:251`, `:277`). Two alternatives were offered and not taken:
`min(rf,lr,cnn)` and `min` of the three margins over `*_canonical`.

`dist_nt` is not recomputed. `dist_to_canon` was verified equal to
`|center - canonical center|` on 50,130 / 50,130 rows, max discrepancy 0, with exactly
one canonical row per gene.

Attrition to n = 4,332: 6,583 canonical rows → 5,898 with APARENT2; 43,547 analysis-set
competitors → 38,186 with APARENT2, covering 4,333 genes; intersection 4,332.
Independently equals `pub/fig2_aparent.tsv` `within_gene_rank n_genes = 4332`.

Distributions: `dist_nt` median 32,696 nt (IQR 13,205–78,021), min 503.
APARENT2 min 2.49e-06 (canonical) / 2.04e-04 (competitor); no zeros or negatives, so
the log10 axes in panel A drop nothing.

## panelB.tsv — reshaped, not recomputed

Source: `pub/shortlist_cpa_density.tsv` (2,001 rows), which already holds this content.
Not recomputed from `shortlist_cpa.parquet`.

| Output column | Source column | Change |
|---|---|---|
| `motif` | `motif` | filtered to PAS, UGUA, U-rich, GU-rich |
| `position` | `pos` | renamed only |
| `class` | `cls` | `Canonical` → `canonical`, `Model false positive` → `competitor` |
| `density` | `density` | unchanged |

1,600 rows = 4 motifs x 2 classes x 200 positions.

Two deviations from the spec, both reported rather than papered over:

1. **Positions are −100…+99, not −100…+100.** The source has 200 positions per series,
   not 201. No +100 row was invented. The 9-point moving average applied in the R
   script drops 4 more at each end, so the drawn lines span −96…+95.
2. **`PAS (dinuc null)` exists in the source and was dropped**, as the spec's motif list
   excludes it. It is 400 of the 2,001 source rows.

**Asymmetric denominators.** `pub/shortlist_cpa_density_counts.tsv`: Canonical **6,583**
against Model false positive **43,547**. Canonical peaks are all canonicals; competitors
are analysis-set only. This is deliberate upstream
(`pub/scripts/featurize_shortlist_cpa.py:23-24`: "Canonical peaks are always kept;
non-canonical peaks are kept only when in_analysis_set is true"). Densities are per-peak
so the curves remain comparable, but they rest on different peak counts. Stated in the
figure caption.

## panelC.tsv — 6-column, 5-row, same-model form

| task | model | auroc | lo | hi | n | source |
|---|---|---|---|---|---|---|
| canonical vs negative | RF | 0.987359 | — | — | 104,031 | `pub/models/prod/rf/metadata.json` → `winner_meta.test_metrics.AUC` |
| canonical vs negative | LR | 0.982677 | — | — | 104,031 | `pub/models/prod/lr/metadata.json` → same path |
| canonical vs negative | CNN | 0.991206 | — | — | 104,031 | `pub/models/prod/cnn/metadata.json` → same path |
| canonical vs competitor | APARENT2 | 0.4782 | 0.4697 | 0.4854 | 44,084 | `pub/fig2_aparent.tsv`, row `kind=pooled_auroc, subset=analysis_set`, cells `value/lo/hi`; n = `n_canon` 5,898 + `n_fp` 38,186 |
| canonical vs negative | APARENT2 | 0.791571 | 0.7856 | 0.7978 | 80,246 | **computed here** — see below |

`lo`/`hi` are blank for RF/LR/CNN: no CI is stored in any of the three metadata files
(`lo`, `hi`, `ci`, `confidence` all absent). `n` is `report` support, 104,031.

A 7th column, `clean_comparison` (bool), marks the APARENT2 canonical-vs-negative row.
It is the only contrast that holds the model fixed and varies only the task; the figure
rings that point.

### The APARENT2 canonical-vs-negative row

APARENT2 scores exist for negative windows. `scripts/score_genes_aparent2.py:13-16` added
`aparent2_prob` to **every window of every per-gene parquet** in
`train/gene_scans/per_gene/` (17,895 genes). Joining
`pub/splits/prod/{pos,neg}_test.bed` on `(gene, start → window_start_genomic)` lands on a
score 100% of the time where the gene's scan file exists.

- Recovered **80,246 of 104,031** prod test windows (77.1%): 2,451 positive, 77,795 negative.
- Coverage is limited by gene-set overlap, not by missing scores, and is **not uniform by
  chromosome**: chrX 365 genes present / 895 absent; chr19 958 / 402.
- AUROC by the same rank estimator used elsewhere (Mann-Whitney, midranks).
- CI = 200-resample stratified bootstrap over sites, `np.random.default_rng(0)`.

**Subset-bias check — passed.** Recomputing the three production models from
`pub/scans/prod/per_gene` on the identical windows:

| | RF | LR | CNN | APARENT2 |
|---|---|---|---|---|
| locked metadata (104,031) | 0.987359 | 0.982677 | 0.991206 | — |
| join, 100,731 windows (96.8%) | 0.988085 | 0.983680 | 0.991566 | — |
| matched subset, 80,246 (77.1%) | 0.987645 | 0.982700 | 0.990940 | 0.791571 |

The join reproduces the locked metadata to +0.0004…+0.0010, and restricting to the
APARENT2-covered subset moves the three models by less than 0.0003. The 0.792 is
therefore comparable to the metadata AUROCs on the same axis.

**Caveat to carry into the text:** the APARENT2 scan in `train/gene_scans/` is the
prior-generation scan, while RF/LR/CNN come from `pub/models/prod/`. The windows and
labels are identical; the model provenance is not the same vintage.

## Figure

`make_paradox_fig.R` reads only the three TSVs and does no modelling. The two statistics
it computes are rank statistics of the plotted rows: the panel-A AUROC and the
canonical-higher fraction. Its `auroc()` is the same estimator as
`pub/scripts/figure2_stats.py:49-60`, reimplemented rather than imported so the figure is
self-contained.

Theme, palette and output conventions copied from `pub/scripts/make_figure2.R` lines
40–69: `theme_classic(base_size = 16)`, `INK #1A1A1A`, `CANON #0072B2`, `CC #E69F00`,
`CCLAB "cCPA site"`, `cairo_pdf`, `dpi 400`. That file was read for reference and is not
sourced. Panel B applies the same 9-point moving average as `make_figure2.R:150` and
`:210`.

Panel A annotates two AUROCs because they answer different questions and differ:
**0.469** on the plotted rows (canonical vs the single best competitor per gene, 4,332 v
4,332) and **0.478** pooled over all competitors (from `panelC.tsv`). The best-competitor
contrast is the harder one, hence lower. Both are below 0.5.

### ggplot2 4.0.3 note

`annotate("text", x = Inf, y = -Inf, ...)` is silently destructive on a log10 scale:
ggplot transforms the annotation position along with the data, `log(-Inf)` is `NaN`, and
the label is dropped with `In log(x, base) : NaNs produced` plus
`Removed 1 row containing missing values (geom_text)`. Panel A places its annotation at
finite data coordinates instead. The two `pub/CLAUDE.md` traps (`label.size` rejected by
`annotate("label", ...)`; `geom_errorbarh` deprecated) do not arise here — neither is used.

## Rebuild

```bash
cd /scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub/chap2_paradox_fig
/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/r/4.5.0/bin/Rscript \
    make_paradox_fig.R .
```

Renders in seconds from the TSVs; no parquet read, no scan. Regenerating the TSVs
themselves requires the phase-1 python and the parquet inputs listed above.
