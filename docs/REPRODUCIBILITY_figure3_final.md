> **SUPERSEDED. Do not use.** Replaced by `REPRODUCIBILITY.md`, which documents
> the merged Figure 3 (`figure3_merged.png`). This file describes an earlier figure
> and, in the case of the spliceai_me document, an earlier acceptor encoding
> (`mean` over 2 kb, AUROC 0.7451) that was replaced by `max` over 2 kb (0.7369).
> Kept only as a record of what was built and why it changed.

# Reproducibility: figure3_final (MaxEnt vs SpliceAI)

Everything `figure3_final.png` / `.pdf` and its headline numbers depend on.
Written 2026-07-21.

**Supersedes `REPRODUCIBILITY_spliceai_me.md`**, which documents the earlier
two-row figure. In particular that document's caveat "panel D does not support its
title" is resolved here: the figure now states the crossover in the title, the
subtitle, an on-panel callout, and a held-out statistic. Keep the old file only as
a record of the superseded figure, or delete it.

Working directory for every command:

```
/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub
```

---

## 1. Headline numbers

| Quantity | Value | Source |
|---|---|---|
| MaxEnt acceptor, peak over windows | 0.606 | `fig3_bcshow_auroc.tsv` |
| MaxEnt 5'ss, peak (oriented) | 0.548 | `fig3_bcshow_auroc.tsv`, raw 0.452 |
| SpliceAI acceptor, **held-out test** | **0.7369** | `sa_marginal_final.tsv` |
| SpliceAI donor sum, held-out test | 0.6754 | `sa_marginal_final.tsv` |
| SpliceAI donor max, held-out test | 0.6541 | `sa_marginal_final.tsv` |
| Base rate P(true end) | 0.09308 | `fig3_class_counts.tsv` (6583 / 70723) |

### 0.737 test vs 0.740 pooled

Two different numbers for the same encoding, and the figure uses the test one:

- **0.7400** is `sa_acc_max` AUROC over all 70,723 peaks (pooled).
- **0.7369** is the same encoding scored on the held-out half only.

Panel E plots held-out test AUROC and its acceptor curve terminates at 0.737, so
the annotation box prints 0.737. Printing 0.740 next to a point drawn at 0.737
would contradict the plot. Both are emitted by `figure3_final.py` to stderr; only
the test value reaches the figure.

### The crossover statistic

The claim "at high acceptor a stronger donor marks the competitor" rests on this
table, not on reading the heatmap. `sa_tertile_final.tsv`:

| acceptor tertile | donor | AUROC (test, raw) | direction | n | n canonical |
|---|---|---|---|---|---|
| weak | sum | 0.6506 | canonical | 12213 | 412 |
| weak | max | 0.6412 | canonical | 12213 | 412 |
| medium | sum | 0.5312 | canonical | 11756 | 521 |
| medium | max | 0.5133 | canonical | 11756 | 521 |
| **strong** | **sum** | **0.4141** | **COMPETITOR** | 11785 | 2346 |
| strong | max | 0.3681 | COMPETITOR | 11785 | 2346 |

Monotonic 0.651 -> 0.531 -> 0.414, crossing 0.5 in the strong tertile. Two
properties make it load-bearing:

1. **Tertile edges are cut on TRAIN and applied to TEST** (`sa_acc_max` quantiles
   at 1/3 and 2/3 of the train half), so the strata are not selected using the half
   they are scored on.
2. Both donor reductions cross, and `max` goes further than `sum` (0.368 vs 0.414),
   so the reversal is not an artifact of the sum encoding.

`figure3_final.py` prints this table to stderr and warns if the strong tertile does
not fall below 0.5.

---

## 2. Dependency contract

`figure3_final.py` checks all of the following before any work and exits 1 naming
each missing path. MaxEnt is never recomputed.

| Required input | Used for |
|---|---|
| `fig3_ratio.tsv` | **panel A** acceptor bubbles (y = `maxscore`, thr 8.0) |
| `fig3_bcshow_ratio.tsv` | panels B, C bubbles (y = `yval`, thr 4.0; `u1` and `donor` only) |
| `fig3_bcshow_auroc.tsv` | panel D curves, peak-AUROC subtitles, MaxEnt donor marginal |
| `fig3_acc_x_interv.tsv` | panel F |
| `fig3_metagene.tsv` | panel H |
| `fig3_class_counts.tsv` | base rate, n labels |
| `fig3_ucube_meta.tsv` | per-peak table: gene, chrom, center, strand, is_canonical, acc |
| `spliceai_bw/*.bw` (4) | SpliceAI acceptor/donor, both strands |

**Both `fig3_ratio.tsv` and `fig3_bcshow_ratio.tsv` are required.** They are not
interchangeable. `fig3_bcshow_ratio.tsv` contains only `u1` and `donor`; the
acceptor bubbles for panel A live in `fig3_ratio.tsv` under a different y column and
a different strong-site threshold. An earlier draft of the contract listed only the
former and panel A could not be built.

All `fig3_*.tsv` come from a prior `scripts/figure3_data.py` run.

`center` is the CPA/polyA site, not the acceptor. `chrom` is chr-prefixed.

---

## 3. Scripts

| File | Role | In git |
|---|---|---|
| `scripts/figure3_data.py` | produces every `fig3_*.tsv` (pre-existing, unmodified) | see §7 |
| `figure3_final.py` | SpliceAI scan, holdout, crossover statistic | **no** |
| `make_figure3_final.R` | assembles the figure, reads only tsvs | **no** |
| `spliceai_grid.py` | provenance: selected the locked encodings | **no** |
| `spliceai_anchor_check.py` | provenance: first-pass check | **no** |

`spliceai_grid.py` and `spliceai_anchor_check.py` do not run in the figure path.
They are what justify the constants hard-coded in `figure3_final.py`
(`EXPECT_ACC_MAX = 0.740`, `EXPECT_DON_SUM = 0.675`, and the choice of max-over-2kb
for acceptor and sum-over-2kb for donor). Delete them and the encoding choice
becomes unexplained.

No pipeline file was modified.

---

## 4. SpliceAI data

UCSC SpliceAI Wildtype track (`spliceAIWt`), hg38, strand-specific. The documented
paths `/gbdb/hg38/spliceAIWt/` and `/gbdb/hg38/bbi/spliceAIWt/` both 404. Real
location:

```
https://hgdownload.soe.ucsc.edu/gbdb/hg38/bbi/spliceAi/wildtype/
```

Fetched 2026-07-21, server mtime Oct 1 2024. UCSC publishes no version tag or
checksum. Size plus a hash of the first 1 MiB is the strongest available pin; these
are **not** full-file checksums.

| File | Bytes | md5 of first 1 MiB |
|---|---|---|
| `spliceAiAcceptorPlus.bw` | 870196921 | `692edfaae854bcf81eeff2254c70eebc` |
| `spliceAiAcceptorMinus.bw` | 868392666 | `bc0a5113302bf75e333060631fe3e450` |
| `spliceAiDonorPlus.bw` | 865449909 | `1f3de517f4a6c675b715d8e86478f91e` |
| `spliceAiDonorMinus.bw` | 862256321 | `5867c5da30c4d5eb0354cbc2d1c6733e` |

```bash
mkdir -p spliceai_bw && cd spliceai_bw
for f in spliceAiAcceptorPlus spliceAiAcceptorMinus spliceAiDonorPlus spliceAiDonorMinus; do
  wget -c https://hgdownload.soe.ucsc.edu/gbdb/hg38/bbi/spliceAi/wildtype/$f.bw
done
```

`--remote` streams instead, but UCSC drops connections long before a 70k-peak run
finishes. Local copies are the supported path.

---

## 5. Environment

```
python    3.11.5      (/home/spour98/splice_env2)
numpy     2.4.2
pyBigWig  0.3.25
R         4.5.0
ggplot2   4.0.3   dplyr 1.2.1   readr 2.2.0   patchwork 1.3.2   scales 1.4.0
R libs    /home/spour98/R/x86_64-pc-linux-gnu-library/4.5
OS        Linux 5.14.0-687.25.1.el9_8.x86_64
```

`sklearn` is deliberately not a dependency. AUROC is an inline Mann-Whitney with
midrank tie handling.

### module load no-ops in a non-interactive shell

On this cluster `module` is not a shell function in a non-interactive shell.
`module load r/4.5.0` returns success, prints nothing, and leaves `PATH` unchanged,
so `Rscript` stays missing. Sourcing
`/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/bash` makes `module
avail` and `module list` work but still does not add R to `PATH`. Call the binary
directly:

```
/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/r/4.5.0/bin/Rscript
```

It resolves `~/R/x86_64-pc-linux-gnu-library/4.5` on its own, so all five packages
are found. In an interactive shell `module load r/4.5.0` works normally.

---

## 6. Commands, in order

```bash
cd /scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub

# 0. UPSTREAM, already done; exact arguments unknown, see gaps
#    python scripts/figure3_data.py --shortlist <?> --reference_fasta <?> \
#        --u1_lookup <?> --me5_lookup <?> --out fig3

# 1. Fetch the four bigWigs (section 4)

# 2. Compute: one scan, ~1.4 min at 12 threads
python figure3_final.py fig3_ucube_meta.tsv --threads 12

# 3. Plot
/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/r/4.5.0/bin/Rscript \
    make_figure3_final.R . fig3
```

Determinism: the gene holdout is `md5("0:" + gene) & 1`. Python's `hash()` is
avoided because it is salted per process. No RNG. Reruns are byte-identical.

`make_figure3_final.R` reads only tsvs and rescans nothing, so figure edits are
cheap. `E_MIN` (default 200) overrides the cell-count floor for F and G.

---

## 7. Outputs and checksums

```
1bc7cf13e507c9de98b088d3dfb4a64b  figure3_final.py
a64fbaebda5aeb08771eebfda1d4d0b0  make_figure3_final.R

52052f3d6e146d480722f028a8146355  sa_auroc_window_final.tsv    10 rows
5142ec7108cdcf150d256ca0c8e8f79c  sa_acc_x_don_final.tsv       56 rows
a8c009b473d38d663e04463de5472010  sa_tertile_final.tsv          6 rows
55dc9a56fb4ab831fcccbb430aef4072  sa_marginal_final.tsv         5 rows
cf21527afd1ac8bcd5dfdf2102909c8d  sa_metagene_final.tsv        50 rows
```

`figure3_final.pdf` 124035 B, `figure3_final.png` 1956611 B.

| Output | Feeds |
|---|---|
| `sa_auroc_window_final.tsv` | panel E curves |
| `sa_acc_x_don_final.tsv` | panel G tiles |
| `sa_tertile_final.tsv` | panel G callout, figure caption |
| `sa_marginal_final.tsv` | panel E annotation box |
| `sa_metagene_final.tsv` | written, **no panel consumes it** |

---

## 8. Built-in verification

`figure3_final.py` refuses to write output if the coordinate frame is wrong:

- Recomputes `sa_acc_max` and **exits non-zero** if neither the pooled nor the test
  value lands within 0.02 of 0.740. Under `--limit` this drops to a warning, because
  the expectation is a full-table number and a subsample drifts past tolerance on
  noise alone (a 4000-peak sample gave 0.7158 on an earlier encoding).
- Same check on `sa_don_sum` as a warning.
- Reports peaks dropped for missing coverage. Full run: 0 for both signals.
- Warns if the strong-acceptor tertile donor AUROC is not below 0.5, which is the
  condition the figure's central claim depends on.

Observed on the full run: `sa_acc_max` pooled 0.7400 / test 0.7369, `sa_don_sum`
test 0.6754. These match the values `spliceai_grid.py` produced independently, and
that agreement is the load-bearing check: the two scripts fetch different genomic
spans (2000 nt vs 2500 nt, the latter to cover the metagene) and still agree, so
the wider fetch did not shift the coordinate frame.

Chrom-edge clipping maps the fetched slice onto its exact array offsets rather than
padding blindly, so a short read at a chromosome end cannot silently shift the
frame.

---

## 9. Figure construction notes

- **Panels F and G use independent symmetric fill limits** (F ±1.818, G ±4.958).
  A shared scale was tried and rejected: the two panels differ in dynamic range by
  about 2.7x, so one scale rendered F nearly white while its own spread is 3.2 log2
  units. Both subtitles carry the range and the line "Panels F and G share the
  pattern, not the scale."
- **Two colour systems.** In fill panels (A, B, C, F, G) blue/orange means class:
  blue = enriched for true ends, orange = competitors, stated in the F and G legend
  title. In line panels (D, E) colour means signal, not class, stated in both
  subtitles.
- **Panel D is oriented** (U1 and 5'ss as 1 - AUROC); **panel E is raw**. Underneath,
  MaxEnt donor raw is 0.4516, below chance and competitor-predicting, while SpliceAI
  donor raw is 0.6754, above chance. Both donor curves plot above 0.5 but point in
  opposite raw directions.
- Panel G deciles collapse to **9 bins, not 10**: `sa_acc_max` has enough tied
  near-zero values that two quantile edges coincide.

---

## 10. Gaps

**1. Nothing is in version control.** `git status` shows the whole `reanal/` tree
untracked. HEAD is `f6dc96d` and contains none of these scripts or outputs. No
commit pins this result.

```bash
git add reanal/clean/pub/{figure3_final.py,spliceai_grid.py,spliceai_anchor_check.py} \
        reanal/clean/pub/make_figure3_final.R \
        reanal/clean/pub/REPRODUCIBILITY_figure3_final.md
echo 'reanal/clean/pub/spliceai_bw/' >> .gitignore   # 3.4 GB
```

**2. The MaxEnt half is not reproducible from this document.** The exact
`figure3_data.py` invocation is unknown; it needs `--shortlist`,
`--reference_fasta`, `--u1_lookup`, `--me5_lookup`. The `fig3_*.tsv` files were
consumed as given. This is the largest hole in the chain.

**3. bigWig checksums are partial** (first 1 MiB) and UCSC provides no version tag.
A silent upstream refresh would not be caught by size alone.

**4. SpliceAI is corroboration, not an independent test.** It integrates 10 kb of
context and was trained on annotated splice sites, so it is not independent of the
annotation defining the canonical labels. Stated in the figure caption.

**5. The low-acceptor reversal is asserted, not explained.** At low acceptor the
donor effect runs the other way (donor AUROC 0.651 in the weak tertile). The
reading offered is that no exon is defined there so donor signal only indicates the
region is spliced. That is an interpretation, not a tested claim.
