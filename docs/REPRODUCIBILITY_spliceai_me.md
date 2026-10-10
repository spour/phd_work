> **SUPERSEDED. Do not use.** Replaced by `REPRODUCIBILITY.md`, which documents
> the merged Figure 3 (`figure3_merged.png`). This file describes an earlier figure
> and, in the case of the spliceai_me document, an earlier acceptor encoding
> (`mean` over 2 kb, AUROC 0.7451) that was replaced by `max` over 2 kb (0.7369).
> Kept only as a record of what was built and why it changed.

# Reproducibility: MaxEnt vs SpliceAI comparison (figure3_spliceai_me)

Documents everything the figure `figure3_spliceai_me.png` / `.pdf` and its headline
numbers depend on. Written 2026-07-21.

Working directory for every command below:

```
/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub
```

---

## 1. What is being claimed

| Number | Value | Where it comes from |
|---|---|---|
| MaxEnt acceptor marginal | 0.606 | `fig3_bcshow_auroc.tsv`, peak over windows (oriented) |
| MaxEnt 5'ss marginal | 0.548 oriented / **0.452 raw** | `fig3_bcshow_auroc.tsv`, peak (predicts competitor) |
| SpliceAI acceptor, held-out | **0.7451** | `figure3_spliceai_me.py`, test half |
| SpliceAI donor, held-out | **0.6754** | `figure3_spliceai_me.py`, test half |
| Base rate P(true end) | 0.09308 | `fig3_class_counts.tsv` (6583 / 70723) |

Encodings are **locked**: `sa_acc` = nan-aware MEAN acceptor probability over the
2000 nt upstream window; `sa_don` = nan-aware SUM donor probability over the same
window. Both strand-oriented with index 0 at the CPA site.

---

## 2. Dependency graph

### Runtime dependencies (needed to regenerate the figure)

```
                        figure3_data.py   [UPSTREAM PIPELINE, not written here]
                               |
        +----------------------+----------------------+-------------------+
        |                      |                      |                   |
fig3_ucube_meta.tsv   fig3_bcshow_auroc.tsv   fig3_class_counts.tsv   fig3_acc_x_interv.tsv
        |                      |                      |                   |
        +----------+-----------+----------+-----------+                   |
                   |                      |                               |
                   v                      v                               |
        spliceai_bw/*.bw (4) --> figure3_spliceai_me.py                    |
                                          |                               |
                   +----------------------+                               |
                   |                                                      |
       auroc_window_spliceai_me.tsv                                       |
       acc_x_don_spliceai_me.tsv       ---------> make_figure3_spliceai_me.R <--+
       marginal_auroc_spliceai_me.tsv                       |
       metagene_spliceai_me.tsv  [written, NOT read]        v
                                              figure3_spliceai_me.png / .pdf
```

Exact reads, verified by grep, not assumed:

- `figure3_spliceai_me.py` reads `fig3_ucube_meta.tsv` (positional),
  `fig3_bcshow_auroc.tsv`, `fig3_class_counts.tsv`, and the four bigWigs.
  It does **not** read `fig3_acc_x_interv.tsv` or `fig3_metagene.tsv`.
- `make_figure3_spliceai_me.R` reads `fig3_bcshow_auroc.tsv`,
  `fig3_acc_x_interv.tsv`, `fig3_class_counts.tsv`, and three of the four
  `*_spliceai_me.tsv` files.
- `metagene_spliceai_me.tsv` is produced but **no panel consumes it**. The
  requested layout is 2x2 with no metagene panel.

### Provenance-only dependencies (NOT needed to rerun, but the result is not honest without them)

These two scripts do not execute in the figure path. They are what justify the
choices hard-coded into `figure3_spliceai_me.py`, and the result is only defensible
if they are kept alongside it.

| Script | What it fixed | Constant it justifies |
|---|---|---|
| `spliceai_anchor_check.py` | First pass: max acceptor over 2 kb beats MaxEnt | baselines 0.737 / 0.648 quoted in the narrative |
| `spliceai_grid.py` | 162-cell grid with gene holdout **selected** mean-over-2kb (acceptor) and sum-over-2kb (donor) | `EXPECT_SA_ACC = 0.745`, `EXPECT_SA_DON = 0.675` sanity gates |

If you delete `spliceai_grid.py`, the figure still builds, but the encoding choice
becomes an unexplained magic constant and the held-out claim is unsupported.

---

## 3. Scripts

| File | Role | Written for this analysis | In git |
|---|---|---|---|
| `scripts/figure3_data.py` | Produces all `fig3_*.tsv` MaxEnt inputs | no, pre-existing | see §7 |
| `spliceai_anchor_check.py` | First-pass SpliceAI check | yes | **no** |
| `spliceai_grid.py` | Encoding grid search + gene holdout | yes | **no** |
| `figure3_spliceai_me.py` | Compute side of the figure | yes | **no** |
| `make_figure3_spliceai_me.R` | Plot side of the figure | yes | **no** |

No pipeline file was modified by this analysis.

---

## 4. Data inputs

### SpliceAI bigWigs

UCSC SpliceAI Wildtype track (trackDb name `spliceAIWt`), hg38, strand-specific.
Directory (found by listing; the paths `/gbdb/hg38/spliceAIWt/` and
`/gbdb/hg38/bbi/spliceAIWt/` both 404):

```
https://hgdownload.soe.ucsc.edu/gbdb/hg38/bbi/spliceAi/wildtype/
```

Fetched to `spliceai_bw/` on 2026-07-21. Server-side mtime Oct 1 2024.
UCSC publishes no version tag or checksum for these, so the size plus a hash of the
first 1 MiB is the strongest pin available. Full-file md5 was skipped as too slow
over 3.4 GB; **these are not full-file checksums**.

| File | Bytes | md5 of first 1 MiB |
|---|---|---|
| `spliceAiAcceptorPlus.bw` | 870196921 | `692edfaae854bcf81eeff2254c70eebc` |
| `spliceAiAcceptorMinus.bw` | 868392666 | `bc0a5113302bf75e333060631fe3e450` |
| `spliceAiDonorPlus.bw` | 865449909 | `1f3de517f4a6c675b715d8e86478f91e` |
| `spliceAiDonorMinus.bw` | 862256321 | `5867c5da30c4d5eb0354cbc2d1c6733e` |

To refetch:

```bash
mkdir -p spliceai_bw && cd spliceai_bw
for f in spliceAiAcceptorPlus spliceAiAcceptorMinus spliceAiDonorPlus spliceAiDonorMinus; do
  wget -c https://hgdownload.soe.ucsc.edu/gbdb/hg38/bbi/spliceAi/wildtype/$f.bw
done
```

Remote streaming (`--remote`) works for small batches but UCSC refuses connections
well before a 70k-peak run finishes. Local copies are the supported path.

### MaxEnt inputs

All produced by `scripts/figure3_data.py`, prefix `fig3`:

| File | Rows | Used by |
|---|---|---|
| `fig3_ucube_meta.tsv` | 70723 | peak table: gene, chrom, center, strand, is_canonical, acc |
| `fig3_bcshow_auroc.tsv` | 120 | panel A curves; MaxEnt donor marginal |
| `fig3_acc_x_interv.tsv` | 65 | panel C |
| `fig3_class_counts.tsv` | 2 | base rate, n labels |

`center` is the CPA/polyA site, **not** the acceptor. `chrom` is chr-prefixed.

---

## 5. Environment

```
python    3.11.5      (/home/spour98/splice_env2)
numpy     2.4.2
pyBigWig  0.3.25
R         4.5.0       module r/4.5.0
ggplot2   4.0.3       dplyr 1.2.1   readr 2.2.0   patchwork 1.3.2   scales 1.4.0
R libs    /home/spour98/R/x86_64-pc-linux-gnu-library/4.5
OS        Linux 5.14.0-687.25.1.el9_8.x86_64
```

`sklearn` is deliberately **not** a dependency. AUROC is an inline Mann-Whitney
with midrank tie handling in both Python scripts.

Note on running R: `module load r/4.5.0` works in an interactive shell. In a
non-interactive shell Lmod may not be a shell function, in which case call the
binary directly:

```
/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/r/4.5.0/bin/Rscript
```

---

## 6. Commands, in order

Step 0 is upstream and was not run in this analysis; see §7 for the gap.

```bash
cd /scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub

# 0. UPSTREAM (already done; exact arguments unknown, see gaps)
#    python scripts/figure3_data.py --shortlist <?> --reference_fasta <?> \
#        --u1_lookup <?> --me5_lookup <?> --out fig3

# 1. Fetch bigWigs (see section 4)

# 2. Provenance: first-pass check           -> stdout only
python spliceai_anchor_check.py fig3_ucube_meta.tsv

# 3. Provenance: grid search + gene holdout -> selects the locked encodings
python spliceai_grid.py fig3_ucube_meta.tsv --threads 12
#    -> spliceai_grid.tsv, spliceai_auroc_vs_window.tsv, spliceai_acc_x_donor.tsv

# 4. Compute side of the figure             (~1.7 min scan, 12 threads)
python figure3_spliceai_me.py fig3_ucube_meta.tsv --threads 12
#    -> auroc_window_spliceai_me.tsv, acc_x_don_spliceai_me.tsv,
#       marginal_auroc_spliceai_me.tsv, metagene_spliceai_me.tsv

# 5. Plot side
module load r/4.5.0 && Rscript make_figure3_spliceai_me.R . fig3
#    -> figure3_spliceai_me.pdf, figure3_spliceai_me.png
```

Determinism: the gene holdout is `md5("0:" + gene) & 1`, seeded and stable across
runs and platforms. Python's built-in `hash()` is deliberately not used because it
is salted per process. No RNG anywhere. Reruns are byte-identical.

---

## 7. Outputs and checksums

Scripts:

```
e6fdd6cba189d2618fd1bd8bebdd6dd9  spliceai_anchor_check.py
40da8dd29907c86627c7df6589daaf0e  spliceai_grid.py
9b2683b7d0461e5bf849ffef9924bc4a  figure3_spliceai_me.py
90990a27fe8de8974efbb79af77635df  make_figure3_spliceai_me.R
```

Data outputs:

```
ef1dd928b33320b83eaf0d8ec3246f15  auroc_window_spliceai_me.tsv
1faa36b055cd565e6adbe313dc45fb7c  acc_x_don_spliceai_me.tsv
0459c622a63296622bb8b1a53747a036  marginal_auroc_spliceai_me.tsv
cf21527afd1ac8bcd5dfdf2102909c8d  metagene_spliceai_me.tsv
9067f846b79301fe2895d334bcf3ac8f  spliceai_grid.tsv
be2bb79acad452054fe00f0d8f238f8d  spliceai_auroc_vs_window.tsv
80c35710c107d55df93353135dd4c1c7  spliceai_acc_x_donor.tsv
```

---

## 8. Built-in verification

`figure3_spliceai_me.py` refuses to write output if the coordinate frame is wrong:

- Recomputes `sa_acc` mean-2kb on the test half and **exits non-zero** if it is more
  than 0.02 from 0.745. Under `--limit` this is downgraded to a warning, because the
  expectation is a full-table number and a subsample drifts past tolerance on noise
  alone (a 4000-peak sample gave 0.7158).
- Same check on `sa_don`, as a warning.
- Reports peaks dropped for missing coverage (0 on the full run).

Observed on the full run: acceptor 0.7451, donor 0.6754, exactly matching the values
`spliceai_grid.py` independently produced. This is the load-bearing check: the two
scripts fetch different genomic spans (grid fetches 2000 nt upstream; the figure
script fetches 2500 nt to cover the metagene) and still agree to four decimals, so
the wider fetch did not shift the frame.

`spliceai_grid.py` cross-checks against `spliceai_anchor_check.py`: donor max-2kb
0.6550 vs 0.648, acceptor max-2kb 0.7400 vs 0.737. The residual drift is a known
1 bp difference in the minus-strand window convention between the two scripts
(`(center, center+2000]` vs `[center, center+2000)`), not an error.

---

## 9. Gaps and caveats

**1. Nothing here is in version control.** `git status` reports the entire
`reanal/` tree as untracked, including all four scripts written for this analysis
and every output. HEAD is `f6dc96d` and contains none of it. No commit pins this
result. Fix before publication:

```bash
git add reanal/clean/pub/{spliceai_anchor_check,spliceai_grid,figure3_spliceai_me}.py \
        reanal/clean/pub/make_figure3_spliceai_me.R \
        reanal/clean/pub/REPRODUCIBILITY_spliceai_me.md
echo 'reanal/clean/pub/spliceai_bw/' >> .gitignore   # 3.4 GB of bigWigs
```

**2. The upstream MaxEnt half is not reproducible from this document.** The exact
`figure3_data.py` invocation is unknown: it requires `--shortlist`,
`--reference_fasta`, `--u1_lookup`, `--me5_lookup`, and the `fig3_*.tsv` files were
consumed as given. Anyone rebuilding from raw data must recover those four paths.
This is the largest hole in the chain.

**3. bigWig checksums are partial** (first 1 MiB), and UCSC provides no version tag.
A silent upstream refresh of the track would not be caught by size alone.

**4. Panel D does not support its title across the whole panel.** The claim "at
fixed acceptor a stronger donor marks the competitor" holds in acceptor deciles 6-9
and **reverses** in deciles 1-5:

```
acc decile 9: log2OR +4.21 -> +1.40   holds
acc decile 8: +4.82 -> -0.25          holds
acc decile 7: +2.11 -> -0.53          holds
acc decile 6: -0.82 -> -2.09          holds
acc decile 5: -1.01 -> -0.16          REVERSES
acc decile 4: -1.44 -> +0.64          REVERSES
acc decile 3: -1.39 -> -0.07          REVERSES
acc decile 2: -2.27 -> -0.13          REVERSES
acc decile 1: -2.42 -> -0.28          REVERSES
```

It is a crossover interaction, not a main effect. Plausible reading: at low acceptor,
adding donor signal indicates the region is spliced at all and raises the odds
slightly; at high acceptor, adding donor means "internal exon" and lowers them. The
mechanism of interest lives in the high-acceptor regime, which is where true ends
are, but the title as printed asserts it unconditionally.

**5. Panels A and B use opposite donor conventions.** A is oriented as published
(MaxEnt 5'ss 0.548); B is raw as specified (SpliceAI donor 0.675). Underneath,
MaxEnt donor raw is 0.4516, i.e. below chance and competitor-predicting, while
SpliceAI donor raw is above chance. The two donor curves both plot above 0.5 but
point in opposite raw directions. Legend and subtitle say so; a skimming reader may
still misread it as agreement.

**6. Deciles collapse to 9 bins, not 10**, because `sa_acc` has enough tied
near-zero values that two quantile edges coincide.

**7. SpliceAI is corroboration, not an independent test.** It integrates 10 kb of
context and was trained on annotated splice sites, so it is not independent of the
annotation that defines the canonical labels. Stated on the figure.

**8. Panels C and D share one fill limit** (max of the two 95th percentiles,
`oob = squish`, limit 4.217) so the two conditioning panels are visually comparable.
This was a judgment call, not a specified requirement.
