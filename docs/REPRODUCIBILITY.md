# Reproducibility: Figure 3 (MaxEnt and SpliceAI)

Authoritative record for `figure3_merged.png` / `.pdf` and every number on it.
Written 2026-07-21.

Supersedes `REPRODUCIBILITY_spliceai_me.md` and `REPRODUCIBILITY_figure3_final.md`,
which remain on disk with a SUPERSEDED banner as a record of the earlier figures.
Everything still current is carried forward here. **Do not quote numbers from
them**: the first documents an acceptor encoding of `mean` over 2 kb (AUROC
0.7451), which was replaced by `max` over 2 kb (0.7369) after the grid search.

Superseded artifacts still present in the directory, kept for provenance and not
inputs to anything current: `figure3_spliceai_me.py`, `make_figure3_spliceai_me.R`,
`figure_summary*.py/R`, `*_spliceai_me.tsv`, `figure3_spliceai_me.*`,
`figure_summary*.png/pdf`, `figure3_final.*`. The grid outputs
`spliceai_grid.tsv`, `spliceai_auroc_vs_window.tsv` and `spliceai_acc_x_donor.tsv`
are the record of the encoding search described in section 2 and are worth
keeping.

Working directory for every command:

```
/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub
```

---

## 1. The claims and where each number comes from

The figure argues three things. Each rests on a different statistic, and they are
not interchangeable.

| Claim | Evidence | Panels |
|---|---|---|
| A strong acceptor 3'ss marks the real 3' end | OR 1.46/SD (MaxEnt), 2.22/SD (SpliceAI) | A, D, E, I |
| At fixed acceptor, a stronger 5'ss lowers the odds | OR 0.912 and 0.606 per SD | F, G, I |
| The real end outranks its own competitors | within-gene win-rate 0.587 / 0.672 | I |

### stats_forest.tsv, the load-bearing table

| block | effect | scorer | estimate | 95% CI | p | n |
|---|---|---|---|---|---|---|
| effect | acceptor 3'ss | MaxEnt | 1.4640 | 1.4271, 1.5018 | 1e-188 | 70723 |
| effect | acceptor 3'ss | SpliceAI | 2.2215 | 2.1723, 2.2718 | < 1e-300 | 70723 |
| effect | donor 5'ss \| acceptor | MaxEnt | 0.9119 | 0.8910, 0.9332 | 5e-15 | 70340 |
| effect | donor 5'ss \| acceptor | SpliceAI | 0.6061 | 0.5835, 0.6296 | 6e-147 | 70723 |
| effect | acceptor x donor | MaxEnt | 0.9963 | 0.9737, 1.0194 | **0.749** | 70340 |
| effect | acceptor x donor | SpliceAI | 0.7975 | 0.7623, 0.8343 | 9e-23 | 70723 |
| ranking | within-gene win-rate | MaxEnt | 0.5865 | 0.5771, 0.5959 | < 1e-300 | 6583 |
| ranking | within-gene win-rate | SpliceAI | 0.6718 | 0.6628, 0.6807 | 1e-221 | 6583 |

Odds ratios are per SD of a z-scored predictor, from IRLS logistic fits on all
peaks with Wald standard errors. Base rate P(real end) = 0.09308 (6583 / 70723).

**The interaction is scorer-dependent.** Significant for SpliceAI, null for
MaxEnt (OR 0.996, p = 0.749). The shared, robust claim is the acceptor main
effect plus the conditional donor suppression. Do not write that both scorers
show an interaction. The figure caption states this.

### Three statements that sound alike and are not

This tripped up the analysis repeatedly. Keep them distinct in the text.

1. **Level**: in the high-acceptor band of panel G, the minimum log2 odds ratio
   is **+0.0015**. Those cells fall to the base rate and stop. They never become
   competitor-enriched. Panel F behaves the same way at high acceptor
   (+1.82 down to +0.06 along the top row).
2. **Ranking**: within the strong-acceptor tertile, donor AUROC is **0.4141**
   (competitor direction, held out). Canonicals carry lower donor than
   competitors there. That is a ranking statement, not a level statement.
3. **Regression**: `donor | acceptor` OR < 1 in both scorers. This is the claim
   the figure title rests on.

All three are true at once. A title saying the site "becomes a competitor" is
contradicted by (1), which a reader can check by eye. The wording settled on is
"competes it back toward baseline".

### 0.737 test vs 0.740 pooled

Same encoding, two numbers. 0.7400 is `sa_acc_max` AUROC over all 70,723 peaks;
**0.7369** is the held-out half. Panel E plots held-out AUROC and its acceptor
curve ends at 0.737, so the annotation prints 0.737. Printing 0.740 beside a
point drawn at 0.737 would contradict the plot.

---

## 2. Files: what to run

```bash
cd /scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub

# 1. stats + reshapes. Reuses the SpliceAI cache; no genome scan.
python figure3_merged.py fig3_ucube_meta.tsv

# 2. the figure
/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/r/4.5.0/bin/Rscript \
    make_figure3_merged.R . fig3
```

Runtime is seconds. Step 1 touches bigWigs only if `sa_peak_scalars_final.tsv`
is missing, in which case it rescans (about 1.5 min at 12 threads) and rewrites
the cache.

### Scripts

| File | Role | Needed to rebuild the figure |
|---|---|---|
| `scripts/figure3_data.py` | upstream, produces all `fig3_*.tsv` (unmodified) | yes, already run |
| `figure3_merged.py` | all stats and reshapes | yes |
| `make_figure3_merged.R` | plots all nine panels | yes |
| `spliceai_grid.py` | selected the locked SpliceAI encodings | no, but see below |
| `spliceai_anchor_check.py` | first-pass SpliceAI check | no |
| `figure3_final.py` / `make_figure3_final.R` | earlier 8-panel figure | no, superseded |
| `joint_score_final.py` | joint-score test (negative result) | no |
| `rule_2x2_final.py` | decision-rule 2x2 (negative result) | no |

`spliceai_grid.py` does not run in the figure path but is what justifies the
encodings hardcoded in `figure3_merged.py`: acceptor = **max** over 2 kb
upstream, donor = **sum** over 2 kb upstream. It searched 162 encodings with a
gene holdout and found the train-selected winner equals the test-selected winner
(gap 0.0000), so the choice is not a search artifact. Delete it and the
encodings become unexplained constants.

### Negative results worth keeping

Both are real findings that shaped the figure. Do not redo them.

- **Decision-rule 2x2** (`rule_2x2_final.py`): "strong acceptor AND no downstream
  5'ss" captures 80.7% of real ends at 10.1% precision, only **1.08-fold** over
  base rate, because `acc >= 8.0` admits 91% of peaks. No threshold makes it a
  usable classifier (precision never clears 28%). Binarizing destroys the
  gradient the AUROC and the logistic use. This is why there is no 2x2 panel.
- **Joint score** (`joint_score_final.py`): `z(acc) - z(donor)` **hurts** SpliceAI
  (0.737 to 0.680, delta -0.057) and is neutral for MaxEnt (+0.004). SpliceAI's
  donor is positively predictive marginally (0.675), so subtracting it removes
  signal. A linear difference cannot represent an acceptor-conditional sign flip.
  This is why the figure shows the interaction rather than a composite score.

---

## 3. Dependency contract

`figure3_merged.py` checks all of these at startup and exits 1 naming each
missing path. MaxEnt is never recomputed.

| Required input | Feeds |
|---|---|
| `fig3_ratio.tsv` | **panel A** acceptor bubbles (y = `maxscore`, thr 8.0) |
| `fig3_bcshow_ratio.tsv` | panels B, C (y = `yval`, thr 4.0; `u1` and `donor` only) |
| `fig3_bcshow_auroc.tsv` | panel D curves, peak-AUROC subtitles, panel E number box |
| `fig3_acc_x_interv.tsv` | panel F |
| `fig3_class_counts.tsv` | base rate, n labels |
| `fig3_ucube_meta.tsv` | per-peak gene, is_canonical, acc, interv_donor, coords |
| `sa_acc_x_don_final.tsv` | panel G |
| `sa_auroc_window_final.tsv` | panel E curves |
| `sa_tertile_final.tsv` | the 0.414 stat in G's subtitle |
| `sa_peak_scalars_final.tsv` | per-peak SpliceAI scalars (cache; rescan if absent) |
| `fig3_acc_paired_summary.tsv` | optional; within-gene MaxEnt stat |
| `spliceai_bw/*.bw` (4) | only if the cache is absent |

**`fig3_ratio.tsv` and `fig3_bcshow_ratio.tsv` are both required and are not
interchangeable.** `bcshow_ratio` contains only `u1` and `donor`; the acceptor
bubbles for panel A live in `fig3_ratio.tsv` under a different y column and a
different strong-site threshold. An earlier draft listed only the former and
panel A could not be built.

---

## 4. SpliceAI data

UCSC SpliceAI Wildtype track (`spliceAIWt`), hg38, strand-specific. The
documented paths `/gbdb/hg38/spliceAIWt/` and `/gbdb/hg38/bbi/spliceAIWt/` both
404. Real location:

```
https://hgdownload.soe.ucsc.edu/gbdb/hg38/bbi/spliceAi/wildtype/
```

Fetched 2026-07-21, server mtime Oct 1 2024. UCSC publishes no version tag or
checksum. Size plus a hash of the first 1 MiB is the strongest pin available;
**these are not full-file checksums**.

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

Remote streaming works for small batches but UCSC refuses connections well
before a 70k-peak run finishes. Local copies (3.4 GB) are the supported path.

### The coordinate convention, and why it is checked

Per peak, 2000 nt upstream, strand-oriented so index 0 sits at the CPA site and
increasing index moves upstream:

```
+ strand: values(chrom, center-2000, center) reversed   # i = center-1-g
- strand: values(chrom, center, center+2000) as-is      # i = g-center
```

`center` is the CPA/polyA site, **not** the acceptor. `chrom` is chr-prefixed.
Chrom-edge clipping maps the fetched slice onto its exact array offsets rather
than padding blindly, so a short read at a chromosome end cannot shift the frame.

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

`sklearn` and `scipy` are deliberately not dependencies. AUROC (Mann-Whitney with
midranks), logistic regression (IRLS with Wald SEs), and the Wilcoxon signed-rank
test are all inline.

### module load no-ops in a non-interactive shell

On this cluster `module` is not a shell function in a non-interactive shell.
`module load r/4.5.0` returns success, prints nothing, and leaves `PATH`
unchanged, so `Rscript` stays missing. Sourcing
`/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/bash` makes
`module avail` and `module list` work but still does not add R to `PATH`. Call
the binary directly:

```
/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/r/4.5.0/bin/Rscript
```

It resolves `~/R/x86_64-pc-linux-gnu-library/4.5` on its own. In an interactive
shell `module load r/4.5.0` works normally.

---

## 6. Determinism and verification

The gene 50/50 holdout is `md5("0:" + gene) & 1`. Python's built-in `hash()` is
deliberately avoided because it is salted per process. No RNG anywhere. Reruns
are byte-identical.

Verification built into the scripts:

- `figure3_merged.py` verifies `sa_peak_scalars_final.tsv` matches the peak table
  in both row count and gene order before joining by index, and exits rather than
  silently misaligning.
- `figure3_final.py` recomputes `sa_acc_max` and exits non-zero if neither the
  pooled nor the held-out value lands within 0.02 of 0.740. Under `--limit` this
  drops to a warning, since the expectation is a full-table number and a
  subsample drifts past tolerance on noise alone.
- It also warns if the strong-acceptor tertile donor AUROC is not below 0.5,
  which is the condition the figure's conditional claim depends on.

**The load-bearing cross-check**: `spliceai_grid.py` and `figure3_final.py` fetch
different genomic spans (2000 nt vs 2500 nt, the latter to cover a metagene) and
still agree to four decimals on 0.7369 / 0.6754. That agreement, not the
tolerance thresholds, is what rules out a coordinate-frame shift.

---

## 7. Checksums

Scripts:

```
df6957139478a3a8e452540cf47f8fd8  figure3_merged.py
1f4687b54e195883307441158140d25f  make_figure3_merged.R
40da8dd29907c86627c7df6589daaf0e  spliceai_grid.py
e6fdd6cba189d2618fd1bd8bebdd6dd9  spliceai_anchor_check.py
02c711205bca20f1461bc7e0f122bbdb  joint_score_final.py
3d9bd6d800bc3be99bf1fd2e829a3114  rule_2x2_final.py
1bc7cf13e507c9de98b088d3dfb4a64b  figure3_final.py
a64fbaebda5aeb08771eebfda1d4d0b0  make_figure3_final.R
```

Data:

```
03aa798d3971958d0e78ca5358849784  stats_forest.tsv
473fa14052fd2840b8c7bb8f9d9ee032  heat_maxent.tsv
dc570a9fa36b51e73ecea6390be5cf3c  heat_spliceai.tsv
7f254e359797b99d66d9dd81e7471d0f  sa_peak_scalars_final.tsv
5142ec7108cdcf150d256ca0c8e8f79c  sa_acc_x_don_final.tsv
52052f3d6e146d480722f028a8146355  sa_auroc_window_final.tsv
a8c009b473d38d663e04463de5472010  sa_tertile_final.tsv
55dc9a56fb4ab831fcccbb430aef4072  sa_marginal_final.tsv
```

`figure3_merged.pdf` 128349 B, `figure3_merged.png` 2300456 B.

---

## 8. Figure construction notes

- **Panels F and G use independent symmetric fill limits** (F ±1.818, G ±4.958).
  A shared scale was tried and rejected: the panels differ in dynamic range by
  about 2.7x, so one scale rendered F nearly white while its own spread is 3.2
  log2 units. Both subtitles carry their range and the note that the scales are
  not comparable.
- **Two colour systems.** In fill panels (A, B, C, F, G) blue/orange encodes
  class: blue = enriched for real ends, orange = competitors. In line panels
  (D, E) colour encodes signal, not class, stated in both subtitles. In the
  forest (I) points are greyscale and encode scorer.
- **Panel D is oriented** (U1 and 5'ss as 1 - AUROC); **panel E is raw**.
  MaxEnt donor raw is 0.4516, below chance and competitor-predicting, while
  SpliceAI donor raw is 0.6754, above chance. Both donor curves plot above 0.5
  in their respective panels but point in opposite raw directions.
- Panel G deciles collapse to **9 bins, not 10**: `sa_acc_max` has enough tied
  near-zero values that two quantile edges coincide.
- Callout anchors are computed from the data (argmax/argmin among cells with
  n >= 200) so they cannot drift onto empty cells. G's blue callout sits at
  donor decile 6 of 9, low relative to what is populated in that row but not low
  in absolute terms: high-acceptor cells with a genuinely weak donor barely
  exist.

---

## 9. Gaps

**1. Nothing is in version control.** `git ls-files reanal/` returns 0. HEAD is
`f6dc96d` and contains none of this. No commit pins the result.

```bash
git add reanal/clean/pub/*.py reanal/clean/pub/*.R \
        reanal/clean/pub/REPRODUCIBILITY.md reanal/clean/pub/CLAUDE.md
echo 'reanal/clean/pub/spliceai_bw/' >> .gitignore   # 3.4 GB
```

**2. The MaxEnt half is not reproducible from this document.** The exact
`figure3_data.py` invocation is unknown; it needs `--shortlist`,
`--reference_fasta`, `--u1_lookup`, `--me5_lookup`. The `fig3_*.tsv` files were
consumed as given. This is the largest hole in the chain and only the author can
close it.

**3. bigWig checksums are partial** (first 1 MiB) and UCSC provides no version
tag. A silent upstream refresh would not be caught by size alone.

**4. SpliceAI is corroboration, not an independent test.** It integrates 10 kb of
context and was trained on annotated splice sites, so it is not independent of
the annotation that defines the canonical labels. Stated in the figure subtitle.

**5. `internal_exon_like` cannot be reconstructed per peak.** The published class
fractions in `fig3_internal_exon_like.tsv` come from `fig3_first5_from_anchor.tsv`
at `[50,300]` inclusive (reproduces 0.163755 / 0.194855 to six decimals), but
that file carries only `cls` and `value`: no gene key, no `acc`, and it is
filtered to 47,336 of 70,723 rows. `rule_2x2_final.py` therefore derives its
`has_dn5` from `nearest_dn`, a different quantity (distance downstream of the
peak's own site, versus distance from the anchor 3'ss in both directions), giving
0.1525 / 0.1805. Its cell counts will not reconcile with the published fractions.

**6. The low-acceptor reversal is described, not tested.** At low acceptor the
donor effect runs the other way (donor AUROC 0.651 in the weak tertile). The
reading offered is that no exon is defined there, so donor signal only indicates
the region is spliced. That is an interpretation.

---

## Appendix (2026-07-27): U1 recalibration to 5.25 — environment and cache validation

Added during the U1 threshold recalibration (Tasks A/B). Records the traps and the
validations so a later session does not repeat them.

**Environment.** `/home` had an NFS fault during this work that corrupted the
`~/splice_env2` numpy and broke shell init; a cvmfs scipy-stack Python was used as a
fallback with a pure-Python Faidx reader in place of pysam. `/home` and
`~/splice_env2` (numpy 2.4.2, pandas 3.0.0) were healthy again by 2026-07-27 and are
the stack used for the analyses below. A separate transient `libgomp: Thread creation
failed` (node thread/process exhaustion, not a mount fault) is worked around by
running single-threaded: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`.

**Upstream pipeline reproduces exactly.** `scripts/figure3_data.py` (invocation in
`scripts/slurm_figure3_data.sh`, `~/splice_env2`, `--imbalanced --max_per_class 70000`,
seed 0, `PYTHONHASHSEED=0`) was re-run to `--out fig3_new`. `fig3_new_bcshow_ratio.tsv`
and `fig3_new_bcshow_auroc.tsv` are byte-identical to the committed `fig3_*`, and
`fig3_new_denswin.npz` has the same md5 as `fig3_denswin.npz`. The environment does not
change upstream results.

**`u1_window_cache.npz` alignment (built by `task_b2_ties.py`).** Row-for-row aligned to
`fig3_ucube_meta.tsv` (70,723 peaks); gene-label match to meta is 1.000000. Pushing its
`dens_4.0` column through the within-gene engine reproduces the known gene-averaged
win-rates to 4 dp (ties=0.5 0.6187; sign-test 0.6200). **All `dens_{t}` columns
(4.0/5.25/6.5/7.0/8.0) are produced in one scan loop by the same `score_window`,
differing only by the constant `t` in `(v>=t).sum()` — so the `dens_4.0` validation
transfers to `dens_5.25` by construction.**

**The 0.834 exact-match with `u1_grid_cache` `up_500|density_per_kb@4` is NOT a rounding
or boundary-truncation effect.** All 70,723 windows are full (`npos==492`, 100%). The
difference is one-sided: the grid cache counts 0–8 *more* U1(negΔG≥4) sites than
`dens_4.0`, never fewer (83.4% exact, 89.3% within 1 count). The maximum difference is
exactly 8 = (9-mer length − 1) and the sign is constant, which is the signature of a
9-mer-boundary-inclusion convention difference (the grid scan counts 9-mers that
straddle the upstream window edge; `dens_*` counts only 9-mers whose sliding-window
start lies inside the 500-nt window). It is rank-preserving within gene, hence the 4-dp
reproduction above. Task C retires the grid cache for U1 (F/G/I/J use `dens_5.25`; panel
B uses a fresh 40-window scan at 5.25), so the offset does not enter the final figure.

---

## Appendix (2026-07-27, cont.): U1 suppression layer REFUTED by matched-decoy control

The two-layer model's U1/telescripting layer does not survive a matched intronic-decoy
control and is dropped. Figure 3 becomes a one-layer model (acceptor competence) plus a
rigorous negative. Provenance: `task_b8/b10/b13/b16/b16b/b18/b19.py`.

**The negative (task_b13, task_b16).** Decoys: intronic positions in the same gene,
>=1 kb from any of 569,005 PolyASite atlas clusters (`../train/data/atlas.chr.bed`),
matched to each real competitor on distance-to-canonical-pA, nearest 3'ss, and nearest
5'ss/donor (tightened match balances all three to ~145 nt median). Competitor-vs-matched-
decoy within-gene win-rate (ties=0.5, null 0.500 permutation-confirmed): **u1_excess
0.5025** (dead null); U1@5.25 0.475. Placebo (canonical-vs-decoy) reproduces the real
canonical-vs-competitor contrast (u1_excess +0.047 both). Robust to repeat/low-complexity
matching (0.507) and to 1/2/5 kb exclusion radii (0.499/0.503/0.511). The apparent U1
signal is compartment + a local depletion at the used canonical site, not CPA-specific.

**The mechanism (task_b19).** A sharp LOCAL U1/GC dip at the pA (-150..0 nt; canonical
-50 bin U1 15.6 vs ~30 elsewhere, GC 0.338 vs 0.44), present in canonical AND competitor,
ABSENT in decoys (flat ~29.6). GC anti-correlates with AT-downstream (-0.45) and strong-PAS
(-0.22). **pA-proximal U1 density is a negative proxy for poly-A architecture (AT-rich
hexamer + U/GU-rich DSE depressing a GC-coupled hybridization score), not U1 occupancy.**

**What survives (task_b17, task_b18).** (i) Competitors ARE architecturally competent CPA
sites: vs matched intronic decoys, strong-PAS win-rate 0.903, GU/UG-DSE 0.654 (CPA-specific,
survives the control U1 failed). (ii) Layer 1 (acceptor) is intact: pooled OR MaxEnt 1.464 /
SpliceAI 2.221; survives same-compartment matching (TERM-TERM win-rate 0.457, CI excludes
0.5); the decoy refutation is structurally degenerate for it because 92.4% of decoys share
the competitor's anchor 3'ss (acceptor strength is architecture-referenced, not window
composition).

**9-mer convention blast radius (B12.3).** The straddling-9-mer (grid-cache) convention was
used only by: the abandoned U1-threshold/FDR analysis (`finalize_u1_threshold.py`,
`u1_threshold.tsv`: 185.8/peak@4.0, 11.7/peak@7.0, 15.9x) and the hero F/G/I/J @4 tables
(`figure3_hero_data.py`). All are dropped (U1 refuted). Every surviving U1 number uses the
`dens_*` sliding-window (starts-inside) convention; the two conventions never coexist in a
manuscript number.

**GC interpretation (B12.1).** The within-gene GC axis is largely poly-A architecture (per
task_b19); a smaller component survives compartment-matching (+0.090) as an unexplained
composition axis. GC is a descriptive covariate, reported with its own observed|null|excess,
not a mechanism.

## Appendix (2026-07-28): REVERSAL — U1 layer reinstated as COMPARTMENT-level suppression

This supersedes the framing of the 2026-07-27 "REFUTED" appendix. The matched-decoy result
stands, but it is a **specificity limit**, not a refutation. Figure 3 is two-layer again.

**Layer 2 = compartment-level suppression.** Intronic CPA competitors sit in a U1-rich,
splicing-competent compartment; terminal-exon 3' ends do not. Evidence (primary cutoff 5.25,
`task_rev_numbers.py` from `peak_compartment.npy` + `u1_window_cache.npz`):
- marginal within-gene win-rate U1@5.25 **0.594 (+0.094)**, composition-corrected u1_excess **+0.047**;
- TERM-TERM (both windows terminal/3'UTR, 1,943 genes) U1@5.25 **+0.049 [.529,.570]**,
  u1_excess **+0.034 [.513,.555]**, both CI exclude 0.5;
- the B19 within-gene paired metagene (`paired_u1_excess_525.tsv`): competitors carry +5.8
  sites/kb more pA-proximal U1, CI above zero in all 20 upstream bins.

**Specificity limit (the decoy result, now text not refutation).** pA-proximal U1 is NOT a
per-site suppression mark: real intronic competitors carry no more composition-corrected U1
than matched non-CPA intronic decoys in the same gene (competitor-vs-decoy u1_excess win-rate
**0.4996**). So the claim is compartment-level, not site-ranking. **Scope of the
compartment-controlled evidence: tandem 3'UTR APA** (the TERM-TERM stratum).

**GC is a covariate, not matched.** Within gene the competitor window is GC-richer (win-rate
0.619, TERM-TERM +0.099) while pooled between-class GC is flat (+0.001). Reported with its own
numbers; nothing is GC-matched in the figure.

**Rebuild (U1 at 5.25, no MaxEnt touched).** `task_rev_scan.py` (one U1 positional pass, all
70,723 peaks, pA-2000..pA+500) → panel B `fig3rev_bcshow_ratio/auroc.tsv` (peak oriented AUROC
0.547 @450nt) and panel F `paired_u1_excess_525.tsv` / `paired_u1_metagene_525.tsv`.
`task_rev_hero.py` (cached `dens_5.25`) → panels G/I/J `acc_x_u1_525.tsv`,
`hero_within_gene_525.tsv`, `forest_u1_525.tsv` (U1 OR/SD 0.848, within-gene 0.594, interaction
OR 1.10). `make_figure3_v2.R` restored to the two-layer A–J layout, U1 panels reframed
compartment-level. The 4.0 hero tables and `bcshow_ratio.tsv` u1 rows are retained but no longer
drawn.

## Appendix (2026-07-28): figure3_v3_2 -- "terminal exon definition" reframe

A reframe (not a rebuild) of Figure 3. Real 3' end = terminal exon (strong 3'ss + pA, no
downstream donor); competitor with a strong 5'ss between its 3'ss and pA = internal exon
(splices through, pA bypassed). `make_figure3_v3_2.R` reuses acceptor bubbles, the MaxEnt
acceptor x intervening-5'ss heat (`fig3_acc_x_interv.tsv`), the held-out SpliceAI donor
tertile forest (`sa_tertile_final.tsv`: weak 0.651 / medium 0.531 / strong 0.414 competitor
direction), and `stats_forest.tsv`, plus three NEW within-gene analyses (`task_v32_analyses.py`;
gene-averaged, permutation null over within-gene label shuffles, excess over 0.5, bootstrap CI).
Donor scoring uses the dense `me5_lookup` array, verified byte-identical to maxentpy `score5`
(not a re-scan of the Figure-3 grid).

1. **Ordered geometry (A1, `v32_A1_geometry.tsv`).** Within-gene win-rate that the competitor
   carries a strong 5'ss (>=8): interval (anchor 3'ss, pA) **+0.0476 [.540,.555], perm p=0.002**;
   interval (pA, pA+2000) **+0.0006 [.495,.506], perm p=0.85 (null)**. The donor contrast is in
   the defined exon and absent downstream -- the asymmetry exon definition predicts.
2. **Exon-size dependence (A2, `v32_A2_exonsize.tsv`).** Donor excess by anchor->donor distance:
   <50 +0.0025 (p=0.09), 50-150 +0.0141, 150-300 +0.0080, 300-1000 **+0.0173**, >1000 +0.0057
   (p<=0.004 for the last four). MIXED: present across 50-1000 nt, weakest at the anchor,
   strongest at 300-1000 nt -- NOT sharply tuned to the 50-300 nt internal-exon window. A generic
   intervening-donor-at-intron-scale reading is not excluded; reported plainly.
3. **Decoy architecture (A3, `v32_A3_decoy_arch.tsv`).** Fraction with a strong 5'ss within 2 kb
   upstream: real end 0.782, competitor 0.823, intronic decoy 0.826 (downstream ~0.81 for all).
   Decoys are donor-flanked like competitors, so B13's placebo (canonical-vs-decoy ~
   canonical-vs-competitor) is PREDICTED by exon definition, not a contradiction.

Panel F metagene (`v32_F_metagene.tsv`): strong-5'ss density is depleted just upstream of the pA
for real ends only (the terminal exon has no donor); competitors and decoys are not, upstream-only.
U1 is demoted to the caption as a proxy for donor-rich intronic territory (within-gene win-rate
0.594 at 5.25), not occupancy. GC not matched.

### figure3_v3_2 fixes (2026-07-28)

**Panel C verdict.** The cached `sa_tertile_final.tsv` (0.651/0.531/0.414) is CORRECT. Protocol
(from `figure3_final.py`, reproduced): tertile edges from SpliceAI acceptor prob (`sa_acc_max`)
on the TRAIN gene-split (md5, HOLDOUT_SEED=0), donor = SUM over 2 kb upstream, AUROC on the
held-out TEST set, canonical positive. Reproduced byte-for-byte (seed 0: 0.6506/0.5312/0.4141,
n 12213/11756/11785). The earlier recompute (0.649/0.655/0.672) used MaxEnt-acceptor tertiles on
ALL data -- a different, cross-model statistic; it is flat because MaxEnt acceptor and SpliceAI
donor are independent, whereas the crossover is intrinsic to SpliceAI. The gradient is driven by
the tertile VARIABLE, not the holdout (SpliceAI-acc on all data: 0.639/0.547/0.413). Panel C
unchanged.

**Length-matched D + distance (`task_v32_fix2.py`, `v32_D_lengthmatched.tsv`, `v32_G_distance.tsv`).**
Both intervals rescored with the same me5 scan; downstream length-matched to (pA, pA+d), d = the
peak's anchor->pA distance.
- presence: (anchor,pA) +0.0466 (p=0.002) ; (pA,pA+d) length-matched +0.0018 (p=0.63, null) ;
  (pA,pA+2000) current +0.0006 (p=0.85). Downstream stays null even length-matched -- not a ceiling.
- distance-to-nearest-strong-donor (ceiling-free): (anchor,pA) +0.0413 (p=0.002) ;
  (pA,pA+d) matched +0.0132 (p=0.010). A small downstream residual remains, ~3x smaller than
  upstream. The asymmetry survives length-matching.
- G distance: nearest upstream strong 5'ss (gene-median) real 981 nt > competitor 830 nt ~
  decoy 810 nt. Decoys match competitors, confirming the presence result.
