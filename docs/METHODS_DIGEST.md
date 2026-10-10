# METHODS_DIGEST.md

Reconstruction of the Methods for a thesis chapter, mapped from the analysis
directory to the 8 chapter figures.

- **Root:** `/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean`
  (also reachable as `/home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean`)
- **Compiled:** 2026-08-31, read-only inspection. Nothing was executed or modified.
- **Convention:** every parameter is quoted verbatim with `path:line`. Where a
  number could not be found in any file, that is stated explicitly as **NOT FOUND**.

## Reading this document

The repository does **not** use the thesis's `0.1 … 0.8` numbering anywhere. Its
own figure names are `figure1`, `figure2`, `supp_fig1`, `figure_overlap_filtered`,
and so on, and there is a *different* numbering scheme in
`pub/scripts/fig1_8_model_timeline.R:2` (`# fig1_8_model_timeline.R -- Figure 1.8`,
a literature timeline that is **not** one of the eight figures). The mapping below
was established by matching the distinctive strings supplied in the brief against
file contents, not by trusting any filename.

| Thesis figure | Repo artifact | Plotting script | Data script |
|---|---|---|---|
| 0.1 data setup | (no single figure file; schematic) | **not found** | `pub/scripts/featurization.py`, `pub/scripts/make_splits.py`, `pub/scripts/score_genes.py` |
| 0.2 models | (no single figure file; schematic) | **not found** | `pub/scripts/cpa_cnn.py`, `pub/scripts/train_{rf,lr,cnn}.py` |
| 0.3 curation | `pub/supp_fig1.pdf` / `.png` | `pub/sfig1.R` | `pub/scripts/select_constitutive_cpa.py` |
| 0.4 performance | `pub/figure1.pdf` / `.png` | `pub/scripts/make_figure1.R` | `pub/scripts/make_figure1_data.py` |
| 0.5 competitor features | `pub/figure2.pdf` / `.png` | `pub/scripts/make_figure2.R` | `pub/scripts/figure2_stats.py`, `pub/scripts/featurize_shortlist_cpa.py` |
| 0.6 reporter schema | — | **NOT FOUND** (no wet-lab code in tree) | **NOT FOUND** |
| 0.7 gel | — | **NOT FOUND** (no wet-lab code in tree) | **NOT FOUND** |
| 0.8 perturbation overlap | `pub/figure_overlap_filtered.pdf` / `.png` | `pub/make_figure_overlap.R` | `pub/task_overlap_accounting.py`, `pub/scripts/03_simple_overlap.py` |

---

# 1. Repository tree (depth ≤ 4, abridged)

Full recursion is ~1,400 text/code files plus large binaries. Directories are
listed with entry counts; only code-bearing subtrees are expanded.

```
clean/                                    305 entries  (figure PDFs/PNGs, TSV/parquet outputs, slurm .out/.err)
├── scripts/                              169   PRIOR-GENERATION pipeline (May–Jul 2026)
│   ├── puffin/                             8   Puffin-style CPA model experiments
│   ├── end_choice/                         4
│   ├── logs/                                   slurm logs
│   └── __pycache__/
├── pub/                                 1394   FINAL pipeline + analyses (Jun 29 – Aug 31 2026); a git repo
│   ├── .git/                                   913 tracked files, HEAD 674764d
│   ├── scripts/                          123   the locked pipeline (see pub/CLAUDE.md "Never edit the pipeline")
│   │   └── __pycache__/
│   ├── scripts2/                           8   fig3 variants (R)
│   ├── data/                               6   Dominant_CPA_baseline_total.bed, negatives_*.bed, HumanPas/
│   │   └── HumanPas/                       2   hg38.PAS.main.tsv, hg38.PAS.max.tsv  (PolyA_DB)
│   ├── models/                             6   fold0..fold4 + prod, each {rf,lr,cnn}/metadata.json + model
│   ├── splits/                             6   fold0..fold4 + prod, each {pos,neg}_{train,val,test}.{bed,fa} + features/
│   ├── folds/                                  fold_assignments.tsv, *_chroms.txt, run_make_splits.sh
│   ├── genebeds/                           6   genes_fold{0..4}.bed, genes_prod.bed
│   ├── scans/                              7   per-gene parquet scans: fold0..4, crossfit, prod
│   ├── external/                          10   geo/, lifted/, peaks/, chain/, bin/, apa_2018/ (vendored Leslie-lab R)
│   ├── spliceai_bw/                            4 UCSC SpliceAI bigWigs (3.4 GB, gitignored)
│   ├── figs/                              12   fig1..fig5, fig_composite (a separate composite, not the 8)
│   ├── final/                             13   counts.json, per_site.tsv, headline.tsv, …
│   ├── figdata_final/                          frozen rerank figure data
│   ├── hiconf_out/, cpa_out/, cpa_out_isect/, expoverlaps/, te/, tesrc/, geom/, mech/,
│   │   grad/, fold/, folds/, gate/, go/, master/, modelcmp/, resolve/, seed/, stab/,
│   │   subfam/, tefig/, tenull/, wb/, ctx/, context/, xlip_out/, atlaste/, splmodel/
│   └── (≈240 top-level .py/.R analysis scripts + their .tsv/.pdf/.png outputs)
├── train/                                 23   PRIOR-GENERATION training area
│   ├── data/                              31   hg38.fa, atlas.clusters.2.0.GRCh38.96.tsv, polyadb_v4/, annotation/
│   │   ├── annotation/                     4   gencode.v49.annotation.gtf, terminal_exons.bed, three_prime_utr.bed, three_prime_zone.bed
│   │   ├── polyadb_v4/HumanPas/            4
│   │   ├── splits/                             the ORIGINAL (fold=30) splits
│   │   ├── te_dataset/, te_features/, end_choice/
│   ├── splice_train/scripts/              65   splicing-model bakeoff (not used by the 8 figures)
│   ├── {rf,lr,cnn,te,puffin}_runs/              per-run metadata for older models
│   └── gene_scans/per_gene/                    older APARENT2-era per-gene scans
├── figures/                                4   legacy PNGs (FGF20_with_aparent2, fig1_ranks_*)
├── figdata_final/                         35   frozen rerank-figure data (Jun 28)
├── clean_final/                            0   EMPTY
├── data/                                   7   te_dataset*, te_features
├── logs/                                 169   aparent2_scan_summary.json, rescan_v2_resume_summary.json, …
├── cisbp_rna/, logos_all_motifs/, tomtom_used/, tomtom_suppressed/
└── (top-level analysis outputs: shortlist_*.parquet, rerank_*.tsv, threess_*, fivess_*, cnn_*, …)
```

**Environment / dependency manifests:** there is **no** `environment.yml`,
`requirements.txt`, `renv.lock`, `sessionInfo`, `Snakefile`, `*.smk`, or
`pyproject.toml` anywhere in the tree. Configuration lives in two YAML files
(`pub/datasets.yaml`, `pub/hiconf.yaml`), two JSON run-configs
(`pub/cpa_out/config_used.json`, `pub/cpa_out_isect/config_used.json`), and prose
(`pub/REPRODUCIBILITY.md` §5). Orchestration is SLURM (`*.slurm`, `slurm_*.sh`,
`submit_all.sh`), not Snakemake.

---

# 2. Per-figure dossiers

## Figure 0.1 — data setup

### (a) Plotting script

**NOT FOUND.** No script in the tree draws a data-setup schematic. The only script
whose *content* covers this material is `scripts/make_figure1.py` (210 KB,
prior-generation), which contains at `scripts/make_figure1.py:2728`:

```
      A. PAS hexamer density (AATAAA + 11 common variants)
```

but it is a multi-figure composite generator for the older "CPCCPA suppression"
story (`scripts/make_figure1.py:1-40`), not the chapter's Figure 0.1. Treat the
figure as hand-drawn.

### (b) Upstream data-generating scripts

| Step | Script |
|---|---|
| positive set (constitutive CPA) | `pub/scripts/select_constitutive_cpa.py` |
| chromosome splits | `pub/scripts/make_splits.py`; wrapper `pub/scripts/make_prod_split.sh`; 5-fold `pub/scripts/assign_folds.py` |
| FASTA extraction | `pub/scripts/extract_fastas.sh` |
| feature matrix | `pub/scripts/featurization.py` (driver `pub/scripts/featurize_all.slurm`) |
| genome-wide scan | `pub/scripts/score_genes.py` (driver `pub/scripts/score_array.slurm`) |

### (c) Exact parameters (verbatim)

**Window = 500 nt, CPA site at the midpoint (i.e. −250/+250).**

- `pub/scripts/featurization.py:212` — `    macro_L, macro_stride = 500, 1`
- `pub/scripts/cpa_cnn.py:12` — `    is at position 250, so position matters), unlike global max pooling.`
- `pub/scripts/score_genes.py:19` — `    is_constitutive_cpa  : bool  # window where the canonical CPA site sits at +250`
- `pub/scripts/score_genes.py:237` — `    # The constitutive window has the CPA base at offset 250.`
- `pub/scripts/score_genes.py:247` — `            canonical_window_start = cpa_offset - 250`
- Confirmed on disk: `pub/splits/prod/pos_train.bed:1` = `chr1	943953	944453	NOC2L	ENST00000327044.7	-` (500 nt).

**Binning inside the window (RF/LR features).**

- `pub/scripts/featurization.py:120` — `    micro_L, micro_stride = 30, 10`
- `pub/scripts/featurization.py:121` — `    n_bins = ((L - micro_L) // micro_stride) + 1  # 48`
- `pub/scripts/featurization.py:213` — `    micro_L, micro_stride = 30, 10`

→ 48 bins of 30 nt at stride 10 across the 500 nt window.

**One-hot encoding.**

- `pub/scripts/featurization.py:70` — `_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}`
- `pub/scripts/featurization.py:73-79` — `def one_hot(seq): … arr = np.zeros((len(seq), 4), dtype=np.uint8)` (N → all-zero row)
- CNN one-hot: `pub/scripts/cpa_cnn.py:86` — `_BASE_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}`
- `pub/scripts/cpa_cnn.py:90` — `    """Return (4, seq_len) one-hot. Ns / unknowns -> all zeros at that pos."""`

**UGUA and the engineered k-mer families.**

- `pub/scripts/featurization.py:114` — `ENG_FAMILIES = ["CA_UA", "DSEs", "poly_U_kmers", "poly_U_kmer", "UGUA_kmers"]`
- `pub/scripts/featurization.py:167` — `    UGUA_kmers = count_pattern([3, 2, 3, 0])`   (T,G,T,A = UGUA)
- `pub/scripts/featurization.py:163-166`:
  ```
      poly_U_kmers = (count_pattern([3, 0, 3, 3, 3, 3])
                      + count_pattern([3, 2, 3, 3, 3, 3])
                      + count_pattern([3, 3, 3, 3, 3, 3]))
      poly_U_kmer = count_pattern([3, 3, 3, 3])
  ```
- `pub/scripts/featurization.py:136-138` — `CA = (Cc[:, :-1] & A_next).sum(axis=1)` / `TA = …` / `CA_UA = CA + TA`
- `pub/scripts/featurization.py:150` — `    DSEs = G_count + T_count + GG + GT + TG + TT + TTT + TGG + TGT + GTG`

**PAS hexamer.** Two different hexamer sets are used in the project; they are not
interchangeable and both are quoted here.

- Feature-matrix PWM route (Fig 0.1/0.2): three PAS PWMs are read from an external
  list, `/scratch/spour98/identify_model/pwm_paths` (default at
  `pub/scripts/featurize_all.slurm:16` — `PWM_LIST="${PWM_LIST:-/scratch/spour98/identify_model/pwm_paths}"`).
  The file (outside the analysis root) contains 10 PWMs:
  `DSE_1..4_PWM.txt, NUDT_PWM.txt, PAS_Hu.txt, PAS_PWM.txt, PAS_Siepel.txt, PolyU_PWM.txt, UGUA_Hu.txt`.
- Explicit 13-hexamer list (used by Figure 0.5's density panels):
  `pub/scripts/featurize_shortlist_cpa.py:52-54`
  ```
  # canonical PAS hexamer + the 11 most common single-substitution variants (Beaudoing 2000)
  PAS_HEXAMERS = ["AATAAA","ATTAAA","TATAAA","AGTAAA","AAGAAA","AATATA",
                  "AATACA","CATAAA","GATAAA","AATGAA","TTTAAA","ACTAAA","AATAGA"]
  ```
- Two-hexamer "strong PAS" set used throughout the mechanism analyses:
  `pub/hiconf_sets.py:43` — `HEX = ("AATAAA", "ATTAAA")`;
  `pub/task_b8.py:47-50` — `# standard canonical PAS hexamer set (Tian/Gruber); AATAAA & ATTAAA are the strong two` / `PAS_STRONG=("AATAAA","ATTAAA")`;
  `pub/task_gc_model.py:40` — `PAS_STRONG = ("AATAAA", "ATTAAA"); PAS_LO, PAS_HI = -60, 0`

**Feature-matrix dimensions.** 15 families × 48 bins = **720 columns**.
Verified from `pub/models/prod/lr/metadata.json` → `winner_meta.columns` has 720
entries over 15 distinct families
(`CA_UA, DSE_1..4_PWM.txt, DSEs, NUDT_PWM.txt, PAS_Hu.txt, PAS_PWM.txt, PAS_Siepel.txt, PolyU_PWM.txt, UGUA_Hu.txt, UGUA_kmers, poly_U_kmer, poly_U_kmers`),
and `pub/models/prod/lr/metadata.json` `model_stats.n_features = 720`.

**PWM log-odds background.**
`pub/scripts/featurization.py:278` — `p.add_argument("--bg", nargs=4, type=float, default=[0.3, 0.2, 0.2, 0.3])`
(used as `--bg 0.3 0.2 0.2 0.3` in `pub/scripts/featurize_all.slurm:28` and
`pub/scripts/score_array.slurm:38`).
`pub/scripts/featurization.py:46` — `    return 10.0 * np.log10(odds)`; `eps=1e-4` at line 41.
`pub/scripts/featurization.py:58` — PWMs longer than `max_len=20` are skipped.

**Negative set and class balance.**

- `pub/scripts/make_splits.py:62-66`:
  ```
      ap.add_argument("--fold",             type=int, default=40,
                      help="target negatives per positive")
      ap.add_argument("--decoy_per_total",  type=int, default=7,
                      help="1 decoy per N total negs (7 -> decoy:nondecoy = 1:6)")
      ap.add_argument("--seed",             type=int, default=42)
  ```
- Decoy = PAS-containing negative, non-decoy = no PAS:
  `pub/scripts/make_splits.py:7-8`
  ```
      --neg_decoy_bed       : PAS-containing negatives (e.g. negatives_w_pa_clean.bed)
      --neg_nondecoy_bed    : non-PAS negatives (e.g. negatives_no_pa_clean.bed)
  ```

**Chromosome holdout (production split).** `pub/scripts/make_prod_split.sh:5`:

```
#   train = chr1,2,4-12,14-17   val = chr3,13   test = chr18-22,X   (chrY excluded)
```
Written literally at `pub/scripts/make_prod_split.sh:17-19`.

**5-fold cross-fit holdout.** Whole chromosomes, balanced by positive count:
`pub/scripts/assign_folds.py:7-12` (`Greedy longest-processing-time bin packing`),
`pub/scripts/assign_folds.py:29` — `ap.add_argument("-k", "--folds", type=int, default=5)`.
For fold *k*: `test = fold k, val = fold (k+1) mod 5, train = the rest`
(`pub/scripts/assign_folds.py:76-82`). Realised assignment is in
`pub/folds/fold_assignments.tsv` (chr1/7/X/13 → fold 0, etc.).

**Strand-aware sequence extraction.** `pub/scripts/extract_fastas.sh:18` —
`      bedtools getfasta -s -name -fi "$GENOME" -bed "$bed" -fo "$fa"`, with
`GENOME=../train/data/hg38.fa` (`pub/scripts/extract_fastas.sh:9`).

### (d) Inputs and outputs

| Input | Path | Provenance |
|---|---|---|
| genome | `train/data/hg38.fa` (3.27 GB, mtime 2014-01-15), `.fai` | **hg38** (UCSC) |
| PolyA_DB PAS table | `pub/data/HumanPas/hg38.PAS.main.tsv`, `hg38.PAS.max.tsv` (mtime 2025-09-12) | version disputed — see §5 |
| PolyASite atlas | `train/data/atlas.clusters.2.0.GRCh38.96.tsv` → `train/data/atlas.chr.bed` (569,005 lines) | **PolyASite 2.0**, GRCh38, Ensembl 96 |
| annotation | `train/data/annotation/gencode.v49.annotation.gtf` (mtime 2025-09-02) | **GENCODE v49** |
| PWMs | `/scratch/spour98/identify_model/pwm_paths` → `BaselineModelPWMs/*.txt` | outside the analysis root |
| protein-coding genes | `train/data/pc_genes.bed` (20,042 lines) = `pub/genebeds/genes_prod.bed` | derived from GENCODE |

| Output | Path | Rows |
|---|---|---|
| positives | `pub/data/Dominant_CPA_baseline_total.bed` | **15,794** |
| positives, chrY dropped | `pub/data/Dominant_CPA_baseline_total.noY.bed` | 15,785 |
| PAS-bearing negatives | `pub/data/negatives_w_pa_clean.bed` | 604,551 |
| PAS-free negatives | `pub/data/negatives_no_pa_clean.bed` | 493,956 |
| prod split | `pub/splits/prod/{pos,neg}_{train,val,test}.bed/.fa` | see below |
| feature TSVs | `pub/splits/{fold*,prod}/features/{pos,neg}_{train,val,test}.tsv` | 720 cols |
| genome-wide scans | `pub/scans/{fold0..4,crossfit,prod}/per_gene/<GENE>.parquet` | 20,012 gene parquets in `prod` |

### (e) Every hard-coded number, with source

| Number | Where it is | Verdict |
|---|---|---|
| **15,794** | `wc -l pub/data/Dominant_CPA_baseline_total.bed` = 15794, **and printed in the run logs** | **RESOLVED.** Not a literal in any script — it is the row count of the positive BED, echoed at runtime. Log evidence: `logs/shortlist_45617229_0.out:1` (and `_2,_5,_7,_8,_11,_14,_15`) — `[canon] 15,794 canonical windows, 15,794 symbols`; `threess_prof_46083225.err:9`, `threess_prof_46084638.err:9`, `threess_prof_46120736.err:9`, `threess_prof_46241732.err:7`, `threess_prof_46267965.err:7`, `threess_prof_46078021.err:4` — `   all CPA              15,794`; `fivess_profile_46239406.err:2,38`, `fivess_profile_46260364.err:2,57`, `fivess_profile_46267928.err:2,57` — `   all CPA                15,794`. So the figure label is correct and is a canonical-site (= gene) count. Note the modelling set is **15,785** (`Dominant_CPA_baseline_total.noY.bed`) because chrY (9 sites) is excluded by the chromosome lists. |
| **376,057** | — | **NOT FOUND.** Exhaustive grep of *every* file in the tree — scripts, notes, TSVs, JSON, YAML **and all SLURM `.out`/`.err` run logs** — returns no `376,057` or `376057` (the only hits are digit-substrings inside floats such as `7.40613760576024`). Nearby real counts that a label could have been drawn from: `303,312` (PolyA_DB main BED lines), `311,500` (`train/data/human.PAS.hg38.slop250.bed`), `363,009` (`pub/splits/prod/neg_train_nondecoy.bed`), `429,849` (prod train negatives), `569,005` (PolyASite atlas clusters), `441,546` (prod train pos+neg). **Flag for the author.** |
| −250 / +250 | `pub/scripts/score_genes.py:19,237,247`; `pub/scripts/cpa_cnn.py:12` | **CONFIRMED** as the 500-nt window centred on the CPA base. |
| 500 (window) | `pub/scripts/featurization.py:212`; `pub/scripts/build_shortlist.py:111` (`--win`, default 500) | CONFIRMED |
| 40 (neg:pos target) | `pub/scripts/make_splits.py:62` | Realised ratio is **36.75**, not 40, because decoy/non-decoy pools run short on some chromosomes; recorded independently as `"class_ratio": 36.74865350089767` in `pub/models/prod/cnn/metadata.json`. |
| split sizes (prod) | `wc -l pub/splits/prod/*.bed` | train 11,697 pos + 429,849 neg; val 1,153 + 46,120; test **2,935 + 101,096 = 104,031** |

### (f) Statistical tests

None at this stage. Subsampling of negatives is the only stochastic step, seeded
at `pub/scripts/make_splits.py:66` (`--seed`, default 42) and applied via
`pub/scripts/make_splits.py:71` — `    rng = random.Random(args.seed)`.

---

## Figure 0.2 — models

### (a) Plotting script

**NOT FOUND.** No script draws a model-architecture schematic. The architecture
prose that a schematic would be drawn from is the docstring of
`pub/scripts/cpa_cnn.py:1-16`.

### (b) Upstream scripts

`pub/scripts/train_rf.py`, `pub/scripts/train_lr.py`, `pub/scripts/train_cnn.py`,
model definition `pub/scripts/cpa_cnn.py`; SLURM arrays
`pub/scripts/train_{rf,lr,cnn}_array.slurm`; orchestrator `pub/scripts/submit_all.sh`.

### (c) Exact hyperparameters (verbatim)

**Random forest** — `pub/scripts/train_rf.py:84-91`:
```
    ap.add_argument("--trees",     type=int, default=30000)
    ap.add_argument("--min_split", type=int, default=5)
    ap.add_argument("--min_leaf",  type=int, default=4)
    ap.add_argument("--criterion", default="log_loss")
    ap.add_argument("--class_weight", default="balanced")
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--n_jobs",    type=int, default=16)
    ap.add_argument("--mode_label", default="logloss_leaf4",
```
Calibration and threshold — `pub/scripts/train_rf.py:11-12`:
```
  3) Fit Platt (sigmoid) calibration on the external validation set
  4) Pick F1-optimal threshold on validation
```
`pub/scripts/train_rf.py:170` — `        calibrator = CalibratedClassifierCV(FrozenEstimator(rf), method="sigmoid")`
`pub/scripts/train_rf.py:178` — `    thresholds = np.linspace(0.05, 0.95, 91)`
Realised forest (from `pub/models/prod/rf/metadata.json`):
`"forest_stats": {"n_estimators": 30000, "avg_depth": 34.085966666666664, "avg_nodes": 7273.8234, "min_depth": 26, "max_depth": 53, "min_nodes": 5969, "max_nodes": 9081}`;
selected threshold `0.29`; `train_time_sec 7499.855143070221`.

**Logistic regression** — `pub/scripts/train_lr.py:6`:
```
    penalty='l1', C=0.1, tol=0.01, solver='saga', class_weight='balanced'
```
and `pub/scripts/train_lr.py:71-79`:
```
    ap.add_argument("--penalty",      default="l1")
    ap.add_argument("--C",            type=float, default=0.1)
    ap.add_argument("--tol",          type=float, default=0.01)
    ap.add_argument("--solver",       default="saga")
    ap.add_argument("--max_iter",     type=int, default=2000)
    ap.add_argument("--class_weight", default="balanced")
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--n_jobs",       type=int, default=8)
    ap.add_argument("--mode_label",   default="lr_l1_C0.1")
```
`pub/scripts/train_lr.py:160` — `    thresholds = np.linspace(0.05, 0.95, 91)`; selected threshold `0.9299999999999999`.
No calibration (`"calibration": "none"`). Realised:
`"n_nonzero_coefs": 720, "sparsity": 0.0, "intercept": -3.9746896370785935e-05, "n_iter": 37`.

**CNN** — architecture, `pub/scripts/cpa_cnn.py:28-59`:
```
        seq_len: int = 500,
        n_filters_1: int = 128,
        n_filters_2: int = 256,
        n_hidden: int = 128,
        dropout: float = 0.3,
        self.conv1 = nn.Conv1d(4, n_filters_1, kernel_size=8, padding="same")
        self.pool1 = nn.MaxPool1d(4)  # 500 -> 125
        self.conv2 = nn.Conv1d(n_filters_1, n_filters_2, kernel_size=8, padding="same")
        self.pool2 = nn.MaxPool1d(4)  # 125 -> 31
        self.conv3 = nn.Conv1d(n_filters_2, n_filters_2, kernel_size=4, padding="same")
        self.adaptive_pool = nn.AdaptiveMaxPool1d(4)  # -> (B, 256, 4)
        self.fc1 = nn.Linear(n_filters_2 * 4, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)
```
Each conv is followed by `BatchNorm1d` then `ReLU` (`pub/scripts/cpa_cnn.py:66-70`);
dropout is applied twice in the head (lines 73, 75).
`pub/scripts/cpa_cnn.py:15` — `~660k parameters.`; realised `"n_params": 661633`.

Training — `pub/scripts/train_cnn.py:121-144`:
```
    ap.add_argument("--epochs",       type=int,   default=40)
    ap.add_argument("--batch_size",   type=int,   default=256)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience",     type=int,   default=3)
    ap.add_argument("--num_workers",  type=int,   default=4)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--warmup_frac",  type=float, default=0.05,
    ap.add_argument("--grad_clip",    type=float, default=1.0)
    ap.add_argument("--label_smooth", type=float, default=0.1,
    ap.add_argument("--pos_weight",   type=float, default=1.0,
    ap.add_argument("--n_filters_1", type=int,   default=128)
    ap.add_argument("--n_filters_2", type=int,   default=256)
    ap.add_argument("--n_hidden",    type=int,   default=128)
    ap.add_argument("--dropout",     type=float, default=0.3)
```
`pub/scripts/train_cnn.py:201` — `    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,`
`pub/scripts/train_cnn.py:19` — `    * Cosine LR schedule with linear warmup (5% of total steps), in place of`
Loss/label smoothing: `pub/scripts/train_cnn.py:229` —
`                y_target = y * (1.0 - args.label_smooth) + 0.5 * args.label_smooth`
Realised (`pub/models/prod/cnn/metadata.json`):
```
"training": {"epochs": 40, "best_epoch": 3, "batch_size": 256, "lr": 0.001,
 "weight_decay": 0.0001, "optimizer": "AdamW", "scheduler": "CosineWithWarmup",
 "warmup_frac": 0.05, "criterion": "BCEWithLogitsLoss + binary label smoothing",
 "label_smooth": 0.1, "pos_weight": 1.0, "class_ratio": 36.74865350089767,
 "patience": 3, "grad_clip": 1.0, "amp": false, "train_time_sec": 1623.1410813331604}
```
Model selection `"selected_by": "best_val_auprc"`; selected threshold `0.36999999999999994`.

**Seeds.** `seed = 42` for all three (`train_rf.py:89`, `train_lr.py:77`, `train_cnn.py:127`);
`pub/scripts/train_cnn.py:52-56` seeds `random`, `numpy`, `torch`, `torch.cuda`.

**Actual invocations — every hyperparameter is passed explicitly, and matches the
defaults above.** The defaults are therefore not merely nominal; verified against
the three SLURM arrays (all `#SBATCH --array=0-5`, task i -> `(fold0..fold4, prod)`):

`pub/scripts/train_rf_array.slurm:25-32`:
```
python3 scripts/train_rf.py \
  --pos_train "$F/pos_train.tsv" --neg_train "$F/neg_train.tsv" \
  --pos_val   "$F/pos_val.tsv"   --neg_val   "$F/neg_val.tsv" \
  --pos_test  "$F/pos_test.tsv"  --neg_test  "$F/neg_test.tsv" \
  --outdir "$OUT" \
  --trees 30000 --min_split 5 --min_leaf 4 \
  --criterion log_loss --class_weight balanced \
  --seed 42 --n_jobs $SLURM_CPUS_PER_TASK --mode_label logloss_leaf4
```
(`--cpus-per-task=16`, `--mem=64G`, `--time=6:00:00`; `:9-10` — `# RF on featurized TSVs. Task i -> split NAMES[i]. Identical hyperparameters across` / `# folds and prod (prod reproduces the original RF).`)

`pub/scripts/train_lr_array.slurm:23-30`:
```
python3 scripts/train_lr.py \
  … --outdir "$OUT" \
  --penalty l1 --C 0.1 --tol 0.01 --solver saga --max_iter 2000 \
  --class_weight balanced --seed 42 --n_jobs $SLURM_CPUS_PER_TASK \
  --mode_label lr_l1_C0.1
```
(`--cpus-per-task=8`, `--mem=32G`, `--time=2:00:00`)

`pub/scripts/train_cnn_array.slurm:31-40`:
```
python3 scripts/train_cnn.py \
  --pos_train "$DIR/pos_train.fa" --neg_train "$DIR/neg_train.fa" \
  … --outdir "$OUT" \
  --seq_len 500 --epochs 40 --batch_size 256 --lr 1e-3 --weight_decay 1e-4 \
  --patience 3 --label_smooth 0.1 --pos_weight 1.0 --grad_clip 1.0 \
  --warmup_frac 0.05 --dropout 0.3 \
  --n_filters_1 128 --n_filters_2 256 --n_hidden 128 \
  --num_workers $SLURM_CPUS_PER_TASK --seed 42 --mode_label cnn_v2
```
(`--gres=gpu:h100:1`, `--cpus-per-task=8`, `--mem=32G`, `--time=8:00:00`; note
`--array=0-5` with **no `%N` throttle**, unlike the scoring array — see the AMP
finding below. Idempotence guard at `:26-28`: a fold is skipped if
`cnn_model.pt` exists and `metadata.json` already contains `"test_metrics"`.)

**RF calibration branch actually taken.** `pub/scripts/train_rf.py:169-172`:
```
    if _HAS_FROZEN:
        calibrator = CalibratedClassifierCV(FrozenEstimator(rf), method="sigmoid")
    else:
        calibrator = CalibratedClassifierCV(rf, cv="prefit", method="sigmoid")
```
The training environment carries **scikit-learn 1.2.1** (§3.2), and
`FrozenEstimator` was only added in scikit-learn 1.6, so `_HAS_FROZEN` is False
and the **`cv="prefit"`** branch ran. Methods should say Platt/sigmoid calibration
of a pre-fit forest on the external validation split, via `CalibratedClassifierCV(..., cv="prefit")`.

**The production CNN was trained on CPU, not on the requested H100.**
`pub/scripts/train_cnn.py:152` — `    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
and `:207` — `    use_amp = (device.type == "cuda") and (not args.no_amp)`.
`--no_amp` is never passed, so `amp=False` in the metadata can only mean the run
saw no CUDA device. Recorded per fold (`pub/models/*/cnn/metadata.json`):

| model | `amp` | `train_time_sec` | `best_epoch` |
|---|---|---|---|
| fold0 | True | 64 | 5 |
| fold1 | True | 68 | 6 |
| fold2 | True | 52 | 4 |
| fold3 | True | 53 | 4 |
| **fold4** | **False** | **1424** | 3 |
| **prod** | **False** | **1623** | 3 |

Direct log confirmation for the production model (array task 5):
`pub/logs/cnn_46303739_5.out` — `Device: cpu`, then
`[Training v2] epochs=40 bs=256 lr=0.001 wd=0.0001 amp=False`.
Folds 0–3 print `amp=True` and finish in ~1 minute; fold4 and prod take ~25x
longer. The likely cause is GPU contention: the CNN array runs all six tasks
concurrently (`--array=0-5`) while the *scoring* array deliberately throttles
(`score_array.slurm:3` — `#SBATCH --array=0-5%2`, with `:13-14` — `# %2 throttles to 2 concurrent tasks so array tasks never share a GPU (avoids the` / `# CUDA OOM seen when all 6 hit the GPU at once).`).

**Consequences to state in Methods:** (i) the CNN behind every Figure 0.4 number
was trained in fp32 on CPU, so it is *not* bit-reproducible from a GPU rerun even
at seed 42 (different RNG stream and different precision); (ii) fold4 is not
numerically comparable to folds 0-3 for the same reason; (iii) `prod` and `fold4`
stopped at `best_epoch` 3, folds 0-3 at 4-6. None of this is recorded in any
README - it is recoverable only from `metadata.json` and the SLURM logs.

**"P(CPA)".** The literal string appears once in the tree, in a *different* model
family (Puffin-style, prior generation): `scripts/puffin/puffin_cpa.py:23` —
`      P(CPA) = sigmoid(w_pas * a_pas + w_splice * a_splice + b)`. The three
production models emit `rf_prob` / `lr_prob` / `cnn_prob`
(`pub/scripts/make_figure1_data.py:48` — `COLS   = {"RF": "rf_prob", "LR": "lr_prob", "CNN": "cnn_prob"}`).
If the figure labels the output "P(CPA)" it is a display label, not a code symbol.

### (d) Inputs / outputs

Inputs: `pub/splits/{fold*,prod}/features/*.tsv` (RF, LR) and
`pub/splits/{fold*,prod}/*.fa` (CNN).
Outputs: `pub/models/{fold0..fold4,prod}/{rf,lr,cnn}/` — `metadata.json` plus
`rf_calibrated.joblib` + `rf_uncalibrated.joblib` (15.7 GB each),
`lr_model.joblib` + `lr_coefficients.tsv`, `cnn_model.pt`.

### (e) Hard-coded numbers

`128` filters — `pub/scripts/cpa_cnn.py:30`, `pub/scripts/train_cnn.py:141`,
`pub/models/prod/cnn/metadata.json` `"n_filters_1": 128` (and `"n_hidden": 128`).
`500` — `pub/scripts/cpa_cnn.py:30`, `train_cnn.py:119`, `"seq_len": 500`.
`kernel_size=8` (layers 1–2) and `kernel_size=4` (layer 3) are **not** in the
figure brief but are the other half of the architecture.

### (f) Statistics

None. Threshold selection is F1-maximisation on validation over a
`np.linspace(0.05, 0.95, 91)` grid for all three models; no test.

---

## Figure 0.3 — curation

### (a) Plotting script

`pub/sfig1.R` → `supp_fig1.pdf` / `supp_fig1.png`
(`pub/sfig1.R:137-138`, `ggsave("supp_fig1.pdf", fig, width=13, height=10, device=cairo_pdf, limitsize=FALSE)`).
Header, `pub/sfig1.R:2-3`:
```
# make_supp1.R -- Supplementary Figure 1: selection of constitutive polyA sites
# from PolyA DB v4 (hg38.PAS.main.tsv), and why the positive set is ~15.7k not 20k.
```
Panels A–D at `pub/sfig1.R:4-7`.

### (b) Upstream script

`pub/scripts/select_constitutive_cpa.py`. It writes `<out>.tsv` and `<out>.bed`
(`pub/scripts/select_constitutive_cpa.py:23-25`); panel D of the figure reads the
TSV (`pub/sfig1.R:11` — `TSV<-if(length(args)>=1) args[1] else "constitutive_cpa.tsv"`).

### (c) Exact parameters

`pub/scripts/select_constitutive_cpa.py:6-10`:
```
Method (as stated), per PolyA DB 3:
  significance = PSE x RPM   (percentage of samples expressed  x  mean reads per million)
  A site is selected for the positive set if it has the HIGHEST PSExRPM AND is
  located in (or at the end of) the 3'UTR of the LONGEST isoform of the
  corresponding gene.  -> one constitutive PAS per gene (the true 3' end).
```
`pub/scripts/select_constitutive_cpa.py:30` — `SOURCES = ["3READS", "ENCODE.PB", "GTEx.ONT"]`
`pub/scripts/select_constitutive_cpa.py:96-97`:
```
    ap.add_argument("--source", default="3reads", choices=["3reads", "mean", "max", "sum", "best"],
                    help="how to build PSExRPM across data sources (default 3reads = PolyA DB 3 native)")
```
Filters:
```
    ap.add_argument("--region", default="3'UTR",          # :98
    ap.add_argument("--biotype", default="protein_coding",# :102
    ap.add_argument("--keep_multimapped", action="store_true",
                    help="keep PAS assigned to >1 gene (default: drop, require PAS2gene==1)")  # :104-105
```
applied at `:123-126`:
```
    if a.biotype != "any" and "gene_biotype" in df.columns:
        df = df[df["gene_biotype"] == a.biotype]
    if not a.keep_multimapped and "PAS2gene" in df.columns:
        df = df[pd.to_numeric(df["PAS2gene"], errors="coerce") == 1]
```
One site per gene: `:157` — `    df = df.sort_values("PSExRPM", ascending=False)` (then dedupe on gene).
`--min_score` and `--top_percentile` both default to `None` (`:108-111`) — no
expression cutoff was applied.
The longest-isoform step was **approximated**: `:153-154` —
```
        iso_note = ("NOTE: no --ucsc_refgene supplied; taking the gene's dominant 3'UTR site "
                    "(highest PSExRPM) as the constitutive site (longest-isoform step approximated)")
```

### (d) Inputs / outputs

Input: `pub/data/HumanPas/hg38.PAS.main.tsv` (also at `train/data/polyadb_v4/HumanPas/hg38.PAS.main.tsv`).
Derived BED: `train/data/polyadb_v4/HumanPas/polyadb_v4_main.bed`, **303,312 lines**.
Output: `constitutive_cpa.tsv` / `.bed` → becomes `pub/data/Dominant_CPA_baseline_total.bed`.
Figure: `pub/supp_fig1.pdf`, `pub/supp_fig1.png` (mtime Jul 14 11:42, matching `sfig1.R` Jul 14 11:41).

### (e) Every hard-coded number

Every count in this figure is a literal in the R script — panels A–C do **not**
read data.

| Number | Line | Verbatim |
|---|---|---|
| 303,312 | `pub/sfig1.R:70` | `       subtitle="One dominant 3′UTR site per gene (highest PSE×RPM), from 303,312 PAS in PolyA DB v4")+` |
| 31,101 / 17,623 / 16,749 / 15,700 | `pub/sfig1.R:25` | `stage_n <-c(31101, 17623, 16749, 15700)` |
| stage names | `pub/sfig1.R:24` | `stage_nm<-c("all gene symbols","protein-coding","uniquely mapped","3′UTR site (kept)")` |
| −13,478 / −874 / −1,049 | `pub/sfig1.R:26` | `drop_rsn<-c("− 13,478 non-coding biotypes","− 874 multi-mapped (PAS2gene≠1)","− 1,049 no 3′UTR PAS")` |
| biotype counts | `pub/sfig1.R:30` | `biocnt <-c(17623,11278,803,607,286,146,116,74,47,24,20,16,61)` |
| dropped-PAS locations | `pub/sfig1.R:34` | `drop_n  <-c(1679,1385,388,270,199,142,28,4,1)` |
| 89.1 % retained | `pub/sfig1.R:66` | `           label=sprintf("%.1f%% of protein-coding genes retained",100*15700/17623),` |
| PSE, RPM, 3READS | `pub/sfig1.R:134` | `caption="Constitutive site = the highest PSE×RPM PAS in the 3′UTR of each protein-coding gene (one per gene). PSE×RPM from the 3READS quantification.")` |

**Verification and mismatches:**

- 303,312 **CONFIRMED** independently: `wc -l train/data/polyadb_v4/HumanPas/polyadb_v4_main.bed` = 303312.
- 31,101 − 13,478 = 17,623 ✓; 17,623 − 874 = 16,749 ✓; 16,749 − 1,049 = 15,700 ✓. The funnel is internally consistent.
- The biotype vector sums to 31,101 ✓ (`17623+11278+803+607+286+146+116+74+47+24+20+16+61`).
- The dropped-location vector sums to 4,096, which is *PAS in those genes*, not genes — as the axis label says (`pub/sfig1.R:101` — `x="PAS in these genes"`). Not a mismatch, but easy to misread.
- **MISMATCH (flag):** the funnel ends at **15,700**, but the positive BED that everything downstream uses has **15,794** rows (and 15,785 after chrY). A 94-site difference between the figure's terminal stage and the file it describes is unexplained by any script in the tree.
- **MISMATCH (flag):** the figure says "**PolyA DB v4**" (`pub/sfig1.R:3,70,133`) while the script that produced the data says "**PolyA DB 3**" (`pub/scripts/select_constitutive_cpa.py:5,14,20,97`). Both read the same file, `hg38.PAS.main.tsv`. The directory name `train/data/polyadb_v4/` sides with v4. The version tag is not recorded inside the data file's path or any manifest. **Resolve before writing Methods.**
- Panel D (`n=`, median) is computed live from the TSV (`pub/sfig1.R:112,121`), so it carries no hard-coded value.

### (f) Statistics

None. Panel D shows median and IQR only (`pub/sfig1.R:112`, `:116-117`).

---

## Figure 0.4 — performance

### (a) Plotting script

`pub/scripts/make_figure1.R` → `pub/figure1.pdf` / `.png` (mtime Aug 18 11:41).
Panel map, `pub/scripts/make_figure1.R:8-11`:
```
#   A metrics [prod,locked]  B corr [prod]  C top-K recovery [crossfit, within-250nt]
#   D ERP27 traces [prod]    E score-by-class [prod, Canonical/Atlas/Background]
#   F Jaccard [prod, vicinity threshold]
```
Panel G is appended (`:11-14`).

### (b) Upstream script

`pub/scripts/make_figure1_data.py` (v3.0, `:3`). It also writes the legend file:
`:20-21` — `  - Writes figure1_legends.md with the panel legends and the real n values, so`
`    the legends cannot drift from the figure.` → `pub/figure1_legends.md`.

Further upstream: `pub/scripts/score_genes.py` via `pub/scripts/score_array.slurm`
(produces `pub/scans/prod/per_gene/*.parquet` and `pub/scans/crossfit/per_gene/*.parquet`).

### (c) Exact parameters

`pub/scripts/make_figure1_data.py:500` —
```
    ap.add_argument("--erp27", default="ERP27"); ap.add_argument("--vic", type=int, default=250)
```
`pub/scripts/make_figure1_data.py:512-513` — `--require_canonical` defaults **True**
(`ap.add_argument("--require_canonical", action="store_true", default=True,`).

**Per-gene peak calling — the central methodological choice.**
`pub/scripts/make_figure1_data.py:129-135`:
```
# PER-GENE THRESHOLD
#
# A gene's peak-calling cut is the score at its own canonical 3' end: the max
# model score within +/- vic nt of the constitutive CPA window. Everything that
# calls peaks (panel F) uses this. There is deliberately no global cut.
```
`:141` — `    near = np.abs(ws - cpos) <= vic`

**Panel C (top-K).** `:331` — `def panel_C_topk(crossfit_dir, out, Kmax=10, tol=250):`;
scored from cross-fit parquets so no gene is scored by a model that saw its
chromosome. `:348` — `Combined` takes `min(best.values())`, i.e. best rank across
the three models. K = 1 is argmax recovery (`:349`).

**Panel E (score by class).** `:147` — `def scan_prod(prod_dir, atlas_idx, gene_chrom, vic, out, per_gene=6, seed=0,`;
`:162` — `    rng = np.random.default_rng(seed)`; 6 random background windows per gene.
Atlas class: `:217` — `                # E "Atlas" class: nearest window within 250, skipping the canonical site`.

**Panel F (Jaccard).** `:362-384`, `_f_one`: peak set per model =
`peak = {m: set(np.where(arr[m] >= gt[m])[0]) for m in MODELS}` (`:379`); pairwise
Jaccard over the three pairs; genes with no canonical site return `None` and are
excluded (`:377-378`); union-empty pairs contribute `NaN` and are dropped
(`:369-371` docstring, `:400`).

**Panel G (score vs expression).** Spearman per site group, computed at
`:306-307` — `.apply(lambda x: x["score"].corr(x["tpm"], method="spearman"),`.
Site groups from `:123-126`:
```
def _group_of(t):
    if t == "TE": return "Terminal"
    if t in ("IN", "EX"): return "Intragenic"
    return "Other"
```
A site's score is the max over windows within `vic` (=250) nt of it.

**SLURM/scan parameters.** `pub/scripts/score_array.slurm:34-44`:
```
python3 scripts/score_genes.py \
  --genes_bed "$GENES" \
  --cpa_bed data/Dominant_CPA_baseline_total.bed --cpa_bed_is_window \
  --genome ../train/data/hg38.fa \
  --pwms "$PWM_LIST" --bg 0.3 0.2 0.2 0.3 \
  --rf_dir  models/$NAME/rf \
  --lr_dir  models/$NAME/lr \
  --cnn_dir models/$NAME/cnn \
  --outdir "$OUT" \
  --cnn_batch 256 \
  --resume
```
with `#SBATCH --gres=gpu:h100:1`, `--cpus-per-task=8`, `--mem=128G` (`:4-6`).
Fold semantics: `:11-12` — `#   folds 0-4: score that fold's TEST-chromosome genes with models/fold{k}/  (leak-free)` / `#   prod     : score ALL genes with models/prod/  (for the published Figure 1-4)`.

### (d) Inputs / outputs

| Input | Path |
|---|---|
| prod scan | `pub/scans/prod/per_gene/*.parquet` (20,012 genes) |
| cross-fit scan | `pub/scans/crossfit/per_gene/*.parquet` |
| canonical BED | `pub/data/Dominant_CPA_baseline_total.bed` |
| atlas BED | `train/data/atlas.chr.bed` (PolyASite 2.0, 569,005 clusters) |
| models | `pub/models/prod/{rf,lr,cnn}/` |

| Output | Path |
|---|---|
| panel n's | `pub/fig1_counts.tsv` |
| B | `pub/fig1_corr.tsv` |
| C | `pub/fig1_topk.tsv` |
| D | `pub/fig1_traces.tsv` |
| E | `pub/fig1_scoredist.tsv` (41 MB) |
| F | `pub/fig1_jaccard.tsv` |
| G | `figG_expr.tsv` (referenced at `pub/make_figureG.R:25`) |
| thresholds (record only) | `pub/fig1_thresholds.tsv` |
| legends | `pub/figure1_legends.md` |

### (e) Every hard-coded number

**Panel A metrics are hard-coded in the R script and do not read any file.**
`pub/scripts/make_figure1.R:83-84`:
```
metrics <- tribble(~model,~AUROC,~AUPRC,~F1,
  "RF",0.987,0.851,0.781, "LR",0.983,0.840,0.761, "CNN",0.991,0.923,0.872) |>
```

Against the locked model metadata (`pub/models/prod/*/metadata.json`,
`winner_meta.test_metrics`, all on `support: 104031`):

| model | metric | figure (`make_figure1.R:84`) | metadata | verdict |
|---|---|---|---|---|
| RF | AUROC | 0.987 | 0.9873593422899334 | ✓ |
| RF | AUPRC | **0.851** | **0.8413127280473581** | **MISMATCH (+0.010)** |
| RF | F1 | **0.781** | **0.771716469770674** | **MISMATCH (+0.009)** |
| LR | AUROC | 0.983 | 0.9826769677587475 | ✓ |
| LR | AUPRC | **0.840** | **0.8274380323547013** | **MISMATCH (+0.013)** |
| LR | F1 | **0.761** | **0.7583983712249747** | **MISMATCH (+0.003)** |
| CNN | AUROC | 0.991 | 0.9912056164269252 | ✓ |
| CNN | AUPRC | 0.923 | 0.9226807205913508 | ✓ |
| CNN | F1 | **0.872** | **0.8650338718082334** | **MISMATCH (+0.007)** |

I searched the whole tree for a table carrying `0.851` together with `0.781` and
found none; the five mismatching values do not correspond to any of the five
cross-fit folds either (fold test metrics range AUPRC 0.793–0.812 for RF,
0.765–0.777 for LR, 0.888–0.904 for CNN; see `pub/models/fold*/`). **These five
numbers are unsourced. Flag for the author** — the honest values for a Methods
section are the metadata ones, which are also the ones the confusion matrices
in the same JSON reproduce (RF `CM: [[100496, 600], [714, 2221]]`,
LR `[[100372, 724], [700, 2235]]`, CNN `[[100764, 332], [445, 2490]]`).

**Panel n's — all confirmed, from `pub/fig1_counts.tsv` (written by the data script):**

```
panel	n	unit
A	104031	test windows
B	15568	genes
C	15559	genes
D	24750	windows
E	15568	genes
F	15568	genes
G	367488	sites
```

| Brief's label | Status |
|---|---|
| **104,031** | ✓ `pub/fig1_counts.tsv:2`; independently = 2,935 + 101,096 from `wc -l pub/splits/prod/{pos,neg}_test.bed`; and `"support": 104031` in all three `pub/models/prod/*/metadata.json`. |
| **15,568** | ✓ `pub/fig1_counts.tsv:3,6,7` (panels B, E, F). Panel C is **15,559**, nine fewer — a real, documented difference (C is cross-fit, B/E/F are prod). |
| **ERP27, 24,750** | ✓ `pub/fig1_counts.tsv:5`; gene name default at `make_figure1_data.py:500`; template at `:441` — `"D": ("Score traces along a single gene (ERP27, {n} windows). Each model's "`. |
| **367,488** | ✓ `pub/fig1_counts.tsv:8`. |
| **Jaccard** | ✓ panel F. Values `pub/fig1_jaccard.tsv`: `RF-LR 0.096`, `RF-CNN 0.166`, `LR-CNN 0.069`. |
| **Spearman** | ✓ panel G, `make_figure1_data.py:306`. Per-group ρ printed to console and written beside the data; not in a small TSV in the tree. |
| **Top-K / recovery** | ✓ `pub/fig1_topk.tsv`; K = 1: RF 0.3158, LR 0.2836, CNN 0.3636, Combined 0.4752. K = 10: RF 0.4157, LR 0.4268, CNN 0.6329, Combined 0.6885. |
| **TPM** | ✓ panel G x-axis; atlas TPM parsed at `make_figure1_data.py:80` — `                tpm = float(p[4]); pct = float(p[6]); typ = p[9]`. |
| **AUROC / AUPRC / F1** | ✓ panel A, but see the mismatch table above. |

Also hard-coded, and **superseded at runtime**: `pub/scripts/make_figure1.R:59` —
`THRESH <- c(RF=0.932, LR=0.980, CNN=0.889)` — overwritten from
`pub/fig1_thresholds.tsv` when present (`:60-64`), which holds
`RF 0.9208 / LR 0.9742 / CNN 0.8824` (means; medians 0.9836 / 0.9914 / 0.9576).
These are read "for the record" and **not plotted** (`:18-23`).

Legend prose reports the per-gene cut distribution: `pub/figure1_legends.md:23` —
`they span the full score range (RF median 0.98, range 0.004-0.998), and low-cut genes do not produce degenerate peak sets (median 0.9% of windows called).`

### (f) Statistical tests

- Pearson correlation between model score vectors, streaming (`pub/figure1_legends.md:7`).
- Spearman correlation of score against site expression per group (`make_figure1_data.py:306`).
- 95 % CI on the *mean* score within each expression bin, panel G (`pub/figure1_legends.md:17`).
- **No multiple-testing correction anywhere in this figure**, and none is needed as
  no hypothesis tests are reported.

**Scope caveat the author should carry into Methods** —
`pub/figure1_legends.md:21`:
```
All panels use the same gene set: genes with an annotated dominant CPA site in PolyA_DB. Genes without one are excluded throughout, since no per-gene peak-calling cut can be defined for them. These are not a random subset — they skew toward low expression and dispersed cleavage — so the figure describes well-annotated genes.
```

---

## Figure 0.5 — competitor features

### (a) Plotting script

`pub/scripts/make_figure2.R` → `pub/figure2.pdf` / `.png`
(`pub/scripts/make_figure2.R:222-223`, `width=18, height=11`, `dpi=400`).
Panel map, `:5-11`:
```
#   A  inter-model peak convergence
#   B  APARENT2 within-gene rank, by contest size
#   C  PAS density, real vs dinucleotide null
#   D  APARENT2 ROC
#   E  APARENT2 precision-recall
#   F  positional motif density
```

### (b) Upstream scripts

- Candidate set: `pub/scripts/build_shortlist.py` → `pub/shortlist_crossfit_windows.parquet`
- Motif composition + null: `pub/scripts/featurize_shortlist_cpa.py` → `pub/shortlist_cpa.parquet`, `pub/shortlist_cpa_density.tsv`
- Statistics: `pub/scripts/figure2_stats.py` → `fig2_*.tsv`
- Analysis-set flags: `pub/task_build_final_parquet.py` → `pub/cpa_peaks_final.parquet` (documented in `pub/cpa_peaks_final.README.md`)

### (c) Exact parameters

**Candidate ("competitor" / model-false-positive) definition** —
`pub/scripts/build_shortlist.py:5-19`:
```
Per gene, from its crossfit scan parquet (leak-free: rf/lr/cnn scored by a model
that never saw the gene's chromosome):

  1. canonical region = the gene's window(s) from Dominant_CPA_baseline_total.bed
     (cols: chrom,start,end,symbol,ENST,strand). A scan window "overlaps" if
     ws < canon_end AND ws+win > canon_start.
  2. per model m in {rf,lr,cnn}:  m_canonical = MAX m_prob over overlapping windows.
  3. CONSENSUS shortlist = windows where rf>=rf_canonical AND lr>=lr_canonical AND
     cnn>=cnn_canonical  (every model independently ranks the window >= its own
     canonical). No single model does the selecting, so RF/LR/CNN are all "spent"
     on selection; APARENT2 (external) + motif composition are the independent
     evidence for Figure 2.
       - windows overlapping the canonical region -> is_canonical=True (reserved)
       - the rest                                 -> internal competitors
  4. NMS-collapse to one peak per site, canonical reserved.
```
`pub/scripts/build_shortlist.py:111-112`:
```
    ap.add_argument("--win", type=int, default=500)
    ap.add_argument("--merge_dist", type=int, default=250,
```
`:122` — `    ap.add_argument("--score_col", default="consensus",`

**Three analysis-set filters** — `pub/figure_twofeature_notes.md:18-29`:
```
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
```

**Motif-density window and null** — `pub/scripts/featurize_shortlist_cpa.py:12-15`:
```
  - fetch the strand-aware window around `center` (default +/-100 nt),
  - count CPA motif hits: PAS hexamers, CFIm/UGUA, downstream U-rich, GU-rich,
  - flag has_PAS in the upstream PAS zone,
  - build a dinucleotide-preserving shuffle N_SHUFFLE times and recount.
```
`:189-196`:
```
    ap.add_argument("--up", type=int, default=100, help="upstream nt around center")
    ap.add_argument("--down", type=int, default=100, help="downstream nt around center")
    ap.add_argument("--n_shuffle", type=int, default=20)
    ap.add_argument("--split", default="test", choices=["all","test","val","train"])
    ap.add_argument("--contested_only", action="store_true")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
```
`:131-132` — `    """Preserve exact dinucleotide counts (Altschul & Erikson 1985) via an Eulerian` / `    walk on the dinucleotide graph. Falls back to identity for degenerate seqs."""`
`:246` — `    MOTIFS = ["PAS","UGUA","U-rich","GU-rich"]`
`:52-54` — the 13 PAS hexamers (quoted in Figure 0.1 above).
Recorded invocation, `:31-36`:
```
  python3 featurize_shortlist_cpa.py \
    --shortlist shortlist_crossfit_windows.parquet \
    --peaks cpa_peaks_final.parquet \
    --reference_fasta ../train/data/hg38.fa \
    --split test --contested_only --n_shuffle 20 \
    --out shortlist_cpa
```

**Statistics parameters** — `pub/scripts/figure2_stats.py:110-117`:
```
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--boot_pr", type=int, default=200,
    ap.add_argument("--out", default=".")
    …
    rng = np.random.default_rng(0)
```
Recorded invocation, `:34-39`:
```
  python3 figure2_stats.py \
    --shortlist shortlist_crossfit_windows.parquet \
    --cpa shortlist_cpa.parquet \
    --peaks cpa_peaks_final.parquet \
    --cpa_bg_auroc 0.991 \
    --out .
```
AUROC is Mann-Whitney with midranks, implemented inline (`:49-60`); AUPRC is
average precision by the step rule with its no-skill baseline (`:63-74`).
Within-gene rank: `:77-79` — `    """Rank percentile of the gene's canonical site among that gene's own false` / `    positives, ties counted as half.`

**Plot-side smoothing (must be stated in Methods).** Panels C and F apply a
9-point moving average before drawing:
`pub/scripts/make_figure2.R:150` and `:210` —
`    mutate(density=stats::filter(density,rep(1/9,9),sides=2) |> as.numeric()) |>`

### (d) Inputs / outputs

| Input | Path |
|---|---|
| shortlist | `pub/shortlist_crossfit_windows.parquet` (6.4 MB) |
| CPA motif features | `pub/shortlist_cpa.parquet` (4.2 MB) |
| analysis flags | `pub/cpa_peaks_final.parquet` (1.9 MB) — see `pub/cpa_peaks_final.README.md` |
| genome | `train/data/hg38.fa` (hg38) |
| annotation for terminal exons | GENCODE v49 |
| APARENT2 scores | joined from `train/gene_scans/per_gene` (`build_shortlist.py:35`) |

| Output | Path |
|---|---|
| A | `pub/fig2_dispersion.tsv` (43,547 rows + header) |
| B | `pub/fig2_rank_binned.tsv` |
| C, F | `pub/shortlist_cpa_density.tsv` |
| D | `pub/fig2_roc.tsv` |
| E | `pub/fig2_pr.tsv` |
| effect sizes | `pub/fig2_aparent.tsv` |
| n's | `pub/fig2_counts.tsv` |
| figure | `pub/figure2.pdf` / `.png` |

### (e) Every hard-coded number, and the brief's labels

`pub/fig2_aparent.tsv` (the load-bearing table):
```
kind	subset	value	lo	hi	baseline	n_genes	n_canon	n_fp
within_gene_rank	analysis_set	0.4768	0.4653	0.4892	0.5	4332.0		
pooled_auroc	analysis_set	0.4782	0.4697	0.4854	0.5		5898.0	38186.0
pooled_auprc	analysis_set	0.1335	0.1292	0.1384	0.1338		5898.0	38186.0
cpa_vs_background_ref	figure1	0.991						
```
`pub/fig2_counts.tsv`:
```
panel	n	unit
A	43547	model false positives
B	43547	model false positives
C	4332	genes
D	50130	peaks: 43,547 false positive + 6,583 canonical
```

| Brief's label | Source | Verdict |
|---|---|---|
| **0.478** | `pub/fig2_aparent.tsv` `pooled_auroc = 0.4782` (CI 0.4697–0.4854; chance 0.5) | ✓ **RESOLVED** — this is APARENT2's pooled AUROC for canonical vs competitor, i.e. *below chance*. |
| **0.470** | — | **PARTIAL.** No `0.470` exists. The nearest values are `within_gene_rank = 0.4768` (CI 0.4653–0.4892) and, in `pub/fig2_rank_binned.tsv`, the 2-competitor bin `0.478`. If the figure prints two numbers side by side they are most likely **0.478 (pooled AUROC)** and **0.477 (within-gene rank)**. **Confirm against the rendered panel.** |
| **0.134** | `pooled_auprc = 0.1335`, `baseline = 0.1338` | ✓ **RESOLVED**, and note the AUPRC *equals its own no-skill baseline*. Rationale hard-coded in the plot script, `pub/scripts/make_figure2.R:29-30`: `# E matters more than D: canonical sites are ~13% of the set, so the no-skill` / `# AUPRC is 0.134, not 0.5.` |
| **median 5** | Panel A, computed live at `pub/scripts/make_figure2.R:102` — `  med <- median(d$spread_nt); f50 <- mean(d$spread_nt<=50)`; printed at `:108` — `             label=paste0("median ",med," nt"))`. Recomputed from `pub/fig2_dispersion.tsv`: n = 43,547, **median = 5.0 nt**, 80.4 % within 50 nt. | ✓ **CONFIRMED** — median inter-model peak spread of 5 nt. Not hard-coded. |
| **dinucleotide null** | Panel C, `pub/scripts/make_figure2.R:8,139,144-148`; generated by `pub/scripts/featurize_shortlist_cpa.py:129-132` (Altschul–Erikson), 20 shuffles, seed 0. | ✓ |
| **metaprofile / density per peak** | Axis labels: `pub/scripts/make_figure2.R:158` — `y="PAS density per peak"`; `:217` — `y="Motif density per peak"`. Window ±100 nt around the site. | ✓ |
| **Rank of canonical** | Panel B y-axis, `pub/scripts/make_figure2.R:136` — `labs(title="B",x="cCPA sites in the gene", y="Rank of canonical site")+th`. Per-bin values in `pub/fig2_rank_binned.tsv`: 1→0.4881 (n=1,465), 2→0.478 (681), 3→0.4966 (388), 4-5→0.4885 (497), 6-10→0.4624 (574), 11+→0.4459 (727). | ✓ |

Other hard-coded numbers in this figure:
- `pub/scripts/make_figure2.R:103` — `  XMAX <- 60; ntail <- mean(d$spread_nt > XMAX)` (panel A x-limit).
- `pub/scripts/make_figure2.R:232-233` — printed caption counts:
  `cat("  D  38,186 cCPA + 5,898 canonical (APARENT2 available)\n")` (twice, D and E).
  These agree with `fig2_aparent.tsv` `n_canon = 5898.0, n_fp = 38186.0`.
- **Denominator note (flag).** Panel C's n is 4,332 genes and panels D/E cover
  38,186 + 5,898 = 44,084 peaks with APARENT2 available, while panels A/B cover
  43,547 and the density panels' underlying file covers 50,130. The R script
  remaps stale panel keys at `:75-78`:
  ```
  ## fig2_counts.tsv is keyed by the OLD panel letters from figure2_stats.py:
  ##   A dispersion, B motif density, C rank, D pas-null
  ## Map them onto the new positions so the printed caption counts are right.
  OLD2NEW <- c(A="A", C="B", D="C", B="F")
  ```
  so the caption counts for panels C and F come from an old key mapping. Verify
  the numbers actually printed under each panel before quoting them.

### (f) Statistical tests

- Bootstrap CIs over genes, 1,000 resamples (`figure2_stats.py:110`), 200 for the
  PR curve (`:111`), `np.random.default_rng(0)` (`:117`).
- AUROC = Mann-Whitney U with midrank tie handling, inline (`:49-60`).
- AUPRC = step-rule average precision with its analytic no-skill baseline (`:63-74`).
- Dinucleotide-preserving-shuffle null for PAS density, 20 shuffles per peak.
- **No multiple-testing correction.** None is applied; the figure reports effect
  sizes with CIs rather than p-values.

---

## Figure 0.6 — reporter schema

### (a)–(f)

**NOT FOUND — no code in this tree produces or supports this figure.**

Exhaustive case-insensitive search over all 662 source files
(`.py`, `.R`, `.sh`, `.slurm`, `.md`, `.json`, `.yaml`) for:
`eGFP`, `GFP`, `Sanger`, `subclone`, `plasmid`, `reporter`, `minigene`,
`oligodT`, `oligo(dT)`, `transfect`, `RT-PCR`, `construct` (in a cloning sense):

| String | Result |
|---|---|
| `eGFP` | 0 files |
| `Sanger` | 0 files |
| `subclone` | 0 files |
| `reporter` | 0 files |
| `minigene` | 0 files |
| `transfect` | 0 files |
| `RT-PCR` | 0 files |
| `oligodT` / `oligo(dT)` | 0 files |
| `GFP` | 1 file — `logs/aparent2_scan_summary.json` (a gene symbol in a scan log, not a construct) |
| `plasmid` | 2 files — `scripts/saliency_aparent2.py`, `scripts/score_genes_aparent2.py` (references to the APARENT2 *training library*, not a cloning step) |

The only two matches that touch the brief's terms at all:

- **HEK293** — `pub/datasets.yaml:73`:
  ```
        # panel, which includes HeLa and HEK293 -- kept separate because an
  ```
  This is a comment about which cell lines appear in the **published Singh/Lee
  3'-seq panel** used for Figure 0.8; it is not a statement about a reporter assay
  performed in HEK293.
- **oligo-dT** — `pub/datasets.yaml:88` and `:111`:
  ```
      assay: "oligo-dT 3'-seq, polyA sites"
  ```
  Again describing the **Oh 2017 published assay**, not a reporter.

`~600` (construct length), "dual-CPA", and every wet-lab step are absent.

**Conclusion for Methods:** Figure 0.6 is a hand-drawn schematic of bench work
whose protocol is not recorded anywhere in this directory. The Methods text for it
must come from the lab notebook, not from this repository. The one thing the
repository *can* supply is the candidate-selection logic that presumably chose the
sites to clone — see Figure 0.7 below.

---

## Figure 0.7 — gel

### (a)–(f)

**NOT FOUND — no code in this tree produces or supports this figure.**

| String | Result |
|---|---|
| `dual-CPA` / `dual CPA` | 0 files |
| `+ctrl` / `-ctrl` | 0 files (as figure labels) |
| `synthetic` (as a construct class) | 0 files |
| `U1AMO` / `U1 AMO` | 19 files — **all** referring to the published Oh 2017 GEO dataset, not to an AMO treatment done here |
| `leukemia` | 1 file — `pub/scripts/00_download_and_liftover.sh:114`: `# leukemia + immune (shared series)`, a comment labelling the GEO downloads `GSE111310` / `GSE111793` |
| `immune` | `pub/datasets.yaml:41` — `cells: "immune subsets, tissues, CLL"`, describing the same published atlas |

So "leukemia", "immune", and "U1 AMO" in the figure are almost certainly the
*sample/condition labels carried over from the public datasets in Figure 0.8*,
not experiments performed here.

**The nearest thing to a candidate-selection method** (i.e. how "predicted" sites
might have been chosen for testing) is:

- `pub/hiconf_sets.py` — builds one high-confidence site set per published
  experiment and cross-tabulates them; outputs in `pub/hiconf_out/sites/`
  (`atlas_3seq.tsv`, `oh_u1amo.tsv`, `pcf11_dep.tsv`, `singh_ipa.tsv`, `lee_ipa.tsv`),
  plus `jaccard_matrix.tsv`, `hexamer.tsv`, `restricted.tsv`.
  Parameters, `pub/hiconf.yaml:10-14`:
  ```
  defaults:
    tolerance: 50          # nt, for the all-by-all
    min_reads: 2           # absolute floor
    enrich_ratio: 3.0      # treatment / background at matched coordinates
    merge_within: 25
  ```
- `pub/task_u1_hit_character.py`, `pub/task_u1_hit_intron.py`,
  `pub/task_u1_hit_ecdf.py` → `pub/make_figure_u1_hits.R` → `pub/figure_u1_hits.pdf`
  characterise the 274 model false positives that a U1 AMO experiment used
  (`pub/figure_u1_hits_notes.md:1-10`).

None of this is a gel, and none of it names constructs.

**Conclusion for Methods:** as with 0.6, the wet-lab protocol is not in this
directory. If the "predicted" lane corresponds to sites picked from this analysis,
the selection criterion is most likely the `pub/hiconf_out/sites/oh_u1amo.tsv`
set or the 274 U1-AMO-hit false positives; **the author must confirm which**, as
no file records the choice.

---

## Figure 0.8 — perturbation overlap

### (a) Plotting script

`pub/make_figure_overlap.R` → `pub/figure_overlap_filtered.pdf` / `.png`
(mtime Aug 14 07:09). It does **zero** computation:
`pub/figure_overlap_notes.md:9-10`:
```
Figure `figure_overlap_filtered.pdf` / `.png`, built by `make_figure_overlap.R`, which does
no computation.
```
`pub/make_figure_overlap.R:11-14`:
```
# Zero computation. Percentages and their CIs come from ovl_panel_a_sites.tsv, which counts
# each external site once; fold and p come from ovl_unfilt_competitor.tsv and
# ovl_ana_competitor.tsv; the UpSet reads ovl_ana_per_peak.tsv; the caption sentence reads
# ovl_pairwise_external_pairwise.tsv.
```

### (b) Upstream scripts

- `pub/task_overlap_prep.py` — filter check against PolyASite 2.0 / PolyA_DB v4, writes `ovl_filter_check.tsv`
- `pub/task_overlap_accounting.py` — verifies counts, writes the peak table `ovl_peaks_analysis.tsv`
- `pub/scripts/03_simple_overlap.py` — the overlap engine
- `pub/task_oh_topn.py` — reproduces and re-runs Oh's top-N peak cut
- Download/liftover: `pub/scripts/00_download_and_liftover.sh`, `pub/scripts/00b_verify_and_repair.sh`, `pub/scripts/01_fetch_gse135140.sh`
- Peak calling from bigWig: `pub/scripts/04_bw_peaks.py`
- Dataset registry: `pub/datasets.yaml`, `pub/hiconf.yaml`

Recorded provenance, `pub/figure_overlap_notes.md:158-166`:
```
python task_overlap_accounting.py                       # verify, account, write the peak table
python scripts/03_simple_overlap.py --peaks ovl_peaks_analysis.tsv \
  --external atlas3seq=ovl_atlas3seq.tsv \
             oh_camo=external/peaks/oh2017_cAMO_hg38.tsv \
             oh_u1amo=external/peaks/oh2017_U1AMO_hg38.tsv \
  --tolerances 25,50,100 --shuffle 200 --out ovl_ana
Rscript make_figure_overlap.R
```

### (c) Exact parameters

**Match tolerance = 50 nt.** `pub/make_figure_overlap.R:21` — `TOL <- 50`;
subtitle at `:83` — `       subtitle = paste("Match within 50 nt. Each external site counted once; the Oh bigWigs are",`;
also `:124` and `:142` (`tolerance 50 nt and exact coordinates`).
The engine computes 25/50/100 (`--tolerances 25,50,100`); the figure filters to 50.
Defaults: `pub/scripts/03_simple_overlap.py:236` — `    ap.add_argument("--tolerances", default="25,50,100")`.

**Shuffle null.** `pub/scripts/03_simple_overlap.py:245` —
`    ap.add_argument("--shuffle", type=int, default=50,` — but the published run
used **200** rounds. `pub/figure_overlap_notes.md:78-80`:
```
Tolerance 50 nt. Percentage and Wilson 95% CI are dataset-side on unique sites. Fold and p
are false-positive-side: the peaks hit against the same sites displaced by a random 10 to
100 kb offset, 200 rounds, so p floors at 0.005.
```
Offset range from `pub/datasets.yaml:28` — `  shuffle_offset: [10000, 100000]`;
`pub/datasets.yaml:27` — `  shuffle_rounds: 200`.

**Merging and read floors.** `pub/datasets.yaml:24-29`:
```
defaults:
  window: 500
  tolerances: [50, 100, 250, 500]
  shuffle_rounds: 200
  shuffle_offset: [10000, 100000]
  quant_half: 50
```
Per-dataset `merge_within: 25`, `min_reads: 2` (`pub/datasets.yaml:102,124,159-160,…`).

**Top-N cut on the Oh bigWigs.** `pub/scripts/03_simple_overlap.py:21-26`:
```
--bw-top-n keeps the N strongest peaks per bigWig. Use it whenever two bigWigs
are compared: cAMO and U1AMO have equal total signal spread over very different
numbers of bases, so a fixed value threshold calls them at different
stringencies and the comparison becomes an artefact of the cutoff.
```
The published run used **top 40,000** per condition (`pub/figure_overlap_notes.md:64-72`).

**liftOver hg19 → hg38.** `pub/scripts/00_download_and_liftover.sh:130-131` —
chain `hg19ToHg38.over.chain.gz` from UCSC; `:160` —
`"$LO" -minMatch=0.95 "$TMP/in.bed" "$CH" "$TMP/out.bed" "$TMP/unmapped.bed" 2>/dev/null`;
multi-mapped rows are dropped (`:162-164`); positions treated as 1-based on input (`:143`).

**"60 nt upstream" + AATAAA/ATTAAA.** This is the hexamer convention used in the
mechanism analyses that accompany this figure:
`pub/task_gc_model.py:22` — `  HEXAMER TEST. AATAAA or ATTAAA in [pA-60, pA), the convention of task_composition_test.py.`
`pub/task_gc_model.py:40` — `PAS_STRONG = ("AATAAA", "ATTAAA"); PAS_LO, PAS_HI = -60, 0`
`pub/figure_twofeature_notes.md:122` — `**The hexamer test, which the mechanism had to pass.** AATAAA or ATTAAA in [pA-60, pA),`
`pub/figure_twofeature_notes.md:464` — `**The hexamer stratification** (was panel h). Hexamer = AATAAA or ATTAAA in [pA-60, pA).`
`pub/figure_twofeature_notes.md:670` — `to pA-5000), \`hexamer_cache.npz\` (AATAAA/ATTAAA in [pA-60, pA)).`
`pub/hiconf_sets.py:43` — `HEX = ("AATAAA", "ATTAAA")`

**"nearest site".** `pub/scripts/03_simple_overlap.py:5` —
`Do our peaks agree with published ones. Strand-aware nearest neighbour, hit`
`within a tolerance, broken down by canonical vs competitor.`
Unstranded Oh data is matched on either strand, by design
(`pub/figure_overlap_notes.md:61-62`).

**"telescripting".** `pub/hiconf.yaml:48` — `    label: "U1 AMO\n(telescripting loss)"`.
The term appears in 31 source files; it is the mechanistic framing of the U1 AMO
arm, not a computed quantity.

**"PCF11".** `pub/datasets.yaml:126-141` documents the sign-flip test
(siPCF11 vs siLUC, and dPAS1-muB vs wt), with the caveat at `:168-170`:
```
    direction: down          # PCF11 favours proximal PAS; depletion REDUCES
                             # intronic use. Overlap cannot see loss.
```
and `pub/hiconf.yaml:56-61` (`pcf11_dep`). **Note:** the PCF11 arm is present in
the registries and in `pub/hiconf_out/sites/pcf11_dep.tsv`, but the published
Figure 0.8 (`figure_overlap_filtered`) shows only three rows — 3'-seq atlas,
Oh control AMO, Oh U1 AMO (`pub/figure_overlap_notes.md:46-48, 84-86`). If the
thesis figure shows a PCF11 row, it comes from the `pub/hiconf_out/` /
`fig_jaccard_hexamer` line of work, not from `figure_overlap_filtered`.

### (d) Inputs — provenance

| Dataset | Accession | Files | Build | Lab / paper |
|---|---|---|---|---|
| 3'-seq IpA atlas | **GSE111310** + **GSE111793** | `GSE111310_ipa_tpm.txt.gz`, `GSE111793_ipa_tpm.txt.gz` | hg19 → lifted | **Singh 2018** (Nat Commun) + **Lee 2018** (Nature), Mayr/Leslie |
| Oh control AMO | **GSE103252** | `GSE103252_cAMO_15nmol_odT_selection.polyA_sites.bw` | hg19 → lifted | **Oh 2017** (NSMB), Dreyfuss |
| Oh U1 AMO | **GSE103252** | `GSE103252_U1AMO_15nmol_odT_selection.polyA_sites.bw` | hg19 → lifted | Oh 2017, Dreyfuss |
| PCF11 (registered, not in the 3-row panel) | **GSE123105** | `GSM3536480 siLUC`, `GSM3536484 siPCF11`, `GSM3536487 wtN`, `GSM3536493 T2/muB` `_mergedReplicates_{F,R}.bw` | hg19 | **Kamieniarz-Gdula 2019**, Proudfoot |
| So 2019 (downloaded, used in the earlier `fig_site_overlap` run) | **GSE135140** | bigWigs | hg38 | `pub/scripts/00_download_and_liftover.sh:112` — `geo_series GSE135140        # So 2019, hg38, primary` |
| Leslie-lab code | bitbucket `leslielab/apa_2018` | vendored at `pub/external/apa_2018/` | — | `pub/scripts/00_download_and_liftover.sh:124-125` |
| journal supplements | DOIs `10.1038/s41586-018-0465-8` (Lee), `10.1038/s41467-018-04112-z` (Singh), `10.1038/nsmb.3473` (Oh) | `pub/external/supp/` | — | `pub/scripts/00_download_and_liftover.sh:118-120` |
| peak set under test | `pub/cpa_peaks_final.parquet` | 70,723 peaks / 6,583 genes | **hg38** | this study |
| gene bodies | `train/data/pc_genes.bed` | 20,042 | hg38 | GENCODE v49 |

Explicitly **excluded** datasets and why: `pub/datasets.yaml:226-241`
(GSM3536494 duplicate; Chiu 2018 GSE100537 is mouse mm9; Beckedorff has no 3'-end
calls; PolyA_DB/PolyASite are circular by construction).

### (d, cont.) Outputs

`pub/ovl_ana_*.tsv` (analysis set), `pub/ovl_unfilt_*.tsv` (all 64,140),
`pub/ovl_pairwise_external_pairwise.tsv`, `pub/ovl_panel_a_sites.tsv`,
`pub/ovl_ana_per_peak.tsv`, `pub/ovl_peaks_analysis.tsv`, `pub/ovl_filter_check.tsv`,
logs `pub/ovl_runs.log`, `pub/ovl_runs_analysis.log`; figure
`pub/figure_overlap_filtered.pdf` / `.png`.

### (e) Every hard-coded number

**Assertion constants in the accounting script** — `pub/task_overlap_accounting.py:51-55`:
```
print(f"in_analysis_set  {n_all:>7,} peaks   expected 48,394   {'OK' if n_all == 48394 else 'MISMATCH'}")
print(f"  canonical      {n_can:>7,}         expected  4,847   {'OK' if n_can == 4847 else 'MISMATCH'}")
print(f"  competitor     {n_com:>7,}         expected 43,547   {'OK' if n_com == 43547 else 'MISMATCH'}")
print(f"  genes          {n_gene:>7,}         expected  4,847   {'OK' if n_gene == 4847 else 'MISMATCH'}")
```
`:71` — `FILT = [("APA-annotated: a PolyASite or PolyA_DB site within 500 nt", T.flag_apa_annotated),`
`:93` — `assert int(cur.sum()) == 43547 and gcount(cur) == 4847`

**The filter cascade** — `pub/figure_overlap_notes.md:33-38`:
```
| step | removed | genes hit | genes emptied | remaining | genes |
| model false positives, all | 0 | 0 | 0 | 64,140 | 6,583 |
| APA-annotated: a PolyASite or PolyA_DB site within 500 nt | 18,119 | 5,122 | 1,575 | 46,021 | 5,008 |
| in a reference terminal exon | 1,076 | 689 | 85 | 44,945 | 4,923 |
| at or downstream of the gene's canonical site | 1,398 | 266 | 76 | 43,547 | 4,847 |
```

**External dataset accounting** — `pub/figure_overlap_notes.md:46-48`:
```
| 3'-seq atlas | 9,557 | - | - | 0 (0.00%) | 4,106 | - | **5,451** | 2,898 | 2,258 |
| Oh 2017 control AMO | 184,174 | 50,472 | 10,472 | 25 (0.06%) | - | 39,978 | **39,972** | 8,643 | 6,275 |
| Oh 2017 U1 AMO | 187,018 | 67,976 | 27,976 | 30 (0.07%) | - | 39,970 | **39,970** | 8,329 | 6,320 |
```

**Panel a results** — `pub/figure_overlap_notes.md:84-86`:
```
| 3'-seq atlas (circular) | 5,451 | 596, **10.93%** [10.13, 11.79], 40.4-fold, p < 0.005 | 19, **0.35%** [0.22, 0.54], 2.1-fold, p = 0.015 |
| Oh 2017 control AMO | 39,972 | 1,307, **3.27%** [3.10, 3.45], 10.5-fold, p < 0.005 | 83, **0.21%** [0.17, 0.26], 1.2-fold, p = 0.065 |
| Oh 2017 U1 AMO | 39,970 | 1,249, **3.12%** [2.96, 3.30], 10.3-fold, p < 0.005 | 273, **0.68%** [0.61, 0.77], 3.9-fold, p < 0.005 |
```

**Panel b (UpSet)** — `pub/figure_overlap_notes.md:118-128`:
```
370 of the 43,547 false positives carry a match within 50 nt. Set sizes: U1 AMO 274,
control AMO 87, 3'-seq atlas 19.
| U1 AMO only | 265 |
| control AMO only | 79 |
| 3'-seq atlas only | 17 |
| control AMO + U1 AMO | 7 |
| all three | 1 |
| 3'-seq atlas + U1 AMO | 1 |
```

**Known internal discrepancy, already documented** —
`pub/figure_u1_hits_notes.md:7-10`:
```
**Count correction.** 274 false positives carry a U1 AMO site within 50 nt, not 273, and 87
carry a control AMO site, not 83. The 273 and 83 are dataset-side, the number of Oh sites
landing on a false positive; the two differ when two Oh sites fall near the same peak. So the
split is 274 against 43,273. Six false positives are hit by both conditions.
```
i.e. 273/83 are dataset-side counts and 274/87 are peak-side counts; both are correct
for their own denominator. Say which one the figure uses.

**Superseded run to avoid quoting** — `pub/figure_overlap_notes.md:24-25`:
```
The earlier run on 44,945 predates the downstream-of-canonical filter and is superseded.
Its outputs are the `ovl_filt_*` files; the corrected run is `ovl_ana_*`.
```

**Brief's labels, checked:**

| Label | Verdict |
|---|---|
| **27,856** | **NOT FOUND.** No `27,856` appears in any script, note, TSV, JSON, YAML or SLURM `.out`/`.err` log in the tree (bare `27856` occurs only as a digit-substring inside floats). The candidate set sizes actually used are **64,140** (all model false positives), **46,021**, **44,945** (superseded), **48,394** (analysis set incl. canonical) and **43,547** (analysis-set competitors). The closest lexical near-miss is **27,976** — the number of Oh U1-AMO blocks discarded by the top-40,000 cut (`pub/figure_overlap_notes.md:48,67`). **Flag for the author:** either the label is a typo for 27,976, or it belongs to a version of the analysis not present in this directory. |
| **50 nt / 50nt** | ✓ `pub/make_figure_overlap.R:21` (`TOL <- 50`), `:83`, `:124`, `:142`; `pub/hiconf.yaml:11`. |
| **telescripting** | ✓ `pub/hiconf.yaml:48` (label text); 31 files mention it. |
| **PCF11** | ✓ `pub/datasets.yaml:126-224`, `pub/hiconf.yaml:56-67`, `pub/hiconf_out/sites/pcf11_dep.tsv`. **But not a row of `figure_overlap_filtered`** — see (c). |
| **Singh / Lee** | ✓ `pub/datasets.yaml:36,46-50`; `pub/hiconf.yaml:6-8,18`. GSE111310 / GSE111793. |
| **AATAAA / ATTAAA** | ✓ `pub/hiconf_sets.py:43`, `pub/task_gc_model.py:40`, `pub/task_b8.py:47-50`. |
| **60 nt upstream** | ✓ `pub/task_gc_model.py:22,40` — `[pA-60, pA)`. |
| **nearest site** | ✓ `pub/scripts/03_simple_overlap.py:5` — strand-aware nearest neighbour. |

### (f) Statistical tests and corrections

- **Wilson 95 % CI** on each dataset's hit percentage, dataset-side, unique sites
  (`pub/figure_overlap_notes.md:78`).
- **Circular-shift permutation null**: peaks tested against the same external sites
  displaced by a uniform random 10–100 kb offset, **200 rounds**, so p floors at
  **0.005** (`pub/figure_overlap_notes.md:78-80`; `pub/datasets.yaml:27-28`).
- Pairwise agreement between external datasets at 50 nt and at exact coordinates
  (`pub/figure_overlap_notes.md:135-138`).
- Related U1-hit characterisation uses **Cliff's delta** with a gene-clustered 95 % CI
  and **Mann-Whitney by normal approximation with tie correction**
  (`pub/figure_u1_hits_notes.md:13-15`), plus a **within-gene permutation null over
  10,000 shuffles** (`pub/figure_u1_hits_notes.md:59`).
- **No multiple-testing correction is applied to Figure 0.8.** Three datasets × two
  peak sets = six reported p-values, all uncorrected. Benjamini–Hochberg is used
  elsewhere in the project (`pub/master_grid.py:18` — `  3. Benjamini-Hochberg q across every reported cell.`;
  `pub/gridsearch_direction.py:10` — `Benjamini-Hochberg q across all cells.`) but
  **not here**. State this, or apply it.

**Circularity caveat that must appear in Methods** —
`pub/figure_overlap_notes.md:100-104`:
```
The first filter removes false positives within 500 nt of a PolyASite or PolyA_DB site. The
3'-seq atlas is an APA atlas of that same class, so its collapse from 10.93% to 0.35% is a
property of the filter and carries no information about the peaks. The row is labelled
circular on the panel. The Oh sets are perturbation data from a different assay and are not
touched by the filter.
```

---

# 3. Cross-cutting

## 3.1 Genome build and annotation

| Item | Value | Evidence |
|---|---|---|
| Genome assembly | **hg38** | `train/data/hg38.fa` (+ `.fai`); `pub/scripts/extract_fastas.sh:9`; `pub/cpa_peaks_final.README.md:15` — `Assembly **hg38** (\`chrom\` is chr-prefixed).` |
| Chromosome sizes | `train/data/hg38.chrom.sizes` | |
| Gene annotation | **GENCODE v49** | `train/data/annotation/gencode.v49.annotation.gtf`; `pub/figure_corr_notes.md:14` — `overlaps the last exon of any GENCODE v49 transcript by at least 1 nt` |
| Derived annotation | `terminal_exons.bed`, `three_prime_utr.bed`, `three_prime_zone.bed` | `train/data/annotation/` |
| PAS atlas (1) | **PolyASite 2.0**, GRCh38, Ensembl 96 | `train/data/atlas.clusters.2.0.GRCh38.96.tsv` → `atlas.chr.bed`, 569,005 clusters |
| PAS atlas (2) | **PolyA_DB** (v3 vs v4 disputed — §5) | `pub/data/HumanPas/hg38.PAS.{main,max}.tsv`; main BED = 303,312 sites |
| External build | **hg19** for GSE103252, GSE111310, GSE111793, GSE123105 → lifted to hg38 | `pub/datasets.yaml` `build:` fields; chain `hg19ToHg38.over.chain.gz`, `-minMatch=0.95` |
| SpliceAI track | UCSC `spliceAi/wildtype/`, hg38, fetched 2026-07-21, server mtime Oct 1 2024 | `pub/REPRODUCIBILITY.md` §4, with per-file byte sizes and first-1-MiB md5s (**not** full-file checksums) |

**Coordinate convention (important, and non-obvious)** —
`pub/cpa_peaks_final.README.md:17-23`:
```
`center` is the **poly(A) site** of the peak, not the acceptor and not the window start.

**`center` is 0-based.** Every sequence fetch in this project, in the pipeline and in the
analysis scripts, indexes the FASTA with `center` as a 0-based offset, and the feature
rebuild reproduced the pipeline's own features to a maximum absolute difference of 0.000000
only under that reading.
```
and the strand asymmetry, `pub/cpa_peaks_final.README.md:49-53`:
```
The upstream window used by the features is `[center-2000, center-126]` in **transcript**
orientation, meaning decreasing genomic coordinate on the minus strand. In the fetch
convention the pipeline uses, the plus strand takes its first upstream base at genomic
`center-2000` and the minus strand at genomic `center+1999`, a one-base asymmetry between
strands that is inherited from the pipeline and preserved here so the features reproduce.
```

## 3.2 Software and versions

There is **no** machine-readable environment file (no `environment.yml`,
`requirements.txt`, or lockfile). However, **all three virtualenvs the scripts
reference still exist on disk**, so the versions below were read directly from
their `site-packages/*.dist-info` directories rather than inferred. This resolves
what was otherwise the largest hole in the record.

**Three distinct environments, used for different stages.** `pub/REPRODUCIBILITY.md` §5
documents only `splice_env2` — the *analysis* environment — and **not** `venv_2025`,
which is the environment that actually trained the models and produced every
Figure 0.1/0.2/0.4 number.

| | `~/venv_2025` | `~/bpnet_env` | `~/splice_env2` |
|---|---|---|---|
| **used for** | **featurization, RF/LR/CNN training, genome-wide scoring** (`featurize_all.slurm:14`, `train_*_array.slurm`, `score_array.slurm:20`) | splice-feature extraction (`scripts/*.sh`, `make_prod_split.sh:9`) | Figure-3 / SpliceAI analyses (`REPRODUCIBILITY.md` §5) |
| Python | **3.10** | — | 3.11.5 (per §5) |
| torch | **2.6.0** | 2.10.0 | 2.13.0 |
| scikit-learn | **1.2.1** | 1.7.1 | 1.8.0 |
| numpy | **1.26.4** | 2.3.3 | 2.4.2 |
| pandas | **2.2.3** | 2.3.3 | 3.0.0 |
| scipy | **1.15.1** | 1.16.2 | 1.17.0 |
| pyBigWig | — | 0.3.24 | 0.3.25 |
| biopython | 1.84 | — | — |
| joblib | 1.5.2 | — | — |
| matplotlib | 3.10.0 | — | — |
| pysam | 0.22.1 | — | — |
| maxentpy | 0.0.3 | — | — |
| tqdm | 4.67.1 | — | — |

(all `+computecanada` builds. `splice_env2`'s numpy 2.4.2 / pyBigWig 0.3.25 match
`pub/REPRODUCIBILITY.md` §5 exactly, confirming which env that document describes.)

**Two consequences worth stating in Methods.**
1. **scikit-learn 1.2.1** in the training env means `FrozenEstimator` is unavailable,
   so the RF used `CalibratedClassifierCV(rf, cv="prefit", method="sigmoid")` (§Figure 0.2).
2. A shell comment claims `sklearn 1.8 (HistGBM monotonic_cst); any env with sklearn>=1.4` —
   that describes `bpnet_env`/`splice_env2`, **not** the env that trained the RF and LR.

The table below records the remaining prose-only entries and the tools with no
version anywhere.

| Software | Version | Where recorded | Used by |
|---|---|---|---|
| Python | 3.11.5 (`splice_env2`) / **3.10 (`venv_2025`, the training env)** | §5; venv `site-packages` path | all Python |
| numpy | 2.4.2 (`splice_env2`) / **1.26.4 (`venv_2025`)** | §5; venv dist-info | everywhere (380 imports) |
| pandas | 3.0.0 (`splice_env2`) / **2.2.3 (`venv_2025`)** | §5 Appendix; venv dist-info | 112 imports |
| pyBigWig | **0.3.25** | `pub/REPRODUCIBILITY.md` §5 | bigWig reads, Fig 0.8 |
| R | **4.5.0** | `pub/REPRODUCIBILITY.md` §5; binary path in `pub/CLAUDE.md` | all figures |
| ggplot2 | **4.0.3** | `pub/REPRODUCIBILITY.md` §5; `pub/CLAUDE.md` | all figures |
| dplyr | **1.2.1** | `pub/REPRODUCIBILITY.md` §5 | all figures |
| readr | **2.2.0** | `pub/REPRODUCIBILITY.md` §5 | all figures |
| patchwork | **1.3.2** | `pub/REPRODUCIBILITY.md` §5 | figure assembly |
| scales | **1.4.0** | `pub/REPRODUCIBILITY.md` §5 | all figures |
| OS | Linux 5.14.0-687.25.1.el9_8.x86_64 | `pub/REPRODUCIBILITY.md` §5 | |
| R library path | `/home/spour98/R/x86_64-pc-linux-gnu-library/4.5` | `pub/REPRODUCIBILITY.md` §5 | |
| scikit-learn | **1.2.1** (`venv_2025`) — **RESOLVED** | venv dist-info | RF, LR (`train_rf.py`, `train_lr.py`) |
| PyTorch | **2.6.0** (`venv_2025`) — **RESOLVED** | venv dist-info | CNN (61 imports) |
| Biopython | **1.84** — **RESOLVED** | venv dist-info | `featurization.py`, `cpa_cnn.py`, `score_genes.py` |
| scipy | **1.15.1** (`venv_2025`) — **RESOLVED**; deliberately excluded from the Fig-3 path | venv dist-info; `pub/CLAUDE.md` | various |
| maxentpy (MaxEntScan) | **0.0.3** — **RESOLVED** (43 imports) | venv dist-info | splice features |
| pysam | **0.22.1** — **RESOLVED**; pyfaidx unversioned | venv dist-info | FASTA access |
| joblib **1.5.2**, tqdm **4.67.1**, matplotlib **3.10.0** — RESOLVED; pyarrow, shap unversioned | | venv dist-info | various |
| **ViennaRNA (`RNA`)** | **2.6.4** (module name only) | shell comment — `module load StdEnv/2023 viennarna/2.6.4` | folding scans |
| **bedtools** | **UNVERSIONED** (`module load bedtools`) | `pub/scripts/extract_fastas.sh:6`, `score_array.slurm:19` | FASTA extraction |
| **UCSC liftOver** | **UNVERSIONED** (binary fetched from `hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver`) | `pub/scripts/00_download_and_liftover.sh:132-135` | Fig 0.8 |
| **Arrow** | `arrow/17.0.0` (cluster module) | `score_array.slurm:18` and others | parquet I/O |
| R: tidyr, forcats, ggpubr, ggseqlogo, viridis, tibble, stringr, grid, ggridges, ggrepel, ggforce, ggbeeswarm | **UNVERSIONED** | — | various figures |

**Python environments referenced (three, inconsistently):**
`~/venv_2025` (`featurize_all.slurm:14`, `score_array.slurm:20`),
`~/bpnet_env` (`scripts/*.sh`, `pub/scripts/make_prod_split.sh:9`),
`~/splice_env2` (`pub/REPRODUCIBILITY.md` §5). None is captured as a lockfile.

**Cluster environment trap, worth a Methods footnote** — `pub/CLAUDE.md`:
```
**`module load` silently no-ops in a non-interactive shell.** It returns
success, prints nothing, and leaves `PATH` unchanged.
…
/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/r/4.5.0/bin/Rscript
```

## 3.3 Random seeds

| Seed | Value | Location |
|---|---|---|
| negative subsampling | 42 | `pub/scripts/make_splits.py:66`, applied `:71` |
| RF | 42 | `pub/scripts/train_rf.py:89` → `random_state=args.seed` (`:148`) |
| LR | 42 | `pub/scripts/train_lr.py:77` → `random_state=args.seed` (`:134`) |
| CNN | 42 | `pub/scripts/train_cnn.py:127`; seeds `random`, `numpy`, `torch`, `torch.cuda` at `:52-56` |
| Fig 0.4 background sampling | 0 | `pub/scripts/make_figure1_data.py:147` (`seed=0`), `:162` (`np.random.default_rng(seed)`) |
| Fig 0.5 dinucleotide shuffle | 0 | `pub/scripts/featurize_shortlist_cpa.py:195`, `:242` (`rng = random.Random(a.seed)`) |
| Fig 0.5 bootstrap | 0 | `pub/scripts/figure2_stats.py:117` (`np.random.default_rng(0)`) |
| gene 20 % holdout (rerank line of work) | 42 | `pub/figure_rerank_primary_notes.md:23-24`, `pub/figure_corr_notes.md:44-45` |
| Fig-3/SpliceAI gene holdout | **no RNG at all** | `pub/REPRODUCIBILITY.md` §6 — `The gene 50/50 holdout is \`md5("0:" + gene) & 1\`. Python's built-in \`hash()\` is deliberately avoided because it is salted per process. No RNG anywhere. Reruns are byte-identical.` |
| upstream pipeline env | `PYTHONHASHSEED=0`, seed 0 | `pub/REPRODUCIBILITY.md` Appendix — `(invocation in \`scripts/slurm_figure3_data.sh\`, \`~/splice_env2\`, \`--imbalanced --max_per_class 70000\`, seed 0, \`PYTHONHASHSEED=0\`)` |

## 3.4 HPC structure

No Snakemake, no Nextflow, no Makefile. Everything is SLURM. The training/scoring
DAG for Figures 0.1/0.2/0.4 is a single script, `pub/scripts/submit_all.sh:14-24`:
```
FEAT=$(sbatch --parsable scripts/featurize_all.slurm)
RF=$(sbatch --parsable --dependency=aftercorr:$FEAT scripts/train_rf_array.slurm)
LR=$(sbatch --parsable --dependency=aftercorr:$FEAT scripts/train_lr_array.slurm)
CNN=$(sbatch --parsable scripts/train_cnn_array.slurm)
```
with the ordering rationale at `:4-5`:
```
# RF/LR wait per-fold on the matching featurize task (aftercorr), so fold k's RF starts
# as soon as fold k's featurization finishes, not all of them.
```
Array indices 0–5 map to `(fold0, fold1, fold2, fold3, fold4, prod)`
(`featurize_all.slurm:17`, `score_array.slurm:26`).
Scoring resources: `#SBATCH --gres=gpu:h100:1 --cpus-per-task=8 --mem=128G --time=12:00:00`,
throttled `--array=0-5%2` to avoid GPU contention (`score_array.slurm:3-14`).
Other SLURM wrappers: `pub/scripts/{cnn_array,train_*_array,score_array_cpu,slurm_prod_*}.slurm`,
`pub/scripts/slurm_*.sh`, plus ~30 in `scripts/`.

Run logs are on disk as `*.out` / `*.err` in the root, `pub/logs/`, `scripts/logs/`,
`logs/` (e.g. `threess_prof_46303975.err` at 16 MB), so job IDs are recoverable.

## 3.5 Data and accession identifiers (consolidated)

| Accession | Description | Build | Used for |
|---|---|---|---|
| **GSE103252** | Oh 2017 (NSMB), Dreyfuss lab: control AMO and U1 AMO, oligo-dT 3'-seq, HeLa | hg19 | Fig 0.8 |
| **GSE111310** | Singh 2018 (Nat Commun): 3'-seq IpA, immune subsets / myeloma / cell lines | hg19 | Fig 0.8 |
| **GSE111793** | Lee 2018 (Nature): 3'-seq IpA, CLL (reuses Singh's immune samples) | hg19 | Fig 0.8 |
| **GSE123105** | Kamieniarz-Gdula 2019, Proudfoot: QuantSeq REV 3' mRNA-seq, nuclear RNA, HeLa; GSMs 3536480/3536484/3536487/3536493 | hg19 | registered; not in `figure_overlap_filtered` |
| **GSE135140** | So 2019: U1 AMO, primary | hg38 | downloaded; used in the earlier `fig_site_overlap` run |
| **GSE100537** | Chiu 2018 | mm9 | **excluded** (`pub/datasets.yaml:231-235`) |
| GSE125534 / GSE125535 | Beckedorff 2020 | hg19 | **excluded** (`pub/datasets.yaml:236-239`) |
| DOI 10.1038/s41586-018-0465-8 | Lee 2018 supplement | — | `pub/external/supp/` |
| DOI 10.1038/s41467-018-04112-z | Singh 2018 supplement | — | `pub/external/supp/` |
| DOI 10.1038/nsmb.3473 | Oh 2017 supplement | — | `pub/external/supp/` |
| bitbucket `leslielab/apa_2018` | Leslie-lab code/data | — | vendored `pub/external/apa_2018/` |
| PolyASite 2.0 | `atlas.clusters.2.0.GRCh38.96.tsv`, 569,005 clusters | GRCh38 | Fig 0.1, 0.4, 0.5, 0.8 |
| PolyA_DB (v3/v4) | `hg38.PAS.main.tsv`, `hg38.PAS.max.tsv`; main BED 303,312 sites | hg38 | Fig 0.1, 0.3, 0.5, 0.8 |
| GENCODE v49 | `gencode.v49.annotation.gtf` | hg38 | throughout |
| UCSC SpliceAI Wildtype | `spliceAiAcceptor{Plus,Minus}.bw`, `spliceAiDonor{Plus,Minus}.bw` | hg38 | Fig-3 line of work (not one of the 8) |
| UCSC hg19→hg38 chain | `hg19ToHg38.over.chain.gz` | — | Fig 0.8 |
| ENCODE eCLIP (present, not in the 8) | `ELAVL1_eclip_K562.bed`, `TIA1_eclip.bed` | — | supplementary overlap tests |
| CISBP-RNA | `cisbp_rna.meme`, `RBP_Information*.txt`, `Homo_sapiens_2026_06_25_4:21_pm.zip` | — | motif matching (TOMTOM) |

## 3.6 Version control

`pub/` **is** a git repository: HEAD `674764d` ("Prune the binned splice model…"),
913 tracked files including all `.py`, `.R`, `.md`, and the `.tsv` figure data.
This supersedes `pub/REPRODUCIBILITY.md` §9-1 ("**Nothing is in version control.**
`git ls-files reanal/` returns 0"), which was written 2026-07-21 before the
repository was initialised. The enclosing `clean/` directory is inside a *different*
repo (branch `main`, HEAD `f6dc96d`) that does **not** track `reanal/`.
`pub/models/prod/rf/*.joblib` (15.7 GB each) and `pub/spliceai_bw/` (3.4 GB) are
not tracked.

---

# 4. Statistical methods, consolidated

| Method | Where | Notes |
|---|---|---|
| Mann-Whitney U / AUROC with midranks, inline | `pub/scripts/figure2_stats.py:49-60`; `pub/CLAUDE.md` | `scipy`/`sklearn` deliberately avoided in the `pub/` analysis layer |
| Average precision (AUPRC) by the step rule + no-skill baseline | `pub/scripts/figure2_stats.py:63-74` | |
| Bootstrap CI over genes | `figure2_stats.py:110-111` (1,000 / 200 resamples) | |
| Wilson 95 % CI on proportions | `pub/figure_overlap_notes.md:78` | Fig 0.8 panel a |
| Circular-shift permutation null (10–100 kb, 200 rounds) | `pub/datasets.yaml:27-28`; `pub/figure_overlap_notes.md:78-80` | p floors at 0.005 |
| Within-gene label-shuffle permutation null | `pub/figure_rerank_primary_notes.md:55-58` (200×); `pub/figure_u1_hits_notes.md:59` (10,000×) | |
| Cliff's delta with gene-clustered CI | `pub/figure_u1_hits_notes.md:13-15` | |
| Spearman / Pearson correlation | `make_figure1_data.py:306`; `pub/figure1_legends.md:7` | Fig 0.4 panels B, G |
| IRLS logistic regression with Wald SEs, inline | `pub/CLAUDE.md`; `pub/REPRODUCIBILITY.md` §1 | Fig-3 line of work |
| Wilcoxon signed-rank, inline | `pub/REPRODUCIBILITY.md` §5 | |
| Fisher exact / hypergeometric / binomial | 9 / 6 / 9 files respectively | GO and enrichment side-analyses |
| Dinucleotide-preserving shuffle (Altschul & Erikson 1985) | `pub/scripts/featurize_shortlist_cpa.py:129-132` | Fig 0.5 panel C |
| **Benjamini–Hochberg FDR** | `pub/master_grid.py:18`; `pub/gridsearch_direction.py:10` | **Applied only in grid-search side-analyses. NOT applied in Figures 0.3, 0.4, 0.5 or 0.8.** |
| Bonferroni | — | never used |

---

# 5. Gaps / could-not-verify

Ordered by how much they would change a Methods section.

1. **Figure 0.4 panel A metrics are unsourced and disagree with the model
   metadata.** `pub/scripts/make_figure1.R:84` hard-codes
   `RF 0.987/0.851/0.781, LR 0.983/0.840/0.761, CNN 0.991/0.923/0.872`.
   The locked test metrics in `pub/models/prod/{rf,lr,cnn}/metadata.json` are
   `RF 0.9874/0.8413/0.7717, LR 0.9827/0.8274/0.7584, CNN 0.9912/0.9227/0.8650`.
   AUROC matches for all three; AUPRC and F1 do not for RF and LR, and F1 does not
   for CNN. No file in the tree contains the figure's values, and no cross-fit fold
   reproduces them. **Cannot be resolved from the repository.**

2. **The production CNN was trained on CPU, not the requested H100 — and this is
   recorded nowhere but the logs.** `pub/models/prod/cnn/metadata.json` has
   `"amp": false` with `train_time_sec 1623`, and `pub/logs/cnn_46303739_5.out`
   prints `Device: cpu`. Folds 0-3 ran on GPU with AMP in ~52-68 s; fold4 and prod
   fell back to CPU (~1424 s, ~1623 s), almost certainly because the CNN array is
   unthrottled (`--array=0-5`) while the scoring array is not (`--array=0-5%2`).
   So the CNN behind every Figure 0.4 number is an fp32 CPU model that a GPU rerun
   at seed 42 will not reproduce bit-for-bit, and fold4 is not numerically
   comparable to folds 0-3. **Decide whether to retrain uniformly or to state this
   in Methods.**

3. **376,057 (Figure 0.1) does not exist anywhere.** Full-tree grep over every text
   file, including all SLURM run logs, returns nothing. Nearest real counts: 303,312 / 311,500 / 363,009 / 429,849 /
   441,546 / 569,005. **Cannot be resolved from the repository.** (By contrast **15,794**,
   the other Figure 0.1 label, *is* confirmed — it is printed in the shortlist and
   profile run logs as the canonical-window count.)

4. **27,856 (Figure 0.8) does not exist anywhere**, including in the run logs. Nearest lexical match is
   **27,976** (Oh U1-AMO blocks discarded by the top-40,000 cut,
   `pub/figure_overlap_notes.md:48,67`). **Cannot be resolved from the repository.**

5. **PolyA_DB version: v3 or v4?** The figure and its script say **v4**
   (`pub/sfig1.R:3,70,133`); the data-generating script says **v3**
   (`pub/scripts/select_constitutive_cpa.py:5,14,20,97`). The directory name says v4
   (`train/data/polyadb_v4/`). Both read the same `hg38.PAS.main.tsv`, which carries
   no version tag. **Author must resolve.**

6. **Figure 0.3's 15,700 vs the 15,794-row positive BED.** The funnel's terminal
   stage is 94 sites short of the file every downstream step actually uses
   (15,785 after chrY exclusion). No script explains the difference.

7. **Figures 0.6 and 0.7 have no computational provenance in this directory.**
   No `eGFP`, `Sanger`, `subclone`, `reporter`, `minigene`, `transfect`, `RT-PCR`,
   `dual-CPA`, or `~600 nt` construct anywhere. The `HEK293` and `oligo-dT` hits are
   descriptions of *published* datasets in `pub/datasets.yaml`, and `leukemia` /
   `immune` are GEO-download comments. Which analysis output supplied the
   "predicted" constructs is not recorded; the most likely candidates are
   `pub/hiconf_out/sites/oh_u1amo.tsv` or the 274 U1-AMO-hit false positives of
   `pub/figure_u1_hits_notes.md`. **Author must confirm.**

8. **No environment lockfile — but the environments themselves survive, so this is
   now RESOLVED by inspection.** `pub/REPRODUCIBILITY.md` §5 documents only
   `~/splice_env2` (the *analysis* env). The env that actually trained the models is
   `~/venv_2025`: **Python 3.10, torch 2.6.0, scikit-learn 1.2.1, numpy 1.26.4,
   pandas 2.2.3, scipy 1.15.1** (§3.2 for the full three-way table). Two caveats
   remain: nothing on disk *pins* these (a rebuild would drift), and the version
   skew across the three envs is large (sklearn 1.2.1 vs 1.7.1 vs 1.8.0; numpy
   1.26 vs 2.3 vs 2.4), so analyses run in `splice_env2` against models trained in
   `venv_2025` cross a major-version boundary in both numpy and scikit-learn.
   Recommend writing a `requirements.txt` from `venv_2025` before the env is lost.

9. **The PWM bank lives outside the analysis root.** `/scratch/spour98/identify_model/pwm_paths`
   and `BaselineModelPWMs/*.txt` (10 PWMs) are not copied into, or tracked by, this
   directory. Ten of the fifteen feature families in every RF/LR result depend on
   them. The individual PWM matrices' provenance (which paper each came from) is not
   recorded beyond the filenames `PAS_Hu.txt`, `PAS_Siepel.txt`, `UGUA_Hu.txt`, `NUDT_PWM.txt`.

10. **`negatives_w_pa_clean.bed` / `negatives_no_pa_clean.bed` have no generating
   script in this tree.** Both are dated 2025-05-17 and are copied in from an earlier
   directory. The rule that decides "PAS-containing" versus "no PAS" — which hexamer
   set, which window — is therefore **unverifiable**. This is a load-bearing gap:
   the decoy/non-decoy 1:6 ratio is a stated design parameter of the negative set.

11. **`Dominant_CPA_baseline_total.bed` was not regenerated by the recorded command.**
    `pub/scripts/select_constitutive_cpa.py` writes `constitutive_cpa.{tsv,bed}`; the
    exact invocation that produced `Dominant_CPA_baseline_total.bed` (which `--source`,
    whether `--ucsc_refgene` was supplied, whether `--min_score` was set) is not
    recorded in any shell script or log. The script's own note says the longest-isoform
    step was approximated when `--ucsc_refgene` is absent, but nothing records whether
    it was absent.

12. **Figure 0.5's 0.470 is unmatched.** `0.478` resolves cleanly to the pooled AUROC
    (0.4782); the second number does not appear. Nearest candidates are the within-gene
    rank `0.4768` and the 2-competitor bin `0.478`. **Check the rendered panel.**

13. **Figure 0.5 panel denominators are remapped from stale keys.**
    `pub/scripts/make_figure2.R:75-78` remaps `fig2_counts.tsv`'s old panel letters
    onto the new positions. The counts printed under panels C and F therefore come
    from a key translation, and `fig2_counts.tsv` itself still records the *old*
    panel-D density set at 50,130 peaks while the rest of the figure uses 43,547.
    Verify what is actually printed before quoting.

14. **Figure 0.8: 273/83 vs 274/87.** Both are correct on their own denominator
    (dataset-side vs peak-side); `pub/figure_u1_hits_notes.md:7-10` documents this.
    State which the figure uses.

15. **Figure 0.8: no multiple-testing correction on six reported p-values.**
    Benjamini–Hochberg is used elsewhere in the project but not here.

16. **Two parallel script trees with overlapping filenames.** `scripts/` and
    `pub/scripts/` both contain `train_rf.py`, `train_lr.py`, `train_cnn.py`,
    `cpa_cnn.py`, `featurization.py`, `make_splits.py`, `score_rerank1.py`,
    `build_shortlist.py`, `make_figure1.R`, `paired_suppression_figure.R`, …
    `pub/scripts/` is the final one (`pub/CLAUDE.md`: "**Never edit the pipeline.**
    `scripts/*.py` and `scripts/*.R` are inputs"). All line references in this
    document are to the version stated. Do not quote the `scripts/` copies for
    the eight figures.

17. **The repo's own figure numbering conflicts with the thesis's.**
    `pub/scripts/fig1_8_model_timeline.R:2` declares itself "Figure 1.8"
    (a literature timeline of 13 published CPA models, 1999–2022) and
    `pub/figs/fig1..fig5 + fig_composite` are a *different* five-panel composite
    (Aug 16–17). Neither belongs to the eight figures analysed here. Do not
    cross-reference by number.

18. **`pub/REPRODUCIBILITY.md` contains three superseding appendices that reverse
    each other** (2026-07-27 "U1 layer REFUTED", 2026-07-28 "REVERSAL — U1 layer
    reinstated", 2026-07-28 "v3_2 terminal-exon reframe"). Those concern the
    Figure-3/SpliceAI work rather than the eight figures, but if any of that
    material reaches the chapter, read the appendices in order and quote only the
    last. §9-1 of the same document ("Nothing is in version control") is now stale —
    `pub/` is a git repo with 913 tracked files.

19. **`clean_final/` is empty** and `figdata_final/` (Jun 28) versus
    `pub/figdata_final/` (Jul 2) hold two different frozen copies of the rerank
    figure data with different values (e.g. `rerank_genomewide.tsv` is 2,069 vs
    1,416 bytes). Neither feeds the eight figures, but do not mine them for numbers.
