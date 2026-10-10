**phd_work**

Code and figures for my thesis chapters on how a gene's 3′ end is chosen.

- chapter 2: does local sequence separate constitutive 3′ ends from competitor cleavage sites
- chapter 3: wider window, and the splicing environment upstream of a 3′ end
- data, model weights, scans and job logs are not included
- scripts read inputs from the original cluster layout (`reanal/clean/` and `reanal/clean/pub/`), so paths inside them refer to that tree

**Layout**

| Folder | Contents |
|---|---|
| `pipeline/` | chapter 2: data, splits, features, the three models, genome scan, candidate tables, Figures 2.3 and 2.5 |
| `baseline_models/` | chapter 2 helpers the jobs ran: gene-scan aggregation, APARENT2 scores, shortlist featurizer, CISBP-RNA conversion |
| `splice_models/` | MaxEnt and U1 lookups, tool installs |
| `analysis/` | analyses for chapters 2 and 3, by topic |
| `thesis_ch3/analysis/` | chapter 3 and 4 analyses |
| `thesis_ch3/figures/` | chapter 3 figure scripts |
| `plotting/` | panel scripts the thesis figures draw from |
| `jobs/` | slurm scripts for the above |
| `figures/` | the 19 thesis figures (`pdf/`, `png/`) and `make_nature_figs.R` |
| `docs/` | per-figure notes, methods digest, legends |
| `testing/` | exploratory work and drafts that are not in the thesis, same layout as above |

- everything outside `testing/` went into the thesis
- `testing/` holds old figure drafts, the MaxEnt vs SpliceAI figure, BPNet-style and terminal-exon models, repeat and TE analyses, one-off checks

`analysis/` topics:

- `acceptor`: 3′ splice site
- `u1`: U1 and 5′ splice sites
- `splice_model`: the 26-feature model and splice-site scans
- `networks`: the chapter 3 CNN and its interpretation
- `implantation`: in silico insertion tests
- `figure_data`: tables for specific figures
- `data_build`: caches and candidate tables
- `tasks`: analysis-set build and checks (`task_*`)
- `other`: overlap and recovery

**Run order, chapter 2**

- positives and external data: `pipeline/select_constitutive_cpa.py`, `pipeline/00_download_and_liftover.sh`, `pipeline/00b_verify_and_repair.sh`, `pipeline/01_fetch_gse135140.sh`, `pipeline/04_bw_peaks.py`
- splits: `pipeline/make_splits.py`, `pipeline/make_prod_split.sh`, `pipeline/assign_folds.py`, `pipeline/split_genes_by_fold.sh`
- features: `pipeline/featurization.py`
- models: `pipeline/train_lr.py`, `pipeline/train_rf.py`, `pipeline/train_cnn.py` (network in `pipeline/cpa_cnn.py`)
- genome scan: `pipeline/score_genes.py` via `pipeline/slurm_prod_sharded.sh` and `pipeline/slurm_prod_cpu_sharded.sh`, then `pipeline/merge_fold_scans.sh`, `baseline_models/aggregate_gene_scans.py`, `baseline_models/score_genes_aparent2.py`
- candidates: `pipeline/build_shortlist.py`, `pipeline/featurize_shortlist.py`, `pipeline/featurize_shortlist_cpa.py`, `pipeline/atlas_join.py`
- external overlap: `pipeline/03_simple_overlap.py`, `analysis/tasks/task_overlap_prep.py`, `analysis/tasks/task_overlap_accounting.py`, `analysis/other/overlap_model_scores.py`, `analysis/other/overlap_rf_dense.py`, `analysis/other/overlap_shared_recompute.py`
- figure tables: `pipeline/make_figure1_data.py` (2.3), `pipeline/figure2_stats.py` (2.5), `analysis/data_build/trace_analysis_set.py` (2.4)

**Run order, chapter 3**

- analysis set: `analysis/tasks/task_build_final_parquet.py` (`cpa_peaks_final.parquet`), `analysis/tasks/task_decoy_build.py`, `analysis/data_build/` caches
- splice-site scores: `splice_models/build_me5_lookup.py`, `splice_models/build_u1_lookup.py`, `pipeline/figure3_data.py`, `analysis/splice_model/splice_score_redundancy.py`, `analysis/splice_model/binned_profile_scan.py`
- calibration: `thesis_ch3/analysis/maxent_calibration.py`, `thesis_ch3/analysis/u1_calibration.py`, `thesis_ch3/analysis/u1_null_analysis.py`, `analysis/u1/u1_threshold_calibrate.py`
- hypothesis: `thesis_ch3/analysis/hypothesis_dose.py`, `thesis_ch3/analysis/hypothesis_grid.py`, `thesis_ch3/analysis/pas_controlled.py`
- use after U1 inhibition (Figure 3.5 inputs): `thesis_ch3/analysis/occupancy_extract.py`, `thesis_ch3/analysis/oh2017_sites.py`, `thesis_ch3/analysis/u1_replication.py`
- splice model: `analysis/splice_model/splice_model_26.py`, `analysis/splice_model/splice26_loco.py`, `thesis_ch3/analysis/splice_model_expand.py`, `thesis_ch3/analysis/splice_model_boost.py`, `thesis_ch3/analysis/recalibrated.py`
- network: `analysis/networks/cnn_train.py` (model in `analysis/networks/cnn_model_def.py`), `analysis/networks/cnn_loco_train.py`, `analysis/networks/cnn_wg_loco.py`
- robustness: `thesis_ch3/analysis/context_sweep.py`, `thesis_ch3/analysis/splice_scramble.py`, `thesis_ch3/analysis/patching.py`, `analysis/acceptor/acceptor_arch_train.py`
- interpretation: `analysis/networks/cnn_interpret.py`, `analysis/networks/cnn_posmap_occlusion.py`, `thesis_ch3/analysis/interpret_deep.py`, `thesis_ch3/analysis/concept_erasure.py`, `thesis_ch3/analysis/filter_annotation.py`
- implantation: `thesis_ch3/analysis/implant_mech.py`, `thesis_ch3/analysis/implant_cisbp.py`, `thesis_ch3/analysis/implant_position.py`, `thesis_ch3/analysis/mechanism.py`, `analysis/implantation/`, `analysis/acceptor/`
- exonic context: `thesis_ch3/analysis/exon_context.py`, `thesis_ch3/analysis/fig3_10_data.py`, `thesis_ch3/analysis/context_matched.py`, `thesis_ch3/analysis/window_features.py`, `thesis_ch3/analysis/exon_content_analysis.py`, `thesis_ch3/analysis/stress_test.py`, `thesis_ch3/analysis/repeat_context.py`
- recovery: `analysis/other/rescue_rate.py`, `analysis/other/rescue_ranks.py`
- statistics: `thesis_ch3/analysis/stats_ch3.py`

**Run order, chapter 4**

- usage in 3′ end sequencing data: `thesis_ch3/analysis/usage_atlas.py`, `thesis_ch3/analysis/usage_ratio.py`, `thesis_ch3/analysis/usage_metaprofile.py`
- sequence design: `thesis_ch3/analysis/design_relay.py`

**Thesis figures**

| Figure | Script |
|---|---|
| 2.1 | `plotting/sfig1.R` |
| 2.3 | `pipeline/make_figure1.R` |
| 2.4 | `plotting/make_analysis_set_funnel.R` |
| 2.5 | `pipeline/make_figure2.R` |
| 2.9 | `plotting/make_figure_overlap_shared.R` |
| 3.1 | `thesis_ch3/figures/fig3_01_hypothesis.R` |
| 3.2 | `thesis_ch3/figures/fig3_02_genesets.R` |
| 3.3, 3.4, 3.6, 3.7, 3.8, 3.9, 3.11 | `thesis_ch3/figures/fig3_regroup.R` (panels from `plotting/make_splicing_environment.R`, `make_splice_model_final.R`, `make_figure_architecture.R`, `make_figure_rescue.R`, `make_figure_loco.R`, `make_figure_cnn_interpretability.R`) |
| 3.5 | `thesis_ch3/figures/fig3_hypothesis_dose.R` |
| 3.10 | `thesis_ch3/figures/fig3_09_robustness.R` |
| 3.12 | `thesis_ch3/figures/fig3_13_mechanism.R` |
| 3.13 | `thesis_ch3/figures/fig3_implant_panel.R` |
| 3.14 | `thesis_ch3/figures/fig3_14_recovery.R` |

- Figures 2.2 and 2.6 to 2.8 are drawings, gels and sequencing traces
- `figures/make_nature_figs.R` runs each script above and restyles the plot: lowercase panel letters, no titles, 7 pt Helvetica, no gridlines, 183 mm wide
- per-figure notes, key numbers and the scripts behind each panel: [docs/thesis_figures/README.md](docs/thesis_figures/README.md)

**Software**

- R 4.5.0, ggplot2 4.0.3, patchwork
- Python 3, numpy, pandas, pyarrow, PyTorch, maxentpy
- chapter 3 analyses use no scikit-learn or scipy; AUROC, logistic regression and Wilcoxon are written inline
