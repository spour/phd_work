**phd_work**

Code and figures for my thesis chapters on how a gene's 3′ end is chosen.

- chapter 2: does local sequence separate constitutive 3′ ends from competitor cleavage sites
- chapter 3: wider window, and the splicing environment upstream of a 3′ end
- data, model weights, scans and job logs are not included
- scripts read inputs from the original cluster layout (`reanal/clean/` and `reanal/clean/pub/`), so paths inside them refer to that tree

**Layout**

| Folder | Contents |
|---|---|
| `baseline_models/` | chapter 2 random forest, logistic regression and CNN: features, training, genome scans, reranking |
| `splice_models/` | splice-feature models and benchmarks |
| `pipeline/` | candidate-site pipeline, chapter 2 figure data, Figures 2.3 and 2.5 |
| `analysis/` | analyses for chapters 2 and 3, by topic |
| `thesis_ch3/analysis/` | chapter 3 analyses and GPU experiments |
| `thesis_ch3/figures/` | chapter 3 figure scripts |
| `plotting/` | figure scripts the thesis figures draw panels from, plus earlier drafts |
| `jobs/` | slurm scripts |
| `figures/` | the 19 thesis figures (`pdf/`, `png/`) and `make_nature_figs.R` |
| `docs/` | per-figure notes, reproducibility notes, methods digest, legends |

`analysis/` topics:

- `acceptor`: 3′ splice site
- `u1`: U1 and 5′ splice sites
- `splice_model`: the 26-feature model, SpliceAI and MaxEnt scorers
- `networks`: CNN models and their interpretation
- `implantation`: in silico insertion tests
- `figure_data`: tables for specific figures
- `data_build`: caches and candidate tables
- `tasks`: one-off checks (`task_*`)
- `other`

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
