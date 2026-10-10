# phd_work

Code and figures for the thesis chapters on choosing a gene's 3′ end.
Chapter 2 asks whether local sequence separates constitutive 3′ ends from competitor
cleavage sites; Chapter 3 widens the window and tests the splicing environment upstream.

Data, model weights, scans and job logs are not included. Scripts read inputs from the
original cluster layout (`reanal/clean/` and `reanal/clean/pub/`), so paths inside them
refer to that tree, not to this repository.

## Layout

| Folder | Contents | Original location |
|---|---|---|
| `baseline_models/` | Chapter 2 random forest, logistic regression and CNN: features, training, genome scans, reranking | `clean/scripts/` |
| `splice_models/` | splice-feature models and benchmarks | `clean/train/splice_train/scripts/` |
| `pipeline/` | candidate-site pipeline, Chapter 2 figure data and figures 2.3 and 2.5 | `pub/scripts/`, `pub/scripts2/` |
| `analysis/` | analyses behind Chapters 2 and 3, grouped by topic (below) | `pub/*.py` |
| `thesis_ch3/analysis/` | Chapter 3 analyses and GPU experiments | `pub/thesis_ch3/py/` |
| `thesis_ch3/figures/` | Chapter 3 figure scripts | `pub/thesis_ch3/R/` |
| `plotting/` | figure scripts whose panels the thesis figures use, and earlier figure drafts | `pub/*.R` |
| `jobs/` | slurm submission scripts | `pub/*.sbatch`, `pub/*.sh` |
| `figures/` | the 19 thesis figures in Nature style (`pdf/`, `png/`) and `make_nature_figs.R` | `pub/nature_figs/` |
| `docs/` | per-figure provenance, reproducibility notes, methods digest, figure legends | `pub/`, `clean/` |

`analysis/` topics: `acceptor` (3′ splice site), `u1` (U1 and 5′ splice sites),
`splice_model` (the 26-feature model, SpliceAI and MaxEnt scorers), `networks` (CNN and
BPNet models and their interpretation), `implantation` (in silico insertion tests),
`figure_data` (tables for specific figures), `data_build` (caches and candidate tables),
`tasks` (one-off checks, named `task_*`), `other`.

## Thesis figures

| Figure | Script |
|---|---|
| 2.1 | `plotting/sfig1.R` |
| 2.3 | `pipeline/make_figure1.R` |
| 2.4 | `plotting/make_analysis_set_funnel.R` |
| 2.5 | `pipeline/make_figure2.R` |
| 2.9 | `plotting/make_figure_overlap_shared.R` |
| 3.1 | `thesis_ch3/figures/fig3_01_hypothesis.R` |
| 3.2 | `thesis_ch3/figures/fig3_02_genesets.R` |
| 3.3, 3.4, 3.6, 3.7, 3.8, 3.9, 3.11 | `thesis_ch3/figures/fig3_regroup.R`, which takes panels from `plotting/make_splicing_environment.R`, `make_splice_model_final.R`, `make_figure_architecture.R`, `make_figure_rescue.R`, `make_figure_loco.R` and `make_figure_cnn_interpretability.R` |
| 3.5 | `thesis_ch3/figures/fig3_hypothesis_dose.R` |
| 3.10 | `thesis_ch3/figures/fig3_09_robustness.R` |
| 3.12 | `thesis_ch3/figures/fig3_13_mechanism.R` |
| 3.13 | `thesis_ch3/figures/fig3_implant_panel.R` |
| 3.14 | `thesis_ch3/figures/fig3_14_recovery.R` |

Figures 2.2 and 2.6 to 2.8 are drawings, gels and sequencing traces; their images are in
`docs/thesis_figures/figures/`.

`figures/make_nature_figs.R` runs each script above unchanged, captures the plot it
saves, and restyles it: lowercase panel letters, no titles or subtitles, 7 pt Helvetica,
no gridlines, 183 mm wide. The data and values drawn are the same. The per-panel n that
the old subtitles carried belongs in each figure legend.

`docs/thesis_figures/THESIS_FIGURES_README.md` documents every figure: what it shows,
key numbers, inputs, how it was made and open issues. Its embedded code is the original
commented source; the thesis versions of the images are in `figures_original/`.

## Software

R 4.5.0 with ggplot2 4.0.3 and patchwork; Python 3 with numpy, pandas, pyarrow, PyTorch
and maxentpy. Chapter 3 analyses avoid scikit-learn and scipy; AUROC, logistic
regression and Wilcoxon tests are written inline.
