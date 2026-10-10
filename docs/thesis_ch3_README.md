# Thesis Chapter 3 figures

Remake of the Chapter 3 figures for `SP_thesis_0826.docx`, restructured to match the
figure density of the Laverty (2023) reference thesis: two-level numbering (Figure 3.N),
one idea per figure.

Scripts here only read tables in `pub/` (`../`). Nothing in `pub/` or `scripts/` is
modified. Outputs go to `out/`.

    R/           plotting scripts, one per thesis figure (fig3_NN_*.R)
    py/          new analyses (context-length sweep)
    schematics/  conceptual diagrams
    out/         PDF + PNG, named by thesis figure number
    logs/        sbatch output

## Figure map

| Thesis fig | Content | Old fig | Source script (pub/) | Inputs | Needs rerun? |
|---|---|---|---|---|---|
| 3.1  | Schematic: terminal exon vs intronic competitor, donor/U1 hypothesis | none | new | none | no, drawn |
| 3.2  | Schematic: gene universes 15,560 > 6,583 > 4,847 > 13,824 | none | new | analysis_set_funnel.tsv | no |
| 3.3  | Donor-depletion metagene, within-gene U1 difference | 3.2.1 A,B | make_splicing_environment.R | splice_env_*.tsv | no |
| 3.4  | U1 vs MaxEnt correlation, threshold selection | 3.2.1 C,D | make_splicing_environment.R | splice_score_*.tsv | no |
| 3.5  | Splice model construction + coefficients | 3.2.2 A,B | make_splice_model_final.R | splice_model_26_*.tsv | no |
| 3.6  | Splice model within-gene recall, score separation | 3.2.2 C,D | make_splice_model_final.R | recall_at_k.tsv, score_separation*.tsv | no |
| 3.7  | CNN architecture + context spans | 3.2.3 arch, 3.2.5A | make_figure_architecture.R, make_figure_rescue.R | none | no |
| 3.8  | CNN vs splice: ROC/PR, within-gene, genome-wide | 3.2.3 A,B,D | make_figure_loco.R | figLC_*.tsv, wg_*.tsv, cnn_wholegenome_p1.tsv | no |
| 3.9  | Robustness: shuffle, seeds, feature ablation, context sweep | 3.2.3 C | make_figure_loco.R + new | loco_shuffle.tsv, acceptor_arch_train.tsv, acceptor_options_metrics.tsv, rerank_ablation.tsv, **context sweep** | **context sweep only** |
| 3.10 | Filter logos, positional occlusion, implantation | 3.2.4 | make_figure_cnn_interpretability.R | cnnmain_*.tsv, panelB_prob_ci.tsv, ... | no |
| 3.11 | Recovery flow, single estimand | 3.2.5B | make_figure_rescue.R | rescue_ranks.tsv, rescue_rate_genes.tsv | no |

## Open decisions

- Headline P@1: one estimand throughout. Candidates: CNN genome-wide 0.821
  (cnn_wholegenome_p1.tsv) or combined z-sum on the 13,824 universe 0.893. The
  per-gene oracle 0.921 is an upper bound, report as such, not as a headline.
- Seeds: base CNN is 0.832 +/- 0.013 over 8 seeds (acceptor_arch_train.tsv);
  thesis currently reports one seed (0.821).
- calibration_diagnosis.tsv records temperature 1225.8. Check before use.

## Status

All eleven figures built (out/fig3_01 .. fig3_11). Chapter text: chapter3/chapter3.md -> chapter3/SP_chapter3_revised.docx
(pandoc, styles from ../SP_thesis_0826.docx; unreferenced media from the reference doc stripped).

| Fig | Script |
|---|---|
| 3.1, 3.2 | R/fig3_01_hypothesis.R, R/fig3_02_genesets.R (drawn; 3.2 counts asserted against tables) |
| 3.3-3.8, 3.10 | R/fig3_regroup.R (panels harvested from pub/ scripts via R/_harvest.R) |
| 3.9 | R/fig3_09_robustness.R (context sweep, splice scramble, shuffle, seeds) |
| 3.11 | R/fig3_11_recovery.R (P@1 vs chance with gene bootstrap; flow relabelled) |

New experiments: py/context_sweep.py (job 61822760), py/splice_scramble.py (job 61824148).
Both reproduce acceptor_arch_train.tsv base seeds exactly on intact input.

Known issues carried forward:
- Two LOCO fits of the splice model: 0.465 (recall_at_k.py / loco_figure_data.py, Figs 3.6, 3.8) and
  0.463 (splice26_loco_scores.npz, combined ranking and Fig 3.11). Disclosed in Methods 3.4.4.
- make_figure_cnn_interpretability.R labels donor CAGGTGAGT with score5 10.86 (that is CAGGTAAGT's
  score); maxentpy gives 10.67. Fixed in fig3_regroup.R by relabelling.

## Third pass (2026-09-27)

New analyses:
- py/maxent_calibration.py -> data/maxent_calibration.tsv (Fig 3.4C): MaxEntScan sensitivity at 99,710
  annotated acceptors/donors vs background calls in 2,000 competitor windows.
- py/exon_context.py -> data/exon_context.tsv, data/exon_bins.npz: GENCODE v49 exon coverage per window/bin.
- py/patching.py (GPU, job 61827403) -> data/patching.npz, data/scramble_logits.npz: input patching and
  per-window logits for the scramble arms.
- py/splice_scramble.py resample mode (job 61825548) -> data/splice_resample.tsv.
- py/fig3_10_data.py + R/fig3_10_exon_context.R -> new Figure 3.10.
- py/stats_ch3.py -> data/stats_ch3.tsv: every test and effect size cited in the chapter.
Renumbered: old 3.10 -> 3.11 (interpretation), old 3.11 -> 3.12 (recovery, R/fig3_12_recovery.R).

## Fourth pass (2026-09-27)

- Figures now 3.1-3.14: new 3.11 (R/fig3_11_content.R: repeats, surrogates, patching by type, 6-mers) and
  3.13 (R/fig3_13_mechanism.R: implant position profile, annotated splice-site knockouts, architecture,
  UGUA/PAS scans). Old 3.11 -> 3.12 (interpretation), old 3.12 -> 3.14 (R/fig3_14_recovery.R; panel B now
  all 15,560 genes from ../wg_per_gene.tsv).
- New scripts: py/implant_position.py (job 61829002), py/mechanism.py (job 61829944), py/repeat_context.py,
  py/window_features.py, py/exon_content_analysis.py, py/recalibrated.py, py/u1_calibration.py,
  py/fig_content_data.py, py/fig_mech_data.py.
- Calibrated (composition-corrected, stricter) features lowered splice-model performance and are NOT used;
  thresholds are placed on the calibration curve instead (Fig 3.4C, with U1 points).

## Fifth pass (2026-09-27): stress test and repeat investigation (not yet in the chapter)

- py/stress_test.py -> data/stress_test.tsv: network vs intronic (0.847), terminal-exon (0.739) and annotated
  alternative-site competitors (0.753), whole-shortlist LOCO scores. Flags from cpa_peaks_final.parquet
  reproduce the funnel (18,119; 1,076). NOTE corr_rowmask.npz is NOT in wg/meta row order (ridx).
- py/context_matched.py -> data/context_matched.tsv: both sites >=90% exonic and <=10% repeat: network 0.625
  [0.561, 0.687], annotated exon 0.512 (219 genes, 262 pairs).
- py/repeat_deep.py -> data/repeat_deep.tsv: PAS present at 90-94% of repeat competitors (80% of real ends);
  structure (RNAfold) not generally different except Alu; repeat-site competitors near annotated PAS 21.0% vs
  27.6%; network penalises antisense L1 (-1.02), Alu (-0.66), LTR (-0.38), not sense L1 or DNA transposons.
- py/repeat_position_background.py, py/repeat_site_positions.py: LTR-sense sites 66% in last 20% of the
  consensus (background 22%, chi2 2858), Alu sense at 5' end and central linker, Alu antisense at 3' end;
  L1 positions follow the 5'-truncation background.
- py/repeat_shuffle.py (job 61834259): shuffling +/-300 nt flanks raises Alu competitors +0.77, LTR sense
  +0.22, L1 antisense +0.11; lowers real ends -0.22.

## Sixth pass (2026-09-27): shortened chapter, mechanism-oriented

Chapter cut from 13.3k to 9.9k words, 13 figures and 1 table. Previous version kept in `chapter3_v1/`.
Removed: exon-context section and figure (old 3.10), repeat/content section and figure (old 3.11),
their Methods, and all repeat deep-dive material (scripts and data kept, not cited).
Figure files keep their old names; chapter numbering is now
3.10 = out/fig3_12_interpretation, 3.11 = out/fig3_13_mechanism, 3.12 = out/fig3_u1amo, 3.13 = out/fig3_14_recovery.

New analyses
| Chapter item | Script | Output |
|---|---|---|
| Table 3.1, poly(A) signal held fixed | py/pas_controlled.py | data/pas_controlled.tsv |
| Section 3.2.4, reduced vs terminal-exon feature model | py/splice_model_expand.py | data/splice_model_expand.tsv |
| Fig 3.12, U1 AMO derepression (So et al. 2019, GSE135140) | py/occupancy_extract.py (+ `decoy`), py/u1amo_derepression.py, R/fig3_u1amo.R | data/u1amo_derepression.tsv, data/u1amo_metagene.tsv |

Not used in the chapter: py/occupancy_analysis.py (XLIP occupancy at sites). The positive control failed
(hnRNPC signal falls with U content, rho = -0.16), so site-level XLIP signal is confounded by composition.
