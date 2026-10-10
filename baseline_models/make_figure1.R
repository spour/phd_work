#!/usr/bin/env Rscript
# make_figures.r render publication-grade svg figures from the tsv "data behind each

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(scales)
  library(stringr)
  library(viridis)
  library(patchwork)
  library(ggforce)
})

PAL <- list(
  purple    = "#7F77DD",
  amber     = "#BA7517",
  teal      = "#1D9E75",
  blue      = "#378ADD",
  coral     = "#D85A30",
  gray      = "#888780",
  purple_dk = "#3C3489",
  canon_red = "#DC2626"
)

MODEL_COLORS <- c(
  "RF"       = PAL$purple,
  "LR"       = PAL$amber,
  "CNN"      = PAL$teal,
  "APARENT2" = PAL$coral,
  "Combined" = PAL$purple_dk
)

REF_COLORS <- c(
  "PolyASite"     = "#85B7EB",
  "PolyA_DB Main" = "#378ADD",
  "PolyA_DB Max"  = "#0C447C",
  "Canonical"     = PAL$canon_red
)

CPCCPA_CLASS_COLORS <- c(
  "Constitutive"        = PAL$teal,
  "tier4_all_four"      = "#7E2C12",
  "tier3_exactly_three" = "#993C1D",
  "tier2_exactly_two"   = "#D85A30",
  "tier1_exactly_one"   = "#F0997B",
  "Random"              = PAL$gray
)

CPCCPA_CLASS_LABEL <- c(
  "Constitutive"        = "Constitutive\n(pos. control)",
  "tier4_all_four"      = "All 4 models",
  "tier3_exactly_three" = "Exactly 3 models",
  "tier2_exactly_two"   = "Exactly 2 models",
  "tier1_exactly_one"   = "Exactly 1 model",
  "Random"              = "Random\n(neg. control)"
)

CPCCPA_CLASS_ORDER <- c(
  "Constitutive", "tier4_all_four", "tier3_exactly_three",
  "tier2_exactly_two", "tier1_exactly_one", "Random"
)

pub_theme <- function(base_size = 10) {
  theme_minimal(base_size = base_size) +
    theme(
      panel.grid.major = element_line(colour = "#D3D1C7", linewidth = 0.25),
      panel.grid.minor = element_line(colour = "#E8E6DD", linewidth = 0.15),
      axis.line        = element_line(colour = "black",  linewidth = 0.4),
      axis.ticks       = element_line(colour = "black",  linewidth = 0.4),
      axis.text        = element_text(colour = "#2C2C2A"),
      axis.title       = element_text(colour = "#2C2C2A"),
      plot.title       = element_text(face = "plain", size = base_size + 2,
                                       hjust = 0, margin = margin(b = 6)),
      plot.subtitle    = element_text(colour = "#5F5E5A", size = base_size - 1,
                                       hjust = 0, margin = margin(b = 8)),
      plot.caption     = element_text(colour = "#5F5E5A", size = base_size - 2,
                                       hjust = 0),
      strip.background = element_rect(fill = "#F1EFE8", colour = NA),
      strip.text       = element_text(colour = "#2C2C2A"),
      legend.background = element_rect(fill = "white", colour = NA),
      legend.key       = element_rect(fill = "white", colour = NA),
      plot.background  = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA)
    )
}

read_tsv_safe <- function(path) {
  if (!file.exists(path)) return(NULL)
  if (file.info(path)$size == 0) return(NULL)
  out <- tryCatch(
    suppressMessages(readr::read_tsv(path, show_col_types = FALSE,
                                      na = c("NA", ""))),
    error = function(e) {
      warning("could not read ", path, ": ", e$message)
      NULL
    }
  )
  if (is.null(out) || nrow(out) == 0) return(NULL)
  out
}

save_svg <- function(plot, out_dir, name, width = 6, height = 4) {
  path <- file.path(out_dir, paste0(name, ".svg"))
  ggsave(path, plot, device = "svg",
         width = width, height = height, units = "in")
  message("  wrote ", path)
  invisible(path)
}

order_models <- function(df) {
  if (!"Model" %in% names(df)) return(df)
  present <- intersect(names(MODEL_COLORS), unique(df$Model))
  df$Model <- factor(df$Model, levels = present)
  df
}

render_fig0 <- function(fig_dir, out_dir) {
  message("[Fig 0] training stats")
  base <- "fig0_training_stats"

  summ <- read_tsv_safe(file.path(fig_dir, paste0(base, "_summary.tsv")))
  if (!is.null(summ)) {
    summ$Model  <- factor(summ$Model,  levels = c("RF", "LR", "CNN"))
    summ$Metric <- factor(summ$Metric, levels = c("AUPRC", "AUROC", "F1"))
    y_lo <- max(0, floor(min(summ$value, na.rm = TRUE) * 10) / 10 - 0.05)
    p <- ggplot(summ, aes(Metric, value, fill = Model)) +
      geom_col(position = position_dodge(width = 0.8),
               width = 0.7, colour = "black", linewidth = 0.25) +
      geom_text(aes(label = sprintf("%.3f", value)),
                position = position_dodge(width = 0.8),
                vjust = -0.4, size = 2.8) +
      scale_fill_manual(values = MODEL_COLORS) +
      coord_cartesian(ylim = c(y_lo, 1.05)) +
      labs(x = NULL, y = "Score",
           title = "Window-level test-set metrics",
           subtitle = "RF / LR / CNN; y-axis zoomed") +
      pub_theme()
    save_svg(p, out_dir, paste0(base, "_summary"), 6.5, 4)
  }

  pr <- read_tsv_safe(file.path(fig_dir, paste0(base, "_pr.tsv")))
  if (!is.null(pr)) {
    pr$Model <- factor(pr$Model, levels = c("RF", "LR", "CNN"))
    p <- ggplot(pr, aes(recall, precision, colour = Model)) +
      geom_line(linewidth = 0.8) +
      scale_x_continuous(labels = percent_format(accuracy = 1),
                          limits = c(0, 1)) +
      scale_y_continuous(labels = percent_format(accuracy = 1),
                          limits = c(0, 1)) +
      scale_colour_manual(values = MODEL_COLORS) +
      labs(x = "Recall", y = "Precision",
           title = "Test-set precision-recall curves",
           subtitle = "Window-level CPA discrimination") +
      pub_theme()
    save_svg(p, out_dir, paste0(base, "_pr"), 5.5, 4.5)
  }

  roc <- read_tsv_safe(file.path(fig_dir, paste0(base, "_roc.tsv")))
  if (!is.null(roc)) {
    roc$Model <- factor(roc$Model, levels = c("RF", "LR", "CNN"))
    p <- ggplot(roc, aes(fpr, tpr, colour = Model)) +
      geom_abline(slope = 1, intercept = 0,
                  colour = "#888780", linetype = "dashed", linewidth = 0.3) +
      geom_line(linewidth = 0.8) +
      scale_x_continuous(labels = percent_format(accuracy = 1),
                          limits = c(0, 1)) +
      scale_y_continuous(labels = percent_format(accuracy = 1),
                          limits = c(0, 1)) +
      scale_colour_manual(values = MODEL_COLORS) +
      labs(x = "False positive rate", y = "True positive rate",
           title = "Test-set ROC curves",
           subtitle = "Window-level CPA discrimination") +
      pub_theme()
    save_svg(p, out_dir, paste0(base, "_roc"), 5.5, 4.5)
  }
}

render_fig1 <- function(fig_dir, out_dir, suffix = "") {
  message("[Fig 1", suffix, "] rank distributions + top-K")
  base <- paste0("fig1_ranks", suffix)

  cdf <- read_tsv_safe(file.path(fig_dir, paste0(base, "_cdf.tsv")))
  if (!is.null(cdf)) {
    cdf <- order_models(cdf)
    p <- ggplot(cdf, aes(rank, frac, colour = Model)) +
      geom_line(linewidth = 0.8) +
      scale_x_log10(breaks = c(1, 10, 100, 1000, 10000),
                     labels = label_comma()) +
      scale_y_continuous(labels = percent_format(accuracy = 1),
                          limits = c(0, 1)) +
      scale_colour_manual(values = MODEL_COLORS) +
      labs(x = "Rank of canonical window (log scale)",
           y = "Genes (cumulative %)",
           title = "Rank of the canonical CPA window",
           subtitle = "Strict per-window rank (no overlap allowance)") +
      pub_theme()
    save_svg(p, out_dir, paste0(base, "_cdf"), 6.5, 4)
  }

  topk <- read_tsv_safe(file.path(fig_dir, paste0(base, "_topk.tsv")))
  if (!is.null(topk)) {
    topk <- order_models(topk)
    topk$K <- factor(topk$K, levels = unique(topk$K[order(topk$K_num)]))
    p <- ggplot(topk, aes(K, pct, fill = Model)) +
      geom_col(position = position_dodge(width = 0.85),
               width = 0.78, colour = "black", linewidth = 0.25) +
      geom_text(aes(label = sprintf("%.0f%%", pct)),
                position = position_dodge(width = 0.85),
                vjust = -0.4, size = 2.5) +
      scale_fill_manual(values = MODEL_COLORS) +
      scale_y_continuous(limits = c(0, 115),
                          breaks = c(0, 25, 50, 75, 100),
                          labels = function(x) paste0(x, "%")) +
      labs(x = NULL, y = "Genes (%)",
           title = "Canonical CPA region in top-K predictions",
           subtitle = "Hit = any of the top-K windows overlaps the canonical 500-nt window") +
      pub_theme()
    save_svg(p, out_dir, paste0(base, "_topk"),
              max(6.5, 1.5 + 1.4 * length(unique(topk$Model))), 4)
  }
}

render_fig2 <- function(fig_dir, out_dir) {
  message("[Fig 2] atlas recovery")
  df <- read_tsv_safe(file.path(fig_dir, "fig2_atlas_recovery.tsv"))
  if (is.null(df)) return(invisible(NULL))
  df <- order_models(df)
  df$Atlas <- factor(df$Atlas, levels = unique(df$Atlas))
  p <- ggplot(df, aes(Atlas, recall, fill = Model)) +
    geom_col(position = position_dodge(width = 0.85),
             width = 0.78, colour = "black", linewidth = 0.25) +
    geom_text(aes(label = sprintf("%.0f%%", recall)),
              position = position_dodge(width = 0.85),
              vjust = -0.4, size = 2.5) +
    scale_fill_manual(values = MODEL_COLORS) +
    scale_y_continuous(limits = c(0, 115),
                        breaks = c(0, 25, 50, 75, 100),
                        labels = function(x) paste0(x, "%")) +
    labs(x = NULL, y = "Recall (% of reference sites with a model peak)",
         title = "Atlas recovery by each model's peak set",
         subtitle = "'Combined' = bedtools-merged union of all model peak BEDs") +
    pub_theme()
  save_svg(p, out_dir, "fig2_atlas_recovery",
            max(7.5, 1.5 + 1.5 * length(unique(df$Atlas))), 5)
}

render_fig3 <- function(fig_dir, out_dir) {
  message("[Fig 3] distance + atlas-by-distance")
  cdf <- read_tsv_safe(file.path(fig_dir, "fig3_distance_atlas_cdf.tsv"))
  if (!is.null(cdf)) {
    cdf <- order_models(cdf)
    p <- ggplot(cdf, aes(distance, frac, colour = Model)) +
      geom_vline(xintercept = c(100, 1000, 10000),
                 colour = "#D3D1C7", linetype = "dotted", linewidth = 0.3) +
      geom_line(linewidth = 0.8) +
      scale_x_log10(breaks = c(1, 10, 100, 1000, 10000, 100000),
                     labels = label_comma()) +
      scale_y_continuous(labels = percent_format(accuracy = 1),
                          limits = c(0, 1)) +
      scale_colour_manual(values = MODEL_COLORS) +
      labs(x = "Distance from canonical CPA (nt, log scale)",
           y = "Genes (cumulative %)",
           title = "How close is each model's top pick to the canonical CPA?",
           subtitle = "Distance of zero means top pick is the canonical window itself") +
      pub_theme()
    save_svg(p, out_dir, "fig3_distance_atlas_cdf", 6.5, 4)
  }

  strat <- read_tsv_safe(file.path(fig_dir, "fig3_distance_atlas_strat.tsv"))
  if (!is.null(strat)) {
    strat <- order_models(strat)
    strat$bin <- factor(strat$bin, levels = unique(strat$bin))
    p <- ggplot(strat, aes(bin, rate, fill = Model)) +
      geom_col(position = position_dodge(width = 0.85),
               width = 0.78, colour = "black", linewidth = 0.25) +
      geom_hline(yintercept = 16.6, linetype = "dashed",
                  colour = "#5F5E5A", linewidth = 0.4) +
      scale_fill_manual(values = MODEL_COLORS) +
      scale_y_continuous(limits = c(0, 100),
                          breaks = c(0, 25, 50, 75, 100),
                          labels = function(x) paste0(x, "%")) +
      labs(x = "Distance from canonical CPA (nt)",
           y = "Atlas overlap rate",
           title = "Are 'wrong' picks real atlas CPA sites?",
           subtitle = "Dashed line: random in-gene window baseline (~17%)") +
      pub_theme()
    save_svg(p, out_dir, "fig3_distance_atlas_strat", 7, 4)
  }
}

render_fig4 <- function(fig_dir, out_dir) {
  message("[Fig 4] score distributions by class")
  long <- read_tsv_safe(file.path(fig_dir, "fig4_score_distributions.tsv"))
  thr  <- read_tsv_safe(file.path(fig_dir, "fig4_score_distributions_thresholds.tsv"))
  if (is.null(long)) return(invisible(NULL))
  long <- order_models(long)
  long$Class <- factor(long$Class,
                        levels = c("Canonical", "Atlas", "Background"))
  class_pal <- c("Canonical"  = PAL$canon_red,
                 "Atlas"      = PAL$blue,
                 "Background" = PAL$gray)

  ns <- long %>%
    group_by(Model, Class) %>%
    summarise(n = n(), .groups = "drop")

  p <- ggplot(long, aes(Class, score, fill = Class)) +
    geom_violin(scale = "width", trim = FALSE,
                colour = "black", linewidth = 0.3, alpha = 0.85) +
    geom_boxplot(width = 0.08, outlier.shape = NA,
                  fill = "white", linewidth = 0.3) +
    facet_wrap(~ Model, nrow = 1) +
    scale_fill_manual(values = class_pal, guide = "none") +
    scale_y_continuous(limits = c(0, 1.05)) +
    labs(x = NULL, y = "Predicted probability",
         title = "Score distributions by window class",
         subtitle = "Per-model violins; Atlas = non-canonical window overlapping the atlas union") +
    pub_theme()

  if (!is.null(thr)) {
    thr <- order_models(thr)
    p <- p + geom_hline(data = thr,
                        aes(yintercept = threshold),
                        linetype = "dashed", colour = "black",
                        linewidth = 0.4)
  }

  save_svg(p, out_dir, "fig4_score_distributions",
            max(7, 2 + 2.5 * length(unique(long$Model))), 4.5)
}

render_fig5 <- function(fig_dir, out_dir) {
  message("[Fig 5] inter-model agreement")
  corr <- read_tsv_safe(file.path(fig_dir, "fig5_agreement_corr.tsv"))
  if (!is.null(corr)) {
    lvls <- intersect(names(MODEL_COLORS), unique(c(corr$x, corr$y)))
    corr$x <- factor(corr$x, levels = lvls)
    corr$y <- factor(corr$y, levels = rev(lvls))
    p <- ggplot(corr, aes(x, y, fill = rho)) +
      geom_tile(colour = "black", linewidth = 0.3) +
      geom_text(aes(label = sprintf("%.2f", rho)),
                colour = "#2C2C2A", size = 3.5) +
      scale_fill_gradient(low = "#F1EFE8", high = PAL$purple_dk,
                          limits = c(0, 1), name = "Pearson r") +
      coord_equal() +
      labs(x = NULL, y = NULL,
           title = "Per-window score agreement between models",
           subtitle = "Pearson correlation of predicted probabilities") +
      pub_theme()
    save_svg(p, out_dir, "fig5_agreement_corr",
              max(5, 1.5 + 1.2 * length(lvls)),
              max(5, 1.5 + 1.2 * length(lvls)))
  }

  jac <- read_tsv_safe(file.path(fig_dir, "fig5_agreement_jaccard.tsv"))
  if (!is.null(jac)) {
    jac$pair <- factor(jac$pair, levels = jac$pair)
    p <- ggplot(jac, aes(pair, median)) +
      geom_col(width = 0.55, fill = PAL$blue,
               colour = "black", linewidth = 0.25) +
      geom_text(aes(label = sprintf("%.2f", median)),
                vjust = -0.4, size = 3) +
      scale_y_continuous(limits = c(0, 1.05),
                          breaks = c(0, .25, .5, .75, 1)) +
      labs(x = NULL, y = "Median Jaccard index (per gene)",
           title = "Peak-set agreement between model pairs",
           subtitle = "Peaks within 250 nt count as the same site") +
      pub_theme() +
      theme(axis.text.x = element_text(angle = 25, hjust = 1))
    save_svg(p, out_dir, "fig5_agreement_jaccard",
              max(6, 1.5 + 1.0 * nrow(jac)), 4)
  }

  inter <- read_tsv_safe(file.path(fig_dir,
                          "fig5_agreement_upset_intersections.tsv"))
  setsz <- read_tsv_safe(file.path(fig_dir,
                          "fig5_agreement_upset_setsizes.tsv"))
  mat   <- read_tsv_safe(file.path(fig_dir,
                          "fig5_agreement_upset_matrix.tsv"))
  if (!is.null(inter) && !is.null(setsz) && !is.null(mat)) {
    cats <- intersect(names(MODEL_COLORS), unique(setsz$Model))
    inter$intersection_id <- factor(inter$intersection_id,
                                     levels = inter$intersection_id)
    mat$intersection_id   <- factor(mat$intersection_id,
                                     levels = levels(inter$intersection_id))
    mat$Model             <- factor(mat$Model, levels = rev(cats))
    setsz$Model           <- factor(setsz$Model, levels = rev(cats))

    bar_top <- ggplot(inter, aes(intersection_id, count)) +
      geom_col(fill = PAL$teal, colour = "black", linewidth = 0.25,
                width = 0.7) +
      geom_text(aes(label = paste0(format(count, big.mark = ","),
                                   "\n(", sprintf("%.0f%%",
                                                   frac_of_total * 100), ")")),
                vjust = -0.2, size = 2.3, lineheight = 0.85) +
      scale_y_continuous(expand = expansion(mult = c(0, 0.25)),
                          labels = label_comma()) +
      labs(x = NULL, y = "Intersection size") +
      pub_theme() +
      theme(axis.text.x = element_blank(),
            axis.ticks.x = element_blank(),
            panel.grid.major.x = element_blank())

    mat_plot <- ggplot(mat) +
      geom_point(aes(intersection_id, Model,
                     colour = factor(in_intersection)),
                  size = 4) +
      geom_line(data = mat %>% filter(in_intersection == 1) %>%
                          group_by(intersection_id) %>%
                          filter(n() > 1),
                aes(intersection_id, Model, group = intersection_id),
                colour = PAL$teal, linewidth = 1) +
      scale_colour_manual(values = c("0" = "#E8E6DD", "1" = PAL$teal),
                           guide = "none") +
      labs(x = NULL, y = NULL) +
      pub_theme() +
      theme(axis.text.x = element_blank(),
            axis.ticks.x = element_blank(),
            panel.grid = element_blank())

    side_set <- ggplot(setsz, aes(set_size, Model)) +
      geom_col(fill = PAL$gray, colour = "black", linewidth = 0.25,
                width = 0.6) +
      geom_text(aes(label = format(set_size, big.mark = ",")),
                hjust = -0.1, size = 2.5) +
      scale_x_reverse(expand = expansion(mult = c(0.25, 0)),
                       labels = label_comma()) +
      labs(x = "Set size", y = NULL) +
      pub_theme() +
      theme(panel.grid.major.y = element_blank())

    blank <- ggplot() + theme_void()

    composite <- (blank | bar_top) / (side_set | mat_plot) +
      plot_layout(widths = c(1, 3), heights = c(2, 1)) +
      plot_annotation(
        title = "High-confidence peak overlap across models",
        subtitle = paste0("Total distinct peak clusters: ",
                           format(sum(inter$count), big.mark = ","),
                           ". Peaks within 250 nt = same site."),
        theme = pub_theme()
      )
    save_svg(composite, out_dir, "fig5_agreement_upset",
              max(7, 1 + 0.7 * nrow(inter)), 5.5)
  }
}

render_fig6 <- function(fig_dir, out_dir) {
  trace_files <- list.files(fig_dir, pattern = "^fig6_trace_[^_]+\\.tsv$",
                             full.names = TRUE)
  if (!length(trace_files)) return(invisible(NULL))
  for (f in trace_files) {
    gene <- sub("^fig6_trace_(.+)\\.tsv$", "\\1", basename(f))
    message("[Fig 6] trace for ", gene)
    trace <- read_tsv_safe(f)
    mark  <- read_tsv_safe(sub("\\.tsv$", "_markers.tsv", f))
    meta  <- read_tsv_safe(sub("\\.tsv$", "_meta.tsv", f))
    if (is.null(trace)) next
    trace <- order_models(trace)
    thr_lines <- trace %>% group_by(Model) %>%
      summarise(thr = first(threshold), .groups = "drop")

    canon_kb <- if (!is.null(mark)) {
      mark$position_kb[mark$kind == "canonical"]
    } else NULL
    atlas_kb <- if (!is.null(mark)) {
      mark$position_kb[mark$kind == "atlas"]
    } else NULL

    p <- ggplot(trace, aes(position_kb, prob)) +
      geom_area(aes(fill = Model), alpha = 0.25) +
      geom_line(aes(colour = Model), linewidth = 0.4) +
      geom_hline(data = thr_lines,
                 aes(yintercept = thr, colour = Model),
                 linetype = "dashed", linewidth = 0.4) +
      facet_wrap(~ Model, ncol = 1, strip.position = "right") +
      scale_fill_manual(values = MODEL_COLORS, guide = "none") +
      scale_colour_manual(values = MODEL_COLORS, guide = "none") +
      scale_y_continuous(limits = c(0, 1.05),
                          breaks = c(0, 0.5, 1.0)) +
      labs(x = "Position along gene, 5' \u2192 3' (kb)",
           y = "Predicted probability",
           title = paste0("Score traces across ", gene)) +
      pub_theme()

    if (!is.null(canon_kb) && length(canon_kb)) {
      p <- p + geom_vline(xintercept = canon_kb,
                          colour = PAL$canon_red,
                          linetype = "dashed", linewidth = 0.5)
    }
    if (!is.null(atlas_kb) && length(atlas_kb)) {
      tick_df <- data.frame(position_kb = atlas_kb)
      p <- p + geom_point(data = tick_df,
                          aes(x = position_kb, y = 1.03),
                          shape = 25, fill = PAL$gray,
                          colour = PAL$gray, size = 1.5,
                          inherit.aes = FALSE)
    }
    if (!is.null(meta) && nrow(meta) == 1) {
      sub <- sprintf("%s:%s-%s (strand %s, %.1f kb)",
                     meta$chrom[1],
                     format(meta$start[1], big.mark = ","),
                     format(meta$end[1],   big.mark = ","),
                     meta$strand[1], meta$length_kb[1])
      p <- p + labs(subtitle = sub)
    }
    n_panels <- length(unique(trace$Model))
    save_svg(p, out_dir, paste0("fig6_trace_", gene),
              8.5, max(3, 1.5 * n_panels + 1))
  }
}

render_fig7 <- function(fig_dir, out_dir) {
  message("[Fig 7] genome coverage")
  df <- read_tsv_safe(file.path(fig_dir, "fig7_genome_coverage.tsv"))
  denom <- read_tsv_safe(file.path(fig_dir,
                          "fig7_genome_coverage_denominator.tsv"))
  if (is.null(df)) return(invisible(NULL))
  df <- order_models(df)
  df$Metric <- factor(df$Metric,
                       levels = c("Above-threshold", "Merged peaks"))
  y_hi <- max(df$pct, na.rm = TRUE) * 1.25
  sub <- "Left: any above-threshold 500-nt window; Right: bedtools-merged representative peaks"
  if (!is.null(denom) && nrow(denom) >= 1) {
    sub <- paste0("Total gene-body footprint: ",
                   sprintf("%.1f Mb", denom$total_genebody_bp[1] / 1e6),
                   ". ", sub)
  }
  p <- ggplot(df, aes(Model, pct, fill = Model)) +
    geom_col(width = 0.6, colour = "black", linewidth = 0.25,
              show.legend = FALSE) +
    geom_text(aes(label = sprintf("%.2f%%", pct)),
              vjust = -0.4, size = 2.8) +
    facet_wrap(~ Metric, nrow = 1) +
    scale_fill_manual(values = MODEL_COLORS) +
    coord_cartesian(ylim = c(0, max(y_hi, 1))) +
    labs(x = NULL, y = "% of protein-coding gene-body bases",
         title = "How much of the protein-coding genome is called positive?",
         subtitle = sub) +
    pub_theme()
  save_svg(p, out_dir, "fig7_genome_coverage",
            max(7, 2 + 1.5 * length(unique(df$Model))), 4.5)
}

render_fig8 <- function(fig_dir, out_dir) {
  message("[Fig 8] 3-way Venn")
  df <- read_tsv_safe(file.path(fig_dir, "fig8_venn_rflrcnn.tsv"))
  if (is.null(df)) return(invisible(NULL))
  total <- sum(df$count)

  circle_df <- data.frame(
    set = c("RF", "LR", "CNN"),
    x0  = c(0.50, 0.34, 0.66),
    y0  = c(0.62, 0.36, 0.36),
    r   = 0.25,
    colour = c(MODEL_COLORS["RF"],
                MODEL_COLORS["LR"],
                MODEL_COLORS["CNN"])
  )
  label_df <- data.frame(
    set = c("RF", "LR", "CNN"),
    x = c(0.50, 0.08, 0.92),
    y = c(0.94, 0.22, 0.22),
    colour = circle_df$colour
  )
  region_pos <- tibble::tribble(
    ~region,       ~x,    ~y,
    "rf",          0.50,  0.78,
    "lr",          0.18,  0.28,
    "cnn",         0.82,  0.28,
    "lr+rf",       0.30,  0.50,
    "cnn+rf",      0.70,  0.50,
    "cnn+lr",      0.50,  0.26,
    "cnn+lr+rf",   0.50,  0.45
  )
  df2 <- df %>%
    inner_join(region_pos, by = "region") %>%
    mutate(label = paste0(format(count, big.mark = ","),
                          "\n(", sprintf("%.1f%%",
                                          frac_of_total * 100), ")"),
            bold  = (region == "cnn+lr+rf"))

  p <- ggplot() +
    ggforce::geom_circle(data = circle_df,
      aes(x0 = x0, y0 = y0, r = r, fill = set),
      alpha = 0.4, colour = "black", size = 0.4) +
    scale_fill_manual(values = c("RF" = MODEL_COLORS["RF"][[1]],
                                  "LR" = MODEL_COLORS["LR"][[1]],
                                  "CNN" = MODEL_COLORS["CNN"][[1]]),
                       guide = "none") +
    geom_text(data = label_df,
              aes(x = x, y = y, label = set, colour = set),
              fontface = "bold", size = 6) +
    scale_colour_manual(values = c("RF" = MODEL_COLORS["RF"][[1]],
                                    "LR" = MODEL_COLORS["LR"][[1]],
                                    "CNN" = MODEL_COLORS["CNN"][[1]]),
                         guide = "none") +
    geom_text(data = df2 %>% filter(!bold),
              aes(x, y, label = label), size = 3.5) +
    geom_text(data = df2 %>% filter(bold),
              aes(x, y, label = label), size = 4, fontface = "bold") +
    coord_fixed(xlim = c(0, 1), ylim = c(0.05, 1)) +
    labs(title = paste0("RF / LR / CNN agreement on intragenic CPA peaks ",
                        "(total: ", format(total, big.mark = ","), ")")) +
    pub_theme() +
    theme(axis.text = element_blank(),
          axis.ticks = element_blank(),
          axis.line = element_blank(),
          axis.title = element_blank(),
          panel.grid = element_blank(),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border = element_blank())
  save_svg(p, out_dir, "fig8_venn_rflrcnn", 7, 7)
}

render_fig9 <- function(fig_dir, out_dir, suffix = "") {
  base <- paste0("fig9_saturation", suffix)
  message("[Fig 9", suffix, "] saturation diagnostic")

  panelA <- read_tsv_safe(file.path(fig_dir, paste0(base, "_panelA.tsv")))
  panelB <- read_tsv_safe(file.path(fig_dir, paste0(base, "_panelB.tsv")))
  panelC <- read_tsv_safe(file.path(fig_dir, paste0(base, "_panelC.tsv")))

  pa <- NULL; pb <- NULL; pc <- NULL
  if (!is.null(panelA)) {
    panelA <- order_models(panelA)
    pa <- ggplot(panelA, aes(n_high, fill = Model, colour = Model)) +
      geom_histogram(position = "identity", bins = 40,
                      alpha = 0.45, linewidth = 0.2) +
      scale_fill_manual(values = MODEL_COLORS) +
      scale_colour_manual(values = MODEL_COLORS, guide = "none") +
      scale_y_continuous(trans = scales::pseudo_log_trans(base = 10),
                          breaks = c(0, 10, 100, 1000, 10000),
                          labels = label_comma()) +
      labs(x = "# windows per gene above threshold",
           y = "Genes (pseudo-log)",
           title = "A. Confident windows per gene") +
      pub_theme()
  }
  if (!is.null(panelB) && nrow(panelB) > 0) {
    panelB <- order_models(panelB)
    pb <- ggplot(panelB, aes(gap, fill = Model, colour = Model)) +
      geom_histogram(position = "identity", bins = 40,
                      alpha = 0.45, linewidth = 0.2) +
      geom_vline(xintercept = 0.01, linetype = "dashed",
                  colour = "#5F5E5A", linewidth = 0.4) +
      scale_fill_manual(values = MODEL_COLORS) +
      scale_colour_manual(values = MODEL_COLORS, guide = "none") +
      scale_x_continuous(limits = c(0, 1)) +
      labs(x = "(top-1 prob) - (canonical prob), top-1 misses",
           y = "Genes",
           title = "B. Are 'misses' ties?") +
      pub_theme()
  }
  if (!is.null(panelC)) {
    panelC <- order_models(panelC)
    cdf_rows <- panelC %>%
      group_by(Model) %>%
      arrange(rank) %>%
      mutate(cdf = seq_along(rank) / n()) %>%
      ungroup()
    pc <- ggplot(cdf_rows, aes(rank, cdf, colour = Model)) +
      geom_step(linewidth = 0.7) +
      scale_x_log10(labels = label_comma()) +
      scale_y_continuous(labels = percent_format(accuracy = 1),
                          limits = c(0, 1)) +
      scale_colour_manual(values = MODEL_COLORS) +
      labs(x = "Rank of canonical window (log)",
           y = "Cumulative fraction of genes",
           title = "C. Where does the canonical rank?") +
      pub_theme()
  }

  panels <- Filter(Negate(is.null), list(pa, pb, pc))
  if (!length(panels)) return(invisible(NULL))
  composite <- Reduce(`|`, panels) +
    plot_annotation(
      title = paste0("Score-saturation diagnostic",
                     if (nzchar(suffix)) " (test chromosomes only)" else ""),
      subtitle = "Many windows at high probability => argmax becomes a coin flip",
      theme = pub_theme()
    )
  save_svg(composite, out_dir, base, 5 * length(panels), 4.5)
}

render_fig10 <- function(fig_dir, out_dir) {
  message("[Fig 10] calibration")
  cal <- read_tsv_safe(file.path(fig_dir, "fig10_calibration.tsv"))
  if (!is.null(cal)) {
    cal <- order_models(cal)
    p <- ggplot(cal, aes(mean_pred, empirical, colour = Model)) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                  colour = "#888780", linewidth = 0.4) +
      geom_line(linewidth = 0.6) +
      geom_point(aes(size = n), alpha = 0.85) +
      scale_x_continuous(limits = c(0, 1),
                          labels = percent_format(accuracy = 1)) +
      scale_y_continuous(limits = c(0, 1),
                          labels = percent_format(accuracy = 1)) +
      scale_colour_manual(values = MODEL_COLORS) +
      scale_size_continuous(name = "Bin count", range = c(1, 5)) +
      labs(x = "Mean predicted probability (in bin)",
           y = "Empirical positive rate (in bin)",
           title = "Are model probabilities calibrated?",
           subtitle = "Predicted vs observed positive rate per bin; diagonal = perfect") +
      pub_theme()
    save_svg(p, out_dir, "fig10_calibration", 6, 6)
  }

  hist <- read_tsv_safe(file.path(fig_dir, "fig10_calibration_dist.tsv"))
  if (!is.null(hist)) {
    hist <- order_models(hist)
    p <- ggplot(hist, aes(bin_mid, count, fill = Model)) +
      geom_col(width = 0.08, colour = "black", linewidth = 0.2,
                show.legend = FALSE) +
      facet_wrap(~ Model, nrow = 1, scales = "free_y") +
      scale_fill_manual(values = MODEL_COLORS) +
      scale_x_continuous(limits = c(0, 1),
                          labels = percent_format(accuracy = 1)) +
      labs(x = "Predicted probability", y = "Sampled windows",
           title = "Distribution of predicted probabilities per model",
           subtitle = "Flat = uncertain everywhere; bimodal = decisive") +
      pub_theme()
    save_svg(p, out_dir, "fig10_calibration_dist",
              max(7, 2 + 1.5 * length(unique(hist$Model))), 4)
  }
}

render_fig11 <- function(fig_dir, out_dir) {
  message("[Fig 11] motif density")
  df <- read_tsv_safe(file.path(fig_dir, "fig11_motif_density.tsv"))
  if (is.null(df)) return(invisible(NULL))

  pal_set <- c("Canonical" = PAL$canon_red)
  others  <- setdiff(unique(df$set), "Canonical")
  pal_set <- c(pal_set, setNames(rep("#7E2C12", length(others)), others))

  df$panel <- factor(df$panel, levels = unique(df$panel))
  p <- ggplot(df, aes(position, value, colour = set)) +
    geom_vline(xintercept = 0, linetype = "dashed",
                colour = "#5F5E5A", linewidth = 0.4) +
    geom_line(linewidth = 0.7) +
    facet_wrap(~ panel, nrow = 2, scales = "free_y") +
    scale_colour_manual(values = pal_set, name = NULL) +
    labs(x = "Position relative to cut site (nt)",
         y = "Per-sequence frequency",
         title = "Canonical CPA motifs at canonical vs CPCCPA cut sites",
         subtitle = "Per-position occurrence rate around each cut site (vertical line)") +
    pub_theme() +
    theme(legend.position = "bottom")
  save_svg(p, out_dir, "fig11_motif_density", 9, 6.5)
}

render_fig13_heatmaps <- function(fig_dir, out_dir) {
  message("[Fig 13] score heatmaps")
  for (kind in c("canonical", "cpccpa")) {
    base <- paste0("fig13_heatmap_", kind)
    df    <- read_tsv_safe(file.path(fig_dir, paste0(base, ".tsv")))
    meta  <- read_tsv_safe(file.path(fig_dir, paste0(base, "_metaprofile.tsv")))
    if (is.null(df)) next

    df <- order_models(df)
    if (!is.null(meta)) meta <- order_models(meta)

    p_heat <- ggplot(df, aes(position, site_idx, fill = prob)) +
      geom_raster() +
      facet_wrap(~ Model, nrow = 1) +
      scale_fill_viridis(option = "viridis", limits = c(0, 1),
                          name = "Probability") +
      scale_y_reverse(expand = c(0, 0)) +
      scale_x_continuous(expand = c(0, 0)) +
      labs(x = "nt from cut site",
            y = paste0(stringr::str_to_title(kind),
                       " sites (sorted by mean center prob)")) +
      pub_theme() +
      theme(panel.grid = element_blank())

    if (!is.null(meta)) {
      p_meta <- ggplot(meta, aes(position, mean_prob, colour = Model)) +
        geom_area(aes(fill = Model), alpha = 0.3,
                   position = "identity", show.legend = FALSE) +
        geom_line(linewidth = 0.7) +
        geom_hline(aes(yintercept = threshold), linetype = "dashed",
                    colour = "black", linewidth = 0.3) +
        facet_wrap(~ Model, nrow = 1) +
        scale_colour_manual(values = MODEL_COLORS, guide = "none") +
        scale_fill_manual(values = MODEL_COLORS) +
        scale_y_continuous(limits = c(0, 1.05)) +
        labs(x = NULL, y = "Mean\nprob") +
        pub_theme() +
        theme(axis.text.x = element_blank(),
              axis.ticks.x = element_blank(),
              strip.text = element_text(face = "bold"))

      composite <- p_meta / p_heat +
        plot_layout(heights = c(1, 4)) +
        plot_annotation(
          title = paste0("Per-site CPA score profile: ", kind, " sites"),
          theme = pub_theme()
        )
      save_svg(composite, out_dir, base,
                3.5 * length(unique(df$Model)) + 1.5, 8)
    } else {
      save_svg(p_heat, out_dir, base,
                3.5 * length(unique(df$Model)) + 1.5, 7)
    }
  }
}

render_fig13_robustness <- function(fig_dir, out_dir) {
  message("[Fig 13] threshold robustness")
  df <- read_tsv_safe(file.path(fig_dir, "fig13_threshold_robustness.tsv"))
  if (is.null(df)) return(invisible(NULL))
  df <- order_models(df)
  df$percentile <- factor(paste0("p", df$percentile),
                           levels = paste0("p", sort(unique(df$percentile))))

  p_a <- ggplot(df, aes(percentile, Model, fill = atlas_overlap_pct)) +
    geom_tile(colour = "white", linewidth = 0.3) +
    geom_text(aes(label = sprintf("%.1f", atlas_overlap_pct)),
              size = 3) +
    scale_fill_distiller(palette = "RdYlBu", direction = -1,
                          name = "% overlap atlas",
                          limits = c(0, max(df$atlas_overlap_pct,
                                            30, na.rm = TRUE))) +
    labs(x = NULL, y = NULL,
         title = paste0("A. Atlas overlap rate (%) of intragenic above-threshold windows ",
                         "  -  ref: ", df$atlas_label[1])) +
    pub_theme() +
    theme(panel.grid = element_blank())

  p_b <- ggplot(df, aes(percentile, Model,
                         fill = log10(pmax(n_intragenic, 1)))) +
    geom_tile(colour = "white", linewidth = 0.3) +
    geom_text(aes(label = ifelse(n_intragenic < 10000,
                                   format(n_intragenic, big.mark = ","),
                                   paste0(round(n_intragenic / 1000), "k"))),
              size = 3) +
    scale_fill_distiller(palette = "Blues", direction = 1,
                          name = "log10(count)") +
    labs(x = "Threshold (percentile of canonical-CPA probabilities)",
         y = NULL,
         title = "B. log10(# intragenic above-threshold windows) per (model, percentile)") +
    pub_theme() +
    theme(panel.grid = element_blank())

  composite <- p_a / p_b +
    plot_annotation(
      title = "Threshold robustness  -  CPCCPA atlas overlap stays low at any threshold",
      subtitle = paste0("Each row sweeps threshold across percentiles of canonical-CPA score distribution; ",
                         "if CPCCPA were a thresholding artifact, atlas overlap in A would rise toward 100%."),
      theme = pub_theme()
    )
  save_svg(composite, out_dir, "fig13_threshold_robustness",
            max(8, 1.2 * length(unique(df$percentile)) + 3), 7.5)
}

render_fig_cpccpa <- function(fig_dir, out_dir) {
  for (variant in c("with_3utr", "no_3utr")) {
    base <- paste0("fig_cpccpa_", variant)
    overlap <- read_tsv_safe(file.path(fig_dir, paste0(base, "_overlap.tsv")))
    rpm     <- read_tsv_safe(file.path(fig_dir, paste0(base, "_rpm.tsv")))
    pse     <- read_tsv_safe(file.path(fig_dir, paste0(base, "_pse.tsv")))
    detect  <- read_tsv_safe(file.path(fig_dir, paste0(base, "_detect.tsv")))
    if (is.null(overlap)) next

    message("[fig_cpccpa] ", variant)
    classes <- intersect(CPCCPA_CLASS_ORDER, unique(overlap$Class))
    overlap$Class <- factor(overlap$Class, levels = classes)
    overlap$Reference <- factor(overlap$Reference,
                                 levels = c("PolyASite",
                                            "PolyA_DB Main",
                                            "PolyA_DB Max"))
    n_by_class <- overlap %>%
      group_by(Class) %>%
      summarise(n_total = max(n_total), .groups = "drop")
    label_with_n <- setNames(
      paste0(CPCCPA_CLASS_LABEL[as.character(n_by_class$Class)],
              "\n(n=", format(n_by_class$n_total, big.mark = ","), ")"),
      as.character(n_by_class$Class)
    )

    p_a <- ggplot(overlap, aes(Class, rate, fill = Reference)) +
      geom_col(position = position_dodge(width = 0.85),
                width = 0.78, colour = "black", linewidth = 0.25) +
      geom_text(aes(label = sprintf("%.0f", rate)),
                position = position_dodge(width = 0.85),
                vjust = -0.3, size = 2.4) +
      scale_fill_manual(values = REF_COLORS, name = "Reference") +
      scale_x_discrete(labels = label_with_n) +
      scale_y_continuous(limits = c(0, 115),
                          breaks = c(0, 25, 50, 75, 100),
                          labels = function(x) paste0(x, "%")) +
      labs(x = NULL, y = "Sites overlapping reference (%)",
           title = "A. Atlas overlap by model-agreement tier") +
      pub_theme()

    add_violin_panel <- function(df, value_col, ylab, title_text,
                                  log_transform = FALSE) {
      if (is.null(df) || nrow(df) == 0) return(NULL)
      df$Class <- factor(df$Class, levels = classes)
      detect_labels <- if (!is.null(detect)) {
        detect %>%
          mutate(label = sprintf("%.0f%% have expr", pct_with_expr))
      } else NULL

      p <- ggplot(df, aes(Class, .data[[value_col]], fill = Class)) +
        geom_violin(scale = "width", trim = FALSE,
                    colour = "black", linewidth = 0.3, alpha = 0.85) +
        geom_boxplot(width = 0.08, outlier.shape = NA,
                      fill = "white", linewidth = 0.3) +
        scale_fill_manual(values = CPCCPA_CLASS_COLORS, guide = "none") +
        scale_x_discrete(labels = label_with_n) +
        labs(x = NULL, y = ylab, title = title_text) +
        pub_theme()
      if (log_transform) p <- p + scale_y_log10(labels = label_comma())
      if (!is.null(detect_labels)) {
        y_max <- if (log_transform) {
          max(df[[value_col]], na.rm = TRUE) * 1.5
        } else {
          max(df[[value_col]], na.rm = TRUE) * 1.05
        }
        detect_labels$Class <- factor(detect_labels$Class, levels = classes)
        p <- p + geom_text(data = detect_labels,
                            aes(x = Class, y = y_max, label = label),
                            inherit.aes = FALSE,
                            size = 2.5, colour = "#5F5E5A")
      }
      p
    }

    p_b <- add_violin_panel(
      rpm, "rpm",
      "RPM where overlaps v4 Max (log)",
      "B. Expression among detected sites (PolyA_DB v4 Max RPM)",
      log_transform = TRUE)
    p_c <- add_violin_panel(
      pse, "pse",
      "PSE (% of samples expressing) where overlaps v4 Max",
      "C. Detection breadth among detected sites (PolyA_DB v4 Max PSE)")

    panels <- Filter(Negate(is.null), list(p_a, p_b, p_c))
    composite <- Reduce(function(a, b) a / b, panels) +
      plot_annotation(
        title = paste0("CPCCPA experimental usage (",
                        if (variant == "with_3utr")
                          "with 3'UTR"
                        else "intragenic only, 3'UTR excluded)"),
        theme = pub_theme()
      )
    save_svg(composite, out_dir, base, 10, 4 * length(panels))
  }
}

render_fig12 <- function(fig_dir, out_dir) {
  sal_files <- list.files(fig_dir,
                           pattern = "^fig12_saliency_.+\\.tsv$",
                           full.names = TRUE)
  if (!length(sal_files)) return(invisible(NULL))
  for (f in sal_files) {
    label <- sub("^fig12_saliency_(.+)\\.tsv$", "\\1", basename(f))
    message("[Fig 12] saliency ", label)
    df <- read_tsv_safe(f)
    if (is.null(df)) next
    df <- order_models(df)
    p <- ggplot(df, aes(position, attribution_norm, fill = Model)) +
      geom_col(width = 1.0, colour = NA, show.legend = FALSE) +
      geom_vline(xintercept = 0, linetype = "dashed",
                  colour = PAL$canon_red, linewidth = 0.4) +
      facet_grid(Model ~ gene, scales = "free") +
      scale_fill_manual(values = MODEL_COLORS) +
      scale_y_continuous(limits = c(0, 1.05)) +
      labs(x = paste0("nt from ", label, " CPA"),
            y = "Normalized attribution",
            title = paste0("Sequence attribution at ", label, " CPAs"),
            subtitle = "Per-position attribution magnitude; bars normalized per track") +
      pub_theme() +
      theme(strip.text.y = element_text(angle = 0))
    n_g <- length(unique(df$gene))
    n_m <- length(unique(df$Model))
    save_svg(p, out_dir, sub("\\.tsv$", "", basename(f)),
              max(6, 2 + 2.5 * n_g), max(4, 1.4 * n_m + 1.5))
  }
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 1) {
    stop("Usage: Rscript make_figures.R <fig_dir> [<out_dir>]")
  }
  fig_dir <- args[1]
  out_dir <- if (length(args) >= 2) args[2] else file.path(fig_dir, "svg")
  if (!dir.exists(fig_dir)) stop("fig_dir does not exist: ", fig_dir)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  message("reading TSVs from: ", fig_dir)
  message("writing SVGs   to: ", out_dir)
  message("")

  render_fig0(fig_dir, out_dir)
  render_fig1(fig_dir, out_dir, suffix = "")
  render_fig1(fig_dir, out_dir, suffix = "_testonly")
  render_fig2(fig_dir, out_dir)
  render_fig3(fig_dir, out_dir)
  render_fig4(fig_dir, out_dir)
  render_fig5(fig_dir, out_dir)
  render_fig6(fig_dir, out_dir)
  render_fig7(fig_dir, out_dir)
  render_fig8(fig_dir, out_dir)
  render_fig9(fig_dir, out_dir, suffix = "")
  render_fig9(fig_dir, out_dir, suffix = "_testonly")
  render_fig10(fig_dir, out_dir)
  render_fig11(fig_dir, out_dir)
  render_fig12(fig_dir, out_dir)
  render_fig13_heatmaps(fig_dir, out_dir)
  render_fig13_robustness(fig_dir, out_dir)
  render_fig_cpccpa(fig_dir, out_dir)

  message("\nAll SVGs written to ", out_dir)
}

main()