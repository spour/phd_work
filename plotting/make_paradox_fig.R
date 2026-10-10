#!/usr/bin/env Rscript
# chapter 2 "paradox" figure

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
  library(readr);   library(scales);   library(hexbin)
})

args  <- commandArgs(trailingOnly = TRUE)
INDIR <- if (length(args) >= 1) args[1] else "."
p_in  <- function(f) file.path(INDIR, f)

# theme + palette
# copied from pub/scripts/make_figure2.r (lines 40-69), unchanged
INK   <- "#1A1A1A"; BASE <- 16
CANON <- "#0072B2"; CC   <- "#E69F00"
CCLAB <- "cCPA site"

th <- theme_classic(base_size = BASE) +
  theme(text            = element_text(color = INK),
        axis.text       = element_text(color = INK, size = BASE - 3),
        axis.title      = element_text(size = BASE - 2),
        axis.line       = element_line(color = INK, linewidth = 0.5),
        plot.title      = element_text(size = BASE + 2, face = "bold", hjust = 0,
                                       margin = margin(b = 1)),
        plot.title.position = "plot",
        plot.subtitle   = element_blank(),
        legend.position = "bottom",
        legend.title    = element_blank(),
        legend.text     = element_text(size = BASE - 4),
        legend.margin   = margin(t = -4, b = 0),
        legend.box.margin = margin(t = -2),
        legend.key.size = unit(0.8, "lines"),
        strip.background = element_blank(),
        strip.text      = element_text(face = "bold", size = BASE - 4,
                                       margin = margin(t = 0, b = 2)),
        plot.margin     = margin(2, 8, 3, 6))

# rank auroc (mann-whitney with midranks) -- same estimator as
# pub/scripts/figure2_stats.py:49-60, reimplemented so the figure is
auroc <- function(y, s) {
  keep <- is.finite(s); y <- y[keep]; s <- s[keep]
  r <- rank(s, ties.method = "average")
  np <- sum(y == 1); nn <- sum(y == 0)
  (sum(r[y == 1]) - np * (np + 1) / 2) / (np * nn)
}

A <- read_tsv(p_in("panelA.tsv"), show_col_types = FALSE)

nA    <- nrow(A)
auA   <- auroc(c(rep(1, nA), rep(0, nA)),
               c(A$canonical_aparent2, A$bestcomp_aparent2))
winA  <- mean(A$canonical_aparent2 > A$bestcomp_aparent2)

Cc    <- read_tsv(p_in("panelC.tsv"), show_col_types = FALSE)
pooled <- Cc |> filter(model == "APARENT2", task == "canonical vs competitor")

labA <- sprintf(
  "AUROC %.3f  (this panel: best competitor per gene)\npooled over all competitors %.3f [%.3f, %.3f]\ncanonical higher in %s of genes",
  auA, pooled$auroc[1], pooled$lo[1], pooled$hi[1], percent(winA, accuracy = 0.1))

pA <- ggplot(A, aes(bestcomp_aparent2, canonical_aparent2)) +
  stat_summary_hex(aes(z = dist_nt), fun = median, bins = 34) +
  geom_abline(slope = 1, intercept = 0, linetype = 2,
              color = INK, linewidth = 0.5) +
  scale_x_log10(labels = label_log()) +
  scale_y_log10(labels = label_log()) +
  scale_fill_viridis_c(
    transform = "log10", option = "D", direction = -1,
    breaks = c(1e3, 1e4, 1e5, 1e6),
    labels = c("1 kb", "10 kb", "100 kb", "1 Mb"),
    name  = "median distance between the two sites",
    guide = guide_colorbar(barwidth = unit(9, "lines"), barheight = unit(0.5, "lines"),
                           title.position = "top", title.hjust = 0.5)) +
  annotate("text", x = max(A$bestcomp_aparent2), y = min(A$canonical_aparent2),
           hjust = 1, vjust = 0,
           size = 3.0, color = INK, lineheight = 1.08, label = labA) +
  coord_fixed() +
  labs(title = "A",
       x = paste0("APARENT2 at the best ", CCLAB),
       y = "APARENT2 at the real 3' end") +
  th +
  theme(legend.title = element_text(size = BASE - 4, color = INK))

B <- read_tsv(p_in("panelB.tsv"), show_col_types = FALSE) |>
  mutate(motif = factor(motif, levels = c("PAS", "UGUA", "U-rich", "GU-rich")),
         class = factor(class, levels = c("canonical", "competitor")))

B <- B |> group_by(motif, class) |> arrange(position) |>
  mutate(density = as.numeric(stats::filter(density, rep(1/9, 9), sides = 2))) |>
  ungroup() |> filter(!is.na(density))

pB <- ggplot(B, aes(position, density, color = class)) +
  geom_vline(xintercept = 0, linetype = 3, color = "#B0B4BA", linewidth = 0.4) +
  geom_line(linewidth = 0.8) +
  facet_wrap(~motif, nrow = 2, scales = "free_y") +
  scale_color_manual(values = c(canonical = CANON, competitor = CC),
                     labels = c(canonical = "real 3' end", competitor = CCLAB)) +
  labs(title = "B", x = "Position relative to site (nt)",
       y = "Motif density per peak") +
  th

Cd <- Cc |>
  mutate(lab = ifelse(model == "APARENT2",
                      paste0("APARENT2\n", sub("canonical vs ", "vs ", task)),
                      paste0(model, "\nvs negative"))) |>
  arrange(match(model, c("RF", "LR", "CNN", "APARENT2")), desc(task)) |>
  mutate(lab = factor(lab, levels = lab))

pC <- ggplot(Cd, aes(lab, auroc, color = task)) +
  geom_hline(yintercept = 0.5, linetype = 2, color = INK, linewidth = 0.5) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.12, linewidth = 0.6,
                na.rm = TRUE) +
  geom_point(size = 3.0) +
  geom_text(aes(label = sprintf("%.3f", auroc)), vjust = -1.25, hjust = 0.5,
            size = 3.0, color = INK, show.legend = FALSE) +
  geom_point(data = filter(Cd, clean_comparison), shape = 21, size = 6.4,
             stroke = 0.7, fill = NA, show.legend = FALSE) +
  annotate("text", x = which(Cd$clean_comparison), y = 0.70,
           label = "same model,\nboth tasks", size = 2.9, color = INK,
           lineheight = 1.05, vjust = 1) +
  annotate("text", x = -Inf, y = 0.5, hjust = -0.06, vjust = -0.6,
           size = 2.9, color = INK, label = "chance") +
  scale_color_manual(values = c(`canonical vs negative`    = CANON,
                                `canonical vs competitor`  = CC)) +
  coord_cartesian(ylim = c(0.40, 1.10)) +
  scale_y_continuous(breaks = seq(0.4, 1.0, 0.1)) +
  labs(title = "C", x = NULL, y = "AUROC") +
  th

CAP <- paste0(
  "A: ", format(nA, big.mark = ","), " genes with an APARENT2 score at both the annotated 3' end and at ",
  "the highest-consensus competitor of the analysis set; consensus = mean(rf_prob, lr_prob, cnn_prob). ",
  "Hex fill is the median distance between the two sites. Dashed line is y = x.\n",
  "B: motif density per peak. Source positions are -100..+99; the 9-point moving average ",
  "(as in make_figure2.R) drops 4 positions at each end, so the drawn lines span -96..+95. ",
  "Denominators are asymmetric by construction: 6,583 canonical peaks against 43,547 analysis-set ",
  "competitors, so densities are comparable but the two curves rest on different peak counts.\n",
  "C: RF/LR/CNN canonical-vs-negative are the locked test metrics and carry no stored CI. ",
  "APARENT2 appears on both tasks and is the only same-model contrast; its CIs are bootstraps over sites.")

fig <- (pA | pB) / pC +
  plot_layout(heights = c(1.15, 0.85)) +
  plot_annotation(
    caption = CAP,
    theme = theme(plot.caption = element_text(size = 8.5, color = "#555555",
                                              hjust = 0, lineheight = 1.15)))

ggsave(p_in("paradox_fig.pdf"), fig, width = 17, height = 12, device = cairo_pdf)
ggsave(p_in("paradox_fig.png"), fig, width = 17, height = 12, dpi = 400)

cat(sprintf("[out] paradox_fig.pdf, paradox_fig.png\n"))
cat(sprintf("      panel A: n=%d genes, AUROC(best competitor)=%.4f, canonical higher in %.4f\n",
            nA, auA, winA))
cat(sprintf("      panel B: %d rows, positions %d..%d\n", nrow(B), min(B$position), max(B$position)))
cat(sprintf("      panel C: %d rows\n", nrow(Cd)))
