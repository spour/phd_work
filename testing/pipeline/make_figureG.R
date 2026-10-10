#!/usr/bin/env Rscript
# figure g: sequence-model score vs measured polyasite expression

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(readr); library(scales)
})
args  <- commandArgs(trailingOnly = TRUE)
INDIR <- if (length(args) >= 1) args[1] else "."
MODELS <- c("RF","LR","CNN"); MCOL <- c(RF="#0072B2", LR="#E69F00", CNN="#009E73")
INK <- "#1A1A1A"; BASE <- 17
th <- theme_classic(base_size = BASE) +
  theme(text=element_text(color=INK), axis.text=element_text(color=INK,size=BASE-3),
        axis.title=element_text(size=BASE-1), axis.line=element_line(color=INK,linewidth=0.5),
        axis.ticks=element_line(color=INK,linewidth=0.5),
        plot.title=element_text(size=BASE+2,face="bold",hjust=0),
        plot.subtitle=element_text(size=BASE-4,color="#555555"),
        legend.position="top", legend.title=element_blank(),
        legend.text=element_text(size=BASE-3), plot.margin=margin(6,8,6,8))

fe <- file.path(INDIR, "figG_expr.tsv"); fc <- file.path(INDIR, "figG_corr.tsv")
if (!file.exists(fe)) stop("figG_expr.tsv not found; run make_figure1_data.py first")

ex <- read_tsv(fe, show_col_types = FALSE) |>
  mutate(model = factor(model, levels = MODELS))
pL <- ggplot(ex, aes(mean_expr, mean_score, color = model)) +
  geom_line(linewidth = 0.9) + geom_point(size = 2.2) +
  scale_color_manual(values = MCOL) +
  scale_x_continuous(trans = "log10", labels = label_number(accuracy = 0.01)) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(title = "G", subtitle = "Model score vs PolyASite expression",
       x = "Atlas expression (median per bin, log)", y = "Mean model score") + th

if (file.exists(fc)) {
  co <- read_tsv(fc, show_col_types = FALSE) |>
    mutate(model = factor(model, levels = MODELS))
  pR <- ggplot(co, aes(model, spearman, fill = model)) +
    geom_col(width = 0.6) +
    geom_text(aes(label = sprintf("%.2f", spearman)), vjust = -0.4, size = 4, color = INK) +
    scale_fill_manual(values = MCOL, guide = "none") +
    scale_y_continuous(limits = c(0, 1), expand = expansion(mult = c(0, 0.05))) +
    labs(subtitle = "Spearman(score, expression)", x = NULL, y = "Spearman rho") + th
  fig <- pL + pR + plot_layout(widths = c(1.5, 1))
} else fig <- pL

ggsave(file.path(INDIR, "figureG.pdf"), fig, width = 10, height = 4.6, device = cairo_pdf)
ggsave(file.path(INDIR, "figureG.png"), fig, width = 10, height = 4.6, dpi = 300)
cat("[out] figureG.pdf, figureG.png\n")