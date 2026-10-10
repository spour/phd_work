#!/usr/bin/env Rscript
# per-gene max-3'ss effect behind figure 3 (mirrors the tested .py)
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(readr); library(tidyr)})
args <- commandArgs(trailingOnly = TRUE)
TSV  <- if (length(args) >= 1) args[1] else "fig3_raster_sites.tsv"
STEM <- if (length(args) >= 2) args[2] else "figure3_delta"

auroc <- function(pos, neg) as.numeric(wilcox.test(pos, neg)$statistic) / (length(pos) * length(neg))

d <- read_tsv(TSV, show_col_types = FALSE) %>% filter(signal == "acceptor")
anc <- d %>% group_by(cls, site_id) %>% slice_max(score, n = 1, with_ties = FALSE) %>% ungroup()
w <- anc %>% select(cls, site_id, score, pos) %>%
  pivot_wider(names_from = cls, values_from = c(score, pos)) %>%
  filter(!is.na(score_Canonical), !is.na(score_Competitor))
cs <- w$score_Canonical; xs <- w$score_Competitor
ps <- abs(w$pos_Canonical); qs <- abs(w$pos_Competitor)
n <- nrow(w); delta <- cs - xs
med <- median(delta); frac <- mean(delta > 0)
wp  <- suppressWarnings(wilcox.test(cs, xs, paired = TRUE)$p.value)
rs  <- auroc(cs, xs); rd <- auroc(ps, qs)

pA <- ggplot(data.frame(delta), aes(delta)) +
  geom_histogram(bins = 40, fill = "#6BAED6", color = "white", linewidth = .2) +
  geom_vline(xintercept = 0, linewidth = .8) +
  geom_vline(xintercept = med, color = "#D55E00", linewidth = 1, linetype = 2) +
  labs(x = "within-gene \u0394 max 3\u2032ss (canonical \u2212 competitor)", y = "genes",
       title = sprintf("A  median %+.2f | %.0f%% canon stronger | p=%.1e", med, 100*frac, wp)) +
  theme_classic(base_size = 11)

el <- bind_rows(data.frame(v = cs, cls = "canonical"), data.frame(v = xs, cls = "competitor"))
pB <- ggplot(el, aes(v, color = cls)) + stat_ecdf(linewidth = .9) +
  scale_color_manual(values = c(canonical = "#0072B2", competitor = "#D55E00"), name = NULL) +
  labs(x = "max 3\u2032ss (MaxEnt bits)", y = "cumulative fraction",
       title = sprintf("B  score AUROC %.3f", rs)) +
  theme_classic(base_size = 11) + theme(legend.position = c(.75, .25))

ed <- bind_rows(data.frame(v = ps, cls = "canonical"), data.frame(v = qs, cls = "competitor"))
pC <- ggplot(ed, aes(v, color = cls)) + stat_ecdf(linewidth = .9) +
  scale_color_manual(values = c(canonical = "#0072B2", competitor = "#D55E00"), name = NULL) +
  labs(x = "|anchor distance to polyA site| (nt)", y = "cumulative fraction",
       title = sprintf("C  distance AUROC %.3f", rd)) +
  theme_classic(base_size = 11) + theme(legend.position = c(.75, .25))

if (requireNamespace("patchwork", quietly = TRUE)) {
  library(patchwork)
  ggsave(paste0(STEM, ".pdf"), pA | pB | pC, width = 15, height = 4.6, device = cairo_pdf)
  ggsave(paste0(STEM, ".png"), pA | pB | pC, width = 15, height = 4.6, dpi = 200)
} else {
  ggsave(paste0(STEM, "_A.pdf"), pA, width = 5, height = 4.6, device = cairo_pdf)
  ggsave(paste0(STEM, "_B.pdf"), pB, width = 5, height = 4.6, device = cairo_pdf)
  ggsave(paste0(STEM, "_C.pdf"), pC, width = 5, height = 4.6, device = cairo_pdf)
  message("[note] patchwork not installed: panels saved separately")
}
cat(sprintf("[out] %s  n=%d AUROC(score)=%.3f frac=%.3f median=%+.3f AUROC(dist)=%.3f\n",
            STEM, n, rs, frac, med, rd))