#!/usr/bin/env Rscript
# per-gene best-in-window motif, position collapsed (mirrors tested .py)
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(readr); library(tidyr)})
args   <- commandArgs(trailingOnly = TRUE)
TSV    <- if (length(args) >= 1) args[1] else "fig3_raster_sites.tsv"
STEM   <- if (length(args) >= 2) args[2] else "figure3_window"
SIGNAL <- if (length(args) >= 3) args[3] else "donor"
LO     <- if (length(args) >= 4) as.numeric(args[4]) else -400
HI     <- if (length(args) >= 5) as.numeric(args[5]) else 100
THR    <- if (length(args) >= 6) as.numeric(args[6]) else 8
U1THR  <- if (length(args) >= 7) as.numeric(args[7]) else -8
auroc <- function(pos, neg) if (!length(pos) || !length(neg)) NA else
  as.numeric(wilcox.test(pos, neg)$statistic) / (length(pos) * length(neg))
unit <- if (SIGNAL == "u1") "\u2212kcal/mol" else "bits"

d <- read_tsv(TSV, show_col_types = FALSE) %>%
  filter(signal == SIGNAL, pos >= LO, pos <= HI) %>%
  mutate(strength = if (SIGNAL == "u1") -score else score,
         strong = if (SIGNAL == "u1") score <= U1THR else score >= THR)
best <- d %>% group_by(cls, site_id) %>% slice_max(strength, n = 1, with_ties = FALSE) %>% ungroup()
w <- best %>% select(cls, site_id, strength, pos) %>%
  pivot_wider(names_from = cls, values_from = c(strength, pos)) %>%
  filter(!is.na(strength_Canonical), !is.na(strength_Competitor))
cs <- w$strength_Canonical; xs <- w$strength_Competitor
cp <- abs(w$pos_Canonical); xp <- abs(w$pos_Competitor)
n <- nrow(w); delta <- cs - xs; med <- median(delta); frac <- mean(delta > 0)
wp <- suppressWarnings(wilcox.test(cs, xs, paired = TRUE)$p.value); rocs <- auroc(cs, xs)
hs <- d %>% filter(strong) %>% distinct(cls, site_id) %>% count(cls)
gv <- function(k) { x <- hs$n[hs$cls == k]; if (length(x)) x else 0 }
tot <- d %>% distinct(cls, site_id) %>% count(cls)
tv <- function(k) { x <- tot$n[tot$cls == k]; if (length(x)) x else 0 }

pA <- ggplot(data.frame(delta), aes(delta)) +
  geom_histogram(bins = 40, fill = "#6BAED6", color = "white", linewidth = .2) +
  geom_vline(xintercept = 0, linewidth = .8) +
  geom_vline(xintercept = med, color = "#D55E00", linewidth = 1, linetype = 2) +
  labs(x = sprintf("within-gene \u0394 best %s (%s)", SIGNAL, unit), y = "genes",
       title = sprintf("A  median %+.2f | %.0f%% canon stronger | p=%.1e", med, 100*frac, wp)) +
  theme_classic(base_size = 11)
elong <- bind_rows(data.frame(v = cs, cls = "canonical"), data.frame(v = xs, cls = "competitor"))
pB <- ggplot(elong, aes(v, color = cls)) + stat_ecdf(linewidth = .9) +
  scale_color_manual(values = c(canonical = "#0072B2", competitor = "#D55E00"), name = NULL) +
  labs(x = sprintf("best %s in window (%s)", SIGNAL, unit), y = "cumulative fraction",
       title = sprintf("B  best-in-window AUROC %.3f (position collapsed)", rocs)) +
  theme_classic(base_size = 11) + theme(legend.position = c(.75, .25))
dlong <- bind_rows(data.frame(v = cp, cls = "canonical"), data.frame(v = xp, cls = "competitor"))
pC <- ggplot(dlong, aes(v, color = cls)) + stat_ecdf(linewidth = .9) +
  scale_color_manual(values = c(canonical = "#0072B2", competitor = "#D55E00"), name = NULL) +
  labs(x = "|distance of best motif to site| (nt)", y = "cumulative fraction",
       title = sprintf("C  canon strong %d/%d, comp strong %d/%d",
                       gv("Canonical"), tv("Canonical"), gv("Competitor"), tv("Competitor"))) +
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
cat(sprintf("[out] %s  n=%d signal=%s window=[%g,%g] AUROC(best)=%.3f frac=%.3f strong canon=%d/%d comp=%d/%d\n",
            STEM, n, SIGNAL, LO, HI, rocs, frac, gv("Canonical"), tv("Canonical"), gv("Competitor"), tv("Competitor")))