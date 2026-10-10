#!/usr/bin/env Rscript
# raster re-anchored on the max 3'ss / acceptor (mirrors tested .py)
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(readr); library(tidyr)})
args   <- commandArgs(trailingOnly = TRUE)
TSV    <- if (length(args) >= 1) args[1] else "fig3_raster_sites.tsv"
STEM   <- if (length(args) >= 2) args[2] else "figure3_anchored"
THR    <- if (length(args) >= 3) as.numeric(args[3]) else 8
U1THR  <- if (length(args) >= 4) as.numeric(args[4]) else -8
LO     <- if (length(args) >= 5) as.numeric(args[5]) else -200
HI     <- if (length(args) >= 6) as.numeric(args[6]) else 800
BIN    <- if (length(args) >= 7) as.numeric(args[7]) else 20
CLSLAB <- c(Competitor = "iCPA (competitors)", Canonical = "ccPA (canonical)")

d <- read_tsv(TSV, show_col_types = FALSE)
anchor <- d %>% filter(signal == "acceptor") %>% group_by(cls, site_id) %>%
  slice_max(score, n = 1, with_ties = FALSE) %>% ungroup() %>% select(cls, site_id, apos = pos)
d <- d %>% inner_join(anchor, by = c("cls", "site_id")) %>% mutate(rel = pos - apos)
ord <- anchor %>% group_by(cls) %>% mutate(row = rank(abs(apos), ties.method = "first")) %>%
  ungroup() %>% select(cls, site_id, row)
d <- d %>% inner_join(ord, by = c("cls", "site_id"))
anchor <- anchor %>% inner_join(ord, by = c("cls", "site_id")) %>% mutate(site_rel = -apos)

don <- d %>% filter(signal == "donor", score >= THR, rel >= LO, rel <= HI)
u1  <- d %>% filter(signal == "u1", score <= U1THR, rel >= LO, rel <= HI)
for (df in c("don", "u1", "anchor")) {}
don$cls    <- factor(don$cls,    levels = c("Competitor", "Canonical"))
u1$cls     <- factor(u1$cls,     levels = c("Competitor", "Canonical"))
anchor$cls <- factor(anchor$cls, levels = c("Competitor", "Canonical"))

hm <- ggplot() +
  geom_point(data = u1,  aes(rel, row), shape = 6, color = "#1D9E75", size = .7, alpha = .55) +
  geom_point(data = don, aes(rel, row), shape = 4, color = "#D55E00", size = 1.1, stroke = .5, alpha = .85) +
  geom_point(data = anchor, aes(site_rel, row), shape = 18, color = "#111111", size = .9) +
  geom_vline(xintercept = 0, color = "#0072B2", linewidth = .6) +
  facet_wrap(~cls, nrow = 1, labeller = labeller(cls = CLSLAB)) +
  scale_y_reverse(expand = c(0, 0)) + coord_cartesian(xlim = c(LO, HI)) +
  labs(x = "position relative to max 3\u2032ss / acceptor (nt)",
       y = "gene  (sorted by acceptor\u2192polyA distance)",
       title = "Raster anchored on the max 3\u2032ss (acceptor)") +
  theme_classic(base_size = 12) +
  theme(strip.background = element_blank(), strip.text = element_text(face = "bold"),
        axis.text.y = element_blank(), axis.ticks.y = element_blank(), panel.spacing = unit(1, "lines"))

edges <- seq(LO, HI, by = BIN)
prof <- don %>% mutate(rb = cut(rel, edges, labels = FALSE)) %>%
  distinct(cls, site_id, rb) %>% count(cls, rb)
ntot <- d %>% distinct(cls, site_id) %>% count(cls, name = "ntot")
prof <- prof %>% left_join(ntot, by = "cls") %>%
  mutate(frac = n / ntot, x = LO + (rb - 0.5) * BIN)
pf <- ggplot(prof, aes(x, frac, color = cls)) + geom_line(linewidth = .8) +
  geom_vline(xintercept = 0, color = "#0072B2", linewidth = .6) +
  scale_color_manual(values = c(Competitor = "#D55E00", Canonical = "#0072B2"), labels = CLSLAB, name = NULL) +
  coord_cartesian(xlim = c(LO, HI)) +
  labs(x = NULL, y = "frac genes\nstrong 5\u2032ss") +
  theme_classic(base_size = 12) + theme(legend.position = "top", axis.text.x = element_blank())

if (requireNamespace("patchwork", quietly = TRUE)) {
  library(patchwork); g <- pf / hm + plot_layout(heights = c(1, 5))
  ggsave(paste0(STEM, ".pdf"), g, width = 13, height = 8, device = cairo_pdf)
  ggsave(paste0(STEM, ".png"), g, width = 13, height = 8, dpi = 300)
} else {
  ggsave(paste0(STEM, ".pdf"), hm, width = 13, height = 7, device = cairo_pdf)
  ggsave(paste0(STEM, "_profile.pdf"), pf, width = 13, height = 2.4, device = cairo_pdf)
  ggsave(paste0(STEM, ".png"), hm, width = 13, height = 7, dpi = 300)
  message("[note] patchwork not installed: raster and profile saved separately")
}
cat(sprintf("[out] %s  rel window [%g,%g]\n", STEM, LO, HI))