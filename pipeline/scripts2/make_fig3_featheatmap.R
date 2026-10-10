#!/usr/bin/env Rscript
# direct icpa vs ccpa heatmap of per-gene features (mirrors tested .py)
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(readr); library(tidyr)})
args <- commandArgs(trailingOnly = TRUE)
TSV  <- if (length(args) >= 1) args[1] else "fig3_raster_sites.tsv"
STEM <- if (length(args) >= 2) args[2] else "figure3_featheat"
LO   <- if (length(args) >= 3) as.numeric(args[3]) else -400
HI   <- if (length(args) >= 4) as.numeric(args[4]) else 100
CLSLAB <- c(Competitor = "iCPA (competitors)", Canonical = "ccPA (canonical)")

d <- read_tsv(TSV, show_col_types = FALSE)
max3 <- d %>% filter(signal == "acceptor") %>% group_by(cls, site_id) %>%
  summarise(value = max(score), .groups = "drop") %>% mutate(feature = "max3")
best5 <- d %>% filter(signal == "donor", pos >= LO, pos <= HI) %>% group_by(cls, site_id) %>%
  summarise(value = max(score), .groups = "drop") %>% mutate(feature = "best5")
bestu <- d %>% filter(signal == "u1", pos >= LO, pos <= HI) %>% group_by(cls, site_id) %>%
  summarise(value = max(-score), .groups = "drop") %>% mutate(feature = "bestU1")
feat <- bind_rows(max3, best5, bestu) %>%
  group_by(feature) %>% mutate(z = (value - mean(value, na.rm = TRUE)) / sd(value, na.rm = TRUE)) %>% ungroup()

m3w <- max3 %>% select(cls, site_id, value) %>% pivot_wider(names_from = cls, values_from = value) %>%
  filter(!is.na(Canonical), !is.na(Competitor)) %>% mutate(delta = Canonical - Competitor) %>%
  arrange(desc(delta)) %>% mutate(row = row_number())
feat <- feat %>% inner_join(m3w %>% select(site_id, row), by = "site_id")

feat$feature <- factor(feat$feature, levels = c("max3", "best5", "bestU1"),
                       labels = c("max 3\u2032ss", "best 5\u2032ss\n(window)", "best U1\n(window)"))
feat$cls <- factor(feat$cls, levels = c("Competitor", "Canonical"))

p <- ggplot(feat, aes(feature, row, fill = pmax(pmin(z, 2), -2))) +
  geom_tile() +
  facet_wrap(~cls, nrow = 1, labeller = labeller(cls = CLSLAB)) +
  scale_fill_distiller(palette = "RdBu", limits = c(-2, 2), name = "z-score\n(per column)") +
  scale_y_reverse(expand = c(0, 0)) + scale_x_discrete(expand = c(0, 0)) +
  labs(title = sprintf("Per-gene splice features: iCPA vs ccPA  (5\u2032ss/U1 in [%g,%g] nt)", LO, HI),
       x = NULL, y = "gene  (ordered by within-gene \u0394 max 3\u2032ss)") +
  theme_classic(base_size = 12) +
  theme(strip.background = element_blank(), strip.text = element_text(face = "bold"),
        axis.text.y = element_blank(), axis.ticks.y = element_blank(),
        axis.text.x = element_text(size = 9), panel.spacing = unit(1, "lines"))
ggsave(paste0(STEM, ".pdf"), p, width = 8.4, height = 8, device = cairo_pdf)
ggsave(paste0(STEM, ".png"), p, width = 8.4, height = 8, dpi = 300)
cat(sprintf("[out] %s  n=%d\n", STEM, nrow(m3w)))