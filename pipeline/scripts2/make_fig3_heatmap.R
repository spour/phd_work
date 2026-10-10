#!/usr/bin/env Rscript
# gene-paired splice-signal metagene heatmap (mirrors tested .py)
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(readr); library(tidyr); library(scales)})
args   <- commandArgs(trailingOnly = TRUE)
TSV    <- if (length(args) >= 1) args[1] else "fig3_raster_sites.tsv"
STEM   <- if (length(args) >= 2) args[2] else "figure3_heatmap"
SIGNAL <- if (length(args) >= 3) args[3] else "donor"
THR    <- if (length(args) >= 4) as.numeric(args[4]) else 8.0
SMOOTH <- if (length(args) >= 5) as.numeric(args[5]) else 1.5
BIN    <- if (length(args) >= 6) as.numeric(args[6]) else 25
SORT   <- if (length(args) >= 7) args[7] else "comp"
COLOR  <- if (length(args) >= 8) args[8] else "occ"
FLOOR  <- if (length(args) >= 9) as.numeric(args[9]) else 4.0
MAX_NT <- 2000; DN_NT <- 500; NBIN <- (MAX_NT + DN_NT) / BIN
CLSLAB <- c(Competitor = "iCPA (competitors)", Canonical = "ccPA (canonical)")
ss <- ifelse(SIGNAL == "donor", "5", "3")

gsmooth <- function(x, sigma) {
  if (sigma <= 0) return(x)
  k <- max(3, round(sigma * 6)); if (k %% 2 == 0) k <- k + 1
  t <- seq(-(k %/% 2), (k %/% 2)); g <- exp(-t^2 / (2 * sigma^2)); g <- g / sum(g)
  h <- k %/% 2; xp <- c(rep(x[1], h), x, rep(x[length(x)], h))
  as.numeric(stats::filter(xp, g, sides = 2))[(h + 1):(h + length(x))]
}

d <- read_tsv(TSV, show_col_types = FALSE) %>% filter(pos >= -MAX_NT, pos <= DN_NT)
anchor <- d %>% filter(signal == "acceptor") %>%
  group_by(cls, site_id) %>% slice_max(score, n = 1, with_ties = FALSE) %>%
  ungroup() %>% select(cls, site_id, apos = pos)
sites <- d %>% distinct(cls, site_id)

sig <- d %>% filter(signal == SIGNAL) %>%
  mutate(bin = pmin(pmax(floor((pos + MAX_NT) / BIN), 0), NBIN - 1))
if (COLOR == "occ") {
  g <- sig %>% filter(score >= THR) %>% distinct(cls, site_id, bin) %>% mutate(val = 1)
} else {
  g <- sig %>% group_by(cls, site_id, bin) %>% summarise(val = max(score), .groups = "drop")
}
grid <- sites %>% tidyr::crossing(bin = 0:(NBIN - 1)) %>%
  left_join(g, by = c("cls", "site_id", "bin")) %>%
  mutate(val = ifelse(is.na(val), 0, val))
if (COLOR == "occ") {
  grid <- grid %>% group_by(cls, site_id) %>% arrange(bin, .by_group = TRUE) %>%
    mutate(v = gsmooth(val, SMOOTH)) %>% ungroup()
} else {
  grid$v <- grid$val
}

if (SORT == "self") {
  ord <- anchor %>% group_by(cls) %>% mutate(row = rank(abs(apos), ties.method = "first")) %>%
    ungroup() %>% select(cls, site_id, row)
  grid   <- grid %>% inner_join(ord, by = c("cls", "site_id"))
  anchor <- anchor %>% inner_join(ord, by = c("cls", "site_id"))
} else {
  both <- intersect(grid$site_id[grid$cls == "Competitor"], grid$site_id[grid$cls == "Canonical"])
  ord <- anchor %>% filter(cls == "Competitor", site_id %in% both) %>%
    mutate(dist = abs(apos)) %>% arrange(dist) %>% mutate(row = row_number()) %>% select(site_id, row)
  grid   <- grid %>% inner_join(ord, by = "site_id")
  anchor <- anchor %>% inner_join(ord, by = "site_id")
}
grid$posc  <- -MAX_NT + (grid$bin + 0.5) * BIN
grid$cls   <- factor(grid$cls,   levels = c("Competitor", "Canonical"))
anchor$cls <- factor(anchor$cls, levels = c("Competitor", "Canonical"))

if (COLOR == "occ") {
  fillsc <- scale_fill_viridis_c(option = "magma", name = paste0("strong ", ss, "\u2032ss\ndensity"))
  ptitle <- sprintf("Strong %s (\u2265%g bits) around the polyA site", SIGNAL, THR)
  ylabp  <- "frac genes\nstrong motif"
} else {
  cap <- as.numeric(quantile(grid$v[grid$v > 0], 0.999, na.rm = TRUE))
  fillsc <- scale_fill_viridis_c(option = "magma", limits = c(FLOOR, cap), oob = squish,
                                 name = paste0(ss, "\u2032ss MaxEnt\n(bits)"))
  ptitle <- sprintf("%s MaxEnt score around the polyA site", tools::toTitleCase(SIGNAL))
  ylabp  <- paste0("mean max\n", ss, "\u2032ss")
}

hm <- ggplot(grid, aes(posc, row, fill = v)) +
  geom_tile(width = BIN) +
  geom_point(data = anchor, aes(apos, row), inherit.aes = FALSE,
             shape = 1, color = "#22D3EE", size = 1, stroke = .5) +
  geom_vline(xintercept = 0, color = "white", linewidth = .4) +
  facet_wrap(~cls, nrow = 1, labeller = labeller(cls = CLSLAB)) + fillsc +
  scale_y_reverse(expand = c(0, 0)) +
  scale_x_continuous(expand = c(0, 0), breaks = seq(-MAX_NT, DN_NT, 500)) +
  labs(title = ptitle, x = "position relative to polyA site (nt)",
       y = sprintf("gene  (sorted by %s anchor distance)", ifelse(SORT == "self", "own", "competitor"))) +
  theme_classic(base_size = 12) +
  theme(strip.background = element_blank(), strip.text = element_text(face = "bold"),
        axis.text.y = element_blank(), axis.ticks.y = element_blank(), panel.spacing = unit(1, "lines"))

prof <- grid %>% group_by(cls, posc) %>% summarise(m = mean(v), .groups = "drop")
pf <- ggplot(prof, aes(posc, m, color = cls)) +
  geom_line(linewidth = .8) + geom_vline(xintercept = 0, color = "grey40", linewidth = .4) +
  scale_color_manual(values = c(Competitor = "#D55E00", Canonical = "#0072B2"), labels = CLSLAB, name = NULL) +
  scale_x_continuous(expand = c(0, 0), breaks = seq(-MAX_NT, DN_NT, 500)) +
  labs(x = NULL, y = ylabp) + theme_classic(base_size = 12) +
  theme(legend.position = "top", axis.text.x = element_blank())

if (requireNamespace("patchwork", quietly = TRUE)) {
  library(patchwork); g2 <- pf / hm + plot_layout(heights = c(1, 5))
  ggsave(paste0(STEM, ".pdf"), g2, width = 13, height = 8, device = cairo_pdf)
  ggsave(paste0(STEM, ".png"), g2, width = 13, height = 8, dpi = 300)
} else {
  ggsave(paste0(STEM, ".pdf"), hm, width = 13, height = 7, device = cairo_pdf)
  ggsave(paste0(STEM, "_profile.pdf"), pf, width = 13, height = 2.4, device = cairo_pdf)
  ggsave(paste0(STEM, ".png"), hm, width = 13, height = 7, dpi = 300)
  message("[note] patchwork not installed: heatmap and profile saved separately")
}
cat(sprintf("[out] %s  signal=%s color=%s sort=%s\n", STEM, SIGNAL, COLOR, SORT))