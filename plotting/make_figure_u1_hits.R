#!/usr/bin/env Rscript
# what distinguishes the model false positives a u1 amo experiment actually used
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr)
  library(patchwork); library(scales)
})
INK <- "#1A1A1A"; BASE <- 7
GRP <- c("canonical" = "#0072B2", "U1 AMO hit" = "#D55E00",
         "control AMO hit" = "#009E73", "not hit" = "#8A8F98")
LV <- names(GRP)

th <- theme_classic(base_size = BASE, base_family = "sans") + theme(
  plot.title = element_text(face = "bold", size = BASE + .5, colour = INK, hjust = 0,
                            margin = margin(b = 2)),
  plot.title.position = "plot",
  plot.subtitle = element_text(size = BASE - .5, colour = INK, hjust = 0, margin = margin(b = 4)),
  axis.text = element_text(colour = INK, size = BASE - .5),
  axis.title = element_text(colour = INK, size = BASE),
  axis.line = element_line(linewidth = .25), axis.ticks = element_line(linewidth = .25),
  legend.position = "none", plot.margin = margin(3, 5, 3, 4))

# a: discriminant distributions
DI <- read_tsv("u1_hit_ecdf.tsv", show_col_types = FALSE, progress = FALSE) %>%
  mutate(group = factor(group, levels = LV))
KEY <- DI %>% distinct(group, n, median) %>%
  mutate(lab = sprintf("%s  n = %s, median %.2f", group, comma(n), median))
pa <- ggplot(DI, aes(value, ecdf, colour = group)) +
  geom_line(linewidth = .5) +
  geom_text(data = KEY, aes(x = -3.95, y = c(.98, .90, .82, .74), label = lab),
            hjust = 0, size = 1.9, show.legend = FALSE) +
  scale_colour_manual(values = GRP) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  coord_cartesian(xlim = c(-4, -0.7)) +
  labs(title = "a  Used false positives score above unused ones",
       subtitle = "Held-out discriminant, ECDF. A curve to the right scores higher.",
       x = "held-out discriminant", y = "of the group at or below", colour = NULL) + th

# b: cliff's delta forest
QL <- c("discriminant" = "discriminant",
        "spl_upstream_max_me3" = "acceptor strength",
        "gc_contrast" = "gc_contrast")
S <- read_tsv("u1_hit_stats.tsv", show_col_types = FALSE, progress = FALSE) %>%
  filter(group_b == "not hit", group_a %in% c("U1 AMO hit", "control AMO hit")) %>%
  mutate(q = factor(QL[quantity], levels = rev(QL)),
         group_a = factor(group_a, levels = c("control AMO hit", "U1 AMO hit")),
         lab = sprintf("%+.3f", cliff_delta))
pb <- ggplot(S, aes(cliff_delta, q, colour = group_a)) +
  geom_vline(xintercept = 0, colour = "#8A8F98", linewidth = .25) +
  geom_errorbar(aes(xmin = delta_lo, xmax = delta_hi), orientation = "y", width = 0,
                linewidth = .45, position = position_dodge(width = .5)) +
  geom_point(size = 1.4, position = position_dodge(width = .5)) +
  geom_text(aes(label = lab, vjust = ifelse(group_a == "U1 AMO hit", -1.3, 2.1)),
            size = 1.9, position = position_dodge(width = .5), show.legend = FALSE) +
  scale_colour_manual(values = GRP[c("control AMO hit", "U1 AMO hit")],
                      breaks = c("U1 AMO hit", "control AMO hit")) +
  scale_x_continuous(limits = c(-.1, .48)) +
  labs(title = "b  The control AMO shows the same shift",
       subtitle = paste("Against the non-hit false positives, gene-clustered 95% CI.",
                        "Control AMO is the\nsame assay without the perturbation, so a",
                        "matching shift means detection."),
       x = "Cliff's delta against not hit", y = NULL, colour = NULL) + th +
  theme(legend.position = "bottom", legend.key.size = unit(3, "mm"),
        legend.text = element_text(size = BASE - .5), legend.margin = margin(t = -3))

# c: host intron rank
IN <- read_tsv("u1_hit_intron.tsv", show_col_types = FALSE, progress = FALSE) %>%
  mutate(group = factor(group, levels = LV[LV != "canonical"]),
         label = factor(label, levels = c(as.character(1:9), "10+")))
F2 <- IN %>% distinct(group, frac_first_two, ci_lo, ci_hi, n_group) %>%
  mutate(lab = sprintf("%s  %.1f%% [%.1f, %.1f]  n = %s", group,
                       100 * frac_first_two, 100 * ci_lo, 100 * ci_hi, comma(n_group)))
pc <- ggplot(IN, aes(label, frac, fill = group)) +
  geom_col(width = .72, position = position_dodge(width = .78)) +
  scale_fill_manual(values = GRP) +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     expand = expansion(c(0, .28))) +
  geom_text(data = F2, aes(x = 4.3, y = c(.50, .44, .38), label = lab, colour = group),
            inherit.aes = FALSE, hjust = 0, size = 1.9, show.legend = FALSE) +
  scale_colour_manual(values = GRP) +
  labs(title = "c  U1 AMO hits sit in the first introns",
       subtitle = paste("Host intron of the Ensembl canonical transcript, ranked from",
                        "the TSS.\nLabels give the share in the first two introns; Oh 2017",
                        "reported 38%."),
       x = "host intron rank from the TSS", y = "of the group", fill = NULL) + th +
  theme(legend.position = "none")

# d: top-n robustness
R <- read_tsv("ovl_topn_competitor.tsv", show_col_types = FALSE, progress = FALSE) %>%
  filter(tolerance == 50) %>%
  mutate(N = as.integer(sub("^.*?(\\d+)k$", "\\1", dataset)) * 1000,
         cond = ifelse(grepl("^u1", dataset), "U1 AMO hit", "control AMO hit"),
         cond = factor(cond, levels = LV[LV != "canonical"]),
         lab = sprintf("%s hits", comma(n_external_in_competitors)))
pd_ <- ggplot(R, aes(N, fold_enrichment, colour = cond)) +
  geom_hline(yintercept = 1, colour = "#8A8F98", linewidth = .25) +
  geom_line(linewidth = .5) + geom_point(size = 1.4) +
  geom_text(aes(label = lab), size = 1.9, vjust = -1.1, show.legend = FALSE) +
  scale_colour_manual(values = GRP) +
  scale_x_continuous(breaks = c(20000, 40000, 60000), labels = comma) +
  scale_y_continuous(limits = c(0, 5.9)) +
  labs(title = "d  The separation survives the peak-call cut",
       subtitle = paste("Every cut lands inside the tie block at the signal floor,",
                        "so which tied sites\nare kept is set by chromosome order.",
                        "Fold is against the displaced null."),
       x = "sites kept by the peak call", y = "fold over the displaced null", colour = NULL) + th +
  theme(legend.position = "bottom", legend.key.size = unit(3, "mm"),
        legend.text = element_text(size = BASE - .5), legend.margin = margin(t = -3))

fig <- (pa | pb) / (pc | pd_)
ggsave("figure_u1_hits.pdf", fig, width = 180, height = 116, units = "mm",
       device = grDevices::cairo_pdf)
ggsave("figure_u1_hits.png", fig, width = 180, height = 116, units = "mm", dpi = 400)
cat("wrote figure_u1_hits.pdf / .png\n")
