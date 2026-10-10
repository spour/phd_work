#!/usr/bin/env Rscript
# do external experiments ever use a model false positive
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr)
  library(patchwork); library(scales)
})
INK <- "#1A1A1A"; CAN <- "#0072B2"; COMP <- "#E69F00"; NUL <- "#8A8F98"
DARK <- "#1F4E5F"; BASE <- 7
TOL <- 50

th <- theme_classic(base_size = BASE, base_family = "sans") + theme(
  plot.title = element_text(face = "bold", size = BASE + .5, colour = INK, hjust = 0,
                            margin = margin(b = 2)),
  plot.title.position = "plot",
  plot.subtitle = element_text(size = BASE - .5, colour = INK, hjust = 0, margin = margin(b = 4)),
  axis.text = element_text(colour = INK, size = BASE - .5),
  axis.title = element_text(colour = INK, size = BASE),
  axis.line = element_line(linewidth = .25), axis.ticks = element_line(linewidth = .25),
  legend.position = "none", plot.margin = margin(3, 5, 3, 4))

SHORT <- c(atlas3seq = "3'-seq atlas", oh_camo = "control AMO", oh_u1amo = "U1 AMO")
CIRC <- "atlas3seq"

# a: paired bars
SI <- read_tsv("ovl_panel_a_sites.tsv", show_col_types = FALSE, progress = FALSE) %>%
  filter(tolerance == TOL)
rd <- function(f, tag) read_tsv(f, show_col_types = FALSE, progress = FALSE) %>%
  filter(tolerance == TOL) %>% mutate(set = tag) %>%
  select(dataset, set, fold_enrichment, p_empirical, p_floor)
FP <- bind_rows(rd("ovl_unfilt_competitor.tsv", "unfiltered"),
                rd("ovl_ana_competitor.tsv", "analysis"))

NSITE <- SI %>% distinct(dataset, n_sites)
LAB <- setNames(sprintf("%s\n%s sites%s",
                        c("3'-seq atlas, Singh 2018 + Lee 2018", "Oh 2017 control AMO",
                          "Oh 2017 U1 AMO")[match(NSITE$dataset, names(SHORT))],
                        comma(NSITE$n_sites),
                        ifelse(NSITE$dataset == CIRC, "; circular on the analysis set", "")),
                NSITE$dataset)
LAB <- LAB[names(SHORT)]

A <- left_join(SI, FP, by = c("dataset", "set")) %>%
  mutate(set = factor(set, levels = c("analysis", "unfiltered")),
         name = factor(LAB[dataset], levels = rev(LAB)),
         ptxt = ifelse(p_empirical <= p_floor,
                       sprintf("p < %.3f", p_floor),
                       sprintf("p = %.3f", p_empirical)),
         lab = sprintf("%s hits   %.1f-fold   %s",
                       comma(n_hit), fold_enrichment, ptxt))
XLAB <- max(A$ci_hi) * 1.06
XMAX <- XLAB * 1.66

pa <- ggplot(A, aes(pct_of_dataset, name, fill = set)) +
  geom_col(width = .62, position = position_dodge(width = .68)) +
  geom_errorbar(aes(xmin = ci_lo, xmax = ci_hi), width = .18, linewidth = .3,
                colour = INK, position = position_dodge(width = .68)) +
  geom_text(aes(x = XLAB, label = lab), hjust = 0, size = 2.0, colour = INK,
            position = position_dodge(width = .68)) +
  scale_fill_manual(values = c(unfiltered = NUL, analysis = CAN),
                    breaks = c("unfiltered", "analysis"),
                    labels = c("all 64,140 model false positives",
                               "43,547 after the three filters, in 4,847 genes")) +
  scale_x_continuous(limits = c(0, XMAX), breaks = c(0, 5, 10),
                     expand = expansion(c(0, 0)), labels = function(x) paste0(x, "%")) +
  labs(title = "a  The U1 AMO set still uses false positives",
       subtitle = paste("Match within 50 nt. Each external site counted once; the Oh bigWigs are",
                        "unstranded, so a site hits on either strand.\nBars are Wilson 95% CI.",
                        "Fold and p compare the false positives hit with the same sites displaced",
                        "10 to 100 kb,\n200 rounds, so p floors at 0.005."),
       x = "sites of the dataset landing on a model false positive", y = NULL, fill = NULL) + th +
  theme(axis.text.y = element_text(size = BASE - 1),
        legend.position = "bottom", legend.direction = "vertical",
        legend.key.size = unit(3, "mm"), legend.text = element_text(size = BASE - .5),
        legend.margin = margin(t = -2, b = 0))

# b: upset on the analysis set
P <- read_tsv("ovl_ana_per_peak.tsv", show_col_types = FALSE, progress = FALSE) %>%
  filter(!as.logical(is_canonical))
hc <- paste0("hit_", names(LAB), "_", TOL)
M <- as.matrix(mutate(select(P, all_of(hc)), across(everything(), as.logical)))
M[is.na(M)] <- FALSE
colnames(M) <- names(LAB)
SETS <- names(LAB)
Mh <- M[rowSums(M) > 0, , drop = FALSE]
key <- apply(Mh, 1, function(r) paste(which(r), collapse = ","))
tb <- sort(table(key), decreasing = TRUE)
ids <- names(tb)
inter <- tibble::tibble(ord = seq_along(ids), n = as.integer(tb))
setsz <- colSums(M)

dots <- bind_rows(lapply(seq_along(ids), function(j) {
  mem <- as.integer(strsplit(ids[j], ",")[[1]])
  tibble::tibble(ord = j, si = seq_along(SETS), member = seq_along(SETS) %in% mem)
}))
dots$lab <- factor(SHORT[SETS][dots$si], levels = rev(SHORT[SETS]))
seg <- dots %>% filter(member) %>% group_by(ord) %>%
  summarise(lo = min(si), hi = max(si), .groups = "drop") %>% filter(hi > lo) %>%
  mutate(lo = length(SETS) - lo + 1, hi = length(SETS) - hi + 1)

pb_top <- ggplot(inter, aes(ord, n)) +
  geom_col(width = .6, fill = DARK) +
  geom_text(aes(label = comma(n)), vjust = -0.45, size = 2.0, colour = INK) +
  scale_y_continuous(expand = expansion(c(0, .3)), labels = comma) +
  scale_x_continuous(breaks = inter$ord, expand = expansion(add = .6)) +
  labs(title = "b  The surviving hits come from one dataset each",
       subtitle = paste0(comma(nrow(Mh)), " of the 43,547 false positives carry a match ",
                         "within 50 nt. Bars are the peaks hit by exactly the marked\ncombination."),
       x = NULL, y = "model false positives") + th +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
        axis.line.x = element_blank())

pb_dot <- ggplot(dots, aes(ord, lab)) +
  geom_point(colour = "grey87", size = 1.9) +
  geom_segment(data = seg, aes(x = ord, xend = ord, y = lo, yend = hi),
               inherit.aes = FALSE, linewidth = .4, colour = DARK) +
  geom_point(data = filter(dots, member), colour = DARK, size = 1.9) +
  scale_x_continuous(breaks = inter$ord, expand = expansion(add = .6)) +
  labs(x = NULL, y = NULL) + th +
  theme(axis.text.x = element_blank(), axis.ticks = element_blank(),
        axis.line = element_blank(), axis.text.y = element_text(size = BASE - 1))

pb <- pb_top / pb_dot + plot_layout(heights = c(2.5, 1))

PW <- read_tsv("ovl_pairwise_external_pairwise.tsv", show_col_types = FALSE, progress = FALSE)
g <- function(a, b, col) PW[[col]][PW$row == a & PW$col == b]
CAP <- sprintf(paste0(
  "Lee 2018 reused the Singh 2018 immune samples: their site lists agree at %.2f and %.2f ",
  "within 50 nt (%.2f and %.2f at exact coordinates), so they are merged\ninto one 3'-seq ",
  "atlas set. The two conditions of the Oh 2017 experiment agree at %.2f, and agreement ",
  "across labs is %.2f to %.2f."),
  g("immune", "cll", "frac_row_in_col"), g("cll", "immune", "frac_row_in_col"),
  g("immune", "cll", "frac_exact_match"), g("cll", "immune", "frac_exact_match"),
  g("oh_camo", "oh_u1amo", "frac_row_in_col"),
  min(g("oh_camo", "immune", "frac_row_in_col"), g("immune", "oh_camo", "frac_row_in_col")),
  max(g("oh_camo", "immune", "frac_row_in_col"), g("immune", "oh_camo", "frac_row_in_col")))

fig <- pa / pb + plot_layout(heights = c(1, 1.45)) +
  plot_annotation(caption = CAP,
                  theme = theme(plot.caption = element_text(size = BASE - 1, colour = INK,
                                                            hjust = 0, lineheight = 1.15)))
ggsave("figure_overlap_filtered.pdf", fig, width = 180, height = 126, units = "mm",
       device = grDevices::cairo_pdf)
ggsave("figure_overlap_filtered.png", fig, width = 180, height = 126, units = "mm", dpi = 400)
cat("\ncaption sentence:\n", CAP, "\n")
cat("wrote figure_overlap_filtered.pdf / .png\n")
cat("\nset sizes on the filtered set:\n")
print(setsz)
cat("\nintersections:\n")
print(inter %>% mutate(sets = sapply(ids, function(i)
  paste(SETS[as.integer(strsplit(i, ",")[[1]])], collapse = " + "))))
