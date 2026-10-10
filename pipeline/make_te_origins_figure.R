#!/usr/bin/env Rscript
# are the cryptic cpa peaks suppressed poly(a) sites derived from transposable elements?

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
  library(readr);  library(scales);    library(forcats)
})
args  <- commandArgs(trailingOnly = TRUE)
D_TE  <- if (length(args) >= 1) args[1] else "tesrc"
D_GE  <- if (length(args) >= 2) args[2] else "geom"
D_SF  <- if (length(args) >= 3) args[3] else "subfam"
STEM  <- if (length(args) >= 4) args[4] else "figure_te_origins"

rd <- function(dir, f, req = TRUE) {
  p <- file.path(dir, f)
  if (!file.exists(p)) { if (req) stop("missing ", p); return(NULL) }
  read_tsv(p, show_col_types = FALSE, progress = FALSE)
}

INK <- "#141414"; MUT <- "#5A5A5A"; BASE <- 12
SENSE <- "#08519C"; ANTI <- "#C8C8C8"
STEP  <- c("#08519C", "#6BAED6", "#E8E8E8")

th <- theme_classic(base_size = BASE) +
  theme(text = element_text(colour = INK),
        axis.text  = element_text(colour = INK, size = BASE - 3),
        axis.title = element_text(size = BASE - 1),
        axis.line  = element_line(colour = INK, linewidth = 0.45),
        axis.ticks = element_line(colour = INK, linewidth = 0.45),
        plot.title = element_text(size = BASE, face = "bold", hjust = 0,
                                  margin = margin(b = 3)),
        plot.title.position = "plot",
        plot.subtitle = element_text(size = BASE - 4, colour = MUT,
                                     margin = margin(b = 5), lineheight = 1.15),
        legend.position = "bottom", legend.title = element_blank(),
        legend.text = element_text(size = BASE - 4),
        legend.margin = margin(t = -4), legend.key.size = unit(0.7, "lines"),
        panel.grid.major = element_line(linewidth = 0.2, colour = "grey94"),
        plot.margin = margin(6, 10, 6, 8))

S  <- rd(D_TE, "te_source.tsv")
HL <- rd(D_SF, "hexamer_landing.tsv",      req = FALSE)
LS <- rd(D_SF, "landing_by_subfamily.tsv", req = FALSE)

# a  derived?
# composition is near strand-symmetric, so a sense:antisense ratio far from 1
# cannot come from base content
# reads as a poly(a) signal one way only -- an element-borne signal
gb <- S |>
  filter(family != "non-TE", orientation %in% c("sense", "antisense")) |>
  select(family, orientation, arch_per_kb, kb) |>
  pivot_wider(names_from = orientation, values_from = c(arch_per_kb, kb)) |>
  filter(kb_sense >= 200) |>
  mutate(ratio = arch_per_kb_sense / pmax(arch_per_kb_antisense, 1e-9)) |>
  arrange(desc(ratio)) |> head(10) |> mutate(family = fct_inorder(family))
gbl <- gb |>
  select(family, sense = arch_per_kb_sense, antisense = arch_per_kb_antisense) |>
  pivot_longer(-family, names_to = "orientation", values_to = "v") |>
  mutate(orientation = factor(orientation, c("sense", "antisense")))
pa <- ggplot(gbl, aes(v, fct_rev(family), fill = orientation)) +
  geom_col(position = position_dodge(width = 0.72), width = 0.66) +
  geom_text(data = gb, aes(x = arch_per_kb_sense, y = fct_rev(family),
                           label = sprintf("%.1f\u00d7", ratio)),
            inherit.aes = FALSE, hjust = -0.2, size = 2.9,
            fontface = "bold", colour = INK) +
  scale_fill_manual(values = c(sense = SENSE, antisense = ANTI)) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.24))) +
  labs(title = "a   Derived from TEs? Orientation dependence",
       subtitle = paste0("poly(A) architectures per kb of element, sense vs ",
                         "antisense.\nthe sense:antisense ratio is the ",
                         "element-borne signature; ERV families lead"),
       x = "architectures per kb", y = NULL) + th

pb <- NULL
if (!is.null(HL) && nrow(HL) && "subs_from_pas" %in% names(HL)) {
  gh <- HL |> filter(is.finite(subs_from_pas)) |>
    mutate(s = pmin(as.integer(subs_from_pas), 6L)) |>
    count(s) |> arrange(s) |>
    mutate(frac = n / sum(n),
           near = s <= 1,
           s = factor(s, levels = 0:6))
  near_pct <- 100 * sum(gh$frac[as.integer(as.character(gh$s)) <= 1])
  pb <- ggplot(gh, aes(s, frac, fill = near)) +
    geom_col(width = 0.78) +
    geom_text(aes(label = percent(frac, accuracy = 1)),
              vjust = -0.5, size = 2.6, colour = MUT) +
    scale_fill_manual(values = c(`TRUE` = SENSE, `FALSE` = ANTI),
                      guide = "none") +
    scale_x_discrete(drop = FALSE) +
    scale_y_continuous(labels = percent, expand = expansion(mult = c(0, 0.18))) +
    labs(title = "b   The element brought the signal",
         subtitle = sprintf(paste0("substitutions from a poly(A) signal in the ",
                                   "ancestral consensus, at the\nposition each ",
                                   "hexamer occupies -- %.0f%% sit at or one ",
                                   "step from a signal"), near_pct),
         x = "substitutions from AATAAA / ATTAAA in the consensus",
         y = "share of hexamers in elements") + th
}

pc <- NULL
if (!is.null(HL) && nrow(HL) &&
    all(c("el_div", "subs_from_pas") %in% names(HL))) {
  brk <- c(0, 5, 10, 15, 20, 25, 30, Inf)
  lb  <- c("0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30+")
  gc <- HL |> filter(is.finite(subs_from_pas), is.finite(el_div)) |>
    mutate(bin = cut(el_div, brk, lb, right = FALSE)) |>
    group_by(bin) |>
    summarise(near = mean(subs_from_pas <= 1), n = n(), .groups = "drop") |>
    filter(!is.na(bin), n >= 30)
  pc <- ggplot(gc, aes(bin, near, group = 1)) +
    geom_line(colour = SENSE, linewidth = 0.7) +
    geom_point(colour = SENSE, size = 2.4) +
    geom_text(aes(label = comma(n)), vjust = -1.1, size = 2.4, colour = MUT) +
    scale_y_continuous(labels = percent, limits = c(0, NA),
                       expand = expansion(mult = c(0, 0.18))) +
    labs(title = "c   A molecular clock",
         subtitle = paste0("fraction of hexamers where the consensus is a ",
                          "signal or one step away,\nagainst copy divergence ",
                          "(RepeatMasker milliDiv). n above each point"),
         x = "% divergence from subfamily consensus",
         y = "near-signal fraction") + th
}

pd_ <- NULL
if (!is.null(LS) && nrow(LS)) {
  gd <- LS |> filter(n >= 20) |> slice_max(frac_one_or_less, n = 14) |>
    mutate(subfamily = fct_reorder(subfamily, frac_one_or_less),
           signal = frac_exact,
           one    = pmax(frac_one_or_less - frac_exact, 0),
           rest   = pmax(1 - frac_one_or_less, 0)) |>
    select(subfamily, family, n, signal, one, rest) |>
    pivot_longer(c(signal, one, rest), names_to = "kind", values_to = "v") |>
    mutate(kind = factor(kind, c("signal", "one", "rest"),
                         labels = c("consensus already a signal",
                                    "one substitution away", "two or more")))
  lab <- LS |> filter(n >= 20) |> slice_max(frac_one_or_less, n = 14)
  pd_ <- ggplot(gd, aes(v, fct_rev(subfamily), fill = kind)) +
    geom_col(width = 0.72) +
    geom_text(data = lab, aes(x = 1.015, y = fct_rev(subfamily),
                              label = paste0(family, "  n=", comma(n))),
              inherit.aes = FALSE, hjust = 0, size = 2.4, colour = MUT) +
    scale_fill_manual(values = STEP) +
    scale_x_continuous(labels = percent, limits = c(0, 1.5),
                       breaks = c(0, 0.5, 1), expand = expansion(0)) +
    labs(title = "d   Native carriers vs mutational targets",
         subtitle = paste0("subfamilies whose observed hexamers most often sit ",
                          "at or one step\nfrom a signal in their own consensus"),
         x = "share of observed hexamers", y = NULL) +
    th + theme(legend.position = "bottom", legend.direction = "vertical",
               axis.text.y = element_text(size = BASE - 5))
}

blank <- function(p) if (is.null(p)) plot_spacer() else p
fig <- (blank(pa) | blank(pb)) / (blank(pc) | blank(pd_)) +
  plot_annotation(
    caption = paste0(
      "Transposable elements are a source of cryptic poly(A) architecture, not ",
      "a general explanation for the peak set. A sense-oriented ERV subset ",
      "carries element-borne signals; of the hexamers falling in elements, most ",
      "sit where the ancestral consensus was already a signal or one ",
      "substitution away, and provenance concentrates in younger copies. Alu, ",
      "the most abundant intronic element, is depleted and contributes almost ",
      "none."),
    theme = theme(plot.caption = element_text(size = BASE - 4, colour = MUT,
                                              hjust = 0, margin = margin(t = 6),
                                              lineheight = 1.2)))

ggsave(paste0(STEM, ".pdf"), fig, width = 9.5, height = 9.0, device = cairo_pdf)
ggsave(paste0(STEM, ".png"), fig, width = 9.5, height = 9.0, dpi = 400)
cat("[out] ", STEM, ".pdf / .png\n", sep = "")