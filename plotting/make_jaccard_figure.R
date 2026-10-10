#!/usr/bin/env Rscript
# panels from hiconf_sets.py (+ xlip_profile.py)

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr)
  library(readr); library(patchwork); library(scales)
})

args    <- commandArgs(trailingOnly = TRUE)
indir   <- if (length(args) >= 1) args[1] else "hiconf_out"
outstem <- if (length(args) >= 2) args[2] else "fig_jaccard"
labfile <- if (length(args) >= 3 && args[3] != "-") args[3] else NA
dropset <- if (length(args) >= 4) strsplit(args[4], ",")[[1]] else "canonical"
xlipdir <- if (length(args) >= 5) args[5] else "xlip_out"

rd <- function(f, req = TRUE, dir = indir) {
  p <- file.path(dir, f)
  if (!file.exists(p)) { if (req) stop("missing ", p); return(NULL) }
  read_tsv(p, show_col_types = FALSE, progress = FALSE)
}

J <- rd("jaccard_matrix.tsv")
H <- rd("hexamer.tsv")
D <- rd("distances.tsv", req = FALSE)

H$label <- gsub("[\r\n]+", " ", H$label)
H$label[H$name == "model_fp"]  <- "internal competitors"
H$label[H$name == "canonical"] <- "canonical CPA"
if (!is.na(labfile) && file.exists(labfile)) {
  ov <- read_tsv(labfile, col_names = c("name", "label"),
                 show_col_types = FALSE)
  H$label[match(ov$name, H$name)] <- ov$label
}

if (length(dropset) && any(nzchar(dropset))) {
  keep <- !H$name %in% dropset
  message("dropping: ", paste(H$name[!keep], collapse = ", "))
  H <- H[keep, ]
  J <- J[J$set_a %in% H$name & J$set_b %in% H$name, ]
  if (!is.null(D)) D <- D[D$set_a %in% H$name & D$set_b %in% H$name, ]
}

R <- rd("restricted.tsv", req = FALSE)
if (!is.null(R)) R <- R[R$set_a %in% H$name & R$set_b %in% H$name, ]

lab <- setNames(H$label, H$name)
siz <- setNames(H$n, H$name)
ord <- H$name[order(-H$n)]
pretty <- function(x) paste0(lab[x], "\n", comma(siz[x]))

theme_set(theme_classic(base_size = 9) + theme(
  axis.text     = element_text(size = 6),
  plot.title    = element_text(size = 9, face = "bold"),
  plot.subtitle = element_text(size = 6.5, colour = "grey35"),
  legend.title  = element_text(size = 6.5),
  legend.text   = element_text(size = 6),
  strip.background = element_blank(),
  strip.text    = element_text(size = 6.2, face = "bold")))

# a: distance ecdf
pa <- NULL
if (!is.null(D)) {
  D <- D %>%
    filter(is.finite(dist)) %>%
    mutate(a = factor(set_a, levels = ord), b = factor(set_b, levels = ord),
           dist = pmin(pmax(dist, 1), 1e7))
  pa <- ggplot(D, aes(dist, colour = b)) +
    stat_ecdf(geom = "step", linewidth = .5, pad = FALSE) +
    geom_vline(xintercept = 50, linetype = 3, linewidth = .35,
               colour = "grey40") +
    facet_wrap(~a, nrow = 2, labeller = as_labeller(function(x) lab[x])) +
    scale_x_log10(labels = label_number(scale_cut = cut_short_scale()),
                  limits = c(1, 1e7)) +
    scale_y_continuous(labels = percent, limits = c(0, 1)) +
    scale_colour_brewer(palette = "Dark2", name = "nearest site in",
                        labels = function(x) lab[x]) +
    labs(title = "a  Distance to the nearest site in each other set",
         subtitle = paste("ECDF per source set (facet). Dotted line = 50 nt,",
                          "the matching tolerance. A curve that rises only",
                          "far right means the sets are unrelated."),
         x = "distance (nt, log scale)", y = "% of sites within") +
    theme(legend.position = "right",
          panel.grid.major.y = element_line(linewidth = .2, colour = "grey92"))
}

# b: containment
Jb <- J %>% mutate(a = factor(set_a, levels = ord),
                   b = factor(set_b, levels = ord),
                   v = containment_a_in_b)
Jb$v[as.character(Jb$set_a) == as.character(Jb$set_b)] <- NA
vmax <- max(Jb$v, na.rm = TRUE)
pb <- ggplot(Jb, aes(b, a, fill = v)) +
  geom_tile(colour = "white", linewidth = .7) +
  geom_text(aes(label = ifelse(is.na(v), "\u2014", sprintf("%.3f", v)),
                colour = !is.na(v) & v > vmax * .55),
            size = 2.0, show.legend = FALSE) +
  scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = "grey15"),
                      na.value = "grey70") +
  scale_fill_gradient(low = "#F2F5EC", high = "#33633A", name = "containment",
                      na.value = "grey96") +
  scale_x_discrete(labels = pretty) +
  scale_y_discrete(labels = function(x) lab[x], limits = rev(ord)) +
  labs(title = "b  Containment, row set found in column set",
       subtitle = paste("|A \u2229 B| / |A| within 50 nt.",
                        "\nAsymmetric: read across a row."),
       x = NULL, y = NULL) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1),
        axis.line = element_blank(), axis.ticks = element_blank(),
        legend.key.height = unit(9, "mm"), legend.key.width = unit(3, "mm"))

# c: pas
Hp <- H %>%
  mutate(fold = pct_obs / pmax(pct_rand, .01),
         name = factor(name, levels = rev(ord))) %>%
  select(name, pct_obs, pct_rand, fold) %>%
  pivot_longer(c(pct_obs, pct_rand), names_to = "which", values_to = "pct") %>%
  mutate(which = factor(which, levels = c("pct_obs", "pct_rand"),
                        labels = c("sites", "+5 kb displaced")))
flab <- Hp %>% group_by(name) %>%
  summarise(x = max(pct), fold = first(fold), .groups = "drop")

pc <- ggplot(Hp, aes(pct, name, fill = which)) +
  geom_col(position = position_dodge(width = .72), width = .64) +
  geom_text(data = flab, aes(x = x + 3, y = name,
                             label = sprintf("%.1f\u00d7", fold)),
            inherit.aes = FALSE, hjust = 0, size = 2.4, colour = "grey25") +
  scale_fill_manual(values = c("sites" = "#B2182B",
                               "+5 kb displaced" = "#CCCCCC"), name = NULL) +
  scale_y_discrete(labels = function(x) lab[x]) +
  scale_x_continuous(limits = c(0, 118), breaks = c(0, 25, 50, 75, 100),
                     expand = expansion(c(0, 0))) +
  labs(title = "c  Canonical PAS at site centres",
       subtitle = "AATAAA or ATTAAA in the 60 nt upstream; fold over displaced",
       x = "% of sites", y = NULL) +
  theme(axis.line.x = element_line(linewidth = .4),
        axis.ticks.x = element_line(linewidth = .4),
        legend.position = "top")

# d: restricted containment
pd_ <- NULL
if (!is.null(R) && nrow(R)) {
  has_fold <- "fold_over_chance" %in% names(R)
  Rb <- R %>% mutate(a = factor(set_a, levels = ord),
                     b = factor(set_b, levels = ord),
                     v = if (has_fold) fold_over_chance else containment_a_in_b,
                     lb = if (has_fold)
                            sprintf("%.2f\u00d7\n%.1f%%", fold_over_chance,
                                    100 * containment_a_in_b)
                          else sprintf("%.3f", containment_a_in_b))
  cap <- if (has_fold) max(2, quantile(Rb$v, .95, na.rm = TRUE)) else
         max(Rb$v, na.rm = TRUE)
  Rb$vc <- pmin(Rb$v, cap)
  mid <- if (has_fold) 1 else cap / 2
  pd_ <- ggplot(Rb, aes(b, a, fill = vc)) +
    geom_tile(colour = "white", linewidth = .7) +
    geom_text(aes(label = lb, colour = vc > (mid + cap) / 2),
              size = 1.9, lineheight = .9, show.legend = FALSE) +
    scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = "grey15")) +
    { if (has_fold)
        scale_fill_gradient2(low = "#4575B4", mid = "grey96", high = "#6A51A3",
                             midpoint = 1, limits = c(0, cap),
                             oob = scales::squish,
                             name = "fold over\nchance\n(capped)")
      else scale_fill_gradient(low = "#F1EEF6", high = "#6A51A3",
                               name = "containment") } +
    scale_x_discrete(labels = function(x) lab[x]) +
    scale_y_discrete(labels = function(x) lab[x], limits = rev(ord)) +
    labs(title = "d  Within the analysis genes, vs within-gene chance",
         subtitle = paste("Both sets restricted to the genes carrying our",
                          "peaks.\nCell: fold over a within-gene shuffle,",
                          "then raw containment.\n1.00x = no more than",
                          "expected from site density alone."),
         x = NULL, y = NULL) +
    theme(axis.text.x = element_text(angle = 35, hjust = 1),
          axis.line = element_blank(), axis.ticks = element_blank(),
          legend.key.height = unit(9, "mm"), legend.key.width = unit(3, "mm"))
}

pe <- NULL
if (xlipdir != "-" && file.exists(file.path(xlipdir, "xlip_enrichment.tsv"))) {
  X <- rd("xlip_enrichment.tsv", req = FALSE, dir = xlipdir)
  inflat <- X %>%
    filter(group %in% c("canonical", "canonical shuffled")) %>%
    group_by(group) %>% summarise(mi = mean(mean_input), .groups = "drop")
  iratio <- if (nrow(inflat) == 2)
    inflat$mi[inflat$group == "canonical"] /
    inflat$mi[inflat$group == "canonical shuffled"] else NA

  X <- X %>%
    filter(group %in% c("canonical", "competitors"),
           is.finite(fold_over_own_shuffle)) %>%
    mutate(
      grp = factor(group, levels = c("canonical", "competitors"),
                   labels = c("canonical CPA  (positive control)",
                              "internal competitors")),
      cls = ifelse(grepl("^U1", factor), "U1  (no positive control here)",
            ifelse(grepl("negative", role), "internal negative", role)),
      factor = factor(factor, levels = rev(c(
        "CFIm25", "Fip1", "CstF64", "SF3B1", "hnRNPC", "U1A", "U1C"))))

  sub_e <- paste0(
    "Raw XLIP signal vs a within-gene shuffle of the same sites (So 2019, ",
    "GSE135140). Dashed line = chance.\nInput not used as a denominator: at ",
    "canonical sites it is ",
    if (is.finite(iratio)) sprintf("%.1f\u00d7", iratio) else "~2.7\u00d7",
    " its own shuffle, so it carries 3'-end structure and\ndividing by it ",
    "inverts the positive control. U1 rows are greyed: U1 is a 5' splice ",
    "site factor, so the canonical\ngroup cannot serve as a positive control ",
    "for that channel.")

  pe <- ggplot(X, aes(fold_over_own_shuffle, factor, fill = cls)) +
    geom_col(width = .68) +
    geom_vline(xintercept = 1, linetype = 2, linewidth = .4,
               colour = "grey30") +
    geom_text(aes(label = sprintf("%.2f\u00d7", fold_over_own_shuffle)),
              hjust = -0.15, size = 2.1, colour = "grey25") +
    facet_wrap(~grp) +
    scale_fill_manual(values = c(
      "CPA" = "#B2182B", "internal negative" = "#BBBBBB",
      "U1  (no positive control here)" = "#E4E4E4"), name = NULL) +
    scale_x_continuous(expand = expansion(c(0, .22))) +
    labs(title = "e  U1 and CPA factor occupancy at these sites",
         subtitle = sub_e,
         x = "fold over within-gene shuffle", y = NULL) +
    theme(legend.position = "top",
          panel.grid.major.x = element_line(linewidth = .2, colour = "grey93"))
} else if (xlipdir != "-") {
  message("no xlip_enrichment.tsv in ", xlipdir, "; panel e omitted")
}

mid_row <- if (is.null(pd_)) pb else (pb | pd_)
parts   <- Filter(Negate(is.null), list(pa, mid_row, pc, pe))
hts     <- c(if (!is.null(pa)) 1.10, 1.25, 0.80, if (!is.null(pe)) 0.85)
fig     <- Reduce(`/`, parts) + plot_layout(heights = hts)

n <- length(ord)
w <- 150 + 26 * n
h <- (if (is.null(pa)) 60 else 140) + 16 * n +
     (if (is.null(pd_)) 0 else 20) + (if (is.null(pe)) 0 else 55)
ok <- tryCatch({
  ggsave(paste0(outstem, ".pdf"), fig, width = w, height = h, units = "mm",
         device = grDevices::cairo_pdf); TRUE
}, error = function(e) FALSE)
if (!ok) ggsave(paste0(outstem, ".pdf"), fig, width = w, height = h, units = "mm")
ggsave(paste0(outstem, ".png"), fig, width = w, height = h, units = "mm", dpi = 400)
message("wrote ", outstem, ".pdf and .png")

if (!is.null(D)) {
  cat("\nmedian nearest-neighbour distance (nt), row set to column set:\n")
  D %>% group_by(set_a, set_b) %>%
    summarise(median_nt = round(median(dist)), .groups = "drop") %>%
    pivot_wider(names_from = set_b, values_from = median_nt) %>%
    as.data.frame() %>% print(row.names = FALSE)
}