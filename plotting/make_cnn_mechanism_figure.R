#!/usr/bin/env Rscript
# the first-layer filters, and where each of them acts
source("cnn_fig_common.R")
suppressPackageStartupMessages({library(ggseqlogo); library(grid)})
POS <- rd("cnn_posmap_occlusion.tsv")
FM  <- rd("filter_matches_readout.tsv")
GIA <- rbind(rd("gia_donor_test.tsv"), rd("gia_donor_test_constitutive.tsv"))
NEG <- COMP
e("[fig2] A <- cnn_posmap_occlusion.tsv (occlusion, NOT the learned gate, NOT attention)")
e("[fig2] A row labels <- filter_pwms_readout.meme; IC/n from filter_matches_readout.tsv")
e("[fig2] C <- gia_donor_test.tsv + gia_donor_test_constitutive.tsv")

read_meme <- function(path){
  L <- readLines(path); out <- list(); nm <- NULL; rows <- list()
  for (ln in L){
    if (startsWith(ln, "MOTIF")){ if(!is.null(nm)) out[[nm]] <- do.call(rbind, rows)
                                  nm <- strsplit(ln, " ")[[1]][2]; rows <- list() }
    else if (!is.null(nm) && grepl("^ ", ln) && nzchar(trimws(ln)))
      rows[[length(rows)+1]] <- as.numeric(strsplit(trimws(ln), " +")[[1]])
  }
  if (!is.null(nm)) out[[nm]] <- do.call(rbind, rows)
  lapply(out, function(m){ m <- t(m); rownames(m) <- c("A","C","G","T"); m })
}
PW <- read_meme("filter_pwms_readout.meme")

LEV <- POS %>% group_by(filter) %>% summarise(tot = sum(abs(contrib)), .groups = "drop") %>%
       arrange(tot) %>% pull(filter)
stopifnot(length(LEV) == 32, !anyDuplicated(LEV), all(LEV %in% names(PW)))

panel_A <- function(){
  d <- POS %>% mutate(filter = factor(filter, levels = LEV), x = bin_start + 50)
  stopifnot(identical(levels(d$filter), LEV))
  lim <- max(abs(d$contrib))
  colsum <- POS %>% group_by(bin_start) %>% summarise(s = sum(contrib), .groups = "drop")
  topb <- colsum %>% slice_max(s, n = 1); botb <- colsum %>% slice_min(s, n = 1)
  e("[fig2A] fill limits +/-", sprintf("%.4f", lim),
    "; strongest positive band [", topb$bin_start, ",", topb$bin_start+100, ") ",
    sprintf("%+.3f", topb$s), "; most negative [", botb$bin_start, ",", botb$bin_start+100,
    ") ", sprintf("%+.3f", botb$s))

  ic <- setNames(FM$info_content_bits, FM$filter); ns <- setNames(FM$n_sites, FM$filter)
  ylab <- sprintf("%s\n%.1f bits, n = %s", LEV, ic[LEV], trimws(nf(ns[LEV])))

  XL <- -3050; XR <- -2090
  p <- ggplot(d, aes(x, filter, fill = contrib)) +
    geom_tile(width = 100, height = .90) +
    geom_vline(xintercept = 0, linetype = "22", linewidth = .3, color = INK) +
    scale_fill_gradient2(low = NEG, mid = "#FCFCFC", high = REAL, midpoint = 0,
                         limits = c(-lim, lim), name = "contribution to\nreal vs competitor",
                         breaks = c(-.06, 0, .06)) +
    scale_x_continuous(breaks = seq(-2000, 500, 500), limits = c(XL, 520), expand = c(0, 0)) +
    scale_y_discrete(labels = ylab, expand = expansion(add = .55))
  nlogo <- 0
  for (i in seq_along(LEV)) {
    f <- LEV[i]
    stopifnot(identical(f, LEV[i]))
    g <- ggplotGrob(ggseqlogo(PW[[f]], method = "bits") + theme_void() +
                    theme(legend.position = "none", plot.margin = margin(0,0,0,0)))
    p <- p + annotation_custom(g, xmin = XL + 40, xmax = XR - 40,
                               ymin = i - .47, ymax = i + .47)
    nlogo <- nlogo + 1
  }
  e("[fig2A] ", nlogo, " of ", length(LEV),
    " rows carry a logo as their row label; label text is the filter ID plus information ",
    "content and site count, and no identity is asserted for any filter")
  p +
    annotate("segment", x = XR + 15, xend = XR + 15, y = .4, yend = length(LEV) + .6,
             colour = "#C9CDD3", linewidth = .3) +
    labs(title = "Positional contribution of first-layer filters (occlusion)",
         subtitle = paste0(
           "Each filter is silenced in one 100 nt band at a time and all ", nf(NPEAK),
           " test sites are rescored. Blue = that filter argues for the real 3' end in that band,\n",
           "orange = it argues for a competitor. Rows are ordered by summed |contribution| and each row is labelled by its own sequence logo.\n",
           "Summed over filters, [", topb$bin_start, ", ", topb$bin_start+100,
           ") is the strongest positive band (", sprintf("%+.3f", topb$s), ") and [",
           botb$bin_start, ", ", botb$bin_start+100, ") is the most negative (",
           sprintf("%+.3f", botb$s), ")."),
         x = "position relative to the 3' end (nt)", y = NULL, tag = "A") +
    th + theme(axis.text.y = element_text(size = BASE-3.2, colour = INK, lineheight = .95),
               plot.margin = margin(5.5, 5.5, 5.5, 20),
               panel.grid.major.y = element_blank(),
               legend.position = "right", legend.title = element_text(size = BASE-2),
               legend.key.height = unit(1.05, "lines"))
}

panel_C <- function(){
  keep <- c(acceptor_pyAG = "acceptor consensus", donor_AGGTAAGT = "5' donor AGGTAAGT",
            donor_perm_ctrl_TAGAGTAG = "donor, scrambled", PAS_AATAAA = "AATAAA",
            PAS_perm_ctrl_AAAATA = "AATAAA, scrambled")
  b <- GIA %>% filter(motif %in% names(keep),
                      background %in% c("real_competitor", "constitutive")) %>%
       group_by(background, motif) %>% slice_max(delta_logit, n = 1, with_ties = FALSE) %>%
       ungroup() %>%
       mutate(lab = factor(keep[motif], levels = rev(unname(keep))),
              bg = factor(ifelse(background == "constitutive",
                                 "constitutive background (real 3' ends)",
                                 "real competitor background"),
                          levels = c("real competitor background",
                                     "constitutive background (real 3' ends)")))
  for (i in seq_len(nrow(b)))
    e("[fig2C] ", as.character(b$bg[i]), " | ", as.character(b$lab[i]), " best delta-logit ",
      sprintf("%+.4f", b$delta_logit[i]), " at ", b$position[i], " nt")
  ggplot(b, aes(delta_logit, lab, fill = bg)) +
    geom_vline(xintercept = 0, linewidth = .3, color = INK) +
    geom_col(position = position_dodge(width = .72), width = .66) +
    geom_errorbar(aes(xmin = delta_logit - se, xmax = delta_logit + se),
                  position = position_dodge(width = .72), width = .22,
                  linewidth = .28, color = INK) +
    geom_text(aes(x = delta_logit + sign(delta_logit)*(se + 0.010),
                  label = ifelse(abs(delta_logit) < 5e-4, "0.000", sprintf("%+.3f", delta_logit)),
                  hjust = ifelse(delta_logit > 0, 0, 1)),
              position = position_dodge(width = .72), size = 1.8, color = INK) +
    scale_fill_manual(values = setNames(c(REAL, SPL), levels(b$bg)), name = NULL) +
    scale_x_continuous(expand = expansion(mult = c(.30, .30))) +
    labs(title = "Effect of motif implantation on the model score",
         subtitle = paste0(
           "Best change in logit over a 10 nt grid near the site plus a coarse sweep of the whole window, in two real-sequence backgrounds.\n",
           "0/32 filters meet the donor criteria; implanted GT is null and below its scramble; ISM at donor consensus p = 0.86. No positioned\n",
           "donor or U1 signal. The GT visible mid-logo in filter19 is not a donor match either: its best donor-reference r is ",
           sprintf("%.2f", FM$donor_r[FM$filter == "filter19"]), ", below the 0.68\n",
           "random-PWM null."),
         x = "change in logit on implantation", y = NULL, tag = "C") +
    th + theme(panel.grid.major.y = element_blank(),
               panel.grid.major.x = element_line(color = "#E8EAED", linewidth = .2),
               legend.position = "bottom", legend.margin = margin(t = -2))
}

A <- panel_A(); C <- panel_C()

CAP <- paste(strwrap(paste0(
  "Held-out test set: ", nf(NPEAK), " peaks in ", nf(NGENE), " genes, base rate ",
  sprintf("%.3f", BASER), "; window [-2,000, +500] nt in transcript orientation; chromosome holdout plus in_gene_holdout genes, no gene spans splits. ",
  "Panel A is causal occlusion on the unchanged model, not the learned positional gate (r = 0.09 with causal contribution) and not attention (inert); neither is shown. ",
  "Filters are read out as position weight matrices from the subsequences that activate them above 30% of their maximum over 4,000 held-out test sequences; ",
  "no reference match is shown, so no identity is asserted for any filter. ",
  U1_RECONCILIATION,
  "COMPOSITION-VERSUS-MOTIF TESTS ARE POSITION-MATCHED. An ordinary dinucleotide shuffle also displaces motifs: AATAAA sits about 190 nt closer to the site in real sequence than in its own shuffle, ",
  "which naive in-silico mutagenesis misread as a hexamer effect (ratio 1.141) that collapsed to null (1.009) once position was held fixed. ",
  "Attribution and in-silico mutagenesis remained at the composition-matched null throughout, so the conclusions rest on occlusion and the shuffle control, not on attribution."),
  width = 162), collapse = "\n")

HA <- 4.5; HC <- 1; H <- 17.0
fig <- (A / C) + plot_layout(heights = c(HA, HC)) +
  plot_annotation(
    title = "The first-layer filters and their positional contribution to within-gene discrimination",
    subtitle = paste0("Each filter's sequence logo and, by occlusion, where along the window it argues for the real 3' end or a competitor. Held-out test sequence, ",
                      nf(NPEAK), " peaks in ", nf(NGENE), " genes."),
    caption = CAP,
    theme = theme(plot.title = element_text(size = BASE+2, face = "bold", color = INK),
                  plot.subtitle = element_text(size = BASE, color = "#555555", lineheight = 1.2),
                  plot.caption = element_text(size = BASE-1.7, color = "#555555", hjust = 0, lineheight = 1.25),
                  plot.tag = element_text(size = BASE+2, face = "bold", color = INK)))
ggsave("figure_cnn_mechanism.pdf", fig, width = 10.4, height = H, device = grDevices::cairo_pdf)
ggsave("figure_cnn_mechanism.png", fig, width = 10.4, height = H, dpi = 300)
e("[fig2] figure height ", H, " in; panel A is allocated ", sprintf("%.1f", H*HA/(HA+HC)),
  " in, so 32 rows give about ", sprintf("%.0f", 300*H*HA/(HA+HC)/32),
  " px per row at 300 dpi (nominal; the panel's title and axis take part of that)")
e("[fig2] no identity is asserted for any filter; best_match is never read")
e("[out] figure_cnn_mechanism.pdf figure_cnn_mechanism.png")
