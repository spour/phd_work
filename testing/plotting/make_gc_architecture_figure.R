#!/usr/bin/env Rscript
# what a real 3' end looks like in base composition
source("cnn_fig_common.R")
PROF <- rd("gc_metaprofile.tsv")
DEC  <- "#7B5EA7"
CLS  <- c(real = "real 3' end", competitor = "intragenic competitor", decoy = "intronic decoy")
PAL  <- setNames(c(REAL, COMP, DEC), unname(CLS))
NCLS <- PROF %>% group_by(class) %>% summarise(n = max(n), .groups = "drop")
nof  <- function(k) nf(NCLS$n[NCLS$class == k])

band <- function(lo, hi, k)
  PROF %>% filter(class == k, offset >= lo, offset < hi) %>% pull(gc) %>% mean()
KEYS  <- c("real", "competitor", "decoy")
BASE_UP <- vapply(KEYS, function(k) band(-2000, -1000, k), 0)
MINROW  <- lapply(KEYS, function(k) PROF %>% filter(class == k) %>% slice_min(gc, n = 1))
names(MINROW) <- KEYS
MING  <- vapply(KEYS, function(k) MINROW[[k]]$gc, 0)
MINO  <- vapply(KEYS, function(k) MINROW[[k]]$offset, 0)
DEPTH <- BASE_UP - MING
for (k in KEYS)
  e("[gcA] ", k, ": far-upstream [-2000,-1000) ", sprintf("%.4f", BASE_UP[k]),
    ", minimum ", sprintf("%.4f", MING[k]), " at ", MINO[k], " nt, dip depth ",
    sprintf("%.4f", DEPTH[k]))
for (b in list(c(-100, 0), c(200, 300)))
  e("[gcA] [", b[1], ",", b[2], ")  real ", sprintf("%.4f", band(b[1], b[2], "real")),
    "  competitor ", sprintf("%.4f", band(b[1], b[2], "competitor")),
    "  decoy ", sprintf("%.4f", band(b[1], b[2], "decoy")))

Pd <- PROF %>% mutate(cl = factor(CLS[class], levels = unname(CLS)))
YMIN <- min(Pd$gc - Pd$se); YMAX <- max(Pd$gc + Pd$se)
RC   <- PROF %>% filter(class == "real")
b_lo <- mean(RC$gc[RC$offset >= -2000 & RC$offset < -1000])
t_lo <- mean(RC$gc[RC$offset >= -150  & RC$offset < 0])
ARX  <- -75
TRX0 <- -150; TRX1 <- 300; TLX <- 15
stopifnot(TRX0 >= -150, TLX >= -150, TLX >= 0)
e("[gcA] depth annotation on A: baseline [-2000,-1000) ", sprintf("%.4f", b_lo),
  ", trough [-150,0) ", sprintf("%.4f", t_lo), ", drop ", sprintf("%.4f", b_lo - t_lo),
  "; arrow drawn at ", ARX, " nt, inside the trough window at the site")
e("[gcA] trough level line drawn from ", TRX0, " to ", TRX1,
  " nt and trough text anchored at ", TLX, " nt, both at the site (>= -150), not mid-profile;",
  " baseline level line and text remain at the far left over [-2000,-1000)")
pA <- ggplot(Pd, aes(offset, gc, colour = cl, fill = cl)) +
  annotate("rect", xmin = -2000, xmax = 0, ymin = -Inf, ymax = Inf,
           fill = "#F4F5F7", colour = NA) +
  annotate("rect", xmin = -2000, xmax = -1000, ymin = -Inf, ymax = Inf,
           fill = "#DFE6F0", colour = NA) +
  annotate("rect", xmin = -150, xmax = 0, ymin = -Inf, ymax = Inf,
           fill = "#DFE6F0", colour = NA) +
  geom_vline(xintercept = 0, linetype = "22", linewidth = .3, colour = INK) +
  geom_ribbon(aes(ymin = gc - se, ymax = gc + se), alpha = .25, colour = NA) +
  geom_line(linewidth = .55) +
  scale_colour_manual(values = PAL, name = NULL) +
  scale_fill_manual(values = PAL, guide = "none") +
  scale_x_continuous(breaks = seq(-2000, 500, 500), expand = c(0, 0)) +
  annotate("segment", x = -2000, xend = ARX, y = b_lo, yend = b_lo,
           linetype = "31", linewidth = .32, colour = INK) +
  annotate("segment", x = TRX0, xend = TRX1, y = t_lo, yend = t_lo,
           linetype = "31", linewidth = .32, colour = INK) +
  annotate("segment", x = ARX, xend = ARX, y = t_lo, yend = b_lo, linewidth = .5,
           colour = INK, arrow = arrow(ends = "both", length = unit(3.4, "pt"),
                                       type = "closed")) +
  annotate("text", x = ARX, y = YMAX + .030, vjust = 1, hjust = .5, size = 2.15,
           colour = INK,
           label = "depth = baseline - trough (per site, own frame)") +
  annotate("text", x = -1960, y = b_lo, vjust = -0.8, hjust = 0, size = 1.9, colour = INK,
           label = sprintf("baseline  [-2,000, -1,000)   %.3f", b_lo)) +
  annotate("text", x = TLX, y = t_lo, vjust = 1.7, hjust = 0, size = 1.9, colour = INK,
           label = sprintf("trough  [-150, 0)   %.3f", t_lo)) +
  scale_y_continuous(breaks = seq(.30, .44, .02)) +
  coord_cartesian(ylim = c(YMIN, YMAX + .030)) +
  labs(title = "A  GC content across the cleavage site",
       subtitle = paste0(
         "Mean G+C fraction per 50 nt bin, +/-1 SE, in transcript orientation; the shaded half is upstream of the site. Real ends and competitors both\n",
         "reach an AT-rich minimum at ", MINO["real"], " nt, but the DEPTH differs: measured against each class's own far-upstream level ([-2,000, -1,000)), the\n",
         "minimum falls ", sprintf("%.3f", DEPTH["real"]), " at real ends, ", sprintf("%.3f", DEPTH["competitor"]),
         " at competitors and only ", sprintf("%.3f", DEPTH["decoy"]), " at intronic decoys, which barely dip at all. At [-100, 0) GC is ",
         sprintf("%.3f", band(-100, 0, "real")), ",\n", sprintf("%.3f", band(-100, 0, "competitor")), " and ",
         sprintf("%.3f", band(-100, 0, "decoy")), "; by [+200, +300) the three have converged to ",
         sprintf("%.3f", band(200, 300, "real")), ", ", sprintf("%.3f", band(200, 300, "competitor")), " and ",
         sprintf("%.3f", band(200, 300, "decoy")), ". Note the sign reversal away from\n",
         "the site: far upstream the real end sits in GC-RICHER sequence than its competitors (",
         sprintf("%.3f", band(-2000, -1000, "real")), " against ", sprintf("%.3f", band(-2000, -1000, "competitor")),
         "), and the curves cross near -400 nt. What marks a real\n3' end is the depth of the dip relative to its own surroundings, not absolute AT-richness."),
       x = "position relative to the 3' end (nt)", y = "GC fraction") +
  th + theme(legend.position = c(.015, .04), legend.justification = c(0, 0),
             legend.background = element_rect(fill = "#FFFFFFCC", colour = NA),
             legend.text = element_text(size = BASE - 1.6))

DEP <- rd("gc_dip_depth.tsv")
DS  <- rd("gc_dip_stats.tsv")
RVD <- rd("gc_dip_rawvsdepth.tsv")
P   <- DS %>% filter(primary == 1)
stopifnot(nrow(P) == 1, P$baseline == "[-2000,-1000)", P$trough == "mean [-150,0)")
DCOL <- "depth_[-2000,-1000)_mean [-150,0)"
BCOL <- "baseline_[-2000,-1000)"; TCOL <- "trough_mean [-150,0)"
D <- DEP %>% transmute(gene, class, depth = .data[[DCOL]]) %>% filter(is.finite(depth))
e("[gcB] gc_dip_depth.tsv ", nf(nrow(D)), " sites; depth column '", DCOL, "'")

PAIR <- D %>% filter(class != "decoy") %>% group_by(gene) %>%
  filter(any(class == "real"), any(class == "competitor")) %>%
  summarise(dr = mean(depth[class == "real"]),
            dk = mean(depth[class == "competitor"]),
            wr = { k <- depth[class == "competitor"]
                   (sum(k < dr[1]) + .5 * sum(k == dr[1])) / length(k) },
            p1 = { k <- depth[class == "competitor"]; m <- max(k)
                   if (dr[1] > m) 1 else if (dr[1] == m) 1 / (1 + sum(k == m)) else 0 },
            nk = sum(class == "competitor"), .groups = "drop") %>%
  mutate(delta = dr - dk)
boot_v <- function(v, nb = 2000) {
  set.seed(42)
  quantile(vapply(seq_len(nb), function(i) mean(v[sample.int(length(v), replace = TRUE)]), 0),
           c(.025, .975), names = FALSE)
}
stopifnot(nrow(PAIR) == P$n_genes)
CHK <- c(winrate = mean(PAIR$wr), median = median(PAIR$delta), mean = mean(PAIR$delta),
         dz = mean(PAIR$delta) / sd(PAIR$delta))
REF <- c(winrate = P$wg_winrate, median = P$wg_median_delta, mean = P$wg_mean_delta,
         dz = P$wg_dz)
NTIE <- D %>% filter(class != "decoy") %>% group_by(gene) %>%
  filter(any(class == "real"), any(class == "competitor")) %>%
  summarise(t = sum(depth[class == "competitor"] == mean(depth[class == "real"])),
            n = sum(class == "competitor"), .groups = "drop")
TOL <- sum(NTIE$t / NTIE$n) / (2 * nrow(NTIE))
e("[gcB] ", sum(NTIE$t), " rounding-induced ties across ", nf(sum(NTIE$n)),
  " comparisons, bounding the win-rate shift at ", sprintf("%.2e", TOL))
BP  <- boot_v(PAIR$p1)
e("[gcB2] within-gene P@1 (real end holds the single deepest dip in its gene) ",
  sprintf("%.4f", mean(PAIR$p1)), " [", sprintf("%.4f", BP[1]), ", ", sprintf("%.4f", BP[2]),
  "], against a null of 1 / (1 + mean competitors per gene) = ",
  sprintf("%.4f", mean(1 / (1 + PAIR$nk))))
stopifnot(abs(CHK["winrate"] - REF["winrate"]) <= TOL + 1e-9,
          all(abs(CHK[c("median", "mean", "dz")] - REF[c("median", "mean", "dz")]) < 1e-5))
e("[gcB] within-gene re-derivation matches gc_dip_stats.tsv to 1e-5: win-rate ",
  sprintf("%.4f", CHK["winrate"]), ", median ", sprintf("%+.5f", CHK["median"]), ", dz ",
  sprintf("%.3f", CHK["dz"]))
e("[gcB] within-gene panel uses DEPTH win-rate ", sprintf("%.3f", P$wg_winrate),
  " (raw window GC over the same window is ", sprintf("%.3f", RVD$raw_wg_winrate[1]),
  " and is NOT plotted, only stated in the caption)")

D2  <- D %>% mutate(cl = factor(CLS[class], levels = rev(unname(CLS))))
MU  <- data.frame(cl = factor(unname(CLS), levels = rev(unname(CLS))),
                  m  = c(P$mean_depth_real, P$mean_depth_competitor, P$mean_depth_decoy))
XR2 <- quantile(D2$depth, c(.004, .996))
pB1 <- ggplot(D2, aes(depth, cl, fill = cl)) +
  geom_vline(xintercept = 0, linetype = "22", linewidth = .3, colour = INK) +
  geom_violin(colour = NA, alpha = .42, width = 1.0, scale = "width") +
  geom_boxplot(width = .13, outlier.shape = NA, fill = "white", colour = INK, linewidth = .26) +
  geom_point(data = MU, aes(m, cl), inherit.aes = FALSE, shape = 23, size = 1.5,
             fill = "white", colour = INK, stroke = .4) +
  geom_text(data = MU, aes(m, cl, label = sprintf("mean %.3f", m)), inherit.aes = FALSE,
            vjust = -1.9, size = 1.95, colour = INK) +
  scale_fill_manual(values = PAL, guide = "none") +
  coord_cartesian(xlim = XR2) +
  labs(title = "B1  Depth by class (pooled)",
       subtitle = paste0(
         "All ", nof("real"), " real, ", nof("competitor"), " competitor and ", nof("decoy"),
         " decoy sites POOLED across\ngenes, so this comparison still carries intron location; the intronic decoy\n",
         "is the null for it, and barely dips. Competitors dip more than intron and\n",
         "less than real ends. Real against competitor, Cohen's d = ",
         sprintf("%.2f", P$d_real_vs_comp), "."),
       x = "dip depth (baseline - trough, GC fraction)", y = NULL) +
  th + theme(panel.grid.major.y = element_blank(),
             panel.grid.major.x = element_line(colour = "#E8EAED", linewidth = .2),
             axis.text.y = element_text(size = BASE - 1.2))

XR3 <- quantile(PAIR$delta, c(.005, .995))
LAB3 <- sprintf("win-rate %.3f [%.3f, %.3f]\nP@1 %.3f [%.3f, %.3f]\nchance P@1 %.3f\nmedian %+.3f    dz %.3f",
                P$wg_winrate, P$wg_ci_lo, P$wg_ci_hi, mean(PAIR$p1), BP[1], BP[2],
                mean(1 / (1 + PAIR$nk)), P$wg_median_delta, P$wg_dz)
pB2 <- ggplot(PAIR, aes(delta)) +
  geom_histogram(bins = 70, fill = REAL, colour = NA, alpha = .55) +
  geom_vline(xintercept = 0, linetype = "22", linewidth = .35, colour = INK) +
  geom_vline(xintercept = P$wg_median_delta, linewidth = .45, colour = REAL) +
  annotate("text", x = XR3[2], y = Inf, hjust = 1, vjust = 1.4, size = 2.0, colour = INK,
           label = LAB3) +
  scale_y_continuous(expand = expansion(mult = c(0, .16))) +
  coord_cartesian(xlim = XR3) +
  labs(title = "B2  Depth within gene (paired)",
       subtitle = paste0(
         "Each real end compared ONLY with competitors in its OWN gene, over ",
         nf(P$n_genes), "\ngenes, which holds gene, expression context and locus fixed. This is the\n",
         "confound-free test that B1 cannot be. Positive means the real end dips\n",
         "deeper. Win-rate is P(real deeper) and P@1 is the fraction of genes whose\n",
         "real end holds the single deepest dip; both ties-aware, bootstrapped over genes."),
       x = "depth(real 3' end) - mean depth(own competitors)", y = "genes") +
  th

auroc <- function(sc, pos) {
  r <- rank(sc); n1 <- sum(pos); n0 <- sum(!pos)
  (sum(r[pos]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
auprc <- function(sc, pos) {
  o <- order(sc, decreasing = TRUE); y <- pos[o]; ss <- sc[o]
  k <- which(c(diff(ss) != 0, TRUE))
  prec <- cumsum(y)[k] / k; rec <- cumsum(y)[k] / sum(pos)
  sum(diff(c(0, rec)) * prec)
}
roc_df <- function(sc, pos, lab) {
  o <- order(sc, decreasing = TRUE); y <- pos[o]; ss <- sc[o]
  k <- which(c(diff(ss) != 0, TRUE))
  tpr <- cumsum(y)[k] / sum(pos); fpr <- (k - cumsum(y)[k]) / sum(!pos)
  keep <- unique(c(1, round(seq(1, length(k), length.out = min(length(k), 1500))), length(k)))
  data.frame(fpr = c(0, fpr[keep]), tpr = c(0, tpr[keep]), curve = lab)
}
RD  <- DEP %>% transmute(class, depth = .data[[DCOL]], rawgc = .data[[TCOL]]) %>%
  filter(is.finite(depth), is.finite(rawgc))
RCk <- RD %>% filter(class != "decoy"); pk <- RCk$class == "real"
RDk <- RD %>% filter(class != "competitor"); pd <- RDk$class == "real"
A_DEP <- auroc(RCk$depth, pk); A_RAW <- auroc(-RCk$rawgc, pk); A_DEC <- auroc(RDk$depth, pd)
stopifnot(abs(A_DEP - P$auroc_real_vs_comp)  < 1e-4,
          abs(A_RAW - RVD$raw_auroc[1])      < 1e-4,
          abs(A_DEC - P$auroc_real_vs_decoy) < 1e-4)
PR_DEP <- auprc(RCk$depth, pk); PR_RAW <- auprc(-RCk$rawgc, pk)
BR <- mean(pk)
e("[gcB3] ROC real vs competitor: depth AUROC ", sprintf("%.3f", A_DEP),
  " (AUPRC ", sprintf("%.3f", PR_DEP), "), raw trough GC AUROC ", sprintf("%.3f", A_RAW),
  " (AUPRC ", sprintf("%.3f", PR_RAW), "), base rate ", sprintf("%.3f", BR),
  "; real vs decoy depth AUROC ", sprintf("%.3f", A_DEC),
  " computed and asserted but NOT plotted, B3 shows depth against raw trough GC only")
e("[gcB3] all three asserted against gc_dip_stats.tsv / gc_dip_rawvsdepth.tsv to 1e-4")
RL <- c(sprintf("dip depth (own frame)   AUROC %.3f", A_DEP),
        sprintf("raw trough GC   AUROC %.3f", A_RAW))
ROC <- bind_rows(roc_df(RCk$depth, pk, RL[1]), roc_df(-RCk$rawgc, pk, RL[2])) %>%
  mutate(curve = factor(curve, levels = RL))
pB3 <- ggplot(ROC, aes(fpr, tpr, colour = curve)) +
  geom_abline(slope = 1, intercept = 0, linetype = "22", linewidth = .3, colour = GREY) +
  geom_line(linewidth = .55) +
  scale_colour_manual(values = setNames(c(REAL, GREY), RL), name = NULL) +
  scale_x_continuous(expand = c(0, 0), breaks = seq(0, 1, .25)) +
  scale_y_continuous(expand = c(0, 0), breaks = seq(0, 1, .25)) +
  labs(title = "B3  Per-site discrimination by dip depth",
       subtitle = paste0(
         "Real against competitor, one point per site. Depth sits above raw\n",
         "trough GC, so the own-frame measurement helps pooled separation.\n",
         "Both stay near the diagonal: a population effect, not a per-site\n",
         "classifier. AUPRC ", sprintf("%.3f", PR_DEP), " for depth, ",
         sprintf("%.3f", PR_RAW), " for raw GC, base rate ", sprintf("%.3f", BR), "."),
       x = "false positive rate", y = "true positive rate") +
  th + theme(legend.position = c(.97, .04), legend.justification = c(1, 0),
             legend.background = element_rect(fill = "#FFFFFFCC", colour = NA),
             legend.text = element_text(size = BASE - 1.8),
             legend.key.width = unit(1.3, "lines"),
             panel.grid.major.y = element_line(colour = "#E8EAED", linewidth = .2))

pB <- (pB1 | pB2 | pB3) + plot_layout(widths = c(1.04, 1, 1.0))

CAP <- paste(strwrap(paste0(
  "Sites from cpa_peaks_final.parquet: real = is_canonical (n = ", nof("real"),
  "), intragenic competitor = the analysis-set model false positives (n = ", nof("competitor"),
  "), and intronic decoys (n = ", nof("decoy"), ", seed-42 subsample restricted to analysis-set genes so all three classes share one gene universe). ",
  "Window [-2,000, +500] nt around the cleavage site in transcript orientation, strand-correct; GC is counted per 50 nt bin over G and C, ignoring N. No model is used anywhere in this figure and no motif is claimed: this is base composition. ",
  "THIS IS A POPULATION DIFFERENCE, NOT A PER-SITE CLASSIFIER: depth separates individual real ends from individual competitors at ", sprintf("%.3f", A_DEP), " pooled AUROC (AUPRC ", sprintf("%.3f", PR_DEP), " against a base rate of ", sprintf("%.3f", BR), "), and the sequence CNN's own signal is diffuse rather than local, with in-silico mutagenesis and attribution both at the composition-matched null. ",
  "The decoy arm bounds how much of the pooled gap is intron location rather than 3'-end identity; B2 removes that confound outright by comparing each real end only with competitors in its own gene, while B3 shows how modestly the same statistic separates individual sites. ",
  "Depth is stable across window choices: over three trough definitions ([-150, 0), [-100, 0), and the minimum 50 nt window in [-250, 0)) crossed with two baselines ([-2,000, -1,000) and [-1,500, -750)), the within-gene win-rate spans ",
  sprintf("%.3f", min(DS$wg_winrate)), " to ", sprintf("%.3f", max(DS$wg_winrate)), " and pooled Cohen's d spans ",
  sprintf("%.2f", min(DS$d_real_vs_comp)), " to ", sprintf("%.2f", max(DS$d_real_vs_comp)), ". ",
  "One honest qualification: measuring depth in the site's own frame helps POOLED separation (", sprintf("%.3f", RVD$depth_auroc[1]), " AUROC against ",
  sprintf("%.3f", RVD$raw_auroc[1]), " for raw GC over the same window) but is marginally WORSE within gene (", sprintf("%.3f", RVD$depth_wg_winrate[1]),
  " against ", sprintf("%.3f", RVD$raw_wg_winrate[1]), "), because paired sites in one gene already share most of their baseline, so subtracting it there mostly adds noise. Both are reported and neither dominates. ",
  "What the panels together describe is an architectural difference in composition, an AT-rich approach that runs deepest into a real 3' end, not a signal that identifies any single site."),
  width = 158), collapse = "\n")

fig <- (pA / pB) + plot_layout(heights = c(1, .95)) +
  plot_annotation(
    title = "The used 3' end sits in a deeper local AT-dip than its intragenic competitors",
    subtitle = paste0("A, GC content across the site for all three classes, with dip depth defined on the real-end curve. B1, depth by class pooled across\n",
                      "genes, against a random-intronic null. B2, the same statistic paired within gene, which is the confound-free test. B3, per-site\n",
                      "discrimination by dip depth against raw window GC. ",
                      nf(P$n_genes), " genes, ", nof("real"), " real ends, ", nof("competitor"),
                      " competitors, ", nof("decoy"), " intronic decoys."),
    caption = CAP,
    theme = theme(plot.title = element_text(size = BASE+2, face = "bold", colour = INK),
                  plot.subtitle = element_text(size = BASE, colour = "#555555", lineheight = 1.2),
                  plot.caption = element_text(size = BASE-1.7, colour = "#555555", hjust = 0, lineheight = 1.25),
                  plot.tag = element_text(size = BASE+2, face = "bold", colour = INK)))
ggsave("figure_gc_architecture.pdf", fig, width = 10.2, height = 8.9, device = grDevices::cairo_pdf)
ggsave("figure_gc_architecture.png", fig, width = 10.2, height = 8.9, dpi = 300)
e("[out] figure_gc_architecture.pdf figure_gc_architecture.png")
