# figures 3.3 to 3.8 and 3.10, regrouped from the panels pub/'s scripts build
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr)})
setwd("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3")
source("R/_harvest.R")
dir.create("out", showWarnings = FALSE)
INK <- "#1A1A1A"; GTX <- "#5A6068"
TT <- theme(plot.title = element_text(face = "bold", size = 11, colour = INK),
            plot.subtitle = element_blank(), plot.caption = element_blank())
save2 <- function(fig, name, w, h, dpi = 300) {
  ggsave(file.path("out", paste0(name, ".pdf")), fig, width = w, height = h, device = cairo_pdf, limitsize = FALSE)
  ggsave(file.path("out", paste0(name, ".png")), fig, width = w, height = h, dpi = dpi, bg = "white", limitsize = FALSE)
  cat("[out]", name, "\n")
}
# 3.3 donor and acceptor environment
E <- harvest("make_splicing_environment.R")
drop_leader <- function(p) { p$layers <- Filter(function(l) !(inherits(l$geom, "GeomSegment") &&
    is.data.frame(l$data) && isTRUE(all(l$data$xend == -330))), p$layers); p }
pA0 <- E$panel_A();
a <- edit_text(drop_leader(pA0), drop = c("-50 nt bin", "upstream of the poly")) +
  labs(title = "A   Strong 5′ splice-site density") + TT
b <- edit_text(E$panel_B(), drop = c("^mean ", "no difference", "median -12")) +
  labs(title = "B   Within-gene difference in U1 site density") + TT
REAL <- "#08519C"; COMP <- "#E8A33D"; DEC <- "#7B5EA7"
AP <- read.delim("../acceptor_position_scan.tsv") %>%
  mutate(class = factor(class, levels = c("real", "competitor", "decoy"),
                        labels = c("real 3′ end", "competitor", "intronic decoy")))
stopifnot(nrow(AP) == 4847 + 43547 + 30000)
CLS <- c("real 3′ end" = REAL, competitor = COMP, "intronic decoy" = DEC)
TH3 <- theme_classic(base_size = 7.5) + theme(axis.text = element_text(colour = GTX), legend.position = "none",
        panel.grid.major.y = element_line(colour = "#E8EAED", linewidth = .2))
c3 <- ggplot(AP, aes(max_me3, colour = class)) +
  stat_ecdf(linewidth = 0.6) + scale_colour_manual(values = CLS) +
  coord_cartesian(xlim = c(4, 16)) +
  labs(x = "strongest upstream 3′ splice-site score (MaxEntScan score3)", y = "cumulative fraction of sites",
       title = "C   Strongest upstream 3′ splice site") + TH3 + TT
d3 <- ggplot(AP %>% filter(max_me3 >= 6.5), aes(position, colour = class)) +
  geom_density(aes(y = after_stat(density) * 1000), linewidth = 0.6, adjust = 0.6, key_glyph = "path") + scale_colour_manual(values = CLS, name = NULL) +
  scale_x_continuous(limits = c(-2000, -126), breaks = c(-2000, -1500, -1000, -500)) +
  labs(x = "position of the strongest upstream 3′ splice site (nt)", y = "density (per kb)",
       title = "D   Position of that site") + TH3 + TT +
  theme(legend.position = "bottom")
save2((a / b / (c3 | d3)) + plot_layout(heights = c(1, 0.8, 0.9)), "fig3_03_splice_environment", 6.8, 9.6)

# 3.4 donor scores
pc <- E$panel_C()
pc[[1]] <- edit_text(pc[[1]], drop = "r = 0.69") + labs(title = "A   U1 duplex free energy against MaxEntScan 5′ score") + TT
if (length(pc) > 1) pc[[2]] <- pc[[2]] + labs(title = NULL, subtitle = NULL, caption = NULL)
d <- edit_text(E$panel_D(), drop = "<mapped txt>") + labs(title = "B   Score distributions and thresholds") + TT
# c: maxentscan sensitivity at annotated sites against background calls in intronic
MC <- read.delim("data/maxent_calibration.tsv") %>% mutate(site = factor(site, levels = c("acceptor", "donor"),
        labels = c("3\u2032 splice site (score3)", "5\u2032 splice site (score5)")))
U1C <- read.delim("data/u1_calibration.tsv")
USED <- MC %>% filter((grepl("^3", site) & threshold == 6.5) | (grepl("^5", site) & threshold == 8))
SC <- c("3\u2032 splice site (score3)" = "#E8A33D", "5\u2032 splice site (score5)" = "#08519C")
c4 <- ggplot(MC %>% filter(per_kb_background > 0), aes(per_kb_background, sensitivity_annotated, colour = site)) +
  geom_path(linewidth = 0.6) + geom_point(size = 1) +
  geom_point(data = USED, size = 3, shape = 21, fill = "white", stroke = 1) +
  geom_point(data = U1C, aes(background_per_kb, sensitivity_annotated_donors), inherit.aes = FALSE,
             shape = 24, size = 2.6, fill = "#7B5EA7", colour = INK, stroke = 0.4) +
  geom_text(data = U1C, aes(background_per_kb, sensitivity_annotated_donors, label = paste0("U1 \u2265 ", threshold)), inherit.aes = FALSE,
            hjust = 1.2, vjust = 0.3, size = 2.4, colour = "#7B5EA7") +
  scale_x_log10(breaks = c(0.01, 0.1, 1, 10, 100), labels = c("0.01", "0.1", "1", "10", "100")) +
  scale_colour_manual(values = SC, name = NULL) +
  labs(x = "background calls per kb of intronic sequence", y = "fraction of annotated sites passing",
       title = "C   Calibration of the thresholds") +
  theme_classic(base_size = 7.5) + TT + theme(axis.text = element_text(colour = GTX), legend.position = c(0.3, 0.85),
        legend.key.size = unit(3, "mm"), legend.text = element_text(size = 7))
save2((pc / ((d | c4) + plot_layout(widths = c(1.9, 1)))) + plot_layout(heights = c(1.1, 0.9)), "fig3_04_donor_scores", 8.2, 8.0)

# 3.5, 3.6 splice model
S <- harvest("make_splice_model_final.R")
s5a <- edit_text(S$pA, drop = "Within-gene evaluation",
                 relab = c("Sequence window" = "25 bins of 100 nt",
                           "U1 duplex density" = "U1 site density per bin",
                           "Maximum MaxEnt" = "strongest upstream 3′ splice-site score",
                           "Ridge \\(L2\\)" = "ridge logistic regression"))
s5b <- edit_text(S$pB, drop = c("-0.23 at", "^\\+0.30$"), relab = c("MaxEnt 3'ss" = "acceptor"))
save2((s5a + labs(title = "A   Features") + TT) / (s5b + labs(title = "B   Fitted coefficients by position") + TT) +
      plot_layout(heights = c(1.15, 1)), "fig3_05_splice_model", 7.4, 6.6)
s6c <- edit_text(S$pC, drop = "<mapped lab>", relab = c("random" = "chance"))
s6d <- edit_text(S$pD, drop = c("<mapped sprintf", "Win-rate 0.699"))
save2((s6c + labs(title = "A   Within-gene recall at rank k") + TT) | (s6d + labs(title = "B   Gene-centred scores") + TT),
      "fig3_06_splice_model_performance", 7.8, 3.6)

Ar <- harvest("make_figure_architecture.R")
R  <- harvest("make_figure_rescue.R")
arch <- edit_text(Ar$p, drop = "one output bin sees")
save2((R$pA + labs(title = "A   Sequence read by the baseline and the extended models") + TT) /
      (arch + labs(title = "B   Network architecture") + TT + theme(plot.background = element_rect(fill = "white", colour = NA))) +
      plot_layout(heights = c(1, 0.72)), "fig3_07_architecture", 9.4, 4.8)

L <- harvest("make_figure_loco.R")
a1 <- edit_text(L$pA1, relab = c("^Sequence CNN" = "Sequence CNN", "^Splice model" = "Splice model"))
a2 <- edit_text(L$pA2, drop = "^AUPRC", relab = c("base rate" = "base rate"))
b8 <- edit_text(L$pB, drop = "recall@1", relab = c("^Splice model" = "Splice model"))
d8 <- edit_text(L$pD, drop = "<mapped lab>")
top <- ((a1 + labs(title = "A   Pooled ROC and precision-recall") + TT) | (a2 + TT) |
        (b8 + labs(title = "B   Within-gene recall at rank k") + TT)) + plot_layout(widths = c(0.78, 0.78, 1.44))
bot <- d8 + labs(title = "C   Annotated end ranked first, by gene length and number of candidates") + TT
save2((top / bot) + plot_layout(heights = c(1, 0.95)), "fig3_08_cnn_vs_splice", 10.6, 7.4)

I <- harvest("make_figure_cnn_interpretability.R")
pa <- I$pA; pa[[2]] <- pa[[2]] + labs(title = "A   First-layer filters and their positional contribution") + TT
pb <- I$pB
pb[[2]] <- edit_text(pb[[2]], drop = c("untouched competitor", "<mapped txt>")) +
  labs(title = "B   Splice-site implantation into competitor windows") + TT
save2(pa / pb + plot_layout(heights = c(4.3, 1.35)), "fig3_12_interpretation", 9.8, 13.6, dpi = 220)
