#!/usr/bin/env Rscript
# interpretable-cnn figure
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(readr); library(tidyr)
  library(ggpubr); library(scales)
})
have_logo <- requireNamespace("ggseqlogo", quietly = TRUE)
if (have_logo) suppressPackageStartupMessages(library(ggseqlogo))

PFX <- if (length(commandArgs(trailingOnly = TRUE)) >= 1) commandArgs(trailingOnly = TRUE)[1] else "cnn_interp"
P <- function(s) paste0(PFX, "_", s)
rd <- function(p) if (file.exists(p)) suppressMessages(read_tsv(p, show_col_types = FALSE)) else NULL
getm <- function(df, k) { if (is.null(df) || !"metric" %in% names(df)) return(NA); v <- df$value[df$metric == k]; if (length(v)) as.numeric(v[1]) else NA }

col_can <- "#7B5EA7"; col_com <- "#0072B2"; col_grey <- "#8A8F98"; col_pos <- "#2E7D32"; col_neg <- "#C62828"
theme_pub <- function(base = 11) theme_pubr(base_size = base) +
  theme(plot.title = element_text(face = "bold", size = base + 1, margin = margin(b = 1)),
        plot.subtitle = element_text(size = base - 2, color = "grey35", margin = margin(b = 3)),
        legend.position = "none", plot.margin = margin(6, 8, 6, 6))
ph <- function(t, why = "no data") ggplot() + annotate("text", 0, 0, label = paste0(t, "\n(", why, ")"), size = 3, color = col_grey) +
  labs(title = t) + theme_void() + theme(plot.title = element_text(face = "bold"))

# reshape a long pwm table (id,pos,base,prob) into a list of 4xw matrices for ggseqlogo
pwm_matrices <- function(df, idcol) {
  ids <- unique(df[[idcol]]); out <- list()
  for (id in ids) {
    d <- df[df[[idcol]] == id, ]
    m <- d %>% select(pos, base, prob) %>% pivot_wider(names_from = pos, values_from = prob) %>%
      as.data.frame()
    rownames(m) <- m$base; m$base <- NULL
    m <- as.matrix(m[c("A", "C", "G", "T"), , drop = FALSE])
    ic <- apply(m, 2, function(col) { col <- pmax(col, 1e-6); sum(col * log2(col / 0.25)) })
    out[[as.character(id)]] <- sweep(m, 2, ic, "*")
  }
  out
}

fallback_logo <- function(df, ttl) {
  df$base <- factor(df$base, levels = c("A", "C", "G", "T"))
  ggplot(df, aes(pos, prob, fill = base)) + geom_col(width = .9) +
    scale_fill_manual(values = c(A = "#2E7D32", C = "#1565C0", G = "#F9A825", T = "#C62828")) +
    labs(title = ttl, x = NULL, y = NULL) + theme_pub(9) + theme(legend.position = "right")
}

metrics <- rd(P("metrics.tsv"))
fmeta  <- rd(P("filter_meta.tsv"))
fpwm   <- rd(P("filters_pwm.tsv"))
attr   <- rd(P("attr_examples.tsv"))
modisco <- rd(P("modisco_motifs.tsv"))

# a: performance
if (!is.null(metrics)) {
  vals <- tibble(metric = c("AUROC", "AUPRC", "within-gene P@1"),
                 value = c(getm(metrics, "auroc"), getm(metrics, "auprc"), getm(metrics, "p1")))
  pos_rate <- getm(metrics, "pos_rate")
  vals$metric <- factor(vals$metric, levels = vals$metric)
  pA <- ggplot(vals, aes(metric, value, fill = metric)) +
    geom_col(width = .62) +
    geom_text(aes(label = sprintf("%.3f", value)), vjust = -0.4, size = 3.4) +
    { if (!is.na(pos_rate)) geom_hline(yintercept = pos_rate, linetype = 2, color = col_grey) } +
    scale_fill_manual(values = c(col_can, col_com, "#3E8E7E"), guide = "none") +
    scale_y_continuous(limits = c(0, 1.05), expand = expansion(c(0, 0))) +
    labs(title = "CNN performance (held-out chromosomes)",
         subtitle = if (!is.na(pos_rate)) sprintf("dashed = positive rate %.2f", pos_rate) else "canonical vs competitor",
         x = NULL, y = NULL) + theme_pub()
} else pA <- ph("performance")

# roc and pr: sequence cnn vs splice reranker (model_compare_*)
mc_pfx <- "model_compare"
roc_df <- rd(paste0(mc_pfx, "_roc.tsv")); pr_df <- rd(paste0(mc_pfx, "_pr.tsv"))
mc_met <- rd(paste0(mc_pfx, "_metrics.tsv"))
mc_cols <- c("Splice reranker (features)" = "#7B5EA7", "Sequence CNN" = "#0072B2")
lab_with <- function(df, kind) {
  if (is.null(mc_met)) return(setNames(unique(df$model), unique(df$model)))
  v <- setNames(character(nrow(mc_met)), mc_met$model)
  for (i in seq_len(nrow(mc_met))) {
    if (kind == "roc") v[mc_met$model[i]] <- sprintf("%s  (AUROC %.3f)", mc_met$model[i], mc_met$auroc[i])
    else               v[mc_met$model[i]] <- sprintf("%s  (AUPRC %.3f)", mc_met$model[i], mc_met$auprc[i])
  }
  v
}
if (!is.null(roc_df)) {
  labs_roc <- lab_with(roc_df, "roc"); roc_df$lab <- labs_roc[roc_df$model]
  pROC <- ggplot(roc_df, aes(fpr, tpr, color = lab)) +
    geom_abline(slope = 1, intercept = 0, linetype = 2, color = col_grey) +
    geom_line(linewidth = 1) +
    scale_color_manual(values = setNames(mc_cols[names(labs_roc)], labs_roc), name = NULL) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(title = "Sequence vs features: ROC", subtitle = "canonical vs competitor, leak-free by chromosome",
         x = "false positive rate", y = "true positive rate") +
    theme_pub() + theme(legend.position = c(.98, .02), legend.justification = c(1, 0),
                        legend.text = element_text(size = 8),
                        legend.background = element_rect(fill = "white", color = NA))
} else pROC <- ph("ROC (run roc_pr_compare.py)")
if (!is.null(pr_df)) {
  base_rate <- if (!is.null(mc_met)) mc_met$base_rate[1] else NA
  labs_pr <- lab_with(pr_df, "pr"); pr_df$lab <- labs_pr[pr_df$model]
  pPR <- ggplot(pr_df, aes(recall, precision, color = lab)) +
    { if (!is.na(base_rate)) geom_hline(yintercept = base_rate, linetype = 2, color = col_grey) } +
    geom_line(linewidth = 1) +
    scale_color_manual(values = setNames(mc_cols[names(labs_pr)], labs_pr), name = NULL) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(title = "Sequence vs features: precision-recall",
         subtitle = if (!is.na(base_rate)) sprintf("dashed = canonical base rate %.3f", base_rate) else "canonical vs competitor",
         x = "recall", y = "precision") +
    theme_pub() + theme(legend.position = c(.98, .98), legend.justification = c(1, 1),
                        legend.text = element_text(size = 8),
                        legend.background = element_rect(fill = "white", color = NA))
} else pPR <- ph("PR (run roc_pr_compare.py)")

# b: top filter logos
if (!is.null(fmeta) && !is.null(fpwm)) {
  live <- fmeta %>% filter(n_hits >= 20) %>% arrange(desc(abs(class_pref)))
  topf <- head(live$filter, 6)
  mats <- pwm_matrices(fpwm %>% filter(filter %in% topf), "filter")
  titles <- sapply(topf, function(f) {
    r <- fmeta[fmeta$filter == f, ]
    sprintf("f%d  %s  (pref %+.2f)", f, ifelse(r$match_score >= 0.45, r$match, "novel"), r$class_pref)
  })
  if (have_logo && length(mats) > 0) {
    mats <- mats[as.character(topf)]; names(mats) <- titles
    pB <- ggseqlogo(mats, method = "custom", ncol = 2) +
      labs(title = "First-layer filters learn splice / polyA motifs",
           subtitle = "top filters by |class preference|; height = information (bits)", y = "bits") +
      theme_pub(9) + theme(strip.text = element_text(face = "bold", size = 8))
  } else {
    sub <- fpwm %>% filter(filter %in% topf)
    sub$filter <- factor(sub$filter, levels = topf, labels = titles)
    pB <- fallback_logo(sub, "First-layer filters (install ggseqlogo for logos)") +
      facet_wrap(~filter, ncol = 2)
  }
} else pB <- ph("filter logos")

if (!is.null(fmeta)) {
  live <- fmeta %>% filter(n_hits >= 20) %>% arrange(class_pref) %>%
    mutate(lab = ifelse(match_score >= 0.45, paste0("f", filter, " ", match), paste0("f", filter)),
           side = ifelse(class_pref >= 0, "marks true end", "marks competitor"))
  live$lab <- factor(live$lab, levels = live$lab)
  pC <- ggplot(live, aes(lab, class_pref, fill = side)) +
    geom_col(width = .74) + coord_flip() +
    geom_hline(yintercept = 0, color = "grey30", linewidth = .3) +
    scale_fill_manual(values = c("marks true end" = col_can, "marks competitor" = col_com), name = NULL) +
    labs(title = "Which filters mark true ends vs competitors",
         subtitle = "class preference = mean filter activation, canonical \u2212 competitor",
         x = NULL, y = "activation difference") +
    theme_pub(10) + theme(axis.text.y = element_text(size = 7), legend.position = "top")
} else pC <- ph("class preference")

if (!is.null(modisco) && nrow(modisco) > 0) {
  modisco <- modisco %>% mutate(mid = paste0(metacluster, ":", motif))
  keep <- modisco %>% distinct(mid, n_seqlets) %>% arrange(desc(n_seqlets)) %>% head(6) %>% pull(mid)
  md <- modisco %>% filter(mid %in% keep)
  mats <- pwm_matrices(md %>% rename(pwmid = mid) %>% select(pwmid, pos, base, prob), "pwmid")
  titles <- sapply(keep, function(k) {
    r <- modisco[modisco$mid == k, ][1, ]
    sprintf("%s  %s (n=%d)", k, ifelse(r$match_score >= 0.45, r$match, "novel"), r$n_seqlets)
  })
  if (have_logo && length(mats) > 0) {
    mats <- mats[keep]; names(mats) <- titles
    pD <- ggseqlogo(mats, method = "custom", ncol = 2) +
      labs(title = "TF-MoDISco motifs (de novo from attributions)",
           subtitle = "recurring attributed patterns across the test set", y = "bits") +
      theme_pub(9) + theme(strip.text = element_text(face = "bold", size = 8))
  } else {
    md$mid <- factor(md$mid, levels = keep, labels = titles)
    pD <- fallback_logo(md %>% select(pos, base, prob, mid), "TF-MoDISco motifs") + facet_wrap(~mid, ncol = 2)
  }
} else pD <- ph("TF-MoDISco motifs", "run --modisco (needs modisco-lite)")

if (!is.null(attr)) {
  gcol <- if ("gene" %in% names(attr)) as.character(attr$gene) else paste0("site ", attr$example + 1)
  attr <- attr %>% mutate(lab = paste0(ifelse(is_canonical %in% c(TRUE, "True", "TRUE"), "true end: ", "competitor: "), gcol))
  attr$lab <- factor(attr$lab, levels = unique(attr$lab))
  span <- 60; ac <- attr %>% filter(abs(pos) <= span)
  pE <- ggplot(ac, aes(pos, contrib, fill = contrib > 0)) +
    geom_col(width = 1) +
    geom_hline(yintercept = 0, color = "grey40", linewidth = .3) +
    facet_wrap(~lab, ncol = 1, scales = "free_y") +
    scale_fill_manual(values = c("TRUE" = col_pos, "FALSE" = col_neg), guide = "none") +
    labs(title = "Per-base attribution on example sites",
         subtitle = sprintf("gradient \u00d7 input; green supports canonical, red opposes (\u00b1%d nt of center)", span),
         x = "position relative to site (nt)", y = "contribution") +
    theme_pub(10) + theme(strip.text = element_text(face = "bold", size = 8))
} else pE <- ph("attribution tracks")

posw <- rd(paste0(PFX, "_pos_weights.tsv"))
if (!is.null(posw)) {
  lab_f <- posw %>% distinct(filter, match, class_pref) %>%
    mutate(flab = ifelse(!is.na(match) & match != "" & match != "none" & match != "dead" & match != "rare",
                         sprintf("f%d %s", filter, match), sprintf("f%d", filter))) %>%
    arrange(class_pref)
  posw <- posw %>% left_join(lab_f %>% select(filter, flab), by = "filter")
  posw$flab <- factor(posw$flab, levels = lab_f$flab)
  lim <- max(abs(posw$weight), na.rm = TRUE)
  pPOS <- ggplot(posw, aes(position_nt, flab, fill = weight)) +
    geom_tile() +
    geom_vline(xintercept = 0, linetype = 3, color = "grey30", linewidth = .3) +
    scale_fill_gradient2(low = col_com, mid = "white", high = col_can, midpoint = 0,
                         limits = c(-lim, lim), name = "toward\ncanonical") +
    labs(title = "Learned motif \u00d7 position map",
         subtitle = "positional-branch weight: filter firing in a position bin pushes toward canonical (purple) or competitor (blue)",
         x = "position relative to site (nt)", y = NULL) +
    theme_pub(10) + theme(axis.text.y = element_text(size = 7), legend.position = "right")
} else pPOS <- NULL

dc <- rd(paste0(PFX, "_dinuc_control.tsv"))
if (!is.null(dc)) {
  dd <- tibble(kind = c("real held-out", "dinucleotide-shuffled"),
               auroc = c(getm(dc, "auroc_real_holdout"), getm(dc, "auroc_dinuc_shuffled")))
  dd$kind <- factor(dd$kind, levels = dd$kind)
  pDC <- ggplot(dd, aes(kind, auroc, fill = kind)) +
    geom_col(width = .58) +
    geom_hline(yintercept = 0.5, linetype = 2, color = col_grey) +
    geom_text(aes(label = sprintf("%.3f", auroc)), vjust = -0.4, size = 3.4) +
    scale_fill_manual(values = c("real held-out" = col_can, "dinucleotide-shuffled" = col_grey), guide = "none") +
    scale_y_continuous(limits = c(0, 1.02), expand = expansion(c(0, 0))) +
    labs(title = "Positioned-motif control",
         subtitle = "AUROC collapses toward chance (0.5) when motif placement is destroyed",
         x = NULL, y = "AUROC") + theme_pub()
} else pDC <- NULL

top <- ggarrange(pROC, pPR, pA, ncol = 3, labels = c("A", "B", "C"), font.label = list(size = 14, face = "bold"))
logos <- ggarrange(pB, pD, ncol = 2, labels = c("D", "E"), font.label = list(size = 14, face = "bold"))
mid <- ggarrange(pC, ncol = 1, labels = "F", font.label = list(size = 14, face = "bold"))
blocks <- list(top, logos, mid, pE); block_h <- c(1, 1.25, 1.05, 1.15); block_lab <- c("", "", "", "G")
if (!is.null(pPOS)) { blocks <- c(blocks, list(pPOS)); block_h <- c(block_h, 1.1); block_lab <- c(block_lab, "H") }
if (!is.null(pDC))  { blocks <- c(blocks, list(pDC));  block_h <- c(block_h, 0.9); block_lab <- c(block_lab, "I") }
fig <- ggarrange(plotlist = blocks, ncol = 1, heights = block_h, labels = block_lab,
                 font.label = list(size = 14, face = "bold"))
fig <- annotate_figure(fig, top = text_grob("Interpretable CNN: sequence determinants of canonical vs competitor CPA sites",
                                            face = "bold", size = 13))
Htot <- 17 + (if (!is.null(pPOS)) 4 else 0) + (if (!is.null(pDC)) 3 else 0)
ggsave("figure_cnn_interpret.pdf", fig, width = 14, height = Htot, device = cairo_pdf, limitsize = FALSE)
ggsave("figure_cnn_interpret.png", fig, width = 14, height = Htot, dpi = 200, limitsize = FALSE)
cat("wrote figure_cnn_interpret.pdf / .png",
    if (!have_logo) "  [note: install ggseqlogo for proper sequence logos]" else "", "\n")