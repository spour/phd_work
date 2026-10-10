#!/usr/bin/env Rscript
# rscript 06_overlap_figure.r <indir> <outstem> [tolerance] [drop1

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(tibble)
  library(readr); library(patchwork); library(scales)
})

args    <- commandArgs(trailingOnly = TRUE)
indir   <- if (length(args) >= 1) args[1] else "."
outstem <- if (length(args) >= 2) args[2] else "fig_site_overlap"
TOL     <- if (length(args) >= 3) as.numeric(args[3]) else NA
DROP    <- if (length(args) >= 4) strsplit(args[4], ",")[[1]] else character(0)

rd <- function(f) {
  p <- file.path(indir, f)
  if (!file.exists(p)) { message("missing: ", f); return(NULL) }
  x <- suppressWarnings(read_tsv(p, show_col_types = FALSE, progress = FALSE))
  if (nrow(x) == 0) { message("empty: ", f); return(NULL) }
  x
}

theme_set(theme_classic(base_size = 9) + theme(
  axis.text       = element_text(colour = "grey20", size = 7.5),
  plot.title      = element_text(size = 9.5, face = "bold"),
  plot.subtitle   = element_text(size = 6.8, colour = "grey35"),
  legend.key.size = unit(3.5, "mm"),
  legend.text     = element_text(size = 7),
  legend.title    = element_text(size = 7.5)
))
panels <- list()

cm <- rd("overlap_competitor.tsv")
if (!is.null(cm)) {
  if (length(DROP)) cm <- cm[!cm$dataset %in% DROP, ]
  if (is.na(TOL)) { tt <- sort(unique(cm$tolerance)); TOL <- tt[ceiling(length(tt)/2)] }
  c1 <- cm %>% filter(tolerance == TOL) %>%
    mutate(fold_lab = ifelse(is.finite(fold_enrichment),
                             sprintf("%.0fx", fold_enrichment), "inf"),
           lab = sprintf("%s / %s   %s   p = %s",
                         comma(n_external_in_competitors), comma(n_external),
                         fold_lab, format.pval(p_empirical, digits = 2)),
           dataset = reorder(dataset, pct_of_dataset))
  panels$a <- ggplot(c1, aes(pct_of_dataset, dataset)) +
    geom_col(width = .58, fill = "#B4462F") +
    geom_errorbar(aes(xmin = ci_lo, xmax = ci_hi), width = .16,
                  linewidth = .4, colour = "grey25") +
    geom_text(aes(x = ci_hi, label = lab), hjust = -0.09, size = 2.4,
              colour = "grey25") +
    scale_x_continuous(expand = expansion(c(0, .6))) +
    labs(title = "a  Fraction of each dataset that falls in the competitor set",
         subtitle = paste0("tolerance ", TOL, " nt. Error bars Wilson 95% CI. ",
                           "Fold and p are against the same sites displaced ",
                           "10-100 kb. ", comma(c1$n_competitors[1]),
                           " competitors."),
         x = "% of that dataset's sites landing on a competitor", y = NULL)
}

pw <- rd("overlap_external_pairwise.tsv")
if (!is.null(pw)) {
  if (length(DROP)) pw <- pw[!pw$row %in% DROP & !pw$col %in% DROP, ]
  lvl <- sort(unique(c(pw$row, pw$col)))
  mk <- function(d, val, ttl, sub) {
    d <- d %>% mutate(row = factor(row, levels = rev(lvl)),
                      col = factor(col, levels = lvl),
                      v = .data[[val]])
    ggplot(d, aes(col, row, fill = v)) +
      geom_tile(colour = "white", linewidth = .6) +
      geom_text(aes(label = sprintf("%.2f", v), colour = v > .5),
                size = 2.2, show.legend = FALSE) +
      scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = "grey15")) +
      scale_fill_gradient(low = "#F5F2ED", high = "#1F4E5F", name = NULL,
                          limits = c(0, 1)) +
      labs(title = ttl, subtitle = sub, x = NULL, y = NULL) +
      theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 6.5),
            axis.text.y = element_text(size = 6.5),
            axis.line = element_blank(), axis.ticks = element_blank())
  }
  b1 <- mk(pw, "frac_row_in_col", "b  Agreement, row set into column set",
           paste0("within ", TOL, " nt"))
  b2 <- if ("frac_exact_match" %in% names(pw))
    mk(pw, "frac_exact_match", "  ", "exact coordinate match") else NULL
  panels$b <- if (is.null(b2)) b1 else (b1 | b2)
}

# c: upset
pp <- rd("overlap_per_peak.tsv")
if (!is.null(pp) && !is.na(TOL)) {
  pp$is_canonical <- as.logical(pp$is_canonical)
  hc <- grep(paste0("^hit_.*_", TOL, "$"), names(pp), value = TRUE)
  sets <- sub(paste0("^hit_(.*)_", TOL, "$"), "\\1", hc)
  k <- !sets %in% DROP
  hc <- hc[k]; sets <- sets[k]
  if (length(hc) >= 2) {
    comp <- pp %>% filter(!is_canonical) %>% select(all_of(hc))
    names(comp) <- sets
    mat <- as.matrix(mutate(comp, across(everything(), as.logical)))
    mat[is.na(mat)] <- FALSE
    mat <- mat[rowSums(mat) > 0, , drop = FALSE]

    key <- apply(mat, 1, function(r) paste(which(r), collapse = ","))
    tb  <- head(sort(table(key), decreasing = TRUE), 12)
    ids <- names(tb)
    inter <- tibble(ord = seq_along(ids), n = as.integer(tb))
    setsz <- tibble(set = sets, size = colSums(mat))

    dots <- bind_rows(lapply(seq_along(ids), function(j) {
      mem <- as.integer(strsplit(ids[j], ",")[[1]])
      tibble(ord = j, set = sets, si = seq_along(sets),
             member = seq_along(sets) %in% mem)
    }))
    dots$set <- factor(dots$set, levels = rev(sets))
    seg <- dots %>% filter(member) %>% group_by(ord) %>%
      summarise(lo = min(si), hi = max(si), .groups = "drop") %>%
      filter(hi > lo) %>%
      mutate(lo = length(sets) - lo + 1, hi = length(sets) - hi + 1)

    p_top <- ggplot(inter, aes(ord, n)) +
      geom_col(width = .66, fill = "#1F4E5F") +
      geom_text(aes(label = comma(n)), vjust = -0.4, size = 2.2) +
      scale_y_continuous(expand = expansion(c(0, .22)), labels = comma) +
      scale_x_continuous(breaks = inter$ord, expand = expansion(add = .6)) +
      labs(title = "c  Competitors shared between datasets",
           subtitle = paste0("tolerance ", TOL, " nt; ", comma(nrow(mat)),
                             " competitors with at least one match"),
           x = NULL, y = "competitors") +
      theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
            axis.line.x = element_blank())

    p_dot <- ggplot(dots, aes(ord, set)) +
      geom_point(colour = "grey87", size = 2.3) +
      geom_segment(data = seg, aes(x = ord, xend = ord, y = lo, yend = hi),
                   inherit.aes = FALSE, linewidth = .5, colour = "#1F4E5F") +
      geom_point(data = filter(dots, member), colour = "#1F4E5F", size = 2.3) +
      scale_x_continuous(breaks = inter$ord, expand = expansion(add = .6)) +
      labs(x = NULL, y = NULL,
           caption = paste0("set sizes: ",
                            paste(setsz$set, comma(setsz$size),
                                  sep = " ", collapse = ",  "))) +
      theme(axis.text.x = element_blank(), axis.ticks = element_blank(),
            axis.line = element_blank(),
            plot.caption = element_text(size = 6, colour = "grey40", hjust = 0))

    panels$c <- p_top / p_dot + plot_layout(heights = c(2.6, 1))
  }
}

rows <- Filter(Negate(is.null), list(panels$a, panels$b, panels$c))
if (!length(rows)) stop("no panels could be built")
fig <- Reduce(`/`, rows) + plot_layout(heights = c(1, 1.3, 1.5)[seq_along(rows)])

h <- 60 * length(rows)
ok <- tryCatch({
  ggsave(paste0(outstem, ".pdf"), fig, width = 180, height = h, units = "mm",
         device = grDevices::cairo_pdf); TRUE
}, error = function(e) FALSE)
if (!ok) ggsave(paste0(outstem, ".pdf"), fig, width = 180, height = h, units = "mm")
ggsave(paste0(outstem, ".png"), fig, width = 180, height = h, units = "mm", dpi = 400)
message("wrote ", outstem, ".pdf and .png")