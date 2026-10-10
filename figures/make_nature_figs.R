#!/usr/bin/env Rscript
# re-render every coded thesis figure (ch2 2.1, 2.3, 2.4, 2.5, 2.9
suppressPackageStartupMessages({library(ggplot2); library(patchwork)})
PUB <- normalizePath(".")
OUT <- file.path(PUB, "nature_figs")
FAM <- "Helvetica"
W_MM <- 183; H_MAX <- 247

JOBS <- list(
  list(script = "sfig1.R", args = "constitutive_cpa.tsv", wd = PUB, map = c(supp_fig1 = "fig2_1")),
  list(script = "scripts/make_figure1.R", args = ".", wd = PUB, map = c(figure1 = "fig2_3")),
  list(script = "make_analysis_set_funnel.R", wd = PUB, map = c(figure_analysis_set_funnel = "fig2_4")),
  list(script = "scripts/make_figure2.R", args = ".", wd = PUB, map = c(figure2 = "fig2_5")),
  list(script = "make_figure_overlap_shared.R", wd = PUB, map = c(figure_overlap_shared = "fig2_9")),
  list(script = "thesis_ch3/R/fig3_01_hypothesis.R", wd = file.path(PUB, "thesis_ch3"), map = c(fig3_01_hypothesis = "fig3_1")),
  list(script = "thesis_ch3/R/fig3_02_genesets.R", wd = file.path(PUB, "thesis_ch3"), map = c(fig3_02_genesets = "fig3_2")),
  list(script = "thesis_ch3/R/fig3_regroup.R", wd = file.path(PUB, "thesis_ch3"),
       map = c(fig3_03_splice_environment = "fig3_3", fig3_04_donor_scores = "fig3_4",
               fig3_05_splice_model = "fig3_6", fig3_06_splice_model_performance = "fig3_7",
               fig3_07_architecture = "fig3_8", fig3_08_cnn_vs_splice = "fig3_9",
               fig3_12_interpretation = "fig3_11")),
  list(script = "thesis_ch3/R/fig3_hypothesis_dose.R", wd = file.path(PUB, "thesis_ch3"), map = c(fig3_hypothesis_dose = "fig3_5")),
  list(script = "thesis_ch3/R/fig3_09_robustness.R", wd = file.path(PUB, "thesis_ch3"), map = c(fig3_09_robustness = "fig3_10")),
  list(script = "thesis_ch3/R/fig3_13_mechanism.R", wd = file.path(PUB, "thesis_ch3"), map = c(fig3_13_mechanism = "fig3_12")),
  list(script = "thesis_ch3/R/fig3_implant_panel.R", wd = file.path(PUB, "thesis_ch3"), map = c(fig3_implant_panel = "fig3_13")),
  list(script = "thesis_ch3/R/fig3_14_recovery.R", wd = file.path(PUB, "thesis_ch3"), map = c(fig3_14_recovery = "fig3_14"))
)

# capture
capture <- function(job) {
  got <- list()
  env <- new.env(parent = globalenv())
  env$ggsave <- function(filename, plot = last_plot(), width = NA, height = NA, units = "in", ...) {
    key <- tools::file_path_sans_ext(basename(filename))
    mm <- switch(units, `in` = 25.4, cm = 10, mm = 1, px = NA)
    got[[key]] <<- list(plot = plot, w = width * mm, h = height * mm)
    invisible(NULL)
  }
  env$write.table <- function(...) invisible(NULL)
  env$write.csv <- function(...) invisible(NULL)
  env$commandArgs <- function(trailingOnly = FALSE) if (trailingOnly) job$args %||% character() else c("R", job$args)
  old <- setwd(job$wd %||% PUB); on.exit(setwd(old))
  suppressMessages(suppressWarnings(sys.source(file.path(PUB, job$script), envir = env)))
  got
}

# restyle
blank <- function(e) is.null(e) || inherits(e, "element_blank")
letter_of <- function(t) {
  if (is.null(t) || !is.character(t) || !nzchar(t)) return(NULL)
  m <- regmatches(t, regexec("^\\s*([A-Za-z])(\\s{2,}|\\s*$|[.)]\\s)", t))[[1]]
  if (length(m)) tolower(m[2]) else NULL
}

style_plot <- function(p, s) {
  full <- tryCatch(ggplot2:::plot_theme(p), error = function(e) theme_classic())
  lab <- letter_of(p$labels$title)
  p <- p + labs(title = lab, subtitle = NULL, caption = NULL)
  axes  <- !blank(full$axis.line) || !blank(full$axis.line.x) || !blank(full$axis.line.y)
  ticks <- !blank(full$axis.ticks)
  tiles <- any(vapply(p$layers, function(l) inherits(l$geom, c("GeomTile", "GeomRaster")), TRUE))
  data_axes <- !blank(full$axis.text) && !blank(full$axis.text.x) && !blank(full$axis.text.y)
  if (data_axes && !tiles) { axes <- TRUE; ticks <- TRUE }
  txt <- function(size, ...) element_text(family = FAM, size = size, colour = "black", ...)
  th <- theme(text = element_text(family = FAM, size = 7, colour = "black"),
              strip.background = element_blank(), strip.clip = "off",
              panel.grid = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
              panel.grid.major.x = element_blank(), panel.grid.major.y = element_blank(),
              panel.grid.minor.x = element_blank(), panel.grid.minor.y = element_blank(),
              panel.border = element_blank(),
              plot.title = element_text(family = FAM, size = 8, face = "bold", hjust = 0,
                                        margin = margin(b = 2)),
              plot.title.position = "plot",
              plot.subtitle = element_blank(), plot.caption = element_blank(),
              plot.tag = element_text(family = FAM, size = 8, face = "bold"),
              legend.key.size = unit(3, "mm"), legend.key.spacing = unit(1, "mm"),
              legend.margin = margin(0, 0, 0, 0), legend.box.spacing = unit(1.5, "mm"),
              plot.margin = margin(3, 4, 3, 3))
  keep <- function(els, size, ...) for (el in els) if (!blank(full[[el]]))
    th <<- th + do.call(theme, setNames(list(txt(size, inherit.blank = isTRUE(full[[el]]$inherit.blank), ...)), el))
  keep(c("axis.title", "axis.title.x", "axis.title.y"), 7)
  keep(c("strip.text", "strip.text.x", "strip.text.y"), 7, face = "plain")
  keep(c("axis.text", "axis.text.x", "axis.text.y"), 6)
  keep("legend.title", 6.5); keep("legend.text", 6, margin = margin(l = 1.2, r = 2.5))
  if (axes) th <- th + theme(axis.line = element_line(linewidth = 0.25, colour = "black"))
  if (ticks) th <- th + theme(axis.ticks = element_line(linewidth = 0.25, colour = "black"),
                              axis.ticks.length = unit(1, "mm"))
  p <- p + th
  g <- 0.6 * s
  for (l in p$layers) {
    ap <- l$aes_params
    if (inherits(l$geom, c("GeomText", "GeomLabel"))) {
      sz <- ap$size %||% 3.88
      l$aes_params$size <- min(max(sz * s, 5 / .pt), 7 / .pt)
      l$aes_params$family <- FAM
    } else {
      if (!is.null(ap$linewidth)) l$aes_params$linewidth <- max(ap$linewidth * g, 0.25)
      if (!is.null(ap$size) && is.numeric(ap$size)) l$aes_params$size <- max(ap$size * g, 0.5)
    }
  }
  p
}

map_leaves <- function(p, f) {
  if (inherits(p, "patchwork")) p$patches$plots <- lapply(p$patches$plots, map_leaves, f = f)
  if (inherits(p, "ggplot")) p <- f(p)
  p
}

restyle <- function(p, s) {
  if (inherits(p, "patchwork")) {
    p$patches$plots <- lapply(p$patches$plots, restyle, s = s)
    a <- p$patches$annotation
    a$title <- NULL; a$subtitle <- NULL; a$caption <- NULL
    if (identical(a$tag_levels, "A") || identical(a$tag_levels, list("A"))) a$tag_levels <- "a"
    p$patches$annotation <- a
  }
  if (inherits(p, "ggplot")) p <- style_plot(p, s)
  p
}

auto_letter <- function(p) {
  n <- 0; has <- FALSE
  map_leaves(p, function(q) { n <<- n + 1; if (!is.null(q$labels$title)) has <<- TRUE; q })
  if (has || n < 2) return(p)
  i <- 0
  map_leaves(p, function(q) { i <<- i + 1; q + labs(title = letters[i]) })
}

# per-figure presentation fixes
lab_of <- function(l) paste(c(l$aes_params$label,
  if (is.data.frame(l$data) && "label" %in% names(l$data)) as.character(l$data$label),
  if (!is.null(l$mapping$label)) rlang::quo_text(l$mapping$label)), collapse = " | ")
drop_text <- function(q, re) {
  q$layers <- Filter(function(l) !(inherits(l$geom, c("GeomText", "GeomLabel")) && grepl(re, lab_of(l))), q$layers); q }
FIX <- list(
  fig2_1 = function(q) drop_text(q, "retained"),
  fig2_9 = function(q) {
    if (inherits(q$facet, "FacetWrap") && "src" %in% names(q$facet$params$facets))
      q$facet$params$labeller <- as_labeller(function(x) vapply(sub("\\s*\\([^)]*sites?\\)$", "", x),
                                             function(y) paste(strwrap(y, 14), collapse = "\n"), ""))
    if (any(vapply(q$layers, function(l) grepl("fold", lab_of(l)), TRUE))) q$coordinates$clip <- "off"
    q },
  fig3_4 = function(q) {
    if (inherits(q$facet, "FacetWrap") && "score" %in% names(q$facet$params$facets))
      q$facet$params$labeller <- label_wrap_gen(32)
    q },
  fig3_12 = function(q) {
    if (is.character(q$labels$x) && grepl("annotated site", q$labels$x))
      q <- q + theme(axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1))
    q }
)

write_fig <- function(obj, name) {
  w <- W_MM
  s <- min(1, W_MM / obj$w)
  h <- min(obj$h * W_MM / obj$w, H_MAX)
  p <- obj$plot
  if (!is.null(FIX[[name]])) p <- map_leaves(p, FIX[[name]])
  fig <- auto_letter(restyle(p, s)) & theme(plot.subtitle = element_blank(), plot.caption = element_blank())
  ggsave(file.path(OUT, paste0(name, ".pdf")), fig, width = w, height = h, units = "mm",
         device = cairo_pdf, limitsize = FALSE)
  ggsave(file.path(OUT, paste0(name, ".png")), fig, width = w, height = h, units = "mm",
         dpi = 450, bg = "white", limitsize = FALSE)
  cat(sprintf("[out] %-8s %3.0f x %3.0f mm  (was %3.0f x %3.0f mm, stroke scale %.2f)\n",
              name, w, h, obj$w, obj$h, s))
}

want <- commandArgs(trailingOnly = TRUE)
for (job in JOBS) {
  if (length(want) && !any(job$map %in% want)) next
  got <- capture(job)
  for (k in names(job$map)) {
    if (length(want) && !(job$map[[k]] %in% want)) next
    if (is.null(got[[k]])) { cat("[miss]", job$script, k, "\n"); next }
    write_fig(got[[k]], job$map[[k]])
  }
}
