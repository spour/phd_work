# run one of pub/'s figure scripts for its panel objects without letting it write anything
PUB <- normalizePath("..")
harvest <- function(script) {
  env <- new.env(parent = globalenv())
  env$ggsave <- function(...) invisible(NULL)
  env$writeLines <- function(text, con = stdout(), ...)
    if (is.character(con)) invisible(NULL) else base::writeLines(text, con, ...)
  old <- setwd(PUB); on.exit(setwd(old))
  suppressMessages(suppressWarnings(sys.source(file.path(PUB, script), envir = env)))
  env
}
retitle <- function(p, title, keep_sub = FALSE)
  p + labs(title = title, subtitle = if (keep_sub) p$labels$subtitle else NULL, caption = NULL)

labtext <- function(l) {
  lab <- c(l$aes_params$label, if (is.data.frame(l$data) && "label" %in% names(l$data)) as.character(l$data$label))
  if (is.null(lab) && !is.null(l$mapping$label)) lab <- paste0("<mapped ", rlang::quo_text(l$mapping$label), ">")
  paste(lab, collapse = " | ")
}
is_text <- function(l) inherits(l$geom, c("GeomText", "GeomLabel"))
edit_text <- function(p, drop = character(), relab = character()) {
  keep <- rep(TRUE, length(p$layers)); hit <- setNames(rep(0L, length(c(drop, names(relab)))), c(drop, names(relab)))
  for (i in seq_along(p$layers)) { l <- p$layers[[i]]; if (!is_text(l)) next
    t <- labtext(l)
    for (d in drop) if (grepl(d, t)) { keep[i] <- FALSE; hit[d] <- hit[d] + 1L }
    for (r in names(relab)) if (keep[i] && grepl(r, t)) { l$aes_params$label <- relab[[r]]; p$layers[[i]] <- l; hit[r] <- hit[r] + 1L }
  }
  miss <- names(hit)[hit == 0]; if (length(miss)) stop("edit_text: no layer matched ", paste(miss, collapse = "; "))
  p$layers <- p$layers[keep]; p
}

