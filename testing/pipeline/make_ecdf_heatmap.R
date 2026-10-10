#!/usr/bin/env Rscript
# distance ecdfs above, containment heatmap below

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
  library(readr); library(scales); library(forcats)
})
args  <- commandArgs(trailingOnly=TRUE)
INDIR <- if(length(args)>=1) args[1] else "hiconf_out"
STEM  <- if(length(args)>=2) args[2] else "fig_ecdf_heatmap"
rd <- function(f, req=TRUE){
  p <- file.path(INDIR,f)
  if(!file.exists(p)){ if(req) stop("missing ",p); return(NULL) }
  read_tsv(p, show_col_types=FALSE, progress=FALSE)
}

INK <- "#141414"; MUT <- "#5A5A5A"; BASE <- 13
TOL <- 50

JM <- rd("jaccard_matrix.tsv")
DD <- rd("distances.tsv", req=FALSE)
HX <- rd("hexamer.tsv", req=FALSE)

sz    <- JM |> distinct(set_a, n_a)
sizes <- setNames(sz$n_a, sz$set_a)
labs  <- if(!is.null(HX)) setNames(HX$label, HX$name) else
         setNames(names(sizes), names(sizes))
ord   <- names(sort(sizes, decreasing=TRUE))
pl    <- function(x) labs[as.character(x)]

th <- theme_classic(base_size=BASE)+
  theme(text=element_text(colour=INK),
        axis.text=element_text(colour=INK, size=BASE-4),
        axis.title=element_text(size=BASE-2),
        axis.line=element_line(colour=INK, linewidth=0.45),
        axis.ticks=element_line(colour=INK, linewidth=0.45),
        plot.title=element_text(size=BASE+1, face="bold", hjust=0,
                                margin=margin(b=6)),
        plot.title.position="plot",
        plot.subtitle=element_blank(),
        legend.title=element_text(size=BASE-4),
        legend.text=element_text(size=BASE-4),
        legend.key.size=unit(0.8,"lines"),
        strip.background=element_blank(),
        strip.text=element_text(face="bold", size=BASE-4, margin=margin(b=3)),
        panel.grid.major=element_line(linewidth=0.2, colour="grey94"),
        plot.margin=margin(5,10,5,8))

# a  distance ecdfs
pe <- NULL
if(!is.null(DD)){
  d <- DD |>
    filter(is.finite(dist)) |>
    mutate(dist=pmin(pmax(dist, 1), 1e7),
           a=factor(set_a, levels=ord), b=factor(set_b, levels=ord))
  facet_n <- function(x) paste0(pl(x), "  (n = ",
                                comma(sizes[as.character(x)]), ")")
  pe <- ggplot(d, aes(dist, colour=b))+
    geom_vline(xintercept=TOL, linetype=3, linewidth=0.4, colour="grey45")+
    stat_ecdf(geom="step", linewidth=0.65, pad=FALSE)+
    facet_wrap(~a, nrow=2, labeller=as_labeller(facet_n))+
    scale_x_log10(labels=label_number(scale_cut=cut_short_scale()),
                  limits=c(1, 1e7))+
    scale_y_continuous(labels=percent, limits=c(0,1))+
    scale_colour_brewer(palette="Dark2", name="nearest site in",
                        labels=pl)+
    labs(title=paste0("a   Distance to the nearest site in every other set  ",
                      "(", comma(sum(sizes)), " sites, ", length(ord),
                      " sets; dotted line = ", TOL, " nt)"),
         x="distance to the nearest site (nt)",
         y="sites within that distance")+th+
    theme(legend.position="right")
}

# b  containment heatmap
hm <- JM |>
  mutate(a=factor(set_a, levels=ord), b=factor(set_b, levels=ord),
         v=ifelse(set_a==set_b, NA_real_, containment_a_in_b))
vmax <- max(hm$v, na.rm=TRUE)
ph <- ggplot(hm, aes(b, fct_rev(a), fill=v))+
  geom_tile(colour="white", linewidth=0.8)+
  geom_text(aes(label=ifelse(is.na(v), "\u2014",
                             ifelse(v < 0.001, "<.001", sprintf("%.3f", v))),
                colour=!is.na(v) & v > vmax*0.55),
            size=2.5, show.legend=FALSE)+
  scale_colour_manual(values=c(`TRUE`="white", `FALSE`="grey15"),
                      na.value="grey60")+
  scale_fill_gradientn(colours=c("#FFFFFF","#DEEBF7","#9ECAE1","#4292C6",
                                 "#2171B5","#08306B"),
                       na.value="grey96", name="fraction of\nthe row set")+
  scale_x_discrete(labels=function(x)
    paste0(pl(x), "\nn = ", comma(sizes[as.character(x)])))+
  scale_y_discrete(labels=function(x)
    paste0(pl(x), "  (", comma(sizes[as.character(x)]), ")"))+
  labs(title=paste0("b   Fraction of each row set's sites lying within ", TOL,
                    " nt of a site in the column set"),
       x=NULL, y=NULL)+th+
  theme(axis.text.x=element_text(angle=35, hjust=1, size=BASE-5,
                                 lineheight=0.9),
        axis.text.y=element_text(size=BASE-5),
        axis.line=element_blank(), axis.ticks=element_blank(),
        panel.grid=element_blank(),
        legend.position="right", legend.key.height=unit(11,"mm"),
        legend.key.width=unit(3.5,"mm"))

# c  canonical pas
ps <- NULL
if(!is.null(HX)){
  hasexp <- "pct_expected" %in% names(HX)
  keep <- HX |> filter(name %in% ord) |> mutate(name=factor(name, levels=ord))
  g <- keep |>
    select(name, `at the sites`=pct_obs, `displaced 5 kb`=pct_rand,
           any_of(c(`expected from base composition`="pct_expected"))) |>
    pivot_longer(-name, names_to="which", values_to="pct") |>
    mutate(which=factor(which, levels=c("at the sites","displaced 5 kb",
                                        "expected from base composition")))
  cols <- c(`at the sites`="#B2182B", `displaced 5 kb`="#9E9E9E",
            `expected from base composition`="#DCDCDC")
  fl <- keep |>
    mutate(fold=pct_obs/pmax(pct_rand, 0.01),
           x=pmax(pct_obs, pct_rand, if(hasexp) pct_expected else 0))
  ps <- ggplot(g, aes(pct, fct_rev(name), fill=which))+
    geom_col(position=position_dodge(width=0.78), width=0.7)+
    geom_text(data=fl, aes(x=x+4, y=fct_rev(name),
                           label=sprintf("%.1f\u00d7", fold)),
              inherit.aes=FALSE, hjust=0, size=2.5, colour="#333333")+
    scale_fill_manual(values=cols, name=NULL, drop=TRUE)+
    scale_x_continuous(limits=c(0,120), breaks=c(0,50,100),
                       expand=expansion(c(0,0)))+
    scale_y_discrete(labels=NULL)+
    labs(title="c   AATAAA or ATTAAA in the 60 nt upstream",
         x="% of sites", y=NULL)+th+
    theme(axis.ticks.y=element_blank(),
          legend.position="bottom", legend.direction="vertical",
          legend.margin=margin(t=-4), legend.key.size=unit(0.7,"lines"))
}

# assemble
bottom <- if(is.null(ps)) ph else (ph | ps) + plot_layout(widths=c(1, 0.42))
fig <- if(is.null(pe)) bottom else pe / bottom + plot_layout(heights=c(1, 1.3))

n <- length(ord)
W <- 14 + 0.6*n
H <- (if(is.null(pe)) 0 else 8) + 4 + 0.66*n
ggsave(paste0(STEM,".pdf"), fig, width=W, height=H, device=cairo_pdf,
       limitsize=FALSE)
ggsave(paste0(STEM,".png"), fig, width=W, height=H, dpi=400, limitsize=FALSE)
cat("[out] ", STEM, ".pdf / .png\n", sep="")

cat("\nset sizes:\n")
data.frame(set=pl(ord), n=comma(sizes[ord])) |> print(row.names=FALSE)
cat("\ncontainment at ", TOL, " nt (row inside column), top pairs:\n", sep="")
JM |> filter(set_a != set_b) |>
  select(set_a, set_b, n_a, n_b, n_a_in_b, containment_a_in_b, jaccard) |>
  arrange(desc(containment_a_in_b)) |> head(12) |>
  as.data.frame() |> print(row.names=FALSE, digits=3)
if(!is.null(HX)){
  cat("\ncanonical PAS:\n")
  HX |> mutate(fold=pct_obs/pmax(pct_rand,0.01)) |>
    select(label, n, n_scored, pct_obs, pct_rand,
           any_of("pct_expected"), fold) |>
    as.data.frame() |> print(row.names=FALSE, digits=3)
}