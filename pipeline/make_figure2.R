#!/usr/bin/env Rscript
# figure 2: competitor cpa (ccpa) sites are sequence-competent and cpa grammar cannot

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
  library(readr); library(scales)
})
args <- commandArgs(trailingOnly=TRUE)
INDIR <- if(length(args)>=1) args[1] else "."
have <- function(f){p<-file.path(INDIR,f); if(file.exists(p)) p else NA_character_}

INK<-"#1A1A1A"; BASE<-16
CANON<-"#0072B2"; CC<-"#E69F00"
CCLAB <- "cCPA site"

relab <- function(x) dplyr::recode(x,
                                   `Competitor` = CCLAB,
                                   `Model false positive` = CCLAB,
                                   .default = x)

# title flush to the panel
# panel letter and the plotting area
th <- theme_classic(base_size=BASE)+
  theme(text=element_text(color=INK),
        axis.text=element_text(color=INK,size=BASE-3),
        axis.title=element_text(size=BASE-2),
        axis.line=element_line(color=INK,linewidth=0.5),
        plot.title=element_text(size=BASE+2,face="bold",hjust=0,
                                margin=margin(b=1)),
        plot.title.position="plot",
        plot.subtitle=element_blank(),
        legend.position="bottom",
        legend.title=element_blank(),
        legend.text=element_text(size=BASE-4),
        legend.margin=margin(t=-4,b=0),
        legend.box.margin=margin(t=-2),
        legend.key.size=unit(0.8,"lines"),
        strip.background=element_blank(),
        strip.text=element_text(face="bold",size=BASE-4,
                                margin=margin(t=0,b=2)),
        plot.margin=margin(2,8,3,6))
ph <- function(t,m="awaiting data") ggplot()+annotate("text",0,0,label=m,size=4,color="#8A8F98")+
  lims(x=c(-1,1),y=c(-1,1))+labs(title=t)+th+
  theme(axis.line=element_blank(),axis.text=element_blank(),
        axis.ticks=element_blank(),axis.title=element_blank())

# fig2_counts.tsv is keyed by the old panel letters from figure2_stats.py
# a dispersion, b motif density, c rank, d pas-null
OLD2NEW <- c(A="A", C="B", D="C", B="F")
NC <- list()
fn <- have("fig2_counts.tsv")
if(!is.na(fn)){
  nc <- read_tsv(fn, show_col_types=FALSE)
  for(i in seq_len(nrow(nc))){
    k <- OLD2NEW[[nc$panel[i]]]
    if(!is.null(k)) NC[[k]] <- list(n=nc$n[i], unit=nc$unit[i])
  }
}

AD <- NULL
fap <- have("fig2_aparent.tsv")
if(!is.na(fap)) AD <- read_tsv(fap, show_col_types=FALSE)
getm <- function(k){
  if(is.null(AD)) return(NULL)
  r <- AD[AD$kind==k,]
  if(nrow(r)==0) NULL else as.list(r[1,])
}

fa <- have("fig2_dispersion.tsv")
if(!is.na(fa)){
  d <- read_tsv(fa,show_col_types=FALSE)
  med <- median(d$spread_nt); f50 <- mean(d$spread_nt<=50)
  XMAX <- 60; ntail <- mean(d$spread_nt > XMAX)
  pA <- ggplot(d, aes(spread_nt))+
    geom_histogram(binwidth=2,fill=CC,color=NA,boundary=0)+
    geom_vline(xintercept=med,linetype=2,color=INK,linewidth=0.5)+
    annotate("text",x=med+2,y=Inf,vjust=1.5,hjust=0,size=3.2,color=INK,
             label=paste0("median ",med," nt"))+
    annotate("text",x=XMAX,y=Inf,vjust=3.2,hjust=1,size=3.0,color=INK,
             label=paste0(percent(f50,accuracy=0.1)," within 50 nt"))+
    annotate("text",x=XMAX,y=Inf,vjust=5.0,hjust=1,size=3.0,color=INK,
             label=paste0(percent(ntail,accuracy=0.1)," beyond ",XMAX," nt"))+
    coord_cartesian(xlim=c(-1,XMAX+1))+
    scale_y_continuous(expand=expansion(mult=c(0,0.20)))+
    labs(title="A",x="Spread across the three model peaks (nt)",y="cCPA sites")+th
} else pA <- ph("A","inter-model spread")

fcb <- have("fig2_rank_binned.tsv")
if(!is.na(fcb)){
  d <- read_tsv(fcb,show_col_types=FALSE) |> arrange(lo) |>
    mutate(bin=factor(bin, levels=bin))
  w <- getm("within_gene_rank")
  rlab <- if(is.null(w)) "" else sprintf("overall %.3f [%.3f, %.3f]", w$value, w$lo, w$hi)
  pB <- ggplot(d, aes(bin, mean))+
    geom_hline(yintercept=0.5, linetype=2, color=INK, linewidth=0.5)+
    geom_errorbar(aes(ymin=ci_lo, ymax=ci_hi), width=0.16, linewidth=0.6, color=CC)+
    geom_point(size=2.6, color=CC)+
    geom_text(aes(label=comma(n_genes), y=ci_hi), vjust=-0.9, size=2.6, color="#555555")+
    annotate("text", x=-Inf, y=0.5, vjust=-0.6, hjust=-0.10, size=2.9, color=INK,
             label="no preference")+
    annotate("text", x=Inf, y=-Inf, vjust=-0.8, hjust=1.05, size=2.9, color=INK,
             label=rlab)+
    coord_cartesian(ylim=c(0.30,0.66), clip="off")+
    scale_y_continuous(breaks=seq(0.3,0.65,0.1))+
    labs(title="B",x="cCPA sites in the gene", y="Rank of canonical site")+th
} else pB <- ph("B","within-gene rank")

fd <- have("shortlist_cpa_density.tsv")
if(is.na(fd)) fd <- have("fig2_density.tsv")
if(!is.na(fd)){
  d <- read_tsv(fd,show_col_types=FALSE) |>
    filter(motif %in% c("PAS","PAS (dinuc null)")) |>
    mutate(src=ifelse(motif=="PAS","Real","Dinucleotide null"),
           cls=relab(cls),
           cls=factor(cls,levels=c("Canonical",CCLAB)),
           src=factor(src,levels=c("Real","Dinucleotide null")))
  d <- d |> group_by(cls,src) |> arrange(pos) |>
    mutate(density=stats::filter(density,rep(1/9,9),sides=2) |> as.numeric()) |>
    ungroup() |> filter(!is.na(density))
  pC <- ggplot(d,aes(pos,density,color=cls,linetype=src))+
    geom_vline(xintercept=0,linetype=3,color="#B0B4BA",linewidth=0.4)+
    geom_line(linewidth=0.8)+
    scale_color_manual(values=setNames(c(CANON,CC), c("Canonical",CCLAB)),name=NULL)+
    scale_linetype_manual(values=c(Real="solid",`Dinucleotide null`="22"),name=NULL)+
    guides(color=guide_legend(order=1,nrow=1),linetype=guide_legend(order=2,nrow=1))+
    labs(title="C",x="Position relative to site (nt)",y="PAS density per peak")+th
} else pC <- ph("C","dinucleotide null")

fe <- have("fig2_roc.tsv")
if(!is.na(fe)){
  d <- read_tsv(fe,show_col_types=FALSE)
  ro <- getm("pooled_auroc")
  lab <- if(is.null(ro)) "" else
    sprintf("AUROC %.3f\n[%.3f, %.3f]\nchance 0.500", ro$value, ro$lo, ro$hi)
  pD <- ggplot(d, aes(fpr, tpr))+
    geom_abline(slope=1,intercept=0,linetype=2,color=INK,linewidth=0.5)+
    geom_line(color=CC,linewidth=1.0)+
    annotate("text", x=0.97, y=0.05, hjust=1, vjust=0, size=3.0, color=INK,
             lineheight=1.05, label=lab)+
    coord_equal(xlim=c(0,1), ylim=c(0,1))+
    scale_x_continuous(breaks=seq(0,1,0.25))+
    scale_y_continuous(breaks=seq(0,1,0.25))+
    labs(title="D",x="False positive rate", y="True positive rate")+th
} else pD <- ph("D","ROC")

ff <- have("fig2_pr.tsv")
if(!is.na(ff)){
  d <- read_tsv(ff,show_col_types=FALSE)
  bl <- if("baseline" %in% names(d)) d$baseline[1] else NA_real_
  pr <- getm("pooled_auprc")
  lab <- if(is.null(pr)) "" else
    sprintf("AUPRC %.3f\n[%.3f, %.3f]\nchance %.3f", pr$value, pr$lo, pr$hi, pr$baseline)
  pE <- ggplot(d, aes(recall, precision))+
    { if(is.finite(bl)) geom_hline(yintercept=bl,linetype=2,color=INK,linewidth=0.5) } +
    geom_line(color=CC,linewidth=1.0)+
    annotate("text", x=0.97, y=0.97, hjust=1, vjust=1, size=3.0, color=INK,
             lineheight=1.05, label=lab)+
    { if(is.finite(bl)) annotate("text", x=0.02, y=bl, hjust=0, vjust=-0.5,
                                 size=2.8, color=INK, label="chance") } +
    coord_equal(xlim=c(0,1), ylim=c(0,1))+
    scale_x_continuous(breaks=seq(0,1,0.25))+
    scale_y_continuous(breaks=seq(0,1,0.25))+
    labs(title="E",x="Recall", y="Precision")+th
} else pE <- ph("E","precision-recall")

fb <- have("shortlist_cpa_density.tsv")
if(is.na(fb)) fb <- have("fig2_density.tsv")
if(!is.na(fb)){
  d <- read_tsv(fb,show_col_types=FALSE) |>
    filter(motif %in% c("PAS","UGUA","U-rich","GU-rich")) |>
    mutate(motif=factor(motif,levels=c("PAS","UGUA","U-rich","GU-rich")),
           cls=relab(cls),
           cls=factor(cls,levels=c("Canonical",CCLAB)))
  d <- d |> group_by(cls,motif) |> arrange(pos) |>
    mutate(density=stats::filter(density,rep(1/9,9),sides=2) |> as.numeric()) |>
    ungroup() |> filter(!is.na(density))
  pF <- ggplot(d,aes(pos,density,color=cls))+
    geom_vline(xintercept=0,linetype=3,color="#B0B4BA",linewidth=0.4)+
    geom_line(linewidth=0.8)+
    facet_wrap(~motif,nrow=2,scales="free_y")+
    scale_color_manual(values=setNames(c(CANON,CC), c("Canonical",CCLAB)))+
    labs(title="F",x="Position relative to site (nt)",y="Motif density per peak")+th
} else pF <- ph("F","motif density")

fig <- (pA | pB | pC) / (pD | pE | pF)
ggsave(file.path(INDIR,"figure2.pdf"),fig,width=18,height=11,device=cairo_pdf)
ggsave(file.path(INDIR,"figure2.png"),fig,width=18,height=11,dpi=400)
cat("[out] figure2.pdf, figure2.png\n")

if(length(NC)){
  cat("\npanel n (for the figure legend):\n")
  for(p in sort(names(NC)))
    cat(sprintf("  %s  %s %s\n", p, comma(NC[[p]]$n), NC[[p]]$unit))
}
cat("  D  38,186 cCPA + 5,898 canonical (APARENT2 available)\n")
cat("  E  38,186 cCPA + 5,898 canonical (APARENT2 available)\n")