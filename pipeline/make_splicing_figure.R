#!/usr/bin/env Rscript
# splice-site features discriminate real 3' ends from competing intragenic sites

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
  library(readr); library(scales); library(forcats)
})
args  <- commandArgs(trailingOnly=TRUE)
INDIR <- if(length(args)>=1) args[1] else "splmodel"
STEM  <- if(length(args)>=2) args[2] else "figure_splicing"
rd <- function(f, req=TRUE){
  p <- file.path(INDIR,f)
  if(!file.exists(p)){ if(req) stop("missing ",p); return(NULL) }
  read_tsv(p, show_col_types=FALSE, progress=FALSE)
}

INK <- "#141414"; MUT <- "#5A5A5A"; BASE <- 13
REAL <- "#08519C"; COMP <- "#D95F02"
GCCOL <- "#8C8C8C"; SPL <- "#00776A"

PAIR  <- c("gc_only","minimal_centred")
SHORT <- c(gc_only="GC content", minimal_centred="Splice-site features")
PAL   <- c(`GC content`=GCCOL, `Splice-site features`=SPL)
fx <- function(x) factor(SHORT[as.character(x)], levels=unname(SHORT[PAIR]))

FEATNAME <- c(
  pas_prox_u1        = "U1 near poly(A) site",
  upstream_max_me3   = "upstream 3'ss strength",
  term_ppt           = "terminal pyrimidine tract",
  sum_5ss            = "total 5'ss strength",
  comp_up_3ss        = "competing upstream 3'ss",
  n_5ss_500          = "5'ss within 500 nt",
  dist_nearest_5ss   = "distance to nearest 5'ss",
  dist_dn_5ss_missing        = "no downstream 5'ss",
  dist_nearest_5ss_missing   = "no 5'ss in range")
pf_ <- function(x){
  k <- sub("^spl_", "", x); o <- FEATNAME[k]
  ifelse(is.na(o), gsub("_", " ", k), o)
}

th <- theme_classic(base_size=BASE)+
  theme(text=element_text(colour=INK),
        axis.text=element_text(colour=INK, size=BASE-4),
        axis.title=element_text(size=BASE-2),
        axis.line=element_line(colour=INK, linewidth=0.45),
        axis.ticks=element_line(colour=INK, linewidth=0.45),
        plot.title=element_text(size=BASE+1, face="bold", hjust=0,
                                margin=margin(b=5)),
        plot.title.position="plot",
        plot.subtitle=element_blank(),
        legend.position="bottom", legend.title=element_blank(),
        legend.text=element_text(size=BASE-4),
        legend.margin=margin(t=-4,b=0), legend.key.size=unit(0.7,"lines"),
        legend.box.spacing=unit(2,"mm"),
        strip.background=element_blank(),
        strip.text=element_text(face="bold", size=BASE-3, margin=margin(b=3)),
        panel.grid.major=element_line(linewidth=0.2, colour="grey94"),
        plot.margin=margin(6,12,6,8))

MET <- rd("metrics.tsv"); DEC <- rd("metrics_by_decile.tsv")
DIS <- rd("discriminant.tsv"); DFI <- rd("discriminant_fits.tsv")
PG  <- rd("gc_within_gene.tsv"); PFI <- rd("paradox_fit.tsv", req=FALSE)
PO  <- rd("gc_pooled.tsv", req=FALSE)
MF  <- rd("minimal_features.tsv", req=FALSE)
LO  <- rd("loco.tsv", req=FALSE)
PAIRED <- rd("gene_feature_paired.tsv", req=FALSE)
OOF <- MET |> filter(split=="loco_oof")

a1 <- NULL
if(!is.null(PO)){
  po <- PO |> mutate(cls=factor(cls, levels=c("Canonical","cCPA site"),
                                labels=c("real 3' end","competing site")))
  mu <- po |> group_by(cls) |> summarise(m=mean(window_gc), .groups="drop")
  a1 <- ggplot(po, aes(window_gc, fill=cls, colour=cls))+
    geom_density(alpha=0.30, linewidth=0.6)+
    geom_vline(data=mu, aes(xintercept=m, colour=cls), linetype=2,
               linewidth=0.6, show.legend=FALSE)+
    annotate("text", x=Inf, y=Inf, hjust=1.06, vjust=1.6, size=2.9,
             colour=INK, label=sprintf("\u0394 = %.3f", abs(diff(mu$m))))+
    scale_fill_manual(values=c(`real 3' end`=REAL, `competing site`=COMP))+
    scale_colour_manual(values=c(`real 3' end`=REAL, `competing site`=COMP))+
    labs(x="GC content (500 nt)", y="density")+th
}
rg <- range(c(PG$gc_canonical, PG$gc_ccpa_mean))
a2 <- ggplot(PG, aes(gc_ccpa_mean, gc_canonical))+
  geom_hex(bins=48)+
  geom_abline(slope=1, intercept=0, linetype=2, colour=INK, linewidth=0.6)+
  annotate("text", x=rg[1], y=rg[2], hjust=0, vjust=1.4, size=2.9, colour=INK,
           lineheight=1.25,
           label=if(is.null(PFI)) "" else
             sprintf("%.0f%% below the line\n\u0394 = %+.3f",
                     100*PFI$frac_below_diagonal[1], PFI$mean_delta[1]))+
  scale_fill_gradientn(colours=c("#F7FBFF","#C6DBEF","#4292C6","#08306B"),
                       trans="sqrt", name="genes")+
  coord_equal(xlim=rg, ylim=rg)+
  labs(x="GC of competing sites", y="GC of real 3' end")+th+
  theme(legend.position="right", legend.title=element_text(size=BASE-4),
        legend.key.width=unit(3,"mm"), legend.key.height=unit(7,"mm"))
pa <- wrap_elements((a1 | a2) + plot_layout(widths=c(1, 1.1))) +
  labs(title="A   GC content of real and competing sites, pooled and within gene") +
  theme(plot.title=element_text(size=BASE+1, face="bold", hjust=0,
                                margin=margin(b=4)))

gb <- DEC |> filter(metric=="Within-gene AUROC", model %in% PAIR) |>
  mutate(model=fx(model))
endlab <- gb |> group_by(model) |> filter(mean_gc==max(mean_gc)) |> ungroup()
pb <- ggplot(gb, aes(mean_gc, value, colour=model, fill=model))+
  annotate("rect", xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=0.5,
           fill="#B2182B", alpha=0.06)+
  geom_hline(yintercept=0.5, linetype=2, colour=INK, linewidth=0.55)+
  geom_ribbon(aes(ymin=lo, ymax=hi), alpha=0.18, colour=NA)+
  geom_line(linewidth=1.3)+ geom_point(size=2.4)+
  geom_text(data=endlab, aes(label=sprintf("%.2f", value)), hjust=-0.5,
            size=3.0, fontface="bold", show.legend=FALSE)+
  annotate("text", x=-Inf, y=0.5, hjust=-0.06, vjust=-0.6, size=2.8,
           colour=MUT, label="chance")+
  scale_colour_manual(values=PAL)+ scale_fill_manual(values=PAL)+
  scale_y_continuous(limits=c(0.15,1.04), breaks=seq(0.2,1,0.2))+
  scale_x_continuous(expand=expansion(mult=c(0.04,0.13)))+
  labs(title="B   Discrimination across the GC range",
       x="GC content of the gene",
       y="competing sites outranked")+th

gc_ <- DIS |> select(window_gc, score=z_minimal_centred) |>
  filter(is.finite(score))
labc <- DFI |> filter(model=="minimal_centred", cls=="cCPA site")
pc <- ggplot(gc_, aes(window_gc, score))+
  geom_hex(bins=55)+
  geom_smooth(method="lm", se=FALSE, colour="#B2182B", linewidth=0.9)+
  annotate("text", x=Inf, y=Inf, hjust=1.08, vjust=1.7, size=3.4,
           colour=INK, fontface="bold",
           label=sprintf("r\u00b2 = %.2f", labc$r2[1]))+
  scale_fill_gradientn(colours=c("#FFF7EC","#FDD49E","#FC8D59","#B30000"),
                       trans="log10", name="sites")+
  labs(title="C   Splicing score against GC content",
       x="GC content (500 nt)", y="splicing score")+th+
  theme(legend.position="right", legend.title=element_text(size=BASE-4),
        legend.key.width=unit(3,"mm"), legend.key.height=unit(7,"mm"))

pd_ <- NULL
if(!is.null(MF) && nrow(MF)){
  gd <- MF |>
    mutate(short=pf_(feature), short=factor(short, levels=rev(short)),
           dir=ifelse(coef>0,"real 3' end","competing sites"))
  xr <- max(abs(gd$coef))
  pd_ <- ggplot(gd, aes(coef, short, fill=dir))+
    geom_vline(xintercept=0, colour=INK, linewidth=0.45)+
    geom_col(width=0.68)+
    scale_fill_manual(values=c(`real 3' end`=REAL, `competing sites`=COMP))+
    coord_cartesian(xlim=c(-xr*1.15, xr*1.15))+
    labs(title="D   Feature weights", x="weight", y=NULL)+th+
    theme(axis.text.y=element_text(size=BASE-4))
}

pe <- NULL
if(!is.null(LO)){
  ge <- LO |> filter(model %in% PAIR) |> mutate(model=fx(model))
  ordc <- ge |> filter(model==unname(SHORT["minimal_centred"])) |>
    arrange(win_rate) |> pull(chrom)
  ge <- ge |> mutate(chrom=factor(chrom, levels=ordc))
  pe <- ggplot(ge, aes(win_rate, chrom, colour=model))+
    geom_vline(xintercept=0.5, linetype=2, colour=INK, linewidth=0.45)+
    geom_line(aes(group=chrom), colour="grey84", linewidth=0.4)+
    geom_point(size=1.9)+
    scale_colour_manual(values=PAL)+
    labs(title="E   Leave-one-chromosome-out",
         x="competing sites outranked", y=NULL)+th+
    theme(axis.text.y=element_text(size=BASE-6))
}

pf <- NULL
if(!is.null(PAIRED)){
  ordf <- PAIRED |> distinct(feature) |> pull(feature)
  gg <- PAIRED |>
    mutate(lab=factor(pf_(feature), levels=pf_(ordf)),
           block=factor(block, levels=c("real 3' end","competing sites")))
  lim <- as.numeric(quantile(abs(gg$z), 0.99, na.rm=TRUE))
  ann <- PAIRED |> distinct(row, gc_canonical, n_ccpa)
  ybrk <- ann |> slice(round(seq(1, n(), length.out=6)))

  th_s <- th + theme(
    axis.text.x=element_text(angle=90, hjust=1, vjust=0.5, size=BASE-6),
    axis.title.x=element_blank(), axis.ticks=element_blank(),
    panel.grid=element_blank(), plot.title=element_blank(),
    plot.margin=margin(6,1,6,2),
    legend.position="bottom", legend.direction="horizontal",
    legend.key.width=unit(6,"mm"), legend.key.height=unit(2.5,"mm"),
    legend.title=element_text(size=BASE-6), legend.text=element_text(size=BASE-6))

  g_gc <- ggplot(ann, aes(x="GC", y=row, fill=gc_canonical))+
    geom_raster()+
    scale_fill_gradientn(colours=c("#FFF7BC","#FEC44F","#D95F0E","#7F2704"),
                         name="gene GC")+
    scale_y_reverse(breaks=ybrk$row, labels=sprintf("%.2f", ybrk$gc_canonical),
                    expand=c(0,0))+
    scale_x_discrete(expand=c(0,0))+ labs(y="genes")+th_s

  g_n <- ggplot(ann, aes(x="n", y=row, fill=log10(n_ccpa)))+
    geom_raster()+
    scale_fill_gradientn(colours=c("#F7F7F7","#BDBDBD","#636363","#1A1A1A"),
                         name="competitors")+
    scale_y_reverse(expand=c(0,0))+ scale_x_discrete(expand=c(0,0))+
    labs(y=NULL)+th_s+ theme(axis.text.y=element_blank())

  mk <- function(bl) ggplot(gg |> filter(block==bl),
                            aes(lab, row, fill=pmax(pmin(z, lim), -lim)))+
    geom_raster()+
    scale_fill_gradientn(
      colours=c("#7F3B08","#D95F02","#FDB863","#F7F7F7",
                "#92C5DE","#2166AC","#053061"),
      limits=c(-lim, lim), name="feature value")+
    scale_y_reverse(expand=c(0,0))+ scale_x_discrete(expand=c(0,0))+
    labs(x=NULL, y=NULL, subtitle=bl)+th+
    theme(axis.text.x=element_text(angle=45, hjust=1, size=BASE-5),
          axis.text.y=element_blank(), axis.ticks.y=element_blank(),
          panel.grid=element_blank(),
          plot.subtitle=element_text(size=BASE-2, face="bold", colour=INK,
                                     hjust=0.5, margin=margin(b=4)),
          plot.margin=margin(6,4,6,4),
          legend.position="bottom", legend.direction="horizontal",
          legend.key.width=unit(14,"mm"), legend.key.height=unit(2.5,"mm"),
          legend.title=element_text(size=BASE-6),
          legend.text=element_text(size=BASE-6))

  pf <- wrap_elements(
    (g_gc | g_n | mk("real 3' end") | mk("competing sites")) +
      plot_layout(widths=c(0.05, 0.05, 1, 1))) +
    labs(title="F   Feature values at real and competing sites, gene by gene") +
    theme(plot.title=element_text(size=BASE+1, face="bold", hjust=0,
                                  margin=margin(b=4)))
}

# assemble
row1 <- pa
row2 <- (pb | pc) + plot_layout(widths=c(1, 1))
row3 <- if(is.null(pd_)) pe else (pd_ | pe) + plot_layout(widths=c(1.3, 1))
parts <- Filter(Negate(is.null), list(row1, row2, row3, pf))
hts   <- c(1, 1, 0.95, 1.35)[seq_along(parts)]
fig   <- Reduce(`/`, parts) + plot_layout(heights=hts)

HH <- 5.6 * length(parts)
ggsave(paste0(STEM,".pdf"), fig, width=15, height=HH, device=cairo_pdf,
       limitsize=FALSE)
ggsave(paste0(STEM,".png"), fig, width=15, height=HH, dpi=400, limitsize=FALSE)
cat("[out] ", STEM, ".pdf / .png\n", sep="")

cat("\nout-of-fold, all genes:\n")
OOF |> filter(model %in% PAIR) |>
  select(label, win_rate, win_lo, win_hi, mrr, p_at_1, n_genes) |>
  as.data.frame() |> print(row.names=FALSE, digits=3)
cat("\nwithin-gene AUROC in the extreme GC groups:\n")
DEC |> filter(metric=="Within-gene AUROC", model %in% PAIR,
              gc_bin %in% c(min(gc_bin), max(gc_bin))) |>
  select(label, gc_bin, mean_gc, n_genes, value, lo, hi) |>
  as.data.frame() |> print(row.names=FALSE, digits=3)