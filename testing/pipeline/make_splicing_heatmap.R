#!/usr/bin/env Rscript
# two models, heatmap-led

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
  library(readr); library(scales); library(forcats)
})
args  <- commandArgs(trailingOnly=TRUE)
INDIR <- if(length(args)>=1) args[1] else "splmodel"
STEM  <- if(length(args)>=2) args[2] else "figure_splicing_heatmap"
rd <- function(f, req=TRUE){
  p <- file.path(INDIR,f)
  if(!file.exists(p)){ if(req) stop("missing ",p); return(NULL) }
  read_tsv(p, show_col_types=FALSE, progress=FALSE)
}

INK<-"#1A1A1A"; BASE<-15
CANON<-"#0072B2"; CC<-"#E69F00"; SPL<-"#009E73"; GCCOL<-"#8A8F98"

th <- theme_classic(base_size=BASE)+
  theme(text=element_text(color=INK),
        axis.text=element_text(color=INK,size=BASE-4),
        axis.title=element_text(size=BASE-3),
        axis.line=element_line(color=INK,linewidth=0.45),
        axis.ticks=element_line(color=INK,linewidth=0.45),
        plot.title=element_text(size=BASE+1,face="bold",hjust=0,
                                margin=margin(b=2)),
        plot.title.position="plot",
        plot.subtitle=element_text(size=BASE-5,colour="#555555",
                                   margin=margin(b=4)),
        legend.position="right", legend.title=element_text(size=BASE-5),
        legend.text=element_text(size=BASE-6),
        legend.key.size=unit(0.8,"lines"),
        strip.background=element_blank(),
        strip.text=element_text(face="bold",size=BASE-4,margin=margin(b=3)),
        panel.grid.major=element_line(linewidth=0.2, colour="grey93"),
        plot.margin=margin(3,8,3,6))
th_hm <- th + theme(panel.grid=element_blank(),
                    axis.line=element_blank(), axis.ticks=element_blank())

MET <- rd("hm_metrics_by_decile.tsv")
DEN <- rd("hm_rank_density.tsv")
STR <- rd("hm_gene_strip.tsv")
OV  <- rd("hm_overall.tsv")
WG  <- rd("gc_within_gene.tsv", req=FALSE)
PO  <- rd("gc_pooled.tsv", req=FALSE)

MLEV <- c("gc_only","splicing_centred")
MLAB <- c(gc_only="Window GC", splicing_centred="Splicing (gene-centred)")
fixm <- function(x) factor(x, levels=MLEV, labels=MLAB[MLEV])

# a: gene-level metrics by gc decile, diverging at chance
ga <- MET |> mutate(model=fixm(model),
                    metric=factor(metric,
                                  levels=c("Within-gene AUROC","MRR","P@1")),
                    gc_bin=factor(gc_bin, levels=1:10))
mid <- c(`Within-gene AUROC`=0.5, MRR=NA, `P@1`=NA)
pa <- ggplot(ga, aes(model, fct_rev(gc_bin), fill=value))+
  geom_tile(colour="white", linewidth=0.9)+
  geom_text(aes(label=sprintf("%.2f", value),
                colour=abs(value-0.5) > 0.22),
            size=2.6, show.legend=FALSE)+
  facet_wrap(~metric, nrow=1)+
  scale_colour_manual(values=c(`TRUE`="white", `FALSE`="grey15"))+
  scale_fill_gradient2(low="#B2182B", mid="#F7F7F7", high="#2166AC",
                       midpoint=0.5, limits=c(0,1), name="value")+
  labs(title="a   Gene-level performance, split by how GC-rich the gene is",
       subtitle="rows are deciles of the canonical site's window GC, decile 1 most GC-poor; red = below chance",
       x=NULL, y="GC decile of the gene")+th_hm+
  theme(axis.text.x=element_text(angle=20, hjust=1))

# b: where the canonical site ranks
gb <- DEN |> mutate(model=fixm(model),
                    gc_bin=factor(gc_bin, levels=1:10),
                    rank_bin=factor(rank_bin, levels=1:10))
pb <- ggplot(gb, aes(rank_bin, fct_rev(gc_bin), fill=frac))+
  geom_tile(colour="white", linewidth=0.6)+
  facet_wrap(~model, nrow=1)+
  scale_fill_gradient(low="#F7F7F7", high="#08306B", labels=percent,
                      name="genes")+
  scale_x_discrete(labels=function(x) ifelse(x %in% c("1","5","10"), x, ""))+
  labs(title="b   Where the real 3' end lands among its own gene's candidates",
       subtitle="column 1 = the canonical site scored top in its gene; each row sums to 100%",
       x="rank of the canonical site (decile within its gene)",
       y="GC decile of the gene")+th_hm

# c: the raw per-gene texture
pc <- NULL
if(!is.null(STR)){
  gc_ <- STR |>
    select(gene, gc_canonical, gc_bin,
           gc_only=gc_only_rank_pct, splicing_centred=splicing_centred_rank_pct) |>
    pivot_longer(c(gc_only, splicing_centred), names_to="model",
                 values_to="rank_pct") |>
    mutate(model=fixm(model))
  ordg <- STR |> arrange(gc_canonical) |> pull(gene)
  gc_ <- gc_ |> mutate(gene=factor(gene, levels=ordg))
  pc <- ggplot(gc_, aes(model, gene, fill=rank_pct))+
    geom_tile()+
    scale_fill_gradient2(low="#2166AC", mid="#F7F7F7", high="#B2182B",
                         midpoint=0.5, limits=c(0,1), labels=percent,
                         name="rank of\ncanonical\n(0 = top)")+
    scale_y_discrete(breaks=NULL)+
    labs(title="c   Every held-out gene, sorted by GC",
         subtitle="one row per gene, most GC-poor at the bottom; blue = the canonical site ranked near the top",
         x=NULL, y="genes, sorted by canonical-site GC \u2192")+th_hm+
    theme(axis.text.x=element_text(angle=20, hjust=1))
}

# d: overall gene-level metrics
gd <- OV |> mutate(model=fixm(model),
                   metric=factor(metric,
                                 levels=c("Within-gene AUROC","MRR","P@1")))
pd_ <- ggplot(gd, aes(value, fct_rev(metric), colour=model))+
  geom_vline(xintercept=0.5, linetype=2, colour=INK, linewidth=0.45)+
  geom_errorbar(aes(xmin=lo, xmax=hi), width=0.16, linewidth=0.7,
                orientation="y", position=position_dodge(width=0.5))+
  geom_point(size=2.8, position=position_dodge(width=0.5))+
  geom_text(aes(label=sprintf("%.3f", value)), vjust=-1.3, size=2.7,
            position=position_dodge(width=0.5), show.legend=FALSE)+
  scale_colour_manual(values=setNames(c(GCCOL, SPL), MLAB[MLEV]))+
  labs(title="d   Overall, averaged across all held-out genes",
       subtitle="the averages are close; panel a shows they are reached in completely different ways",
       x="value, 95% CI", y=NULL)+th+
  theme(legend.position="bottom", legend.title=element_blank())

pe <- NULL
if(!is.null(WG)){
  frac <- mean(WG$delta_gc < 0)
  rg <- range(c(WG$gc_canonical, WG$gc_ccpa_mean))
  pe <- ggplot(WG, aes(gc_ccpa_mean, gc_canonical))+
    geom_abline(slope=1, intercept=0, linetype=2, colour=INK, linewidth=0.5)+
    geom_point(size=0.45, alpha=0.20, colour=SPL)+
    annotate("text", x=-Inf, y=Inf, hjust=-0.07, vjust=1.5, size=2.9,
             colour=INK,
             label=sprintf("%.1f%% of genes below the diagonal\nmean shift %+.3f",
                           100*frac, mean(WG$delta_gc)))+
    coord_equal(xlim=rg, ylim=rg)+
    labs(title="e   Why GC works at all: within a gene the real 3' end is GC-poor",
         subtitle="pooled, the two classes differ by 0.006 GC; per gene, by 0.037",
         x="mean GC of that gene's cCPA sites",
         y="GC of the canonical site")+th
}

row1 <- pa
row2 <- pb
row3 <- if(is.null(pc)) pd_ else (pc | pd_) + plot_layout(widths=c(1, 1.15))
parts <- Filter(Negate(is.null), list(row1, row2, row3, pe))
hts <- c(1.15, 1.0, 1.15, if(!is.null(pe)) 1.0)
fig <- Reduce(`/`, parts) + plot_layout(heights=hts)

HH <- 4.3 * length(parts)
ggsave(paste0(STEM,".pdf"), fig, width=14, height=HH, device=cairo_pdf,
       limitsize=FALSE)
ggsave(paste0(STEM,".png"), fig, width=14, height=HH, dpi=400, limitsize=FALSE)
cat("[out] ", STEM, ".pdf / .png\n", sep="")

cat("\ngene-level metrics, held-out genes:\n")
OV |> select(label, metric, value, lo, hi, n_genes) |>
  as.data.frame() |> print(row.names=FALSE, digits=3)
cat("\nwithin-gene AUROC in the extreme GC deciles:\n")
MET |> filter(metric=="Within-gene AUROC", gc_bin %in% c(1,10)) |>
  select(label, gc_bin, mean_gc, n_genes, value, lo, hi) |>
  as.data.frame() |> print(row.names=FALSE, digits=3)