#!/usr/bin/env Rscript
# two models, one question: can either pick the real 3' end out of a gene's own candidate

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
  library(readr); library(scales); library(forcats)
})
args  <- commandArgs(trailingOnly=TRUE)
INDIR <- if(length(args)>=1) args[1] else "modelcmp"
STEM  <- if(length(args)>=2) args[2] else "figure_model_compare"
rd <- function(f, req=TRUE){
  p <- file.path(INDIR,f)
  if(!file.exists(p)){ if(req) stop("missing ",p); return(NULL) }
  read_tsv(p, show_col_types=FALSE, progress=FALSE)
}

INK<-"#1A1A1A"; BASE<-15
CANON<-"#0072B2"; CC<-"#E69F00"
MCOL <- c(gc_model="#0072B2", splicing_only="#009E73",
          splicing_gcres="#66C2A5", gc_only="#BFC3C7")
MLAB <- c(gc_model="MaxEnt 3'ss + GC contrast",
          splicing_only="Splicing features only",
          splicing_gcres="Splicing, GC-residualised",
          gc_only="Window GC alone")
MSHORT <- c(gc_model="3'ss + GC", splicing_only="Splicing",
            splicing_gcres="Splicing, GC-resid.", gc_only="GC alone")

th <- theme_classic(base_size=BASE)+
  theme(text=element_text(color=INK),
        axis.text=element_text(color=INK,size=BASE-4),
        axis.title=element_text(size=BASE-3),
        axis.line=element_line(color=INK,linewidth=0.45),
        axis.ticks=element_line(color=INK,linewidth=0.45),
        plot.title=element_text(size=BASE+1,face="bold",hjust=0,
                                margin=margin(b=2)),
        plot.title.position="plot",
        legend.position="bottom", legend.title=element_blank(),
        legend.text=element_text(size=BASE-5),
        legend.margin=margin(t=-5,b=0), legend.key.size=unit(0.75,"lines"),
        strip.background=element_blank(),
        strip.text=element_text(face="bold",size=BASE-5,margin=margin(b=2)),
        panel.grid.major=element_line(linewidth=0.2, colour="grey93"),
        plot.margin=margin(3,8,3,6))

M  <- rd("metrics.tsv")
LO <- rd("loco.tsv")
CO <- rd("coefficients.tsv")
GS <- rd("gc_scatter.tsv")
GC <- rd("gc_correlation.tsv")
GR <- rd("gc_r2_per_feature.tsv")
PN <- rd("perm_null.tsv", req=FALSE)
FV <- rd("feature_values.tsv", req=FALSE)

H <- M |> filter(split=="holdout") |>
  mutate(model=factor(model, levels=names(MCOL)))
MAIN <- c("gc_model","splicing_only")

# a: within-gene win rate, the headline
ga <- H |> filter(model %in% MAIN)
nullw <- if(!is.null(PN)) PN |> select(model, null_win_rate_mean) else NULL
pa <- ggplot(ga, aes(win_rate, fct_rev(model), colour=model))+
  annotate("rect", xmin=0.5, xmax=Inf, ymin=-Inf, ymax=Inf,
           fill="grey96", colour=NA)+
  geom_vline(xintercept=0.5, linetype=2, colour=INK, linewidth=0.45)+
  geom_errorbarh(aes(xmin=win_rate_lo, xmax=win_rate_hi), height=0.12,
                 linewidth=0.7)+
  geom_point(size=3.2)+
  geom_text(aes(label=sprintf("%.3f", win_rate)), vjust=-1.2, size=3.0,
            colour=INK)+
  annotate("text", x=0.5, y=Inf, vjust=1.5, hjust=1.05, size=2.7,
           colour="#555555", label="permutation null")+
  scale_colour_manual(values=MCOL, guide="none")+
  scale_y_discrete(labels=MSHORT)+
  coord_cartesian(xlim=c(0.45,0.72))+
  labs(title="a   Within-gene: does the real 3' end outrank the gene's cCPA sites?",
       x="win rate against a random cCPA site in the same gene", y=NULL)+th

# b: pooled metrics, each against its own measured null
gb <- H |> filter(model %in% MAIN) |>
  select(model, auroc, auprc, auprc_baseline) |>
  left_join(if(is.null(PN)) tibble(model=character(), null_auroc_mean=numeric())
            else PN |> select(model, null_auroc_mean), by="model") |>
  pivot_longer(c(auroc,auprc), names_to="metric", values_to="v") |>
  mutate(chance=ifelse(metric=="auroc",
                       ifelse(is.na(null_auroc_mean), 0.5, null_auroc_mean),
                       auprc_baseline),
         excess=v-chance,
         metric=factor(metric, levels=c("auroc","auprc"),
                       labels=c("AUROC","AUPRC")),
         model=factor(model, levels=MAIN))
pb <- ggplot(gb, aes(v, fct_rev(model), colour=model))+
  geom_segment(aes(x=chance, xend=v, yend=fct_rev(model)), linewidth=1.6,
               alpha=0.35, lineend="round")+
  geom_point(aes(x=chance), shape=124, size=4.5, colour=INK)+
  geom_point(size=3.0)+
  geom_text(aes(label=sprintf("+%.3f", excess)), vjust=-1.2, size=2.8,
            colour=INK)+
  facet_wrap(~metric, nrow=1, scales="free_x")+
  scale_colour_manual(values=MCOL, guide="none")+
  scale_y_discrete(labels=MSHORT)+
  labs(title="b   Pooled, measured against each model's own null (tick)",
       x="held-out value", y=NULL)+th

# c: per-chromosome loco
gc_ <- LO |> filter(model %in% MAIN) |>
  mutate(model=factor(model, levels=names(MCOL)))
ord <- gc_ |> filter(model=="gc_model") |> arrange(auroc) |> pull(chrom)
gc_ <- gc_ |> mutate(chrom=factor(chrom, levels=ord))
med <- gc_ |> group_by(model) |> summarise(m=median(auroc), .groups="drop")
pc <- ggplot(gc_, aes(auroc, chrom, colour=model))+
  geom_vline(data=med, aes(xintercept=m, colour=model), linetype=2,
             linewidth=0.45, show.legend=FALSE)+
  geom_point(size=1.9, alpha=0.9)+
  scale_colour_manual(values=MCOL, labels=MLAB)+
  labs(title="c   Leave-one-chromosome-out",
       x="held-out AUROC (dashed = median)", y=NULL)+th+
  theme(axis.text.y=element_text(size=BASE-7))

# d: is splicing just gc?
gd <- H |> mutate(model=factor(model, levels=names(MCOL)))
r2max <- if(!is.null(GR)) max(GR$r2_on_gc, na.rm=TRUE) else NA_real_
sub_d <- if(is.na(r2max)) "" else
  sprintf("largest single-feature R\u00b2 on GC = %.2f", r2max)
pd_ <- ggplot(gd, aes(auroc, fct_rev(model), fill=model))+
  geom_col(width=0.6)+
  geom_vline(xintercept=0.5, linetype=2, colour=INK, linewidth=0.45)+
  geom_text(aes(label=sprintf("%.3f", auroc)), hjust=-0.18, size=2.9,
            colour=INK)+
  scale_fill_manual(values=MCOL, guide="none")+
  scale_y_discrete(labels=MSHORT)+
  scale_x_continuous(limits=c(0, max(gd$auroc)*1.22),
                     expand=expansion(mult=c(0,0)))+
  labs(title="d   Splicing features are not a GC detector",
       subtitle=sub_d, x="held-out AUROC (pooled)", y=NULL)+th+
  theme(plot.subtitle=element_text(size=BASE-5, colour="#555555",
                                   margin=margin(b=3)))

ge <- GS |> select(is_canonical, window_gc, score_gc_model, score_splicing_only) |>
  pivot_longer(starts_with("score_"), names_to="model", values_to="score") |>
  mutate(model=sub("^score_","",model),
         model=factor(model, levels=MAIN, labels=MSHORT[MAIN]),
         cls=ifelse(is_canonical, "Canonical", "cCPA site"))
lab_e <- GC |> filter(model %in% MAIN) |>
  mutate(model=factor(model, levels=MAIN, labels=MSHORT[MAIN])) |>
  group_by(model) |>
  summarise(lab=paste(sprintf("%s  r = %+.3f", cls, r), collapse="\n"),
            .groups="drop")
pe <- ggplot(ge, aes(window_gc, score, colour=cls))+
  geom_point(size=0.2, alpha=0.07)+
  geom_smooth(method="loess", se=FALSE, linewidth=0.8, span=0.6)+
  geom_text(data=lab_e, aes(x=Inf, y=Inf, label=lab), inherit.aes=FALSE,
            hjust=1.04, vjust=1.3, size=2.6, colour=INK, lineheight=1.1)+
  facet_wrap(~model, nrow=1, scales="free_y")+
  scale_colour_manual(values=c(Canonical=CANON, `cCPA site`=CC))+
  guides(colour=guide_legend(override.aes=list(size=2.2, alpha=1)))+
  labs(title="e   Model score against window GC",
       x="GC fraction, 500 nt window", y="held-out linear predictor")+th

gf <- GR |> arrange(desc(r2_on_gc)) |> head(12) |>
  mutate(feature=factor(feature, levels=rev(feature)))
pf <- ggplot(gf, aes(r2_on_gc, feature))+
  geom_col(fill="#BFC3C7", width=0.64)+
  geom_text(aes(label=sprintf("%.2f", r2_on_gc)), hjust=-0.2, size=2.5,
            colour=INK)+
  scale_x_continuous(limits=c(0, max(gf$r2_on_gc)*1.3),
                     expand=expansion(mult=c(0,0)))+
  labs(title="f   How much of each splicing feature is GC?",
       x=expression(R^2~"of the feature regressed on window GC"), y=NULL)+th+
  theme(axis.text.y=element_text(size=BASE-7))

gg <- CO |> filter(model=="splicing_only") |>
  mutate(absc=abs(coef)) |> arrange(desc(absc)) |> head(12) |>
  mutate(feature=factor(feature, levels=rev(feature)),
         dir=ifelse(coef>0, "higher at canonical", "higher at cCPA"))
pg <- ggplot(gg, aes(coef, feature, fill=dir))+
  geom_col(width=0.64)+
  geom_vline(xintercept=0, colour=INK, linewidth=0.45)+
  scale_fill_manual(values=c(`higher at canonical`=CANON,
                             `higher at cCPA`=CC))+
  labs(title="g   Splicing model, standardised coefficients",
       x="coefficient", y=NULL)+th+
  theme(axis.text.y=element_text(size=BASE-7))

ph_ <- NULL
if(!is.null(FV)){
  gh <- FV |> mutate(cls=factor(cls, levels=c("Canonical","cCPA site")))
  ph_ <- ggplot(gh, aes(value, fill=cls, colour=cls))+
    geom_density(alpha=0.28, linewidth=0.5)+
    facet_wrap(~feature, scales="free", ncol=4)+
    scale_fill_manual(values=c(Canonical=CANON, `cCPA site`=CC))+
    scale_colour_manual(values=c(Canonical=CANON, `cCPA site`=CC))+
    labs(title="h   Top features, canonical vs cCPA",
         x="feature value", y="density")+th+
    theme(axis.text=element_text(size=BASE-7))
}

row1 <- (pa | pb)  + plot_layout(widths=c(1, 1.25))
row2 <- (pd_ | pc) + plot_layout(widths=c(1.15, 1))
row3 <- (pe | pf)  + plot_layout(widths=c(1.3, 1))
if(is.null(ph_)){
  row4 <- pg
  hts  <- c(0.72, 1, 1, 1)
  HH   <- 15
} else {
  row4 <- (pg | ph_) + plot_layout(widths=c(1, 1.4))
  hts  <- c(0.72, 1, 1, 1.15)
  HH   <- 16.5
}
fig <- row1 / row2 / row3 / row4 + plot_layout(heights=hts)

HH <- if(is.null(ph_)) 15 else 16.5
ggsave(paste0(STEM,".pdf"), fig, width=16, height=HH, device=cairo_pdf,
       limitsize=FALSE)
ggsave(paste0(STEM,".png"), fig, width=16, height=HH, dpi=400, limitsize=FALSE)
cat("[out] ", STEM, ".pdf / .png\n", sep="")

cat("\nheld-out, all four models:\n")
H |> select(model, win_rate, win_rate_lo, win_rate_hi, auroc, auprc,
            auprc_baseline, p_at_1, mrr, n_genes) |>
  as.data.frame() |> print(row.names=FALSE, digits=3)
if(!is.null(PN)){
  cat("\nwithin-gene permutation nulls:\n")
  PN |> select(model, null_win_rate_mean, null_auroc_mean) |>
    as.data.frame() |> print(row.names=FALSE, digits=3)
  cat("\nP@1 and MRR are omitted from the figure: the GC-only model scores best\n",
      "on both while sitting at chance on AUROC, which is a tie artifact.\n", sep="")
}