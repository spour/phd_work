#!/usr/bin/env Rscript
# builds one combined figure (panels a-g) in a single layout, shared theme/palette

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
  library(readr);  library(scales)
})
args  <- commandArgs(trailingOnly = TRUE)
INDIR <- if (length(args) >= 1) args[1] else "."
have  <- function(f){ p <- file.path(INDIR,f); if (file.exists(p)) p else NA_character_ }

# shared theme + palette
MODELS <- c("RF","LR","CNN")
MCOL   <- c(RF="#0072B2", LR="#E69F00", CNN="#009E73")
CLSCOL <- c(Canonical="#0072B2", Atlas="#5FA8D3", Background="#C9CCD1")
TCOL   <- c(TE="#0072B2", EX="#56B4E9", IN="#E69F00", DS="#009E73",
            AE="#CC79A7", AI="#D55E00", AU="#8A8F98", IG="#B0B4BA")
INK    <- "#1A1A1A"; BASE <- 17
th <- theme_classic(base_size = BASE) +
  theme(text=element_text(color=INK), axis.text=element_text(color=INK,size=BASE-3),
        axis.title=element_text(size=BASE-1), axis.line=element_line(color=INK,linewidth=0.5),
        axis.ticks=element_line(color=INK,linewidth=0.5),
        plot.title=element_text(size=BASE+2,face="bold",hjust=0),
        plot.subtitle=element_text(size=BASE-4,color="#555555"),
        legend.position="top", legend.title=element_blank(),
        legend.key.size=unit(0.9,"lines"), legend.text=element_text(size=BASE-3),
        strip.background=element_blank(), strip.text=element_text(face="bold",size=BASE-3),
        plot.margin=margin(6,8,6,8))
placeholder <- function(title, msg="awaiting scan output"){
  ggplot()+annotate("text",0,0.15,label=msg,size=4.4,color="#8A8F98")+
    annotate("text",0,-0.05,label="(fills in when data lands)",size=3.4,color="#B0B4BA")+
    lims(x=c(-1,1),y=c(-1,1))+labs(title=title)+th+
    theme(axis.line=element_blank(),axis.text=element_blank(),
          axis.ticks=element_blank(),axis.title=element_blank())
}

# thresholds: read for the record only
THRESH <- c(RF=0.932, LR=0.980, CNN=0.889)
fT <- have("fig1_thresholds.tsv")
if(!is.na(fT)){
  td <- read_tsv(fT, show_col_types=FALSE)
  THRESH <- setNames(td$threshold, td$model)[MODELS]
}

# per-panel sample sizes (annotate each subtitle with n)
NCOUNT <- list()
fN <- have("fig1_counts.tsv")
if(!is.na(fN)){
  nc <- read_tsv(fN, show_col_types=FALSE)
  for(i in seq_len(nrow(nc))) NCOUNT[[nc$panel[i]]] <- list(n=nc$n[i], unit=nc$unit[i])
}
nsub <- function(panel, base){
  if(!is.null(NCOUNT[[panel]])){
    u <- NCOUNT[[panel]]$unit; nn <- format(NCOUNT[[panel]]$n, big.mark=",")
    paste0(base, "  (n = ", nn, " ", u, ")")
  } else base
}

# figure 1

metrics <- tribble(~model,~AUROC,~AUPRC,~F1,
  "RF",0.987,0.851,0.781, "LR",0.983,0.840,0.761, "CNN",0.991,0.923,0.872) |>
  pivot_longer(-model,names_to="metric",values_to="value") |>
  mutate(model=factor(model,levels=MODELS),
         metric=factor(metric,levels=c("AUROC","AUPRC","F1")))
pA <- ggplot(metrics,aes(metric,value,fill=model))+
  geom_col(position=position_dodge(width=0.80),width=0.66)+
  geom_text(aes(label=sprintf("%.3f",value)),position=position_dodge(width=0.80),
            vjust=-0.5,size=2.6,color=INK)+
  scale_fill_manual(values=MCOL)+
  scale_y_continuous(limits=c(0,1.08),breaks=seq(0,1,.25),expand=expansion(mult=c(0,0.02)))+
  labs(title="A",subtitle=nsub("A","Held-out test performance"),x=NULL,y="Score")+th

fB <- have("fig1_corr.tsv")
if(!is.na(fB)){
  d <- read_tsv(fB,show_col_types=FALSE)
  if(!all(c("model_x","model_y","r") %in% names(d))){
    m <- as.matrix(d[,-1]); rownames(m) <- d[[1]]
    d <- as.data.frame(as.table(m)); names(d) <- c("model_x","model_y","r")
  }
  d <- mutate(d, model_x=factor(model_x,levels=MODELS), model_y=factor(model_y,levels=rev(MODELS)))
  pB <- ggplot(d,aes(model_x,model_y,fill=r))+
    geom_tile(color="white",linewidth=1.2)+
    geom_text(aes(label=sprintf("%.2f",r)),size=4.2,color=INK)+
    scale_fill_gradient(low="#F2F6FA",high="#0072B2",limits=c(0,1),name="r")+
    coord_equal()+labs(title="B",subtitle=nsub("B","Per-window score correlation"),x=NULL,y=NULL)+
    th+theme(legend.position="right",legend.title=element_text(size=BASE-4),panel.grid=element_blank())
} else pB <- placeholder("B","per-window correlation")

fC <- have("fig1_topk.tsv")
if(!is.na(fC)){
  d <- read_tsv(fC,show_col_types=FALSE) |> mutate(model=factor(model,levels=c(MODELS,"Combined")))
  ymax <- min(1, max(d$recovery)*1.15)
  pC <- ggplot(d,aes(K,recovery,color=model))+
    geom_line(linewidth=0.9)+geom_point(size=1.8)+
    scale_color_manual(values=c(MCOL,Combined=INK))+
    scale_y_continuous(labels=percent_format(accuracy=1),limits=c(0,ymax),
                       expand=expansion(mult=c(0,0.02)))+
    scale_x_continuous(breaks=pretty(d$K))+
    labs(title="C",subtitle=nsub("C","Top-K canonical recovery"),x="K (top windows)",y="Recovery")+th
} else pC <- placeholder("C","top-K recovery")

fD <- have("fig1_traces.tsv")
if(!is.na(fD)){
  d <- read_tsv(fD,show_col_types=FALSE) |> mutate(model=factor(model,levels=MODELS))
  pD <- ggplot(d,aes(pos,prob))+
    geom_line(aes(color=model),linewidth=0.6,show.legend=FALSE)+
    facet_grid(model~.,scales="free_y")+scale_color_manual(values=MCOL)+
    scale_y_continuous(breaks=c(0,0.5,1))+
    labs(title="D",subtitle=nsub("D","Per-gene score traces (ERP27)"),x="Position in gene (nt)",y="Score")+
    th+theme(strip.text=element_text(face="bold",size=BASE-4))
} else pD <- placeholder("D","ERP27 score traces")

fE <- have("fig1_scoredist.tsv")
if(!is.na(fE)){
  d <- read_tsv(fE,show_col_types=FALSE)
  present <- intersect(names(CLSCOL), unique(d$class))
  d <- d |> mutate(model=factor(model,levels=MODELS), class=factor(class,levels=present))
  pE <- ggplot(d,aes(class,score,fill=class))+
    geom_violin(color=NA,alpha=0.9,scale="width")+
    facet_wrap(~model,nrow=1)+scale_fill_manual(values=CLSCOL[present],guide="none")+
    scale_y_continuous(limits=c(0,1))+
    labs(title="E",subtitle=nsub("E","Score by class"),x=NULL,y="Score")+
    th+theme(axis.text.x=element_text(angle=30,hjust=1),strip.text=element_text(face="bold",size=BASE-4))
} else pE <- placeholder("E","score distributions")

fF <- have("fig1_jaccard.tsv")
if(!is.na(fF)){
  d <- read_tsv(fF,show_col_types=FALSE)
  if(!"pair" %in% names(d)) names(d)[1] <- "pair"
  if(!"jaccard" %in% names(d)) names(d)[2] <- "jaccard"
  d <- mutate(d, pair=reorder(pair,jaccard))
  pF <- ggplot(d,aes(jaccard,pair))+
    geom_col(fill="#0072B2",width=0.62)+
    geom_text(aes(label=sprintf("%.2f",jaccard)),hjust=-0.2,size=3.4,color=INK)+
    scale_x_continuous(limits=c(0,max(d$jaccard)*1.25),expand=expansion(mult=c(0,0.02)))+
    labs(title="F",subtitle=nsub("F","Peak-set overlap (Jaccard)"),x="Jaccard",y=NULL)+th
} else pF <- placeholder("F","pairwise Jaccard")

GCOL <- c(Terminal="#0072B2", Intragenic="#E69F00", Other="#B0B4BA")
fg <- have("figG_binned.tsv")
if(!is.na(fg)){
  bl <- read_tsv(fg, show_col_types=FALSE) |>
    mutate(model=factor(model,levels=MODELS),
           group=factor(group, levels=c("Terminal","Intragenic","Other")))
  gcl <- setNames(levels(bl$group), levels(bl$group))
  gcf <- have("figG_group_counts.tsv")
  if(!is.na(gcf)){
    gc <- read_tsv(gcf, show_col_types=FALSE)
    for(g in names(gcl)){ k <- gc$n[gc$group==g]
      if(length(k)) gcl[g] <- sprintf("%s (n=%s)", g, format(k, big.mark=",")) }
  }
  ann <- NULL
  fgc <- have("figG_corr.tsv")
  if(!is.na(fgc)){
    xleft <- min(bl$med_tpm, na.rm=TRUE)
    ann <- read_tsv(fgc, show_col_types=FALSE) |>
      filter(!is.na(spearman)) |>
      mutate(model=factor(model, levels=MODELS),
             group=factor(group, levels=c("Terminal","Intragenic","Other"))) |>
      arrange(model, group) |>
      group_by(model) |>
      mutate(x=xleft, y=0.20 - 0.070*(as.integer(group)-1),
             lab=sprintf("%s rho=%+.2f", substr(as.character(group),1,4), spearman)) |>
      ungroup()
  }
  pG <- ggplot(bl, aes(med_tpm, mean_score, color=group, fill=group)) +
    geom_ribbon(aes(ymin=lo, ymax=hi), alpha=0.15, color=NA) +
    geom_line(linewidth=1.0) + geom_point(size=1.6) +
    facet_wrap(~model, nrow=1) +
    scale_x_continuous(trans="log10", breaks=c(0.1,1,10,100),
                       labels=c("0.1","1","10","100")) +
    scale_y_continuous(limits=c(0,1)) +
    scale_color_manual(values=GCOL, labels=gcl, name="Site group") +
    scale_fill_manual(values=GCOL, guide="none") +
    labs(title="G", subtitle=nsub("G","Model score vs expression (mean +/- 95% CI per expression bin; rho = Spearman score vs expr)"),
         x="Average expression (TPM, log scale)", y="Mean model score") +
    th + theme(legend.position="right", legend.title=element_text(size=BASE-4))
  if(!is.null(ann))
    pG <- pG + geom_text(data=ann, aes(x=x, y=y, label=lab, color=group),
                         hjust=0, size=3.0, show.legend=FALSE, inherit.aes=FALSE)
} else pG <- placeholder("G","score vs expression by site group")

top <- pA+pB+pC+pD+pE+pF+plot_layout(design="AB\nCD\nEF",heights=c(1,1,1))
fig <- top / pG + plot_layout(heights=c(3, 1.15))
ggsave(file.path(INDIR,"figure1.pdf"),fig,width=15,height=21,device=cairo_pdf,
       limitsize=FALSE)
ggsave(file.path(INDIR,"figure1.png"),fig,width=15,height=21,dpi=400,
       limitsize=FALSE)
cat("[out] figure1.pdf, figure1.png (panels A-G)\n")
cat("peaks are called per gene from that gene's own 3'-end score;",
    "no global threshold line is drawn.\n")
cat("thresholds in fig1_thresholds.tsv (recorded, not plotted):",
    paste(sprintf("%s=%.3f",MODELS,THRESH[MODELS]),collapse=" "),"\n")