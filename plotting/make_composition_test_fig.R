#!/usr/bin/env Rscript
# one figure, three small panels: auroc / auprc / mrr by feature block, with chance lines
suppressMessages({library(ggplot2);library(dplyr);library(readr);library(tidyr);library(patchwork)})
INK<-"#1A1A1A"; BASE<-13; CAN<-"#0072B2"; COMP<-"#E69F00"; NUL<-"#8A8F98"
D<-read_tsv("hypothesis_test_results.tsv",show_col_types=FALSE)
ORD<-c("COMP","COMP_fine","PAS","ARCH","SPLICE","COMP+PAS","COMP+SPLICE","COMP_fine+SPLICE","COMP+PAS+ARCH","ALL")
D<-D[D$block %in% ORD,]
D$block<-factor(D$block,levels=rev(ORD))
th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-4),axis.line=element_line(color=INK,linewidth=.4),
  plot.title=element_text(size=BASE,face="bold"),plot.subtitle=element_text(size=BASE-5,color="#555555"))
bar<-function(val,lo,hi,ref,ttl,sub,xlab,fmt){
  ggplot(D,aes(.data[[val]],block))+
    geom_vline(xintercept=ref,linetype=2,color=NUL,linewidth=.5)+
    geom_errorbar(aes(xmin=.data[[lo]],xmax=.data[[hi]]),width=.25,linewidth=.5,color=CAN)+
    geom_point(size=3,color=CAN)+
    geom_text(aes(label=sprintf(fmt,.data[[val]])),hjust=-0.25,size=2.7,color=INK)+
    scale_x_continuous(expand=expansion(mult=c(.08,.16)))+
    labs(title=ttl,subtitle=sub,x=xlab,y=NULL)+th
}
pA<-bar("auroc","auroc_lo","auroc_hi",0.5,
        "A   AUROC by block","Dashed = chance 0.5. PAS alone is above 0.5: the hexamer predicts the real end, not the competitor.","AUROC (held-out)"," %.3f")
pB<-bar("auprc","auprc_lo","auprc_hi",D$auprc_baseline[1],
        "B   AUPRC by block",sprintf("Dashed = %.3f base rate.",D$auprc_baseline[1]),"AUPRC (held-out)"," %.3f")
pC<-bar("mrr","mrr_lo","mrr_hi",D$mrr_chance[1],
        "C   MRR by block",sprintf("Dashed = chance %.3f. Reciprocal rank of the real end within its gene.",D$mrr_chance[1]),"MRR (held-out)"," %.3f")
NTC<-c(A="#1B9E77",C="#D95F02",G="#7570B3",T="#CC0066")
CF<-read_tsv("comp_fine_coefs.tsv",show_col_types=FALSE)
CF<-CF[CF$bin_center<200,]
CF$nt<-factor(CF$nt,levels=c("A","C","G","T"))
pD<-ggplot(CF,aes(bin_center,coef,color=nt))+
  geom_hline(yintercept=0,linetype=2,color=NUL,linewidth=.4)+
  geom_vline(xintercept=0,linetype=3,color=INK,linewidth=.4)+
  geom_line(linewidth=.7)+geom_point(size=1.3)+
  scale_color_manual(values=NTC,name="nt")+
  scale_x_continuous(breaks=seq(-1000,150,250))+
  labs(title="D   Where the compositional signal lives",
       subtitle="Per-bin standardised coefficient (COMP_fine: mono-nt freq in 50-nt bins, canonical positive). Signal sits in [-250,+50] around the cleavage site. Dotted = pA; +200..+250 bin dropped (edge artifact).",
       x="position relative to pA (nt)",y="standardised coefficient")+th
WA<-read_tsv("window_auroc.tsv",show_col_types=FALSE)
pk<-WA$win_center[which.max(WA$auroc)]
pE<-ggplot(WA,aes(win_center,auroc))+
  geom_hline(yintercept=0.5,linetype=2,color=NUL,linewidth=.5)+
  geom_vline(xintercept=0,linetype=3,color=INK,linewidth=.4)+
  geom_line(linewidth=.7,color=CAN)+geom_point(size=2,color=CAN)+
  geom_point(data=WA[which.max(WA$auroc),],size=3.2,color=COMP)+
  scale_x_continuous(breaks=seq(-1000,250,250))+
  labs(title="E   Where the signal lives (proper localiser)",
       subtitle=sprintf("Held-out AUROC of a separate mono-only model per 100-nt window. Peak at %+d nt (orange). Dashed = chance 0.5; dotted = pA.",as.integer(pk)),
       x="window center relative to pA (nt)",y="held-out AUROC (per-window)")+th
cap<-paste0("Logistic ridge, standardised features, 5-fold CV split BY GENE (all peaks of a gene in one fold; composition leaks locus identity otherwise). ",
  "AUROC/AUPRC pooled over folds; MRR gene-averaged. Bootstrap CI over genes.\nHypothesis (PAS hexamer is the only discrete element; splice = composition in disguise): supported if COMP alone matches ALL and SPLICE adds nothing over COMP. ",
  "Panel D = coefficients of one joint mono-only fit (weak localiser); panel E = separate per-window models (proper localiser).")
fig<-(pA|pB|pC)/(pD|pE)+plot_layout(heights=c(1,0.9))+plot_annotation(
  title="Is every non-PAS signal just base composition? Feature-block ablation on 70,723 contested peaks",
  caption=cap,theme=theme(plot.title=element_text(face="bold",size=BASE+1,color=INK),
    plot.caption=element_text(size=BASE-6,color="#666666",hjust=0)))
ggsave("figure_composition_test.pdf",fig,width=15,height=9.4,device=cairo_pdf)
ggsave("figure_composition_test.png",fig,width=15,height=9.4,dpi=200)
cat("[out] figure_composition_test.pdf/.png\n")
