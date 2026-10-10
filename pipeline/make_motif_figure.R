#!/usr/bin/env Rscript
# positional motif density, canonical vs competitor
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr);library(readr)})
A <- if(length(commandArgs(TRUE))>=1) commandArgs(TRUE)[1] else "."
D <- suppressMessages(read_tsv(file.path(A,"motif_density_density.tsv"),show_col_types=FALSE))
S <- suppressMessages(read_tsv(file.path(A,"motif_density_composition.tsv"),show_col_types=FALSE))
NAVY<-"#103A5C"; ORNG<-"#E69F00"; INK<-"#1A1A1A"; B<-14
th<-theme_classic(base_size=B)+theme(text=element_text(color=INK),
  plot.title=element_text(face="bold",size=B),plot.subtitle=element_text(size=B-4,color="grey35"),
  legend.position="top",legend.title=element_blank(),axis.text=element_text(color=INK,size=B-3))
CLS<-c(canonical=NAVY,competitor=ORNG)
elems<-c("PAS","CFIm(UGUA)","U-rich","GU-rich")
mk<-function(e){
  d<-D[D$element==e,]; d$cls<-factor(d$cls,levels=c("canonical","competitor"))
  diff<-S$diff[S$element==e]
  ggplot(d,aes(pos,density,color=cls))+geom_line(linewidth=.9)+
    geom_vline(xintercept=0,linetype=3,color="grey60")+
    scale_color_manual(values=CLS)+
    labs(title=e,subtitle=sprintf("core presence diff = %+.3f",diff),
         x="distance from peak center (nt)",y="density")+th
}
fig<-(mk(elems[1])|mk(elems[2]))/(mk(elems[3])|mk(elems[4]))+
  plot_annotation(title="CPA motif composition: canonical vs competitor peaks",
    theme=theme(plot.title=element_text(face="bold",size=B+2)))
ggsave(file.path(A,"figure_motif.pdf"),fig,width=11,height=8,device=cairo_pdf)
ggsave(file.path(A,"figure_motif.png"),fig,width=11,height=8,dpi=300)
cat("[out] figure_motif.pdf/png\n")