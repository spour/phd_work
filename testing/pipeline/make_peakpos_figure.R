#!/usr/bin/env Rscript
# internal-peak position figure from peak_position.py tsvs
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr);library(readr);library(scales)})
A <- if(length(commandArgs(TRUE))>=1) commandArgs(TRUE)[1] else "."
rd <- function(f){p<-file.path(A,f); if(file.exists(p)) suppressMessages(read_tsv(p,show_col_types=FALSE)) else NULL}
PUR<-"#7B5EA7"; BLU<-"#0072B2"; GRY<-"#8A8F98"; INK<-"#1A1A1A"; B<-15
th<-theme_classic(base_size=B)+theme(text=element_text(color=INK),
  plot.title=element_text(face="bold",size=B+1),plot.subtitle=element_text(size=B-4,color="grey30"),
  legend.position="top",legend.title=element_blank(),axis.text=element_text(color=INK,size=B-3))
CLS<-c(canonical=PUR,competitor=BLU)

an<-rd("peak_position_annotation.tsv")
pA<- if(!is.null(an)){
  an$cls<-factor(an$cls,levels=c("canonical","competitor"))
  an$annotation<-factor(an$annotation,levels=c("exon","intron","outside","unknown"))
  ggplot(an,aes(annotation,frac*100,fill=cls))+geom_col(position=position_dodge(.75),width=.68)+
    scale_fill_manual(values=CLS)+scale_y_continuous(expand=expansion(c(0,.05)))+
    labs(title="A  Genomic annotation",x=NULL,y="% of peaks")+th
} else ggplot()+theme_void()

ip<-rd("peak_position_intronpos.tsv")
pB<- if(!is.null(ip)){
  ip$cls<-factor(ip$cls,levels=c("canonical","competitor"))
  ggplot(ip,aes(bin_mid,frac,color=cls))+geom_line(linewidth=1)+geom_point(size=1.4)+
    geom_hline(yintercept=1/20,linetype=2,color=GRY)+
    annotate("text",x=.15,y=1/20,label="uniform",vjust=-.5,size=3.2,color=GRY)+
    scale_color_manual(values=CLS)+
    labs(title="B  Within-intron position",subtitle="0 = 5' end, 1 = 3' end of host intron",
         x="position within intron",y="fraction of peaks")+th
} else ggplot()+theme_void()

gp<-rd("peak_position_genepos.tsv")
pC<- if(!is.null(gp)){
  gp$cls<-factor(gp$cls,levels=c("canonical","competitor"))
  ggplot(gp,aes(bin_mid,frac,color=cls))+geom_line(linewidth=1)+geom_point(size=1.4)+
    scale_color_manual(values=CLS)+
    labs(title="C  Within-gene position",subtitle="0 = TSS, 1 = polyA (strand-aware)",
         x="position within gene",y="fraction of peaks")+th
} else ggplot()+theme_void()

at<-rd("peak_position_atlastype.tsv")
pD<- if(!is.null(at)){
  cc<-at[at$cls=="competitor",]; cc<-cc[order(-cc$frac),]
  cc$atlas_type<-factor(cc$atlas_type,levels=rev(cc$atlas_type))
  ggplot(cc,aes(frac*100,atlas_type))+geom_col(fill=BLU,width=.7)+
    geom_text(aes(label=sprintf("%.1f%%",frac*100)),hjust=-.15,size=3.2)+
    scale_x_continuous(expand=expansion(c(0,.15)))+
    labs(title="D  Atlas site type (competitor)",x="% of competitor peaks",y=NULL)+th
} else ggplot()+theme_void()

fig<-(pA|pB)/(pC|pD)
ggsave(file.path(A,"figure_peakpos.pdf"),fig,width=12,height=9,device=cairo_pdf)
ggsave(file.path(A,"figure_peakpos.png"),fig,width=12,height=9,dpi=300)
cat("[out] figure_peakpos.pdf/png\n")
