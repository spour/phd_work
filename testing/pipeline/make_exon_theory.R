#!/usr/bin/env Rscript
# the intervening-5'ss theory, four ways
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork);library(scales)})
args<-commandArgs(trailingOnly=TRUE)
INDIR<-if(length(args)>=1) args[1] else "."
PFX  <-if(length(args)>=2) args[2] else "fig3"
rd<-function(f){p<-file.path(INDIR,paste0(PFX,"_",f)); if(file.exists(p)) read_tsv(p,show_col_types=FALSE) else NULL}
INK<-"#1A1A1A"; BASE<-13; CAN<-"#0072B2"; COMP<-"#E69F00"
th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),axis.text=element_text(color=INK,size=BASE-4),
  axis.line=element_line(color=INK,linewidth=.4),plot.title=element_text(size=BASE,face="bold",hjust=0),
  plot.subtitle=element_text(size=BASE-5,color="#555555"),legend.position="top",legend.title=element_blank())

# a: interaction heatmap
IN<-rd("interaction.tsv")
if(!is.null(IN)){
  lim<-quantile(abs(IN$log2ratio),.98,na.rm=TRUE); IN$clip<-pmax(pmin(IN$log2ratio,lim),-lim)
  pA<-ggplot(IN,aes(interv_donor,acceptor,fill=clip))+geom_tile()+
    geom_hline(yintercept=6.5,linetype=2,color=INK,linewidth=.3)+
    geom_vline(xintercept=6.5,linetype=2,color=INK,linewidth=.3)+
    scale_fill_gradient2(low=COMP,mid="white",high=CAN,midpoint=0,limits=c(-lim,lim),
                         name=expression(log[2]~frac(canon,comp)))+
    labs(title="A",subtitle="Acceptor \u00d7 best intervening donor (diagonal = two-layer)",
         x="intervening donor 5\u2032ss",y="anchor acceptor 3\u2032ss")+th+
    theme(legend.position="right",legend.title=element_text(size=BASE-6))
} else pA<-ggplot()+theme_void()

# b: asymmetry paired
AP<-rd("asymmetry_paired.tsv")
if(!is.null(AP)){
  med<-median(AP$diff); fpos<-mean(AP$diff>0)
  pB<-ggplot(AP,aes(diff))+geom_vline(xintercept=0,linetype=2,color="#8A8F98")+
    geom_density(fill=CAN,color=CAN,alpha=.25,linewidth=.7)+
    geom_vline(xintercept=med,color=CAN,linewidth=.6)+
    annotate("text",x=med,y=Inf,label=sprintf("median %+.2f  %.0f%%>0",med,fpos*100),hjust=-0.05,vjust=1.6,size=2.8,color=INK)+
    labs(title="B",subtitle="Splice-site asymmetry (acceptor \u2212 intervening donor), within-gene",
         x="canonical \u2212 competitor asymmetry",y="density")+th
} else pB<-ggplot()+theme_void()

# c: intervening 5'ss -- raw count vs length-normalized density (exposes confound)
IVA<-rd("interv_density_auroc.tsv"); IVV<-rd("interv_density_values.tsv")
if(!is.null(IVV)){
  IVV$cls<-factor(IVV$cls,levels=c("Canonical","Competitor"))
  aul<-function(sig){a<-IVA$auroc[IVA$signal==sig]; if(length(a)) sprintf(" (AUROC %.3f)",a[1]) else ""}
  raw<-IVV%>%filter(signal=="n_strong_interv")%>%group_by(cls,value)%>%summarise(n=n(),.groups="drop")%>%
       group_by(cls)%>%mutate(frac=n/sum(n))%>%filter(value<=20)
  pC<-ggplot(raw,aes(value,frac,fill=cls))+geom_col(position="dodge",width=.9)+
    scale_fill_manual(values=c(Canonical=CAN,Competitor=COMP))+scale_y_continuous(labels=percent_format(1))+
    labs(title="C",subtitle=paste0("RAW # strong intervening 5′ss",aul("n_strong_interv")),
         x="# strong intervening 5′ss",y="fraction")+th
} else pC<-ggplot()+theme_void()

# d: length-normalized density -- the confound control
if(!is.null(IVV)){
  den<-IVV%>%filter(signal=="interv_density5")
  lo<-quantile(den$value,.01);hi<-quantile(den$value,.99);den<-den%>%mutate(value=pmax(pmin(value,hi),lo))
  pD<-ggplot(den,aes(value,fill=cls,color=cls))+geom_density(alpha=.25,linewidth=.7)+
    scale_fill_manual(values=c(Canonical=CAN,Competitor=COMP))+scale_color_manual(values=c(Canonical=CAN,Competitor=COMP))+
    labs(title="D",subtitle=paste0("Length-normalized: strong 5′ss / kb",aul("interv_density5")),
         x="strong intervening 5′ss per kb",y="density")+th
} else pD<-ggplot()+theme_void()

fig<-(pA|pB)/(pC|pD)+plot_annotation(
  title="Intervening 5\u2032ss: competitors have MORE (C) but not DENSER (D) donors \u2014 a length effect",
  theme=theme(plot.title=element_text(face="bold",size=BASE+2,hjust=0)))
ggsave(file.path(INDIR,"figure3_intervening.pdf"),fig,width=13,height=10,device=cairo_pdf)
ggsave(file.path(INDIR,"figure3_intervening.png"),fig,width=13,height=10,dpi=300)
cat("[out] figure3_intervening.pdf / .png\n")