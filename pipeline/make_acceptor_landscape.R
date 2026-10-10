#!/usr/bin/env Rscript
# the strongest upstream 3'ss, two ways
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork);library(scales);library(tidyr)})
args<-commandArgs(trailingOnly=TRUE)
INDIR<-if(length(args)>=1) args[1] else "."
PFX  <-if(length(args)>=2) args[2] else "fig3_landscape"
rd<-function(f){p<-file.path(INDIR,paste0(PFX,"_",f)); if(file.exists(p)) read_tsv(p,show_col_types=FALSE) else NULL}

INK<-"#1A1A1A"; BASE<-14; CAN<-"#0072B2"; COMP<-"#E69F00"; NUL<-"#8A8F98"
th<-theme_classic(base_size=BASE)+
  theme(text=element_text(color=INK),axis.text=element_text(color=INK,size=BASE-3),
        axis.line=element_line(color=INK,linewidth=.5),
        plot.title=element_text(size=BASE+1,face="bold",hjust=0),
        plot.subtitle=element_text(size=BASE-4,color="#555555"),
        legend.position="top",legend.title=element_blank(),legend.text=element_text(size=BASE-4))
ph<-function(t,m) ggplot()+annotate("text",0,0,label=m,size=4,color="#8A8F98")+lims(x=c(-1,1),y=c(-1,1))+labs(title=t)+th+
  theme(axis.line=element_blank(),axis.text=element_blank(),axis.ticks=element_blank(),axis.title=element_blank())

# h: log2-ratio enrichment heatmap (both classes per cell)
R<-rd("ratio.tsv")
if(!is.null(R)){
  R<-R%>%filter(maxscore>=0)
  lim<-quantile(abs(R$log2ratio),0.98,na.rm=TRUE)
  R$clip<-pmax(pmin(R$log2ratio,lim),-lim)
  pH<-ggplot(R,aes(window,maxscore,fill=clip))+
    geom_tile()+
    geom_hline(yintercept=6.5,linetype=2,color=INK,linewidth=.35)+
    scale_fill_gradient2(low=COMP,mid="white",high=CAN,midpoint=0,limits=c(-lim,lim),
                         name=expression(log[2]~frac(canonical,competitor)))+
    scale_x_continuous(expand=c(0,0))+scale_y_continuous(expand=c(0,0))+
    labs(title="A",
         subtitle="Enrichment of the running max: blue = true ends, orange = competitors",
         x="Upstream window (0 to X nt)",y="Strongest 3′ss so far (MaxEntScan)")+th+
    theme(legend.position="right",legend.title=element_text(size=BASE-6))
} else pH<-ph("A","ratio heatmap")

# a: upper-quantile running-max curves
Q<-rd("quantiles.tsv")
if(!is.null(Q)){
  Q<-Q%>%filter(quantile %in% c(0.9,0.95))%>%
    mutate(lab=paste0(cls," p",quantile*100),
           cls=factor(cls,levels=c("Canonical","Competitor")))
  pA<-ggplot(Q,aes(window,value,color=cls,linetype=factor(quantile)))+
    geom_line(linewidth=1)+
    scale_color_manual(values=c(Canonical=CAN,Competitor=COMP))+
    scale_linetype_manual(values=c(`0.9`="solid",`0.95`="22"),name="quantile")+
    labs(title="B",subtitle="Upper-tail strongest acceptor (p90 solid, p95 dashed)",
         x="Upstream window (0 to X nt)",y="Strongest 3\u2032ss (MaxEntScan)")+th+
    guides(linetype=guide_legend(order=2),color=guide_legend(order=1))
} else pA<-ph("A","quantile curves")

# b: p(strong acceptor exists) threshold sweep
Th<-rd("threshold.tsv")
if(!is.null(Th)){
  Th<-Th%>%mutate(cls=factor(cls,levels=c("Canonical","Competitor")),
                  thr=factor(threshold,labels=paste0("\u2265 ",sort(unique(threshold)))))
  pB<-ggplot(Th,aes(window,frac,color=cls))+
    geom_line(linewidth=1)+facet_wrap(~thr,nrow=1)+
    scale_color_manual(values=c(Canonical=CAN,Competitor=COMP))+
    scale_y_continuous(labels=percent_format(accuracy=1))+
    labs(title="D",subtitle="Fraction of peaks with a strong acceptor by window X",
         x="Upstream window (0 to X nt)",y="P(strongest acceptor \u2265 threshold)")+th
} else pB<-ph("B","threshold sweep")

# c: auroc vs window
A<-rd("auroc.tsv")
if(!is.null(A)){
  pC<-ggplot(A,aes(window,auroc))+
    geom_hline(yintercept=0.5,linetype=2,color=NUL)+
    geom_line(color=CAN,linewidth=1)+geom_point(color=CAN,size=1.4)+
    annotate("text",x=max(A$window),y=0.5,label="chance",hjust=1,vjust=-0.6,size=3,color="#555555")+
    labs(title="C",subtitle="Can the strongest upstream acceptor classify true end vs competitor?",
         x="Upstream window (0 to X nt)",y="AUROC")+th+theme(legend.position="none")
} else pC<-ph("C","AUROC vs window")

PS<-rd("paired_summary.tsv")
if(!is.null(PS)){
  PS$agg<-factor(PS$agg,levels=PS$agg)
  D<-PS%>%pivot_longer(c(observed,null),names_to="which",values_to="val")
  pD<-ggplot(D,aes(val,agg,color=which))+
    geom_line(aes(group=agg),color="#CCCCCC",linewidth=3,lineend="round")+
    geom_point(size=4)+
    geom_text(data=PS,aes(x=observed,y=agg,label=sprintf("%.2f",observed)),
              inherit.aes=FALSE,vjust=-1.1,size=3,color=INK)+
    scale_color_manual(values=c(observed=CAN,null=NUL),
                       labels=c(observed="observed",null="order-statistic null"))+
    scale_x_continuous(labels=percent_format(accuracy=1),limits=c(0,1))+
    labs(title="E",subtitle="Within-gene: does the true end's acceptor beat its competitors'?",
         x="fraction of genes canonical > competitor",y=NULL)+th
} else pD<-ph("D","within-gene paired")

fig<-pH/(pA|pC)/pB/pD+plot_layout(heights=c(1.1,1,1,0.9))
ggsave(file.path(INDIR,paste0(PFX,"_summary.pdf")),fig,width=13,height=16,device=cairo_pdf)
ggsave(file.path(INDIR,paste0(PFX,"_summary.png")),fig,width=13,height=16,dpi=300)
cat("[out] ",PFX,"_summary.pdf / .png\n",sep="")