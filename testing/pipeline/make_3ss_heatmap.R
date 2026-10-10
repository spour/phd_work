#!/usr/bin/env Rscript
# where in (distance, 3'ss strength) space does a peak look like a true end?
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork);library(scales)})
args<-commandArgs(trailingOnly=TRUE)
INDIR<-if(length(args)>=1) args[1] else "."
PFX  <-if(length(args)>=2) args[2] else "fig3_3ss"
rd<-function(f){p<-file.path(INDIR,f); if(file.exists(p)) read_tsv(p,show_col_types=FALSE) else NULL}

INK<-"#1A1A1A"; BASE<-15
th<-theme_classic(base_size=BASE)+
  theme(text=element_text(color=INK),axis.text=element_text(color=INK,size=BASE-4),
        axis.line=element_line(color=INK,linewidth=.5),
        plot.title=element_text(size=BASE+1,face="bold",hjust=0),
        plot.subtitle=element_text(size=BASE-5,color="#555555"),
        legend.key.height=unit(1.1,"lines"),legend.title=element_text(size=BASE-5))

ax<-rd(paste0(PFX,"_axes.tsv"))
centered<-!is.null(ax) && "centered" %in% names(ax) && ax$centered[1]==1
xlab<-if(centered) "Distance to strongest upstream 3\u2032ss (nt, gene-centered)" else "Distance to strongest upstream 3\u2032ss (nt)"
ylab<-if(centered) "3\u2032ss strength (MaxEntScan, gene-centered)" else "3\u2032ss strength (MaxEntScan)"

E<-rd(paste0(PFX,"_enrich.tsv"))
if(is.null(E)) stop("missing ",PFX,"_enrich.tsv — rerun heatmap_3ss.py")

lim<-quantile(abs(E$log2_enrich),0.98,na.rm=TRUE)
E$clipped<-pmax(pmin(E$log2_enrich,lim),-lim)

pA<-ggplot(E,aes(x,y,fill=clipped))+
  geom_tile()+
  scale_fill_gradient2(low="#E69F00",mid="#F7F7F7",high="#0072B2",midpoint=0,
                       limits=c(-lim,lim),
                       name=expression(log[2]~frac(P(canonical~"|"~bin),P(canonical))),
                       labels=label_number(accuracy=0.1))+
  labs(title="A",
       subtitle="Blue = a peak here looks like a true end; orange = looks like a competitor",
       x=xlab,y=ylab)+th

M<-E%>%group_by(y)%>%summarise(nc=sum(n_canon),nt=sum(n),.groups="drop")%>%
  mutate(frac=(nc+0.5)/(nt+1))
base<-sum(E$n_canon)/sum(E$n)
pB<-ggplot(M,aes(y,frac))+
  geom_hline(yintercept=base,linetype=2,color="#8A8F98")+
  annotate("text",x=min(M$y),y=base,label="base rate",hjust=0,vjust=-0.6,size=3,color="#555555")+
  geom_line(color="#0072B2",linewidth=1)+geom_point(color="#0072B2",size=1.6)+
  scale_y_continuous(labels=percent_format(accuracy=1))+
  labs(title="B",subtitle="Collapsing distance: stronger acceptor \u2192 more likely a true end",
       x=ylab,y="P(peak is canonical)")+th

fig<-pA/pB+plot_layout(heights=c(1.6,1))
ggsave(file.path(INDIR,paste0(PFX,".pdf")),fig,width=10,height=10,device=cairo_pdf)
ggsave(file.path(INDIR,paste0(PFX,".png")),fig,width=10,height=10,dpi=300)
cat("[out] ",PFX,".pdf / .png\n",sep="")