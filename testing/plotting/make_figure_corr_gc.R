#!/usr/bin/env Rscript
# gc confound check for the corrected-negative-set reranker
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(patchwork)})
INK<-"#1A1A1A"; CAN<-"#0072B2"; COMP<-"#E69F00"; BASE<-13
th<-theme_classic(base_size=BASE)+theme(
  plot.title=element_text(face="bold",size=BASE,colour=INK,hjust=0),
  axis.text=element_text(colour=INK),axis.title=element_text(colour=INK),
  legend.position="none",plot.margin=margin(6,10,6,8))
P<-read.delim("rerank_corr_gc_points.tsv")
S<-readLines("rerank_corr_gc.tsv")
val<-function(k){v<-grep(paste0("^#",k,"\t"),S,value=TRUE); as.numeric(strsplit(v[1],"\t")[[1]][2])}
R<-read.delim("rerank_corr_gc.tsv",comment.char="#")
rr<-function(l) R$pearson[R$quantity==l]
P$class<-factor(P$class,levels=c("competitor","canonical"))
P<-P[order(P$class),]
CO<-c(canonical=CAN,competitor=COMP)

pa<-ggplot(P,aes(gc,discriminant,colour=class))+
  geom_point(size=.25,alpha=.12)+
  geom_smooth(method="lm",formula=y~x,se=FALSE,linewidth=.8)+
  scale_colour_manual(values=CO)+
  scale_x_continuous(limits=c(.15,.75))+
  labs(title=sprintf("GC explains %.1f percent of discriminant variance",100*val("resid_r2")),
       x="GC fraction, 500 nt window ending at the pA",y="held-out discriminant")+th+
  annotate("text",x=.72,y=max(P$discriminant),hjust=1,vjust=1,size=3.4,colour=INK,
           label=sprintf("all peaks r = %+.3f\ncanonical r = %+.3f\ncompetitor r = %+.3f",
                         rr("discriminant vs GC, all peaks"),
                         rr("discriminant vs GC, canonical only"),
                         rr("discriminant vs GC, competitor only")))

q<-quantile(P$gc,seq(0,1,length.out=11)); q[1]<-q[1]-1e-9
B<-P %>% mutate(b=cut(gc,q,labels=FALSE)) %>% group_by(b,class) %>%
  summarise(gc=mean(gc),m=mean(discriminant),se=sd(discriminant)/sqrt(n()),n=n(),.groups="drop")
pb<-ggplot(B,aes(gc,m,colour=class))+
  geom_errorbar(aes(ymin=m-1.96*se,ymax=m+1.96*se),width=0,linewidth=.5)+
  geom_line(linewidth=.7)+geom_point(size=2)+
  scale_colour_manual(values=CO)+
  labs(title="Canonical scores hold flat, competitor scores fall",
       x="GC decile, mean GC of the decile",y="mean held-out discriminant")+th+
  annotate("text",x=min(B$gc),y=max(B$m),hjust=0,vjust=1,size=3.4,colour=CAN,label="canonical")+
  annotate("text",x=min(B$gc),y=min(B$m),hjust=0,vjust=0,size=3.4,colour=COMP,label="competitor")

LEG<-data.frame(k=factor(c("canonical (real 3' end)","competitor (model false positive)"),
                         levels=c("canonical (real 3' end)","competitor (model false positive)")),v=1)
pleg<-ggplot(LEG,aes(k,v,fill=k))+geom_col()+
  scale_fill_manual(values=setNames(c(CAN,COMP),levels(LEG$k)),name=NULL)+
  coord_cartesian(xlim=c(10,11))+theme_void()+
  theme(legend.position="bottom",legend.text=element_text(size=BASE-2,colour=INK),
        legend.key.size=unit(9,"pt"))

fig<-wrap_plots(wrap_plots(pa,pb,ncol=2),pleg,ncol=1,heights=c(1,.07))+
  plot_annotation(title="Composition does not carry the corrected-set reranker",
                  theme=theme(plot.title=element_text(face="bold",size=BASE+2,colour=INK)))
ggsave("figure_corr_gc.pdf",fig,width=11,height=4.6,device=grDevices::cairo_pdf)
ggsave("figure_corr_gc.png",fig,width=11,height=4.6,dpi=300)
cat("wrote figure_corr_gc.pdf / .png\n")
