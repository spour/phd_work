#!/usr/bin/env Rscript
# per-site figure, upstream-only candidate set
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(tidyr);library(patchwork)})
INK<-"#1A1A1A"; CAN<-"#0072B2"; COMP<-"#E69F00"; GRY<-"#8A8F98"
LITE<-"#56B4E9"; GRN<-"#009E73"; PLM<-"#CC79A7"; VER<-"#D55E00"; BASE<-7
th<-theme_classic(base_size=BASE,base_family="sans")+theme(
  plot.title=element_text(face="bold",size=BASE+.5,colour=INK,hjust=0,margin=margin(b=2)),
  plot.title.position="plot",
  axis.text=element_text(colour=INK,size=BASE-.5),axis.title=element_text(colour=INK,size=BASE),
  axis.line=element_line(linewidth=.25),axis.ticks=element_line(linewidth=.25),
  strip.background=element_blank(),strip.text=element_text(colour=INK,size=BASE-.5),
  legend.position="none",plot.margin=margin(3,4,3,3))
fc<-function(x) formatC(x,format="d",big.mark=",")
rdp<-function(f) read.delim(f,comment.char="#")
hv<-function(f,k){L<-readLines(f);v<-grep(paste0("^#",k,"\t"),L,value=TRUE);as.numeric(strsplit(v[1],"\t")[[1]][2])}
BLK<-c("acceptor alone (1)"=GRN,"acceptor + gc_contrast (2)"=CAN)

ANCH<- -954
exn<-data.frame(x=c(-2000,-1500),xe=c(-1750,-1150))
pa<-ggplot()+
  geom_vline(xintercept=0,linetype="dashed",colour=GRY,linewidth=.25)+
  geom_segment(aes(x=-2000,xend=500,y=3.1,yend=3.1),colour=INK,linewidth=.3)+
  geom_rect(data=exn,aes(xmin=x,xmax=xe,ymin=2.95,ymax=3.25),fill="#D9DDE1",colour=INK,linewidth=.25)+
  annotate("rect",xmin=-950,xmax=500,ymin=2.95,ymax=3.25,fill="#D9DDE1",colour=INK,linewidth=.25)+
  annotate("text",x=-500,y=3.25,label="terminal exon",size=1.9,colour=INK,vjust=-.5)+
  geom_point(aes(x=0,y=3.1),colour=CAN,size=1.5)+
  annotate("text",x=0,y=2.95,label="pA",size=1.9,colour=CAN,vjust=1.5)+
  annotate("rect",xmin=-2000,xmax=-126,ymin=2.05,ymax=2.35,fill=GRN,alpha=.22)+
  geom_point(aes(x=ANCH,y=2.2),colour=GRN,size=1.4)+
  annotate("text",x=ANCH,y=2.35,label="anchor 3'ss",size=1.9,colour=GRN,vjust=-.6)+
  annotate("text",x=-2000,y=1.66,hjust=0,size=2.0,colour=GRN,
           label="1  max MaxEnt 3'ss,\n    pA-2000..-126")+
  annotate("rect",xmin=-2000,xmax=-438,ymin=.75,ymax=1.05,fill=LITE,alpha=.45)+
  annotate("rect",xmin=-437,xmax=-126,ymin=.75,ymax=1.05,fill=PLM,alpha=.55)+
  annotate("text",x=-1219,y=.9,label="distal",size=1.9,colour=INK)+
  annotate("text",x=-282,y=1.05,label="proximal",size=1.9,colour=INK,vjust=-.6)+
  annotate("text",x=-2000,y=.36,hjust=0,size=2.0,colour=INK,
           label="2  GC, distal minus\n    proximal")+
  scale_x_continuous(breaks=c(-2000,-1000,-437,0,500))+
  coord_cartesian(xlim=c(-2050,550),ylim=c(0,3.6))+
  labs(title="a  Two features",x="position relative to the pA (nt)",y=NULL)+th+
  theme(axis.line.y=element_blank(),axis.ticks.y=element_blank(),axis.text.y=element_blank())

B<-rdp("up_panel_b.tsv")
B$category<-factor(B$category,levels=c("removed: APA-annotated","removed: reference terminal exon",
                                       "removed: downstream of the canonical",
                                       "kept: upstream, unannotated, intronic"))
FILL<-c("removed: APA-annotated"=PLM,"removed: reference terminal exon"=GRN,
        "removed: downstream of the canonical"=VER,"kept: upstream, unannotated, intronic"="#D9DDE1")
pb<-ggplot(B,aes(frac,category,fill=category))+geom_col(width=.62)+
  scale_fill_manual(values=FILL)+
  geom_text(aes(label=fc(n)),hjust=-.12,size=2.1,colour=INK)+
  scale_x_continuous(limits=c(0,.95),expand=expansion(mult=c(0,.02)),labels=scales::percent_format(accuracy=1))+
  scale_y_discrete(labels=function(x) sub("^(removed|kept): ","",x))+
  labs(title="b  Three filters on the competitor pool",
       x="fraction of the competitor pool",y=NULL)+th

CU<-rdp("gc_curves.tsv")
BR<-hv("gc_curves.tsv","base_rate")
pc1<-ggplot(CU,aes(fpr,tpr))+
  geom_abline(slope=1,intercept=0,linetype="dashed",colour=GRY,linewidth=.25)+
  geom_line(linewidth=.4,colour=CAN)+
  scale_x_continuous(n.breaks=3)+scale_y_continuous(n.breaks=3)+
  labs(title="c  Per-site discrimination",x="false positive rate",y="true positive rate")+th
pc2<-ggplot(CU,aes(tpr,precision))+
  geom_hline(yintercept=BR,linetype="dashed",colour=GRY,linewidth=.25)+
  geom_line(linewidth=.4,colour=CAN)+
  scale_x_continuous(n.breaks=3)+scale_y_continuous(n.breaks=3)+
  labs(title=" ",x="recall",y="precision")+th

PT<-read.delim("gc_blocks.tsv") %>%
  filter(block %in% c("acceptor alone (1)","acceptor + gc_contrast (2)"))
E<-bind_rows(
  PT %>% transmute(block,metric="win-rate",value=winrate,lo=win_lo,hi=win_hi,
                   lab=sprintf("%.3f",winrate)),
  PT %>% transmute(block,metric="AUROC excess",value=excess,lo=auroc_lo-null,
                   hi=auroc_hi-null,lab=sprintf("%+.3f",excess)))
E$block<-factor(E$block,levels=rev(c("acceptor alone (1)","acceptor + gc_contrast (2)")))
E$metric<-factor(E$metric,levels=c("win-rate","AUROC excess"))
REF<-data.frame(metric=factor(c("win-rate","AUROC excess"),levels=levels(E$metric)),r=c(.5,0))
DI<-read.delim("d_increment.tsv")
DI$metric<-factor(DI$quantity,levels=levels(E$metric))
DI$lab<-sprintf("%+.3f [%+.3f,%+.3f]\np %s",DI$increment,DI$ci_lo,DI$ci_hi,sub("^= ","= ",DI$p_text))
pd1<-ggplot(E,aes(value,block,colour=block))+
  geom_vline(data=REF,aes(xintercept=r),linetype="dashed",colour=GRY,linewidth=.25)+
  geom_errorbar(aes(xmin=lo,xmax=hi),orientation="y",width=0,linewidth=.45)+
  geom_point(size=1.4)+scale_colour_manual(values=BLK)+
  geom_text(aes(label=lab),vjust=-1.2,size=2.0)+
  geom_text(data=DI,aes(x=-Inf,y=.62,label=lab),hjust=-.05,vjust=1,size=1.85,colour=INK,
            inherit.aes=FALSE)+
  facet_wrap(~metric,scales="free_x")+scale_x_continuous(n.breaks=3)+
  scale_y_discrete(expand=expansion(add=c(1.15,.85)))+
  labs(title="d  Contribution of each component",x=NULL,y=NULL)+th+
  theme(axis.text.y=element_text(size=BASE-1.5))
CO<-read.delim("gc_coefs.tsv")
CO$feature<-factor(CO$feature,levels=rev(c("spl_upstream_max_me3","gc_contrast")),
                   labels=rev(c("MaxEnt 3'ss","GC contrast")))
pd2<-ggplot(CO,aes(beta_std,feature))+
  geom_vline(xintercept=0,linetype="dashed",colour=GRY,linewidth=.25)+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),orientation="y",width=0,linewidth=.45,colour=CAN)+
  geom_point(colour=CAN,size=1.4)+
  geom_text(aes(label=sprintf("%+.3f",beta_std)),vjust=-1.3,size=2.0,colour=INK)+
  scale_x_continuous(limits=c(-.1,.62),n.breaks=3)+
  scale_y_discrete(expand=expansion(add=c(.7,1.0)))+
  labs(title="   standardized coefficients",x="coefficient, 95% CI",y=NULL)+th+
  theme(axis.text.y=element_text(size=BASE-1.5))

PR<-read.delim("gc_profile.tsv")
NP_<-read.delim("gc_profile_n.tsv",comment.char="#")
NLAB<-NP_ %>% group_by(class) %>% summarise(n=max(n),.groups="drop") %>%
  left_join(PR %>% filter(position==min(PR$position)) %>% select(class,y=value),by="class") %>%
  mutate(x=-1980,lab=sprintf("n = %s",fc(n)),y=y+ifelse(class=="canonical",.0045,-.0055))
gwv<-function(k){L<-readLines("gc_windows.tsv");v<-grep(paste0("^#",k,"\t"),L,value=TRUE);as.numeric(strsplit(v[1],"\t")[[1]][2])}
XC<-gwv("crossover")
pe<-ggplot(PR,aes(position,value))+
  annotate("rect",xmin=-2000,xmax=-438,ymin=-Inf,ymax=Inf,fill=LITE,alpha=.16)+
  annotate("rect",xmin=-437,xmax=-126,ymin=-Inf,ymax=Inf,fill=PLM,alpha=.20)+
  geom_ribbon(aes(ymin=ci_lo,ymax=ci_hi,fill=class),alpha=.20,colour=NA)+
  geom_line(aes(colour=class),linewidth=.4)+
  geom_vline(xintercept=XC,linetype="dotted",colour=INK,linewidth=.3)+
  annotate("text",x=XC,y=max(PR$ci_hi),hjust=-.07,vjust=1,size=1.85,colour=INK,
           label=sprintf("classes\ncross at\n%d nt",round(XC)))+
  scale_colour_manual(values=c(canonical=CAN,competitor=COMP))+
  scale_fill_manual(values=c(canonical=CAN,competitor=COMP))+
  scale_x_continuous(breaks=c(-2000,-1000,-437,0,500))+
  geom_text(data=NLAB,aes(x=x,y=y,label=lab,colour=class),hjust=0,size=1.85,inherit.aes=FALSE)+
  labs(title="e  GC profile around the poly(A) site",
       x="position relative to the pA (nt)",y="mean GC fraction per 100 nt bin")+th

FL<-c("gc_contrast","MaxEnt 3'ss")
hv2<-function(f,k){L<-readLines(f);v<-grep(paste0("^#",k,"\t"),L,value=TRUE);as.numeric(strsplit(v[1],"\t")[[1]][2])}
HS<-bind_rows(read.delim("gcc_hist.tsv",comment.char="#") %>% mutate(feature=FL[1]),
              read.delim("me3_hist.tsv",comment.char="#") %>% mutate(feature=FL[2]))
HS$feature<-factor(HS$feature,levels=FL)
HS$signed<-ifelse(HS$class=="canonical",HS$density,-HS$density)
HSTAT<-bind_rows(
  data.frame(feature=FL[1],auroc=hv2("gcc_hist.tsv","auroc"),
             mc=hv2("gcc_hist.tsv","median_canonical"),mk=hv2("gcc_hist.tsv","median_competitor")),
  data.frame(feature=FL[2],auroc=hv2("me3_hist.tsv","auroc"),
             mc=hv2("me3_hist.tsv","median_canonical"),mk=hv2("me3_hist.tsv","median_competitor")))
HSTAT$feature<-factor(HSTAT$feature,levels=FL)
TOP<-HS %>% group_by(feature) %>% summarise(top=max(density),x=min(centre),.groups="drop")
HSTAT<-left_join(HSTAT,TOP,by="feature")
MED<-bind_rows(HSTAT %>% transmute(feature,class="canonical",x=mc,y1=top*.72),
               HSTAT %>% transmute(feature,class="competitor",x=mk,y1=-top*.72))
pf1<-ggplot(HS,aes(centre,signed,fill=class))+
  geom_area(position="identity",alpha=.55,colour=NA)+
  geom_hline(yintercept=0,colour=INK,linewidth=.25)+
  geom_segment(data=MED,aes(x=x,xend=x,y=0,yend=y1,colour=class),
               linetype="dotted",linewidth=.35,inherit.aes=FALSE)+
  scale_fill_manual(values=c(canonical=CAN,competitor=COMP))+
  scale_colour_manual(values=c(canonical=CAN,competitor=COMP))+
  scale_y_continuous(labels=function(x) abs(x))+
  facet_wrap(~feature,scales="free")+
  geom_text(data=HSTAT,aes(x=x,y=top*.98,label=sprintf("AUROC %.3f\nmedians %+.3f, %+.3f",auroc,mc,mk)),
            hjust=0,vjust=1,size=1.85,colour=INK,inherit.aes=FALSE)+
  labs(title="f  Per peak, the classes overlap",x=NULL,y="density, canonical up")+th
PD<-bind_rows(read.delim("gcc_paired.tsv",comment.char="#") %>% mutate(feature=FL[1]),
              read.delim("me3_paired.tsv",comment.char="#") %>% mutate(feature=FL[2]))
PD$feature<-factor(PD$feature,levels=FL)
PD$pos<-pmax(PD$diff,0); PD$neg<-pmin(PD$diff,0)
pfmt<-function(f){e<-hv2(f,"p_exceed"); n<-hv2(f,"nperm")
  if (e==0) sprintf("p < %.0e",1/n) else sprintf("p = %.1e",hv2(f,"p_value"))}
PSTAT<-bind_rows(
  data.frame(feature=FL[1],lab=sprintf("%.3f above zero\n[%.3f,%.3f], %s",
             hv2("gcc_paired.tsv","frac_above"),hv2("gcc_paired.tsv","frac_lo"),
             hv2("gcc_paired.tsv","frac_hi"),pfmt("gcc_paired.tsv")),
             y=max(PD$diff[PD$feature==FL[1]])),
  data.frame(feature=FL[2],lab=sprintf("%.3f above zero\n[%.3f,%.3f], %s",
             hv2("me3_paired.tsv","frac_above"),hv2("me3_paired.tsv","frac_lo"),
             hv2("me3_paired.tsv","frac_hi"),pfmt("me3_paired.tsv")),
             y=max(PD$diff[PD$feature==FL[2]])))
PSTAT$feature<-factor(PSTAT$feature,levels=FL)
pf2<-ggplot(PD,aes(rank))+
  geom_area(aes(y=neg),fill=COMP,alpha=.75,colour=NA)+
  geom_area(aes(y=pos),fill=CAN,alpha=.75,colour=NA)+
  geom_hline(yintercept=0,colour=INK,linewidth=.25)+
  facet_wrap(~feature,scales="free_y")+
  geom_text(data=PSTAT,aes(x=1,y=y*.98,label=lab),hjust=0,vjust=1,size=1.85,colour=INK,
            inherit.aes=FALSE)+
  labs(title="   per gene, the paired difference",
       x="gene, sorted",y="canonical - mean competitor")+th

CH<-read.delim("gc_bychrom.tsv")
CH$chrom<-factor(CH$chrom,levels=CH$chrom[order(CH$excess)])
POOL<-hv("gc_curves.tsv","excess_model")
pg<-ggplot(CH,aes(excess,chrom))+
  geom_vline(xintercept=0,linetype="dashed",colour=GRY,linewidth=.25)+
  geom_vline(xintercept=POOL,linetype="solid",colour=INK,linewidth=.3)+
  geom_errorbar(aes(xmin=ci_lo-null,xmax=ci_hi-null),orientation="y",width=0,linewidth=.3,colour=CAN)+
  geom_point(colour=CAN,size=1)+
  annotate("text",x=POOL,y=nrow(CH)+.9,hjust=.5,vjust=0,size=1.9,colour=INK,
           label=sprintf("pooled %+.3f",POOL))+
  scale_y_discrete(expand=expansion(add=c(.6,1.9)))+
  labs(title="g  Held-out chromosomes",x="AUROC excess over null",y=NULL)+th+
  theme(axis.text.y=element_text(size=BASE-2))

K<-read.delim("gc_scatter.tsv",comment.char="#")
K$class<-factor(K$class,levels=c("competitor","canonical")); K<-K[order(K$class),]
kv<-function(k){L<-readLines("gc_scatter.tsv");v<-grep(paste0("^#",k,"\t"),L,value=TRUE);as.numeric(strsplit(v[1],"\t")[[1]][2])}
ph<-ggplot(K,aes(gc,discriminant,colour=class))+
  geom_point(size=.12,alpha=.10)+
  scale_colour_manual(values=c(canonical=CAN,competitor=COMP))+
  scale_x_continuous(limits=c(.15,.75))+
  labs(title=sprintf("h  Single-window GC explains %.0f%% of the discriminant",100*kv("r2_all")),
       x="GC fraction, 500 nt window",
       y="held-out linear predictor of the model")+th+
  annotate("text",x=.74,y=max(K$discriminant),hjust=1,vjust=1,size=2.0,colour=INK,
           label=sprintf("canonical r = %+.3f\ncompetitor r = %+.3f",
                         kv("r_canonical"),kv("r_competitor")))

# shared legend
KEYS<-c("real 3' end","model false positive","acceptor alone","acceptor + gc_contrast",
        "permutation null")
VOC<-c(CAN,COMP,GRN,CAN,GRY); names(VOC)<-KEYS
LEG<-data.frame(k=factor(KEYS,levels=KEYS),v=1)
pleg<-ggplot(LEG,aes(k,v,fill=k))+geom_col()+
  scale_fill_manual(values=VOC,breaks=KEYS,name=NULL)+
  coord_cartesian(xlim=c(10,11))+theme_void(base_size=BASE,base_family="sans")+
  theme(legend.position="bottom",legend.text=element_text(size=BASE-1.5,colour=INK),
        legend.key.size=unit(4,"pt"))+guides(fill=guide_legend(nrow=1,byrow=TRUE))

body<-((pa|pb|(pc1|pc2))+plot_layout(widths=c(1,1,1.1)))/
      ((pd1|pd2|pe)+plot_layout(widths=c(1.15,.8,1.1)))/
      (pf1|pf2)/
      (pg|ph)
fig<-wrap_plots(body,pleg,ncol=1,heights=c(1,.045))
ggsave("figure_twofeature.pdf",fig,width=180,height=228,units="mm",device=grDevices::cairo_pdf)
ggsave("figure_twofeature.png",fig,width=180,height=228,units="mm",dpi=400)
cat("wrote figure_twofeature.pdf / .png\n")
