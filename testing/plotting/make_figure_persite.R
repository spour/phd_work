#!/usr/bin/env Rscript
# per-site figure, upstream-only candidate set
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(tidyr);library(patchwork)})
INK<-"#1A1A1A"; CAN<-"#0072B2"; COMP<-"#E69F00"; GRY<-"#8A8F98"
LITE<-"#56B4E9"; GRN<-"#009E73"; PLM<-"#CC79A7"; VER<-"#D55E00"; BASE<-7
th<-theme_classic(base_size=BASE,base_family="sans")+theme(
  plot.title=element_text(face="bold",size=BASE+.5,colour=INK,hjust=0,margin=margin(b=2)),
  axis.text=element_text(colour=INK,size=BASE-.5),axis.title=element_text(colour=INK,size=BASE),
  axis.line=element_line(linewidth=.25),axis.ticks=element_line(linewidth=.25),
  legend.position="none",plot.margin=margin(3,4,3,3))
fc<-function(x) formatC(x,format="d",big.mark=",")
rdp<-function(f) read.delim(f,comment.char="#")
hv<-function(f,k){L<-readLines(f);v<-grep(paste0("^#",k,"\t"),L,value=TRUE);as.numeric(strsplit(v[1],"\t")[[1]][2])}

ex<-data.frame(x=c(0,3.1,6.0),xe=c(1.0,3.9,9.0))
pa<-ggplot()+
  geom_segment(aes(x=0,xend=9,y=2.2,yend=2.2),colour=INK,linewidth=.3)+
  geom_rect(data=ex,aes(xmin=x,xmax=xe,ymin=2.05,ymax=2.35),fill="#D9DDE1",colour=INK,linewidth=.25)+
  annotate("text",x=2.05,y=2.2,label="intron",size=2.0,colour=INK,vjust=-.7)+
  annotate("text",x=7.5,y=2.05,label="terminal exon",size=2.0,colour=INK,vjust=1.4)+
  geom_point(aes(x=8.6,y=2.2),colour=CAN,size=1.5)+
  annotate("text",x=9.3,y=2.45,label="real 3' end",colour=CAN,size=2.0,fontface="bold",hjust=1)+
  geom_point(aes(x=5.2,y=2.2),colour=COMP,size=1.5)+
  annotate("text",x=0,y=2.72,label="model false positive",colour=COMP,size=2.0,fontface="bold",hjust=0)+
  geom_segment(aes(x=1.2,xend=5.05,y=1.55,yend=1.55),colour=GRN,linewidth=.4,
               arrow=arrow(ends="both",length=unit(1.1,"pt")))+
  annotate("text",x=1.2,y=1.25,hjust=0,size=2.0,colour=GRN,
           label="3'ss max, pA-2000..-126")+
  geom_segment(aes(x=1.2,xend=5.05,y=.85,yend=.85),colour=CAN,linewidth=.4,
               arrow=arrow(ends="both",length=unit(1.1,"pt")))+
  annotate("text",x=1.2,y=.55,hjust=0,size=2.0,colour=CAN,label="mean U1, same window")+
  geom_segment(aes(x=4.15,xend=5.05,y=.15,yend=.15),colour=PLM,linewidth=.7)+
  annotate("text",x=1.2,y=-.15,hjust=0,size=2.0,colour=PLM,label="mean U1, pA-437..-126")+
  coord_cartesian(xlim=c(-.1,9.4),ylim=c(-.35,2.95))+
  labs(title="a  What the three features measure")+
  theme_void(base_size=BASE,base_family="sans")+
  theme(plot.title=element_text(face="bold",size=BASE+.5,colour=INK,hjust=0,margin=margin(b=2)),
        plot.margin=margin(3,4,3,3))

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
       x="fraction of the competitor pool",y=NULL)+th+
  annotate("text",x=.95,y=.62,hjust=1,size=2.1,colour=INK,
           label=sprintf("%s kept in %s genes",fc(hv("up_panel_b.tsv","kept")),fc(hv("up_panel_b.tsv","genes"))))

C<-rdp("up_panel_c.tsv")
C$group<-factor(C$group,levels=c("retained competitor","removed competitor (APA)","canonical"))
C$measure<-factor(C$measure,levels=c("RPM","PSE"),labels=c("RPM","PSE (%)"))
GC3<-c("canonical"=CAN,"removed competitor (APA)"=PLM,"retained competitor"=COMP)
pc<-ggplot(C,aes(median,group,colour=group))+
  geom_errorbar(aes(xmin=q25,xmax=q75),orientation="y",width=0,linewidth=.3,alpha=.5)+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),orientation="y",width=0,linewidth=1)+
  geom_point(size=1.3)+
  geom_text(aes(label=sprintf("%.2f",median)),vjust=-1.0,size=2.0,show.legend=FALSE)+
  facet_wrap(~measure,scales="free_x")+scale_x_log10(n.breaks=4)+scale_colour_manual(values=GC3)+
  labs(title="c  Removed were used",x="PolyA_DB, median and IQR",y=NULL)+th+
  theme(strip.background=element_blank(),strip.text=element_text(colour=INK,size=BASE-.5),
        axis.text.y=element_text(size=BASE-1))

CU<-read.delim("up_curves.tsv",comment.char="#")
CU$model<-factor(CU$model,levels=c("three features","GC-binned","baseline model"))
MC<-c("three features"=CAN,"GC-binned"=GRN,"baseline model"=VER)
BR<-hv("up_curves.tsv","base_rate")
pd1<-ggplot(CU,aes(fpr,tpr,colour=model))+
  geom_abline(slope=1,intercept=0,linetype="dashed",colour=GRY,linewidth=.25)+
  geom_line(linewidth=.4)+scale_colour_manual(values=MC)+
  labs(title="d  Per-site discrimination",x="false positive rate",y="true positive rate")+th+
  annotate("text",x=.97,y=.16,hjust=1,size=2.1,colour=INK,
           label=sprintf("AUROC\nthree features %.3f\nGC-binned %.3f\nbaseline %.3f",
                         hv("up_curves.tsv","auroc_three"),hv("up_curves.tsv","auroc_gc"),
                         hv("up_curves.tsv","auroc_rf")))
pd2<-ggplot(CU,aes(tpr,precision,colour=model))+
  geom_hline(yintercept=BR,linetype="dashed",colour=GRY,linewidth=.25)+
  geom_line(linewidth=.4)+scale_colour_manual(values=MC)+
  labs(title="   precision-recall",x="recall",y="precision")+th+
  annotate("text",x=.03,y=BR,vjust=-.5,hjust=0,size=2.0,colour=GRY,
           label=sprintf("base rate %.3f",BR))

E<-read.delim("upstream_blocks.tsv")
E<-E %>% filter(block!="old three features (3)")
E$block<-factor(E$block,levels=rev(c("new three features (3)","GC-binned (25)","splice-binned (150)","union (178)")),
                labels=rev(c("three features (3)","GC-binned (25)","splice-binned (150)","union (178)")))
EC<-c("three features (3)"=CAN,"GC-binned (25)"=GRN,"splice-binned (150)"=LITE,"union (178)"=INK)
EL<-bind_rows(E %>% transmute(block,metric="AUROC",v=auroc,lo=auroc_lo,hi=auroc_hi),
              E %>% transmute(block,metric="win-rate",v=winrate,lo=win_lo,hi=win_hi))
pe<-ggplot(EL,aes(v,block,colour=block))+
  geom_vline(xintercept=.5,linetype="dashed",colour=GRY,linewidth=.25)+
  geom_errorbar(aes(xmin=lo,xmax=hi),orientation="y",width=0,linewidth=.45)+
  geom_point(size=1.3)+facet_wrap(~metric)+scale_colour_manual(values=EC)+
  geom_text(aes(label=sprintf("%.3f",v)),vjust=-1.0,size=2.0)+
  scale_x_continuous(limits=c(.48,.78))+
  labs(title="e  Block comparison",
       x="95% CI over genes",y=NULL)+th+
  theme(strip.background=element_blank(),strip.text=element_text(colour=INK,size=BASE-.5),
        axis.text.y=element_text(size=BASE-1))

CH<-read.delim("up_bychrom.tsv")
CH$chrom<-factor(CH$chrom,levels=CH$chrom[order(CH$auroc)])
pf<-ggplot(CH,aes(auroc,chrom))+
  geom_vline(xintercept=.5,linetype="dashed",colour=GRY,linewidth=.25)+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),orientation="y",width=0,linewidth=.3,colour=CAN)+
  geom_point(colour=CAN,size=1)+
  labs(title="f  Held-out chromosomes",x="AUROC, leave-one-chromosome-out",y=NULL)+th+
  theme(axis.text.y=element_text(size=BASE-2))

CO<-read.delim("upstream_coefs.tsv")
CO$feature<-factor(CO$feature,levels=rev(c("spl_upstream_max_me3","spl_pas_prox_u1","spl_upstream_mean_u1")))
pg1<-ggplot(CO,aes(beta_std,feature))+
  geom_vline(xintercept=0,colour=GRY,linewidth=.25)+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),orientation="y",width=0,linewidth=.45,colour=CAN)+
  geom_point(colour=CAN,size=1.3)+
  geom_text(aes(label=sprintf("%+.3f, flip %.2f",beta_std,sign_flip_frac)),vjust=-1.0,size=2.0,colour=INK)+
  scale_x_continuous(limits=c(-.55,.62))+
  labs(title="g  Coefficients",
       x="standardized coefficient, 95% CI over genes",y=NULL)+th+
  theme(axis.text.y=element_text(size=BASE-1))
PR<-read.delim("u1_gc_profile_wide.tsv") %>% filter(measure=="u1_mean_negdG")
pg2<-ggplot(PR,aes(position,value,colour=class,fill=class))+
  geom_ribbon(aes(ymin=ci_lo,ymax=ci_hi),alpha=.18,colour=NA)+
  geom_line(linewidth=.4)+
  geom_vline(xintercept=0,linetype="dashed",colour=GRY,linewidth=.25)+
  scale_colour_manual(values=c(canonical=CAN,competitor=COMP))+
  scale_fill_manual(values=c(canonical=CAN,competitor=COMP))+
  labs(title="   U1 profile",
       x="position relative to the pA (nt)",y="mean U1 -dG per 100 nt bin")+th+
  annotate("text",x=-1950,y=min(PR$value),hjust=0,vjust=0,size=1.9,colour=INK,
           label="canonical below competitor\nat every position")

I<-read.delim("corr_panel_i.tsv")
I$block<-factor(I$block,levels=rev(c("three features (3)","splice-binned (150)","GC-binned (25)")))
I$contrast<-factor(I$contrast,levels=c("canonical - competitor","competitor - intronic decoy"))
ph<-ggplot(I,aes(gap,block,colour=contrast))+
  geom_vline(xintercept=0,colour=GRY,linewidth=.25)+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),orientation="y",width=0,linewidth=.45,
                position=position_dodge(width=.5))+
  geom_point(size=1.3,position=position_dodge(width=.5))+
  geom_text(aes(label=sprintf("%+.3f",gap)),size=2.0,vjust=-1.0,position=position_dodge(width=.5))+
  scale_colour_manual(values=c("canonical - competitor"=CAN,"competitor - intronic decoy"=PLM))+
  scale_x_continuous(limits=c(-.02,.44))+
  scale_y_discrete(expand=expansion(add=c(.6,1.0)))+
  labs(title="h  Compartment, not CPA-likeness",
       x="difference in discriminant, 95% CI",y=NULL)+th+
  theme(axis.text.y=element_text(size=BASE-1))+
  annotate("text",x=.44,y=3.85,hjust=1,size=2.0,colour=CAN,label="canonical - competitor")+
  annotate("text",x=.44,y=3.55,hjust=1,size=2.0,colour=PLM,label="competitor - intronic decoy")

P<-read.delim("up_gc_points.tsv")
P$class<-factor(P$class,levels=c("competitor","canonical")); P<-P[order(P$class),]
rr<-function(k) hv("up_gc_stats.tsv",k)
pi1<-ggplot(P,aes(gc,discriminant,colour=class))+
  geom_point(size=.12,alpha=.10)+
  geom_smooth(method="lm",formula=y~x,se=FALSE,linewidth=.4)+
  scale_colour_manual(values=c(canonical=CAN,competitor=COMP))+
  scale_x_continuous(limits=c(.15,.75))+
  labs(title="i  Discriminant vs GC",
       x="GC fraction, 500 nt window",y="held-out discriminant")+th+
  annotate("text",x=.74,y=max(P$discriminant),hjust=1,vjust=1,size=2.0,colour=INK,
           label=sprintf("canonical r = %+.3f\ncompetitor r = %+.3f",rr("r_canonical"),rr("r_competitor")))
G2<-read.delim("up_gc_stats.tsv",comment.char="#")
G2$quantity<-factor(G2$quantity,levels=rev(c("GC, single 500 nt window","GC, 25 binned profile",
                                             "three features","three features, GC removed")))
GC2<-c("GC, single 500 nt window"=GRN,"GC, 25 binned profile"=GRN,
       "three features"=CAN,"three features, GC removed"=CAN)
pi2<-ggplot(G2,aes(auroc,quantity,colour=quantity))+
  geom_vline(xintercept=.5,linetype="dashed",colour=GRY,linewidth=.25)+
  geom_errorbar(aes(xmin=auroc_lo,xmax=auroc_hi),orientation="y",width=0,linewidth=.45)+
  geom_point(size=1.3)+scale_colour_manual(values=GC2)+
  geom_text(aes(label=sprintf("%.3f",auroc)),vjust=-1.0,size=2.0,colour=INK)+
  scale_x_continuous(limits=c(.42,.72))+
  labs(title="   GC controls",x="AUROC, 95% CI over genes",y=NULL)+th+
  theme(axis.text.y=element_text(size=BASE-1))

# legend
KEYS<-c("canonical / real 3' end","competitor / model false positive","three-feature model",
        "GC-binned","splice-binned","union","baseline model","chance")
VOC<-c(CAN,COMP,CAN,GRN,LITE,INK,VER,GRY); names(VOC)<-KEYS
LEG<-data.frame(k=factor(KEYS,levels=KEYS),v=1)
pleg<-ggplot(LEG,aes(k,v,fill=k))+geom_col()+
  scale_fill_manual(values=VOC,breaks=KEYS,name=NULL)+
  coord_cartesian(xlim=c(10,11))+theme_void(base_size=BASE,base_family="sans")+
  theme(legend.position="bottom",legend.text=element_text(size=BASE-1.5,colour=INK),
        legend.key.size=unit(4,"pt"))+guides(fill=guide_legend(nrow=1,byrow=TRUE))

body<-(pa|pb|pc)/(pd1|pd2|pe)/(pf|pg1|pg2)/(ph|pi1|pi2)
fig<-wrap_plots(body,pleg,ncol=1,heights=c(1,.04))
ggsave("figure_persite.pdf",fig,width=180,height=232,units="mm",device=grDevices::cairo_pdf)
ggsave("figure_persite.png",fig,width=180,height=232,units="mm",dpi=400)
cat("wrote figure_persite.pdf / .png\n")
