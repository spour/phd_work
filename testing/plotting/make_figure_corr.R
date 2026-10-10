#!/usr/bin/env Rscript
# corrected-negative-set reranker, panels a-i
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(tidyr);library(patchwork)})
INK<-"#1A1A1A"; CAN<-"#0072B2"; COMP<-"#E69F00"; GRY<-"#8A8F98"
LITE<-"#56B4E9"; GRN<-"#009E73"; PLM<-"#CC79A7"; VER<-"#D55E00"; BASE<-12
th<-theme_classic(base_size=BASE)+theme(
  plot.title=element_text(face="bold",size=BASE,colour=INK,hjust=0),
  axis.text=element_text(colour=INK),axis.title=element_text(colour=INK),
  legend.position="none",plot.margin=margin(6,9,6,7))
fc<-function(x) formatC(x,format="d",big.mark=",")
rdp<-function(f) read.delim(f,comment.char="#")
hashv<-function(f,k){L<-readLines(f);v<-grep(paste0("^#",k,"\t"),L,value=TRUE);as.numeric(strsplit(v[1],"\t")[[1]][2])}

VOC<-c("full model"=CAN,"three features"=LITE,"binned GC"=GRN,"baseline model"=VER,"chance"=GRY)

A<-rdp("corr_panel_a.tsv")
A$category<-factor(A$category,levels=c("neither","terminal exon only","APA-annotated only","both"))
A$class<-factor(A$class,levels=c("canonical","competitor"))
FILL<-c("neither"="#D9DDE1","terminal exon only"=GRN,"APA-annotated only"=PLM,"both"="#7E4F8F")
pa<-ggplot(A,aes(class,frac,fill=category))+geom_col(width=.62)+
  scale_fill_manual(values=FILL,name=NULL)+
  scale_y_continuous(expand=expansion(mult=c(0,.02)),labels=scales::percent_format(accuracy=1))+
  geom_text(data=A%>%filter(frac>.05),aes(label=fc(n)),position=position_stack(vjust=.5),
            size=2.9,colour="white",fontface="bold")+
  labs(title="a  What the competitor pool is made of",x=NULL,y="fraction of peaks")+
  th+theme(legend.position="right",legend.text=element_text(size=BASE-3),
           legend.key.size=unit(8,"pt"))+
  annotate("text",x=1.5,y=1.06,size=2.9,colour=INK,
           label=sprintf("removed %s APA-annotated and %s terminal exon (%s both)\n%s competitors retained, %s genes lost every competitor",
                         fc(hashv("corr_panel_a.tsv","removed_apa")),
                         fc(hashv("corr_panel_a.tsv","removed_term")),
                         fc(hashv("corr_panel_a.tsv","removed_both")),
                         fc(hashv("corr_panel_a.tsv","retained")),
                         fc(hashv("corr_panel_a.tsv","genes_dropped"))))+
  coord_cartesian(ylim=c(0,1.10),clip="off")

Bd<-rdp("corr_panel_b.tsv")
Bd$group<-factor(Bd$group,levels=c("retained competitor","removed competitor (APA)","canonical"))
Bd$measure<-factor(Bd$measure,levels=c("RPM","PSE"),labels=c("PolyA_DB RPM","PolyA_DB PSE (%)"))
GC3<-c("canonical"=CAN,"removed competitor (APA)"=PLM,"retained competitor"=COMP)
pb<-ggplot(Bd,aes(median,group,colour=group))+
  geom_errorbar(aes(xmin=q25,xmax=q75),orientation="y",width=0,linewidth=.5,alpha=.5)+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),orientation="y",width=0,linewidth=1.6)+
  geom_point(size=2.4)+
  geom_text(aes(label=sprintf("%.2f",median)),vjust=-1.1,size=2.9,show.legend=FALSE)+
  facet_wrap(~measure,scales="free_x")+scale_x_log10()+
  scale_colour_manual(values=GC3)+
  labs(title="b  Removed competitors were used sites",
       x="median with 95% CI, bar shows IQR, log scale",y=NULL)+
  th+theme(strip.background=element_blank(),strip.text=element_text(colour=INK,size=BASE-1),
           axis.text.y=element_text(size=BASE-2))

CU<-read.delim("corr_curves.tsv")
CU$model<-factor(CU$model,levels=c("reranker","three features","baseline model"),
                 labels=c("full model","three features","baseline model"))
MC<-c("full model"=CAN,"three features"=LITE,"baseline model"=VER)
MET<-rdp("corr_metrics.tsv")
au<-function(m) MET$auroc[MET$model==m]; ap<-function(m) MET$auprc[MET$model==m]
pc<-ggplot(CU,aes(fpr,tpr,colour=model))+
  geom_abline(slope=1,intercept=0,linetype="dashed",colour=GRY,linewidth=.4)+
  geom_line(linewidth=.8)+scale_colour_manual(values=MC)+
  labs(title="c  The model separates real ends from competitors",
       x="false positive rate",y="true positive rate")+th+
  annotate("text",x=.97,y=.20,hjust=1,size=3.1,colour=INK,
           label=sprintf("AUROC\nfull %.3f\nthree features %.3f\nbaseline %.3f",
                         au("full_loco_all"),au("k3_loco_all"),au("baseline_rf")))
BR<-MET$base_rate[MET$model=="full_loco_all"]
pd<-ggplot(CU,aes(tpr,precision,colour=model))+
  geom_hline(yintercept=BR,linetype="dashed",colour=GRY,linewidth=.4)+
  geom_line(linewidth=.8)+scale_colour_manual(values=MC)+
  labs(title="d  Precision stays above the base rate",
       x="recall",y="precision")+th+
  annotate("text",x=.97,y=.42,hjust=1,size=3.1,colour=INK,
           label=sprintf("AUPRC\nfull %.3f\nthree features %.3f\nbaseline %.3f\nbase rate %.3f",
                         ap("full_loco_all"),ap("k3_loco_all"),ap("baseline_rf"),BR))+
  annotate("text",x=.03,y=BR,vjust=-.6,hjust=0,size=2.9,colour=GRY,label="base rate")

CH<-read.delim("corr_bychrom.tsv")
CH$chrom<-factor(CH$chrom,levels=CH$chrom[order(CH$p1)])
pe<-ggplot(CH,aes(p1,chrom))+
  geom_segment(aes(x=chance,xend=p1,yend=chrom),colour=GRY,linewidth=.4)+
  geom_point(aes(x=chance),colour=GRY,size=1.6)+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),orientation="y",width=0,linewidth=.5,colour=CAN)+
  geom_point(colour=CAN,size=1.9)+
  labs(title="e  Every chromosome beats chance",
       x="P@1, leave-one-chromosome-out",y=NULL)+th+
  theme(axis.text.y=element_text(size=BASE-3))

CO<-read.delim("corr_coefs.tsv")
CO$fit<-factor(CO$fit,levels=c("three features","full model"))
CO$feature<-factor(CO$feature,levels=rev(c("spl_upstream_max_me3","spl_pas_prox_u1","spl_u1_suppression_up")))
pf<-ggplot(CO,aes(beta_std,feature,colour=fit))+
  geom_vline(xintercept=0,colour=GRY,linewidth=.4)+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),orientation="y",width=0,linewidth=.7,
                position=position_dodge(width=.55))+
  geom_point(size=2.3,position=position_dodge(width=.55))+
  geom_text(aes(label=sprintf("flip %.2f",sign_flip_frac)),size=2.5,vjust=-1.1,
            position=position_dodge(width=.55),show.legend=FALSE)+
  scale_colour_manual(values=c("three features"=LITE,"full model"=CAN))+
  labs(title="f  The acceptor term survives, U1 proximity does not",
       x="standardized coefficient, 95% CI over genes",y=NULL)+th+
  theme(axis.text.y=element_text(size=BASE-3))

P<-read.delim("corr_gc_points.tsv")
P$class<-factor(P$class,levels=c("competitor","canonical")); P<-P[order(P$class),]
GS<-read.delim("corr_gc_stats.tsv")
rr<-function(l) GS$pearson[GS$quantity==l]
pg<-ggplot(P,aes(gc,discriminant,colour=class))+
  geom_point(size=.22,alpha=.10)+
  geom_smooth(method="lm",formula=y~x,se=FALSE,linewidth=.8)+
  scale_colour_manual(values=c(canonical=CAN,competitor=COMP))+
  scale_x_continuous(limits=c(.15,.75))+
  labs(title="g  The three-feature discriminant is flat in GC",
       x="GC fraction, 500 nt window ending at the pA",y="held-out discriminant")+th+
  annotate("text",x=.74,y=max(P$discriminant),hjust=1,vjust=1,size=3.0,colour=INK,
           label=sprintf("canonical r = %+.3f\ncompetitor r = %+.3f",
                         rr("discriminant vs GC, canonical"),rr("discriminant vs GC, competitor")))+
  annotate("text",x=.17,y=min(P$discriminant),hjust=0,vjust=0,size=3.0,colour=CAN,label="canonical")+
  annotate("text",x=.17,y=min(P$discriminant),hjust=0,vjust=-1.4,size=3.0,colour=COMP,label="competitor")

H<-read.delim("corr_panel_h.tsv")
H$predictor<-factor(H$predictor,levels=rev(c("GC, single 500 nt window","GC, 25 binned profile",
  "three features","three features, GC removed","full model")))
HC<-c("GC, single 500 nt window"=GRN,"GC, 25 binned profile"=GRN,"three features"=LITE,
      "three features, GC removed"=LITE,"full model"=CAN)
ph<-ggplot(H,aes(p1,predictor,colour=predictor))+
  geom_vline(aes(xintercept=chance),linetype="dashed",colour=GRY,linewidth=.4)+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),orientation="y",width=0,linewidth=.7)+
  geom_point(size=2.4)+scale_colour_manual(values=HC)+
  geom_text(aes(label=sprintf("AUROC %.3f, win-rate %.3f",auroc,winrate)),
            hjust=0,nudge_y=.28,size=2.7,colour=INK)+
  scale_x_continuous(limits=c(.15,.62))+
  labs(title="h  Binned GC sets the ceiling, one GC window does not",
       x="P@1, dashed line is chance (0.285)",y=NULL)+th+
  theme(axis.text.y=element_text(size=BASE-3))

I<-read.delim("corr_panel_i.tsv")
I$block<-factor(I$block,levels=rev(c("three features (3)","splice-binned (150)","GC-binned (25)")))
I$contrast<-factor(I$contrast,levels=c("canonical - competitor","competitor - intronic decoy"))
IC<-c("canonical - competitor"=CAN,"competitor - intronic decoy"=PLM)
pi_<-ggplot(I,aes(gap,block,colour=contrast))+
  geom_vline(xintercept=0,colour=GRY,linewidth=.4)+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),orientation="y",width=0,linewidth=.7,
                position=position_dodge(width=.55))+
  geom_point(size=2.4,position=position_dodge(width=.55))+
  geom_text(aes(label=sprintf("%+.3f",gap)),size=2.7,vjust=-1.1,
            position=position_dodge(width=.55),show.legend=FALSE)+
  scale_colour_manual(values=IC)+
  scale_x_continuous(limits=c(-.02,.42))+
  labs(title="i  The three features read compartment, not CPA-likeness",
       x="difference in held-out discriminant, 95% CI over genes",y=NULL)+th+
  theme(axis.text.y=element_text(size=BASE-3))+
  annotate("text",x=.40,y=3.34,hjust=1,size=2.9,colour=CAN,label="canonical - competitor")+
  annotate("text",x=.40,y=3.10,hjust=1,size=2.9,colour=PLM,label="competitor - matched intronic decoy")

# shared legend
KEYS<-c("full model","three features","binned GC","baseline model","chance")
LEG<-data.frame(k=factor(KEYS,levels=KEYS),v=1)
pleg<-ggplot(LEG,aes(k,v,fill=k))+geom_col()+
  scale_fill_manual(values=VOC,breaks=KEYS,name=NULL)+
  coord_cartesian(xlim=c(10,11))+theme_void()+
  theme(legend.position="bottom",legend.text=element_text(size=BASE-2,colour=INK),
        legend.key.size=unit(9,"pt"))+guides(fill=guide_legend(nrow=1))

body<-(pa+pb+plot_layout(widths=c(1.15,1)))/(pc|pd|pe)/(pf+pg+plot_layout(widths=c(1,1.1)))/(ph|pi_)
fig<-wrap_plots(body,pleg,ncol=1,heights=c(1,.035))+
  plot_annotation(title="A three-feature splicing model ranks the real 3' end on a corrected competitor set",
                  theme=theme(plot.title=element_text(face="bold",size=BASE+3,colour=INK)))
ggsave("figure_corr.pdf",fig,width=14,height=17,device=grDevices::cairo_pdf,limitsize=FALSE)
ggsave("figure_corr.png",fig,width=14,height=17,dpi=200,limitsize=FALSE)
cat("wrote figure_corr.pdf / .png\n")
