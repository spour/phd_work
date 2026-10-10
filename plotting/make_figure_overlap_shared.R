#!/usr/bin/env Rscript
# the site-overlap figure, on the genes every dataset covers
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(tidyr);library(readr)
                               library(patchwork);library(scales)})
INK<-"#1A1A1A"; NUL<-"#8A8F98"; GTX<-"#5A6068"; FAINT<-"#E3E6E9"; BASE<-7.4
PAL<-c(canonical="#08519C", model_fp="#8A8F98", atlas_3seq="#E8A33D",
       pcf11_dep="#7D3C98", oh_u1amo="#B03A2E")
LAB<-c(canonical="Canonical CPA", model_fp="Internal competitors",
       atlas_3seq="3'-seq IpA atlas", pcf11_dep="PCF11-dependent CPA", oh_u1amo="U1 AMO")
ORD<-c("canonical","model_fp","atlas_3seq","pcf11_dep","oh_u1amo")
fset<-function(x) factor(LAB[x],levels=unname(LAB[ORD]))
th<-theme_minimal(base_size=BASE,base_family="sans")+theme(
  panel.grid.minor=element_blank(),
  panel.grid.major=element_line(colour=FAINT,linewidth=.3),
  plot.title=element_text(size=BASE+1.4,face="bold",colour=INK),
  plot.subtitle=element_text(size=BASE-0.6,colour=GTX,lineheight=1.15),
  axis.text=element_text(colour=INK),axis.title=element_text(colour=GTX,size=BASE-0.4),
  strip.text=element_text(size=BASE-0.5,colour=INK,face="bold",hjust=0),
  legend.key.size=unit(8,"pt"),legend.text=element_text(size=BASE-0.8),
  legend.title=element_text(size=BASE-0.8,colour=GTX),
  plot.tag=element_text(size=BASE+3,face="bold",colour=INK))
rd<-function(f) read_tsv(f,show_col_types=FALSE,progress=FALSE)
DI<-rd("shared_distances.tsv"); CO<-rd("shared_containment.tsv")
HX<-rd("shared_hexamer.tsv");   SI<-rd("overlap_scores_sites.tsv"); RFD<-rd("overlap_rf_dense.tsv")
NG<-nrow(rd("overlap_shared_genes.tsv"))
NS<-SI%>%filter(in_shared==1)%>%count(set)
stopifnot(setequal(NS$set,ORD))
sub<-function(x) sprintf("%s  (%s sites)",LAB[x],comma(NS$n[match(x,NS$set)]))

# a  nearest-neighbour distance
Ap<-DI%>%mutate(src=factor(sub(set_a),levels=sub(ORD)),tgt=fset(set_b))%>%
  group_by(src,tgt)%>%arrange(dist,.by_group=TRUE)%>%
  mutate(f=row_number()/n())%>%ungroup()%>%
  mutate(dist=pmax(dist,1))
pA<-ggplot(Ap,aes(dist,f,colour=tgt))+
  geom_vline(xintercept=50,linetype="22",colour=NUL,linewidth=.35)+
  geom_step(linewidth=.55)+
  facet_wrap(~src,nrow=1)+
  scale_colour_manual(values=setNames(unname(PAL[ORD]),unname(LAB[ORD])),name="nearest site in")+
  scale_x_log10(breaks=c(1,1e2,1e4,1e6),labels=c("1","100","10K","1M"))+
  scale_y_continuous(labels=percent_format(accuracy=1),limits=c(0,1))+
  labs(title="A   Nearest-neighbour distance between site sets",

       x="distance to nearest site (nt, log scale)",y="cumulative fraction of source-set sites")+th+
  theme(legend.position="right",panel.spacing.x=unit(7,"pt"))

# b  containment
Bp<-CO%>%mutate(a=fset(set_a),b=fset(set_b))
pB<-ggplot(Bp,aes(b,a,fill=containment))+
  geom_tile(colour="white",linewidth=1.1)+
  geom_text(aes(label=sprintf("%.3f",containment),
                colour=I(ifelse(containment>.14,"white",INK))),size=2.25)+
  scale_fill_gradient(low="#F4F7FA",high="#08519C",limits=c(0,max(Bp$containment)),
                      name="containment",labels=number_format(accuracy=.01))+
  scale_x_discrete()+scale_y_discrete(limits=rev)+
  labs(title="B   Pairwise containment at 50 nt",

       x=NULL,y=NULL)+th+
  theme(panel.grid.major=element_blank(),
        axis.text.x=element_text(angle=20,hjust=1,size=BASE-1),
        axis.text.y=element_text(size=BASE-1),legend.position="right")

# c  canonical pas
Cp<-HX%>%mutate(s=fset(set))%>%
  select(s,set,Observed=pct_obs,`Displaced 5 kb`=pct_rand,fold)%>%
  pivot_longer(c(Observed,`Displaced 5 kb`),names_to="w",values_to="v")%>%
  mutate(w=factor(w,levels=c("Displaced 5 kb","Observed")))
pC<-ggplot(Cp,aes(v,s,fill=w))+
  geom_col(position=position_dodge(width=.72),width=.62)+
  geom_text(data=Cp%>%filter(w=="Observed"),
            aes(label=sprintf("%.1f%%  %.1f-fold",v,fold)),
            hjust=-0.12,size=2.2,colour=INK,position=position_nudge(y=.17))+
  scale_fill_manual(values=c(`Observed`="#B03A2E",`Displaced 5 kb`=FAINT),name=NULL)+
  scale_x_continuous(limits=c(0,132),breaks=c(0,25,50,75,100),
                     labels=function(x) paste0(x,"%"))+
  scale_y_discrete(limits=rev)+
  labs(title="C   Canonical polyadenylation signal at site centres",

       x="sites carrying a canonical PAS",y=NULL)+th+
  theme(legend.position="bottom",legend.margin=margin(t=-4),
        panel.grid.major.y=element_blank())

# d  what the baselines make of each set
DR<-RFD%>%transmute(set,v=rf_dense_max,model="Random forest, probability")
DC<-SI%>%filter(in_shared==1,!is.na(cnn_pct))%>%
    transmute(set,v=cnn_pct/100,model="CNN, percentile of all windows")
Dp<-bind_rows(DR,DC)%>%mutate(s=fset(set),model=factor(model,
      levels=c("Random forest, probability","CNN, percentile of all windows")))
MED<-Dp%>%group_by(model,s,set)%>%summarise(m=median(v),n=n(),.groups="drop")
pD<-ggplot(Dp,aes(v,s,fill=set,colour=set))+
  geom_violin(scale="width",width=.82,alpha=.30,linewidth=.28)+
  geom_boxplot(width=.16,outlier.shape=NA,alpha=1,linewidth=.3,fill="white",colour=INK)+
  geom_text(data=MED,aes(x=1.02,y=s,label=sprintf("%.2f",m)),hjust=0,size=2.15,colour=INK,
            inherit.aes=FALSE)+
  facet_wrap(~model,nrow=1)+
  scale_fill_manual(values=PAL,guide="none")+scale_colour_manual(values=PAL,guide="none")+
  scale_x_continuous(limits=c(0,1.14),breaks=c(0,.25,.5,.75,1))+
  scale_y_discrete(limits=rev)+
  labs(title="D   Baseline model scores at the sites of each set",

       x="score",y=NULL)+th+theme(panel.grid.major.y=element_blank())

stopifnot(all(sapply(list(pA,pB,pC,pD), function(g) is.null(g$labels$subtitle))))
FIG<-pA/(pB|pC)/pD+plot_layout(heights=c(1,1.05,1.05))+
  plot_annotation(
    title="Concordance of published 3' end datasets with each other and with model calls, on commonly covered genes",
    caption=paste0(
"All panels restricted to the ",comma(NG)," protein-coding genes in which ALL five sets place at least one site. Low overlap between two datasets otherwise has two\n",
"explanations, real disagreement and different gene coverage, and the unrestricted panels cannot separate them. Sites per set after restriction are in panel A.\n",
"D The forest is read from the dense per-nucleotide genome scan (score_genome/out_scores), so all ",comma(nrow(RFD))," sites carry a score; the 73,801-window table\n",
"  used elsewhere covers only 14 to 23% of them and would have compared one model on a seventh of the sites against the other on all of them. The CNN is\n",
"  leave-one-chromosome-out and so out of fold. LOGISTIC REGRESSION IS ABSENT: fig1_scoredist.tsv has LR scores but no coordinates, the per-gene parquet scans\n",
"  that produced them are gone from the tree, and no lr_model.joblib survives, so LR cannot be placed on a genomic site without retraining it.\n",
"  ON THE COMPETITOR ROW. The internal competitors were defined as sites the forest ranks ABOVE the canonical site OF THE SAME GENE, so their low marginal CNN\n",
"  score is not a contradiction of how they were selected. The selection is a WITHIN-GENE statement and panel D is a marginal one, pooled over genes, and the two\n",
"  need not agree: measured within gene on this scope, the best competitor still outscores the canonical site in 71.9% of genes under the forest, while the CNN\n",
"  reverses that to 44.6%. Over all 6,583 genes the out-of-fold CNN places the canonical site first in 57.7%. That reversal is what the CNN was trained to do,\n",
"  so neither model's verdict on this row is independent of how the row was defined; the three external sets are the honest test, and both models rank U1 AMO\n",
"  lowest of them."),
    theme=theme(plot.title=element_text(size=BASE+5,face="bold",colour=INK,family="sans",
                                        margin=margin(b=3)),
                plot.caption=element_text(size=BASE-1.8,colour=GTX,hjust=0,lineheight=1.25,
                                          family="sans")))
ggsave("figure_overlap_shared.pdf",FIG,width=11.6,height=12.2,device=cairo_pdf)
ggsave("figure_overlap_shared.png",FIG,width=11.6,height=12.2,dpi=200)
cat("[out] figure_overlap_shared.pdf figure_overlap_shared.png\n")
