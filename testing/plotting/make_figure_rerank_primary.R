#!/usr/bin/env Rscript
# the 3-feature splicing reranker, unannotated competitors
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(tidyr);library(patchwork)})
INK<-"#1A1A1A"; BASE<-12; CAN<-"#0072B2"; COMP<-"#E69F00"; GRY<-"#8A8F98"
th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-3),axis.title=element_text(size=BASE-2),
  axis.line=element_line(color=INK,linewidth=.4),axis.ticks=element_line(color=INK,linewidth=.4),
  plot.title=element_text(size=BASE,face="bold",hjust=0),
  legend.title=element_blank(),legend.text=element_text(size=BASE-3),legend.key.height=unit(.8,"lines"))
fc<-function(x) formatC(x,big.mark=",",format="d")
rd<-function(f) read_tsv(f,show_col_types=FALSE,comment="#")
LINECOL<-c("reranker"=CAN,"RF baseline"=COMP,"chance"=GRY)
BARCOL<-c("reranker"=CAN,"chance"=GRY)

MET<-rd("rerank_pri_metrics.tsv"); g<-function(m,k) MET[[k]][MET$model==m]
HOLD<-g("primary_hold","p1"); LOCO<-g("primary_loco_all","p1"); BR<-g("primary_loco_all","base_rate")

# a -- roc
CU<-rd("rerank_pri_curves.tsv") %>% mutate(model=factor(model,levels=c("reranker","RF baseline")))
DIAG<-data.frame(x=c(0,1),y=c(0,1))
pa<-ggplot()+
  geom_line(data=DIAG,aes(x,y,color="chance"),linetype=2,linewidth=.5)+
  geom_line(data=CU,aes(fpr,tpr,color=model),linewidth=.9)+
  scale_color_manual(values=LINECOL,breaks=c("reranker","RF baseline","chance"))+
  annotate("text",x=.97,y=.10,hjust=1,size=BASE*0.25,color=INK,
           label=sprintf("AUROC %.3f [%.3f, %.3f]\nRF baseline %.3f",
                         g("primary_loco_all","auroc"),g("primary_loco_all","auroc_lo"),
                         g("primary_loco_all","auroc_hi"),g("rf_baseline","auroc")))+
  labs(title="a   ROC, reranker vs baseline",x="false positive rate",y="true positive rate")+th
# b -- precision-recall
pb<-ggplot()+
  geom_hline(aes(yintercept=BR,color="chance"),linetype=2,linewidth=.5)+
  geom_line(data=CU,aes(tpr,precision,color=model),linewidth=.9)+
  scale_color_manual(values=LINECOL,breaks=c("reranker","RF baseline","chance"))+
  annotate("text",x=.97,y=.95,hjust=1,vjust=1,size=BASE*0.25,color=INK,
           label=sprintf("AUPRC %.3f [%.3f, %.3f]\nRF baseline %.3f\nbase rate %.3f",
                         g("primary_loco_all","auprc"),g("primary_loco_all","auprc_lo"),
                         g("primary_loco_all","auprc_hi"),g("rf_baseline","auprc"),BR))+
  labs(title="b   Precision-recall, base rate 9.8%",x="recall",y="precision")+th
# c -- per chromosome
CH<-rd("rerank_pri_bychrom.tsv") %>%
  mutate(chrom=factor(chrom,levels=chrom[order(as.numeric(sub("chr","",sub("X","23",chrom))))]))
pc<-ggplot(CH,aes(p1,chrom))+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),orientation="y",width=.25,linewidth=.45,color=INK)+
  geom_point(aes(x=chance,color="chance"),size=2.2,shape=18)+
  geom_point(aes(color="reranker"),size=2.3)+
  scale_color_manual(values=LINECOL,breaks=c("reranker","RF baseline","chance"))+
  annotate("text",x=Inf,y=1,hjust=1.04,vjust=-0.6,size=BASE*0.24,color=INK,
           label=sprintf("pooled LOCO P@1 %.3f",LOCO))+
  labs(title="c   P@1 by held-out chromosome",x="P@1 (held out)",y=NULL)+th
# d -- contest difficulty
CO<-rd("rerank_pri_bycontest.tsv") %>% mutate(bucket=factor(bucket,levels=c("1","2","3-4","5-8","9+")))
CL<-CO %>% select(bucket,reranker=p1,chance) %>% pivot_longer(c(reranker,chance),names_to="src",values_to="v")
pd<-ggplot()+
  geom_col(data=CL,aes(bucket,v,fill=src),position=position_dodge(width=.7),width=.62,color=INK,linewidth=.2)+
  geom_errorbar(data=CO,aes(x=bucket,ymin=ci_lo,ymax=ci_hi),width=.14,linewidth=.45,color=INK,
                position=position_nudge(x=.175))+
  geom_text(data=CO,aes(bucket,ci_hi+.045,label=sprintf("%+.2f",lift)),size=BASE*0.26,color=INK)+
  geom_text(data=CO,aes(bucket,-.05,label=fc(n_genes)),size=BASE*0.22,color=INK)+
  scale_fill_manual(values=BARCOL,breaks=c("reranker","chance"))+
  scale_y_continuous(limits=c(-.08,.82),breaks=seq(0,.8,.2),expand=expansion(mult=0))+
  labs(title="d   P@1 by number of competitors",x="competitors in the gene",y="P@1")+th
PN<-rd("rerank_pri_permnull.tsv"); praw<-readLines("rerank_pri_permnull.tsv")
gv<-function(t) as.numeric(sub(paste0("^#",t,"\t"),"",grep(paste0("^#",t,"\t"),praw,value=TRUE)))
OBS<-gv("observed"); PV<-gv("p")
pe<-ggplot(PN)+
  geom_histogram(aes(p1,fill="chance"),bins=30,color="white",linewidth=.2)+
  geom_vline(aes(xintercept=OBS,color="reranker"),linewidth=.9)+
  scale_fill_manual(values=BARCOL,breaks=c("reranker","chance"))+
  scale_color_manual(values=LINECOL,breaks=c("reranker","RF baseline","chance"))+
  annotate("text",x=OBS,y=Inf,vjust=1.6,hjust=1.06,size=BASE*0.25,color=INK,
           label=sprintf("observed %.3f\np %s %.3f",OBS,ifelse(PV<=1/(nrow(PN)+1),"<","="),
                         max(PV,1/(nrow(PN)+1))))+
  labs(title="e   Permutation null",x="P@1 under the null",y="permutations")+th
RK<-rd("rerank_pri_rank.tsv") %>% mutate(category=factor(category,levels=c("1","2","3","4+","last")))
RL<-RK %>% select(category,reranker=observed,chance) %>%
  pivot_longer(c(reranker,chance),names_to="src",values_to="v")
pg<-ggplot()+
  geom_col(data=RL,aes(category,v,fill=src),position=position_dodge(width=.7),width=.62,color=INK,linewidth=.2)+
  geom_text(data=RK,aes(category,pmax(observed,chance)+.022,label=sprintf("%.2fx",observed/chance)),
            size=BASE*0.24,color=INK)+
  scale_fill_manual(values=BARCOL,breaks=c("reranker","chance"))+
  scale_y_continuous(limits=c(0,.47),breaks=seq(0,.4,.1),expand=expansion(mult=0),
                     labels=scales::percent_format(accuracy=1))+
  labs(title="g   Rank of the used 3' end",x="rank within the gene (last takes precedence)",
       y="fraction of genes")+th
CF<-rd("rerank_pri_coefs.tsv") %>% mutate(feature=reorder(feature,abs(beta_std)))
ph<-ggplot(CF,aes(beta_std,feature))+
  geom_vline(xintercept=0,linetype=2,color=GRY,linewidth=.4)+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),orientation="y",width=.2,linewidth=.5,color=INK)+
  geom_point(size=2.6,color=INK)+
  geom_text(aes(x=ci_hi+0.02,label=sprintf("%+.2f  flip %.2f",beta_std,sign_flip_frac)),
            hjust=0,size=BASE*0.22,color=INK)+
  scale_x_continuous(expand=expansion(mult=c(.10,.45)))+
  labs(title="h   Standardized coefficients",x="standardized coefficient",y=NULL)+th
TT<-rd("rerank_pri_termterm.tsv") %>%
  filter(stratum %in% c("primary (unrestricted compartment)","TERM-TERM, all competitors")) %>%
  mutate(xpos=ifelse(stratum=="TERM-TERM, all competitors",2,1))
WD<-TT %>% select(feature,xpos,beta_std) %>% pivot_wider(names_from=xpos,values_from=beta_std,names_prefix="x") %>%
  mutate(retain=sprintf("%.0f%% retained",100*x2/x1),mid=(x1+x2)/2)
pi<-ggplot()+
  geom_hline(yintercept=0,linetype=2,color=GRY,linewidth=.35)+
  geom_line(data=TT,aes(xpos,beta_std,group=feature),color=INK,linewidth=.6)+
  geom_point(data=TT,aes(xpos,beta_std),size=3.0,color=INK)+
  geom_text(data=WD,aes(0.96,x1,label=sprintf("%s   %+.2f",feature,x1)),hjust=1,size=BASE*0.23,color=INK)+
  geom_text(data=WD,aes(2.04,x2,label=sprintf("%+.2f",x2)),hjust=0,size=BASE*0.23,color=INK)+
  geom_text(data=WD,aes(1.5,mid,label=retain),vjust=-1.25,size=BASE*0.23,color=INK)+
  scale_x_continuous(breaks=c(1,2),labels=c("primary","compartment fixed\n(TERM-TERM)"),
                     limits=c(0.30,2.30),expand=c(0,0))+
  labs(title="i   Coefficients with compartment held fixed",x=NULL,y="standardized coefficient")+th+
  theme(axis.line.x=element_blank(),axis.ticks.x=element_blank(),
        axis.text.x=element_text(size=BASE-2,color=INK))+coord_cartesian(clip="off")
PA<-rd("rerank_primary_path.tsv"); KS<-3
PL<-PA %>% transmute(k,p1=hold_p1,lo=hold_lo,hi=hold_hi)
pj<-ggplot(PL,aes(k,p1))+
  geom_hline(aes(yintercept=PA$hold_chance[1],color="chance"),linetype=2,linewidth=.5)+
  geom_ribbon(aes(ymin=lo,ymax=hi),alpha=.18,fill=CAN)+
  geom_line(aes(color="reranker"),linewidth=.9)+geom_point(size=1.9,color=CAN)+
  annotate("point",x=KS,y=PA$hold_p1[PA$k==KS],size=3.8,shape=21,fill="white",color=INK,stroke=.7)+
  annotate("text",x=KS+0.2,y=PA$hold_p1[PA$k==KS],hjust=0,vjust=2.2,size=BASE*0.24,color=INK,
           label=sprintf("k = 3, P@1 %.3f",PA$hold_p1[PA$k==KS]))+
  scale_color_manual(values=LINECOL,breaks=c("reranker","RF baseline","chance"))+
  scale_x_continuous(breaks=PA$k)+
  labs(title="j   P@1 vs model size",x="features in the model",y="P@1 (untouched 20%)")+th

LEG<-data.frame(k=factor(c("reranker","RF baseline","chance"),levels=c("reranker","RF baseline","chance")),v=1)
pleg<-ggplot(LEG,aes(k,v,fill=k))+geom_col()+
  scale_fill_manual(values=LINECOL,breaks=c("reranker","RF baseline","chance"))+
  coord_cartesian(xlim=c(10,11))+theme_void()+
  theme(legend.position="bottom",legend.title=element_blank(),
        legend.text=element_text(size=BASE-2,color=INK),legend.key.height=unit(.9,"lines"))
body<-(pa|pb|pc)/(pd|pe|pg)/(ph|pi|pj) & theme(legend.position="none")
fig<-wrap_plots(body,pleg,ncol=1,heights=c(1,0.045))+
  plot_annotation(title="A 3-feature splicing reranker picks the used 3' end among unannotated competitors",
    theme=theme(plot.title=element_text(face="bold",size=BASE+3,color=INK)))
ggsave("figure_rerank_primary.pdf",fig,width=18,height=13,device=cairo_pdf)
ggsave("figure_rerank_primary.png",fig,width=18,height=13,dpi=180)
cat(sprintf("wrote figure_rerank_primary.pdf / .png  (3 features, holdout P@1 %.4f)\n",HOLD))
