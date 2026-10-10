#!/usr/bin/env Rscript
# aparent2 roc + pr curves (not bars)
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr);library(readr)})
A <- if(length(commandArgs(TRUE))>=1) commandArgs(TRUE)[1] else "."
cv <- suppressMessages(read_tsv(file.path(A,"cpa_discrim_curves.tsv"),show_col_types=FALSE))
mt <- suppressMessages(read_tsv(file.path(A,"cpa_discrim_auroc_auprc.tsv"),show_col_types=FALSE))
a2 <- mt[mt$score=="APARENT2",]
cv <- cv[cv$score=="APARENT2",]
NAVY<-"#103A5C"; RED<-"#B0453C"; INK<-"#1A1A1A"; B<-16
th<-theme_classic(base_size=B)+theme(text=element_text(color=INK),
  plot.title=element_text(face="bold",size=B),plot.subtitle=element_text(size=B-4,color="grey35"),
  axis.text=element_text(color=INK,size=B-3),legend.position="none")

roc<-cv[cv$curve=="ROC",]
pRoc<-ggplot(roc,aes(x,y))+
  geom_abline(slope=1,intercept=0,linetype=2,color=RED,linewidth=.7)+
  geom_line(color=NAVY,linewidth=1.1)+
  annotate("text",x=.62,y=.30,label=sprintf("AUROC = %.3f",a2$auroc),size=4.6,color=NAVY,fontface="bold")+
  annotate("text",x=.7,y=.62,label="chance",angle=32,size=3.6,color=RED)+
  coord_equal(xlim=c(0,1),ylim=c(0,1))+
  labs(title="ROC",subtitle="canonical vs competitor",x="false positive rate",y="true positive rate")+th

pr<-cv[cv$curve=="PR",]
pPr<-ggplot(pr,aes(x,y))+
  geom_hline(yintercept=a2$auprc_baseline,linetype=2,color=RED,linewidth=.7)+
  geom_line(color=NAVY,linewidth=1.1)+
  annotate("text",x=.5,y=max(pr$y)*.9,label=sprintf("AUPRC = %.3f",a2$auprc),size=4.6,color=NAVY,fontface="bold")+
  annotate("text",x=.75,y=a2$auprc_baseline,label=sprintf("baseline %.3f",a2$auprc_baseline),
           vjust=-.5,size=3.4,color=RED)+
  coord_cartesian(xlim=c(0,1),ylim=c(0,max(pr$y,a2$auprc_baseline)*1.3))+
  labs(title="Precision-Recall",subtitle="canonical prevalence baseline",x="recall",y="precision")+th

fig<-(pRoc|pPr)+plot_annotation(
  title="APARENT2 cannot distinguish canonical from internal competitor peaks",
  subtitle=sprintf("n = %s contested peaks; ROC tracks the diagonal and PR sits at baseline -> CPA score is uninformative for site choice",format(a2$n,big.mark=",")),
  theme=theme(plot.title=element_text(face="bold",size=B+2),plot.subtitle=element_text(size=B-4,color="grey30")))
ggsave(file.path(A,"figure_aparent_auroc.pdf"),fig,width=10,height=5.2,device=cairo_pdf)
ggsave(file.path(A,"figure_aparent_auroc.png"),fig,width=10,height=5.2,dpi=300)
cat("[out] figure_aparent_auroc.pdf/png\n")