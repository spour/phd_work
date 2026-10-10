#!/usr/bin/env Rscript
# panel d: the model's score at true 3' ends against their own competitors, within gene
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr)})
e<-function(...) cat(...,"\n",file=stderr())
INK<-"#1A1A1A"; BASE<-7; TRU<-"#08519C"; CMP<-"#E8A33D"; GREY<-"#9AA0A6"; GTX<-"#5A6068"
S<-read_tsv("score_separation.tsv",show_col_types=FALSE)
sv<-function(k) S$value[S$key==k][1]
D<-read_tsv("score_separation_sites.tsv",show_col_types=FALSE)%>%
  mutate(class=factor(class,levels=c("true","competitor"),
                      labels=c("true 3' end","competitor")))
NG<-as.numeric(sv("n_genes")); WR<-as.numeric(sv("winrate_loco"))
WLO<-as.numeric(sv("winrate_lo")); WHI<-as.numeric(sv("winrate_hi"))
MED<-as.numeric(sv("median_delta")); FP<-as.numeric(sv("frac_positive"))
e(sprintf("[panel] %d genes; win-rate %.4f [%.4f, %.4f]; Wilcoxon V = %s, n = %s, z = %s, p %s",
          NG,WR,WLO,WHI,sv("wilcoxon_V"),sv("wilcoxon_n"),sv("wilcoxon_z"),sv("wilcoxon_p")))
Q<-D%>%group_by(class)%>%summarise(m=median(centred_score),
       lo=quantile(centred_score,.25),hi=quantile(centred_score,.75),n=n(),.groups="drop")
e(sprintf("[panel] gene-centred median: %s %.4f (n=%s)",Q$class,Q$m,format(Q$n,big.mark=",")))
YL<-quantile(D$centred_score,c(.002,.998))
YTOP<-YL[2]+0.50*diff(YL)
p<-ggplot(D,aes(class,centred_score,fill=class,color=class))+
  geom_hline(yintercept=0,linetype=2,color=GREY,linewidth=.4)+
  geom_violin(width=.86,linewidth=.35,alpha=.45,trim=TRUE)+
  geom_boxplot(width=.13,outlier.shape=NA,linewidth=.35,fill="white",color=INK)+
  scale_fill_manual(values=c(`true 3' end`=TRU,competitor=CMP))+
  scale_color_manual(values=c(`true 3' end`=TRU,competitor=CMP))+
  geom_text(data=Q,aes(x=class,y=m,label=sprintf("%+.2f",m)),inherit.aes=FALSE,
            hjust=-0.55,size=2.2,color=INK)+
  annotate("text",x=0.52,y=YTOP,hjust=0,vjust=1.1,size=2.15,color=INK,lineheight=1.35,
    label=paste0("Within-gene win-rate ",sprintf("%.3f",WR)," [",sprintf("%.3f",WLO),", ",
      sprintf("%.3f",WHI),"]\n",
      "Paired Wilcoxon signed-rank on the per-gene difference\n",
      "V = ",format(as.numeric(sv("wilcoxon_V")),big.mark=","),
      ", n = ",format(as.numeric(sv("wilcoxon_n")),big.mark=",")," genes, z = ",
      sprintf("%+.1f",as.numeric(sv("wilcoxon_z"))),", p ",sv("wilcoxon_p"),"\n",
      "median difference ",sprintf("%+.2f",MED),", positive in ",
      sprintf("%.0f%%",100*FP)," of genes"))+
  coord_cartesian(ylim=c(YL[1],YTOP))+
  labs(title="Gene-centred score distributions at real 3' ends and competitors",
       x=NULL,y="gene-centred model score")+
  theme_classic(base_size=BASE)+
  theme(text=element_text(color=INK),
        axis.text=element_text(color=INK,size=BASE-1),
        axis.text.x=element_text(size=BASE),
        axis.line=element_line(color=INK,linewidth=.3),axis.title=element_text(size=BASE),
        plot.title=element_text(size=BASE+1,face="bold",hjust=0,margin=margin(b=5)),
        panel.grid.major.y=element_line(color="#EEF0F3",linewidth=.25),
        legend.position="none")
ggsave("panelD_score_separation.pdf",p,width=4.6,height=3.4,device=grDevices::cairo_pdf)
ggsave("panelD_score_separation.png",p,width=4.6,height=3.4,dpi=300)
e("[test] the annotated test is the PAIRED within-gene signed-rank; no pooled two-sample test",
  "is computed or drawn")
e("[out] panelD_score_separation.pdf panelD_score_separation.png")
