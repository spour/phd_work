#!/usr/bin/env Rscript
# the sequence cnn against the locked 26-feature splice model
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(tidyr)
                               library(patchwork)})
e<-function(...) cat(...,"\n",file=stderr())
INK<-"#1A1A1A"; BASE<-7
CNNC<-"#08519C"; SPLC<-"#E8A33D"; GREY<-"#9AA0A6"; GTX<-"#5A6068"; MUTE<-"#C6CACF"
CHANCE<-"#8A8F98"

PO<-read_tsv("figCS_pooled.tsv",show_col_types=FALSE)
RC<-read_tsv("figCS_recall.tsv",show_col_types=FALSE)
SH<-read_tsv("figCS_shuffle.tsv",show_col_types=FALSE)
PA<-read_tsv("figCS_partition.tsv",show_col_types=FALSE)
NG<-PO$n_genes[1]; NP<-PO$n_peaks[1]; BR<-PO$base_rate[1]

th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-1),axis.line=element_line(color=INK,linewidth=.3),
  axis.title=element_text(size=BASE),axis.ticks=element_line(color=INK,linewidth=.3),
  plot.title=element_text(size=BASE+1,face="bold",hjust=0,margin=margin(b=2)),
  plot.subtitle=element_text(size=BASE-1,color=GTX,hjust=0,margin=margin(b=5)),
  legend.position="none")

# a: pooled, for reference only
SHORT<-c("Sequence CNN"="Sequence CNN","Splice model (26 features)"="Splice model, 26 feat.",
         "Acceptor 3'ss alone"="Acceptor 3'ss","Window GC alone"="GC, central 500 nt")
A<-PO%>%mutate(lab=factor(SHORT[model],levels=rev(unname(SHORT))),
               met=factor(ifelse(metric=="auroc","Pooled AUROC","Pooled AUPRC"),
                          levels=c("Pooled AUROC","Pooled AUPRC")),
               fill=ifelse(model=="Sequence CNN",CNNC,
                    ifelse(model=="Splice model (26 features)",SPLC,MUTE)))
BRL<-data.frame(met=factor("Pooled AUPRC",levels=levels(A$met)),v=BR)
CHL<-data.frame(met=factor("Pooled AUROC",levels=levels(A$met)),v=0.5)
pA<-ggplot(A,aes(lab,value,fill=I(fill)))+
  geom_col(width=.62)+
  geom_errorbar(aes(ymin=ci_lo,ymax=ci_hi),width=.16,linewidth=.3,color=INK)+
  geom_hline(data=CHL,aes(yintercept=v),linetype=2,color=CHANCE,linewidth=.35)+
  geom_hline(data=BRL,aes(yintercept=v),linetype=2,color=CHANCE,linewidth=.35)+
  geom_text(aes(y=ci_hi+.035,label=sprintf("%.3f",value)),size=1.95,color=INK,hjust=0)+
  facet_wrap(~met,nrow=1,scales="free_x")+coord_flip(ylim=c(0,1.02))+
  labs(title="A   Pooled discrimination, for reference only",
       subtitle=paste0("Pooled metrics are inflated by between-gene structure; the within-gene ",
                       "comparison in panel B is the headline.\nDashed: chance 0.5 for AUROC, ",
                       "base rate ",sprintf("%.3f",BR)," for AUPRC. Bars are 95% CI over genes."),
       x=NULL,y=NULL)+
  th+theme(strip.background=element_blank(),strip.text=element_text(size=BASE-1,color=GTX,
           hjust=0,margin=margin(b=2)),panel.spacing.x=unit(9,"pt"))

# b: within-gene recall@k
stopifnot(nrow(RC)==20)
R1<-RC%>%filter(k==1)
r1c<-R1$recall[R1$model=="Sequence CNN"]; r1s<-R1$recall[R1$model!="Sequence CNN"]
stopifnot(abs(r1c-0.64757)<1e-4, abs(r1s-0.47197)<1e-4)
CHDF<-RC%>%filter(model=="Sequence CNN")%>%select(k,chance)
rc<-function(k) approx(RC$k[RC$model=="Sequence CNN"],RC$recall[RC$model=="Sequence CNN"],k)$y
rs<-function(k) approx(RC$k[RC$model!="Sequence CNN"],RC$recall[RC$model!="Sequence CNN"],k)$y
ch<-function(k) approx(CHDF$k,CHDF$chance,k)$y
pB<-ggplot(RC,aes(k,recall,color=I(ifelse(model=="Sequence CNN",CNNC,SPLC))))+
  geom_ribbon(aes(ymin=ci_lo,ymax=ci_hi,
                  fill=I(ifelse(model=="Sequence CNN",CNNC,SPLC))),alpha=.14,color=NA)+
  geom_line(data=CHDF,aes(k,chance),inherit.aes=FALSE,linetype=2,color=CHANCE,linewidth=.4)+
  geom_line(linewidth=.55)+geom_point(size=.85)+
  annotate("point",x=1,y=r1c,size=1.9,shape=21,fill="white",color=CNNC,stroke=.55)+
  annotate("point",x=1,y=r1s,size=1.9,shape=21,fill="white",color=SPLC,stroke=.55)+
  annotate("text",x=1.18,y=r1c,label=sprintf("recall@1 = %.3f",r1c),size=2.05,color=CNNC,
           hjust=0,vjust=0.1)+
  annotate("text",x=1.18,y=r1s,label=sprintf("recall@1 = %.3f",r1s),size=2.05,color=SPLC,
           hjust=0,vjust=0.9)+
  annotate("text",x=6,y=rc(6)+.038,label="Sequence CNN",size=2.15,color=CNNC,hjust=.5)+
  annotate("text",x=6.35,y=rs(6)-.058,label="Splice model, 26 features",size=2.15,color=SPLC,
           hjust=.5)+
  annotate("text",x=8.4,y=ch(8.4)-.048,label="chance",size=2.05,color=CHANCE,hjust=.5)+
  scale_x_continuous(breaks=1:10)+
  scale_y_continuous(limits=c(0.25,1.04),breaks=seq(.3,1,.1))+
  labs(title="B   Within-gene recall of the real 3' end, by rank threshold",
       subtitle=paste0("Same ",format(NG,big.mark=",")," held-out genes and the same chromosome ",
        "holdout for both models. Rank counts ties against the model,\nso recall@1 is the strict ",
        "P@1. Chance is the mean over genes of min(k, candidates)/candidates."),
       x="Rank threshold k",y="Fraction of genes with the real end in the top k")+th

# c: dinucleotide shuffle
SH2<-SH%>%mutate(model=factor(model,levels=c("Sequence CNN","Splice model (51 features)")),
                 state=factor(state,levels=c("Real sequence","Dinucleotide shuffled")))
ANN<-SH2%>%group_by(model)%>%summarise(top=max(ci_hi),drop=first(drop),pct=first(pct_lost),
                                       .groups="drop")
pC<-ggplot(SH2,aes(model,winrate,fill=I(ifelse(state=="Real sequence",
      ifelse(model=="Sequence CNN",CNNC,SPLC),"#FFFFFF")),
      color=I(ifelse(model=="Sequence CNN",CNNC,SPLC))))+
  geom_col(position=position_dodge(width=.66),width=.56,linewidth=.4)+
  geom_errorbar(aes(ymin=ci_lo,ymax=ci_hi),position=position_dodge(width=.66),width=.14,
                linewidth=.3,color=INK)+
  geom_hline(yintercept=.5,linetype=2,color=CHANCE,linewidth=.35)+
  geom_text(data=ANN,aes(model,top+.055,label=sprintf("-%.3f  (%.0f%% of the lift)",drop,pct)),
            inherit.aes=FALSE,size=2.05,color=INK)+
  annotate("text",x=0.52,y=.503,label="chance",size=2.0,color=CHANCE,hjust=0,vjust=-0.4)+
  annotate("text",x=1.5,y=.933,label="filled = real sequence,  open = shuffled",size=2.0,
           color=GTX,hjust=0.5)+
  scale_x_discrete(labels=c("Sequence CNN","Splice model,\n51 features"))+
  coord_cartesian(ylim=c(.45,.95))+scale_y_continuous(breaks=seq(.5,.9,.1))+
  labs(title="C   Cost of destroying sequence order",
       subtitle=paste0("Altschul-Erikson shuffling holds mono- and dinucleotide\ncontent, so ",
        "GC and CpG are unchanged and only order is lost."),
       x=NULL,y="Within-gene win-rate")+th

# d: where the cnn wins
PT<-PA%>%filter(kind=="partition")%>%
  mutate(label=factor(label,levels=c("Both rank the real end first","CNN only",
                                     "Splice model only","Neither")),
         fill=c("#4A7CB8",CNNC,SPLC,MUTE)[as.integer(label)])
PRP<-PA%>%filter(kind=="property")%>%
  mutate(label=factor(label,levels=rev(label[order(value)])))
pD1<-ggplot(PT,aes(label,value,fill=I(fill)))+geom_col(width=.62)+
  geom_text(aes(y=value+.012,label=sprintf("%d",n)),size=2.0,color=INK)+
  scale_y_continuous(limits=c(0,.46),breaks=seq(0,.4,.1))+
  scale_x_discrete(labels=c("both","CNN\nonly","splice\nonly","neither"))+
  labs(title="D   Which model ranks the real end first",
       subtitle="Strict rank 1; ties count against\nthe model.",
       x=NULL,y="Fraction of genes")+th
pD2<-ggplot(PRP,aes(value,label))+
  geom_vline(xintercept=.5,linetype=2,color=CHANCE,linewidth=.35)+
  geom_segment(aes(x=.5,xend=value,yend=label),color=GTX,linewidth=.35)+
  geom_point(aes(color=I(ifelse(value<.5,SPLC,CNNC))),size=1.7)+
  geom_text(aes(x=value+ifelse(value<.5,-.018,.018),label=sprintf("%.3f",value),
                hjust=ifelse(value<.5,1,0)),size=1.95,color=INK)+
  scale_x_continuous(limits=c(.22,.86),breaks=seq(.3,.8,.1))+
  labs(title="     What separates the CNN-only genes",
       subtitle=paste0("Rank-sum AUC, the ",PRP$n[1]," CNN-only against the ",PRP$n_ref[1],
        " both-right genes.\nBelow 0.5: the CNN's exclusive wins sit at LOWER values."),
       x="AUC separating CNN-only from both-right genes",y=NULL)+th
stopifnot(abs(PRP$value[PRP$label=="Acceptor 3'ss of the real end"]-0.35381)<1e-4)

CAP<-paste0(
 "Sequence CNN (cnn_model_def.Net, cnn_model.pt) against the locked 26-feature splice model ",
 "(U1 duplex density in 25 bins of 100 nt across [-2,000, +500] plus the maximum upstream ",
 "MaxEnt 3' splice-site score).\n",
 "Panels A, B and D: ",format(NG,big.mark=",")," held-out test genes, ",
 format(NP,big.mark=",")," peaks, base rate ",sprintf("%.3f",BR),
 "; both models fitted on the train and validation genes and applied once to test, so the ",
 "comparison is like for like.\n",
 "Panel B's splice recall@1 is therefore not the 0.465 reported elsewhere, which is the ",
 "leave-one-chromosome-out figure over all 4,847 analysis-set genes: a different protocol on a ",
 "different scope.\n",
 "Panel C plots the 51-feature positional model, the version the shuffle control was run on ",
 "(within-gene 0.695 against the 26-feature model's 0.699); the control was not re-derived for ",
 "the locked model.\n",
 "Reading panel C: a large drop means the signal is POSITIONED composition, where the sequence ",
 "texture sits, not that it is bulk base content, which the shuffle holds fixed. Both models ",
 "are more compositional than not; the CNN least so.\n",
 "Pooled AUROC and AUPRC in panel A are shown only for comparability with the wider literature. ",
 "They reward separating genes rather than sites: window GC alone reaches a higher pooled AUROC ",
 "than the splice model while ranking worse within gene.")

FIG<-((pA|pB)/((pC|pD1|pD2)+plot_layout(widths=c(1.05,0.78,1.17)))
      +plot_layout(heights=c(1,0.92)))+
  plot_annotation(
    title="A sequence model ranks real 3' ends above their intragenic competitors more accurately than splice features alone",
    subtitle=paste0("Within-gene comparison on ",format(NG,big.mark=",")," held-out genes; ",
                    "the sequence model's advantage concentrates in genes with weak upstream ",
                    "acceptors"),
    caption=CAP,
    theme=theme(plot.title=element_text(size=BASE+2,face="bold",color=INK,hjust=0),
                plot.subtitle=element_text(size=BASE,color=GTX,hjust=0,margin=margin(b=6)),
                plot.caption=element_text(size=BASE-2.2,color=GTX,hjust=0,lineheight=1.28,
                                          margin=margin(t=7))))
ggsave("figure_cnn_vs_splice.pdf",FIG,width=11.6,height=8.2,device=grDevices::cairo_pdf)
ggsave("figure_cnn_vs_splice.png",FIG,width=11.6,height=8.2,dpi=320)

TXT<-c(CAP,pA$labels$title,pA$labels$subtitle,pB$labels$title,pB$labels$subtitle,
       pC$labels$title,pC$labels$subtitle,pD1$labels$title,pD2$labels$title,pD2$labels$subtitle)
BAD<-grep("—|--- |dramatically|strikingly|remarkably|clearly|obviously",TXT,value=TRUE)
stopifnot(length(BAD)==0)
e("[fig] panel A pooled AUROC: CNN",sprintf("%.4f",PO$value[PO$model=="Sequence CNN"&PO$metric=="auroc"]),
  " splice",sprintf("%.4f",PO$value[PO$model=="Splice model (26 features)"&PO$metric=="auroc"]))
e("[fig] panel B recall@1: CNN",sprintf("%.4f",r1c)," splice",sprintf("%.4f",r1s))
e("[fig] panel C drops:",paste(sprintf("%s -%.3f (%.0f%%)",ANN$model,ANN$drop,ANN$pct),collapse="  "))
e("[fig] panel D acceptor AUC",sprintf("%.4f",PRP$value[PRP$label=="Acceptor 3'ss of the real end"]))
e("[out] figure_cnn_vs_splice.pdf figure_cnn_vs_splice.png")
