#!/usr/bin/env Rscript
# sequence cnn against the 26-feature splice model
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(tidyr);library(patchwork);library(hexbin);library(scales)})
e<-function(...) cat(...,"\n",file=stderr())
INK<-"#1A1A1A"; BASE<-7
CNNC<-"#08519C"; SPLC<-"#E8A33D"; GREY<-"#8A8F98"; GTX<-"#5A6068"

CV<-read_tsv("figLC_curves.tsv",show_col_types=FALSE)
RC<-read_tsv("figLC_recall.tsv",show_col_types=FALSE)
SH<-read_tsv("loco_shuffle.tsv",show_col_types=FALSE)
WG<-read_tsv("cnn_wholegenome_p1.tsv",show_col_types=FALSE)
WL<-read_tsv("wg_by_length.tsv",show_col_types=FALSE)
PG<-read_tsv("wg_per_gene.tsv",show_col_types=FALSE)
SM<-read_tsv("figLC_summary.tsv",show_col_types=FALSE)
NG<-SM$n_genes[1]; NP<-SM$n_peaks[1]; BR<-SM$base_rate[1]; NGA<-NG
CNNL<-"Sequence CNN"; SPLL<-"Splice model (26 features)"
ANC<-RC$recall[RC$model==CNNL&RC$k==1]; ANS<-RC$recall[RC$model==SPLL&RC$k==1]
col<-function(m) ifelse(m==CNNL,CNNC,SPLC)
stopifnot(nrow(SM)==2, setequal(SM$model,c(CNNL,SPLL)))
stopifnot(setequal(SH$model,c(CNNL,SPLL)), all(grepl("26 features",SH$model[SH$model!=CNNL])))

th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-1),axis.line=element_line(color=INK,linewidth=.3),
  axis.title=element_text(size=BASE),axis.ticks=element_line(color=INK,linewidth=.3),
  plot.title=element_text(size=BASE+1,face="bold",hjust=0,margin=margin(b=4)),
  legend.position="none")

g<-function(m,f) SM[[f]][SM$model==m]
au<-function(m,c_) CV$auc[CV$model==m&CV$curve==c_][1]

# a: roc and pr
ROC<-CV%>%filter(curve=="ROC"); PR<-CV%>%filter(curve=="PR")
pA1<-ggplot(ROC,aes(x,y,color=I(col(model))))+
  geom_abline(slope=1,intercept=0,linetype=2,color=GREY,linewidth=.35)+
  geom_line(linewidth=.55)+
  annotate("text",x=.97,y=.20,hjust=1,size=2.15,color=CNNC,
           label=sprintf("Sequence CNN   AUROC %.3f",au(CNNL,"ROC")))+
  annotate("text",x=.97,y=.12,hjust=1,size=2.15,color=SPLC,
           label=sprintf("Splice model, 26 features   AUROC %.3f",au(SPLL,"ROC")))+
  annotate("text",x=.97,y=.04,hjust=1,size=2.05,color=GREY,label="chance")+
  coord_equal(xlim=c(0,1),ylim=c(0,1))+
  scale_x_continuous(breaks=seq(0,1,.25))+scale_y_continuous(breaks=seq(0,1,.25))+
  labs(title="A   Pooled discrimination",x="False positive rate",y="True positive rate")+th
pA2<-ggplot(PR,aes(x,y,color=I(col(model))))+
  geom_hline(yintercept=BR,linetype=2,color=GREY,linewidth=.35)+
  geom_line(linewidth=.55)+
  annotate("text",x=.97,y=.93,hjust=1,size=2.15,color=CNNC,
           label=sprintf("AUPRC %.3f",au(CNNL,"PR")))+
  annotate("text",x=.97,y=.85,hjust=1,size=2.15,color=SPLC,
           label=sprintf("AUPRC %.3f",au(SPLL,"PR")))+
  annotate("text",x=.45,y=BR+.045,hjust=0,size=2.05,color=GREY,
           label=sprintf("base rate %.3f",BR))+
  coord_equal(xlim=c(0,1),ylim=c(0,1))+
  scale_x_continuous(breaks=seq(0,1,.25))+scale_y_continuous(breaks=seq(0,1,.25))+
  labs(title=" ",x="Recall",y="Precision")+th

# b: within-gene recall@k
r1c<-RC$recall[RC$model==CNNL&RC$k==1]; r1s<-RC$recall[RC$model==SPLL&RC$k==1]
stopifnot(abs(r1c-g(CNNL,"p_at_1"))<1e-6, abs(r1s-g(SPLL,"p_at_1"))<1e-6)
CHD<-RC%>%filter(model==CNNL)%>%select(k,chance)
rc<-function(k) approx(RC$k[RC$model==CNNL],RC$recall[RC$model==CNNL],k)$y
rs<-function(k) approx(RC$k[RC$model==SPLL],RC$recall[RC$model==SPLL],k)$y
ch<-function(k) approx(CHD$k,CHD$chance,k)$y
KM<-max(RC$k)
pB<-ggplot(RC,aes(k,recall,color=I(col(model))))+
  geom_ribbon(aes(ymin=ci_lo,ymax=ci_hi,fill=I(col(model))),alpha=.14,color=NA)+
  geom_line(data=CHD,aes(k,chance),inherit.aes=FALSE,linetype=2,color=GREY,linewidth=.4)+
  geom_line(linewidth=.55)+geom_point(size=.85)+
  annotate("point",x=1,y=r1c,size=2.0,shape=21,fill="white",color=CNNC,stroke=.55)+
  annotate("point",x=1,y=r1s,size=2.0,shape=21,fill="white",color=SPLC,stroke=.55)+
  annotate("text",x=1.2,y=r1c,label=sprintf("recall@1 = %.3f",r1c),size=2.1,color=CNNC,
           hjust=0,vjust=0.0)+
  annotate("text",x=1.2,y=r1s,label=sprintf("recall@1 = %.3f",r1s),size=2.1,color=SPLC,
           hjust=0,vjust=1.0)+
  annotate("text",x=KM*0.6,y=rc(KM*0.6)+.038,label="Sequence CNN",size=2.15,color=CNNC)+
  annotate("text",x=KM*0.63,y=rs(KM*0.63)-.055,label="Splice model, 26 features",size=2.15,
           color=SPLC)+
  annotate("text",x=KM*0.85,y=ch(KM*0.85)-.048,label="chance",size=2.05,color=GREY)+
  scale_x_continuous(breaks=1:KM)+scale_y_continuous(limits=c(.25,1.04),breaks=seq(.3,1,.1))+
  labs(title="B   Within-gene ranking",x="Rank threshold k",
       y="Fraction of genes with the real end in the top k")+th

# c: dinucleotide shuffling
SH2<-SH%>%select(model,real,real_lo,real_hi,shuffled,shuffled_lo,shuffled_hi,drop,pct_of_lift_lost)
LONG<-bind_rows(
  SH2%>%transmute(model,state="Real sequence",v=real,lo=real_lo,hi=real_hi),
  SH2%>%transmute(model,state="Dinucleotide shuffled",v=shuffled,lo=shuffled_lo,hi=shuffled_hi))%>%
  mutate(model=factor(model,levels=c(CNNL,SPLL)),
         state=factor(state,levels=c("Real sequence","Dinucleotide shuffled")))
ANN<-SH2%>%mutate(model=factor(model,levels=c(CNNL,SPLL)))%>%
  transmute(model,top=real_hi,lab=sprintf("-%.3f  (%.0f%% of the lift)",drop,pct_of_lift_lost))
pC<-ggplot(LONG,aes(model,v,fill=I(ifelse(state=="Real sequence",col(as.character(model)),"#FFFFFF")),
                    color=I(col(as.character(model)))))+
  geom_col(position=position_dodge(width=.66),width=.56,linewidth=.4)+
  geom_errorbar(aes(ymin=lo,ymax=hi),position=position_dodge(width=.66),width=.14,
                linewidth=.3,color=INK)+
  geom_hline(yintercept=.5,linetype=2,color=GREY,linewidth=.35)+
  geom_text(data=ANN,aes(model,top+.052,label=lab),inherit.aes=FALSE,size=2.05,color=INK)+
  annotate("text",x=0.52,y=.503,label="chance",size=2.0,color=GREY,hjust=0,vjust=-0.4)+
  annotate("text",x=1.5,y=.935,label="filled = real sequence,  open = shuffled",size=2.0,
           color=GTX,hjust=0.5)+
  scale_x_discrete(labels=c("Sequence CNN","Splice model,\n26 features"))+
  coord_cartesian(ylim=c(.45,.96))+scale_y_continuous(breaks=seq(.5,.9,.1))+
  labs(title="C   Effect of dinucleotide shuffling",
       x=NULL,y="Within-gene win-rate")+th

# d: genome-wide p@1 over gene length x candidate count
BASEW<-WG$baseline[WG$scope=="genome-wide"][1]
LV3<-c("Baseline",SPLL,CNNL)
AGG<-setNames(WG$p_at_1[WG$scope=="genome-wide"],WG$model[WG$scope=="genome-wide"])
PGL<-PG%>%filter(!is.na(gene_length),gene_length>0)%>%
  select(gene_length,n_candidates,Baseline=baseline,splice=splice,cnn=cnn)%>%
  pivot_longer(c(Baseline,splice,cnn),names_to="m",values_to="hit")%>%
  mutate(model=factor(ifelse(m=="cnn",CNNL,ifelse(m=="splice",SPLL,"Baseline")),levels=LV3),
         kb=gene_length/1000)
NHEX<-PGL%>%filter(model=="Baseline")%>%nrow()
MINN<-12
mean_or_na<-function(z) if (length(z) >= MINN) mean(z) else NA_real_
pD<-ggplot(PGL,aes(kb,n_candidates,z=hit))+
  stat_summary_hex(bins=20,fun=mean_or_na,na.rm=TRUE)+
  facet_wrap(~model,nrow=1)+
  scale_x_log10(breaks=c(1,10,100,1000),labels=c("1","10","100","1,000"))+
  scale_y_log10(breaks=c(1,3,10,30,100,300),labels=c("1","3","10","30","100","300"))+
  scale_fill_gradientn(colours=c("#FBE9D2","#FDF7F0","#DCE6F2","#9FC0E0","#4E86BF",CNNC),
                       limits=c(0,1),breaks=seq(0,1,.25),na.value="transparent",
                       name="used 3' end\nranked first")+
  geom_text(data=data.frame(model=factor(LV3,levels=LV3),
                            lab=sprintf("genome-wide %.3f",AGG[LV3])),
            aes(x=0.62,y=330,label=lab),inherit.aes=FALSE,hjust=0,size=2.15,color=INK)+
  labs(title="D   Genome-wide identification of the used 3' end",
       x="Gene length (kb)",y="Candidate 3' ends in the gene")+
  th+theme(legend.position="right",legend.key.width=unit(6,"pt"),
           legend.key.height=unit(15,"pt"),legend.title=element_text(size=BASE-2.2),
           legend.text=element_text(size=BASE-2.2),
           strip.background=element_blank(),
           strip.text=element_text(size=BASE-0.5,color=INK,face="bold",hjust=0,
                                   margin=margin(b=2)),
           panel.spacing.x=unit(7,"pt"))

CAP<-paste0(
 "Leave-one-chromosome-out over the analysis set: ",format(NG,big.mark=",")," genes and ",
 format(NP,big.mark=",")," peaks (one real 3' end per gene, ",
 format(NP-NG,big.mark=",")," intragenic competitors), 23 folds. Every peak is scored by a fit ",
 "that never saw its chromosome.\n",
 "Both models use the identical fold definition, and inside each fold a 25% gene holdout of ",
 "that fold's training genes: the splice model retunes its ridge penalty on it, the CNN early-",
 "stops on it. The two differ in the model and in nothing else.\n",
 "Panel A is pooled over peaks and is shown for reference; the within-gene comparison in panel ",
 "B is the primary metric, because a competitor is a different site in the same gene and pooled ",
 "metrics reward separating genes rather than sites.\n",
 "Panel B rank counts ties against the model, so recall@1 is the strict P@1; chance is the mean ",
 "over genes of min(k, candidates)/candidates, which is not a straight line because genes differ ",
 "in size.\n",
 "Panel C applies each fitted model, unchanged and not refitted, to sequence shuffled with the ",
 "Altschul-Erikson algorithm across the whole window. That holds mononucleotide and dinucleotide ",
 "content exactly, so the global GC and CpG of each window are unchanged, and destroys both local ",
 "order and where along the window any composition sits.\n",
 "The fraction of each model's lift over 0.5 that SURVIVES is therefore what global composition ",
 "alone can do: ", sprintf("%.0f%%",100-SH$pct_of_lift_lost[SH$model==CNNL]), " for the CNN and ",
 sprintf("%.0f%%",100-SH$pct_of_lift_lost[SH$model==SPLL]), " for the splice model. The fraction ",
 "LOST is the part that depends on order and position: ",
 sprintf("%.0f%%",SH$pct_of_lift_lost[SH$model==CNNL]), " and ",
 sprintf("%.0f%%",SH$pct_of_lift_lost[SH$model==SPLL]), ".\n",
 "Panel D covers all ",format(NHEX,big.mark=",")," genes with a GENCODE span, 85 of the 15,560 ",
 "having none. A gene is uncontested when it carries one candidate, so every ",
 "model ranks it first and ",
 "all ",format(WG$n_genes[1]-WG$n_genes[WG$scope=="contested"][1],big.mark=",")," of them enter ",
 "the denominator as correct for the baseline as well, which is why the genome-wide baseline is ",
 sprintf("%.3f",BASEW),".\nLength is what buys candidates, so that baseline is the bottom row ",
 "of each map, and how far each model stays blue above it is what sequence adds.\n",
 "Every gene appears once in each map and a hex is drawn where at least ",MINN,
 " genes fall in it.\nAll ",format(WG$n_genes[WG$scope=="contested"][1],big.mark=","),
 " contested genes are ranked over their full candidate sets, ",format(70723,big.mark=","),
 " candidates in total, scored out of fold by the same 23-fold scheme. No baseline probability ",
 "enters either model as an input.\n",
 "Bands and bars are 95% bootstrap intervals over genes.")

FIG<-((pA1|pA2|pB)+plot_layout(widths=c(0.78,0.78,1.44)))/
     ((pC|pD)+plot_layout(widths=c(1,2.05)))+
  plot_layout(heights=c(1,0.85))+
  plot_annotation(
    title=paste0("A sequence CNN outperforms splice features at within-gene 3' end ",
                 "discrimination"),
    caption=CAP,
    theme=theme(plot.title=element_text(size=BASE+2,face="bold",color=INK,hjust=0,
                                        margin=margin(b=6)),
                plot.caption=element_text(size=BASE-2.2,color=GTX,hjust=0,lineheight=1.3,
                                          margin=margin(t=8))))
ggsave("figure_loco_cnn_vs_splice.pdf",FIG,width=10.6,height=7.4,device=grDevices::cairo_pdf)
ggsave("figure_loco_cnn_vs_splice.png",FIG,width=10.6,height=7.4,dpi=320)

stopifnot(is.null(pA1$labels$subtitle),is.null(pA2$labels$subtitle),
          is.null(pB$labels$subtitle),is.null(pC$labels$subtitle))
stopifnot(setequal(unique(CV$model),c(CNNL,SPLL)),setequal(unique(RC$model),c(CNNL,SPLL)),
          setequal(unique(SH$model),c(CNNL,SPLL)))
stopifnot(!any(grepl("GC|acceptor|Acceptor",c(pA1$labels$title,pA2$labels$title,pB$labels$title,
                                              pC$labels$title))))
e("[fig] LOCO within-gene: CNN",sprintf("%.4f",g(CNNL,"within_gene")),
  " splice",sprintf("%.4f",g(SPLL,"within_gene")))
e("[fig] pooled AUROC/AUPRC: CNN",sprintf("%.4f / %.4f",au(CNNL,"ROC"),au(CNNL,"PR")),
  " splice",sprintf("%.4f / %.4f",au(SPLL,"ROC"),au(SPLL,"PR")))
e("[fig] recall@1: CNN",sprintf("%.4f",r1c)," splice",sprintf("%.4f",r1s))
e("[fig] shuffle drops:",paste(sprintf("%s -%.3f (%.0f%%)",SH$model,SH$drop,
                                       SH$pct_of_lift_lost),collapse="  "))
e("[check] two models only; no panel subtitles; no GC or acceptor panel")
e("[out] figure_loco_cnn_vs_splice.pdf figure_loco_cnn_vs_splice.png")
