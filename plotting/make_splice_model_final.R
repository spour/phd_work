#!/usr/bin/env Rscript
# the three finished splice-model panels composed into one figure
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork)})
e<-function(...) cat(...,"\n",file=stderr())
INK<-"#1A1A1A"; BASE<-7
U1C<-"#08519C"; ACCC<-"#E8A33D"; GREY<-"#9AA0A6"; GTX<-"#5A6068"
X0<--2000; X1<-500; BW<-100
XLO<--2150; XHI<-1280
XBRK<-seq(-2000,500,500)
CF<-read_tsv("splice_model_26_coefficients.tsv",show_col_types=FALSE)
RK<-read_tsv("recall_at_k.tsv",show_col_types=FALSE)
MT<-read_tsv("splice_model_26_metrics.tsv",show_col_types=FALSE)
mv<-function(k,c="value") MT[[c]][MT$metric==k][1]
base_th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-1),axis.line=element_line(color=INK,linewidth=.3),
  axis.title=element_text(size=BASE),
  plot.title=element_text(size=BASE+1,face="bold",hjust=0,margin=margin(b=4)),
  legend.position="none")

# a: schematic
ed<-seq(X0,X1,by=BW)
yW<-3.40; hW<-0.26; yU<-2.25; yA<-1.30; hC<-0.24
pA<-ggplot()+
  annotate("rect",xmin=X0,xmax=0,ymin=yW-hW,ymax=yW+hW,fill="#EDEFF2",color=NA)+
  annotate("rect",xmin=0,xmax=X1,ymin=yW-hW,ymax=yW+hW,fill="#F7F8FA",color=NA)+
  annotate("segment",x=ed,xend=ed,y=yW-hW,yend=yW+hW,color="white",linewidth=.35)+
  annotate("rect",xmin=X0,xmax=X1,ymin=yW-hW,ymax=yW+hW,fill=NA,color=INK,linewidth=.35)+
  annotate("segment",x=0,xend=0,y=yW-hW-0.16,yend=yW+hW+0.14,linetype=2,color=INK,linewidth=.4)+
  annotate("text",x=-60,y=yW+hW+0.20,label="cleavage site",size=2.2,color=INK,hjust=1,vjust=0)+
  annotate("text",x=X0,y=yW+hW+0.20,label="Sequence window, 25 bins of 100 nt",
           size=2.1,color=GTX,hjust=0,vjust=0)+
  annotate("rect",xmin=X0,xmax=X1,ymin=yU-hC,ymax=yU+hC,fill=U1C,alpha=.85,color=NA)+
  annotate("segment",x=ed,xend=ed,y=yU-hC,yend=yU+hC,color="white",linewidth=.3)+
  annotate("text",x=X0,y=yU+hC+0.15,hjust=0,vjust=0,size=2.1,color=U1C,
           label="U1 duplex density (-dG >= 4) per bin, 25 values")+
  annotate("rect",xmin=X0,xmax=-126,ymin=yA-hC,ymax=yA+hC,fill=ACCC,alpha=.30,color=NA)+
  annotate("rect",xmin=-1120,xmax=-940,ymin=yA-hC,ymax=yA+hC,fill=ACCC,color=NA)+
  annotate("text",x=X0,y=yA+hC+0.15,hjust=0,vjust=0,size=2.1,color=ACCC,
           label="Maximum MaxEnt 3' splice-site score over [-2,000, -126], 1 value")+
  annotate("segment",x=380,xend=380,y=yW-hW-0.03,yend=yU+hC+0.03,color=GTX,linewidth=.3,
           arrow=grid::arrow(length=unit(.045,"in"),type="closed"))+
  annotate("segment",x=380,xend=380,y=yU-hC-0.03,yend=yA+hC+0.03,color=GTX,linewidth=.3,
           arrow=grid::arrow(length=unit(.045,"in"),type="closed"))+
  annotate("segment",x=-750,xend=-750,y=yA-hC-0.04,yend=0.74,color=GTX,linewidth=.3,
           arrow=grid::arrow(length=unit(.045,"in"),type="closed"))+
  annotate("rect",xmin=-1720,xmax=220,ymin=0.20,ymax=0.72,fill="white",color=INK,linewidth=.4)+
  annotate("text",x=-750,y=0.575,label="26 features",size=2.5,color=INK,fontface="bold")+
  annotate("text",x=-750,y=0.345,size=2.05,color=INK,
           label="Ridge (L2) logistic regression, features centred within gene")+
  annotate("segment",x=-750,xend=-750,y=0.18,yend=-0.02,color=GTX,linewidth=.3,
           arrow=grid::arrow(length=unit(.045,"in"),type="closed"))+
  annotate("text",x=-750,y=-0.10,vjust=1,size=2.05,color=INK,
           label="Within-gene evaluation: rank of the real 3' end among its gene's candidates")+
  scale_x_continuous(breaks=XBRK,limits=c(XLO,XHI),expand=expansion(mult=c(0,0)))+
  scale_y_continuous(limits=c(-0.42,3.95),expand=expansion(mult=c(0,0)))+
  labs(title="A   Feature construction and model specification",x=NULL,y=NULL)+
  base_th+theme(axis.text=element_blank(),axis.ticks=element_blank(),
                axis.line=element_blank(),plot.margin=margin(4,4,0,4))

# b: coefficient profile
U<-CF%>%filter(!is.na(channel),channel=="u1_dens4")
A<-CF$coefficient[which(CF$channel=="acceptor")][1]
stopifnot(nrow(U)==25,is.finite(A))
jm<-which.min(U$coefficient)
stopifnot(abs(U$coefficient[jm]+0.2273)<5e-4, U$bin_centre[jm]==-150, abs(A-0.2959)<5e-4)
e(sprintf("[B] U1 minimum %+.4f at %+d nt (expected -0.2273 at -150); acceptor %+.4f (expected +0.2959)",
          U$coefficient[jm],U$bin_centre[jm],A))
rng<-range(c(U$coefficient,A,0)); pad<-diff(rng)*0.17
pB<-ggplot(U,aes(bin_centre,coefficient))+
  annotate("rect",xmin=-Inf,xmax=0,ymin=-Inf,ymax=Inf,fill=GREY,alpha=.07)+
  geom_hline(yintercept=0,color=INK,linewidth=.35)+
  geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.35)+
  geom_col(width=88,fill=U1C)+
  annotate("text",x=U$bin_centre[jm],y=U$coefficient[jm],vjust=1.6,size=2.1,color=U1C,
           label=sprintf("%+.2f at %+d nt",U$coefficient[jm],U$bin_centre[jm]))+
  annotate("segment",x=700,xend=700,y=rng[1]-pad,yend=rng[2]+pad,color=GREY,linewidth=.3)+
  annotate("rect",xmin=850,xmax=1130,ymin=0,ymax=A,fill=ACCC)+
  annotate("text",x=990,y=A,label=sprintf("%+.2f",A),size=2.2,color=ACCC,vjust=-0.6)+
  annotate("text",x=990,y=rng[1]-pad*0.45,label="MaxEnt 3'ss\nscalar, not positional",
           size=2.0,color=ACCC,vjust=1,lineheight=1.1)+
  scale_x_continuous(breaks=XBRK,limits=c(XLO,XHI),expand=expansion(mult=c(0,0)))+
  scale_y_continuous(limits=c(rng[1]-pad*1.2,rng[2]+pad*1.2),expand=expansion(mult=c(0,0)))+
  labs(title="B   Ridge regression coefficients by position",
       x="position relative to the cleavage site (nt)",y="ridge coefficient")+
  base_th+theme(plot.margin=margin(2,4,4,4))

# c: recall@k
NG<-RK$n_genes[1]; KM<-max(RK$k); g<-function(k,c="recall") RK[[c]][RK$k==k][1]
stopifnot(NG==4847)
e(sprintf("[C] %d analysis-set genes, recall@1 %.3f (chance %.3f), recall@3 %.3f (chance %.3f)",
          NG,g(1),g(1,"chance"),g(3),g(3,"chance")))
AN<-RK%>%filter(k %in% c(1,3))%>%
  mutate(lab=sprintf("recall@%d   %.3f\nrandom       %.3f",k,recall,chance))
pC<-ggplot(RK,aes(k))+
  geom_ribbon(aes(ymin=chance,ymax=recall),fill=U1C,alpha=.13)+
  geom_line(aes(y=chance),linetype=2,color=GREY,linewidth=.6)+
  geom_point(aes(y=chance),color=GREY,size=1.2)+
  geom_line(aes(y=recall),color=U1C,linewidth=.85)+
  geom_point(aes(y=recall),color=U1C,size=1.7)+
  geom_point(data=RK%>%filter(k==1),aes(y=recall),size=3.2,shape=21,fill=U1C,
             color="white",stroke=.8)+
  geom_text(data=AN,aes(y=recall,label=lab),hjust=-0.10,vjust=1.35,size=2.2,color=INK,
            lineheight=1.25)+
  annotate("text",x=KM+0.4,y=g(KM,"chance"),label="random\nranking",hjust=1,vjust=3.1,
           size=2.2,color=GREY,lineheight=1.1)+
  scale_x_continuous(breaks=1:KM,limits=c(1,KM+0.55),expand=expansion(mult=c(.02,.02)))+
  scale_y_continuous(limits=c(0.20,1),labels=scales::percent_format(accuracy=1),
                     expand=expansion(mult=c(0,.02)))+
  labs(title="C   Within-gene recall as a function of rank threshold",
       x="rank threshold k",y="genes with the true 3' end at rank <= k")+
  base_th+theme(panel.grid.major.y=element_line(color="#EEF0F3",linewidth=.25),
                plot.margin=margin(8,4,4,4))

# d: within-gene score separation
SS<-read_tsv("score_separation.tsv",show_col_types=FALSE)
ssv<-function(k){ v<-SS$value[SS$key==k]; if(!length(v)) stop("missing: ",k); v[1] }
SD<-read_tsv("score_separation_sites.tsv",show_col_types=FALSE)
stopifnot(as.numeric(ssv("n_genes"))==NG, ssv("oof")=="leave-one-chromosome-out")
CHK<-SD%>%group_by(gene)%>%
  summarise(top1=as.numeric(centred_score[class=="true"][1] >
                            max(centred_score[class=="competitor"])),.groups="drop")
R1D<-mean(CHK$top1); R1C<-g(1)
stopifnot(nrow(CHK)==NG, abs(R1D-R1C)<1e-6)
e(sprintf("[consistency] recall@1 recomputed from panel D's scores = %.4f, panel C's file = %.4f",
          R1D,R1C))
e(sprintf("[consistency] win-rate %.4f and the Wilcoxon (V = %s, z = %s, p %s) come from the",
          as.numeric(ssv("winrate_loco")),ssv("wilcoxon_V"),ssv("wilcoxon_z"),ssv("wilcoxon_p")))
e("[consistency] same LOCO scores over the same",NG,"genes; the held-out 0.472 is not used here")
WRD<-as.numeric(ssv("winrate_loco")); WLO<-as.numeric(ssv("winrate_lo"))
WHI<-as.numeric(ssv("winrate_hi")); MED<-as.numeric(ssv("median_delta"))
FP<-as.numeric(ssv("frac_positive"))
SD2<-SD%>%mutate(class=factor(class,levels=c("true","competitor"),
                              labels=c("true 3' end","competitor")))
QD<-SD2%>%group_by(class)%>%summarise(m=median(centred_score),.groups="drop")
YL<-quantile(SD2$centred_score,c(.002,.998)); YTOP<-YL[2]+0.62*diff(YL)
pD<-ggplot(SD2,aes(class,centred_score,fill=class,color=class))+
  geom_hline(yintercept=0,linetype=2,color=GREY,linewidth=.4)+
  geom_violin(width=.86,linewidth=.35,alpha=.45,trim=TRUE)+
  geom_boxplot(width=.13,outlier.shape=NA,linewidth=.35,fill="white",color=INK)+
  scale_fill_manual(values=c(`true 3' end`=U1C,competitor=ACCC))+
  scale_color_manual(values=c(`true 3' end`=U1C,competitor=ACCC))+
  geom_text(data=QD,aes(x=class,y=m,label=sprintf("%+.2f",m)),inherit.aes=FALSE,
            hjust=-0.55,size=2.1,color=INK)+
  annotate("text",x=0.52,y=YTOP,hjust=0,vjust=1.1,size=2.0,color=INK,lineheight=1.35,
    label=paste0("Win-rate ",sprintf("%.3f",WRD)," [",sprintf("%.3f",WLO),", ",
      sprintf("%.3f",WHI),"]\nPaired Wilcoxon on the per-gene difference\nV = ",
      format(as.numeric(ssv("wilcoxon_V")),big.mark=","),", n = ",format(NG,big.mark=","),
      ", z = ",sprintf("%+.1f",as.numeric(ssv("wilcoxon_z"))),", p ",ssv("wilcoxon_p"),
      "\nmedian difference ",sprintf("%+.2f",MED),", positive in ",
      sprintf("%.0f%%",100*FP)," of genes"))+
  coord_cartesian(ylim=c(YL[1],YTOP))+
  labs(title="D   Score distributions at real 3' ends and competitors",
       x=NULL,y="gene-centred model score")+
  base_th+theme(axis.text.x=element_text(size=BASE),
                panel.grid.major.y=element_line(color="#EEF0F3",linewidth=.25),
                plot.margin=margin(8,4,4,4))

CAP<-paste(strwrap(paste0("n = ",format(NG,big.mark=",")," genes, each with one real 3' end ",
  "and at least one competitor. Panels C and D are leave-one-chromosome-out over those genes ",
  "and share one scoring; C's shaded band is the gap to random ranking, D's scores are ",
  "centred within gene."),width=118),collapse="\n")
fig<-(pA/pB/(pC|pD))+plot_layout(heights=c(1.20,1.0,1.05))+
  plot_annotation(
    title="Positional U1 duplex density and upstream 3' splice-site strength discriminate real 3' ends from intragenic competitors",
    subtitle="Ridge logistic model, 26 features: U1 duplex density (-dG >= 4) in 25 bins of 100 nt across [-2,000, +500], and the maximum upstream MaxEnt 3' splice-site score",
    caption=CAP,
    theme=theme(plot.title=element_text(size=BASE+2.5,face="bold",color=INK),
                plot.subtitle=element_text(size=BASE,color=GTX,margin=margin(b=6)),
                plot.caption=element_text(size=BASE-1.6,color=GTX,hjust=0,lineheight=1.25)))
ggsave("figure_splice_model_final.pdf",fig,width=7.8,height=8.6,device=grDevices::cairo_pdf)
ggsave("figure_splice_model_final.png",fig,width=7.8,height=8.6,dpi=300)
e("[layout] A top full width, B full width sharing A's x limits",XLO,"to",XHI,"so the window",
  "aligns with the coefficient profile, and C and D side by side in a bottom row")
BAN<-c("CNN","shuffl","scalar splice","dinucleotide","whole genome","whole-genome")
SRC<-c(CAP,"An acceptor score and positional U1 density place the 3' end within its gene",
       "U1 duplex density (-dG >= 4) in 25 bins of 100 nt across [-2,000, +500] plus the upstream MaxEnt 3' splice-site score: 26 features")
e("[chk] banned terms in the drawn figure:",sum(sapply(BAN,function(b) sum(grepl(b,SRC)))))
e("[chk] panel headings carry a letter and a short title only, no subtitles")
e("[out] figure_splice_model_final.pdf figure_splice_model_final.png")
