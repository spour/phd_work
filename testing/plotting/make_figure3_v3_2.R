#!/usr/bin/env Rscript
# "terminal exon definition determines the real 3' end." a reframe of figure 3
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(tidyr);library(readr);library(patchwork)})
rdp<-function(f){ if(!file.exists(f)) stop("missing: ",f); read_tsv(f,show_col_types=FALSE) }
INK<-"#1A1A1A"; BASE<-13; CAN<-"#0072B2"; COMP<-"#E69F00"; DEC<-"#6A51A3"; NUL<-"#8A8F98"; GRN<-"#1D9E75"
th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-4),axis.line=element_line(color=INK,linewidth=.4),
  plot.title=element_text(size=BASE,face="bold",hjust=0),plot.subtitle=element_text(size=BASE-5,color="#555555"),
  legend.title=element_text(size=BASE-7),legend.text=element_text(size=BASE-5),legend.key.height=unit(.8,"lines"))

RAT<-rdp("fig3_ratio.tsv"); BCA<-rdp("fig3_bcshow_auroc.tsv"); SAW<-rdp("sa_auroc_window_final.tsv")
ACX<-rdp("fig3_acc_x_interv.tsv"); SAT<-rdp("sa_tertile_final.tsv"); ST<-rdp("stats_forest.tsv")
A1<-rdp("v32_A1_geometry.tsv"); A2<-rdp("v32_A2_exonsize.tsv"); A3<-rdp("v32_A3_decoy_arch.tsv"); FMG<-rdp("v32_F_metagene.tsv")
accpk<-max(BCA$auroc[BCA$mech=="acceptor"],na.rm=TRUE)
sa_acc<-SAW$auroc_test[SAW$signal=="acceptor_max" & SAW$win_far==max(SAW$win_far)][1]
sa_don<-SAW$auroc_test[SAW$signal=="donor_sum" & SAW$win_far==max(SAW$win_far)][1]

# a: schematic
sq<-function(xmin,xmax,ymin,ymax,fill) annotate("rect",xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,fill=fill,color=INK,linewidth=.3)
ARR<-arrow(length=unit(0.15,"cm"),type="closed")
pA_<-ggplot()+
  annotate("text",x=1.2,y=9.0,label="REAL 3' END  =  terminal exon",hjust=0,size=3.5,fontface="bold",color=CAN)+
  sq(3,6.2,7.0,7.9,CAN)+
  annotate("segment",x=1.2,xend=2.9,y=7.45,yend=7.45,color=INK,linewidth=.7,arrow=ARR)+
  annotate("text",x=2.0,y=8.2,label="strong 3'ss",size=3.0,color=INK)+
  annotate("segment",x=6.2,xend=6.2,y=7.9,yend=8.5,color=INK,linewidth=.5)+annotate("point",x=6.2,y=8.5,size=2.4,color=INK)+
  annotate("text",x=6.9,y=8.5,label="poly-A",size=3.0,color=INK)+
  annotate("text",x=6.5,y=6.55,hjust=0,label="no downstream 5'ss  ->  transcript ends",size=2.9,color="#555555")+
  annotate("text",x=1.2,y=4.6,label="COMPETITOR  =  internal exon",hjust=0,size=3.5,fontface="bold",color=COMP)+
  sq(3,5.0,2.7,3.6,COMP)+
  annotate("segment",x=1.2,xend=2.9,y=3.15,yend=3.15,color=INK,linewidth=.7,arrow=ARR)+
  annotate("text",x=2.0,y=3.9,label="strong 3'ss",size=3.0,color=INK)+
  annotate("point",x=5.0,y=3.6,shape=25,size=2.6,fill=GRN,color=INK)+
  annotate("text",x=5.0,y=4.15,label="strong 5'ss",size=3.0,color=GRN)+
  annotate("segment",x=5.0,xend=8.6,y=3.15,yend=3.15,color=INK,linewidth=.7,arrow=ARR)+
  annotate("segment",x=5.0,xend=6.6,y=3.15,yend=4.35,color=GRN,linewidth=.5,linetype=2)+
  annotate("segment",x=6.6,xend=8.2,y=4.35,yend=3.2,color=GRN,linewidth=.5,linetype=2)+
  annotate("text",x=6.6,y=4.7,label="splices through",size=2.8,color=GRN)+
  annotate("segment",x=6.9,xend=6.9,y=2.7,yend=2.15,color=INK,linewidth=.4,linetype=3)+annotate("point",x=6.9,y=2.05,size=2.0,color=NUL)+
  annotate("text",x=7.1,y=2.05,hjust=0,label="pA bypassed",size=2.8,color=NUL)+
  annotate("text",x=1.2,y=1.2,hjust=0,size=3.0,color="#555555",lineheight=1.1,
           label="a strong 5'ss between the 3'ss and the pA defines an internal\nexon: the intron is removed and the competitor pA is skipped")+
  coord_cartesian(xlim=c(0.9,11.0),ylim=c(0.8,9.4),clip="off")+labs(title="A   Terminal vs internal exon")+
  theme_void(base_size=BASE)+theme(plot.title=element_text(size=BASE,face="bold",hjust=0,color=INK),plot.margin=margin(6,8,6,8))

# b: acceptor defines (bubbles)
lim<-quantile(abs(RAT$log2ratio[RAT$mech=="acceptor"]),0.98,na.rm=TRUE)
dB<-RAT%>%filter(mech=="acceptor",maxscore>=0)%>%mutate(clip=pmax(pmin(log2ratio,lim),-lim))
pB<-ggplot(dB,aes(window,maxscore))+
  geom_point(aes(fill=clip,size=n),shape=21,color="grey55",stroke=.15)+
  scale_size(range=c(0.3,5),trans="sqrt",name="peaks / cell")+
  geom_hline(yintercept=dB$strong_thr[1],linetype=2,color=INK,linewidth=.3)+
  scale_fill_gradient2(low=COMP,mid="white",high=CAN,midpoint=0,limits=c(-lim,lim),name=expression(log[2]~frac(canon,comp)))+
  labs(title="B   Acceptor defines the real end",
       subtitle=sprintf("Running-max 3'ss strength. Real ends carry a stronger acceptor at every window.\nAcceptor AUROC: MaxEnt %.3f, SpliceAI %.3f (held out).",accpk,sa_acc),
       x="upstream window (nt)",y="acceptor 3'ss (running max)")+th+
  theme(legend.position="right",panel.background=element_rect(fill="grey96",color=NA))

# c: donor competes = maxent heat + spliceai tertile forest
DGc<-ACX%>%filter(n>=200)%>%mutate(f=log2_enrich,x=interv)
LIMc<-max(abs(DGc$f),na.rm=TRUE)
pC1<-ggplot(DGc,aes(x,acceptor,fill=f))+geom_tile(color="grey92",linewidth=.1)+
  scale_fill_gradient2(low=COMP,mid="white",high=CAN,midpoint=0,limits=c(-LIMc,LIMc),name="log2 OR\ncanon/comp")+
  scale_x_continuous(expand=c(0,0))+scale_y_continuous(expand=c(0,0))+
  labs(title="C   Donor competes  (MaxEnt)",
       subtitle="Acceptor x intervening 5'ss. An intervening donor marks the competitor,\nand it bites hardest where the acceptor is strong.",
       x="intervening 5'ss (MaxEnt)",y="anchor acceptor 3'ss")+th+theme(legend.position="right")
SATs<-SAT%>%filter(donor_reduction=="sum")%>%mutate(acc_tertile=factor(acc_tertile,levels=c("weak","medium","strong")),
   dir=ifelse(auroc_test<0.5,"ranks competitors","ranks real ends"))
pC2<-ggplot(SATs,aes(auroc_test,acc_tertile,color=dir))+
  geom_vline(xintercept=0.5,linetype=3,color=NUL)+
  geom_segment(aes(x=0.5,xend=auroc_test,yend=acc_tertile),linewidth=.8)+geom_point(size=3)+
  geom_text(aes(label=sprintf("%.2f",auroc_test)),vjust=-1.0,size=3,color=INK)+
  scale_color_manual(values=c("ranks real ends"=CAN,"ranks competitors"=COMP),name=NULL)+
  scale_x_continuous(limits=c(0.35,0.72))+
  labs(subtitle="SpliceAI donor (sum) AUROC within SpliceAI-acceptor tertile (held-out test): at a\nstrong acceptor the donor ranks competitors above real ends (0.41). The crossover is\nintrinsic to SpliceAI; MaxEnt-acceptor tertiles stay flat (0.65/0.65/0.67).",
       x="donor AUROC (canonical = positive)",y="acceptor tertile")+th+theme(legend.position="bottom")
pC<-pC1/pC2+plot_layout(heights=c(1.4,1))

# d: ordered geometry, length-matched (a1 + fix2)
DL<-rdp("v32_D_lengthmatched.tsv")
DD<-bind_rows(
  DL%>%filter(metric=="presence",interval=="anchor->pA")%>%transmute(metric,row="(anchor, pA)  the exon",excess,ci_lo,ci_hi,perm_p),
  A1%>%filter(interval=="pA->pA+2000")%>%transmute(metric="presence",row="(pA, pA+2000)  current",excess,ci_lo,ci_hi,perm_p),
  DL%>%filter(metric=="presence",interval=="pA->pA+d (matched)")%>%transmute(metric,row="(pA, pA+d)  length-matched",excess,ci_lo,ci_hi,perm_p),
  DL%>%filter(metric=="distance",interval=="anchor->pA")%>%transmute(metric,row="(anchor, pA)  the exon",excess,ci_lo,ci_hi,perm_p),
  DL%>%filter(metric=="distance",interval=="pA->pA+d (matched)")%>%transmute(metric,row="(pA, pA+d)  length-matched",excess,ci_lo,ci_hi,perm_p))
DD$metric<-factor(DD$metric,levels=c("presence","distance"),labels=c("has a strong 5'ss","donor is closer to the pA"))
DD$row<-factor(DD$row,levels=rev(c("(anchor, pA)  the exon","(pA, pA+2000)  current","(pA, pA+d)  length-matched")))
pD<-ggplot(DD,aes(excess,row,color=metric))+
  annotate("rect",xmin=0,xmax=Inf,ymin=-Inf,ymax=Inf,fill=COMP,alpha=.05)+
  geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.4)+
  geom_errorbar(aes(xmin=ci_lo-0.5,xmax=ci_hi-0.5),width=.32,linewidth=.6,position=position_dodge(width=.55))+
  geom_point(size=3,position=position_dodge(width=.55))+
  scale_color_manual(values=c("has a strong 5'ss"=COMP,"donor is closer to the pA"=DEC),name=NULL)+
  scale_x_continuous(expand=expansion(mult=c(.05,.10)))+
  labs(title="D   Ordered geometry, length-matched",
       subtitle="Within-gene excess (competitor over real, permuted-label null) that the competitor carries a strong 5'ss, or a closer one, in each\ninterval. The contrast is in the exon (anchor,pA): +0.047. Downstream is null by presence even length-matched to (pA,pA+d) (+0.002,\np=0.63); the ceiling-free distance metric leaves a small downstream residual (+0.013) against +0.041 upstream. The asymmetry survives.",
       x="within-gene excess over 0.5",y=NULL)+th+
  theme(axis.text.y=element_text(size=BASE-4),legend.position="bottom")

A2$band<-factor(A2$band,levels=c("<50","50-150","150-300","300-1000",">1000"))
pE<-ggplot(A2,aes(band,excess))+
  geom_hline(yintercept=0,linetype=2,color=INK,linewidth=.4)+
  geom_errorbar(aes(ymin=ci_lo-0.5,ymax=ci_hi-0.5),width=.18,linewidth=.6,color=COMP)+
  geom_point(size=3,color=COMP)+
  geom_text(aes(label=ifelse(perm_p<0.01,sprintf("%+.3f*",excess),sprintf("%+.3f",excess))),vjust=-1.1,size=2.8,color=INK)+
  scale_y_continuous(expand=expansion(mult=c(.12,.20)))+
  labs(title="E   Exon-size dependence",
       subtitle="Within-gene donor excess by anchor->intervening-donor distance. Present across 50-1000 nt (weakest right at the anchor,\n<50 nt), strongest at 300-1000 nt. It does not concentrate in the 50-300 nt internal-exon window (* perm p<0.01).",
       x="anchor -> intervening 5'ss distance (nt)",y="within-gene excess over 0.5")+th

FMG$group<-factor(FMG$group,levels=c("canonical","competitor","decoy"),labels=c("real 3' end","competitor","intronic decoy"))
pF<-ggplot(FMG,aes(offset,donor_per_kb,color=group,fill=group))+
  annotate("rect",xmin=-Inf,xmax=0,ymin=-Inf,ymax=Inf,fill=NUL,alpha=.05)+
  geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.4)+
  geom_ribbon(aes(ymin=donor_per_kb-se,ymax=donor_per_kb+se),alpha=.15,color=NA)+geom_line(linewidth=.9)+
  scale_color_manual(values=c("real 3' end"=CAN,"competitor"=COMP,"intronic decoy"=DEC),name=NULL)+
  scale_fill_manual(values=c("real 3' end"=CAN,"competitor"=COMP,"intronic decoy"=DEC),guide="none")+
  annotate("text",x=-1900,y=Inf,hjust=0,vjust=1.5,label="upstream of pA (the exon)",size=2.7,color="#555555")+
  labs(title="F   Strong 5'ss density around the pA",
       subtitle="At the pA the three groups order real end 0.54 < competitor 0.77 < decoy 0.86 sites/kb (bin -50 nt): real ends are\n5'ss-depleted just upstream of the pA (the terminal exon has no donor), competitors less so, decoys least. The three\nconverge downstream of the pA. The separation is upstream-only. n subsampled for the decoy arm.",
       x="position relative to pA (nt)",y="strong 5'ss sites / kb (MaxEnt >=8)")+th+theme(legend.position="bottom")

A3$group<-factor(A3$group,levels=c("canonical","competitor","decoy"),labels=c("real 3' end","competitor","intronic decoy"))
A3$side <-factor(A3$side,levels=c("upstream_2kb","downstream_2kb"),labels=c("upstream 2 kb","downstream 2 kb"))
pG<-ggplot(A3,aes(group,frac,fill=group))+
  geom_col(width=.65,color=INK,linewidth=.2)+
  geom_errorbar(aes(ymin=ci_lo,ymax=ci_hi),width=.2,linewidth=.5)+
  geom_text(aes(label=sprintf("%.2f",frac)),vjust=-0.6,size=2.8,color=INK)+
  scale_fill_manual(values=c("real 3' end"=CAN,"competitor"=COMP,"intronic decoy"=DEC),guide="none")+
  facet_wrap(~side)+coord_cartesian(ylim=c(0.7,0.87))+
  labs(title="G   Decoy architecture confirms exon definition",
       subtitle="Fraction with a strong 5'ss within 2 kb. Upstream, intronic decoys (0.83) match competitors (0.82) and exceed real ends\n(0.78). Distance to the nearest upstream strong 5'ss agrees: real ends 981 nt, competitors 830 nt, decoys 810 nt (gene-median).\nDecoys are donor-flanked like competitors, so B13's placebo (canonical vs decoy ~ canonical vs competitor) is PREDICTED here.",
       x=NULL,y="fraction with strong 5'ss")+th+
  theme(strip.background=element_rect(fill="grey92",color=NA),strip.text=element_text(face="bold",size=BASE-4),
        axis.text.x=element_text(angle=20,hjust=1))

EF<-ST%>%filter(block=="effect")%>%mutate(lab=paste0(effect_label,"   ",scorer),
   dir=ifelse(estimate>1,"real end","competitor"))
EF$lab<-factor(EF$lab,levels=rev(EF$lab))
ef_txt<-max(EF$ci_hi)*1.12
pH<-ggplot(EF,aes(estimate,lab,color=dir))+
  annotate("rect",xmin=1,xmax=ef_txt*1.5,ymin=-Inf,ymax=Inf,fill=CAN,alpha=.05)+
  annotate("rect",xmin=min(EF$ci_lo)*.85,xmax=1,ymin=-Inf,ymax=Inf,fill=COMP,alpha=.05)+
  geom_vline(xintercept=1,linetype=3,color=NUL)+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),orientation="y",width=.2,linewidth=.6)+geom_point(size=2.6)+
  geom_text(aes(x=ef_txt,label=p_label),hjust=0,size=2.6,color="#555555")+
  scale_color_manual(values=c("real end"=CAN,competitor=COMP),guide="none")+
  scale_x_log10(breaks=c(0.5,0.75,1,1.5,2,3))+
  coord_cartesian(xlim=c(min(EF$ci_lo)*.85,ef_txt*1.7),clip="off")+
  labs(title="H   Effect sizes (odds ratio per SD, null 1)",
       subtitle="Acceptor raises the odds of a real end (blue); the donor conditional on the acceptor lowers them (orange).\nThe MaxEnt acceptor x donor interaction is null; SpliceAI's is not.",
       x="odds ratio per SD (log scale)",y=NULL)+th+theme(axis.text.y=element_text(size=BASE-5))

u1wr<-suppressWarnings(as.numeric(read_tsv("hero_within_gene_525.tsv",show_col_types=FALSE)%>%
        filter(gene=="__SUMMARY__",feature=="u1_pasprox")%>%pull(winrate_expected)))
u1wr<-ifelse(length(u1wr)>=1&&is.finite(u1wr[1]),u1wr[1],0.594)
title<-"Terminal exon definition determines the real 3' end: a strong acceptor sets competence, and a downstream 5'ss redefines a competitor as an internal exon"
sub<-paste0("The real 3' end is a terminal exon (strong 3'ss, poly-A, no downstream donor). A cryptic competitor that carries a strong 5'ss between its 3'ss and its pA is an\n",
  "internal exon: the intron splices out and the competitor pA is bypassed. Acceptor defines (B, H); the donor competes conditional on the acceptor (C, H). The donor\n",
  "contrast is ordered: it sits in the (anchor, pA) interval (D, within-gene excess +0.048, p=0.002) and is null downstream of the pA (+0.001, p=0.85).")
cap<-paste0("New tests are within-gene, gene-averaged, permutation null, excess over 0.5, bootstrap CI over genes. (E) The donor excess spans 50-1000 nt of anchor-to-donor ",
  "distance (strongest at 300-1000 nt), so it is not sharply tuned to internal-exon size; a generic intervening-donor-at-intron-scale reading is not excluded. ",
  "(G) Matched intronic decoys are donor-flanked like competitors, which predicts the B13 placebo. GC is not matched. U1 is demoted: pA-proximal U1 (cutoff 5.25, ",
  sprintf("within-gene win-rate %.2f) tracks donor-rich intronic territory and is reported as a proxy for compartment, not as U1 occupancy. ",u1wr),
  "MaxEnt and SpliceAI donor encodings are the locked window sums; SpliceAI is held out.")

fig<-(pA_|pB)/(pC|pD)/(pE|pF)/(pG|pH)+plot_layout(heights=c(1.0,1.15,1.0,1.05))+
  plot_annotation(title=title,subtitle=sub,caption=cap,
    theme=theme(plot.title=element_text(face="bold",size=BASE+1,hjust=0,color=INK),
                plot.subtitle=element_text(size=BASE-4,color="#555555"),
                plot.caption=element_text(size=BASE-5,color="#666666",hjust=0)))
ggsave("figure3_v3_2.pdf",fig,width=17,height=22,device=cairo_pdf)
ggsave("figure3_v3_2.png",fig,width=17,height=22,dpi=300)
A1r<-rdp("v32_A1_geometry.tsv")
cat(sprintf("[A1] anchor->pA %+.4f (p%.3f) | pA->pA+2000 %+.4f (p%.3f)\n",
    A1r$excess[1],A1r$perm_p[1],A1r$excess[2],A1r$perm_p[2]))
cat("[out] figure3_v3_2.pdf / .png\n")
