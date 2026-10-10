#!/usr/bin/env Rscript
# figure 3, two-layer (reinstated 2026-07-28)
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(tidyr);library(readr);library(patchwork)})
args<-commandArgs(trailingOnly=TRUE)
INDIR<-if(length(args)>=1) args[1] else "."
PFX  <-if(length(args)>=2) args[2] else "fig3"
need<-function(p){ if(!file.exists(p)) stop("missing required input: ",p,call.=FALSE); p }
rd <-function(f) read_tsv(need(file.path(INDIR,paste0(PFX,"_",f))),show_col_types=FALSE)
rdp<-function(f) read_tsv(need(file.path(INDIR,f)),show_col_types=FALSE)

INK<-"#1A1A1A"; BASE<-13; CAN<-"#0072B2"; COMP<-"#E69F00"; NUL<-"#8A8F98"
MECHS<-c("acceptor","u1","donor")
MLAB<-c(acceptor="Acceptor 3′ss  (definition)",u1="U1 hybridization  (suppression)",donor="Donor 5′ss  (suppression)")
MCOL<-c(acceptor="#0072B2",u1="#E69F00",donor="#1D9E75")
SLAB<-c(acceptor_max="Acceptor 3′ss  (max over window)",donor_sum="Donor 5′ss  (sum over window)")
SACOL<-c(acceptor_max="#0072B2",donor_sum="#1D9E75")
FCOL<-c(MaxEnt=INK,SpliceAI=NUL,U1=COMP)
th<-theme_classic(base_size=BASE)+
  theme(text=element_text(color=INK),axis.text=element_text(color=INK,size=BASE-4),
        axis.line=element_line(color=INK,linewidth=.4),
        plot.title=element_text(size=BASE,face="bold",hjust=0),
        plot.subtitle=element_text(size=BASE-5,color="#555555"),
        legend.title=element_text(size=BASE-7,lineheight=1.05),
        legend.text=element_text(size=BASE-5),
        legend.key.height=unit(.8,"lines"))
FILLNAME<-"enriched for\nreal 3' ends (blue)\n<-   ->\ncompetitors (orange)"
E_MIN<-as.integer(Sys.getenv("E_MIN","200"))

# dependencies (checked up front)
RAT<-rd("ratio.tsv"); BCS<-rd("bcshow_ratio.tsv"); BCA<-rd("bcshow_auroc.tsv")
CC <-rd("class_counts.tsv")
SAW<-rdp("sa_auroc_window_final.tsv"); ST<-rdp("stats_forest.tsv")
HERO<-rdp("hero_within_gene_525.tsv"); ACU<-rdp("acc_x_u1_525.tsv"); FU1<-rdp("forest_u1_525.tsv")
PMX<-rdp("paired_u1_excess_525.tsv")
BCS_U1<-rdp("fig3rev_bcshow_ratio.tsv")
BCA_U1<-rdp("fig3rev_bcshow_auroc.tsv")

ncan<-as.numeric(CC$n[CC$cls=="Canonical"][1]); ncomp<-as.numeric(CC$n[CC$cls=="Competitor"][1])
nlab_cc<-sprintf("n = %s canon / %s comp",format(ncan,big.mark=","),format(ncomp,big.mark=","))
b<-ACU$base_rate[1]
getv <-function(lab,sc,col="estimate") ST[[col]][ST$effect_label==lab & ST$scorer==sc][1]
getu <-function(lab,col="estimate") FU1[[col]][FU1$effect_label==lab][1]

HS<-HERO%>%filter(gene=="__SUMMARY__")%>%
    mutate(across(c(winrate_expected,pair_wtd,wilcoxon_p),as.numeric))
HG<-HERO%>%filter(gene!="__SUMMARY__")%>%
    mutate(across(c(canonical_val,competitor_val,canonical_z,competitor_z),as.numeric))
hw<-function(f) HS$winrate_expected[HS$feature==f][1]
hp<-function(f) HS$p_label[HS$feature==f][1]
ngene<-HS$n_genes[HS$feature=="acc_MaxEnt"][1]
npair<-HS$n_pairs[HS$feature=="acc_MaxEnt"][1]

# a b c: mechanism bubbles
bubble<-function(d,xcol,ycol,thr,letter,sub,ylab){
  lim<-quantile(abs(d$log2ratio),0.98,na.rm=TRUE); d$clip<-pmax(pmin(d$log2ratio,lim),-lim)
  ggplot(d,aes(.data[[xcol]],.data[[ycol]]))+
    geom_point(aes(fill=clip,size=n),shape=21,color="grey55",stroke=.15)+
    scale_size(range=c(0.3,5.5),trans="sqrt",name="peaks / cell")+
    {if(is.finite(thr)) geom_hline(yintercept=thr,linetype=2,color=INK,linewidth=.3)}+
    scale_fill_gradient2(low=COMP,mid="white",high=CAN,midpoint=0,limits=c(-lim,lim),
                         name=expression(log[2]~frac(canon,comp)))+
    scale_x_continuous(expand=expansion(mult=.02))+scale_y_continuous(expand=expansion(mult=.03))+
    labs(title=letter,subtitle=sub,x="upstream window (nt)",y=ylab)+th+
    coord_cartesian(clip="off")+
    theme(legend.position="right",legend.title=element_text(size=BASE-6),
          panel.background=element_rect(fill="grey96",color=NA))
}
accpk<-max(BCA$auroc[BCA$mech=="acceptor"],na.rm=TRUE)
dA<-RAT%>%filter(mech=="acceptor",maxscore>=0)
pA<-bubble(dA,"window","maxscore",dA$strong_thr[1],"A",
           sprintf("Definition signal, predicts the true end\nRunning-max peak strength.  Peak AUROC %.3f.",accpk),
           "strongest signal so far")
# panel b: u1 bubbles at cutoff 5.25
u1pk<-max(BCA_U1$auroc[BCA_U1$mech=="u1"],na.rm=TRUE)
dB<-BCS_U1[BCS_U1$mech=="u1",]; dB$yval<-dB$yval
pB<-bubble(dB,"window","yval",NA_real_,"B",
           sprintf("Suppression signal, predicts the competitor\nU1 sites/kb, negated ΔG ≥ 5.25.  Peak AUROC %.3f.",u1pk),
           "U1 sites/kb, negated ΔG ≥ 5.25")
# panel c: donor bubbles (unchanged)
dC<-BCS[BCS$mech=="donor",]; donpk<-max(BCA$auroc[BCA$mech=="donor"],na.rm=TRUE)
pC<-bubble(dC,"window","yval",dC$strong_thr[1],"C",
           sprintf("Suppression signal, predicts the competitor\n5′ss sites/kb, MaxEnt ≥ 4.  Peak AUROC %.3f.",donpk),
           "5′ss sites/kb (MaxEnt ≥ 4)")

BCA_D<-bind_rows(BCA%>%filter(mech!="u1"),BCA_U1%>%filter(mech=="u1"))
BCA_D$mlab<-factor(MLAB[BCA_D$mech],levels=MLAB)
pkD<-BCA_D%>%group_by(mech)%>%slice_max(auroc,n=1,with_ties=FALSE)%>%ungroup()
pkD_txt<-paste(sprintf("%s %.3f",c(acceptor="acceptor",u1="U1",donor="5′ss")[pkD$mech],pkD$auroc),collapse="   ")
pD<-ggplot(BCA_D,aes(window,auroc,color=mlab))+
  geom_hline(yintercept=0.5,linetype=3,color=NUL)+
  annotate("text",x=max(BCA_D$window),y=0.5,label="chance",hjust=1,vjust=-0.5,size=2.6,color="#555555")+
  geom_line(linewidth=1)+geom_point(data=pkD,aes(window,auroc),size=2.4,show.legend=FALSE)+
  scale_color_manual(values=setNames(MCOL[MECHS],MLAB[MECHS]))+
  scale_y_continuous(expand=expansion(mult=.10))+
  labs(title="D   Discrimination vs window   ·   MaxEnt",
       subtitle=paste0("Peak AUROC:  ",pkD_txt,".   ",nlab_cc,".\n",
                       "U1 and 5′ss predict the competitor, so their curves are oriented as 1 - AUROC (U1 cutoff 5.25).\n",
                       "Line colour = signal, not class."),
       x="upstream window (nt)",y="AUROC (oriented)",color=NULL)+th+
  theme(legend.position="bottom",legend.title=element_blank())

SAW$slab<-factor(SLAB[SAW$signal],levels=SLAB)
pkE<-SAW%>%group_by(signal)%>%slice_max(auroc_test,n=1,with_ties=FALSE)%>%ungroup()
sa_at<-function(sig) SAW$auroc_test[SAW$signal==sig & SAW$win_far==max(SAW$win_far)][1]
ann<-paste(c(sprintf("MaxEnt acceptor    %.3f",accpk),
             sprintf("MaxEnt 5′ss        %.3f",donpk),
             sprintf("SpliceAI acceptor  %.3f",sa_at("acceptor_max")),
             sprintf("SpliceAI donor     %.3f",sa_at("donor_sum"))),collapse="\n")
pE<-ggplot(SAW,aes(win_far,auroc_test,color=slab))+
  geom_hline(yintercept=0.5,linetype=3,color=NUL)+
  annotate("text",x=max(SAW$win_far),y=0.5,label="chance",hjust=1,vjust=-0.5,size=2.6,color="#555555")+
  geom_line(linewidth=1)+geom_point(data=pkE,aes(win_far,auroc_test),size=2.4,show.legend=FALSE)+
  scale_color_manual(values=setNames(SACOL[c("acceptor_max","donor_sum")],SLAB[c("acceptor_max","donor_sum")]))+
  scale_y_continuous(expand=expansion(mult=.22))+
  labs(title="E   Discrimination vs window   ·   SpliceAI (held out)",
       subtitle=paste0("Acceptor is the max probability over the window, donor is the sum over the window.\n",
                       "Both curves are raw test AUROC with canonical as the positive class, not oriented.\n",
                       "Line colour = signal, not class."),
       x="upstream window (nt)",y="AUROC (raw, held-out test)",color=NULL)+th+
  theme(legend.position="bottom",legend.title=element_blank())+
  annotate("label",x=-Inf,y=Inf,hjust=-0.05,vjust=1.05,label=ann,size=2.7,
           color=INK,fill="white",lineheight=1.15)

ngp<-PMX$n_genes[1]; nboot<-PMX$n_boot[1]
upx  <-mean(PMX$excess_mean[PMX$pos_bin<0])
uplo <-mean(PMX$ci_lo[PMX$pos_bin<0]); uphi<-mean(PMX$ci_hi[PMX$pos_bin<0])
nabove<-sum(PMX$ci_lo[PMX$pos_bin<0]>0); nupb<-sum(PMX$pos_bin<0)
message(sprintf("[F] paired upstream excess %+.2f sites/kb, CI [%+.2f, %+.2f]; above zero %d/%d bins",
        upx,uplo,uphi,nabove,nupb))
REF<-data.frame(pos_bin=PMX$pos_bin,y=0)
pF<-ggplot(PMX,aes(pos_bin,excess_mean))+
  annotate("rect",xmin=0,xmax=Inf,ymin=-Inf,ymax=Inf,fill=NUL,alpha=.06)+
  geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.4)+
  geom_ribbon(aes(ymin=ci_lo,ymax=ci_hi),fill=COMP,alpha=.25)+
  geom_line(data=REF,aes(pos_bin,y),color=CAN,linewidth=.6)+
  geom_line(color=COMP,linewidth=1)+
  annotate("text",x=-495,y=0,hjust=0,vjust=-0.6,size=2.6,color=CAN,
           label="real end, its own baseline")+
  annotate("text",x=6,y=Inf,hjust=0,vjust=1.6,label="downstream of the pA",size=2.6,color="#555555")+
  scale_x_continuous(breaks=seq(-500,500,250))+
  labs(title="F   Within a gene, competitors sit under more U1",
       subtitle=sprintf(paste0("Each gene's real end is its own baseline (blue flat at zero); orange is the competitor's pA-proximal U1 excess over that end,\n",
                               "gene-averaged with a %s resample bootstrap CI. Competitors carry %+.1f sites/kb more U1 upstream (CI %+.1f to %+.1f,\n",
                               "above zero in %d of %d bins). Within-gene win-rate %.3f (excess +0.094); this is a COMPARTMENT contrast, holding inside the\n",
                               "terminal-exon stratum (TERM-TERM excess +0.049, CI excludes 0.5), not a per-site mark (competitors match intronic decoys, see caption). n = %s genes."),
                        format(nboot,big.mark=","),upx,uplo,uphi,nabove,nupb,
                        hw("u1_pasprox"),format(ngp,big.mark=",")),
       x="position relative to the pA site (nt)",
       y="U1 excess over own real end (sites/kb)")+th

heat<-function(d,lim,letter,ttl,sub,xlab)
  ggplot(d,aes(x,acceptor,fill=f))+
    geom_tile(color="grey92",linewidth=.1)+
    scale_fill_gradient2(low=COMP,mid="white",high=CAN,midpoint=0,limits=c(-lim,lim),name=FILLNAME)+
    scale_x_continuous(expand=c(0,0))+scale_y_continuous(expand=c(0,0))+
    labs(title=paste0(letter,"   ",ttl),subtitle=sub,x=xlab,y="anchor acceptor 3′ss")+th+
    theme(legend.position="right")
DG<-ACU%>%filter(n>=E_MIN)%>%mutate(f=log2_enrich,x=u1)
LIMG<-max(abs(DG$f),na.rm=TRUE)
pG<-heat(DG,LIMG,"G","Acceptor × U1 (telescripting)   ·   MaxEnt",
  sprintf(paste0("More PAS-proximal U1 marks the competitor, and it bites hardest where the acceptor is weak: within-gene win-rate\n",
                 "%.2f at a weak acceptor and %.2f at a strong one (interaction OR %.2f, %s). Overall win-rate %.2f.\n",
                 "Fill as in F; cells >=%d sites; fill range ±%.2f."),
          getu("U1 win-rate, weak acceptor"),getu("U1 win-rate, strong acceptor"),
          getu("acceptor x U1"),getu("acceptor x U1","p_label"),
          hw("u1_pasprox"),E_MIN,LIMG),
  "PAS-proximal U1 sites per kb (pA-500 to pA, negated ΔG ≥ 5.25)")
message(sprintf("[G] %d cells, limit +/-%.3f",nrow(DG),LIMG))

sq<-function(xmin,xmax,ymin,ymax,fill) annotate("rect",xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,
                                                fill=fill,color=INK,linewidth=.3)
ARR<-arrow(length=unit(0.17,"cm"),type="closed")
u1mark<-function(xs,y) annotate("point",x=xs,y=rep(y,length(xs)),shape=18,size=2.3,color=COMP)
pH<-ggplot()+
  annotate("text",x=1.3,y=8.95,label="REAL 3' END",hjust=0,size=3.5,fontface="bold",color=CAN)+
  sq(3,6.4,6.9,7.9,CAN)+
  annotate("segment",x=1.3,xend=2.85,y=7.4,yend=7.4,color=INK,linewidth=.7,arrow=ARR)+
  annotate("text",x=2.05,y=8.15,label="strong 3'ss",size=3.1,color=INK)+
  annotate("segment",x=6.4,xend=6.4,y=7.9,yend=8.55,color=INK,linewidth=.5)+
  annotate("point",x=6.4,y=8.55,size=2.6,color=INK)+
  annotate("text",x=7.15,y=8.55,label="poly-A",size=3.1,color=INK)+
  u1mark(c(7.2),6.45)+
  annotate("text",x=7.5,y=6.45,hjust=0,label="terminal exon, low U1",size=2.9,color="#555555")+
  annotate("text",x=1.3,y=5.75,hjust=0,size=3.1,color="#555555",
           label="strong 3'ss in a U1-poor compartment  ->  transcript ends here")+
  annotate("text",x=1.3,y=4.65,label="COMPETITOR",hjust=0,size=3.5,fontface="bold",color=COMP)+
  sq(3,6.4,2.6,3.6,COMP)+
  annotate("segment",x=1.3,xend=2.85,y=3.1,yend=3.1,color=INK,linewidth=.7,arrow=ARR)+
  annotate("text",x=2.05,y=3.85,label="strong 3'ss",size=3.1,color=INK)+
  annotate("segment",x=6.4,xend=6.4,y=3.6,yend=4.2,color=INK,linewidth=.5)+
  annotate("point",x=6.4,y=4.2,size=2.6,color=INK)+
  annotate("text",x=6.4,y=4.55,label="pA",size=3.1,color=INK)+
  u1mark(c(5.15,5.65,6.15,6.65,7.15,7.65),2.15)+
  annotate("text",x=7.95,y=2.15,hjust=0,label="intronic, U1-rich compartment",size=2.9,color=COMP)+
  annotate("segment",x=6.4,xend=9.4,y=3.1,yend=3.1,color=INK,linewidth=.7,arrow=ARR,linetype=1)+
  annotate("text",x=6.6,y=3.55,hjust=0,label="splicing-competent, transcription reads on",size=2.9,color="#555555")+
  annotate("text",x=1.3,y=1.35,hjust=0,size=3.1,color="#555555",lineheight=1.15,
           label="strong 3'ss in a U1-rich intronic compartment  ->  the site\nloses to the real end downstream")+
  coord_cartesian(xlim=c(0.9,11.6),ylim=c(0.9,9.5),clip="off")+
  labs(title="H   The two-layer logic")+
  theme_void(base_size=BASE)+
  theme(plot.title=element_text(size=BASE,face="bold",hjust=0,color=INK),
        plot.margin=margin(6,10,6,10))

FLAB<-c(acc_MaxEnt="Acceptor 3'ss  ·  MaxEnt",
        acc_SpliceAI="Acceptor 3'ss  ·  SpliceAI",
        u1_pasprox="PAS-proximal U1  (5.25)")
set.seed(0)
gsamp<-sample(unique(HG$gene),min(150,length(unique(HG$gene))))
samp<-HG%>%filter(gene %in% gsamp)
LONG<-samp%>%
  select(gene,feature,competitor=competitor_z,`real end`=canonical_z)%>%
  pivot_longer(c(competitor,`real end`),names_to="end",values_to="z")
MEAN<-HG%>%group_by(feature)%>%
  summarise(competitor=mean(competitor_z),`real end`=mean(canonical_z),.groups="drop")%>%
  pivot_longer(c(competitor,`real end`),names_to="end",values_to="z")
ANN<-HS%>%transmute(feature,
                    lab=sprintf("win-rate %.2f   %s",winrate_expected,p_label),
                    col=ifelse(expected_direction=="canonical",CAN,COMP))
LONG$flab<-factor(FLAB[LONG$feature],levels=FLAB)
MEAN$flab<-factor(FLAB[MEAN$feature],levels=FLAB)
ANN$flab <-factor(FLAB[ANN$feature], levels=FLAB)
LONG$end<-factor(LONG$end,levels=c("competitor","real end"))
MEAN$end<-factor(MEAN$end,levels=c("competitor","real end"))
LONG$col<-ifelse(LONG$feature=="u1_pasprox",COMP,CAN)
pI<-ggplot(LONG,aes(end,z,group=gene))+
  geom_hline(yintercept=0,linetype=3,color=NUL)+
  geom_line(aes(color=col),linewidth=.22,alpha=.18)+
  geom_line(data=MEAN,aes(end,z,group=1),color=INK,linewidth=1.5,inherit.aes=FALSE)+
  geom_point(data=MEAN,aes(end,z),color=INK,size=2.6,inherit.aes=FALSE)+
  geom_text(data=ANN,aes(x=1.5,y=Inf,label=lab,color=col),inherit.aes=FALSE,
            vjust=1.4,size=2.9,show.legend=FALSE)+
  scale_color_identity()+
  scale_y_continuous(expand=expansion(mult=c(.06,.20)))+
  facet_wrap(~flab,nrow=1)+
  labs(title="I   Within a gene: stronger 3′ss, more competitor-side U1",
       subtitle=sprintf(paste0("Each line is one gene, from its competitor mean to its real end;\n",
                               "%d genes drawn at random, bold line is the mean over all %s.\n",
                               "Gene fixed, so gene composition cannot explain it.\n",
                               "Scores are z scaled to share an axis. n = %s genes, %s pairs."),
                        length(gsamp),format(as.numeric(ngene),big.mark=","),
                        format(as.numeric(ngene),big.mark=","),format(as.numeric(npair),big.mark=",")),
       x=NULL,y="feature value (z)")+th+
  theme(strip.background=element_blank(),strip.text=element_text(size=BASE-5,color=INK,face="bold"),
        panel.spacing=unit(0.7,"lines"))

EF<-bind_rows(
  ST%>%filter(block=="effect",grepl("^acceptor 3",effect_label))%>%
     transmute(effect_label,scorer,estimate,ci_lo,ci_hi,p_label),
  FU1%>%filter(block=="effect")%>%
     transmute(effect_label,scorer,estimate,ci_lo,ci_hi,p_label))%>%
  mutate(lab=paste0(effect_label,"   ",scorer),
         dir=ifelse(estimate>1,"real end","competitor"))
EF$lab<-factor(EF$lab,levels=rev(EF$lab))
RK<-bind_rows(
  ST%>%filter(block=="ranking")%>%transmute(lab=paste0("acceptor 3'ss   ",scorer),
                                            scorer,estimate,ci_lo,ci_hi,p_label),
  FU1%>%filter(block=="ranking")%>%transmute(lab="PAS-proximal U1   U1",
                                             scorer,estimate,ci_lo,ci_hi,p_label))
RK$lab<-factor(RK$lab,levels=rev(RK$lab))
ef_txt<-max(EF$ci_hi)*1.20
ef_lo <-min(EF$ci_lo)*0.80
pJ1<-ggplot(EF,aes(estimate,lab,color=dir))+
  annotate("rect",xmin=1,xmax=ef_txt*1.55,ymin=-Inf,ymax=Inf,fill=CAN,alpha=.05)+
  annotate("rect",xmin=ef_lo,xmax=1,ymin=-Inf,ymax=Inf,fill=COMP,alpha=.05)+
  geom_vline(xintercept=1,linetype=3,color=NUL)+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),orientation="y",width=.18,linewidth=.6)+
  geom_point(size=2.4)+
  geom_text(aes(x=ef_txt,label=p_label),hjust=0,size=2.7,color="#555555")+
  annotate("text",x=1.06,y=0.40,hjust=0,size=2.7,color="#555555",label="->  toward real 3' end")+
  annotate("text",x=0.94,y=0.40,hjust=1,size=2.7,color="#555555",label="toward competitor  <-")+
  scale_color_manual(values=c("real end"=CAN,competitor=COMP),guide="none")+
  scale_x_log10(breaks=c(0.5,0.75,1,1.5,2,3))+
  scale_y_discrete(expand=expansion(add=c(0.95,0.6)))+
  coord_cartesian(xlim=c(ef_lo,ef_txt*1.55),clip="off")+
  labs(title="J   Effect sizes",
       subtitle=paste0("Three signals on one odds axis (null 1). A stronger acceptor 3′ss raises the odds of a\n",
                       "real end (blue, right); more PAS-proximal U1 (5.25) lowers them (orange, left)."),
       x="odds ratio per SD (log scale)",y=NULL)+th+
  theme(axis.text.y=element_text(size=BASE-5),plot.margin=margin(6,8,2,6))
rk_txt<-max(RK$ci_hi)+0.055
pJ2<-ggplot(RK,aes(estimate,lab,color=scorer))+
  geom_vline(xintercept=0.5,linetype=3,color=NUL)+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),orientation="y",width=.14,linewidth=.6)+
  geom_point(size=2.4)+
  geom_text(aes(x=rk_txt,label=p_label),hjust=0,size=2.7,color="#555555")+
  scale_color_manual(values=FCOL,guide="none")+
  coord_cartesian(xlim=c(0.46,rk_txt+0.115),clip="off")+
  labs(subtitle="Within-gene win-rate, each in its expected direction",x="win-rate",y=NULL)+th+
  theme(axis.text.y=element_text(size=BASE-5),plot.margin=margin(2,8,6,6))
pJ<-pJ1/pJ2+plot_layout(heights=c(1.5,1))

sub<-sprintf(paste0("A strong acceptor 3′ss marks the real end (both scorers, A to E and I). Cryptic competitors sit in a U1-richer, splicing-competent compartment\n",
  "(B, F, G, I, J): within gene, U1@5.25 win-rate %.2f (excess +0.094), composition-corrected u1_excess +0.047. This is COMPARTMENT-level suppression, holding\n",
  "inside the terminal-exon stratum (TERM-TERM +0.049 / +0.034, CI excludes 0.5); it is not a per-site mark (competitors match intronic decoys, caption). n = %s real / %s competitors."),
  hw("u1_pasprox"),format(ncan,big.mark=","),format(ncomp,big.mark=","))
cap<-paste0("Specificity limit: pA-proximal U1 is not a per-site suppression mark. Real intronic competitors carry no more composition-corrected U1 than matched non-CPA intronic\n",
  "decoys in the same gene (competitor-vs-decoy u1_excess win-rate 0.50), so the layer is compartment-level, not site-ranking. Scope of the compartment-controlled evidence: tandem\n",
  "3'UTR APA (the TERM-TERM stratum, 1,943 genes). GC is a covariate, not matched: within gene the competitor window is GC-richer (win-rate 0.62, TERM-TERM +0.099) while pooled\n",
  "between-class GC is flat (+0.001). SpliceAI's donor track does not resolve the U1 contrast. MaxEnt and U1 are annotation-free; SpliceAI is shown as acceptor corroboration.")

fig<-(pA|pB|pC)/(pD|pE)/(pF|pG)/((pH|pI|pJ)+plot_layout(widths=c(0.9,1.15,1.0)))+
  plot_layout(heights=c(1,1,1.15,1.05))+
  plot_annotation(
    title="Two layers define the real 3′ end: a strong 3′ss sets competence, and a U1-poor terminal-exon compartment licenses it against cryptic competitors",
    subtitle=sub,caption=cap,
    theme=theme(plot.title=element_text(face="bold",size=BASE+2,hjust=0,color=INK),
                plot.subtitle=element_text(size=BASE-3,color="#555555"),
                plot.caption=element_text(size=BASE-5,color="#666666",hjust=0)))
ggsave(file.path(INDIR,"figure3_v2.pdf"),fig,width=17,height=22,device=cairo_pdf)
ggsave(file.path(INDIR,"figure3_v2.png"),fig,width=17,height=22,dpi=300)
cat(sprintf("[fill] G +/-%.3f (%d cells)\n",LIMG,nrow(DG)))
cat(sprintf("[hero] %.4f / %.4f / %.4f  (MaxEnt acc, SpliceAI acc, U1@5.25)\n",
            hw("acc_MaxEnt"),hw("acc_SpliceAI"),hw("u1_pasprox")))
cat("[out] figure3_v2.pdf / .png\n")
