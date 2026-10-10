#!/usr/bin/env Rscript
# single assembled figure 3 (no new computation
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(tidyr);library(readr);library(patchwork)})
rdp<-function(f){ if(!file.exists(f)) stop("missing: ",f); read_tsv(f,show_col_types=FALSE) }
INK<-"#1A1A1A"; BASE<-12
CAN<-"#0072B2"; COMP<-"#E69F00"; BG<-"#9AA0A6"; GRN<-"#1D9E75"
th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-3),axis.title=element_text(size=BASE-2),
  axis.line=element_line(color=INK,linewidth=.4),axis.ticks=element_line(color=INK,linewidth=.4),
  plot.title=element_text(size=BASE+1,face="bold",hjust=0),
  plot.subtitle=element_text(size=BASE-3,color="#555555",lineheight=1.05),
  legend.title=element_text(size=BASE-4,lineheight=1.0),legend.text=element_text(size=BASE-4),
  legend.key.height=unit(.7,"lines"),legend.key.width=unit(.7,"lines"))
fc<-function(n) formatC(as.numeric(n),big.mark=",",format="d")
onp<-function(l,x,y,hj=0.5,sz=BASE*0.26) annotate("text",x=x,y=y,label=l,size=sz,color=INK,hjust=hj,fontface="italic")

NMIN_BUB<-100
SZ_RANGE<-c(1.1,6.0)
sz_breaks<-function(n){
  n<-n[is.finite(n)]; lo<-min(n); hi<-max(n)
  b<-unique(round(exp(seq(log(max(lo,1)),log(hi),length.out=4))))
  sort(unique(c(lo,b[b>lo&b<hi],hi)))
}
FILLNAME<-expression(atop(log[2]~frac(canonical,competitor),"blue = real end   orange = competitor"))

NN<-rdp("fig3env_groupn.tsv"); nof<-setNames(NN$n_peaks,NN$group)
NCAN<-nof["FAILURE"]; NCOMP<-nof["COMPETITOR"]; NBG<-nof["DECOY"]
xlab2<-c(canonical=sprintf("canonical\n(%s)",fc(NCAN)),competitor=sprintf("competitor\n(%s)",fc(NCOMP)))
ST<-rdp("stats_forest.tsv") %>% filter(scorer=="MaxEnt")
ef<-function(lab) ST %>% filter(effect_label==lab)
e_acc<-ef("acceptor 3'ss"); e_don<-ef("donor 5'ss | acceptor"); e_int<-ef("acceptor x donor")
bcmp<-rdp("fig3env_panelB_fine_cmp.tsv"); acmp<-rdp("fig3env_panelA_cmp.tsv") %>% filter(comparison=="COMPETITOR - DECOY")

# a: compartment
LEV<-c("first exon","internal exon","last exon-3'UTR","first intron","internal intron","last intron","intergenic-other")
CFILL<-c("first exon"="#A6CEE3","internal exon"="#4A90C2","last exon-3'UTR"="#0072B2",
         "first intron"="#FDD9A6","internal intron"="#FDAE6B","last intron"="#E6690B","intergenic-other"="#BDBDBD")
B<-rdp("fig3env_panelB_fine.tsv") %>% mutate(group=factor(group,levels=c("canonical","competitor")),class=factor(class,levels=LEV))
pa<-ggplot(B,aes(group,frac,fill=class))+geom_col(width=.66,color=INK,linewidth=.15)+
  scale_fill_manual(values=CFILL,name="compartment",limits=LEV)+scale_x_discrete(labels=xlab2)+
  scale_y_continuous(limits=c(0,1.12),breaks=seq(0,1,.25),expand=expansion(mult=c(0,0)))+
  onp(sprintf("intronic:  competitor - canonical  +%.2f [%.2f, %.2f]",bcmp$effect,bcmp$lo,bcmp$hi),1.5,1.06)+
  labs(title="a   Competitors sit in introns, real ends in the terminal exon",
       subtitle="GENCODE compartment of the 500 nt window; blue = exon, amber = intron.",x=NULL,y="fraction of peaks")+
  th+theme(legend.position="right")

# b: atlas overlap
A<-rdp("fig3env_panelA.tsv") %>% filter(group!="SUCCESS") %>%
  mutate(group=recode(group,FAILURE="canonical",COMPETITOR="competitor",DECOY="background"),
         group=factor(group,levels=c("canonical","competitor","background")))
xlabA<-c(canonical=sprintf("canonical\n(%s)",fc(NCAN)),competitor=sprintf("competitor\n(%s)",fc(NCOMP)),
         background=sprintf("matched intronic\nbackground (%s)",fc(NBG)))
pb<-ggplot(A,aes(group,any_atlas,fill=group))+geom_col(width=.66,color=INK,linewidth=.25)+
  geom_errorbar(aes(ymin=lo,ymax=hi),width=.2,linewidth=.4,color=INK)+
  geom_text(aes(y=any_atlas+.05,label=sprintf("%.0f%%",100*any_atlas)),size=BASE*0.3,color=INK)+
  scale_fill_manual(values=c(canonical=CAN,competitor=COMP,background=BG),guide="none")+scale_x_discrete(labels=xlabA)+
  scale_y_continuous(limits=c(0,1.15),breaks=seq(0,1,.25),expand=expansion(mult=c(0,0)))+
  annotate("segment",x=2,xend=3,y=1.04,yend=1.04,color=INK,linewidth=.4)+
  onp(sprintf("competitor - background  +%.2f [%.2f, %.2f]",acmp$effect,acmp$lo,acmp$hi),2.5,1.10)+
  labs(title="b   Competitors rarely coincide with a real polyA site",
       subtitle="Fraction with any PolyASite / PolyA_DB site in the 500 nt window; canonical is definitional.",
       x=NULL,y="fraction with atlas site")+th

# c,d: bubbles (one size scale + one fill scale over both panels)
RAT<-rdp("fig3_ratio.tsv")
dC<-RAT %>% filter(mech=="acceptor",maxscore>=0,n>=NMIN_BUB) %>% rename(yv=maxscore)
BCS<-rdp("fig3rev_bcshow_ratio.tsv")
dD<-BCS %>% filter(mech=="u1",n>=NMIN_BUB) %>% rename(yv=yval)
NALL<-c(dC$n,dD$n); LIMB<-1.0
sizescale<-scale_size(range=SZ_RANGE,transform="sqrt",name="peaks / cell",
                      breaks=sz_breaks(NALL),labels=function(x) fc(x),limits=range(NALL))
divfill_bub<-scale_fill_gradient2(low=COMP,mid="white",high=CAN,midpoint=0,limits=c(-LIMB,LIMB),
                                  breaks=c(-1,-.5,0,.5,1),oob=scales::squish,name=FILLNAME)
bubble<-function(d,thr,ttl,sub,ylab){
  ggplot(d,aes(window,yv))+
    geom_point(aes(fill=log2ratio,size=n),shape=21,color="grey55",stroke=.15)+
    sizescale+divfill_bub+
    {if(is.finite(thr)) geom_hline(yintercept=thr,linetype=2,color=INK,linewidth=.3)}+
    scale_x_continuous(expand=expansion(mult=.02))+scale_y_continuous(expand=expansion(mult=.04))+
    labs(title=ttl,subtitle=sub,x="upstream window (nt)",y=ylab)+th+coord_cartesian(clip="off")+
    theme(legend.position="right",panel.background=element_rect(fill="grey96",color=NA))
}
pc<-bubble(dC,dC$strong_thr[1],
  "c   A stronger acceptor marks the real end",
  sprintf("Running-max 3'ss strength.  Acceptor OR %.2f/SD [%.2f, %.2f], peak AUROC 0.606.\nCells with at least %d peaks; area = peaks, fill = class ratio.",e_acc$estimate,e_acc$ci_lo,e_acc$ci_hi,NMIN_BUB),
  "strongest 3'ss so far")
pd<-bubble(dD,NA_real_,
  "d   U1 density is higher around competitors",
  "U1 negated dG >= 5.25, calibrated at 75% donor recall and 27x enrichment over intronic background.\nDescriptive panel, not a model feature: U1 contributes +0.0017 AUROC.  Same cell floor and scales as c.",
  "U1 sites / kb (negated dG >= 5.25)")

# e: auroc vs window, maxent only (5'ss = 4, u1 = 5.25)
BCA<-rdp("fig3_bcshow_auroc.tsv"); U1A<-rdp("fig3rev_bcshow_auroc.tsv")
D<-bind_rows(BCA %>% filter(mech!="u1"), U1A %>% filter(mech=="u1")) %>%
  mutate(signal=factor(recode(mech,acceptor="acceptor 3'ss",u1="U1 hybridization",donor="5' ss (donor)"),
                       levels=c("acceptor 3'ss","U1 hybridization","5' ss (donor)")))
SIGCOL<-c("acceptor 3'ss"=CAN,"U1 hybridization"=COMP,"5' ss (donor)"=GRN)
pk<-D %>% group_by(signal) %>% slice_max(auroc,n=1,with_ties=FALSE) %>% ungroup()
pkt<-paste(sprintf("%s %.3f",c("acceptor 3'ss"="acceptor","U1 hybridization"="U1","5' ss (donor)"="5'ss")[as.character(pk$signal)],pk$auroc),collapse="   ")
pe<-ggplot(D,aes(window,auroc,color=signal))+
  geom_hline(yintercept=0.5,linetype=3,color=BG)+
  annotate("text",x=max(D$window),y=0.5,label="chance",hjust=1,vjust=-0.5,size=2.6,color="#555555")+
  geom_line(linewidth=1)+geom_point(data=pk,aes(window,auroc),size=2.2,show.legend=FALSE)+
  scale_color_manual(values=SIGCOL,name=NULL)+scale_y_continuous(expand=expansion(mult=.10))+
  labs(title="e   The acceptor predicts the end; 5' ss and U1 predict the competitor",
       subtitle=paste0(sprintf("Peak AUROC: %s (MaxEnt only; colour = signal).\n",pkt),
                       "5' ss density MaxEnt >= 4, U1 negated dG >= 5.25; both oriented as 1 - AUROC."),
       x="upstream window (nt)",y="AUROC (oriented)")+th+theme(legend.position="bottom")

HM<-rdp("heat_maxent.tsv"); E_MIN<-200
HD<-HM %>% filter(n>=E_MIN, !is.na(log2_or))
LIMF<-max(abs(HD$log2_or),na.rm=TRUE)
pf<-ggplot(HD,aes(interv,acceptor,fill=log2_or))+geom_tile(color="grey92",linewidth=.1)+
  scale_fill_gradient2(low=COMP,mid="white",high=CAN,midpoint=0,limits=c(-LIMF,LIMF),
                       name=expression(atop(log[2]~"odds ratio","real end vs base rate")))+
  scale_x_continuous(expand=c(0,0))+scale_y_continuous(expand=c(0,0))+
  labs(title="f   An intervening 5' ss competes every acceptor level toward the competitor",
       subtitle=sprintf(paste0("Real-end enrichment by acceptor strength and best intervening 5' ss (cells >= %d peaks; fill range +/-%.2f).\n",
                       "The 5' ss shifts cells toward the competitor at every acceptor level (interaction null: OR %.2f, p = %.3f)."),
                       E_MIN,LIMF,e_int$estimate,e_int$p),
       x="best intervening 5' ss (MaxEnt)",y="anchor acceptor 3'ss")+th+theme(legend.position="right")

row1<-wrap_plots(pa,pb,ncol=2)
row2<-wrap_plots(pc,pd,ncol=2,guides="collect")
row3<-wrap_plots(pe,pf,ncol=2)
fig<-wrap_plots(row1,row2,row3,ncol=1,heights=c(0.82,1.25,1.02))+
  plot_annotation(
    title="Figure 3.  A real 3' end is a strong acceptor in a terminal-exon compartment; competitors are strong acceptors in introns",
    subtitle=sprintf("Competitors are model false positives: intragenic regions scoring higher than the gene's true CPA site under the baseline LR / RF / CNN ensemble.  n = %s canonical / %s competitor.",fc(NCAN),fc(NCOMP)),
    caption=paste0("SpliceAI is dropped: its 10 kb context means the acceptor head already encodes downstream donor presence, so it is not an independent scorer -- the acceptor x donor\n",
                   "interaction appears only when both terms are SpliceAI (0.651 / 0.531 / 0.414) and is flat with MaxEnt acceptor tertiles (0.649 / 0.655 / 0.672). U1 appears at one cutoff\n",
                   "(negated dG >= 5.25) in panels d and e. The 5' ss bubble map is not shown: 5' ss counts in a 500 nt window are integers, so its density axis is discrete and renders as\n",
                   "disconnected stripes; U1 sites are dense enough for a continuous axis. The 5' ss signal is in panels e and f, where it is a model term; U1 contributes +0.0017 AUROC."),
    theme=theme(plot.title=element_text(face="bold",size=BASE+3,hjust=0,color=INK),
                plot.subtitle=element_text(size=BASE-2,color="#555555"),
                plot.caption=element_text(size=BASE-5,color="#666666",hjust=0)))
ggsave("figure3.pdf",fig,width=16,height=15.5,device=cairo_pdf)
ggsave("figure3.png",fig,width=16,height=15.5,dpi=200)
cat(sprintf("wrote figure3.pdf / .png  (panel f: %d cells, +/-%.3f)\n",nrow(HD),LIMF))
