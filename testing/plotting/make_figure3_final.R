#!/usr/bin/env Rscript
# maxent and spliceai on one canonical-vs-competitor figure
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork);library(scales)})
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
SCOL<-c(acceptor_max="#0072B2",donor_sum="#1D9E75")
th<-theme_classic(base_size=BASE)+
  theme(text=element_text(color=INK),axis.text=element_text(color=INK,size=BASE-4),
        axis.line=element_line(color=INK,linewidth=.4),
        plot.title=element_text(size=BASE,face="bold",hjust=0),
        plot.subtitle=element_text(size=BASE-5,color="#555555"),
        legend.title=element_blank(),legend.text=element_text(size=BASE-5),
        legend.key.height=unit(.8,"lines"))
mfac<-function(x) factor(MLAB[x],levels=MLAB)

# inputs (all required
RAT<-rd("ratio.tsv")
BCS<-rd("bcshow_ratio.tsv")
BCA<-rd("bcshow_auroc.tsv")
AXI<-rd("acc_x_interv.tsv")
MG <-rd("metagene.tsv")
CC <-rd("class_counts.tsv")
SAW<-rdp("sa_auroc_window_final.tsv")
SAD<-rdp("sa_acc_x_don_final.tsv")
SAT<-rdp("sa_tertile_final.tsv")
SAM<-rdp("sa_marginal_final.tsv")

ncan <-as.numeric(CC$n[CC$cls=="Canonical"][1])
ncomp<-as.numeric(CC$n[CC$cls=="Competitor"][1])
b<-ncan/(ncan+ncomp)
nlab_cc<-sprintf("n = %s canon / %s comp",format(ncan,big.mark=","),format(ncomp,big.mark=","))
E_MIN<-as.integer(Sys.getenv("E_MIN","200"))
l2or<-function(p){p<-pmin(pmax(p,1e-4),1-1e-4); log2((p/(1-p))/(b/(1-b)))}
accpk<-max(BCA$auroc[BCA$mech=="acceptor"],na.rm=TRUE)

# a b c: mechanism bubbles
bubble<-function(d,xcol,ycol,thr,letter,sub,ylab){
  lim<-quantile(abs(d$log2ratio),0.98,na.rm=TRUE); d$clip<-pmax(pmin(d$log2ratio,lim),-lim)
  ggplot(d,aes(.data[[xcol]],.data[[ycol]]))+
    geom_point(aes(fill=clip,size=n),shape=21,color="grey55",stroke=.15)+
    scale_size(range=c(0.3,5.5),trans="sqrt",name="peaks / cell")+
    geom_hline(yintercept=thr,linetype=2,color=INK,linewidth=.3)+
    scale_fill_gradient2(low=COMP,mid="white",high=CAN,midpoint=0,limits=c(-lim,lim),
                         name=expression(log[2]~frac(canon,comp)))+
    scale_x_continuous(expand=expansion(mult=.02))+
    scale_y_continuous(expand=expansion(mult=.03))+
    labs(title=letter,subtitle=sub,x="upstream window (nt)",y=ylab)+th+
    coord_cartesian(clip="off")+
    theme(legend.position="right",legend.title=element_text(size=BASE-6),
          panel.background=element_rect(fill="grey96",color=NA))
}
dA<-RAT%>%filter(mech=="acceptor",maxscore>=0)
pA<-bubble(dA,"window","maxscore",dA$strong_thr[1],"A",
           sprintf("Definition signal, predicts the true end\nRunning-max peak strength.  Peak AUROC %.3f.",accpk),
           "strongest signal so far")
mk<-function(m,letter,ylab,encl){
  d<-BCS[BCS$mech==m,]
  pk<-max(BCA$auroc[BCA$mech==m],na.rm=TRUE)
  bubble(d,"window","yval",d$strong_thr[1],letter,
         sprintf("Suppression signal, predicts the competitor\n%s.  Peak AUROC %.3f.",encl,pk),ylab)
}
pB<-mk("u1","B","U1 sites/kb (negated ΔG ≥ 4.0)","U1 sites/kb, negated ΔG ≥ 4.0")
pC<-mk("donor","C","5′ss sites/kb (MaxEnt ≥ 4)","5′ss sites/kb, MaxEnt ≥ 4")

# d: maxent auroc vs window (oriented)
BCA$mlab<-mfac(BCA$mech)
pkD<-BCA%>%group_by(mech)%>%slice_max(auroc,n=1,with_ties=FALSE)%>%ungroup()
pkD_txt<-paste(sprintf("%s %.3f",c(acceptor="acceptor",u1="U1",donor="5′ss")[pkD$mech],pkD$auroc),collapse="   ")
pD<-ggplot(BCA,aes(window,auroc,color=mlab))+
  geom_hline(yintercept=0.5,linetype=3,color=NUL)+
  annotate("text",x=max(BCA$window),y=0.5,label="chance",hjust=1,vjust=-0.5,size=2.6,color="#555555")+
  geom_line(linewidth=1)+geom_point(data=pkD,aes(window,auroc),size=2.4,show.legend=FALSE)+
  scale_color_manual(values=setNames(MCOL[MECHS],MLAB[MECHS]))+
  scale_y_continuous(expand=expansion(mult=.10))+
  labs(title="D   Discrimination vs window   ·   MaxEnt",
       subtitle=paste0("Peak AUROC:  ",pkD_txt,".   ",nlab_cc,".\n",
                       "U1 and 5′ss predict the competitor, so their curves are oriented as 1 - AUROC.\n",
                       "Line colour = signal, not class."),
       x="upstream window (nt)",y="AUROC (oriented)")+th+theme(legend.position="bottom")

# e: spliceai auroc vs window (raw)
SAW$slab<-factor(SLAB[SAW$signal],levels=SLAB)
pkE<-SAW%>%group_by(signal)%>%slice_max(auroc_test,n=1,with_ties=FALSE)%>%ungroup()
mget1<-function(k) SAM$auroc_test[SAM$metric==k][1]
ann<-paste(c(sprintf("MaxEnt acceptor    %.3f",mget1("maxent_acceptor")),
             sprintf("MaxEnt 5′ss        %.3f",mget1("maxent_donor_oriented")),
             sprintf("SpliceAI acceptor  %.3f",mget1("acceptor_max")),
             sprintf("SpliceAI donor     %.3f",mget1("donor_sum"))),collapse="\n")
pE<-ggplot(SAW,aes(win_far,auroc_test,color=slab))+
  geom_hline(yintercept=0.5,linetype=3,color=NUL)+
  annotate("text",x=max(SAW$win_far),y=0.5,label="chance",hjust=1,vjust=-0.5,size=2.6,color="#555555")+
  geom_line(linewidth=1)+geom_point(data=pkE,aes(win_far,auroc_test),size=2.4,show.legend=FALSE)+
  scale_color_manual(values=setNames(SCOL[c("acceptor_max","donor_sum")],SLAB[c("acceptor_max","donor_sum")]))+
  scale_y_continuous(expand=expansion(mult=.22))+
  labs(title="E   Discrimination vs window   ·   SpliceAI (held out)",
       subtitle=paste0("Acceptor is the max probability over the window, donor is the sum over the window.\n",
                       "Both curves are raw test AUROC with canonical as the positive class, not oriented.\n",
                       "Line colour = signal, not class."),
       x="upstream window (nt)",y="AUROC (raw, held-out test)")+th+
  theme(legend.position="bottom")+
  annotate("label",x=-Inf,y=Inf,hjust=-0.05,vjust=1.05,label=ann,size=2.7,
           color=INK,fill="white",lineheight=1.15)

Fd<-AXI%>%filter(n>=E_MIN)%>%mutate(fill=l2or(p_canonical))
Gd<-SAD%>%filter(n>=E_MIN)%>%mutate(fill=log2_or)
limF<-max(abs(Fd$fill),na.rm=TRUE)
limG<-max(abs(Gd$fill),na.rm=TRUE)
FILLNAME<-"enriched for\ntrue ends (blue)\n←  →\ncompetitors (orange)"
fillsc<-function(lim) scale_fill_gradient2(low=COMP,mid="white",high=CAN,midpoint=0,
                                           limits=c(-lim,lim),name=FILLNAME)
DEFN<-"fill = log2 odds ratio of P(true end) vs the 9.3% base rate."
NOTCOMP<-"Panels F and G share the pattern, not the scale."
tstrong<-SAT$auroc_test[SAT$acc_tertile=="strong" & SAT$donor_reduction=="sum"][1]

pF<-ggplot(Fd,aes(interv,acceptor,fill=fill))+geom_tile(color="grey92",linewidth=.1)+fillsc(limF)+
  scale_x_continuous(expand=c(0,0))+
  scale_y_continuous(expand=expansion(mult=c(0,.10)))+
  annotate("label",x=Inf,y=Inf,hjust=1.03,vjust=1.1,label="same crossover, MaxEnt",
           size=2.7,color=INK,fill="white")+
  labs(title="F   Acceptor × donor interaction   ·   MaxEnt",
       subtitle=sprintf("At high acceptor 3′ss, a stronger intervening 5′ss lowers P(true end). At low acceptor the gradient reverses.\n%s\n%s  (≥%d sites/cell; n = %s sites; fill range ±%.2f)",
                        DEFN,NOTCOMP,E_MIN,format(sum(Fd$n),big.mark=","),limF),
       x="best intervening 5′ss (MaxEnt)",y="anchor acceptor 3′ss")+th+
  theme(legend.position="right",legend.title=element_text(size=BASE-7,lineheight=1.05))

pG<-ggplot(Gd,aes(don_bin,acc_bin,fill=fill))+geom_tile(color="grey92",linewidth=.1)+fillsc(limG)+
  scale_x_continuous(expand=c(0,0),breaks=1:10)+
  scale_y_continuous(expand=expansion(mult=c(0,.10)),breaks=1:10)+
  annotate("label",x=Inf,y=Inf,hjust=1.02,vjust=1.1,
           label=sprintf("strong acceptor + strong donor → competitor  (donor AUROC %.2f, held out)",tstrong),
           size=2.7,color=INK,fill="white")+
  labs(title="G   Acceptor × donor interaction   ·   SpliceAI (held out)",
       subtitle=sprintf("Same conditioning with learned scores, and the same crossover.\n%s\n%s  (≥%d peaks/cell; n = %s peaks; fill range ±%.2f)",
                        DEFN,NOTCOMP,E_MIN,format(sum(Gd$n),big.mark=","),limG),
       x="SpliceAI donor decile (sum over 2 kb)",y="SpliceAI acceptor decile (max over 2 kb)")+th+
  theme(legend.position="right",legend.title=element_text(size=BASE-7,lineheight=1.05))

MG$cls<-factor(MG$cls,levels=c("Canonical","Competitor"))
pH<-ggplot(MG,aes(pos,n5_per_site,color=cls))+
  annotate("rect",xmin=0,xmax=Inf,ymin=-Inf,ymax=Inf,fill=NUL,alpha=.06)+
  geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.4)+
  geom_line(linewidth=1)+scale_color_manual(values=c(Canonical=CAN,Competitor=COMP))+
  labs(title="H",
       subtitle=paste0("Strong 5′ss density around the site (MaxEnt ≥ 8): competitors carry more upstream.   ",nlab_cc),
       x="position relative to center site (nt)",y="strong 5′ss density (per site)")+th+
  theme(legend.position="top")

tdir<-SAT$direction[SAT$acc_tertile=="strong" & SAT$donor_reduction=="sum"][1]
sub<-paste0(
 "Two scorers, one conclusion. MaxEnt (hand-built, local) and SpliceAI (learned, 10 kb context) agree: a strong acceptor 3′ss marks the true end; at a\n",
 sprintf("strong acceptor, a strong donor 5′ss marks the competitor. n = %s true ends / %s competitors.",
         format(ncan,big.mark=","),format(ncomp,big.mark=",")))
cap<-sprintf(paste0("At low acceptor no exon is defined and the donor effect reverses, so F and G show a crossover, not a main effect. ",
             "Donor AUROC within the strong-acceptor tertile = %.3f (%s direction),\n",
             "tertiles cut on train and applied to the held-out half. ",
             "SpliceAI integrates 10 kb of context and is shown as corroboration, not an independent mechanistic test."),
             tstrong,tolower(tdir))
fig<-(pA|pB|pC)/(pD|pE)/(pF|pG)/(pH)+plot_layout(heights=c(1,1.1,1.1,0.8))+
  plot_annotation(
    title="MaxEnt and SpliceAI agree: the acceptor 3′ss defines the true 3′ end, and at high acceptor a stronger donor 5′ss marks the intragenic competitor",
    subtitle=sub,caption=cap,
    theme=theme(plot.title=element_text(face="bold",size=BASE+1,hjust=0),
                plot.subtitle=element_text(size=BASE-3,color="#555555"),
                plot.caption=element_text(size=BASE-5,color="#666666",hjust=0)))
ggsave(file.path(INDIR,"figure3_final.pdf"),fig,width=15,height=18,device=cairo_pdf)
ggsave(file.path(INDIR,"figure3_final.png"),fig,width=15,height=18,dpi=300)
cat(sprintf("[fill] independent limits: F ±%.3f, G ±%.3f (base rate b = %.5f)\n",limF,limG,b))
cat(sprintf("[crossover] strong-acceptor donor_sum AUROC = %.4f (%s)\n",tstrong,tdir))
cat("[out] figure3_final.pdf / .png\n")
