#!/usr/bin/env Rscript
# the merged figure 3
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork)})
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
FCOL<-c(MaxEnt=INK,SpliceAI=NUL)
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

RAT<-rd("ratio.tsv"); BCS<-rd("bcshow_ratio.tsv"); BCA<-rd("bcshow_auroc.tsv")
CC<-rd("class_counts.tsv")
SAW<-rdp("sa_auroc_window_final.tsv"); SAT<-rdp("sa_tertile_final.tsv")
HM<-rdp("heat_maxent.tsv"); ST<-rdp("stats_forest.tsv")
MG<-rd("metagene.tsv")

ncan<-as.numeric(CC$n[CC$cls=="Canonical"][1]); ncomp<-as.numeric(CC$n[CC$cls=="Competitor"][1])
nlab_cc<-sprintf("n = %s canon / %s comp",format(ncan,big.mark=","),format(ncomp,big.mark=","))
b<-HM$base_rate[1]
tstrong<-SAT$auroc_test[SAT$acc_tertile=="strong" & SAT$donor_reduction=="sum"][1]
getv<-function(lab,sc,col="estimate") ST[[col]][ST$effect_label==lab & ST$scorer==sc][1]

D <-HM%>%filter(n>=E_MIN)%>%mutate(f=log2_or)
LIMF<-max(abs(D$f),na.rm=TRUE)

message(sprintf("[base rate] %.5f",b))
message(sprintf("[F] %d cells, limit +/-%.3f",nrow(D),LIMF))
message(sprintf("[G tertile] weak/medium/strong donor AUROC = %.4f / %.4f / %.4f",
        SAT$auroc_test[SAT$acc_tertile=="weak"&SAT$donor_reduction=="sum"][1],
        SAT$auroc_test[SAT$acc_tertile=="medium"&SAT$donor_reduction=="sum"][1],
        SAT$auroc_test[SAT$acc_tertile=="strong"&SAT$donor_reduction=="sum"][1]))
for(i in seq_len(nrow(ST)))
  message(sprintf("[forest] %-8s %-30s %-9s %8.4f [%7.4f,%7.4f] %s",
          ST$block[i],ST$effect_label[i],ST$scorer[i],ST$estimate[i],ST$ci_lo[i],ST$ci_hi[i],ST$p_label[i]))

# a b c: mechanism bubbles
bubble<-function(d,xcol,ycol,thr,letter,sub,ylab){
  lim<-quantile(abs(d$log2ratio),0.98,na.rm=TRUE); d$clip<-pmax(pmin(d$log2ratio,lim),-lim)
  ggplot(d,aes(.data[[xcol]],.data[[ycol]]))+
    geom_point(aes(fill=clip,size=n),shape=21,color="grey55",stroke=.15)+
    scale_size(range=c(0.3,5.5),trans="sqrt",name="peaks / cell")+
    geom_hline(yintercept=thr,linetype=2,color=INK,linewidth=.3)+
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
mk<-function(m,letter,ylab,encl){
  d<-BCS[BCS$mech==m,]; pk<-max(BCA$auroc[BCA$mech==m],na.rm=TRUE)
  bubble(d,"window","yval",d$strong_thr[1],letter,
         sprintf("Suppression signal, predicts the competitor\n%s.  Peak AUROC %.3f.",encl,pk),ylab)
}
pB<-mk("u1","B","U1 sites/kb (negated ΔG ≥ 4.0)","U1 sites/kb, negated ΔG ≥ 4.0")
pC<-mk("donor","C","5′ss sites/kb (MaxEnt ≥ 4)","5′ss sites/kb, MaxEnt ≥ 4")

# d: maxent auroc vs window (oriented)
BCA$mlab<-factor(MLAB[BCA$mech],levels=MLAB)
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
       x="upstream window (nt)",y="AUROC (oriented)",color=NULL)+th+
  theme(legend.position="bottom",legend.title=element_blank())

# e: spliceai auroc vs window (raw)
SAW$slab<-factor(SLAB[SAW$signal],levels=SLAB)
pkE<-SAW%>%group_by(signal)%>%slice_max(auroc_test,n=1,with_ties=FALSE)%>%ungroup()
sa_at<-function(sig) SAW$auroc_test[SAW$signal==sig & SAW$win_far==max(SAW$win_far)][1]
ann<-paste(c(sprintf("MaxEnt acceptor    %.3f",accpk),
             sprintf("MaxEnt 5′ss        %.3f",max(BCA$auroc[BCA$mech=="donor"],na.rm=TRUE)),
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

# f and g: interaction heatmaps, independent scales
pF<-ggplot(D,aes(interv,acceptor,fill=f))+
  geom_tile(color="grey92",linewidth=.1)+
  scale_fill_gradient2(low=COMP,mid="white",high=CAN,midpoint=0,limits=c(-LIMF,LIMF),name=FILLNAME)+
  scale_x_continuous(expand=c(0,0))+scale_y_continuous(expand=c(0,0))+
  labs(title="F   Acceptor × donor interaction   ·   MaxEnt",
       subtitle=sprintf("At every acceptor level a stronger intervening 5′ss erodes real-end enrichment toward baseline; the effect is uniform\nacross acceptor strength (interaction null, %s). fill = log2 odds ratio of P(real end) vs the %.1f%% base rate; cells >=%d sites; fill range ±%.2f.",
                        getv("acceptor x donor","MaxEnt","p_label"),100*b,E_MIN,LIMF),
       x="best intervening 5′ss (MaxEnt)",y="anchor acceptor 3′ss")+th+
  theme(legend.position="right")

TT<-SAT[SAT$donor_reduction=="sum",]
TT$acc_tertile<-factor(TT$acc_tertile,levels=c("strong","medium","weak"))
hm_se<-function(A,n1,n2){
  Q1<-A/(2-A); Q2<-2*A*A/(1+A)
  sqrt((A*(1-A)+(n1-1)*(Q1-A*A)+(n2-1)*(Q2-A*A))/(n1*n2))
}
TT$se <-mapply(hm_se,TT$auroc_test,TT$n_canonical,TT$n-TT$n_canonical)
TT$lo <-pmax(0,TT$auroc_test-1.96*TT$se); TT$hi<-pmin(1,TT$auroc_test+1.96*TT$se)
TT$dir<-ifelse(TT$auroc_test>0.5,"donor marks real end","donor marks competitor")
TT$lab<-sprintf("%.2f  %s",TT$auroc_test,TT$dir)
tw<-TT$auroc_test[TT$acc_tertile=="weak"][1]
tm<-TT$auroc_test[TT$acc_tertile=="medium"][1]
ts<-TT$auroc_test[TT$acc_tertile=="strong"][1]
pG<-ggplot(TT,aes(auroc_test,acc_tertile,color=dir))+
  annotate("rect",xmin=0.5,xmax=Inf,ymin=-Inf,ymax=Inf,fill=CAN,alpha=.05)+
  annotate("rect",xmin=-Inf,xmax=0.5,ymin=-Inf,ymax=Inf,fill=COMP,alpha=.05)+
  geom_vline(xintercept=0.5,linetype=3,color=NUL)+
  annotate("text",x=0.5,y=0.42,label="chance",size=2.6,color="#555555",vjust=1.1)+
  geom_errorbar(aes(xmin=lo,xmax=hi),orientation="y",width=.14,linewidth=.6)+
  geom_point(size=3)+
  geom_text(aes(label=lab,hjust=ifelse(auroc_test>0.5,1.10,-0.10)),
            vjust=-1.15,size=2.9,show.legend=FALSE)+
  scale_color_manual(values=setNames(c(CAN,COMP),
                     c("donor marks real end","donor marks competitor")),guide="none")+
  scale_x_continuous(limits=c(0.35,0.72),breaks=seq(0.4,0.7,0.05))+
  scale_y_discrete(expand=expansion(add=c(0.75,0.75)))+
  labs(title="G   SpliceAI: at strong acceptor, the donor marks the competitor",
       subtitle=sprintf(paste0("Donor AUROC within acceptor tertiles. As the acceptor strengthens the donor flips from marking the real end to\n",
                "marking the competitor (%.2f -> %.2f -> %.2f), the same conditional suppression as F. Tertiles cut on train, scored on held-out test."),
                tw,tm,ts),
       x="donor AUROC within tertile (held-out test, raw)",y="anchor acceptor tertile")+th

sq<-function(xmin,xmax,ymin,ymax,fill) annotate("rect",xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,
                                                fill=fill,color=INK,linewidth=.3)
ARR<-arrow(length=unit(0.17,"cm"),type="closed")
pH<-ggplot()+
  annotate("text",x=1.3,y=8.75,label="REAL 3' END",hjust=0,size=3.5,fontface="bold",color=CAN)+
  sq(3,6.4,6.7,7.7,CAN)+
  annotate("segment",x=1.3,xend=2.85,y=7.2,yend=7.2,color=INK,linewidth=.7,arrow=ARR)+
  annotate("text",x=2.05,y=7.95,label="strong 3'ss",size=3.1,color=INK)+
  annotate("segment",x=6.4,xend=6.4,y=7.7,yend=8.4,color=INK,linewidth=.5)+
  annotate("point",x=6.4,y=8.4,size=2.6,color=INK)+
  annotate("text",x=6.4,y=8.85,label="poly-A",size=3.1,color=INK)+
  annotate("text",x=1.3,y=5.95,hjust=0,size=3.1,color="#555555",
           label="strong 3'ss, no downstream 5'ss  ->  transcript ends here")+
  annotate("text",x=1.3,y=4.75,label="COMPETITOR",hjust=0,size=3.5,fontface="bold",color=COMP)+
  sq(3,6.4,2.5,3.5,COMP)+sq(8.7,10.4,2.5,3.5,COMP)+
  annotate("segment",x=1.3,xend=2.85,y=3.0,yend=3.0,color=INK,linewidth=.7,arrow=ARR)+
  annotate("text",x=2.05,y=3.75,label="strong 3'ss",size=3.1,color=INK)+
  annotate("segment",x=5.6,xend=5.6,y=3.5,yend=4.1,color=INK,linewidth=.5)+
  annotate("point",x=5.6,y=4.1,size=2.6,color=INK)+
  annotate("text",x=5.6,y=4.45,label="pA",size=3.1,color=INK)+
  annotate("segment",x=6.4,xend=7.25,y=3.0,yend=3.0,color=INK,linewidth=.7,arrow=ARR)+
  annotate("text",x=6.85,y=2.3,label="5'ss",size=3.1,color=INK)+
  annotate("curve",x=6.5,xend=8.6,y=3.55,yend=3.55,curvature=-0.55,color=INK,linewidth=.5,linetype=2)+
  annotate("text",x=7.55,y=4.9,label="splice",size=2.9,color="#555555")+
  annotate("text",x=1.3,y=1.75,hjust=0,size=3.1,color="#555555",lineheight=1.15,
           label="strong 3'ss AND a downstream 5'ss  ->  loses to an\ninternal exon that splices through")+
  coord_cartesian(xlim=c(0.9,10.7),ylim=c(1.1,9.4),clip="off")+
  labs(title="H   The splicing logic")+
  theme_void(base_size=BASE)+
  theme(plot.title=element_text(size=BASE,face="bold",hjust=0,color=INK),
        plot.margin=margin(6,10,6,10))

EF<-ST%>%filter(block=="effect",effect_label!="acceptor x donor")%>%
    mutate(lab=paste0(effect_label,"   ",scorer),
           dir=ifelse(grepl("^acceptor",effect_label),"real end","competitor"))
EF$lab<-factor(EF$lab,levels=rev(EF$lab))
RK<-ST%>%filter(block=="ranking")%>%mutate(lab=paste0("acceptor 3'ss   ",scorer))
RK$lab<-factor(RK$lab,levels=rev(RK$lab))
ef_txt<-max(EF$ci_hi)*1.35
ef_lo <-min(EF$ci_lo)*0.80
pI1<-ggplot(EF,aes(estimate,lab,color=dir))+
  annotate("rect",xmin=1,xmax=ef_txt*3,ymin=-Inf,ymax=Inf,fill=CAN,alpha=.05)+
  annotate("rect",xmin=ef_lo*0.5,xmax=1,ymin=-Inf,ymax=Inf,fill=COMP,alpha=.05)+
  geom_vline(xintercept=1,linetype=3,color=NUL)+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),orientation="y",width=.18,linewidth=.6)+
  geom_point(size=2.4)+
  geom_text(aes(x=ef_txt,label=p_label),hjust=0,size=2.7,color="#555555")+
  annotate("text",x=1.06,y=0.40,hjust=0,size=2.7,color="#555555",label="->  toward real 3' end")+
  annotate("text",x=0.94,y=0.40,hjust=1,size=2.7,color="#555555",label="toward competitor  <-")+
  scale_color_manual(values=c("real end"=CAN,competitor=COMP),guide="none")+
  scale_x_log10(breaks=c(0.5,0.75,1,1.5,2,3))+
  scale_y_discrete(expand=expansion(add=c(0.95,0.6)))+
  coord_cartesian(xlim=c(ef_lo,ef_txt*2.4),clip="off")+
  labs(title="I   Effect sizes and robustness",
       subtitle=paste0("Two forces on the same odds axis (null 1). A stronger acceptor 3′ss raises the odds of a\n",
                       "real end (blue, right); at fixed acceptor a stronger 5′ss lowers them (orange, left).\n",
                       "Both scorers, both significant."),
       x="odds ratio per SD (log scale)",y=NULL)+th+
  theme(axis.text.y=element_text(size=BASE-5),plot.margin=margin(6,8,2,6))
rk_txt<-max(RK$ci_hi)+0.055
pI2<-ggplot(RK,aes(estimate,lab,color=scorer))+
  geom_vline(xintercept=0.5,linetype=3,color=NUL)+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),orientation="y",width=.14,linewidth=.6)+
  geom_point(size=2.4)+
  geom_text(aes(x=rk_txt,label=p_label),hjust=0,size=2.7,color="#555555")+
  scale_color_manual(values=FCOL,guide="none")+
  coord_cartesian(xlim=c(0.46,rk_txt+0.115),clip="off")+
  labs(subtitle="Within-gene win-rate (P real end outranks competitor)",x="win-rate",y=NULL)+th+
  theme(axis.text.y=element_text(size=BASE-5),plot.margin=margin(2,8,6,6))
pI<-pI1/pI2+plot_layout(heights=c(1.7,1))

MG$cls<-factor(MG$cls,levels=c("Canonical","Competitor"))
for(k in c("Canonical","Competitor"))
  message(sprintf("[J] %s metagene n5_per_site: upstream %.5f, downstream %.5f",k,
          mean(MG$n5_per_site[MG$cls==k & MG$pos<0]),mean(MG$n5_per_site[MG$cls==k & MG$pos>0])))
pJ<-ggplot(MG,aes(pos,n5_per_site,color=cls))+
  annotate("rect",xmin=0,xmax=Inf,ymin=-Inf,ymax=Inf,fill=NUL,alpha=.06)+
  geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.4)+
  geom_line(linewidth=1)+
  scale_color_manual(values=c(Canonical=CAN,Competitor=COMP))+
  labs(title="J   Where the competing 5′ss sits (metagene)",
       subtitle="Strong 5′ss density around the pA site (MaxEnt >= 8): competitors carry more strong 5′ss upstream; downstream the classes are equal.",
       x="position relative to center site (nt)",y="strong 5′ss density (per site)")+th+
  theme(legend.position="top",legend.title=element_blank())

fmt<-function(x) sprintf("%.2f",x)
sub<-sprintf(paste0("MaxEnt (hand-built, local) and SpliceAI (learned, 10 kb context) agree. Acceptor 3′ss raises the odds of a real end (OR %s / %s per SD);\n",
  "at fixed acceptor a stronger 5′ss lowers them (donor OR %s / %s per SD). Within genes the real end outranks its competitors %.0f%% / %.0f%% of the time.\n",
  "SpliceAI shown as corroboration, not an independent mechanistic test. n = %s real ends / %s competitors."),
  fmt(getv("acceptor 3'ss","MaxEnt")),fmt(getv("acceptor 3'ss","SpliceAI")),
  fmt(getv("donor 5'ss | acceptor","MaxEnt")),fmt(getv("donor 5'ss | acceptor","SpliceAI")),
  100*getv("within-gene win-rate, acceptor","MaxEnt"),
  100*getv("within-gene win-rate, acceptor","SpliceAI"),
  format(ncan,big.mark=","),format(ncomp,big.mark=","))
cap<-sprintf(paste0("Interaction is scorer-dependent: significant for SpliceAI (%s), null for MaxEnt (%s); the shared, robust claim is the acceptor main effect\n",
  "and the conditional donor suppression. MaxEnt and SpliceAI use different score scales; the shared feature is the pattern."),
  getv("acceptor x donor","SpliceAI","p_label"),getv("acceptor x donor","MaxEnt","p_label"))

fig<-(pA|pB|pC)/(pD|pE)/(pF|pG)/((pH|pI)+plot_layout(widths=c(0.95,1.05)))/pJ+
  plot_layout(heights=c(1,1,1.15,1,1.0))+
  plot_annotation(
    title="A strong 3′ss defines the real 3′ end; a downstream 5′ss competes it back toward baseline",
    subtitle=sub,caption=cap,
    theme=theme(plot.title=element_text(face="bold",size=BASE+2,hjust=0,color=INK),
                plot.subtitle=element_text(size=BASE-3,color="#555555"),
                plot.caption=element_text(size=BASE-5,color="#666666",hjust=0)))
ggsave(file.path(INDIR,"figure3_merged.pdf"),fig,width=17,height=23,device=cairo_pdf)
ggsave(file.path(INDIR,"figure3_merged.png"),fig,width=17,height=23,dpi=300)
cat(sprintf("[fill] F fill limit +/-%.3f; G is the tertile plot (no fill scale)\n",LIMF))
cat("[out] figure3_merged.pdf / .png\n")
