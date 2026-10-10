#!/usr/bin/env Rscript
# maxent vs spliceai on the same discrimination
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork);library(scales)})
args<-commandArgs(trailingOnly=TRUE)
INDIR<-if(length(args)>=1) args[1] else "."
PFX  <-if(length(args)>=2) args[2] else "fig3"
rd <-function(f){p<-file.path(INDIR,paste0(PFX,"_",f)); if(file.exists(p)) read_tsv(p,show_col_types=FALSE) else NULL}
rdp<-function(f){p<-file.path(INDIR,f);                 if(file.exists(p)) read_tsv(p,show_col_types=FALSE) else NULL}

INK<-"#1A1A1A"; BASE<-13; CAN<-"#0072B2"; COMP<-"#E69F00"; NUL<-"#8A8F98"
MECHS<-c("acceptor","u1","donor")
MLAB<-c(acceptor="Acceptor 3′ss  (definition)",u1="U1 hybridization  (suppression)",donor="Donor 5′ss  (suppression)")
MCOL<-c(acceptor="#0072B2",u1="#E69F00",donor="#1D9E75")
SLAB<-c(acceptor="Acceptor 3′ss  (mean over window)",donor="Donor 5′ss  (sum over window)")
SCOL<-c(acceptor="#0072B2",donor="#1D9E75")
th<-theme_classic(base_size=BASE)+
  theme(text=element_text(color=INK),axis.text=element_text(color=INK,size=BASE-4),
        axis.line=element_line(color=INK,linewidth=.4),
        plot.title=element_text(size=BASE,face="bold",hjust=0),
        plot.subtitle=element_text(size=BASE-5,color="#555555"),
        legend.title=element_blank(),legend.text=element_text(size=BASE-5),
        legend.key.height=unit(.8,"lines"))

# inputs
BCA<-rd("bcshow_auroc.tsv")
AXI<-rd("acc_x_interv.tsv")
CC <-rd("class_counts.tsv")
SAW<-rdp("auroc_window_spliceai_me.tsv")
SAD<-rdp("acc_x_don_spliceai_me.tsv")
MAR<-rdp("marginal_auroc_spliceai_me.tsv")

# class sizes and base rate
ncan<-NA; ncomp<-NA
if(!is.null(CC) && all(c("cls","n")%in%names(CC))){
  ncan <-suppressWarnings(as.numeric(CC$n[CC$cls=="Canonical"][1]))
  ncomp<-suppressWarnings(as.numeric(CC$n[CC$cls=="Competitor"][1]))
}
b<-if(!is.na(ncan)&&!is.na(ncomp)) ncan/(ncan+ncomp) else NA
if(is.na(b) && !is.null(AXI))
  b<-median((AXI$p_canonical+1e-3)/2^AXI$log2_enrich-1e-3,na.rm=TRUE)
nlab_cc<-if(!is.na(ncan)) sprintf("n = %s canon / %s comp",
            format(ncan,big.mark=","),format(ncomp,big.mark=",")) else ""
E_MIN<-as.integer(Sys.getenv("E_MIN","200"))
l2or<-function(p){p<-pmin(pmax(p,1e-4),1-1e-4); log2((p/(1-p))/(b/(1-b)))}

# a: maxent auroc vs window (oriented, as published)
pA<-ggplot()+theme_void()
if(!is.null(BCA) && nrow(BCA)>0){
  BCA$mlab<-factor(MLAB[BCA$mech],levels=MLAB)
  pkA<-BCA%>%group_by(mech)%>%slice_max(auroc,n=1,with_ties=FALSE)%>%ungroup()
  pkA_txt<-paste(sprintf("%s %.3f",c(acceptor="acceptor",u1="U1",donor="5′ss")[pkA$mech],pkA$auroc),collapse="   ")
  pA<-ggplot(BCA,aes(window,auroc,color=mlab))+
    geom_hline(yintercept=0.5,linetype=3,color=NUL)+
    annotate("text",x=max(BCA$window),y=0.5,label="chance",hjust=1,vjust=-0.5,size=2.6,color="#555555")+
    geom_line(linewidth=1)+
    geom_point(data=pkA,aes(window,auroc),size=2.4,show.legend=FALSE)+
    scale_color_manual(values=setNames(MCOL[MECHS],MLAB[MECHS]))+
    scale_y_continuous(expand=expansion(mult=.12))+
    labs(title="A  MaxEnt",
         subtitle=paste0("Hand-built motif scores. Peak AUROC:  ",pkA_txt,".\n",
                         "U1 and 5′ss predict the competitor, so their curves are ORIENTED (1 - AUROC).\n",
                         "Higher is better discrimination in all three."),
         x="upstream window (nt)",y="AUROC (oriented)")+th+
    theme(legend.position="bottom")
}

# b: spliceai auroc vs window (raw, held-out test)
pB<-ggplot()+theme_void()
if(!is.null(SAW) && nrow(SAW)>0){
  SAW$slab<-factor(SLAB[SAW$signal],levels=SLAB)
  pkB<-SAW%>%group_by(signal)%>%slice_max(auroc_test,n=1,with_ties=FALSE)%>%ungroup()
  ann<-""
  if(!is.null(MAR) && nrow(MAR)>0){
    nm<-c(maxent_acceptor="MaxEnt acceptor",maxent_donor="MaxEnt 5′ss (oriented)",
          spliceai_acceptor="SpliceAI acceptor",spliceai_donor="SpliceAI donor (raw)")
    MAR$lab<-nm[MAR$metric]
    MAR<-MAR[!is.na(MAR$lab),]
    ann<-paste(sprintf("%s  %.3f",MAR$lab,MAR$auroc),collapse="\n")
  }
  pB<-ggplot(SAW,aes(win_far,auroc_test,color=slab))+
    geom_hline(yintercept=0.5,linetype=3,color=NUL)+
    annotate("text",x=max(SAW$win_far),y=0.5,label="chance",hjust=1,vjust=-0.5,size=2.6,color="#555555")+
    geom_line(linewidth=1)+
    geom_point(data=pkB,aes(win_far,auroc_test),size=2.4,show.legend=FALSE)+
    scale_color_manual(values=setNames(SCOL[c("acceptor","donor")],SLAB[c("acceptor","donor")]))+
    scale_y_continuous(expand=expansion(mult=.22))+
    labs(title="B  SpliceAI",
         subtitle=paste0("Acceptor is the MEAN probability over the window, donor is the SUM over the window.\n",
                         "Both curves are RAW test AUROC, not oriented, so donor above 0.5 means donor is\n",
                         "higher at true ends marginally. Held-out half, gene-level 50/50 split."),
         x="upstream window (nt)",y="AUROC (raw, held-out test)")+th+
    theme(legend.position="bottom")
  if(nzchar(ann))
    pB<-pB+annotate("label",x=-Inf,y=Inf,hjust=-0.05,vjust=1.05,label=ann,
                    size=2.7,color=INK,fill="white",lineheight=1.15)
}

Cdat<-NULL; Ddat<-NULL
if(!is.null(AXI) && nrow(AXI)>0){
  Cdat<-AXI%>%filter(n>=E_MIN)%>%mutate(fill=l2or(p_canonical))
}
if(!is.null(SAD) && nrow(SAD)>0){
  Ddat<-SAD%>%filter(n>=E_MIN)%>%mutate(fill=log2_or)
}
lim<-max(c(if(!is.null(Cdat)) quantile(abs(Cdat$fill),.95,na.rm=TRUE) else NA,
           if(!is.null(Ddat)) quantile(abs(Ddat$fill),.95,na.rm=TRUE) else NA),na.rm=TRUE)
fillsc<-scale_fill_gradient2(low=COMP,mid="white",high=CAN,midpoint=0,
                            limits=c(-lim,lim),oob=squish,
                            name=expression(log[2]~"odds ratio vs base rate"))

pC<-ggplot()+theme_void()
if(!is.null(Cdat)){
  pC<-ggplot(Cdat,aes(interv,acceptor,fill=fill))+
    geom_tile(color="grey92",linewidth=.1)+fillsc+
    scale_x_continuous(expand=c(0,0))+scale_y_continuous(expand=c(0,0))+
    labs(title="C  MaxEnt: acceptor x intervening 5′ss",
         subtitle=sprintf("At fixed acceptor 3′ss, a stronger intervening 5′ss lowers P(true end).  (≥%d sites/cell; n = %s sites)",
                          E_MIN,format(sum(Cdat$n),big.mark=",")),
         x="best intervening 5′ss (MaxEnt)",y="anchor acceptor 3′ss")+th+
    theme(legend.position="right",legend.title=element_text(size=BASE-6))
}

pD<-ggplot()+theme_void()
if(!is.null(Ddat)){
  pD<-ggplot(Ddat,aes(don_bin_ctr,acc_bin_ctr,fill=fill))+
    geom_tile(color="grey92",linewidth=.1)+fillsc+
    scale_x_continuous(expand=c(0,0),breaks=seq(1,10,1))+
    scale_y_continuous(expand=c(0,0),breaks=seq(1,10,1))+
    labs(title="D  SpliceAI: acceptor x donor",
         subtitle=sprintf("Same conditioning, learned scores. At fixed acceptor, a stronger donor lowers P(true end).  (≥%d peaks/cell; n = %s peaks)",
                          E_MIN,format(sum(Ddat$n),big.mark=",")),
         x="SpliceAI donor decile (sum over 2 kb)",y="SpliceAI acceptor decile (mean over 2 kb)")+th+
    theme(legend.position="right",legend.title=element_text(size=BASE-6))
}

sub<-paste0(
 "Acceptor 3′ss is stronger at canonical true ends in both scorers (A, B). SpliceAI donor is positive marginally (B) but reverses under acceptor conditioning:\n",
 "at fixed acceptor a stronger 5′ss lowers P(true end) (D), matching the MaxEnt intervening-5′ss interaction (C). SpliceAI integrates 10 kb of context, so it is\n",
 "shown as corroboration, not as an independent mechanistic test. ",nlab_cc,".")
fig<-(pA|pB)/(pC|pD)+plot_layout(heights=c(1,1))+
  plot_annotation(
    title="MaxEnt and SpliceAI agree: acceptor 3′ss marks the true end, and at fixed acceptor a stronger donor 5′ss marks the intragenic competitor",
    subtitle=sub,
    theme=theme(plot.title=element_text(face="bold",size=BASE+1,hjust=0),
                plot.subtitle=element_text(size=BASE-3,color="#555555")))
ggsave(file.path(INDIR,"figure3_spliceai_me.pdf"),fig,width=15,height=11,device=cairo_pdf)
ggsave(file.path(INDIR,"figure3_spliceai_me.png"),fig,width=15,height=11,dpi=300)
cat(sprintf("[fill] shared log2 OR limit = %.3f, base rate b = %.5f\n",lim,b))
cat("[out] figure3_spliceai_me.pdf / .png\n")
