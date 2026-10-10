#!/usr/bin/env Rscript
# terminal-exon definition, both directions of the site
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork);library(scales)})
args<-commandArgs(trailingOnly=TRUE)
INDIR<-if(length(args)>=1) args[1] else "."
PFX  <-if(length(args)>=2) args[2] else "fig3"
rd<-function(f){p<-file.path(INDIR,paste0(PFX,"_",f)); if(file.exists(p)) read_tsv(p,show_col_types=FALSE) else NULL}

INK<-"#1A1A1A"; BASE<-13; CAN<-"#0072B2"; COMP<-"#E69F00"; NUL<-"#8A8F98"
MECHS<-c("acceptor","u1","donor")
MLAB<-c(acceptor="Acceptor 3\u2032ss  (definition)",u1="U1 hybridization  (suppression)",donor="Donor 5\u2032ss  (suppression)")
MCOL<-c(acceptor="#0072B2",u1="#E69F00",donor="#1D9E75")
th<-theme_classic(base_size=BASE)+
  theme(text=element_text(color=INK),axis.text=element_text(color=INK,size=BASE-4),
        axis.line=element_line(color=INK,linewidth=.4),
        plot.title=element_text(size=BASE,face="bold",hjust=0),
        plot.subtitle=element_text(size=BASE-5,color="#555555"),
        legend.title=element_blank(),legend.text=element_text(size=BASE-5),
        legend.key.height=unit(.8,"lines"))
mfac<-function(x) factor(MLAB[x],levels=MLAB)
wilson<-function(k,n,z=1.96){ if(n==0) return(c(NA,NA,NA)); ph<-k/n; d<-1+z*z/n
  c(ph,(ph+z*z/(2*n)-z*sqrt(ph*(1-ph)/n+z*z/(4*n*n)))/d,(ph+z*z/(2*n)+z*sqrt(ph*(1-ph)/n+z*z/(4*n*n)))/d) }

# read everything up front
RAT<-rd("ratio.tsv"); AUR<-rd("auroc.tsv"); DAU<-rd("density_auroc.tsv")
AXI<-rd("acc_x_interv.tsv"); MG<-rd("metagene.tsv")
F5<-rd("first5_from_anchor.tsv"); IEL<-rd("internal_exon_like.tsv")

# global class sizes (canonical / competitor), from the richest source
class_n<-function(){
  cc<-rd("class_counts.tsv")
  if(!is.null(cc) && all(c("cls","n")%in%names(cc))){
    can<-suppressWarnings(as.numeric(cc$n[cc$cls=="Canonical"][1]))
    comp<-suppressWarnings(as.numeric(cc$n[cc$cls=="Competitor"][1]))
    if(!is.na(can)&&!is.na(comp)) return(c(can=can,comp=comp))
  }
  if(!is.null(AXI) && all(c("n","p_canonical")%in%names(AXI)))
    return(c(can=round(sum(AXI$n*AXI$p_canonical)),comp=round(sum(AXI$n*(1-AXI$p_canonical)))))
  if(!is.null(F5) && "cls"%in%names(F5)){ tb<-table(F5$cls)
    return(c(can=as.numeric(tb["Canonical"]),comp=as.numeric(tb["Competitor"]))) }
  c(can=NA,comp=NA)
}
cn<-class_n()
nlab_cc<-if(!is.na(cn["can"])) sprintf("n = %s canon / %s comp",
              format(cn["can"],big.mark=","),format(cn["comp"],big.mark=",")) else ""
ppr<-function(p){ if(is.na(p)) "" else if(p<1e-3) "p < 0.001" else paste0("p = ",formatC(p,format="g",digits=2)) }
auroc_p<-function(A,n1,n2){
  if(any(is.na(c(A,n1,n2)))||n1<2||n2<2) return(NA)
  Q1<-A/(2-A); Q2<-2*A*A/(1+A)
  se<-sqrt((A*(1-A)+(n1-1)*(Q1-A*A)+(n2-1)*(Q2-A*A))/(n1*n2))
  if(se<=0) return(NA); 2*pnorm(-abs((A-0.5)/se))
}

# a/b/c: heatmaps
# shared bubble-size scale for a/b/c
SZ_RANGE<-c(1.1,6.0)
sz_breaks<-function(n){
  n<-n[is.finite(n)]; if(!length(n)) return(waiver())
  lo<-min(n); hi<-max(n)
  b<-unique(round(exp(seq(log(max(lo,1)),log(hi),length.out=4))))
  sort(unique(c(lo,b[b>lo&b<hi],hi)))
}
sz_report<-function(tag,n){
  n<-n[is.finite(n)]
  message(sprintf("[size %s] cells=%d  min=%g  q05=%g  q50=%g  q95=%g  max=%g",
                  tag,length(n),min(n),quantile(n,.05),median(n),quantile(n,.95),max(n)))
}
size_scale<-function(n) scale_size(range=SZ_RANGE,trans="sqrt",name="peaks / cell",
                                   breaks=sz_breaks(n),limits=range(n[is.finite(n)]))
heat<-function(m,letter){
  d<-RAT%>%filter(mech==m,maxscore>=0); if(nrow(d)==0) return(ggplot()+theme_void())
  lim<-quantile(abs(d$log2ratio),0.98,na.rm=TRUE); d$clip<-pmax(pmin(d$log2ratio,lim),-lim)
  thr<-if("strong_thr" %in% names(d)) d$strong_thr[1] else 8.0
  has_n<-"n" %in% names(d) && any(is.finite(d$n))
  p<-ggplot(d,aes(window,maxscore))
  if(has_n){
    sz_report(paste0("A/",m),d$n)
    p<-p+geom_point(aes(fill=clip,size=n),shape=21,color="grey55",stroke=.15)+
      size_scale(d$n)
  }else{
    p<-p+geom_tile(aes(fill=clip))
  }
  p+geom_hline(yintercept=thr,linetype=2,color=INK,linewidth=.3)+
    scale_fill_gradient2(low=COMP,mid="white",high=CAN,midpoint=0,limits=c(-lim,lim),
                         name=expression(log[2]~frac(canon,comp)))+
    scale_x_continuous(expand=expansion(mult=.02))+
    scale_y_continuous(expand=expansion(mult=.02))+
    labs(title=letter,subtitle=paste0(MLAB[m],"   ",nlab_cc),
         x="upstream window (nt)",y="strongest signal so far")+th+
    coord_cartesian(clip="off")+
    theme(legend.position="right",legend.title=element_text(size=BASE-6),
          panel.background=element_rect(fill="grey96",color=NA))
}
BCS<-rd("bcshow_ratio.tsv")
densheat<-function(m,letter,role,ylab,encl){
  if(is.null(BCS)) return(ggplot()+theme_void())
  d<-BCS[BCS$mech==m,]; if(nrow(d)==0) return(ggplot()+theme_void())
  lim<-quantile(abs(d$log2ratio),0.98,na.rm=TRUE); d$clip<-pmax(pmin(d$log2ratio,lim),-lim)
  thr<-d$strong_thr[1]
  pk<-if(!is.null(BCA)) max(BCA$auroc[BCA$mech==m],na.rm=TRUE) else NA
  sz_report(paste0("BC/",m),d$n)
  ggplot(d,aes(window,yval))+
    geom_point(aes(fill=clip,size=n),shape=21,color="grey55",stroke=.15)+
    size_scale(d$n)+
    scale_fill_gradient2(low=COMP,mid="white",high=CAN,midpoint=0,limits=c(-lim,lim),
                         name=expression(log[2]~frac(canon,comp)))+
    scale_x_continuous(expand=expansion(mult=.02))+
    scale_y_continuous(expand=expansion(mult=.03))+
    labs(title=letter,
         subtitle=sprintf("%s\n%s.  Peak AUROC %.3f.",role,encl,pk),
         x="upstream window (nt)",y=ylab)+th+coord_cartesian(clip="off")+
    theme(legend.position="right",legend.title=element_text(size=BASE-6),
          panel.background=element_rect(fill="grey96",color=NA))
}
BCA<-rd("bcshow_auroc.tsv")
accpk<-if(!is.null(BCA)) max(BCA$auroc[BCA$mech=="acceptor"],na.rm=TRUE) else NA
pA<-heat("acceptor","A")+labs(subtitle=sprintf("Definition signal, predicts the true end\nRunning-max peak strength.  Peak AUROC %.3f.",accpk))
pB<-densheat("u1","B","Suppression signal, predicts the competitor","U1 sites/kb, negated ΔG ≥ 4.0","U1 sites/kb (negated ΔG ≥ 4.0)")
pC<-densheat("donor","C","Suppression signal, predicts the competitor","5′ss sites/kb, MaxEnt ≥ 4","5′ss sites/kb (MaxEnt ≥ 4)")

pD<-ggplot()+theme_void()
if(!is.null(BCA) && nrow(BCA)>0){
  BCA$mlab<-mfac(BCA$mech)
  pk<-BCA%>%group_by(mech)%>%slice_max(auroc,n=1,with_ties=FALSE)%>%ungroup()
  pk_txt<-paste(sprintf("%s %.3f",c(acceptor="acceptor",u1="U1",donor="5′ss")[pk$mech],pk$auroc),collapse="   ")
  enc_txt<-"Acceptor: running-max peak strength, predicts the true end.  U1: sites/kb at negated ΔG ≥ 4.0, predicts the competitor.  5′ss: sites/kb at MaxEnt ≥ 4, predicts the competitor."
  pD<-ggplot(BCA,aes(window,auroc,color=mlab))+
    geom_hline(yintercept=0.5,linetype=3,color=NUL)+
    annotate("text",x=max(BCA$window),y=0.5,label="chance",hjust=1,vjust=-0.5,size=2.6,color="#555555")+
    geom_line(linewidth=1)+
    geom_point(data=pk,aes(window,auroc),size=2.4,show.legend=FALSE)+
    scale_color_manual(values=setNames(MCOL[MECHS],MLAB[MECHS]))+
    scale_y_continuous(expand=expansion(mult=.10))+
    labs(title="D",
         subtitle=paste0("Peak AUROC:  ",pk_txt,"   ",nlab_cc,".\n",enc_txt,"\n",
                         "All three exceed chance. Acceptor and U1 are the strong ones. Curves are oriented so higher is better discrimination."),
         x="upstream window (nt)",y="AUROC (oriented)")+th+
    theme(legend.position="bottom")
  cat(sprintf("[D] peak AUROC  %s\n",paste(sprintf("%s=%.4f@%.0fnt",pk$mech,pk$auroc,pk$window),collapse="  ")))
}

E_FILL<-Sys.getenv("E_FILL","orig")
if(!is.null(AXI) && nrow(AXI)>0){
  E_MIN<-as.integer(Sys.getenv("E_MIN","200"))
  b<-median((AXI$p_canonical + 1e-3) / 2^AXI$log2_enrich - 1e-3, na.rm=TRUE)
  AXI<-AXI%>%filter(n>=E_MIN)
  p<-pmin(pmax(AXI$p_canonical,1e-4),1-1e-4)
  AXI$f_prob<-AXI$log2_enrich
  AXI$f_or  <-log2( (p/(1-p)) / (b/(1-b)) )
  AXI<-AXI%>%group_by(acceptor)%>%
       mutate(f_or_rc=f_or-weighted.mean(f_or,n))%>%ungroup()
  fillv<-switch(E_FILL, or=AXI$f_or, or_rc=AXI$f_or_rc, AXI$f_prob)
  fillnm<-switch(E_FILL,
                 or   =expression(log[2]~"odds ratio vs base rate"),
                 or_rc=expression(log[2]~"odds ratio, within acceptor bin"),
                 expression(log[2]~frac("P(true end)","base rate")))
  ftag<-switch(E_FILL, or="  [fill: log2 odds ratio]",
               or_rc="  [fill: conditional log2 OR]", "")
  message(sprintf("[E] mode=%s  E_MIN=%d  cells kept=%d  base rate b=%.5f  (fill range %.3f to %.3f)",
                  E_FILL, E_MIN, nrow(AXI), b, min(fillv,na.rm=TRUE), max(fillv,na.rm=TRUE)))
  lim<-quantile(abs(fillv),.95,na.rm=TRUE); AXI$clip<-pmax(pmin(fillv,lim),-lim)
  nE<-sum(AXI$n)
  EC<-rd("e_coef.tsv"); ecoef<-""
  if(!is.null(EC) && nrow(EC)>0)
    ecoef<-sprintf("  (OR %.2f per SD, 95%% %.2f\u2013%.2f)", EC$OR[1], EC$ci_lo[1], EC$ci_hi[1])
  pE<-ggplot(AXI,aes(interv,acceptor,fill=clip))+
    geom_tile(color="grey92",linewidth=.1)+
    scale_fill_gradient2(low=COMP,mid="white",high=CAN,midpoint=0,limits=c(-lim,lim),
                         name=fillnm)+
    scale_x_continuous(expand=c(0,0))+scale_y_continuous(expand=c(0,0))+
    labs(title="E",subtitle=sprintf("At fixed acceptor 3\u2032ss, a stronger intervening 5\u2032ss lowers P(true end)%s  (\u2265%d sites/cell; n = %s sites)%s",
                                     ecoef, E_MIN, format(nE,big.mark=","), ftag),
         x="best intervening 5\u2032ss (MaxEnt)",y="anchor acceptor 3\u2032ss")+th+
    theme(legend.position="right",legend.title=element_text(size=BASE-6))
} else pE<-ggplot()+theme_void()

if(!is.null(MG) && nrow(MG)>0){
  MG$cls<-factor(MG$cls,levels=c("Canonical","Competitor"))
  pMeta5<-ggplot(MG,aes(pos,n5_per_site,color=cls))+
    annotate("rect",xmin=0,xmax=Inf,ymin=-Inf,ymax=Inf,fill=NUL,alpha=.06)+
    geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.4)+
    geom_line(linewidth=1)+scale_color_manual(values=c(Canonical=CAN,Competitor=COMP))+
    labs(title="F",subtitle=paste0("Strong 5\u2032ss density around the site (MaxEnt \u2265 8): competitors carry more upstream.   ",nlab_cc),
         x="position relative to center site (nt)",
         y="strong 5\u2032ss density (per site)")+th+theme(legend.position="top")
} else pMeta5<-ggplot()+theme_void()

fig<-(pA|pB|pC)/(pD|pE)/(pMeta5)+plot_layout(heights=c(1,1.2,0.8))+
  plot_annotation(
    title="Between-class discrimination: all three splicing signals separate true 3\u2032 ends from intragenic competitors",
    subtitle=paste0("Raw pooled canonical-vs-competitor view, no location control. Each signal is shown at the encoding that discriminates best. The acceptor 3\u2032ss is the definition\n",
      "signal and predicts the true end (A). U1 hybridization and the 5\u2032ss are suppression signals and predict the competitor, so their curves in D are oriented to show\n",
      "discrimination rather than direction (B, C, D). All three exceed chance. Acceptor and U1 are the strong ones. E and F show the acceptor by intervening 5\u2032ss\n",
      "interaction and the strong-5\u2032ss density around the site."),
    theme=theme(plot.title=element_text(face="bold",size=BASE+1,hjust=0),
                plot.subtitle=element_text(size=BASE-3,color="#555555")))
ggsave(file.path(INDIR,"figure3.pdf"),fig,width=15,height=11,device=cairo_pdf)
ggsave(file.path(INDIR,"figure3.png"),fig,width=15,height=11,dpi=300)
cat("[out] figure3.pdf / .png\n")

