#!/usr/bin/env Rscript
# what the reported cnn reads, in two panels
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(tidyr)
                               library(patchwork);library(ggseqlogo);library(scales)})
e<-function(...) cat(...,"\n",file=stderr())
INK<-"#1A1A1A"; BASE<-7; BLUE<-"#08519C"; ORNG<-"#E8A33D"; GREY<-"#8A8F98"; GTX<-"#5A6068"
FAINT<-"#C6CACF"

PM<-read_tsv("cnnmain_posmap.tsv",show_col_types=FALSE)
LG<-read_tsv("cnnmain_logos.tsv",show_col_types=FALSE)
GA<-read_tsv("gia_dose_matched_all.tsv",show_col_types=FALSE)
GE<-read_tsv("gia_dose_evolved.tsv",show_col_types=FALSE)
MPW<-read_tsv("me3_pwm.tsv",show_col_types=FALSE)
NPW<-read_tsv("me3_pwm_noc.tsv",show_col_types=FALSE)
UPW<-read_tsv("u1_pwm.tsv",show_col_types=FALSE)
MAPW<-read_tsv("canon_acc_pwm.tsv",show_col_types=FALSE)
MAMETA<-read_tsv("canon_acc_pwm_meta.tsv",show_col_types=FALSE)
NT<-read_tsv("cnnmain_notes.tsv",col_types=cols(.default=col_character()))
nv<-function(k) NT$value[NT$key==k][1]
PPT<-nv("ppt_filter"); NF<-as.integer(nv("n_filters"))
NTE<-as.integer(nv("n_test")); NGE<-as.integer(nv("n_genes")); RF<-as.integer(nv("A:receptive_field"))

th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-1),axis.line=element_line(color=INK,linewidth=.3),
  axis.title=element_text(size=BASE),axis.ticks=element_line(color=INK,linewidth=.3),
  plot.title=element_text(size=BASE+1,face="bold",hjust=0,margin=margin(b=4)),
  legend.position="none")

ORD<-PM%>%distinct(filter,row_order)%>%arrange(row_order)%>%pull(filter)
stopifnot(length(ORD)==NF, !any(duplicated(ORD)))
CONS<-PM%>%distinct(filter,consensus)%>%tibble::deframe()
mk<-function(f){
  m<-LG%>%filter(filter==f)%>%select(position,base,prob)%>%
     pivot_wider(names_from=position,values_from=prob)%>%as.data.frame()
  rn<-m$base; m<-as.matrix(m[,-1]); rownames(m)<-rn; m[c("A","C","G","T"),,drop=FALSE]}
mats<-lapply(ORD,mk)
derived<-sapply(mats,function(m) paste(rownames(m)[apply(m,2,which.max)],collapse=""))
stopifnot(identical(unname(derived),unname(CONS[ORD])))
LABF<-PM%>%distinct(filter,row_order,ic_per_col,n_sites)%>%arrange(row_order)%>%
  mutate(txt=sprintf("%s   %.2f b/col   %s sites",sub("filter","filter ",filter),
                     ic_per_col,format(n_sites,big.mark=",")))
names(mats)<-LABF$txt
pLog<-ggseqlogo(mats,method="bits",ncol=1)+
  theme_classic(base_size=BASE)+
  theme(text=element_text(color=INK),legend.position="none",strip.background=element_blank(),
        strip.text=element_text(size=BASE-2.4,color=GTX,hjust=0,margin=margin(b=0,t=1.5)),
        axis.text=element_blank(),axis.ticks=element_blank(),axis.line=element_blank(),
        axis.title=element_blank(),plot.margin=margin(l=2,r=1,t=18,b=2))

PM2<-PM%>%mutate(filter=factor(filter,levels=rev(ORD)))
stopifnot(identical(rev(levels(PM2$filter)),ORD))
LIM<-max(abs(PM2$contrib))
pHM<-ggplot(PM2,aes(centre,filter,fill=contrib))+geom_tile()+
  scale_fill_gradient2(low=ORNG,mid="white",high=BLUE,midpoint=0,limits=c(-LIM,LIM),
                       name="contribution")+
  scale_x_continuous(breaks=seq(-2000,500,500),expand=c(0,0))+
  scale_y_discrete(expand=c(0,0))+
  coord_cartesian(xlim=c(-2065,500),clip="off")+
  labs(title="A   First-layer filters and their positional contribution to within-gene ranking",
       x="Position relative to the cleavage site (nt)",y=NULL)+
  th+theme(axis.text.y=element_blank(),axis.ticks.y=element_blank(),axis.line.y=element_blank(),
           legend.position="right",legend.key.width=unit(6,"pt"),
           legend.key.height=unit(20,"pt"),legend.title=element_text(size=BASE-2),
           legend.text=element_text(size=BASE-2),plot.margin=margin(l=1,r=2,t=2,b=2))
pA<-(pLog|pHM)+plot_layout(widths=c(1.05,2.0))

PB<-read_tsv("panelB_prob_ci.tsv",show_col_types=FALSE)%>%filter(background=="real_competitor")
PBASE<-PB$base_median_p[1]; PNBG<-PB$n_bg[1]
UL<-c(acc_canon="3' splice site, tract 83% pyrimidine, no G, score3 10.28",
      donor="5' splice site, strongest U1 duplex, score5 10.86")
UC<-c(acc_canon=ORNG,donor=BLUE)
PART<-c(acc_canon="3' splice site",donor="5' splice site")
PB<-PB%>%filter(name%in%names(UL))%>%
   mutate(part=factor(PART[name],levels=c("3' splice site","5' splice site")),
          lab=UL[name],col=UC[name])
NSHUF<-20; NDER<-9
CTL<-bind_rows(
  PB%>%transmute(part,n=n_copies,lo=shuffle_p-shuffle_sd/sqrt(NSHUF),
                 hi=shuffle_p+shuffle_sd/sqrt(NSHUF)),
  PB%>%transmute(part,n=n_copies,lo=derange_p-derange_sd/sqrt(NDER),
                 hi=derange_p+derange_sd/sqrt(NDER)))%>%
  group_by(part,n)%>%summarise(lo=min(lo),hi=max(hi),.groups="drop")
ENDS<-PB%>%group_by(name)%>%slice_max(n_copies,n=1)%>%ungroup()
YB<-range(c(PB$motif_lo,PB$motif_hi,CTL$lo,CTL$hi,PBASE)); YB<-YB+c(-1,1)*.09*diff(YB)
SEQL<-c(acc_canon="TTTAACTTTCTTTTTTACAGACG",donor="CAGGTGAGT")
SHORT<-c(acc_canon="83% pyrimidine, no G, score3 10.28",
         donor="strongest U1 duplex, score5 10.86")
TAGB<-PB%>%filter(n_copies==1)%>%
  left_join(ENDS%>%select(name,nend=n_copies,pend=motif_p),by="name")%>%
  mutate(y=motif_p,
         vj=c(acc_canon=1.25,donor=1.25)[name],
         txt=sprintf("%s\n%s\n%.3f [%.3f, %.3f] at 1 copy to %.3f at %d copies",
                     SEQL[name],SHORT[name],motif_p,motif_lo,motif_hi,pend,nend))
pB<-ggplot(PB,aes(n_copies,motif_p,group=name))+
  geom_ribbon(data=CTL,aes(n,ymin=lo,ymax=hi),fill=GREY,alpha=.30,inherit.aes=FALSE)+
  geom_hline(yintercept=PBASE,linetype=2,color=INK,linewidth=.35)+
  geom_text(data=PB%>%group_by(part)%>%summarise(x=max(n_copies),.groups="drop"),
            aes(x,PBASE,label=sprintf("untouched competitor, %.2f",PBASE)),
            hjust=1,vjust=-0.6,size=2.2,color=INK,inherit.aes=FALSE)+
  geom_errorbar(aes(ymin=motif_lo,ymax=motif_hi,color=I(col)),width=.10,linewidth=.45)+
  geom_line(aes(color=I(col)),linewidth=.62)+geom_point(aes(color=I(col)),size=1.15)+
  geom_text(data=TAGB,aes(1.06,y,label=txt,color=I(col),vjust=vj),hjust=0,size=1.95,
            lineheight=1.08,inherit.aes=FALSE)+
  facet_wrap(~part,nrow=1,scales="free_x")+
  scale_x_continuous(breaks=c(1,2,3,4,6,8))+
  scale_y_continuous(limits=YB,breaks=seq(.3,1,.1))+
  labs(title="B   Implanted into a competitor, does the model call it a real 3' end",
       x="Tandem copies implanted",
       y="Median predicted probability")+
  th+theme(strip.background=element_blank(),
           strip.text.x=element_text(size=BASE-1,color=GTX,hjust=0,margin=margin(b=2)),
           panel.spacing.x=unit(11,"pt"),legend.position="none")

lmat<-function(d){m<-t(as.matrix(d[,c("A","C","G","T")])); rownames(m)<-c("A","C","G","T"); m}
MAM<-lmat(MAPW%>%arrange(offset)); U9<-lmat(UPW%>%arrange(offset))
stopifnot(ncol(MAM)==23,ncol(U9)==9)
cons<-function(m) paste(rownames(m)[apply(m,2,which.max)],collapse="")
CM<-cons(MAM); CU<-cons(U9)
stopifnot(substr(CM,19,20)=="AG",substr(CU,4,5)=="GT")
LGN<-c(sprintf("3' splice site ensemble, n = %s",format(MAMETA$n_sequences[1],big.mark=",")),
       "5' splice site ensemble, n = 115")
LMATS<-setNames(list(MAM,U9),LGN)
stopifnot(identical(levels(PB$part),c("3' splice site","5' splice site")))
pBlogo<-ggseqlogo(LMATS,method="bits",ncol=1)+
  scale_y_continuous(limits=c(0,2.05),breaks=c(0,1,2),expand=c(0,0))+
  theme_classic(base_size=BASE)+
  theme(text=element_text(color=INK),legend.position="none",strip.background=element_blank(),
        strip.text=element_text(size=BASE-2.8,color=GTX,hjust=0,lineheight=1.05,
                                margin=margin(b=1,t=1)),
        axis.text.x=element_blank(),axis.ticks.x=element_blank(),axis.line.x=element_blank(),
        axis.text.y=element_text(size=BASE-2.4,color=GTX),axis.title.x=element_blank(),
        axis.title.y=element_text(size=BASE-2,color=GTX),
        plot.margin=margin(l=2,r=6,t=26,b=40))+
  labs(y="bits")
pB<-(pBlogo|pB)+plot_layout(widths=c(1,3.5))

gp<-function(nm,k,col) PB[[col]][PB$name==nm&PB$n_copies==k]
EVV<-GE%>%filter(arm=="motif",site=="evolved_23",n_copies==1,background=="real_canonical")%>%
     distinct(best_delta)%>%pull(best_delta)
NA_<-max(PB$n_copies[PB$name=="acc_canon"]); ND_<-max(PB$n_copies[PB$name=="donor"])
shufmax<-max(abs(PB$shuffle_p-PBASE)); dermax<-max(abs(PB$derange_p-PBASE))
dnr<-PB%>%filter(name=="donor")
donshuf<-max(abs(dnr$shuffle_p-PBASE)); donder<-max(abs(dnr$derange_p-PBASE))
tsh<-read_tsv("panelB_prob_ci.tsv",show_col_types=FALSE)%>%
     filter(background=="real_competitor",name=="acc_noC")
tshmax<-max(tsh$shuffle_p-tsh$base_median_p)

# caption
CAP<-NULL

FIG<-pA/pB+plot_layout(heights=c(4.3,1.35))+
  plot_annotation(theme=theme(plot.margin=margin(4,6,4,6)))
ggsave("figure_cnn_interpretability.pdf",FIG,width=9.8,height=13.6,device=grDevices::cairo_pdf,
       limitsize=FALSE)
ggsave("figure_cnn_interpretability.png",FIG,width=9.8,height=13.6,dpi=220,limitsize=FALSE)
stopifnot(is.null(pHM$labels$subtitle),is.null(pB$labels$subtitle))
e("[A] logo/tile order ASSERTED: consensus re-derived from every drawn matrix matches the table")
e("[A] rows ordered by summed |contribution|; top:",paste(sub("filter","",ORD[1:5]),collapse=","),
  "| PPT filter",PPT,"at row",PM$row_order[PM$filter==PPT][1])
e("[B] competitor background, median predicted probability, baseline",sprintf("%.4f",PBASE))
for (st in names(UL))
  e("   ",sprintf("%-46s %s", UL[st],
    paste(sprintf("n%d %.3f (shuf %.3f)",PB$n_copies[PB$name==st],PB$motif_p[PB$name==st],
                  PB$shuffle_p[PB$name==st]),collapse="  ")))
e("[B] shuffle controls stay within",sprintf("%.4f",shufmax),
  "of baseline; the T-rich acceptor's reaches",sprintf("%+.4f",tshmax),"for contrast")
e("[B] logo consensus: acceptor",CM," donor",CU)
e("[out] figure_cnn_interpretability.pdf figure_cnn_interpretability.png")
