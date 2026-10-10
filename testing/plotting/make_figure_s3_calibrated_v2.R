#!/usr/bin/env Rscript
# figure s3 calibrated v2
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork);library(scales)})
INK<-"#1A1A1A"; BASE<-12; GRN<-"#1D9E75"; COMP<-"#E69F00"; NUL<-"#8A8F98"
ME3DOT<-"#1A1A1A"; ME5DOT<-"#6A3D9A"; ANCH<-"#E31A1C"; PACOL<-"#1F78B4"; U1HI<-"#1F78B4"
CLEV<-c("real","competitor"); CLAB<-c("real 3′ ends","competitors")

D<-read_tsv("s3_raster_binned.tsv",show_col_types=FALSE)
RO<-read_tsv("s3_row_order.tsv",show_col_types=FALSE)
NG<-sum(RO$class=="real"); set.seed(42); perm<-sample(NG)
RO$rank<-ifelse(RO$class=="real",RO$row_index,RO$row_index-NG); RO$ynew<-perm[RO$rank]
MAP<-RO%>%transmute(row_index,ynew,d=peak_d_anchor_to_pa)
D<-D%>%left_join(MAP,by="row_index")
D$class<-factor(D$class,levels=CLEV,labels=CLAB)
TICK<-RO%>%transmute(ynew,d=peak_d_anchor_to_pa,class=factor(class,levels=CLEV,labels=CLAB))

XB<-c(-2000,-1500,-1000,-500,0,500); XL<-c("-2 kb","-1.5 kb","-1 kb","-500","pA","+500")
th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  plot.title=element_text(size=BASE,face="bold"),plot.subtitle=element_text(size=BASE-4,color="#555555"),
  axis.text=element_text(color=INK,size=BASE-4),axis.text.y=element_blank(),axis.ticks.y=element_blank(),
  axis.line=element_line(linewidth=.35),strip.background=element_rect(fill="grey92",color=NA),
  strip.placement="outside",strip.text.y.left=element_text(size=BASE-4,face="bold",color=INK,angle=90),
  legend.key.height=unit(.7,"lines"),legend.key.width=unit(.5,"lines"),panel.spacing=unit(.15,"lines"))
facet_cls<-facet_grid(rows=vars(class),scales="free_y",space="free",switch="y")

# pa-aligned maxent panels: hits only, white bg, anchor tick
pa_points<-function(feat,cut,col,ttl,sub){
  ggplot()+
    geom_point(data=TICK,aes(-d,ynew),shape=124,size=1.1,color=ANCH)+
    geom_point(data=D%>%filter(feature==feat,score_max>=cut),aes(bin_start,ynew),shape=16,size=.55,color=col)+
    geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.4)+
    scale_x_continuous(breaks=XB,labels=XL,expand=expansion(mult=c(.01,.01)),limits=c(-2050,520))+
    scale_y_reverse(expand=c(0,0))+facet_cls+
    labs(title=ttl,subtitle=sub,x="position relative to pA (nt)",y="gene (paired: same row = same gene)")+th
}
p3<-pa_points("me3",8.19,ME3DOT,"MaxEnt 3′ss (acceptor), pA-aligned — hits only",
   "MaxEnt 3′ss ≥ 8.19 (50% sensitivity, 475x over intronic). Red tick = anchor 3′ss per row.")
p5<-pa_points("me5",8.46,ME5DOT,"MaxEnt 5′ss (donor), pA-aligned — hits only",
   "MaxEnt 5′ss ≥ 8.46 (50% sensitivity, 468x; top-1-optimal). Red tick = anchor 3′ss; a donor between it and the pA defines an internal exon.")

# u1 panel: keep the fill (density is the point) + two dot layers
du<-D%>%filter(feature=="u1"); upu<-as.numeric(quantile(du$score_max,0.99,na.rm=TRUE))
pu<-ggplot(du,aes(bin_start,ynew))+
  geom_raster(aes(fill=score_max))+
  geom_point(data=du%>%filter(score_max>=5.25),shape=16,size=.22,color=INK,alpha=.85)+
  geom_point(data=du%>%filter(score_max>=8.0),shape=16,size=.5,color=U1HI)+
  geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.4)+
  scale_fill_gradient(low="white",high=COMP,limits=c(0,upu),oob=squish,na.value="grey93",name="U1 score")+
  scale_x_continuous(breaks=XB,labels=XL,expand=expansion(mult=c(0,.01)))+
  scale_y_reverse(expand=c(0,0))+facet_cls+
  labs(title="U1 hybridization (-ΔG), pA-aligned — fill kept",
       subtitle="U1 ≥ 5.25 (75% donor recall, 27x; dark dots) and ≥ 8.0 (blue). Density is the point, so the raster stays.",
       x="position relative to pA (nt)",y="gene (paired)")+th

# anchor-aligned maxent panels: x = position relative to anchor 3'ss
XA<-c(0,500,1000,1500,2000); XAL<-c("anchor","+500","+1 kb","+1.5 kb","+2 kb")
anc_points<-function(feat,cut,col,ttl,sub){
  h<-D%>%filter(feature==feat,score_max>=cut)%>%mutate(xa=bin_start+d)
  ggplot()+
    geom_point(data=TICK,aes(d,ynew),shape=124,size=1.1,color=PACOL)+
    geom_point(data=h,aes(xa,ynew),shape=16,size=.55,color=col)+
    geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.4)+
    scale_x_continuous(breaks=XA,labels=XAL,expand=expansion(mult=c(.01,.01)),limits=c(-700,2450))+
    scale_y_reverse(expand=c(0,0))+facet_cls+
    labs(title=ttl,subtitle=sub,x="position relative to anchor 3′ss (nt)",y="gene (paired)")+th
}
p3a<-anc_points("me3",8.19,ME3DOT,"MaxEnt 3′ss (acceptor), anchor-aligned — hits only",
   "x = 0 at the anchor 3′ss (dashed). Blue tick = pA per row. MaxEnt 3′ss ≥ 8.19.")
p5a<-anc_points("me5",8.46,ME5DOT,"MaxEnt 5′ss (donor), anchor-aligned — hits only",
   "x = 0 at the anchor 3′ss (dashed). Blue tick = pA per row; the (anchor, pA) span is the exon interval. MaxEnt 5′ss ≥ 8.46.")

cap<-paste0("Rows are in RANDOM order (fixed seed), pairing kept (row i real = row i competitor): any residual diagonal is not the d_anchor sort. The two MaxEnt panels plot HITS ONLY as\n",
  "points on white, since real acceptors/donors are sparse and several-hundred-fold enriched (475x / 468x) -- the raster was hiding them; U1 keeps its fill because its density (only 27x over\n",
  "intronic at 5.25) is the point. Cutoffs differ because each is separately calibrated against annotated GENCODE v49 sites. The anchor is a thin per-row tick, not a filled region. The anchor-\n",
  "aligned panels re-reference x to the anchor 3′ss (0), with the pA marked per row, since the mechanism is anchor-referenced. Same 200 gene-paired peaks, same scan; bins 10 nt (max in bin).")
fig<-(p3/p5/pu/p3a/p5a)+plot_layout(guides="collect")+
  plot_annotation(title="Figure S3 (calibrated v2).  MaxEnt hits as points (pA- and anchor-aligned) + U1 density, random row order",
    caption=cap,theme=theme(plot.title=element_text(face="bold",size=BASE+2,color=INK),
                            plot.caption=element_text(size=BASE-5,color="#666666",hjust=0)))
ggsave("figure_s3_calibrated_v2.pdf",fig,width=13,height=24,device=cairo_pdf)
ggsave("figure_s3_calibrated_v2.png",fig,width=13,height=24,dpi=170)
for(x in list(list(p3,"me3"),list(p5,"me5"),list(pu,"u1"),list(p3a,"me3_anchor"),list(p5a,"me5_anchor"))){
  ggsave(paste0("figure_s3_calibrated_v2_",x[[2]],".pdf"),x[[1]],width=12,height=5.4,device=cairo_pdf)
  ggsave(paste0("figure_s3_calibrated_v2_",x[[2]],".png"),x[[1]],width=12,height=5.4,dpi=190)
}
cat("[out] figure_s3_calibrated_v2.* + 5 standalones\n")
