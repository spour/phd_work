#!/usr/bin/env Rscript
# figure s3 (calibrated) -- three positional heatmaps
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork);library(scales)})
INK<-"#1A1A1A"; BASE<-12; CAN<-"#0072B2"; GRN<-"#1D9E75"; COMP<-"#E69F00"; NUL<-"#8A8F98"
ME5DOT<-"#6A3D9A"; U1HI<-"#1F78B4"
D<-read_tsv("s3_raster_binned.tsv",show_col_types=FALSE)
D$class<-factor(D$class,levels=c("real","competitor"),labels=c("real 3′ ends","competitors"))
XB<-c(-2000,-1500,-1000,-500,0,500); XL<-c("-2 kb","-1.5 kb","-1 kb","-500","pA","+500")

RO<-read_tsv("s3_row_order.tsv",show_col_types=FALSE)
INT<-RO%>%transmute(row_index,class=factor(class,levels=c("real","competitor"),
        labels=c("real 3′ ends","competitors")),xmin=-peak_d_anchor_to_pa,xmax=0)

th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  plot.title=element_text(size=BASE,face="bold"),plot.subtitle=element_text(size=BASE-4,color="#555555"),
  axis.text=element_text(color=INK,size=BASE-4),axis.text.y=element_blank(),axis.ticks.y=element_blank(),
  axis.line=element_line(linewidth=.35),strip.background=element_rect(fill="grey92",color=NA),
  strip.placement="outside",strip.text.y.left=element_text(size=BASE-4,face="bold",color=INK,angle=90),
  legend.key.height=unit(.7,"lines"),legend.key.width=unit(.5,"lines"),panel.spacing=unit(.15,"lines"))

base_panel<-function(feat,fill_hi){
  d<-D%>%filter(feature==feat); up<-as.numeric(quantile(d$score_max,0.99,na.rm=TRUE))
  ggplot(d,aes(bin_start,row_index))+
    geom_raster(aes(fill=score_max))+
    geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.4)+
    scale_fill_gradient(low="white",high=fill_hi,limits=c(0,up),oob=squish,na.value="grey93",name="score")+
    scale_x_continuous(breaks=XB,labels=XL,expand=expansion(mult=c(0,.01)))+
    scale_y_reverse(expand=c(0,0))+
    facet_grid(rows=vars(class),scales="free_y",space="free",switch="y")+
    labs(x="position relative to pA (nt)",y="gene (paired: same row = same gene)")+th
}

# me3 (acceptor) : dots >= 8.19
p3<-base_panel("me3",CAN)+
  geom_point(data=D%>%filter(feature=="me3",score_max>=8.19),shape=16,size=.32,color=INK)+
  labs(title="MaxEnt 3′ss (acceptor) — 23-mer [-20..+3], AG-anchored",
       subtitle="hit dots: MaxEnt 3′ss ≥ 8.19  (50% sensitivity, 475x over intronic)")

# me5 (donor) : dots >= 8.46, with per-row (anchor,pa) interval shading
p5<-base_panel("me5",GRN)+
  geom_rect(data=INT,aes(xmin=xmin,xmax=xmax,ymin=row_index-0.5,ymax=row_index+0.5),
            inherit.aes=FALSE,fill=NUL,alpha=.16)+
  geom_point(data=D%>%filter(feature=="me5",score_max>=8.46),shape=16,size=.32,color=ME5DOT)+
  labs(title="MaxEnt 5′ss (donor) — 9-mer [-3..+6], GT-anchored",
       subtitle="hit dots: MaxEnt 5′ss ≥ 8.46 (50% sensitivity, 468x; top-1-optimal donor-absence). Shaded = (anchor 3′ss, pA): donors inside define an internal exon")

# u1 : dots >= 5.25 (primary) and >= 8.0 (contrast)
pu<-base_panel("u1",COMP)+
  geom_point(data=D%>%filter(feature=="u1",score_max>=5.25),shape=16,size=.22,color=INK,alpha=.85)+
  geom_point(data=D%>%filter(feature=="u1",score_max>=8.0),shape=16,size=.5,color=U1HI)+
  labs(title="U1 hybridization (-ΔG) — same 9-mer as the donor, thermodynamic scoring",
       subtitle="hit dots: U1 ≥ 5.25 (75% donor recall, 27x over intronic; dark) and U1 ≥ 8.0 (blue, for contrast)")

cap<-paste0("The three hit cutoffs differ because each is separately calibrated against annotated GENCODE v49 sites (50% sensitivity for MaxEnt 3′ss/5′ss at 8.19/8.46; 75% donor\n",
  "recall for U1 at 5.25). MaxEnt hits are several-hundred-fold enriched over intronic background (475x / 468x), so real acceptors and donors show up as sparse, sharp dots; U1 at 5.25\n",
  "is only 27x enriched, so its dots saturate the field. That density difference is the point: MaxEnt marks specific sites, U1 at a permissive cutoff marks composition. me5 and U1 read the\n",
  "SAME 9-mer in the same register and differ only in weighting; me3 reads a different motif at the other end of the intron. Rows are gene-paired (row i real = row i competitor); genes\n",
  "ordered by competitor distance-to-anchor. Bins are 10 nt (max score in bin); per-panel fill scale; dashed line = pA.")
fig<-(p3/p5/pu)+plot_layout(guides="collect")+
  plot_annotation(title="Figure S3 (calibrated).  Positional MaxEnt 3′ss / 5′ss and U1 scans, per-panel calibrated hit cutoffs",
    caption=cap,theme=theme(plot.title=element_text(face="bold",size=BASE+2,color=INK),
                            plot.caption=element_text(size=BASE-5,color="#666666",hjust=0)))
ggsave("figure_s3_calibrated.pdf",fig,width=13,height=15.5,device=cairo_pdf)
ggsave("figure_s3_calibrated.png",fig,width=13,height=15.5,dpi=200)
for(x in list(list(p3,"me3"),list(p5,"me5"),list(pu,"u1"))){
  ggsave(paste0("figure_s3_calibrated_",x[[2]],".pdf"),x[[1]],width=12,height=6,device=cairo_pdf)
  ggsave(paste0("figure_s3_calibrated_",x[[2]],".png"),x[[1]],width=12,height=6,dpi=200)
}
cat("[out] figure_s3_calibrated.pdf/.png + figure_s3_calibrated_me3/me5/u1 standalone\n")
