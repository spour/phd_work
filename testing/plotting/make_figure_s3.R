#!/usr/bin/env Rscript
# figure s3 -- the within-gene contrast the statistics actually measure
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork);library(scales)})
INK<-"#1A1A1A"; BASE<-12; CAN<-"#0072B2"; GRN<-"#1D9E75"; COMP<-"#E69F00"; NUL<-"#8A8F98"
th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  plot.title=element_text(size=BASE,face="bold"),plot.subtitle=element_text(size=BASE-4,color="#555555"),
  axis.text=element_text(color=INK,size=BASE-4),axis.line=element_line(linewidth=.35),
  strip.background=element_rect(fill="grey92",color=NA),strip.text=element_text(size=BASE-4,face="bold",color=INK))
FL<-c(me3="MaxEnt 3′ss (acceptor)",me5="MaxEnt 5′ss (donor)",u1="U1 density ≥5.25")

# panel 1: paired difference
P1<-read_tsv("c_s3_panel1.tsv",show_col_types=FALSE)
P1$feature<-factor(P1$feature,levels=c("me3","me5","u1"),labels=FL)
P1$sign<-ifelse(P1$diff>0,"real end higher","competitor higher")
fr<-P1%>%group_by(feature)%>%summarise(f=mean(diff>0),.groups="drop")
p1<-ggplot(P1,aes(rank,diff,color=sign))+
  geom_hline(yintercept=0,color=INK,linewidth=.3)+
  geom_segment(aes(xend=rank,yend=0),linewidth=.5)+geom_point(size=.9)+
  geom_text(data=fr,aes(x=3,y=Inf,label=sprintf("real > competitor: %.2f",f)),
            inherit.aes=FALSE,hjust=0,vjust=1.5,size=3,color=INK)+
  scale_color_manual(values=c("real end higher"=CAN,"competitor higher"=COMP),name=NULL)+
  facet_wrap(~feature,ncol=1,scales="free")+
  labs(title="1  Within-gene paired difference (real end − its own competitor)",
       subtitle="Each bar is one gene, sorted. This is what an acceptor win-rate of 0.59 looks like: most bars positive, many negative.",
       x="gene (sorted by difference)",y="real − competitor")+th+
  theme(legend.position="bottom",axis.text.x=element_blank(),axis.ticks.x=element_blank())

# panel 2: within-gene-pair z raster
P2<-read_tsv("c_s3_panel2.tsv",show_col_types=FALSE)
P2$feature<-factor(P2$feature,levels=c("me3","me5","u1"),labels=FL)
P2$class<-factor(P2$class,levels=c("real","competitor"),labels=c("real 3′ ends","competitors"))
zc<-as.numeric(quantile(abs(P2$z),0.98,na.rm=TRUE))
p2<-ggplot(P2,aes(bin_start,row_index,fill=z))+
  geom_raster()+geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.3)+
  scale_fill_gradient2(low=COMP,mid="white",high=CAN,midpoint=0,limits=c(-zc,zc),oob=squish,name="z (within gene-pair)")+
  scale_x_continuous(breaks=c(-2000,-1000,0),labels=c("-2 kb","-1 kb","pA"),expand=c(0,0))+
  scale_y_reverse(expand=c(0,0))+
  facet_grid(rows=vars(class),cols=vars(feature),scales="free_y",space="free_y",switch="y")+
  labs(title="2  Within-gene-pair z-scored (between-locus variance removed)",
       subtitle="Each gene's real+competitor scores standardised to their shared mean/SD. If a contrast is visible anywhere, it is here.",
       x="position relative to pA (nt)",y=NULL)+th+
  theme(axis.text.y=element_blank(),axis.ticks.y=element_blank(),strip.placement="outside",
        strip.text.y.left=element_text(angle=90),legend.position="bottom",
        legend.key.height=unit(.6,"lines"),panel.spacing=unit(.15,"lines"))

# panel 3a: acceptor maxent vs distance-to-pa
P3a<-read_tsv("c_s3_panel3a.tsv",show_col_types=FALSE)%>%filter(!is.na(acc),!is.na(d_anchor_to_pa))
P3a$class<-factor(P3a$class,levels=c("real","competitor"),labels=c("real 3′ ends","competitors"))
CLR<-c("real 3′ ends"=CAN,"competitors"=COMP)
sc<-ggplot(P3a,aes(d_anchor_to_pa,acc,color=class))+
  geom_point(size=1.3,alpha=.7)+
  scale_color_manual(values=CLR,name=NULL)+
  scale_x_log10()+
  labs(x="distance anchor 3′ss → pA (nt, log)",y="anchor acceptor MaxEnt")+th+
  theme(legend.position="none")
mx<-ggplot(P3a,aes(d_anchor_to_pa,fill=class,color=class))+geom_density(alpha=.25,linewidth=.4)+
  scale_x_log10()+scale_fill_manual(values=CLR)+scale_color_manual(values=CLR)+
  theme_void()+theme(legend.position="none")
my<-ggplot(P3a,aes(acc,fill=class,color=class))+geom_density(alpha=.25,linewidth=.4)+coord_flip()+
  scale_fill_manual(values=CLR)+scale_color_manual(values=CLR)+theme_void()+theme(legend.position="none")
p3a<-(mx+plot_spacer()+sc+my+plot_layout(ncol=2,widths=c(4,1),heights=c(1,4)))

# panel 3b: me3 re-aligned on anchor
P3b<-read_tsv("c_s3_panel3b.tsv",show_col_types=FALSE)
P3b$class<-factor(P3b$class,levels=c("real","competitor"),labels=c("real 3′ ends","competitors"))
up3<-as.numeric(quantile(P3b$me3_max,0.99,na.rm=TRUE))
p3b<-ggplot(P3b,aes(xbin,row_index,fill=me3_max))+
  geom_raster()+geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.4)+
  annotate("text",x=0,y=Inf,label="anchor 3′ss",angle=90,hjust=1.1,vjust=-0.3,size=2.6,color=INK)+
  scale_fill_gradient(low="white",high=CAN,limits=c(0,up3),oob=squish,na.value="grey93",name="MaxEnt 3′ss")+
  scale_x_continuous(breaks=seq(-500,400,250),expand=c(0,0))+scale_y_reverse(expand=c(0,0))+
  facet_grid(rows=vars(class),scales="free_y",space="free_y",switch="y")+
  labs(x="position relative to anchor 3′ss (nt)",y=NULL)+th+
  theme(axis.text.y=element_blank(),axis.ticks.y=element_blank(),strip.placement="outside",
        strip.text.y.left=element_text(angle=90),legend.position="bottom",
        legend.key.height=unit(.6,"lines"))
p3<-(p3a|p3b)+plot_layout(widths=c(1,1))+
  plot_annotation(title="3  Anchor-referenced: where Layer 1 lives",
    subtitle=paste0("(a) the anchor acceptor is one discrete site; within gene it is STRONGER at real ends (paired win-rate 0.61) but NOT closer to the pA ",
                    "(paired win-rate 0.47, median −50 nt, consistent with the established negative 0.506); scatter and marginals are unpaired/between-gene ",
                    "(median distance 895 vs 881 nt).  (b) me3 re-aligned on the anchor (x=0)."),
    theme=theme(plot.title=element_text(size=BASE,face="bold",color=INK),
                plot.subtitle=element_text(size=BASE-5,color="#555555")))

cap<-paste0("These are aggregate tendencies of SMALL effect size (acceptor pooled AUROC 0.606, within-gene win-rate 0.586) and are not expected to be visible per gene. ",
  "Panel 1 shows the paired differences directly; panel 2 removes between-locus variance by z-scoring within each gene-pair; panel 3 references the discrete anchor 3′ss where\n",
  "the acceptor signal actually lives. me5 and U1 score the same GT-anchored 9-mer and differ only in weighting, which is why their panel-1 fractions are identical (0.39 both); ",
  "me3 scores a different, AG-anchored, pyrimidine-rich 23-mer at the other end of the intron.\n",
  "Same 200 gene-paired peaks and same scan as the raw-score version. U1 analysis cutoff is 5.25; no hit dots are shown (they saturate the field).")
fig<-(p1/p2/wrap_elements(p3))+plot_layout(heights=c(1.1,1.0,1.0))+
  plot_annotation(title="Figure S3.  The within-gene contrast the Figure 3 statistics measure (200 gene-paired peaks)",
    caption=cap,theme=theme(plot.title=element_text(face="bold",size=BASE+2,color=INK),
                            plot.caption=element_text(size=BASE-5,color="#666666",hjust=0)))
ggsave("figure_s3.pdf",fig,width=13,height=19,device=cairo_pdf)
ggsave("figure_s3.png",fig,width=13,height=19,dpi=170)
ggsave("figure_s3_panel1.pdf",p1,width=9,height=8,device=cairo_pdf); ggsave("figure_s3_panel1.png",p1,width=9,height=8,dpi=200)
ggsave("figure_s3_panel2.pdf",p2,width=11,height=6,device=cairo_pdf); ggsave("figure_s3_panel2.png",p2,width=11,height=6,dpi=200)
ggsave("figure_s3_panel3.pdf",wrap_elements(p3),width=13,height=6,device=cairo_pdf); ggsave("figure_s3_panel3.png",wrap_elements(p3),width=13,height=6,dpi=200)
cat("[out] figure_s3 + panel1/2/3 standalone\n")
