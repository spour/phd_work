#!/usr/bin/env Rscript
# what distinguishes terminal from internal exons, in the terms your competitor heatmaps use
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork)})
args<-commandArgs(trailingOnly=TRUE)
INDIR<-if(length(args)>=1) args[1] else "."
PFX  <-if(length(args)>=2) args[2] else "exon_ref"
D<-read_tsv(file.path(INDIR,paste0(PFX,"_scores.tsv")),show_col_types=FALSE)
INK<-"#1A1A1A"; BASE<-13; COL<-c(internal="#0072B2", terminal="#E69F00")
th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),axis.text=element_text(color=INK,size=BASE-4),
  axis.line=element_line(color=INK,linewidth=.4),plot.title=element_text(size=BASE,face="bold",hjust=0),
  plot.subtitle=element_text(size=BASE-5,color="#555555"),legend.position="top",legend.title=element_blank())
D2<-D%>%filter(cls %in% c("internal","terminal")); D2$cls<-factor(D2$cls,levels=c("internal","terminal"))

# a: 3'ss strength -- both strong, terminal slightly stronger
a3<-D2%>%filter(is.finite(three)); mm3<-a3%>%group_by(cls)%>%summarise(m=median(three),.groups="drop")
pA<-ggplot(a3,aes(three,fill=cls,color=cls))+geom_density(alpha=.3,linewidth=.7)+
  geom_vline(data=mm3,aes(xintercept=m,color=cls),linewidth=.5,linetype=2,show.legend=FALSE)+
  scale_fill_manual(values=COL)+scale_color_manual(values=COL)+
  labs(title="A",subtitle="3\u2032ss (acceptor): terminal exons have acceptors as strong as internal",
       x="MaxEntScan 3\u2032ss score",y="density")+th

# b: distance to nearest strong downstream 5'ss -- the distinguishing context
dd<-D2%>%filter(is.finite(dn_donor_dist), dn_donor_dist<=600)
frac<-D2%>%group_by(cls)%>%summarise(has=mean(is.finite(dn_donor_dist)),.groups="drop")
med<-D2%>%filter(is.finite(dn_donor_dist))%>%group_by(cls)%>%summarise(m=median(dn_donor_dist),.groups="drop")
pB<-ggplot(dd,aes(dn_donor_dist,fill=cls,color=cls))+
  geom_histogram(aes(y=after_stat(density)*100*40),binwidth=40,position="identity",alpha=.35,linewidth=.3)+
  geom_vline(data=med,aes(xintercept=m,color=cls),linewidth=.6,linetype=2,show.legend=FALSE)+
  scale_fill_manual(values=COL)+scale_color_manual(values=COL)+
  labs(title="B",subtitle=sprintf("Nearest downstream 5\u2032ss: internal median %.0fnt, terminal %.0fnt",
       med$m[med$cls=="internal"],med$m[med$cls=="terminal"]),
       x="distance from acceptor to nearest strong downstream 5\u2032ss (nt)",
       y="% of exons (per 40 nt)")+th

fig<-(pA|pB)+plot_layout(guides="collect")+
  plot_annotation(title="Terminal vs internal exons: equally strong acceptors, but only internal exons have a nearby downstream 5\u2032ss",
    theme=theme(plot.title=element_text(face="bold",size=BASE,hjust=0)))&theme(legend.position="top")
ggsave(file.path(INDIR,"figure3_exon_reference.pdf"),fig,width=12,height=4.5,device=cairo_pdf)
ggsave(file.path(INDIR,"figure3_exon_reference.png"),fig,width=12,height=4.5,dpi=300)
cat("[out] figure3_exon_reference.pdf / .png\n")