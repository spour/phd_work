#!/usr/bin/env Rscript
# supplementary: the intronic-decoy contrast
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(patchwork)})
INK<-"#1A1A1A"; CAN<-"#0072B2"; GRY<-"#8A8F98"; PLM<-"#CC79A7"; BASE<-7
th<-theme_classic(base_size=BASE,base_family="sans")+theme(
  plot.title=element_text(face="bold",size=BASE+.5,colour=INK,hjust=0,margin=margin(b=2)),
  plot.title.position="plot",
  plot.subtitle=element_text(size=BASE-.5,colour=INK,hjust=0,margin=margin(b=4)),
  axis.text=element_text(colour=INK,size=BASE-.5),axis.title=element_text(colour=INK,size=BASE),
  axis.line=element_line(linewidth=.25),axis.ticks=element_line(linewidth=.25),
  strip.background=element_blank(),strip.text=element_text(colour=INK,size=BASE-.5),
  legend.position="none",plot.margin=margin(3,5,3,4))
BLK<-c("acceptor + gc_contrast (2)"=CAN,"GC-binned (25)"=PLM)
J<-read.delim("gc_decoy.tsv")
J$block<-factor(J$block,levels=rev(names(BLK)))
J$contrast<-factor(J$contrast,levels=c("canonical - competitor","competitor - intronic decoy"),
                   labels=c("canonical - competitor: + = real ends score higher",
                            "competitor - decoy: + = CPA-like, - = below background"))
p<-ggplot(J,aes(gap,block,colour=block))+
  geom_vline(xintercept=0,colour=GRY,linewidth=.25)+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),orientation="y",width=0,linewidth=.5)+
  geom_point(size=1.5)+scale_colour_manual(values=BLK)+
  geom_text(aes(label=sprintf("%+.3f",gap)),size=2.1,vjust=-1.3)+
  facet_wrap(~contrast,ncol=1)+
  scale_x_continuous(limits=c(-.14,.46))+
  scale_y_discrete(expand=expansion(add=c(.7,1.0)))+
  labs(title="Supplementary: compartment, not CPA-likeness",
       subtitle="A measure of CPA-likeness orders canonical > competitor > intronic background, because the\ncompetitors are the positions a CPA model already picked out of that background. GC-binned does\nthat. The two-feature model does not: it places competitors slightly below unselected intronic\nsequence, because a competitor is a locally AT-rich site inside an intron that is AT-rich\nthroughout, so its distal-minus-proximal contrast is lower than a random intronic position's.",
       x="difference in the held-out discriminant, 95% CI over genes",y=NULL)+th+
  theme(axis.text.y=element_text(size=BASE-1))
ggsave("figure_supp_decoy.pdf",p,width=140,height=78,units="mm",device=grDevices::cairo_pdf)
ggsave("figure_supp_decoy.png",p,width=140,height=78,units="mm",dpi=400)
cat("wrote figure_supp_decoy.pdf / .png\n")
