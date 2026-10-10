# figure 3.10, the network reads the exonic identity of upstream sequence
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr)})
setwd("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3")
INK<-"#1A1A1A"; GTX<-"#5A6068"; GREY<-"#8A8F98"; REAL<-"#08519C"; COMP<-"#E8A33D"; EXON<-"#08519C"; INTRON<-"#A9B0BA"
TH<-theme_classic(base_size=9)+theme(plot.title=element_text(face="bold",size=11,colour=INK),axis.text=element_text(colour=GTX))
A<-read.delim("data/fig3_10_bins.tsv")%>%mutate(x=bin_start+50,class=factor(class,levels=c("real 3′ end","competitor")))
pA<-ggplot(A,aes(x,exonic,colour=class))+geom_line(linewidth=0.7)+geom_point(size=1.2)+
  geom_vline(xintercept=0,linetype="22",colour=INK,linewidth=0.3)+
  scale_colour_manual(values=c("real 3′ end"=REAL,competitor=COMP),name=NULL)+
  scale_y_continuous(limits=c(0,1))+
  labs(x="position relative to the candidate site (nt)",y="fraction annotated exon",title="A   Exonic sequence in the window")+
  TH+theme(legend.position=c(0.25,0.85),legend.key.size=unit(3,"mm"))
B<-read.delim("data/fig3_10_p1.tsv")%>%mutate(method=factor(method,levels=method),kind=ifelse(grepl("annotated",method),"annotation","model"))
pB<-ggplot(B,aes(method,p1,fill=kind))+geom_col(width=0.65,colour=INK,linewidth=0.25)+
  geom_errorbar(aes(ymin=lo,ymax=hi),width=0.15,linewidth=0.35)+
  scale_fill_manual(values=c(model="#C9D6E6",annotation="white"),guide="none")+
  scale_y_continuous(limits=c(0,1),expand=c(0,0))+scale_x_discrete(labels=function(x) gsub(", | (?=network|model)", "\n", x, perl=TRUE))+
  labs(x=NULL,y="precision at 1",title="B   Ranking by annotated exon content")+TH+theme(axis.text.x=element_text(size=7.5))
C<-read.delim("data/fig3_10_byfrac.tsv")%>%mutate(bin=factor(bin,levels=bin))
pC<-ggplot(C,aes(bin,p1))+geom_col(width=0.6,fill="#C9D6E6",colour=INK,linewidth=0.25)+
  geom_errorbar(aes(ymin=lo,ymax=hi),width=0.15,linewidth=0.35)+
  scale_y_continuous(limits=c(0,1),expand=c(0,0))+
  labs(x="exonic fraction of the real 3′ end's upstream 2 kb",y="network precision at 1",title="C   Network accuracy by upstream exon content")+TH
Dd<-read.delim("data/fig3_10_patch.tsv")%>%mutate(x=bin_start+50,kind=factor(kind,levels=c("exonic","intronic"),
     labels=c("transplanted bin exonic","transplanted bin intronic")))
pD<-ggplot(Dd,aes(x,gain,colour=kind,fill=kind))+
  geom_hline(yintercept=0,colour=GREY,linewidth=0.3)+
  geom_ribbon(aes(ymin=lo,ymax=hi),alpha=0.25,colour=NA)+geom_line(linewidth=0.7)+geom_point(size=1.1)+
  geom_vline(xintercept=0,linetype="22",colour=INK,linewidth=0.3)+
  scale_colour_manual(values=c("transplanted bin exonic"=EXON,"transplanted bin intronic"=GREY),name=NULL)+
  scale_fill_manual(values=c("transplanted bin exonic"=EXON,"transplanted bin intronic"=GREY),name=NULL)+
  labs(x="position of the transplanted 100 nt bin (nt)",y="increase in competitor logit",title="D   Transplanting 100 nt from the real 3′ end window")+
  TH+theme(legend.position=c(0.25,0.85),legend.key.size=unit(3,"mm"))
fig<-(pA|pB)/(pC|pD)+plot_layout(heights=c(1,1))
ggsave("out/fig3_10_exon_context.pdf",fig,width=8.6,height=6.8,device=cairo_pdf)
ggsave("out/fig3_10_exon_context.png",fig,width=8.6,height=6.8,dpi=300,bg="white")
cat("[out] out/fig3_10_exon_context.{pdf,png}\n")
