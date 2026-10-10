# frequency of real 3' ends by maxentscan score of the upstream splice sites
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr)})
setwd("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3")
INK<-"#1A1A1A"; GTX<-"#5A6068"; CAN<-"#0072B2"; COMP<-"#E69F00"; ACC<-"#0072B2"; DON<-"#D55E00"
th<-theme_classic(base_size=10)+theme(plot.title=element_text(face="bold",size=11,colour=INK),axis.text=element_text(colour=GTX),legend.key.size=unit(4,"mm"))
C<-read.delim("data/hypothesis_curve.tsv",check.names=FALSE)%>%filter(candidates>=200)%>%
  mutate(score=factor(score,levels=c("3' splice site","5' splice site"),labels=c("Upstream 3' splice site","Upstream 5' splice site")))
pA<-ggplot(C,aes(mid,observed_over_expected,colour=score,fill=score))+geom_hline(yintercept=1,colour=GTX,linetype="dashed",linewidth=0.3)+
  geom_ribbon(aes(ymin=lo,ymax=hi),alpha=0.18,colour=NA)+geom_line(linewidth=1)+geom_point(size=2.4)+
  scale_colour_manual(values=c(ACC,DON))+scale_fill_manual(values=c(ACC,DON))+scale_y_continuous(trans="log2",breaks=c(0.5,0.75,1,1.5,2),limits=c(0.28,2.1))+
  scale_x_continuous(breaks=seq(4,14,2))+
  labs(x="MaxEntScan score",y="real 3' ends, observed / expected",title="A   Each splice site")+th+theme(legend.title=element_blank(),legend.position="bottom")
G<-read.delim("data/hypothesis_grid.tsv",check.names=FALSE)
ax<-unique(G[order(G$acc_mid),"acc_bin"]); dx<-unique(G[order(G$don_mid),"don_bin"])
G<-G%>%mutate(a=factor(acc_bin,levels=ax),d=factor(don_bin,levels=dx),l=log2(observed_over_expected))
L<-max(abs(G$l))
pB<-ggplot(G,aes(a,d,fill=l))+geom_tile(colour="white",linewidth=0.6)+
  scale_fill_gradient2(low=COMP,mid="#F7F7F7",high=CAN,midpoint=0,limits=c(-L,L),breaks=log2(c(0.5,1,2)),labels=c("0.5","1","2"),name="observed /\nexpected")+
  labs(x="3' splice site score",y="5' splice site score",title="B   Both splice sites")+th+
  theme(axis.line=element_blank(),axis.ticks=element_blank(),axis.text=element_text(colour=INK,size=8.5),legend.position="right",legend.key.height=unit(7,"mm"),legend.key.width=unit(3.5,"mm"))+coord_fixed()
fig<-pA+pB+plot_layout(widths=c(1,1.15))
ggsave("out/fig3_hypothesis_dose.pdf",fig,width=7.8,height=4.0,device=cairo_pdf); ggsave("out/fig3_hypothesis_dose.png",fig,width=7.8,height=4.0,dpi=300,bg="white")
cat("[out] out/fig3_hypothesis_dose\n")
