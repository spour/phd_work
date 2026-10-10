# figure 3.13, in silico tests of mechanisms coupling splicing and 3' end formation
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr)})
setwd("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3")
INK<-"#1A1A1A"; GTX<-"#5A6068"; GREY<-"#8A8F98"; ACC<-"#E8A33D"; DON<-"#08519C"
TH<-theme_classic(base_size=8.5)+theme(plot.title=element_text(face="bold",size=10.5,colour=INK),axis.text=element_text(colour=GTX),
      legend.key.size=unit(3,"mm"),legend.text=element_text(size=7.5),legend.title=element_blank())
P<-read.delim("data/implant_position.tsv")%>%mutate(lab=paste0(ifelse(construct=="acceptor","3′ splice site","5′ splice site")," ×",copies))
pA<-ggplot(P,aes(offset,diff,colour=lab))+geom_hline(yintercept=0,colour=GREY,linewidth=0.3)+
  geom_vline(xintercept=-230,linetype="22",colour=GREY,linewidth=0.3)+
  geom_ribbon(aes(ymin=diff_lo,ymax=diff_hi,fill=lab),alpha=0.25,colour=NA)+geom_line(linewidth=0.6)+geom_point(size=0.9)+
  scale_colour_manual(values=c("3′ splice site ×1"="#F2C58A","3′ splice site ×4"=ACC,"5′ splice site ×1"="#8FB0D3","5′ splice site ×8"=DON))+
  scale_fill_manual(values=c("3′ splice site ×1"="#F2C58A","3′ splice site ×4"=ACC,"5′ splice site ×1"="#8FB0D3","5′ splice site ×8"=DON),guide="none")+
  labs(x="offset of the implanted sequence (nt)",y="change in competitor logit\n(construct minus shuffled control)",title="A   Implantation by position")+TH+theme(legend.position="bottom")+guides(colour=guide_legend(nrow=2))
K<-read.delim("data/figm_knockout.tsv")%>%filter(bin!="all")%>%
  mutate(bin=factor(sub("site at ","",sub(" nt","",bin)),levels=c("-2000 to -1000","-1000 to -500","-500 to -250","-250 to -60")),
         size=factor(size,levels=c("9 nt","20 nt","23 nt","50 nt")))
kplot<-function(e,ttl,col){ ggplot(K%>%filter(experiment==e),aes(bin,diff,fill=size))+geom_hline(yintercept=0,colour=GREY,linewidth=0.3)+
  geom_col(position=position_dodge(0.75),width=0.7,colour=INK,linewidth=0.2)+
  geom_errorbar(aes(ymin=lo,ymax=hi),position=position_dodge(0.75),width=0.15,linewidth=0.3)+
  scale_fill_manual(values=col)+labs(x="position of the annotated site (nt)",y="change in logit\n(knockout minus control)",title=ttl)+TH+
  theme(legend.position="bottom",axis.text.x=element_text(size=7)) }
pB<-kplot("terminal-exon 3' splice site","B   Terminal-exon 3′ splice site removed",c("23 nt"="#F2C58A","50 nt"=ACC))
pC<-kplot("upstream 5' splice site","C   Upstream 5′ splice site removed",c("9 nt"="#8FB0D3","20 nt"=DON))
Ar<-read.delim("data/figm_architecture.tsv")
pD<-ggplot(Ar,aes(factor(donor_offset),interaction,fill=arrangement))+geom_hline(yintercept=0,colour=GREY,linewidth=0.3)+
  geom_col(width=0.65,colour=INK,linewidth=0.2)+
  scale_fill_manual(values=c("internal exon"="#C9D6E6","intron, then terminal exon"="#6C7A89"))+
  labs(x="offset of the 5′ splice site; 3′ splice site at -300 (nt)",y="interaction (both - each alone)",title="D   Exon architecture")+
  TH+theme(legend.position="bottom")
S<-read.delim("data/figm_scan.tsv")%>%mutate(element=recode(element,"TTGTAC (UGUA)"="TTGTAC (UGUA)","AATAAA (PAS)"="AATAAA"))
pE<-ggplot(S,aes(offset,diff,colour=element))+geom_hline(yintercept=0,colour=GREY,linewidth=0.3)+
  geom_vline(xintercept=0,linetype="22",colour=INK,linewidth=0.3)+geom_line(linewidth=0.6)+geom_point(size=0.8)+
  scale_colour_manual(values=c("TTGTAC (UGUA)"="#2E7D32","AATAAA"="#8E44AD"))+
  labs(x="offset of the implanted 6-mer (nt)",y="change in competitor logit\n(6-mer minus shuffled control)",title="E   Core polyadenylation elements")+
  TH+theme(legend.position="bottom")
fig<-(pA|pB|pC)
ggsave("out/fig3_13_mechanism.pdf",fig,width=10.4,height=3.4,device=cairo_pdf)
ggsave("out/fig3_13_mechanism.png",fig,width=10.4,height=3.4,dpi=300,bg="white")
cat("[out] out/fig3_13_mechanism.{pdf,png}\n")
