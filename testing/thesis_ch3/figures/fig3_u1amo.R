# 3' end sequencing signal around unannotated competitors and random intronic positions
suppressPackageStartupMessages({library(ggplot2);library(dplyr)})
setwd("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3")
INK<-"#1A1A1A"; GTX<-"#5A6068"; COMP<-"#E69F00"; NUL<-"#8A8F98"
D<-read.delim("data/usage_metaprofile.tsv")%>%mutate(x=bin_start+5,
  class=factor(class,levels=c("real 3' ends","unannotated competitors","random intronic positions"),labels=c("Real 3' ends","Unannotated competitors","Random intronic positions")),
  condition=factor(condition,levels=c("control","U1 AMO"),labels=c("Control","U1 inhibited")),
  experiment=factor(experiment,levels=c("HeLa (Feng 2024)","HeLa (Feng 2025)","HEK293 (Feng 2025)","SW480 (Feng 2025)")))
p<-ggplot(D,aes(x,signal,colour=class,linetype=condition,linewidth=condition))+geom_vline(xintercept=0,colour=GTX,linetype="dashed",linewidth=0.3)+
  geom_line()+facet_grid(class~experiment,scales="free_y")+
  scale_colour_manual(values=c("Real 3' ends"="#0072B2","Unannotated competitors"=COMP,"Random intronic positions"=NUL),guide="none")+
  scale_linetype_manual(values=c("Control"="22","U1 inhibited"="solid"))+scale_linewidth_manual(values=c("Control"=0.5,"U1 inhibited"=0.8))+
  scale_x_continuous(breaks=c(-500,0,500))+
  labs(x="position relative to site (nt)",y="3' end reads (signal per site)")+theme_classic(base_size=10)+
  theme(strip.background=element_blank(),strip.text=element_text(colour=INK,size=9.5),axis.text=element_text(colour=GTX),legend.title=element_blank(),
        legend.position="bottom",legend.key.width=unit(9,"mm"),panel.spacing=unit(4,"mm"))
ggsave("out/fig3_u1amo.pdf",p,width=7.8,height=6.2,device=cairo_pdf); ggsave("out/fig3_u1amo.png",p,width=7.8,height=6.2,dpi=300,bg="white")
cat("[out] out/fig3_u1amo\n")
