# figure 3.9, what the network's ranking depends on
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr)})
setwd("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3")
source("R/_harvest.R")
INK<-"#1A1A1A"; GTX<-"#5A6068"; GREY<-"#8A8F98"; CAN<-"#0B55A0"; LIGHT<-"#C9D6E6"
TH<-theme_classic(base_size=10)+theme(plot.title=element_text(face="bold",size=11,colour=INK),
  axis.text=element_text(colour=GTX),plot.subtitle=element_blank(),plot.caption=element_blank())
CH<-0.292
ACC<-read.delim("data/acceptor_alone_test.tsv"); acc_wr<-ACC$value[ACC$metric=="win_rate"]

D<-read.delim("data/context_sweep.tsv")
S<-D%>%group_by(context,start,end)%>%summarise(m=mean(test_p_at_1),lo=min(test_p_at_1),hi=max(test_p_at_1),.groups="drop")%>%mutate(up=-start)
U<-S%>%filter(end==500); B5<-S%>%filter(context=="span500")
pA<-ggplot(U,aes(up,m))+
  geom_hline(yintercept=CH,linetype="22",colour=GREY)+
  annotate("text",x=2000,y=CH+0.012,label="chance",hjust=1,colour=GREY,size=2.8)+
  geom_line(colour=CAN,linewidth=0.7)+geom_errorbar(aes(ymin=lo,ymax=hi),width=45,colour=CAN,linewidth=0.35)+
  geom_point(colour=CAN,size=1.9)+
  geom_errorbar(data=B5,aes(x=330,ymin=lo,ymax=hi),width=45,colour=INK,linewidth=0.35)+
  geom_point(data=B5,aes(x=330,y=m),shape=21,fill="white",colour=INK,size=2.2,stroke=0.6)+
  scale_x_continuous(breaks=c(100,250,500,1000,1500,2000))+scale_y_continuous(limits=c(0.25,0.7))+
  labs(x="upstream sequence read (nt), downstream to +500",y="precision at 1",title="A   Window length")+TH

SC<-rbind(read.delim("data/splice_scramble.tsv")%>%mutate(mode="bases permuted"),
          read.delim("data/splice_resample.tsv")%>%mutate(mode="bases resampled"))
lv<-c("acc","don","both")
B<-SC%>%filter(arm!="intact")%>%mutate(target=sub("ctrl_","",arm),kind=ifelse(grepl("^ctrl",arm),"random positions","splice sites"))%>%
  group_by(mode,target,kind)%>%summarise(m=mean(test_p_at_1),lo=min(test_p_at_1),hi=max(test_p_at_1),.groups="drop")%>%
  mutate(target=factor(target,levels=lv,labels=c("acceptors","donors","both")),kind=factor(kind,levels=c("splice sites","random positions")))
I0<-SC%>%filter(arm=="intact",mode=="bases permuted")%>%summarise(m=mean(test_p_at_1))
pB<-ggplot(B,aes(target,m,shape=kind,fill=kind))+
  geom_hline(yintercept=I0$m,colour=CAN,linewidth=0.4)+
  annotate("text",x=-Inf,y=I0$m+0.014,label="  intact",hjust=0,colour=CAN,size=2.8)+
  geom_hline(yintercept=CH,linetype="22",colour=GREY)+
  annotate("text",x=-Inf,y=CH+0.014,label="  chance",hjust=0,colour=GREY,size=2.8)+
  geom_errorbar(aes(ymin=lo,ymax=hi),position=position_dodge(0.45),width=0.15,linewidth=0.35,colour=INK)+
  geom_point(position=position_dodge(0.45),size=2.4,colour=INK,stroke=0.6)+
  scale_shape_manual(values=c("splice sites"=21,"random positions"=21),name=NULL)+
  scale_fill_manual(values=c("splice sites"=CAN,"random positions"="white"),name=NULL)+
  scale_y_continuous(limits=c(0.25,0.7))+
  facet_wrap(~mode)+
  labs(x="sites modified",y="precision at 1",title="B   Strong splice sites modified in place")+TH+
  theme(strip.background=element_blank(),strip.text=element_text(size=9,colour=INK))+
  theme(legend.position=c(0.98,0.35),legend.justification=c(1,0.5),legend.key.size=unit(3,"mm"),legend.text=element_text(size=8))

L<-harvest("make_figure_loco.R")
pC<-edit_text(L$pC,drop=c("<mapped lab>","filled = real"))+labs(title="C   Dinucleotide shuffle",subtitle=NULL,caption=NULL)+TH

AT<-read.delim("../acceptor_arch_train.tsv")%>%mutate(variant=factor(variant,levels=c("base","aux"),
      labels=c("sequence only","sequence +\nacceptor score")))
M<-AT%>%group_by(variant)%>%summarise(m=mean(test_within_gene),.groups="drop")
pD<-ggplot(AT,aes(variant,test_within_gene))+
  geom_hline(yintercept=acc_wr,linetype="22",colour=GREY)+
  annotate("text",x=-Inf,y=acc_wr+0.01,label="  acceptor score alone",hjust=0,colour=GREY,size=2.8)+
  geom_point(position=position_jitter(width=0.08,seed=1),colour=CAN,size=1.8,alpha=0.85)+
  geom_errorbar(data=M,aes(y=m,ymin=m,ymax=m),width=0.4,colour=INK,linewidth=0.6)+
  scale_y_continuous(limits=c(0.58,0.88))+
  labs(x=NULL,y="within-gene win-rate",title="D   Eight seeds per network")+TH

fig<-(pA|pB)/(pC|pD)
ggsave("out/fig3_09_robustness.pdf",fig,width=8.6,height=7,device=cairo_pdf)
ggsave("out/fig3_09_robustness.png",fig,width=8.6,height=7,dpi=300,bg="white")
cat("[out] out/fig3_09_robustness.{pdf,png}\n")
