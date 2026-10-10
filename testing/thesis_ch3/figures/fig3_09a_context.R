# draft panel for figure 3.9: the same network retrained at seven window lengths
suppressPackageStartupMessages({library(ggplot2);library(dplyr)})
INK<-"#1A1A1A"; GTX<-"#5A6068"; GREY<-"#8A8F98"; CAN<-"#0072B2"
D<-read.delim("data/context_sweep.tsv")
S<-D%>%group_by(context,start,end,length)%>%
  summarise(m=mean(test_p_at_1),lo=min(test_p_at_1),hi=max(test_p_at_1),.groups="drop")%>%
  mutate(up=-start)
U<-S%>%filter(end==500); B<-S%>%filter(context=="span500")
p<-ggplot(U,aes(up,m))+
  geom_line(colour=CAN,linewidth=0.8)+
  geom_errorbar(aes(ymin=lo,ymax=hi),width=40,colour=CAN,linewidth=0.4)+
  geom_point(colour=CAN,size=2.2)+
  geom_point(data=B,aes(x=250,y=m),shape=21,fill="white",colour=INK,size=2.6,stroke=0.7)+
  geom_errorbar(data=B,aes(x=250,ymin=lo,ymax=hi),width=40,colour=INK,linewidth=0.4)+
  annotate("text",x=290,y=B$m-0.012,label="baseline span, −250 to +250",hjust=0,colour=INK,size=3)+
  scale_x_continuous(breaks=c(100,250,500,1000,1500,2000))+
  labs(x="upstream sequence read (nt); downstream fixed at +500",y="precision at 1 (held-out genes)",
       title="Retraining one network at seven window lengths",
       subtitle="5 seeds per window; points are means, bars the seed range; 1,623 held-out genes")+
  theme_classic(base_size=11)+
  theme(plot.title=element_text(face="bold",colour=INK),plot.subtitle=element_text(colour=GTX,size=9),
        axis.text=element_text(colour=GTX))
ggsave("out/fig3_09a_context_draft.png",p,width=130,height=90,units="mm",dpi=300,bg="white")
cat("[out] out/fig3_09a_context_draft.png\n")
