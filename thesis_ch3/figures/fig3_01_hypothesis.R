# figure 3.1, the chapter's hypothesis as a cartoon
suppressPackageStartupMessages({library(ggplot2);library(patchwork)})
INK<-"#1A1A1A"; GTX<-"#5A6068"; GREY<-"#8A8F98"; PAPER<-"#FCFCFD"
CAN<-"#0072B2"; COMP<-"#E69F00"; EXON<-"#D9DDE3"; U1C<-"#6A5ACD"
TS<-3.1
tt<-function(s) ggtitle(s)
TH<-theme_void(base_size=10)+theme(
  plot.title=element_text(face="bold",colour=INK,size=11,margin=margin(b=4)),
  plot.margin=margin(6,8,6,8), plot.background=element_rect(fill="white",colour=NA))

# a  gene
ex<-data.frame(x1=c(2,20,40,60,84), x2=c(8,25,45,65,98))
comp<-c(12,31,34,52,72,77)
pA<-ggplot()+
  annotate("segment",x=2,xend=98,y=0,yend=0,colour=GREY,linewidth=0.5)+
  geom_rect(data=ex,aes(xmin=x1,xmax=x2,ymin=-1,ymax=1),fill=EXON,colour=INK,linewidth=0.3)+
  annotate("segment",x=comp,xend=comp,y=0,yend=2.2,colour=COMP,linewidth=0.6)+
  annotate("point",x=comp,y=2.2,colour=COMP,size=2.4)+
  annotate("segment",x=97,xend=97,y=1,yend=3.2,colour=CAN,linewidth=0.8)+
  annotate("point",x=97,y=3.2,colour=CAN,size=3.4)+
  annotate("text",x=97,y=4.3,label="used 3′ end",colour=CAN,size=TS,fontface="bold",hjust=0.8)+
  annotate("text",x=44,y=3.6,colour=COMP,size=TS,fontface="bold",
           label="competitors")+
  annotate("text",x=91,y=-1.9,label="terminal exon",colour=GTX,size=TS-0.3)+
  annotate("text",x=0,y=0,label="5′",colour=GTX,size=TS,hjust=1)+
  annotate("text",x=100,y=0,label="3′",colour=GTX,size=TS,hjust=0)+
  coord_cartesian(xlim=c(-2,102),ylim=c(-2.4,4.8),clip="off")+
  tt("A   One used 3′ end and its intragenic competitors")+TH

# b  the two windows
yR<-2.2; yC<-0
u1R<-c(-1800,-1450,-1150)
u1C<-c(-1250,-1000,-760,-520,-380,-240,-120,160,380)
pB<-ggplot()+
  annotate("segment",x=-2000,xend=-700,y=yR,yend=yR,colour=GREY,linewidth=0.5)+
  annotate("rect",xmin=-700,xmax=0,ymin=yR-0.3,ymax=yR+0.3,fill=EXON,colour=INK,linewidth=0.3)+
  annotate("segment",x=0,xend=500,y=yR,yend=yR,colour=GREY,linewidth=0.4,linetype="22")+
  annotate("segment",x=-700,xend=-700,y=yR+0.3,yend=yR+0.75,colour=INK,linewidth=0.5)+
  annotate("text",x=-700,y=yR+0.98,label="acceptor (3′ ss)",colour=INK,size=TS-0.3)+
  annotate("segment",x=u1R,xend=u1R,y=yR-0.18,yend=yR+0.18,colour=U1C,linewidth=0.6)+
  annotate("segment",x=0,xend=0,y=yR+0.3,yend=yR+0.85,colour=CAN,linewidth=0.8)+
  annotate("point",x=0,y=yR+0.85,colour=CAN,size=3)+
  annotate("text",x=-350,y=yR-0.62,label="terminal exon",colour=GTX,size=TS-0.3)+
  annotate("text",x=530,y=yR,label="real 3′ end",colour=CAN,size=TS,fontface="bold",hjust=0)+
  annotate("segment",x=-2000,xend=500,y=yC,yend=yC,colour=GREY,linewidth=0.5)+
  annotate("rect",xmin=-1750,xmax=-1500,ymin=yC-0.3,ymax=yC+0.3,fill=EXON,colour=INK,linewidth=0.3)+
  annotate("segment",x=-1500,xend=-1500,y=yC+0.3,yend=yC+0.75,colour=INK,linewidth=0.5)+
  annotate("text",x=-1500,y=yC+0.98,label="donor (5′ ss)",colour=INK,size=TS-0.3)+
  annotate("segment",x=u1C,xend=u1C,y=yC-0.18,yend=yC+0.18,colour=U1C,linewidth=0.6)+
  annotate("point",x=c(-1500,-380),y=yC-0.55,colour=U1C,size=5.2,alpha=0.9)+
  annotate("text",x=c(-1500,-380),y=yC-0.55,label="U1",colour="white",size=2.3,fontface="bold")+
  annotate("segment",x=0,xend=0,y=yC+0.3,yend=yC+0.85,colour=COMP,linewidth=0.8)+
  annotate("point",x=0,y=yC+0.85,colour=COMP,size=3)+
  annotate("point",x=0,y=yC+0.85,shape=4,colour=INK,size=3.6,stroke=1)+
  annotate("text",x=530,y=yC,label="competitor",colour=COMP,size=TS,fontface="bold",hjust=0)+
  annotate("segment",x=-2000,xend=-1960,y=-1.6,yend=-1.6,colour=U1C,linewidth=0.6)+
  annotate("segment",x=-1980,xend=-1980,y=-1.78,yend=-1.42,colour=U1C,linewidth=0.6)+
  annotate("text",x=-1930,y=-1.6,label="donor-like site",colour=GTX,size=TS-0.3,hjust=0)+
  annotate("segment",x=-2000,xend=500,y=-2.15,yend=-2.15,colour=INK,linewidth=0.3,
           arrow=arrow(ends="both",length=unit(1.6,"mm"),type="closed"))+
  annotate("text",x=-750,y=-2.45,label="2,500 nt window",
           colour=INK,size=TS-0.3)+
  scale_x_continuous(breaks=c(-2000,-1500,-1000,-500,0,500))+
  coord_cartesian(xlim=c(-2050,850),ylim=c(-2.6,3.4),clip="off")+
  labs(x="position relative to the site (nt)")+
  tt("B   Hypothesised splicing environment of each site")+TH+
  theme(axis.text.x=element_text(colour=GTX,size=8),axis.ticks.x=element_line(colour=GREY),
        axis.ticks.length.x=unit(1.2,"mm"),
        axis.title.x=element_text(colour=GTX,size=8,margin=margin(t=3)))

# c  predictions
P<-data.frame(
  x=c(1,2,3),
  head=c("Prediction 1","Prediction 2","Prediction 3"),
  body=c("Fewer strong donors and\nU1 sites upstream of\na used 3′ end",
         "A strong upstream acceptor\nmarks a used 3′ end",
         "A network reading the same\n2,500 nt learns these\nsignals, and others"),
  test=c("Figures 3.3, 3.4","Figures 3.5, 3.6","Figures 3.7 to 3.11"))
W<-0.46
pC<-ggplot(P)+
  geom_rect(aes(xmin=x-W,xmax=x+W,ymin=0,ymax=1),fill=PAPER,colour=GREY,linewidth=0.3)+
  geom_text(aes(x=x-W+0.04,y=0.84,label=head),hjust=0,fontface="bold",colour=INK,size=TS)+
  geom_text(aes(x=x-W+0.04,y=0.5,label=body),hjust=0,colour=INK,size=TS-0.2,lineheight=0.95)+
  geom_text(aes(x=x-W+0.04,y=0.15,label=test),hjust=0,colour=CAN,size=TS-0.3,fontface="italic")+
  coord_cartesian(xlim=c(0.5,3.5),ylim=c(0,1))+
  tt("C   Three predictions, and where each is tested")+TH

fig<-pA/pB+plot_layout(heights=c(0.9,1.7))
dir.create("out",showWarnings=FALSE)
ggsave("out/fig3_01_hypothesis.pdf",fig,width=180,height=140,units="mm",device=cairo_pdf)
ggsave("out/fig3_01_hypothesis.png",fig,width=180,height=140,units="mm",dpi=300,bg="white")
cat("[out] out/fig3_01_hypothesis.{pdf,png}\n")
