#!/usr/bin/env Rscript
# where the max-scoring upstream 3'ss sits, by class
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr)})
e<-function(...) cat(...,"\n",file=stderr())
INK<-"#1A1A1A"; BASE<-7
REAL<-"#08519C"; COMP<-"#E8A33D"; DEC<-"#7B5EA7"; GREY<-"#9AA0A6"; GTX<-"#5A6068"
LO<--2000; HI<--126
D<-read_tsv("acceptor_position_scan.tsv",show_col_types=FALSE)%>%filter(!is.na(position))
S<-read_tsv("acceptor_position_stats.tsv",show_col_types=FALSE)%>%filter(subset=="all windows")
LV<-c("real","competitor","decoy"); LB<-c("real 3' end","competitor","intronic decoy")
D<-D%>%mutate(class=factor(class,levels=LV,labels=LB))
S<-S%>%mutate(class=factor(class,levels=LV,labels=LB))
U<-1/(HI-LO)
e(sprintf("[fig] %s median %+.0f IQR %.0f",S$class,S$median,S$iqr))
p<-ggplot(D,aes(position,color=class,fill=class))+
  geom_hline(yintercept=U,linetype=2,color=GREY,linewidth=.45)+
  geom_density(alpha=.10,linewidth=.6,adjust=1.1)+
  geom_point(data=S,aes(x=median,y=0),size=1.8,shape=17,show.legend=FALSE)+
  scale_color_manual(values=setNames(c(REAL,COMP,DEC),LB))+
  scale_fill_manual(values=setNames(c(REAL,COMP,DEC),LB))+
  annotate("text",x=LO+20,y=U,label="uniform across the window",hjust=0,vjust=-0.7,
           size=2.1,color=GTX)+
  annotate("text",x=HI-20,y=0,hjust=1,vjust=-0.5,size=2.1,color=INK,lineheight=1.35,
    label=paste0("median position, IQR\n",
      paste(sprintf("%s  %+.0f, %.0f nt",S$class,S$median,S$iqr),collapse="\n"))) +
  scale_x_continuous(breaks=seq(-2000,0,500),limits=c(LO,HI),expand=expansion(mult=c(0,0)))+
  scale_y_continuous(expand=expansion(mult=c(0,.06)))+
  labs(title="Position of the maximum-scoring upstream 3' splice site relative to the cleavage site",
       x="position of the winning acceptor relative to the cleavage site (nt)",
       y="density of sites")+
  theme_classic(base_size=BASE)+
  theme(text=element_text(color=INK),axis.text=element_text(color=INK,size=BASE-1),
        axis.line=element_line(color=INK,linewidth=.3),axis.title=element_text(size=BASE),
        plot.title=element_text(size=BASE+1,face="bold",hjust=0,margin=margin(b=5)),
        legend.title=element_blank(),legend.position=c(.02,.02),
        legend.justification=c(0,0),legend.text=element_text(size=BASE-1),
        legend.key.height=unit(.7,"lines"),legend.background=element_rect(fill="white",color=NA),
        panel.grid.major.y=element_line(color="#EEF0F3",linewidth=.25))
ggsave("figure_acceptor_position.pdf",p,width=5.4,height=3.3,device=grDevices::cairo_pdf)
ggsave("figure_acceptor_position.png",p,width=5.4,height=3.3,dpi=300)
e("[out] figure_acceptor_position.pdf figure_acceptor_position.png")
