#!/usr/bin/env Rscript
# panel b: within-gene recall@k on the analysis set
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr)})
e<-function(...) cat(...,"\n",file=stderr())
INK<-"#1A1A1A"; BASE<-7; BLU<-"#08519C"; GREY<-"#9AA0A6"; GTX<-"#5A6068"
D<-read_tsv("recall_at_k.tsv",show_col_types=FALSE)
NG<-D$n_genes[1]; KM<-max(D$k)
stopifnot(NG==4847)
g<-function(k,col="recall") D[[col]][D$k==k][1]
e(sprintf("[panel] %d analysis-set genes, k up to %d; recall@1 %.3f (chance %.3f), recall@3 %.3f (chance %.3f)",
          NG,KM,g(1),g(1,"chance"),g(3),g(3,"chance")))
A<-D%>%filter(k %in% c(1,3))%>%
  mutate(lab=sprintf("recall@%d   %.3f\nrandom       %.3f",k,recall,chance))
p<-ggplot(D,aes(k))+
  geom_ribbon(aes(ymin=chance,ymax=recall),fill=BLU,alpha=.13)+
  geom_line(aes(y=chance),linetype=2,color=GREY,linewidth=.6)+
  geom_point(aes(y=chance),color=GREY,size=1.2)+
  geom_line(aes(y=recall),color=BLU,linewidth=.85)+
  geom_point(aes(y=recall),color=BLU,size=1.7)+
  geom_point(data=D%>%filter(k==1),aes(y=recall),size=3.2,shape=21,fill=BLU,
             color="white",stroke=.8)+
  geom_text(data=A,aes(y=recall,label=lab),hjust=-0.10,vjust=1.35,size=2.3,
            color=INK,lineheight=1.25)+
  annotate("text",x=KM+0.4,y=g(KM,"chance"),label="random\nranking",hjust=1,vjust=3.1,
           size=2.2,color=GREY,lineheight=1.1)+
  scale_x_continuous(breaks=1:KM,limits=c(1,KM+0.55),expand=expansion(mult=c(.02,.02)))+
  scale_y_continuous(limits=c(0.20,1),labels=scales::percent_format(accuracy=1),
                     expand=expansion(mult=c(0,.02)))+
  labs(title="Within-gene recall as a function of rank threshold",
       x="rank threshold k",y="genes with the true 3' end at rank <= k")+
  theme_classic(base_size=BASE)+
  theme(text=element_text(color=INK),
        axis.text=element_text(color=INK,size=BASE-1),
        axis.line=element_line(color=INK,linewidth=.3),axis.title=element_text(size=BASE),
        plot.title=element_text(size=BASE+1,face="bold",hjust=0,margin=margin(b=5)),
        panel.grid.major.y=element_line(color="#EEF0F3",linewidth=.25),
        legend.position="none")
ggsave("panelB_v3_recall_at_k.pdf",p,width=4.6,height=3.3,device=grDevices::cairo_pdf)
ggsave("panelB_v3_recall_at_k.png",p,width=4.6,height=3.3,dpi=300)
e("[out] panelB_v3_recall_at_k.pdf panelB_v3_recall_at_k.png")
