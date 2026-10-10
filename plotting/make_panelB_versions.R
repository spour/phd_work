#!/usr/bin/env Rscript
# three candidate panel b designs for the 26-feature model
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork)})
e<-function(...) cat(...,"\n",file=stderr())
INK<-"#1A1A1A"; BASE<-7
BLU<-"#08519C"; PALE<-"#9EC4E4"; GREY<-"#9AA0A6"; GTX<-"#5A6068"
S<-read_tsv("panelB_rank_summary.tsv",show_col_types=FALSE)
GR<-read_tsv("panelB_gene_ranks.tsv",show_col_types=FALSE)
gv<-function(sc,kd,k) S$value[S$scope==sc & S$kind==kd & S$k==k][1]
hv<-function(m,k) S$value[S$scope=="headline" & S$kind==m & S$k==k][1]
th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-1),axis.line=element_line(color=INK,linewidth=.3),
  axis.title=element_text(size=BASE),
  plot.title=element_text(size=BASE+1,face="bold",hjust=0,margin=margin(b=5)),
  legend.position="none")
WR<-hv("within_gene_winrate","value"); WRL<-hv("within_gene_winrate","ci_lo")
WRH<-hv("within_gene_winrate","ci_hi")
P1<-hv("p_at_1","value"); P1R<-hv("p_at_1","reference")
WG<-hv("p_at_1_whole_genome","value"); WGR<-hv("p_at_1_whole_genome","reference")
CONT1<-gv("contested","dist","1"); CH1<-gv("contested","dist_chance","1")
NC<-S$n_genes[S$scope=="contested" & S$kind=="dist"][1]
WGCH<-gv("all genes","dist_chance","1")
e(sprintf("[ref] contested rank-1 %.3f against random ranking %.3f; whole-genome %.3f against random %.3f and baseline %.3f",
          CONT1,CH1,WG,WGCH,WGR))

# v1: rank occupancy
LB<-c("1","2","3","4","5","6+")
D<-data.frame(rank=factor(LB,levels=LB),
              v=sapply(LB,function(k) gv("contested","dist",k)),
              ch=sapply(LB,function(k) gv("contested","dist_chance",k)))
v1<-ggplot(D,aes(rank,v))+
  geom_col(aes(fill=rank=="1"),width=.68,color=NA)+
  scale_fill_manual(values=c(`TRUE`=BLU,`FALSE`=PALE))+
  geom_segment(aes(x=as.numeric(rank)-.34,xend=as.numeric(rank)+.34,y=ch,yend=ch),
               linetype=2,color=GREY,linewidth=.45)+
  geom_text(aes(label=sprintf("%.3f",v)),vjust=-0.6,size=2.4,color=INK,fontface="bold")+
  annotate("text",x=1.42,y=CH1,label="random ranking",hjust=0,vjust=.4,size=2.0,color=GTX)+
  annotate("text",x=4.6,y=Inf,vjust=1.4,hjust=0,size=2.15,color=INK,lineheight=1.35,
    label=paste0("Win-rate ",sprintf("%.3f",WR)," [",sprintf("%.3f",WRL),", ",
      sprintf("%.3f",WRH),"]\n(true site against one random\ncompetitor; chance 0.500)\n\n",
      "All ",format(15560,big.mark=",")," genes: ",sprintf("%.3f",WG),
      " at rank 1\n(baseline ",sprintf("%.3f",WGR),")"))+
  scale_y_continuous(expand=expansion(mult=c(0,.16)))+
  labs(title="B   Where the true 3' end ranks among its gene's candidates",
       x="rank of the true 3' end",y="fraction of genes")+
  th+theme(panel.grid.major.y=element_line(color="#EEF0F3",linewidth=.25))

# v2: rank against acceptor strength
Q<-GR%>%filter(contested==1,!is.na(acceptor))
BR<-Q%>%mutate(g=ntile(acceptor,12))%>%group_by(g)%>%
  summarise(x=median(acceptor),p1=mean(rank==1),pc=mean(percentile),n=n(),.groups="drop")
e(sprintf("[v2] acceptor deciles: P@1 %.3f in the weakest twelfth, %.3f in the strongest",
          BR$p1[1],BR$p1[nrow(BR)]))
v2<-ggplot(Q,aes(acceptor,percentile))+
  geom_hex(bins=34)+
  scale_fill_gradient(low="#EEF3F8",high=BLU,trans="log10",name="genes")+
  geom_line(data=BR,aes(x,pc),inherit.aes=FALSE,color="#B2182B",linewidth=.7)+
  geom_point(data=BR,aes(x,pc),inherit.aes=FALSE,color="#B2182B",size=1.1)+
  annotate("text",x=Inf,y=-Inf,hjust=1.05,vjust=-0.6,size=2.15,color=INK,lineheight=1.35,
    label=paste0("Win-rate ",sprintf("%.3f",WR),"   P@1 ",sprintf("%.3f",CONT1),
                 "\n",format(NC,big.mark=",")," contested genes"))+
  annotate("text",x=-Inf,y=-Inf,hjust=-0.05,vjust=-2.2,size=2.0,color="#B2182B",
           label="median per twelfth of acceptor strength")+
  scale_y_continuous(labels=scales::percent_format(accuracy=1),
                     expand=expansion(mult=c(.02,.04)))+
  labs(title="B   The model places the true end better where its acceptor is stronger",
       x="MaxEnt 3' splice-site score of the true 3' end",
       y="competitors ranked below the true end")+
  th+theme(legend.position="right",legend.title=element_text(size=BASE-1.5,color=INK),
           legend.text=element_text(size=BASE-1.5),legend.key.width=unit(.45,"lines"),
           legend.key.height=unit(.9,"lines"))

# v3: cumulative rank
KK<-1:10
CU<-data.frame(k=rep(KK,2),
               v=c(sapply(KK,function(k) gv("contested","cum",as.character(k))),
                   sapply(KK,function(k) gv("contested","cum_chance",as.character(k)))),
               who=rep(c("model","random ranking"),each=length(KK)))
v3<-ggplot(CU,aes(k,v,color=who,linetype=who))+
  geom_line(linewidth=.65)+geom_point(size=1.3)+
  scale_color_manual(values=c(model=BLU,`random ranking`=GREY))+
  scale_linetype_manual(values=c(model=1,`random ranking`=2))+
  annotate("point",x=1,y=CONT1,size=2.4,shape=21,fill=BLU,color="white",stroke=.6)+
  annotate("text",x=1.15,y=CONT1,hjust=0,vjust=.4,size=2.3,color=BLU,fontface="bold",
           label=sprintf("P@1 = %.3f",CONT1))+
  annotate("text",x=10,y=gv("contested","cum","10"),hjust=1,vjust=-1.0,size=2.15,color=BLU,
           label=sprintf("rank <= 10 for %.1f%% of genes",100*gv("contested","cum","10")))+
  annotate("text",x=6.2,y=gv("contested","cum_chance","6"),hjust=0,vjust=1.9,size=2.0,
           color=GTX,label="random ranking")+
  annotate("text",x=Inf,y=-Inf,hjust=1.05,vjust=-0.6,size=2.15,color=INK,lineheight=1.35,
    label=paste0("Win-rate ",sprintf("%.3f",WR),"\n",format(NC,big.mark=",")," contested genes"))+
  scale_x_continuous(breaks=KK)+
  scale_y_continuous(limits=c(0,1),labels=scales::percent_format(accuracy=1),
                     expand=expansion(mult=c(0,.04)))+
  labs(title="B   How far down the list the true 3' end falls",
       x="rank k",y="genes with the true end at rank <= k")+
  th+theme(panel.grid.major.y=element_line(color="#EEF0F3",linewidth=.25))

sv<-function(p,f,w,h){ ggsave(paste0(f,".pdf"),p,width=w,height=h,device=grDevices::cairo_pdf)
                       ggsave(paste0(f,".png"),p,width=w,height=h,dpi=300) }
sv(v1,"panelB_v1_rankdist",6.6,3.5); sv(v2,"panelB_v2_scatter",6.6,3.6)
sv(v3,"panelB_v3_cumulative",6.6,3.5)
e("[out] panelB_v1_rankdist panelB_v2_scatter panelB_v3_cumulative (.pdf and .png)")
