#!/usr/bin/env Rscript
# supplementary: why feature 2 is gc_contrast and not u1_contrast
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(patchwork)})
INK<-"#1A1A1A"; CAN<-"#0072B2"; COMP<-"#E69F00"; GRY<-"#8A8F98"
LITE<-"#56B4E9"; GRN<-"#009E73"; PLM<-"#CC79A7"; VER<-"#D55E00"; BASE<-7
th<-theme_classic(base_size=BASE,base_family="sans")+theme(
  plot.title=element_text(face="bold",size=BASE+.5,colour=INK,hjust=0,margin=margin(b=2)),
  axis.text=element_text(colour=INK,size=BASE-.5),axis.title=element_text(colour=INK,size=BASE),
  axis.line=element_line(linewidth=.25),axis.ticks=element_line(linewidth=.25),
  legend.position="none",plot.margin=margin(3,5,3,4))
CC<-read.delim("control_contrast.tsv")

PR<-read.delim("u1_gc_profile_wide.tsv") %>% filter(measure=="u1_mean_negdG")
pa<-ggplot(PR,aes(position,value))+
  annotate("rect",xmin=-2000,xmax=-438,ymin=-Inf,ymax=Inf,fill=LITE,alpha=.16)+
  annotate("rect",xmin=-437,xmax=-126,ymin=-Inf,ymax=Inf,fill=PLM,alpha=.20)+
  geom_ribbon(aes(ymin=ci_lo,ymax=ci_hi,fill=class),alpha=.20,colour=NA)+
  geom_line(aes(colour=class),linewidth=.4)+
  scale_colour_manual(values=c(canonical=CAN,competitor=COMP))+
  scale_fill_manual(values=c(canonical=CAN,competitor=COMP))+
  scale_x_continuous(breaks=c(-2000,-1000,-437,0,500))+
  labs(title="a  The retired U1 contrast",
       x="position relative to the pA (nt)",y="mean U1 -dG per 100 nt bin")+th+
  annotate("text",x=-1960,y=min(PR$ci_lo),hjust=0,vjust=-.2,size=1.9,colour=INK,
           label="same two windows as gc_contrast;\ncanonical below competitor at every position")

MG<-CC %>% filter(kind=="marginal") %>%
  mutate(short=recode(name,"u1_contrast (real U1 table)"="real U1 table",
                      "permuted U1 table, seed 0"="permuted table, seed 0",
                      "permuted U1 table, seed 1"="permuted table, seed 1",
                      "permuted U1 table, seed 2"="permuted table, seed 2",
                      "U1 table on reverse complement"="U1 table, reverse complement",
                      "MaxEnt 5'ss over the same windows"="MaxEnt 5'ss, same windows",
                      "GC fraction, same windows"="GC, same windows"))
MG$short<-factor(MG$short,levels=rev(MG$short))
MC<-c("real U1 table"=CAN,"permuted table, seed 0"=GRY,"permuted table, seed 1"=GRY,
      "permuted table, seed 2"=GRY,"U1 table, reverse complement"=VER,
      "MaxEnt 5'ss, same windows"=GRN,"GC, same windows"=PLM)
pb<-ggplot(MG,aes(excess,short,colour=short))+
  geom_vline(xintercept=0,linetype="dashed",colour=GRY,linewidth=.25)+
  geom_point(size=1.4)+scale_colour_manual(values=MC)+
  geom_text(aes(label=sprintf("%+.3f",excess)),vjust=-1.1,size=1.9)+
  scale_x_continuous(limits=c(-.02,.115))+
  labs(title="b  Same windows, different table",
       x="AUROC excess over each score's own null",y=NULL)+th+
  theme(axis.text.y=element_text(size=BASE-1.5))+
  scale_y_discrete(expand=expansion(add=c(.7,.9)))

PA<-CC %>% filter(kind=="partition") %>%
  mutate(short=ifelse(grepl("^u1",name),"u1_contrast,\nGC gradient removed",
                      "gc_contrast,\nu1_contrast removed"))
PA$short<-factor(PA$short,levels=rev(PA$short))
pc<-ggplot(PA,aes(excess,short,colour=short))+
  geom_vline(xintercept=0,linetype="dashed",colour=GRY,linewidth=.25)+
  geom_point(size=1.5)+
  scale_colour_manual(values=setNames(c(CAN,PLM),levels(PA$short)[c(2,1)]))+
  geom_text(aes(label=sprintf("%+.3f, win-rate %.3f",excess,winrate)),vjust=-1.2,size=1.9,colour=INK)+
  scale_x_continuous(limits=c(-.01,.10))+
  scale_y_discrete(expand=expansion(add=c(.8,1.0)))+
  labs(title="c  Which one is the signal",
       x="AUROC excess after removing the other",y=NULL)+th+
  theme(axis.text.y=element_text(size=BASE-1.5))

MD<-CC %>% filter(kind=="model") %>%
  filter(name %in% c("acceptor alone","acceptor + u1_contrast","acceptor + GC fraction, same windows",
                     "acceptor + u1_contrast, GC gradient removed","acceptor + both contrasts")) %>%
  mutate(short=recode(name,"acceptor alone"="acceptor alone",
                      "acceptor + u1_contrast"="+ u1_contrast",
                      "acceptor + GC fraction, same windows"="+ gc_contrast",
                      "acceptor + u1_contrast, GC gradient removed"="+ u1_contrast, GC removed",
                      "acceptor + both contrasts"="+ both"))
MD$short<-factor(MD$short,levels=rev(c("acceptor alone","+ u1_contrast",
                                       "+ u1_contrast, GC removed","+ gc_contrast","+ both")))
pd<-ggplot(MD,aes(winrate,short))+
  geom_vline(xintercept=.5,linetype="dashed",colour=GRY,linewidth=.25)+
  geom_errorbar(aes(xmin=win_lo,xmax=win_hi),orientation="y",width=0,linewidth=.45,colour=CAN)+
  geom_point(size=1.4,colour=CAN)+
  geom_text(aes(label=sprintf("%.3f",winrate)),vjust=-1.2,size=1.9,colour=INK)+
  scale_x_continuous(limits=c(.49,.70))+
  scale_y_discrete(expand=expansion(add=c(.7,.9)))+
  labs(title="d  At the model level",
       x="within-gene win-rate, holdout, null 0.500",y=NULL)+th+
  theme(axis.text.y=element_text(size=BASE-1.5))

fig<-(pa|pb)/(pc|pd)+plot_annotation(
  title="Supplementary: the U1 contrast is a GC gradient, and gc_contrast measures it directly",
  theme=theme(plot.title=element_text(face="bold",size=BASE+1.5,colour=INK,family="sans")))
ggsave("figure_supp_u1.pdf",fig,width=180,height=110,units="mm",device=grDevices::cairo_pdf)
ggsave("figure_supp_u1.png",fig,width=180,height=110,units="mm",dpi=400)
cat("wrote figure_supp_u1.pdf / .png\n")
