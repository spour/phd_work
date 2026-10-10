#!/usr/bin/env Rscript
# supplementary: the almada 2013 protection-window sweep and the controls that show it
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(patchwork)})
INK<-"#1A1A1A"; CAN<-"#0072B2"; COMP<-"#E69F00"; GRY<-"#8A8F98"
GRN<-"#009E73"; PLM<-"#CC79A7"; VER<-"#D55E00"; BASE<-7
th<-theme_classic(base_size=BASE,base_family="sans")+theme(
  plot.title=element_text(face="bold",size=BASE+.5,colour=INK,hjust=0,margin=margin(b=2)),
  axis.text=element_text(colour=INK,size=BASE-.5),axis.title=element_text(colour=INK,size=BASE),
  axis.line=element_line(linewidth=.25),axis.ticks=element_line(linewidth=.25),
  legend.position="none",plot.margin=margin(3,5,3,4))
PC<-c(upstream=CAN,downstream=VER)
sweep_panel<-function(d,ttl,sub=NULL,dashed=NULL){
  p<-ggplot(d,aes(W,winrate,colour=direction,fill=direction))+
    geom_hline(yintercept=0.5,linetype="dashed",colour=GRY,linewidth=.25)+
    geom_ribbon(aes(ymin=lo,ymax=hi),alpha=.18,colour=NA)+
    geom_line(linewidth=.4)+geom_point(size=1.1)+
    scale_colour_manual(values=PC)+scale_fill_manual(values=PC)+
    scale_x_log10(breaks=c(100,200,500,1000,2000),labels=c("100","200","500","1k","2k"))+
    coord_cartesian(ylim=c(.34,.52))+
    labs(title=ttl,x="window W (nt), log scale",y="within-gene win-rate")+th
  if(!is.null(dashed)) p<-p+geom_line(data=dashed,aes(W,winrate,colour=direction),
                                      linetype="22",linewidth=.35,inherit.aes=FALSE)
  p
}
A<-read.delim("almada_sweep.tsv",comment.char="#") %>%
  filter(grepl("^protected_W",feature)) %>%
  mutate(W=as.numeric(sub("protected_W","",feature)),cut=ifelse(grepl("^8.46",cutoff),"8.46","8.77"),
         direction=factor(direction,levels=c("upstream","downstream")),lo=win_lo,hi=win_hi)
C<-read.delim("almada_controls.tsv") %>%
  filter(grepl("^W",feature)) %>%
  mutate(W=as.numeric(sub("W","",feature)),direction=factor(direction,levels=c("upstream","downstream")))

pa<-sweep_panel(A %>% filter(cut=="8.46"),"a  Scan-defined strong donors",
                dashed=A %>% filter(cut=="8.77"))+
  annotate("text",x=2000,y=.512,hjust=1,size=1.9,colour=CAN,label="upstream")+
  annotate("text",x=2000,y=.487,hjust=1,size=1.9,colour=VER,label="downstream")+
  annotate("text",x=100,y=.352,hjust=0,size=1.9,colour=INK,
           label="solid, cutoff 8.46; dashed, 8.77\nupstream deficit at every W, none downstream")
pb<-sweep_panel(C %>% filter(analysis=="annotated donors"),"b  GENCODE-annotated 5'ss only")+
  annotate("text",x=100,y=.352,hjust=0,size=1.9,colour=INK,
           label="only 9.1% of scan hits are annotated 5'ss;\nthe asymmetry inverts and grows downstream")
pc<-sweep_panel(C %>% filter(analysis=="control motif"),"c  Composition-matched control motif")+
  annotate("text",x=100,y=.352,hjust=0,size=1.9,colour=INK,
           label="column-permuted donor PWM, GT destroyed,\nmatched hit density: the deficit is reproduced")
CMP<-bind_rows(
  C %>% filter(analysis=="scan donors 8.46",direction=="upstream",W %in% c(500,1000)) %>%
    transmute(W,what="real donor motif",winrate,lo,hi),
  C %>% filter(analysis=="control motif",direction=="upstream",W %in% c(500,1000)) %>%
    transmute(W,what="control motif",winrate,lo,hi),
  C %>% filter(analysis=="annotated donors",direction=="upstream",W %in% c(500,1000)) %>%
    transmute(W,what="annotated 5'ss",winrate,lo,hi)) %>%
  mutate(what=factor(what,levels=rev(c("real donor motif","control motif","annotated 5'ss"))),
         Wl=factor(paste0("W = ",W),levels=c("W = 500","W = 1000")))
pd<-ggplot(CMP,aes(winrate,what,colour=what))+
  geom_vline(xintercept=.5,linetype="dashed",colour=GRY,linewidth=.25)+
  geom_errorbar(aes(xmin=lo,xmax=hi),orientation="y",width=0,linewidth=.45)+
  geom_point(size=1.3)+facet_wrap(~Wl)+
  scale_colour_manual(values=c("real donor motif"=CAN,"control motif"=PLM,"annotated 5'ss"=GRN))+
  geom_text(aes(label=sprintf("%.3f",winrate)),vjust=-1.0,size=2.0)+
  coord_cartesian(xlim=c(.44,.52))+
  labs(title="d  A non-donor motif does the same work",
       x="upstream win-rate, 95% CI over genes",y=NULL)+th+
  theme(strip.background=element_blank(),strip.text=element_text(colour=INK,size=BASE-.5),
        axis.text.y=element_text(size=BASE-1))
fig<-(pa|pb)/(pc|pd)+plot_annotation(
  title="Supplementary: the protection-window sweep is composition, not U1 recognition",
  theme=theme(plot.title=element_text(face="bold",size=BASE+1.5,colour=INK,family="sans")))
ggsave("figure_supp_almada.pdf",fig,width=180,height=110,units="mm",device=grDevices::cairo_pdf)
ggsave("figure_supp_almada.png",fig,width=180,height=110,units="mm",dpi=400)
cat("wrote figure_supp_almada.pdf / .png\n")
