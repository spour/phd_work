#!/usr/bin/env Rscript
# task a figure: u1 cutoff calibrated against annotated gencode donors (positive control)
suppressMessages({library(ggplot2); library(dplyr); library(tidyr); library(patchwork)})
INK<-"#1A1A1A"; BASE<-13; CAN<-"#0072B2"; COMP<-"#E69F00"; NUL<-"#8A8F98"
OLD<-4.0; PRIM<-5.25
th<-theme_classic(base_size=BASE)+theme(
  plot.title=element_text(face="bold",colour=INK),
  plot.subtitle=element_text(colour=INK,size=BASE-2),
  legend.position="top", legend.title=element_blank(),
  axis.text=element_text(colour=INK), axis.title=element_text(colour=INK))

D<-read.delim("u1_positive_dist.tsv")
Dl<-pivot_longer(D,c(donor_gt,intronic,all_9mer),names_to="src",values_to="d")
Dl$src<-factor(Dl$src,levels=c("donor_gt","intronic","all_9mer"),
               labels=c("annotated donors","random intronic","all 9-mers"))
pA<-ggplot(Dl,aes(score,d,colour=src))+
  geom_step(direction="mid",linewidth=0.8)+
  geom_vline(xintercept=OLD,linetype="dashed",colour=NUL,linewidth=0.4)+
  geom_vline(xintercept=PRIM,linetype="dashed",colour=INK,linewidth=0.5)+
  annotate("text",x=OLD-0.15,y=Inf,label="old 4.0",hjust=1,vjust=1.4,size=3,colour=NUL)+
  annotate("text",x=PRIM+0.15,y=Inf,label="primary 5.25",hjust=0,vjust=1.4,size=3,colour=INK)+
  scale_colour_manual(values=c("annotated donors"=CAN,"random intronic"=COMP,"all 9-mers"=NUL))+
  labs(title="Real 5'ss carry high U1 scores",
       subtitle="annotated donors sit far above intronic background",
       x="U1 score (-dG)", y="density")+th

C<-read.delim("u1_positive_calibration.tsv",comment.char="#")
C<-C[is.finite(C$sensitivity)&is.finite(C$enrich_vs_intron)&C$enrich_vs_intron>0,]
lab<-C%>%filter(threshold %in% c(4.0,5.25,6.5,7.0,8.0))
pB<-ggplot(C,aes(sensitivity,enrich_vs_intron))+
  geom_line(colour=CAN,linewidth=0.8)+geom_point(colour=CAN,size=1.1)+
  geom_point(data=lab,colour=INK,size=2)+
  geom_text(data=lab,aes(label=sprintf("%.2f",threshold)),hjust=-0.2,vjust=0.3,size=3,colour=INK)+
  scale_y_log10()+scale_x_reverse()+
  labs(title="Sensitivity and enrichment trade off with the cutoff",
       subtitle="labels are U1 score thresholds; enrichment is donors over intronic",
       x="sensitivity (fraction of annotated donors kept)",
       y="fold enrichment vs intronic (log)")+th
fig<-pA|pB
ggsave("u1_positive_calibration.png",fig,width=13,height=5.2,dpi=170,bg="white")
cat("[out] u1_positive_calibration.png\n")
