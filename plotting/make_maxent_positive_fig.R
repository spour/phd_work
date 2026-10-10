#!/usr/bin/env Rscript
# maxent positive-control calibration, mirroring u1_positive_calibration.png
suppressMessages({library(ggplot2);library(dplyr);library(tidyr);library(patchwork)})
INK<-"#1A1A1A"; BASE<-13; CAN<-"#0072B2"; COMP<-"#E69F00"; NUL<-"#8A8F98"; CUT<-8.0
th<-theme_classic(base_size=BASE)+theme(
  plot.title=element_text(face="bold",colour=INK),plot.subtitle=element_text(colour=INK,size=BASE-3),
  legend.position="top",legend.title=element_blank(),
  axis.text=element_text(colour=INK),axis.title=element_text(colour=INK))

mk<-function(pfx,klab,anch){
  D<-read.delim(paste0(pfx,"_positive_dist.tsv"))
  Dl<-pivot_longer(D,c(annotated,intronic,allkmer),names_to="src",values_to="d")
  Dl$src<-factor(Dl$src,levels=c("annotated","intronic","allkmer"),
                 labels=c(paste0("annotated ",anch),"random intronic",paste0("all ",klab)))
  pA<-ggplot(Dl,aes(score,d,colour=src))+
    geom_step(direction="mid",linewidth=0.8)+
    geom_vline(xintercept=CUT,linetype="dashed",colour=INK,linewidth=0.5)+
    annotate("text",x=CUT+0.3,y=Inf,label="cutoff 8",hjust=0,vjust=1.4,size=3,colour=INK)+
    scale_colour_manual(values=setNames(c(CAN,COMP,NUL),levels(Dl$src)))+
    labs(title=sprintf("Real %s carry high MaxEnt scores",anch),
         subtitle="annotated splice sites sit far above intronic background",
         x=sprintf("MaxEnt %s score (bits)",anch),y="density")+th
  C<-read.delim(paste0(pfx,"_positive_calibration.tsv"),comment.char="#")
  C<-C[is.finite(C$sensitivity)&is.finite(C$enrich_vs_intron)&C$enrich_vs_intron>0&C$sensitivity>0,]
  lab<-C[C$threshold %in% c(4,6,8,10),]
  cut8<-C[which.min(abs(C$threshold-CUT)),]
  pB<-ggplot(C,aes(sensitivity,enrich_vs_intron))+
    geom_line(colour=CAN,linewidth=0.8)+geom_point(colour=CAN,size=1.1)+
    geom_point(data=lab,colour=INK,size=2)+
    geom_point(data=cut8,colour="red",size=2.6)+
    geom_text(data=lab,aes(label=sprintf("%.0f",threshold)),hjust=-0.3,vjust=0.3,size=3,colour=INK)+
    geom_text(data=cut8,aes(label=sprintf("8 (sens %.0f%%, %.0fx)",100*sensitivity,enrich_vs_intron)),
              hjust=-0.15,vjust=1.6,size=3,colour="red")+
    scale_y_log10()+scale_x_reverse()+
    labs(title="Sensitivity and enrichment trade off with the cutoff",
         subtitle="labels are MaxEnt score thresholds; enrichment is annotated over intronic",
         x="sensitivity (fraction of annotated sites kept)",y="fold enrichment vs intronic (log)")+th
  fig<-pA|pB
  ggsave(paste0(pfx,"_positive_calibration.png"),fig,width=13,height=5.2,dpi=170,bg="white")
  ggsave(paste0(pfx,"_positive_calibration.pdf"),fig,width=13,height=5.2,device=cairo_pdf,bg="white")
  cat(sprintf("[out] %s_positive_calibration.png/.pdf\n",pfx))
}
mk("maxent5","9-mers","5'ss")
mk("maxent3","23-mers","3'ss")
