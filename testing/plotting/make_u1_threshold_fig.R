#!/usr/bin/env Rscript
# item 1 figure: u1 "strong site" threshold calibration
suppressMessages({library(ggplot2); library(dplyr); library(tidyr); library(patchwork)})

INK<-"#1A1A1A"; BASE<-13; CAN<-"#0072B2"; COMP<-"#E69F00"; NUL<-"#8A8F98"
OLD<-4.0; NEW<-7.0
th<-theme_classic(base_size=BASE)+theme(
  plot.title=element_text(face="bold",colour=INK),
  plot.subtitle=element_text(colour=INK,size=BASE-2),
  legend.position="top", legend.title=element_blank(),
  axis.text=element_text(colour=INK), axis.title=element_text(colour=INK))

H<-read.delim("u1_score_hist.tsv")
H$bin<-round(H$score/0.25)*0.25
Hs<-H%>%group_by(bin)%>%summarise(lookup=sum(lookup),observed=sum(observed),
                                  dinuc_null=sum(dinuc_null),.groups="drop")
Hl<-Hs%>%pivot_longer(c(lookup,observed,dinuc_null),names_to="src",values_to="d")
Hl$src<-factor(Hl$src,levels=c("lookup","observed","dinuc_null"),
               labels=c("all 9-mers","peak upstream","dinucleotide shuffle"))

pA<-ggplot(Hl,aes(bin,d,colour=src))+
  geom_step(direction="mid",linewidth=0.7)+
  geom_vline(xintercept=OLD,linetype="dashed",colour=INK,linewidth=0.4)+
  geom_vline(xintercept=NEW,linetype="dashed",colour=COMP,linewidth=0.5)+
  annotate("text",x=OLD-0.15,y=Inf,label="old 4.0",hjust=1,vjust=1.4,size=3,colour=INK)+
  annotate("text",x=NEW+0.15,y=Inf,label="FDR min 7.0",hjust=0,vjust=1.4,size=3,colour=COMP)+
  scale_colour_manual(values=c("all 9-mers"=NUL,"peak upstream"=CAN,"dinucleotide shuffle"=COMP))+
  labs(title="U1 hybridization score distribution",
       subtitle="score = negated duplex free energy, higher = stronger",
       x="U1 score (-dG)", y="density")+th

F<-read.delim("u1_threshold.tsv",comment.char="#")
F<-F[!is.na(F$fdr_dinuc),]
Fl<-F%>%select(threshold,fdr_dinuc,fdr_uniform)%>%
  pivot_longer(c(fdr_dinuc,fdr_uniform),names_to="null",values_to="fdr")
Fl$null<-factor(Fl$null,levels=c("fdr_dinuc","fdr_uniform"),
                labels=c("dinucleotide shuffle","uniform 9-mer"))

pB<-ggplot(Fl,aes(threshold,fdr,colour=null))+
  geom_hline(yintercept=c(0.05,0.10),linetype="dotted",colour=INK,linewidth=0.4)+
  annotate("text",x=9.9,y=0.10,label="0.10",hjust=1,vjust=-0.4,size=2.8,colour=INK)+
  annotate("text",x=9.9,y=0.05,label="0.05",hjust=1,vjust=-0.4,size=2.8,colour=INK)+
  geom_line(linewidth=0.8)+geom_point(size=1.1)+
  geom_vline(xintercept=OLD,linetype="dashed",colour=INK,linewidth=0.4)+
  geom_vline(xintercept=NEW,linetype="dashed",colour=COMP,linewidth=0.5)+
  scale_colour_manual(values=c("dinucleotide shuffle"=CAN,"uniform 9-mer"=NUL))+
  scale_y_continuous(limits=c(0,1.15),breaks=seq(0,1,0.25))+
  labs(title="No threshold removes the composition confound",
       subtitle="FDR = null hit rate over observed hit rate; minimum 0.84 at score 7.0",
       x="U1 score threshold (-dG)", y="FDR (fraction of hits from composition)")+th

fig<-pA|pB
ggsave("u1_threshold_calibration.png",fig,width=13,height=5.2,dpi=170,bg="white")
cat("[out] u1_threshold_calibration.png\n")
