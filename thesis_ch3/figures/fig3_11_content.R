# figure 3.11, which properties of the upstream sequence the network reads
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr)})
setwd("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3")
INK<-"#1A1A1A"; GTX<-"#5A6068"; GREY<-"#8A8F98"; REAL<-"#08519C"; COMP<-"#E8A33D"
TH<-theme_classic(base_size=8.5)+theme(plot.title=element_text(face="bold",size=10.5,colour=INK),axis.text=element_text(colour=GTX),
      legend.key.size=unit(3,"mm"),legend.text=element_text(size=7.5))
FAM<-c("Alu","other SINE (MIR)","L1","L2 and other LINE","LTR","DNA transposon","simple repeat or low complexity")
CLS<-c("real 3′ end"=REAL,competitor=COMP)
A<-read.delim("data/figc_repeat_cov.tsv")%>%filter(family %in% FAM)%>%mutate(family=factor(family,levels=rev(FAM)),class=factor(class,levels=names(CLS)))
pA<-ggplot(A,aes(coverage,family,fill=class))+geom_col(position=position_dodge(0.75),width=0.7,colour=INK,linewidth=0.2)+
  scale_fill_manual(values=CLS,name=NULL)+scale_x_continuous(breaks=seq(0,0.14,0.04),labels=function(x) paste0(round(100*x),"%"),expand=c(0,0))+
  labs(x="fraction of upstream 2 kb",y=NULL,title="A   Repeat coverage upstream")+TH+theme(legend.position=c(0.75,0.2))
B<-read.delim("data/figc_repeat_site.tsv")%>%filter(family %in% FAM)%>%
  mutate(family=factor(family,levels=rev(FAM)),class=factor(class,levels=names(CLS)),orientation=factor(orientation,levels=c("antisense","sense")))
pB<-ggplot(B,aes(fraction,family,fill=orientation))+geom_col(width=0.7,colour=INK,linewidth=0.2)+facet_grid(.~class)+
  scale_fill_manual(values=c(sense="#6C7A89",antisense="#D5DAE0"),name="relative to gene")+
  scale_x_continuous(breaks=seq(0,0.12,0.04),labels=function(x) paste0(round(100*x),"%"),expand=c(0,0))+
  labs(x="fraction of sites within the element",y=NULL,title="B   Repeat elements at the site")+TH+
  theme(strip.background=element_blank(),strip.text=element_text(size=8.5,colour=INK),legend.position="bottom",axis.text.y=element_blank())
C<-read.delim("data/figc_surrogate.tsv",na.strings="NA")%>%
  mutate(model=recode(model,"composition (GC, T, A, CpG o/e, ATTTA, TTTTT)"="composition","period-3 coding signal"="period-3 signal",
                      "all sequence-derived"="all of these","annotation: CDS and UTR fraction (reference)"="annotated CDS and UTR"),
         kind=ifelse(model=="annotated CDS and UTR","annotation",ifelse(model=="sequence network","network","features")))%>%filter(model!="chance")
C$model<-factor(C$model,levels=c("composition","period-3 signal","repeat fraction","all of these","sequence network","annotated CDS and UTR"))
ch<-read.delim("data/figc_surrogate.tsv")%>%filter(model=="chance")%>%pull(p1)
pC<-ggplot(C,aes(model,p1,fill=kind))+geom_col(width=0.65,colour=INK,linewidth=0.2)+geom_errorbar(aes(ymin=lo,ymax=hi),width=0.15,linewidth=0.3)+
  geom_hline(yintercept=ch,linetype="22",colour=GREY)+
  scale_fill_manual(values=c(features="#C9D6E6",network=REAL,annotation="white"),guide="none")+
  scale_y_continuous(limits=c(0,1),expand=c(0,0))+scale_x_discrete(labels=function(x) gsub(" ","\n",x))+
  labs(x=NULL,y="precision at 1 (held-out genes)",title="C   Ranking by sequence properties")+TH+theme(axis.text.x=element_text(size=6.5,angle=35,hjust=1,lineheight=0.85))
Dd<-read.delim("data/figc_patch_kind.tsv")%>%mutate(transplanted=factor(transplanted,levels=c("CDS","3' UTR","repeat-free intron","intronic repeat"),
      labels=c("CDS","3′ UTR","repeat-free intron","intronic repeat")),replaced=factor(replaced,levels=c("repeat-free intronic","repeat"),
      labels=c("replacing repeat-free intron","replacing repeat")))
pD<-ggplot(Dd,aes(transplanted,gain,fill=replaced))+geom_hline(yintercept=0,colour=GREY,linewidth=0.3)+
  geom_col(position=position_dodge(0.75),width=0.7,colour=INK,linewidth=0.2)+
  geom_errorbar(aes(ymin=lo,ymax=hi),position=position_dodge(0.75),width=0.15,linewidth=0.3)+
  scale_fill_manual(values=c("replacing repeat-free intron"="#C9D6E6","replacing repeat"="#6C7A89"),name=NULL)+
  scale_x_discrete(labels=function(x) gsub(" ","\n",x))+
  labs(x="100 nt bin transplanted from the real 3′ end window",y="increase in competitor logit",title="D   Transplanting by sequence type")+
  TH+theme(legend.position=c(0.7,0.88),axis.text.x=element_text(size=7))
E<-read.delim("data/figc_kmer.tsv")
pE<-ggplot(E,aes(utr_enrichment,effect,colour=gc))+geom_point(size=0.5,alpha=0.6)+
  scale_colour_gradient(low="#F2C14E",high="#2B4C7E",name="G+C fraction")+
  labs(x="log2 enrichment in 3′ UTR over repeat-free intron",y="change in logit on implanting",title="E   6-mers")+TH+theme(legend.position=c(0.8,0.2))
fig<-(pA|pB)/(free(pC)|pD|pE)+plot_layout(heights=c(1,1.05))
ggsave("out/fig3_11_content.pdf",fig,width=10.2,height=7.4,device=cairo_pdf)
ggsave("out/fig3_11_content.png",fig,width=10.2,height=7.4,dpi=300,bg="white")
cat("[out] out/fig3_11_content.{pdf,png}\n")
