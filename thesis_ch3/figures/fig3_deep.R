# natural splice sites by strength, and sufficiency against necessity, in the network
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr);library(ggrepel)})
setwd("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3")
INK<-"#1A1A1A"; GTX<-"#5A6068"; CAN<-"#0072B2"; COMP<-"#E69F00"; ACC<-"#0072B2"; DON<-"#D55E00"; NUL<-"#B8BDC4"
th<-theme_classic(base_size=10)+theme(plot.title=element_text(face="bold",size=11,colour=INK),axis.text=element_text(colour=GTX),legend.title=element_blank(),legend.key.size=unit(4,"mm"))
S<-read.delim("data/deep_strength.tsv",check.names=FALSE)%>%filter(offset==-300)%>%
  mutate(site=factor(site,levels=c("3' splice site","5' splice site"),labels=c("Annotated 3' splice sites","Annotated 5' splice sites")),
         g=factor(score_group,levels=c("<6","6-8","8-10","10-12",">=10",">=12")))
pA<-ggplot(S,aes(g,diff,colour=site,group=site))+geom_hline(yintercept=0,colour=GTX,linewidth=0.3)+
  geom_errorbar(aes(ymin=diff_lo,ymax=diff_hi),width=0.12,linewidth=0.4)+geom_line(linewidth=0.9)+geom_point(size=2.4)+
  facet_wrap(~site,scales="free_x")+scale_colour_manual(values=c(ACC,DON),guide="none")+
  labs(x="MaxEntScan score of the site written in",y="change in score (logit)",title="A   Natural splice sites, by strength")+th+
  theme(strip.background=element_blank(),strip.text=element_text(colour=INK,size=9.5))
N<-read.delim("data/deep_necessity.tsv",check.names=FALSE)%>%filter(class=="competitor")
M<-read.delim("data/implant_mech.tsv",check.names=FALSE)%>%filter(offset==-200)
map<-c("UGUA"="CFIm25|UGUA","U run (6 or more)"="Fip1|U-rich upstream element","GAAGAA"="SRSF1|exonic splicing enhancer","CUCUCU"="PTB|CU tract","UAGGG"="hnRNP A1|UAGGG",
       "poly(A) signal"="CPSF|AAUAAA","strong 5' splice site (score5 >= 8)"="U1 snRNP|annotated 5' splice sites")
lab<-c("UGUA"="UGUA","U run (6 or more)"="U run","GAAGAA"="GAAGAA","CUCUCU"="CUCUCU","UAGGG"="UAGGG","poly(A) signal"="AAUAAA","strong 5' splice site (score5 >= 8)"="5' splice site")
B<-N%>%mutate(add=M$diff[match(map[element],M$insert)],add_lo=M$diff_lo[match(map[element],M$insert)],add_hi=M$diff_hi[match(map[element],M$insert)],l=lab[element],
              dir=ifelse(add>0,"raises score when added","lowers score when added"))
stopifnot(!any(is.na(B$add)))
pB<-ggplot(B,aes(add,diff,colour=dir))+geom_hline(yintercept=0,colour=GTX,linewidth=0.3)+geom_vline(xintercept=0,colour=GTX,linewidth=0.3)+
  geom_errorbar(aes(ymin=diff_lo,ymax=diff_hi),width=0.004,linewidth=0.35)+geom_errorbar(aes(xmin=add_lo,xmax=add_hi),orientation="y",width=0.008,linewidth=0.35)+
  geom_point(size=2.8)+geom_text_repel(aes(label=l),size=3,colour=INK,seed=1,box.padding=0.35,min.segment.length=1)+
  scale_colour_manual(values=c("raises score when added"=CAN,"lowers score when added"=COMP),guide="none")+
  labs(x="change when one copy is written in",y="change when native copies are removed",title="A   Adding against removing")+th
K<-read.delim("data/deep_ism_kmers.tsv",check.names=FALSE)%>%filter(class=="real 3' end")%>%arrange(mean_importance)%>%mutate(rank=row_number())
top<-rbind(head(K,10),tail(K,10))
K$grp<-ifelse(grepl("UGUA",K$kmer),"contains UGUA",ifelse(grepl("[CU]AG",substr(K$kmer,3,6))&K$mean_importance<0,"contains CAG or UAG","other"))
top$grp<-K$grp[match(top$kmer,K$kmer)]
COL<-c("contains UGUA"=CAN,"contains CAG or UAG"=COMP,"other"=NUL)
pC<-ggplot(K,aes(rank,mean_importance))+geom_hline(yintercept=0,colour=GTX,linewidth=0.3)+geom_point(aes(colour=grp),size=0.7)+
  geom_point(data=K%>%filter(grp!="other"),aes(colour=grp),size=1.1)+
  geom_text_repel(data=top,aes(label=kmer,colour=grp),size=2.6,max.overlaps=Inf,segment.size=0.2,segment.colour=GTX,seed=1,family="mono",
                  nudge_x=ifelse(top$mean_importance>0,-260,260),direction="y",hjust=ifelse(top$mean_importance>0,1,0),show.legend=FALSE)+
  scale_colour_manual(values=COL)+labs(x=paste0(nrow(K)," 6-mers, ranked"),y="importance in real 3' end windows",title="B   Saturation mutagenesis")+th+
  theme(axis.text.x=element_blank(),axis.ticks.x=element_blank(),legend.position="bottom")+guides(colour=guide_legend(override.aes=list(size=2.5)))
fig<-wrap_elements(full=pB)+wrap_elements(full=pC)+plot_layout(widths=c(1,1.1))
ggsave("out/fig3_deep.pdf",fig,width=7.8,height=4.6,device=cairo_pdf); ggsave("out/fig3_deep.png",fig,width=7.8,height=4.6,dpi=300,bg="white")
cat("[out] out/fig3_deep\n")
