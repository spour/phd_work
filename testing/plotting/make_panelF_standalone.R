#!/usr/bin/env Rscript
# panel f of make_figure3_v3_2.r as its own figure
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr)})
INK<-"#1A1A1A"; BASE<-13; CAN<-"#0072B2"; COMP<-"#E69F00"; DEC<-"#6A51A3"; NUL<-"#8A8F98"
th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-4),axis.line=element_line(color=INK,linewidth=.4),
  plot.title=element_text(size=BASE,face="bold",hjust=0),plot.subtitle=element_text(size=BASE-5,color="#555555"),
  plot.caption=element_text(size=BASE-6,color="#555555",hjust=0),
  legend.title=element_text(size=BASE-7),legend.text=element_text(size=BASE-5),legend.key.height=unit(.8,"lines"))

FMG<-read_tsv("v32_F_metagene_analysisset.tsv",show_col_types=FALSE)
n<-FMG%>%group_by(group)%>%summarise(n=max(n),.groups="drop")
gn<-function(g) format(n$n[n$group==g],big.mark=",")
pa<-FMG%>%filter(offset==-50)
v<-function(g) sprintf("%.2f",pa$donor_per_kb[pa$group==g])

LAB<-c(sprintf("real 3' end (n = %s)",gn("canonical")),
       sprintf("competitor (n = %s)",gn("competitor")),
       sprintf("intronic decoy (n = %s)",gn("decoy")))
FMG$group<-factor(FMG$group,levels=c("canonical","competitor","decoy"),labels=LAB)
COLS<-setNames(c(CAN,COMP,DEC),LAB)

pF<-ggplot(FMG,aes(offset,donor_per_kb,color=group,fill=group))+
  annotate("rect",xmin=-Inf,xmax=0,ymin=-Inf,ymax=Inf,fill=NUL,alpha=.05)+
  geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.4)+
  geom_ribbon(aes(ymin=donor_per_kb-se,ymax=donor_per_kb+se),alpha=.15,color=NA)+
  geom_line(linewidth=.9)+
  scale_color_manual(values=COLS,name=NULL)+scale_fill_manual(values=COLS,guide="none")+
  scale_x_continuous(breaks=seq(-2000,2000,500))+
  annotate("text",x=-1950,y=Inf,hjust=0,vjust=1.6,label="upstream of pA (the exon)",size=2.9,color="#555555")+
  annotate("text",x=1950,y=Inf,hjust=1,vjust=1.6,label="downstream of pA",size=2.9,color="#555555")+
  labs(title="Strong 5'ss density around the pA",
       subtitle=paste0("At the pA the three groups order real end ",v("canonical")," < competitor ",v("competitor"),
         " < decoy ",v("decoy")," sites/kb (bin -50 nt): real ends\nare 5'ss-depleted just upstream of the pA (the terminal exon has no donor), competitors less so, decoys\nleast. The three converge downstream of the pA. The separation is upstream-only."),
       x="position relative to pA (nt)",y="strong 5'ss sites / kb (MaxEnt >=8)",
       caption=paste0("Competitors are the analysis set of cpa_peaks_final.parquet: baseline-model false positives after removing APA-annotated\npeaks, reference terminal exons and peaks at or downstream of the canonical pA. ",
         gn("competitor")," competitors against the ",gn("canonical"),
         " real 3' ends of\nthe genes that keep one (4,847 genes), plus ",gn("decoy"),
         " intronic decoys subsampled from 134,969 in those genes (seed 42, 4,189 genes).\n40 bins of 100 nt over +/-2 kb; ribbons are 1 SE. Total plotted: ",
         format(sum(n$n),big.mark=","),"."))+
  th+theme(legend.position="bottom")

ggsave("figure3_panelF.pdf",pF,width=7.2,height=5.0,device=grDevices::cairo_pdf)
ggsave("figure3_panelF.png",pF,width=7.2,height=5.0,dpi=300)
cat("pA bin (-50 nt) sites/kb: real",v("canonical"),"competitor",v("competitor"),"decoy",v("decoy"),"\n")
cat("[out] figure3_panelF.pdf figure3_panelF.png\n")
