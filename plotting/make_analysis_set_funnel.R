#!/usr/bin/env Rscript
# how cpa_peaks_final.parquet's in_analysis_set is built
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr)})
e<-function(...) cat(...,"\n",file=stderr())
INK<-"#1A1A1A"; BASE<-8
REAL<-"#08519C"; COMP<-"#E8A33D"; BAR<-"#59677A"; GREY<-"#9AA0A6"; GTX<-"#5A6068"
F<-read_tsv("analysis_set_funnel.tsv",show_col_types=FALSE)
stopifnot(nrow(F)==6, F$n_peaks==F$n_canonical+F$n_competitor)
lab<-function(x) format(as.numeric(x),big.mark=",",trim=TRUE)
TOT<-F$n_peaks[1]

STAGE<-c("All called peaks",
         "Annotated alternative 3' ends removed",
         "Reference terminal-exon peaks removed",
         "Genes without a surviving competitor dropped",
         "Peaks at or downstream of the canonical removed",
         "Analysis set")
CUT<-c("APA-annotated",
       "reference terminal exon",
       "canonical peaks (gene lost all competitors)",
       "at or downstream of the canonical",
       "canonical peaks (gene rule re-applied)")
F$k<-seq_len(nrow(F)); F$y<--F$k; F$stage<-STAGE
H<-0.30; PAD<-TOT*0.012

B<-bind_rows(
  F[1:5,]%>%transmute(y,x0=0,x1=n_peaks,seg="stage"),
  data.frame(y=F$y[6],x0=0,x1=F$n_canonical[6],seg="real"),
  data.frame(y=F$y[6],x0=F$n_canonical[6],x1=F$n_peaks[6],seg="competitor"))%>%
  mutate(seg=factor(seg,levels=c("stage","real","competitor")))
L<-data.frame(y=F$y[-1]+0.5, x0=F$n_peaks[-1], x1=F$n_peaks[-nrow(F)],
              n=F$n_removed[-1], cut=CUT)%>%
  mutate(txt=paste0("−",lab(n),"  ",cut))
stopifnot(all(L$x1-L$x0==L$n))

p<-ggplot()+
  geom_rect(data=L,aes(xmin=x0,xmax=x1,ymin=y-0.5+H,ymax=y+0.5-H),
            fill=GREY,alpha=.30,color=NA)+
  geom_rect(data=B,aes(xmin=x0,xmax=x1,ymin=y-H,ymax=y+H,fill=seg),
            color="white",linewidth=.3)+
  scale_fill_manual(values=c(stage=BAR,real=REAL,competitor=COMP),guide="none")+
  geom_text(data=F,aes(x=n_peaks+PAD,y=y,label=paste0(lab(n_peaks)," peaks, ",lab(n_genes)," genes")),
            hjust=0,vjust=0.5,size=2.5,color=INK)+
  geom_text(data=L,aes(x=x1+PAD,y=y,label=txt),hjust=0,vjust=0.5,size=2.4,color=GTX)+
  annotate("text",x=F$n_canonical[6]/2,y=F$y[6],hjust=.5,vjust=.5,size=2.4,color="white",
           fontface="bold",label=lab(F$n_canonical[6]))+
  annotate("text",x=F$n_canonical[6]+(F$n_competitor[6]-F$n_canonical[6])/2,y=F$y[6],
           hjust=.5,vjust=.5,size=2.4,color="white",fontface="bold",label=lab(F$n_competitor[6]))+
  annotate("text",x=F$n_canonical[6]/2,y=F$y[6]-H-0.13,hjust=.5,vjust=1,size=2.3,color=REAL,
           label="real 3' ends")+
  annotate("text",x=F$n_canonical[6]+(F$n_competitor[6]-F$n_canonical[6])/2,y=F$y[6]-H-0.13,
           hjust=.5,vjust=1,size=2.3,color="#A5751D",label="competitors")+
  scale_x_continuous(breaks=seq(0,70000,10000),
                     labels=function(x) format(x/1000,trim=TRUE),
                     expand=expansion(mult=c(.004,.24)))+
  scale_y_continuous(breaks=F$y,labels=F$stage,
                     limits=c(min(F$y)-0.95,max(F$y)+0.55),expand=expansion(mult=c(0,0)))+
  coord_cartesian(clip="off")+
  labs(title="Construction of the analysis set by successive filtering of candidate peaks",
       subtitle="Bar length is the number of peaks surviving that stage; the shaded notch between two bars is what the step removed",
       x="peaks (thousands)",y=NULL,
       caption=paste0(
"Filters act on the competitor pool, but two of the five steps remove canonical peaks rather than competitors: a gene that runs out of\n",
"competitors is dropped and its canonical peak leaves with it, so the canonical count falls from ",lab(F$n_canonical[1])," to ",
lab(F$n_canonical[4])," at step 4 and to ",lab(F$n_canonical[6])," at step 6.\n",
"A peak is APA-annotated when the nearest same-strand PolyASite 2.0 cluster or PolyA_DB v4 site lies within 500 nt; a peak is in a reference\n",
"terminal exon when its 500 nt upstream window overlaps a GENCODE v49 last exon by at least 1 nt. Counts are reconstructed from those\n",
"annotation caches by trace_analysis_set.py and asserted equal to the in_analysis_set column of cpa_peaks_final.parquet."))+
  theme_classic(base_size=BASE)+
  theme(text=element_text(color=INK),
        axis.text.x=element_text(color=INK,size=BASE-1),
        axis.text.y=element_text(color=INK,size=BASE-0.5,hjust=1),
        axis.ticks.y=element_blank(),axis.line.y=element_blank(),
        axis.line.x=element_line(color=INK,linewidth=.3),
        axis.title.x=element_text(size=BASE,margin=margin(t=4)),
        plot.title=element_text(size=BASE+2.5,face="bold",hjust=0),
        plot.subtitle=element_text(size=BASE-0.5,color=GTX,margin=margin(b=8)),
        plot.caption=element_text(size=BASE-2,color=GTX,hjust=0,lineheight=1.3),
        plot.title.position="plot", plot.caption.position="plot",
        plot.margin=margin(8,10,6,6),
        panel.grid.major.x=element_line(color="#ECEEF1",linewidth=.25))

ggsave("figure_analysis_set_funnel.pdf",p,width=9.0,height=4.5,device=grDevices::cairo_pdf)
ggsave("figure_analysis_set_funnel.png",p,width=9.0,height=4.5,dpi=300)

e("[design] vertical attrition funnel: one left-aligned bar per stage, length = peaks surviving,")
e("         shared 0..",lab(TOT)," count axis. Not a waterfall (no peeled slices) and not a Sankey (no ribbons).")
e("[stages]")
for(i in seq_len(nrow(F)))
  e(sprintf("   %d  %-48s %8s peaks  %6s genes",i,F$stage[i],lab(F$n_peaks[i]),lab(F$n_genes[i])))
e("[removed labels drawn in the gaps]")
for(i in seq_len(nrow(L))) e("  ",L$txt[i])
e("[final]  bar 6 is a single stacked bar split by class:",lab(F$n_canonical[6]),
  "real 3' ends (",REAL,") +",lab(F$n_competitor[6]),"competitors (",COMP,") =",lab(F$n_peaks[6]))
e("[text]   the two canonical-removing steps are stated in the caption, not drawn as tracks")
e("[out]    figure_analysis_set_funnel.pdf  figure_analysis_set_funnel.png")
