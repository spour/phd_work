#!/usr/bin/env Rscript
# the tie-it-together panel
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork)})
args<-commandArgs(trailingOnly=TRUE)
INDIR<-if(length(args)>=1) args[1] else "."
PFX  <-if(length(args)>=2) args[2] else "fig3"
need<-function(p){ if(!file.exists(p)) stop("missing required input: ",p,call.=FALSE); p }

INK<-"#1A1A1A"; BASE<-13; CAN<-"#0072B2"; COMP<-"#E69F00"; NUL<-"#8A8F98"
th<-theme_classic(base_size=BASE)+
  theme(text=element_text(color=INK),axis.text=element_text(color=INK,size=BASE-4),
        axis.line=element_line(color=INK,linewidth=.4),
        plot.title=element_text(size=BASE,face="bold",hjust=0),
        plot.subtitle=element_text(size=BASE-5,color="#555555"),
        legend.title=element_text(size=BASE-7,lineheight=1.05),
        legend.text=element_text(size=BASE-5),
        legend.key.height=unit(.8,"lines"))
FILLNAME<-"enriched for\nreal 3' ends (blue)\n<-   ->\ncompetitors (orange)"
E_MIN<-as.integer(Sys.getenv("E_MIN","200"))

# maxent table
AXI<-read_tsv(need(file.path(INDIR,paste0(PFX,"_acc_x_interv.tsv"))),show_col_types=FALSE)
b<-median((AXI$p_canonical+1e-3)/2^AXI$log2_enrich-1e-3,na.rm=TRUE)
D<-AXI%>%filter(n>=E_MIN)%>%
   mutate(p=pmin(pmax(p_canonical,1e-4),1-1e-4),
          f=log2((p/(1-p))/(b/(1-b))))
LIMB<-max(abs(D$f),na.rm=TRUE)
aBlue  <-D%>%slice_max(f,n=1,with_ties=FALSE)
aOrange<-D%>%slice_min(f,n=1,with_ties=FALSE)

# spliceai table
SAD<-read_tsv(need(file.path(INDIR,"sa_acc_x_don_final.tsv")),show_col_types=FALSE)
G<-SAD%>%filter(n>=E_MIN)
if(!"log2_or" %in% names(G))
  G<-G%>%mutate(p=pmin(pmax(p_canonical,1e-4),1-1e-4),
                log2_or=log2((p/(1-p))/(b/(1-b))))
G<-G%>%mutate(f=log2_or)
LIMC<-max(abs(G$f),na.rm=TRUE)
# anchors are restricted to the high-acceptor band, which is the region the panel
# is about
HI<-max(G$acc_bin)-2
Ghi<-G%>%filter(acc_bin>=HI)
cBlue<-Ghi%>%slice_max(f,n=1,with_ties=FALSE)
cGone<-Ghi%>%slice_min(f,n=1,with_ties=FALSE)

SAT<-read_tsv(need(file.path(INDIR,"sa_tertile_final.tsv")),show_col_types=FALSE)
tstrong<-SAT$auroc_test[SAT$acc_tertile=="strong" & SAT$donor_reduction=="sum"][1]

message(sprintf("[base rate] b = %.5f  (%.1f%%)",b,100*b))
message(sprintf("[B MaxEnt  ] %d cells kept, fill limit +/-%.3f",nrow(D),LIMB))
message(sprintf("[B anchor blue  ] acceptor=%.2f interv=%.2f  log2OR=%+.2f  n=%d",
                aBlue$acceptor,aBlue$interv,aBlue$f,aBlue$n))
message(sprintf("[B anchor orange] acceptor=%.2f interv=%.2f  log2OR=%+.2f  n=%d",
                aOrange$acceptor,aOrange$interv,aOrange$f,aOrange$n))
message(sprintf("[C SpliceAI] %d cells kept, fill limit +/-%.3f",nrow(G),LIMC))
message(sprintf("[C anchor blue  ] acc_decile=%d don_decile=%d  log2OR=%+.2f  n=%d",
                cBlue$acc_bin,cBlue$don_bin,cBlue$f,cBlue$n))
message(sprintf("[C anchor gone  ] acc_decile=%d don_decile=%d  log2OR=%+.2f  n=%d",
                cGone$acc_bin,cGone$don_bin,cGone$f,cGone$n))
message(sprintf("[C high-acceptor band acc_bin>=%d] min log2OR = %+.4f  (negative = competitor)",
                HI,min(Ghi$f)))
message(sprintf("[tertile] strong-acceptor donor_sum AUROC = %.4f",tstrong))

sq<-function(xmin,xmax,ymin,ymax,fill) annotate("rect",xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,
                                                fill=fill,color=INK,linewidth=.3)
ARR<-arrow(length=unit(0.17,"cm"),type="closed")
pA<-ggplot()+
  annotate("text",x=1.3,y=8.75,label="REAL 3' END",hjust=0,size=3.5,fontface="bold",color=CAN)+
  sq(3,6.4,6.7,7.7,CAN)+
  annotate("segment",x=1.3,xend=2.85,y=7.2,yend=7.2,color=INK,linewidth=.7,arrow=ARR)+
  annotate("text",x=2.05,y=7.95,label="strong 3'ss",size=3.1,color=INK)+
  annotate("segment",x=6.4,xend=6.4,y=7.7,yend=8.4,color=INK,linewidth=.5)+
  annotate("point",x=6.4,y=8.4,size=2.6,color=INK)+
  annotate("text",x=6.4,y=8.85,label="poly-A",size=3.1,color=INK)+
  annotate("text",x=1.3,y=5.95,hjust=0,size=3.1,color="#555555",
           label="strong 3'ss, no downstream 5'ss  ->  transcript ends here")+
  annotate("text",x=1.3,y=4.75,label="COMPETITOR",hjust=0,size=3.5,fontface="bold",color=COMP)+
  sq(3,6.4,2.5,3.5,COMP)+
  sq(8.7,10.4,2.5,3.5,COMP)+
  annotate("segment",x=1.3,xend=2.85,y=3.0,yend=3.0,color=INK,linewidth=.7,arrow=ARR)+
  annotate("text",x=2.05,y=3.75,label="strong 3'ss",size=3.1,color=INK)+
  annotate("segment",x=5.6,xend=5.6,y=3.5,yend=4.1,color=INK,linewidth=.5)+
  annotate("point",x=5.6,y=4.1,size=2.6,color=INK)+
  annotate("text",x=5.6,y=4.45,label="pA",size=3.1,color=INK)+
  annotate("segment",x=6.4,xend=7.25,y=3.0,yend=3.0,color=INK,linewidth=.7,arrow=ARR)+
  annotate("text",x=6.85,y=2.3,label="5'ss",size=3.1,color=INK)+
  annotate("curve",x=6.5,xend=8.6,y=3.55,yend=3.55,curvature=-0.55,
           color=INK,linewidth=.5,linetype=2)+
  annotate("text",x=7.55,y=4.9,label="splice",size=2.9,color="#555555")+
  annotate("text",x=1.3,y=1.75,hjust=0,size=3.1,color="#555555",lineheight=1.15,
           label="strong 3'ss AND a downstream 5'ss  ->  the site loses to\nan internal exon that splices through")+
  coord_cartesian(xlim=c(0.9,10.7),ylim=c(1.1,9.4),clip="off")+
  labs(title="A   The splicing logic")+
  theme_void(base_size=BASE)+
  theme(plot.title=element_text(size=BASE,face="bold",hjust=0,color=INK),
        plot.margin=margin(6,10,6,10))

pB<-ggplot(D,aes(interv,acceptor,fill=f))+
  geom_tile(color="grey92",linewidth=.1)+
  scale_fill_gradient2(low=COMP,mid="white",high=CAN,midpoint=0,
                       limits=c(-LIMB,LIMB),name=FILLNAME)+
  scale_x_continuous(expand=c(0,0))+scale_y_continuous(expand=c(0,0))+
  annotate("label",x=aBlue$interv,y=aBlue$acceptor,
           label="REAL 3' END\nstrong 3'ss, weak 5'ss",
           size=2.9,color=CAN,fill="white",lineheight=1.1,fontface="bold")+
  annotate("label",x=aOrange$interv,y=aOrange$acceptor,hjust=.9,
           label="COMPETITOR\nstrong 5'ss, weaker 3'ss",
           size=2.9,color=COMP,fill="white",lineheight=1.1,fontface="bold")+
  labs(title="B   MaxEnt: the advantage is erased",
       subtitle=sprintf("fill = log2 odds ratio of P(real end) vs the %.1f%% base rate; cells >=%d sites.",
                        100*b,E_MIN),
       x="intervening 5'ss (MaxEnt)",y="acceptor 3'ss (MaxEnt)")+th+
  theme(legend.position="right")

pC<-ggplot(G,aes(don_bin,acc_bin,fill=f))+
  geom_tile(color="grey92",linewidth=.1)+
  scale_fill_gradient2(low=COMP,mid="white",high=CAN,midpoint=0,
                       limits=c(-LIMC,LIMC),name=FILLNAME)+
  scale_x_continuous(expand=c(0,0),breaks=1:10)+
  scale_y_continuous(expand=c(0,0),breaks=1:10)+
  annotate("label",x=cBlue$don_bin,y=cBlue$acc_bin,hjust=.15,
           label="REAL 3' END\nstrong 3'ss",
           size=2.9,color=CAN,fill="white",lineheight=1.1,fontface="bold")+
  annotate("label",x=cGone$don_bin,y=cGone$acc_bin,hjust=.9,
           label="ADVANTAGE GONE\nstrong 3'ss + strong 5'ss",
           size=2.9,color=COMP,fill="white",lineheight=1.1,fontface="bold")+
  labs(title="C   SpliceAI: donor ranks against the real end",
       subtitle=sprintf("learned scores, held-out; fill = log2 odds ratio of P(real end) vs the %.1f%% base rate; cells >=%d peaks.\ndonor AUROC within the strong-acceptor tertile = %.3f (competitor direction).",
                        100*b,E_MIN,tstrong),
       x="SpliceAI donor decile (sum over 2 kb)",
       y="SpliceAI acceptor decile (max over 2 kb)")+th+
  theme(legend.position="right")

cap<-sprintf(paste0("MaxEnt and SpliceAI use different score scales; the shared feature is the pattern, not the magnitude. ",
        "Within the strong-acceptor band the donor ranks against the real end (AUROC %.3f), while those cells stay enriched for real ends: ",
        "the acceptor advantage falls to the base rate, it does not invert."),tstrong)

fig<-(pA|pB|pC)+plot_layout(widths=c(0.9,1.1,1.1))+
  plot_annotation(
    title="A strong 3'ss defines the real 3' end; a downstream 5'ss cancels the advantage",
    subtitle="The acceptor 3'ss sets the true end. A donor 5'ss after it means the site sits in an internal exon: both scorers show the acceptor advantage falling to the base rate, and SpliceAI ranks the donor against the real end. n = 6,583 real ends / 64,140 competitors.",
    caption=cap,
    theme=theme(plot.title=element_text(face="bold",size=BASE+2,hjust=0,color=INK),
                plot.subtitle=element_text(size=BASE-3,color="#555555"),
                plot.caption=element_text(size=BASE-5,color="#666666",hjust=0)))
ggsave(file.path(INDIR,"figure_summary.pdf"),fig,width=17,height=6,device=cairo_pdf)
ggsave(file.path(INDIR,"figure_summary.png"),fig,width=17,height=6,dpi=300)
cat(sprintf("[fill] independent limits: B +/-%.3f, C +/-%.3f\n",LIMB,LIMC))
cat("[out] figure_summary.pdf / .png\n")
