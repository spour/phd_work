#!/usr/bin/env Rscript
# the complete standalone summary figure
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork)})
args<-commandArgs(trailingOnly=TRUE)
INDIR<-if(length(args)>=1) args[1] else "."
PFX  <-if(length(args)>=2) args[2] else "fig3"
need<-function(p){ if(!file.exists(p)) stop("missing required input: ",p,call.=FALSE); p }
rdp<-function(f) read_tsv(need(file.path(INDIR,f)),show_col_types=FALSE)

INK<-"#1A1A1A"; BASE<-13; CAN<-"#0072B2"; COMP<-"#E69F00"; NUL<-"#8A8F98"
SCOL<-c(MaxEnt=INK,SpliceAI=NUL)
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

HM<-rdp("heat_maxent.tsv"); HS<-rdp("heat_spliceai.tsv"); ST<-rdp("stats_forest.tsv")
b<-HM$base_rate[1]

D <-HM%>%filter(n>=E_MIN)%>%mutate(f=log2_or)
G <-HS%>%filter(n>=E_MIN)%>%mutate(f=log2_or)
LIMB<-max(abs(D$f),na.rm=TRUE); LIMC<-max(abs(G$f),na.rm=TRUE)
aBlue  <-D%>%slice_max(f,n=1,with_ties=FALSE)
aOrange<-D%>%slice_min(f,n=1,with_ties=FALSE)
HI<-max(G$acc_bin)-2
Ghi<-G%>%filter(acc_bin>=HI)
cBlue<-Ghi%>%slice_max(f,n=1,with_ties=FALSE)
cGone<-Ghi%>%slice_min(f,n=1,with_ties=FALSE)

getv<-function(lab,sc,col="estimate") ST[[col]][ST$effect_label==lab & ST$scorer==sc][1]
message(sprintf("[base rate] %.5f",b))
message(sprintf("[B] %d cells, limit +/-%.3f | blue acc=%.2f iv=%.2f (%+.2f, n=%d) | orange acc=%.2f iv=%.2f (%+.2f, n=%d)",
        nrow(D),LIMB,aBlue$acceptor,aBlue$interv,aBlue$f,aBlue$n,
        aOrange$acceptor,aOrange$interv,aOrange$f,aOrange$n))
message(sprintf("[C] %d cells, limit +/-%.3f | blue acc=%d don=%d (%+.2f, n=%d) | gone acc=%d don=%d (%+.2f, n=%d)",
        nrow(G),LIMC,cBlue$acc_bin,cBlue$don_bin,cBlue$f,cBlue$n,
        cGone$acc_bin,cGone$don_bin,cGone$f,cGone$n))
message(sprintf("[C] high-acceptor band acc_bin>=%d: min log2OR %+.4f",HI,min(Ghi$f)))
for(i in seq_len(nrow(ST)))
  message(sprintf("[forest] %-8s %-28s %-9s %8.4f [%7.4f,%7.4f] %s",
          ST$block[i],ST$effect_label[i],ST$scorer[i],ST$estimate[i],
          ST$ci_lo[i],ST$ci_hi[i],ST$p_label[i]))

# a: schematic
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

# b and c: heatmaps, independent scales
pB<-ggplot(D,aes(interv,acceptor,fill=f))+
  geom_tile(color="grey92",linewidth=.1)+
  scale_fill_gradient2(low=COMP,mid="white",high=CAN,midpoint=0,limits=c(-LIMB,LIMB),name=FILLNAME)+
  scale_x_continuous(expand=c(0,0))+scale_y_continuous(expand=c(0,0))+
  annotate("label",x=aBlue$interv,y=aBlue$acceptor,label="REAL 3' END\nstrong 3'ss, weak 5'ss",
           size=2.9,color=CAN,fill="white",lineheight=1.1,fontface="bold")+
  annotate("label",x=aOrange$interv,y=aOrange$acceptor,hjust=.9,
           label="COMPETITOR\nstrong 5'ss, weaker 3'ss",
           size=2.9,color=COMP,fill="white",lineheight=1.1,fontface="bold")+
  labs(title="B   MaxEnt: the advantage is erased",
       subtitle=sprintf("fill = log2 odds ratio of P(real end) vs the %.1f%% base rate; cells >=%d sites.",100*b,E_MIN),
       x="intervening 5'ss (MaxEnt)",y="acceptor 3'ss (MaxEnt)")+th+
  theme(legend.position="right")

pC<-ggplot(G,aes(don_bin,acc_bin,fill=f))+
  geom_tile(color="grey92",linewidth=.1)+
  scale_fill_gradient2(low=COMP,mid="white",high=CAN,midpoint=0,limits=c(-LIMC,LIMC),name=FILLNAME)+
  scale_x_continuous(expand=c(0,0),breaks=1:10)+scale_y_continuous(expand=c(0,0),breaks=1:10)+
  annotate("label",x=cBlue$don_bin,y=cBlue$acc_bin,hjust=.15,label="REAL 3' END\nstrong 3'ss",
           size=2.9,color=CAN,fill="white",lineheight=1.1,fontface="bold")+
  annotate("label",x=cGone$don_bin,y=cGone$acc_bin,hjust=.9,
           label="ADVANTAGE GONE\nstrong 3'ss + strong 5'ss",
           size=2.9,color=COMP,fill="white",lineheight=1.1,fontface="bold")+
  labs(title="C   SpliceAI: the advantage is erased",
       subtitle=sprintf("learned scores, held-out; fill = log2 odds ratio of P(real end) vs the %.1f%% base rate; cells >=%d peaks.",100*b,E_MIN),
       x="SpliceAI donor decile (sum over 2 kb)",y="SpliceAI acceptor decile (max over 2 kb)")+th+
  theme(legend.position="right")

# d: forest, two sub-sections with their own null
EF<-ST%>%filter(block=="effect")%>%
    mutate(lab=paste0(effect_label,"   ",scorer))
EF$lab<-factor(EF$lab,levels=rev(EF$lab))
RK<-ST%>%filter(block=="ranking")%>%
    mutate(lab=paste0("acceptor 3'ss   ",scorer))
RK$lab<-factor(RK$lab,levels=rev(RK$lab))

ef_hi<-max(EF$ci_hi); ef_txt<-ef_hi*1.35
pD1<-ggplot(EF,aes(estimate,lab,color=scorer))+
  geom_vline(xintercept=1,linetype=3,color=NUL)+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),orientation="y",width=.18,linewidth=.6)+
  geom_point(size=2.4)+
  geom_text(aes(x=ef_txt,label=p_label),hjust=0,size=2.7,color="#555555")+
  scale_color_manual(values=SCOL,guide="none")+
  scale_x_log10(breaks=c(0.5,0.75,1,1.5,2,3))+
  coord_cartesian(xlim=c(min(EF$ci_lo)*0.85,ef_txt*2.4),clip="off")+
  labs(title="D   Effect sizes and robustness",
       subtitle=paste0("Odds ratios per SD (null 1); within-gene win-rate (null 0.5). OR<1 for donor|acceptor and\n",
                       "interaction means a stronger 5'ss lowers the odds of a real end at fixed 3'ss."),
       x="odds ratio per SD (log scale)",y=NULL)+th+
  theme(axis.text.y=element_text(size=BASE-5),plot.margin=margin(6,8,2,6))

rk_txt<-max(RK$ci_hi)+0.055
pD2<-ggplot(RK,aes(estimate,lab,color=scorer))+
  geom_vline(xintercept=0.5,linetype=3,color=NUL)+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),orientation="y",width=.14,linewidth=.6)+
  geom_point(size=2.4)+
  geom_text(aes(x=rk_txt,label=p_label),hjust=0,size=2.7,color="#555555")+
  scale_color_manual(values=SCOL,guide="none")+
  coord_cartesian(xlim=c(0.46,rk_txt+0.115),clip="off")+
  labs(subtitle="Within-gene win-rate (P real end outranks competitor)",
       x="win-rate",y=NULL)+th+
  theme(axis.text.y=element_text(size=BASE-5),plot.margin=margin(2,8,6,6))

pD<-pD1/pD2+plot_layout(heights=c(1.7,1))

# assemble
fmt<-function(x) sprintf("%.2f",x)
sub<-sprintf(paste0("Two scorers agree. Acceptor 3'ss raises the odds of a real end (OR %s per SD MaxEnt, %s SpliceAI); a stronger 5'ss lowers them at fixed acceptor\n",
             "(OR %s and %s per SD; interaction %s MaxEnt, %s SpliceAI). Within genes the real end outranks its competitors %.0f%% and %.0f%% of the time. ",
             "n = 6,583 real ends / 64,140 competitors."),
     fmt(getv("acceptor 3'ss","MaxEnt")),fmt(getv("acceptor 3'ss","SpliceAI")),
     fmt(getv("donor 5'ss | acceptor","MaxEnt")),fmt(getv("donor 5'ss | acceptor","SpliceAI")),
     getv("acceptor x donor","MaxEnt","p_label"),getv("acceptor x donor","SpliceAI","p_label"),
     100*getv("within-gene win-rate, acceptor","MaxEnt"),
     100*getv("within-gene win-rate, acceptor","SpliceAI"))

row1<-(pA|pB)+plot_layout(widths=c(0.9,1.15))
row2<-(pC|pD)+plot_layout(widths=c(1.15,1.0))
fig<-row1/row2+
  plot_annotation(
    title="A strong 3'ss defines the real 3' end; a downstream 5'ss competes it back toward baseline",
    subtitle=sub,
    caption=paste0("MaxEnt and SpliceAI use different score scales; the shared feature is the pattern, not the magnitude. ",
                   "Effect sizes are pooled logistic fits; win-rate is gene-fixed."),
    theme=theme(plot.title=element_text(face="bold",size=BASE+2,hjust=0,color=INK),
                plot.subtitle=element_text(size=BASE-3,color="#555555"),
                plot.caption=element_text(size=BASE-5,color="#666666",hjust=0)))
ggsave(file.path(INDIR,"figure_summary_full.pdf"),fig,width=17,height=11,device=cairo_pdf)
ggsave(file.path(INDIR,"figure_summary_full.png"),fig,width=17,height=11,dpi=300)
cat(sprintf("[fill] independent limits: B +/-%.3f, C +/-%.3f\n",LIMB,LIMC))
cat("[out] figure_summary_full.pdf / .png\n")
