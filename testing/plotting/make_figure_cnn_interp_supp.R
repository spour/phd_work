#!/usr/bin/env Rscript
# the two methods-defence panels for the cnn interpretability figure
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork)})
e<-function(...) cat(...,"\n",file=stderr())
INK<-"#1A1A1A"; BASE<-7; BLUE<-"#08519C"; GREY<-"#8A8F98"; GTX<-"#5A6068"; FAINT<-"#C6CACF"

OC<-read_tsv("figCI_occlusion.tsv",show_col_types=FALSE)
RK<-read_tsv("figCI_rank.tsv",show_col_types=FALSE)
NT<-read_tsv("cnnmain_notes.tsv",col_types=cols(.default=col_character()))
nv<-function(k) NT$value[NT$key==k][1]
RF<-as.integer(nv("A:receptive_field"))

th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-1),axis.line=element_line(color=INK,linewidth=.3),
  axis.title=element_text(size=BASE),axis.ticks=element_line(color=INK,linewidth=.3),
  plot.title=element_text(size=BASE+1,face="bold",hjust=0,margin=margin(b=4)),
  legend.position="none")
XBRK<-seq(-2000,500,500)

AD<-OC%>%filter(adequate==1)%>%mutate(w=factor(width,levels=c(50,100,200,400)))
NA_<-OC%>%filter(adequate==0)
SH<-scales::seq_gradient_pal("#7FB0DC",BLUE)(seq(0,1,length.out=4))
pS1<-ggplot()+
  geom_hline(yintercept=0,linetype=2,color=GREY,linewidth=.35)+
  geom_line(data=NA_,aes(centre,z),color=FAINT,linewidth=.4)+
  geom_line(data=AD,aes(centre,z,color=w,group=w),linewidth=.5)+
  scale_color_manual(values=setNames(SH,c("50","100","200","400")))+
  annotate("text",x=-2020,y=9.6,hjust=0,size=2.1,color=GTX,
           label="25 nt shown faint: below receptive-field resolution")+
  annotate("text",x=490,y=c(10.2,7.7,7.0,4.3),hjust=1,size=2.1,color=SH[c(4,3,2,1)],
           label=c("400 nt","200 nt","100 nt","50 nt"))+
  scale_x_continuous(breaks=XBRK)+coord_cartesian(xlim=c(-2050,500))+
  labs(title="S1   Positional importance across occlusion scales",
       x="Position relative to the cleavage site (nt)",
       y="Positional importance (z vs shuffled-window null)")+th

LV<-c("rank 1","rank 2-3","rank 4-10","rank >10")
RK2<-RK%>%mutate(stratum=factor(stratum,levels=LV))
TOT<-RK2%>%group_by(stratum)%>%summarise(t=first(total_signal),n=first(n_genes),.groups="drop")
RCOL<-setNames(c(BLUE,"#4A7CB8","#8FAFD4","#C9A227"),LV)
pS2<-ggplot(RK2,aes(centre,shift,color=stratum,group=stratum))+
  geom_hline(yintercept=0,linetype=2,color=GREY,linewidth=.35)+
  geom_line(linewidth=.5)+scale_color_manual(values=RCOL)+
  annotate("text",x=490,y=c(-0.22,-0.28,-0.34,-0.40),hjust=1,size=2.1,color=RCOL,
           label=sprintf("%s  (n = %d, total signal %.2f)",LV,TOT$n,TOT$t))+
  scale_x_continuous(breaks=XBRK)+coord_cartesian(xlim=c(-2050,500))+
  labs(title="S2   Positional importance by ranking outcome",
       x="Position relative to the cleavage site (nt)",
       y="Score shift of the real end relative to its competitors")+th

CAP<-paste0(
 "Reported model (cnn_model.pt), ",format(as.integer(nv("n_test")),big.mark=",")," held-out ",
 "peaks in ",format(as.integer(nv("n_genes")),big.mark=",")," genes. Both panels come from the ",
 "same occlusion scan as the main figure.\n",
 "S1: one output bin sees ",RF," nt, so a mask much narrower than that cannot move the score. ",
 "The ",nv("A:adequate_scales")," nt scales, the ones that register a drop, peak within ",
 nv("A:peak_spread_adequate")," nt of each other, ",
 sprintf("%.2f",as.numeric(nv("A:peak_spread_adequate"))/RF)," of one receptive field; the 25 nt ",
 "probe is 7% of that field, reaches only z = 2.18 and is drawn faint and excluded.\n",
 "S2: genes are stratified by where the model ranks the real 3' end, so a per-stratum win-rate ",
 "would sit on a ceiling or a floor. The quantity plotted is the shift of the real end's score ",
 "relative to its own competitors when a 100 nt window is masked, which no bound constrains.\n",
 "The model uses ",sprintf("%.0f",as.numeric(nv("B:rank_1_total"))/as.numeric(nv("B:rank_>10_total"))),
 "x more positional signal on the genes it ranks first than on the genes it ranks past 10 (",
 nv("B:rank_1_total")," against ",nv("B:rank_>10_total"),"), and what little it uses on those is ",
 "displaced downstream: the two profiles are anticorrelated at r = ",
 nv("B:shape_r_rank1_vs_rankgt10"),".")

FIG<-pS1/pS2+plot_layout(heights=c(1,1))+
  plot_annotation(
    title="Supplementary: robustness of the positional readout",
    caption=CAP,
    theme=theme(plot.title=element_text(size=BASE+2,face="bold",color=INK,hjust=0,
                                        margin=margin(b=6)),
                plot.caption=element_text(size=BASE-2,color=GTX,hjust=0,lineheight=1.3,
                                          margin=margin(t=8))))
ggsave("figure_cnn_interp_supp.pdf",FIG,width=8.6,height=7.4,device=grDevices::cairo_pdf)
ggsave("figure_cnn_interp_supp.png",FIG,width=8.6,height=7.4,dpi=260)
stopifnot(is.null(pS1$labels$subtitle),is.null(pS2$labels$subtitle))
e("[S1] adequate scales",nv("A:adequate_scales"),"peak spread",nv("A:peak_spread_adequate"),
  "nt vs receptive field",RF,"nt; 25 nt faint and excluded")
e("[S2] totals:",paste(sprintf("%s %.2f",TOT$stratum,TOT$t),collapse="  "),
  "| rank1 vs rank>10 r",nv("B:shape_r_rank1_vs_rankgt10"))
e("[out] figure_cnn_interp_supp.pdf figure_cnn_interp_supp.png")
