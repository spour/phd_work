#!/usr/bin/env Rscript
# sequence determinants learned by the reported cnn
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(tidyr)
                               library(patchwork);library(ggseqlogo)})
e<-function(...) cat(...,"\n",file=stderr())
INK<-"#1A1A1A"; BASE<-7; BLUE<-"#08519C"; GREY<-"#8A8F98"; GTX<-"#5A6068"; FAINT<-"#C6CACF"

OC<-read_tsv("figCI_occlusion.tsv",show_col_types=FALSE)
RK<-read_tsv("figCI_rank.tsv",show_col_types=FALSE)
GA<-read_tsv("figCI_gia.tsv",show_col_types=FALSE)
FL<-read_tsv("figCI_filters.tsv",show_col_types=FALSE)
LG<-read_tsv("figCI_logos.tsv",show_col_types=FALSE)
NT<-read_tsv("figCI_notes.tsv",col_types=cols(.default=col_character()))
nv<-function(k) NT$value[NT$key==k][1]
RF<-as.integer(as.numeric(nv("A:receptive_field")))
NGN<-OC$n_genes[1]; NPK<-OC$n_peaks[1]

th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-1),axis.line=element_line(color=INK,linewidth=.3),
  axis.title=element_text(size=BASE),axis.ticks=element_line(color=INK,linewidth=.3),
  plot.title=element_text(size=BASE+1,face="bold",hjust=0,margin=margin(b=4)),
  legend.position="none")
XBRK<-seq(-2000,500,500)

AD<-OC%>%filter(adequate==1)%>%mutate(w=factor(width,levels=c(50,100,200,400)))
NA_<-OC%>%filter(adequate==0)
SH<-scales::seq_gradient_pal("#7FB0DC",BLUE)(seq(0,1,length.out=4))
pA<-ggplot()+
  geom_hline(yintercept=0,linetype=2,color=GREY,linewidth=.35)+
  geom_line(data=NA_,aes(centre,z),color=FAINT,linewidth=.4)+
  geom_line(data=AD,aes(centre,z,color=w,group=w),linewidth=.5)+
  scale_color_manual(values=setNames(SH,c("50","100","200","400")))+
  annotate("text",x=-2020,y=9.6,hjust=0,size=2.0,color=GTX,
           label="25 nt shown faint: below receptive-field resolution")+
  annotate("text",x=490,y=c(10.2,7.7,7.0,4.3),hjust=1,size=2.0,color=SH[c(4,3,2,1)],
           label=c("400 nt","200 nt","100 nt","50 nt"))+
  scale_x_continuous(breaks=XBRK)+coord_cartesian(xlim=c(-2050,500))+
  labs(title="A   Positional importance by occlusion across perturbation scales",
       x="Position relative to the cleavage site (nt)",
       y="Positional importance (z vs shuffled-window null)")+th

LV<-c("rank 1","rank 2-3","rank 4-10","rank >10")
RK2<-RK%>%mutate(stratum=factor(stratum,levels=LV))
TOT<-RK2%>%group_by(stratum)%>%summarise(t=first(total_signal),n=first(n_genes),.groups="drop")
stopifnot(abs(TOT$t[TOT$stratum=="rank 1"]-9.3632)<1e-3,
          abs(TOT$t[TOT$stratum=="rank >10"]-0.1339)<1e-3)
RCOL<-setNames(c(BLUE,"#4A7CB8","#8FAFD4","#C9A227"),LV)
pB<-ggplot(RK2,aes(centre,shift,color=stratum,group=stratum))+
  geom_hline(yintercept=0,linetype=2,color=GREY,linewidth=.35)+
  geom_line(linewidth=.5)+
  scale_color_manual(values=RCOL)+
  annotate("text",x=490,y=c(-0.22,-0.28,-0.34,-0.40),hjust=1,size=2.0,color=RCOL,
           label=sprintf("%s  (n = %d, total %.2f)",LV,TOT$n,TOT$t))+
  scale_x_continuous(breaks=XBRK)+coord_cartesian(xlim=c(-2050,500))+
  labs(title="B   Positional importance stratified by ranking outcome",
       x="Position relative to the cleavage site (nt)",
       y="Score shift of the real end relative to its competitors")+th

ORD<-c("3' splice site (pyrimidine tract + AG)","5' splice site (AGGTAAGT)",
       "5' splice site, scrambled","Polyadenylation signal (AATAAA)",
       "Polyadenylation signal, scrambled")
GA2<-GA%>%mutate(label=factor(label,levels=rev(ORD)),
                 bg_label=factor(bg_label,levels=c("Dinucleotide-shuffled background",
                                                   "Real competitor background")),
                 fill=ifelse(kind=="control",FAINT,BLUE))
pC<-ggplot(GA2,aes(label,delta_logit,fill=I(fill)))+
  geom_hline(yintercept=0,linetype=2,color=GREY,linewidth=.35)+
  geom_col(width=.6)+
  geom_errorbar(aes(ymin=delta_logit-1.96*se,ymax=delta_logit+1.96*se),width=.15,
                linewidth=.3,color=INK)+
  facet_wrap(~bg_label,nrow=1)+coord_flip()+
  scale_y_continuous(limits=c(-0.09,0.36),breaks=seq(0,0.3,.1))+
  labs(title="C   Effect of motif implantation on the model score",
       x=NULL,y="Change in model score at the most favourable position")+
  th+theme(strip.background=element_blank(),
           strip.text=element_text(size=BASE-1,color=GTX,hjust=0,margin=margin(b=2)),
           panel.spacing.x=unit(8,"pt"))

FL2<-FL%>%mutate(matched=best_match!="none",
                 lab=ifelse(filter %in% c("filter27","filter12"),filter,NA_character_))
pD1<-ggplot(FL2,aes(ic_per_col,n_sites))+
  geom_point(aes(color=I(ifelse(matched,BLUE,FAINT))),size=1.5,alpha=.9)+
  geom_text(aes(label=ifelse(is.na(lab),"",sub("filter","filter ",lab))),
            size=2.0,color=INK,hjust=1.15,vjust=.4)+
  scale_y_log10(breaks=c(50,500,5000,25000),labels=c("50","500","5,000","25,000"))+
  scale_x_continuous(limits=c(.52,.86),breaks=seq(.55,.85,.1))+
  annotate("text",x=.52,y=27000,hjust=0,size=2.0,color=GTX,
           label="filled = matches a reference motif")+
  labs(title="D   First-layer filter information content",
       x="Information content (bits per column, maximum 2.0)",y="Activating sites")+th

mk<-function(f){
  m<-LG%>%filter(filter==f)%>%select(position,base,prob)%>%
    pivot_wider(names_from=position,values_from=prob)%>%as.data.frame()
  rn<-m$base; m<-as.matrix(m[,-1]); rownames(m)<-rn; m[c("A","C","G","T"),,drop=FALSE]
}
lab27<-with(FL[FL$filter=="filter27",],sprintf("filter 27   %s, r = %.2f",best_match,r))
lab12<-with(FL[FL$filter=="filter12",],sprintf("filter 12   %s, r = %.2f",best_match,r))
lo<-function(f,ti) ggseqlogo(mk(f),method="bits")+
  labs(title=ti,x=NULL,y="bits")+
  theme_classic(base_size=BASE)+
  theme(text=element_text(color=INK),legend.position="none",
        plot.title=element_text(size=BASE-1,color=GTX,hjust=0,margin=margin(b=1)),
        axis.text.x=element_blank(),axis.ticks.x=element_blank(),
        axis.text.y=element_text(color=INK,size=BASE-2),
        axis.line=element_line(color=INK,linewidth=.3),
        axis.title.y=element_text(size=BASE-1))
pD2<-lo("filter27",lab27)/lo("filter12",lab12)

# compose
CAP<-paste0(
 "Reported model (cnn_model_def.Net, cnn_model.pt), evaluated on ",format(NPK,big.mark=","),
 " held-out peaks in ",format(NGN,big.mark=",")," genes.\n",
 "Panel A: one output bin has an effective receptive field of ",RF," nt, so the important region ",
 "is broad rather than 100 nt wide. The ",nv("A:adequate_scales")," nt scales peak within ",
 nv("A:peak_spread_adequate")," nt of each other, ",
 sprintf("%.2f",as.numeric(nv("A:peak_spread_adequate"))/RF)," of one receptive field. The 25 nt ",
 "probe is 7% of that field, reaches only z = 2.18, and its apparent peak is noise; it is drawn ",
 "faint and excluded from the agreement statement.\n",
 "Panel B: strata are defined by the model's own ranking, so a per-stratum win-rate would sit on ",
 "a ceiling or a floor. The quantity plotted is the shift of the real end's score relative to its ",
 "own competitors when a 100 nt window is masked, which no bound constrains. Summed signal is ",
 paste(sprintf("%.2f",TOT$t),collapse=", ")," for ",paste(LV,collapse=", "),
 ".\nThe rank 1 and rank >10 profiles correlate r = ",nv("B:shape_r_rank1_vs_rankgt10"),
 ", so the stratum the model ranks worst weights different positions rather than fewer.\n",
 "Panel C: each motif is implanted at 60 positions and reported at the position most favourable ",
 "to it, so a null result is not an artifact of placement. Bars are 1.96 standard errors over ",
 GA$n_bg[1]," backgrounds.\nThe implantation set carries permutation controls for the 5' splice ",
 "site and the polyadenylation signal but not for the 3' splice site, so the acceptor bar has no ",
 "paired scramble.\n",
 "Panel D: all 32 filters lie between ",sprintf("%.2f",as.numeric(nv("D:ic_min")))," and ",
 sprintf("%.2f",as.numeric(nv("D:ic_max")))," bits per column against a maximum of 2.0, so ",
 "information content alone does not separate them; ",nv("D:n_matched")," match a reference ",
 "motif.\nUsage is shown because it, together with information content, is what distinguishes ",
 "the polypyrimidine filter, which is both the highest in information content and the most used.\n",
 "Integrated gradients agree with occlusion on location (r = ",nv("IG:excess_vs_occlusion_z_r"),
 " against the occlusion profile) but their magnitude sits at the composition null (ratio ",
 nv("IG:null_ratio"),"), so occlusion is the primary method here.")

FIG<-(pA|pB)/(pC|(pD1|pD2)+plot_layout(widths=c(1,0.62)))+
  plot_layout(heights=c(1,0.9))+
  plot_annotation(title="Sequence determinants learned by the CNN",caption=CAP,
    theme=theme(plot.title=element_text(size=BASE+2,face="bold",color=INK,hjust=0,
                                        margin=margin(b=6)),
                plot.caption=element_text(size=BASE-2.2,color=GTX,hjust=0,lineheight=1.3,
                                          margin=margin(t=8))))
ggsave("figure_cnn_interp.pdf",FIG,width=11.4,height=8.0,device=grDevices::cairo_pdf)
ggsave("figure_cnn_interp.png",FIG,width=11.4,height=8.0,dpi=320)

stopifnot(is.null(pA$labels$subtitle),is.null(pB$labels$subtitle),
          is.null(pC$labels$subtitle),is.null(pD1$labels$subtitle))
BAD<-grep("honest|exactly|push back|—",c(CAP,pA$labels$title,pB$labels$title,pC$labels$title,
                                          pD1$labels$title),value=TRUE)
stopifnot(length(BAD)==0)
e("[A] adequate scales:",nv("A:adequate_scales"),"| 25 nt drawn faint, excluded from the verdict")
e("[A] peak spread adequate",nv("A:peak_spread_adequate"),"nt vs all-five",nv("A:peak_spread_all"),
  "nt; receptive field",RF,"nt")
e("[B] totals:",paste(sprintf("%s %.2f",TOT$stratum,TOT$t),collapse="  "),
  "| ratio",sprintf("%.0fx",TOT$t[1]/TOT$t[4]))
e("[C] acceptor",sprintf("%+.3f / %+.3f",GA$delta_logit[GA$motif=="acceptor_pyAG"][1],
                         GA$delta_logit[GA$motif=="acceptor_pyAG"][2]),
  "| donor",sprintf("%+.3f / %+.3f",GA$delta_logit[GA$motif=="donor_AGGTAAGT"][1],
                    GA$delta_logit[GA$motif=="donor_AGGTAAGT"][2]),
  "| PAS",sprintf("%+.3f / %+.3f",GA$delta_logit[GA$motif=="PAS_AATAAA"][1],
                  GA$delta_logit[GA$motif=="PAS_AATAAA"][2]))
e("[D] IC per column",sprintf("%.3f-%.3f",min(FL$ic_per_col),max(FL$ic_per_col)),
  "| matched",sum(FL$best_match!="none"),"/32 | logos filter27, filter12")
e("[check] no panel subtitles; two logos; caption carries the receptive-field and IG notes")
e("[out] figure_cnn_interp.pdf figure_cnn_interp.png")
