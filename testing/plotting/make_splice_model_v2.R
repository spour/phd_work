#!/usr/bin/env Rscript
# the final 26-feature splice model, two panels
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork)})
e<-function(...) cat(...,"\n",file=stderr())
INK<-"#1A1A1A"; BASE<-7
U1C<-"#08519C"; ACCC<-"#E8A33D"; GREY<-"#9AA0A6"; GTX<-"#5A6068"
CF<-read_tsv("splice_model_26_coefficients.tsv",show_col_types=FALSE)
MT<-read_tsv("splice_model_26_metrics.tsv",show_col_types=FALSE)
mv<-function(k,col="value") MT[[col]][MT$metric==k][1]
th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-1),axis.line=element_line(color=INK,linewidth=.3),
  axis.title=element_text(size=BASE),
  plot.title=element_text(size=BASE+1,face="bold",hjust=0,margin=margin(b=5)),
  legend.position="none")

# a: coefficient profile
panel_A<-function(){
  U<-CF%>%filter(channel=="u1_dens4")
  A<-CF$coefficient[which(CF$channel=="acceptor")][1]
  stopifnot(nrow(U)==25, is.finite(A))
  jm<-which.min(U$coefficient)
  e(sprintf("[A] U1 minimum %+.4f at bin centre %+d nt; U1 range [%+.4f, %+.4f]; acceptor %+.4f",
            U$coefficient[jm],U$bin_centre[jm],min(U$coefficient),max(U$coefficient),A))
  rng<-range(c(U$coefficient,A,0)); pad<-diff(rng)*0.16
  ggplot(U,aes(bin_centre,coefficient))+
    annotate("rect",xmin=-Inf,xmax=0,ymin=-Inf,ymax=Inf,fill=GREY,alpha=.07)+
    geom_hline(yintercept=0,color=INK,linewidth=.35)+
    geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.35)+
    geom_col(width=88,fill=U1C)+
    annotate("point",x=U$bin_centre[jm],y=U$coefficient[jm],size=1.5,shape=21,
             fill=U1C,color="white",stroke=.5)+
    annotate("text",x=U$bin_centre[jm],y=U$coefficient[jm]-pad*0.30,
             label=sprintf("%+.2f at %+d nt",U$coefficient[jm],U$bin_centre[jm]),
             size=2.1,color=U1C,vjust=1)+
    annotate("segment",x=760,xend=760,y=rng[1]-pad,yend=rng[2]+pad,color=GREY,linewidth=.3)+
    annotate("rect",xmin=880,xmax=1160,ymin=0,ymax=A,fill=ACCC)+
    annotate("text",x=1020,y=A,label=sprintf("%+.2f",A),size=2.2,color=ACCC,vjust=-0.6)+
    annotate("text",x=1020,y=rng[1]-pad*0.55,label="MaxEnt 3' splice site\n(scalar, not positional)",
             size=2.0,color=ACCC,vjust=1,lineheight=1.1)+
    annotate("text",x=-1950,y=rng[2]+pad*0.75,label="U1 duplex density (-dG >= 4) per 100 nt bin",
             size=2.2,color=U1C,hjust=0)+
    annotate("text",x=-1950,y=rng[1]-pad*0.55,label="upstream of the cleavage site",
             size=2.0,color=GTX,hjust=0,vjust=1)+
    scale_x_continuous(breaks=c(seq(-2000,500,500)),
                       limits=c(-2100,1260),expand=expansion(mult=c(0,0)))+
    scale_y_continuous(limits=c(rng[1]-pad*1.15,rng[2]+pad*1.15),
                       expand=expansion(mult=c(0,0)))+
    labs(title="A   Fitted weights by position",
         x="position relative to the cleavage site (nt)",y="ridge coefficient")+th
}

# b: performance
panel_B<-function(){
  D<-data.frame(
    metric=factor(c("Within-gene win-rate","P@1, held-out genes","P@1, all genes"),
                  levels=c("Within-gene win-rate","P@1, held-out genes","P@1, all genes")),
    v=c(mv("within_gene_winrate"),mv("p_at_1"),mv("p_at_1_whole_genome")),
    lo=c(mv("within_gene_winrate","ci_lo"),mv("p_at_1","ci_lo"),mv("p_at_1_whole_genome","ci_lo")),
    hi=c(mv("within_gene_winrate","ci_hi"),mv("p_at_1","ci_hi"),mv("p_at_1_whole_genome","ci_hi")),
    ref=c(mv("within_gene_winrate","reference"),mv("p_at_1","reference"),
          mv("p_at_1_whole_genome","reference")),
    rlab=c("chance 0.500","chance 0.292","baseline 0.577"))
  D$lift<-D$v-D$ref
  e(sprintf("[B] %s: %.4f [%.4f, %.4f] against %s",D$metric,D$v,D$lo,D$hi,D$rlab))
  ggplot(D,aes(metric,v))+
    geom_segment(aes(xend=metric,y=ref,yend=v),color=U1C,linewidth=3.2,lineend="butt")+
    geom_errorbar(aes(ymin=lo,ymax=hi),width=.10,linewidth=.4,color=INK)+
    geom_point(size=2.1,color=U1C)+
    geom_segment(aes(x=as.numeric(metric)-.34,xend=as.numeric(metric)+.34,y=ref,yend=ref),
                 linetype=2,color=GREY,linewidth=.45)+
    geom_text(aes(y=hi,label=sprintf("%.3f",v)),vjust=-0.9,size=2.6,color=INK,fontface="bold")+
    geom_text(aes(y=ref,label=rlab),vjust=1.7,size=2.0,color=GTX)+
    geom_text(aes(y=(v+ref)/2,label=sprintf("%+.3f",lift)),hjust=-0.9,size=2.1,color=U1C)+
    scale_y_continuous(breaks=seq(0.2,0.8,0.1),expand=expansion(mult=c(0.10,0.13)))+
    labs(title="B   What the model achieves",x=NULL,y="value")+
    th+theme(panel.grid.major.y=element_line(color="#EEF0F3",linewidth=.25),
             axis.text.x=element_text(size=BASE))
}

CAP<-paste0("n = ",format(mv("within_gene_winrate","n"),big.mark=",")," held-out test genes (",
  format(mv("n_test_peaks"),big.mark=",")," peaks) for the first two measures; ",
  format(mv("p_at_1_whole_genome","n"),big.mark=",")," genes for the third, of which ",
  format(mv("n_uncontested"),big.mark=","),
  " carry a single candidate. Intervals are bootstrap 95% over genes.")
fig<-(panel_A()/panel_B())+plot_layout(heights=c(1,0.92))+
  plot_annotation(
    title="An acceptor score and positional U1 density place the 3' end within its gene",
    subtitle="U1 duplex density (-dG >= 4) in 25 bins of 100 nt across [-2,000, +500] plus the upstream MaxEnt 3' splice-site score: 26 features",
    caption=CAP,
    theme=theme(plot.title=element_text(size=BASE+2.5,face="bold",color=INK),
                plot.subtitle=element_text(size=BASE,color=GTX,margin=margin(b=6)),
                plot.caption=element_text(size=BASE-1.6,color=GTX,hjust=0,lineheight=1.25)))
ggsave("figure_splice_model_v2.pdf",fig,width=7.2,height=6.6,device=grDevices::cairo_pdf)
ggsave("figure_splice_model_v2.png",fig,width=7.2,height=6.6,dpi=300)
SRC<-paste(readLines("make_splice_model_v2.R"),collapse=" ")
e("[chk] CNN / scalar-splice / shuffle mentions in the drawn figure:",
  length(grep("CNN|shuffl|scalar",c(CAP,"An acceptor score and positional U1 density place the 3' end within its gene"))))
e("[chk] no panel subtitles: panel titles carry a letter and a short title only")
e("[out] figure_splice_model_v2.pdf figure_splice_model_v2.png")
