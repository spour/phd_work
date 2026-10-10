#!/usr/bin/env Rscript
# the locked splice comparator: where it sits, and what it learnt
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork)})
e<-function(...) cat(...,"\n",file=stderr())
rd<-function(f){ if(!file.exists(f)) stop("missing: ",f); read_tsv(f,show_col_types=FALSE) }
INK<-"#1A1A1A"; BASE<-8
REAL<-"#08519C"; COMP<-"#E8A33D"; GREY<-"#8A8F98"; GTX<-"#5A6068"
M<-rd("splice_model_final_metrics.tsv"); CF<-rd("splice_model_final_coefficients.tsv")
gv<-function(mod,met,col="value") M[[col]][M$model==mod & M$metric==met][1]
th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-1),axis.line=element_line(color=INK,linewidth=.3),
  axis.title=element_text(size=BASE),
  plot.title=element_text(size=BASE+1,face="bold",hjust=0,margin=margin(b=4)),
  plot.subtitle=element_text(size=BASE-1,color=GTX,margin=margin(b=6),lineheight=1.25),
  legend.title=element_blank(),legend.text=element_text(size=BASE-1),
  legend.key.height=unit(.7,"lines"),legend.key.width=unit(.9,"lines"),
  legend.margin=margin(0,0,0,0),legend.box.spacing=unit(2,"pt"))

FINAL<-"binned U1 + acceptor (51)"
LAD<-c("acceptor alone (1)","scalar splice model (3)",FINAL,
       "full binned + acceptor (126)","sequence CNN")
SHOW<-c("acceptor alone (1)"="MaxEnt 3' splice site alone",
        "scalar splice model (3)"="Scalar splice features",
        "binned U1 + acceptor (51)"="Positional U1 density + acceptor",
        "full binned + acceptor (126)"="All binned channels + acceptor",
        "sequence CNN"="Sequence CNN")
NF<-c(1,3,51,126,NA)

# a: performance ladder
panel_A<-function(){
  D<-bind_rows(lapply(c("winrate","p_at_1"),function(m)
      data.frame(model=LAD,metric=m,
                 v=sapply(LAD,gv,met=m),
                 lo=sapply(LAD,gv,met=m,col="ci_lo"),
                 hi=sapply(LAD,gv,met=m,col="ci_hi"))))%>%
    mutate(lab=factor(SHOW[model],levels=rev(SHOW[LAD])),
           metric=factor(metric,levels=c("winrate","p_at_1"),
                         labels=c("within-gene win-rate","P@1")),
           final=model==FINAL)
  ACCW<-gv("acceptor alone (1)","winrate")
  P1C<-gv("reference: P@1 chance level","p_at_1")
  stopifnot(is.finite(P1C))
  LV<-levels(D$metric)
  RL<-data.frame(metric=factor(c(LV[1],LV[1],LV[2]),levels=LV),
                 x=c(0.5,ACCW,P1C),
                 kind=c("chance","acceptor","chance"),
                 lty=c(2,3,2))
  e(sprintf("[A] ladder win-rates: %s",paste(sprintf("%s %.4f",SHOW[LAD],
      sapply(LAD,gv,met="winrate")),collapse="   ")))
  e(sprintf("[A] reference lines: win-rate chance 0.500 and acceptor %.4f; P@1 chance %.4f",ACCW,P1C))
  ggplot(D,aes(v,lab,color=final))+
    geom_vline(data=RL,aes(xintercept=x),linetype=RL$lty,
               color=ifelse(RL$kind=="chance",INK,GREY),linewidth=.32,inherit.aes=FALSE)+
    geom_errorbar(aes(xmin=lo,xmax=hi),orientation="y",width=.22,linewidth=.4)+
    geom_point(size=1.7)+
    geom_text(aes(x=hi,label=sprintf("%.3f",v)),hjust=-0.28,size=2.3,show.legend=FALSE)+
    facet_wrap(~metric,scales="free_x")+
    scale_color_manual(values=c(`TRUE`=REAL,`FALSE`=INK),guide="none")+
    scale_x_continuous(expand=expansion(mult=c(.06,.16)))+
    geom_text(data=RL,aes(x=x,label=kind),y=0.42,size=2.1,hjust=-0.12,vjust=0,
              color=ifelse(RL$kind=="chance",INK,GREY),inherit.aes=FALSE)+
    labs(title="A   Where hand-built splice biology sits",
         subtitle=paste0("Held-out test genes, bootstrap 95% intervals over genes; the locked ",
           "comparator is in blue.\nChance differs between the metrics: 0.500 for the win-rate, ",
           sprintf("%.3f",P1C)," for P@1, where a gene with k competitors\nscored at random ",
           "gives 1/(1 + k)."),
         x=NULL,y=NULL)+
    th+theme(panel.grid.major.y=element_line(color="#EEF0F3",linewidth=.25),
             strip.background=element_blank(),
             strip.text=element_text(size=BASE-0.5,color=INK,face="bold"),
             axis.text.y=element_text(size=BASE-0.5))
}

# b: coefficient profile by position
panel_B<-function(){
  P<-CF%>%filter(channel %in% c("u1_dens4","u1_dens7"))%>%
    mutate(channel=factor(channel,levels=c("u1_dens4","u1_dens7"),
             labels=c("U1 sites, -dG >= 4","U1 sites, -dG >= 7")))
  A<-CF$coefficient[which(CF$channel=="acceptor")][1]
  stopifnot(is.finite(A))
  rng<-range(c(P$coefficient,A,0)); pad<-diff(rng)*0.13
  e(sprintf("[B] acceptor coefficient %+.4f; U1 coefficients span %+.4f to %+.4f",
            A,min(P$coefficient),max(P$coefficient)))
  neg<-P%>%filter(bin_centre>=-600,bin_centre<=100)%>%group_by(channel)%>%
       summarise(m=mean(coefficient),.groups="drop")
  e(sprintf("[B] mean U1 coefficient over [-600,+100]: %s",
            paste(sprintf("%s %+.4f",neg$channel,neg$m),collapse="   ")))
  ggplot(P,aes(bin_centre,coefficient,color=channel))+
    annotate("rect",xmin=-Inf,xmax=0,ymin=-Inf,ymax=Inf,fill=GREY,alpha=.05)+
    geom_hline(yintercept=0,color=INK,linewidth=.3)+
    geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.3)+
    geom_line(linewidth=.5)+geom_point(size=1.0)+
    annotate("segment",x=-2000,xend=-1650,y=A,yend=A,color=REAL,linewidth=.5)+
    annotate("text",x=-1620,y=A,hjust=0,vjust=0.5,size=2.3,color=REAL,
             label=sprintf("MaxEnt 3' splice site (scalar), %+.2f",A))+
    scale_color_manual(values=c("U1 sites, -dG >= 4"=COMP,"U1 sites, -dG >= 7"="#8C6D1F"))+
    scale_x_continuous(breaks=seq(-2000,500,500))+
    scale_y_continuous(limits=c(rng[1]-pad,rng[2]+pad),expand=expansion(mult=c(0,0)))+
    labs(title="B   What the model weights, by position",
         subtitle=paste0("Ridge coefficients on gene-centred, standardised features, so ",
           "magnitudes are comparable across bins and channels."),
         x="position relative to poly(A) site (nt)",y="coefficient")+
    th+theme(legend.position=c(.985,.06),legend.justification=c(1,0),
             legend.background=element_rect(fill="white",color=NA),
             panel.grid.major.y=element_line(color="#EEF0F3",linewidth=.25))
}

SC<-gv("scalar splice model (3)","winrate"); FI<-gv(FINAL,"winrate")
FU<-gv("full binned + acceptor (126)","winrate"); CN<-gv("sequence CNN","winrate")
SURV<-M$value[M$model=="dinuc-shuffle: fraction of lift surviving"][1]
PD<-M[M$model=="paired: final minus scalar",]
CAP<-paste(strwrap(paste0(
 "Positional resolution, not extra channels, is what the model buys: the same splice biology as ",
 "three scalars reaches ",sprintf("%.3f",SC)," within gene, and reading U1 density at 100 nt ",
 "resolution over the same window reaches ",sprintf("%.3f",FI)," (paired difference ",
 sprintf("%+.4f",PD$value[1])," [",sprintf("%+.4f",PD$ci_lo[1]),", ",sprintf("%+.4f",PD$ci_hi[1]),
 "]). Adding the MaxEnt 5' splice-site channels back takes it to ",sprintf("%.3f",FU),
 ", and the sequence CNN reaches ",sprintf("%.3f",CN),
 " on the same genes. This is base composition placed in space, not a motif model: ",
 sprintf("%.0f%%",100*SURV)," of the binned model's lift over chance survives a dinucleotide ",
 "shuffle that preserves mononucleotide and dinucleotide frequency exactly. Window [-2000, +500] ",
 "throughout, 25 bins of 100 nt, evaluated once on held-out test genes."),width=128),collapse="\n")

fig<-(panel_A()/panel_B())+plot_layout(heights=c(1,1.12))+
  plot_annotation(
    title="An acceptor score and positional U1 composition are the ceiling of hand-built splice biology",
    subtitle="Locked comparator: U1 site density at two thresholds across 25 bins of 100 nt, plus the upstream MaxEnt 3' splice-site score",
    caption=CAP,
    theme=theme(plot.title=element_text(size=BASE+2.5,face="bold",color=INK),
                plot.subtitle=element_text(size=BASE,color=GTX),
                plot.caption=element_text(size=BASE-2,color=GTX,hjust=0,lineheight=1.3)))

ggsave("figure_splice_model.pdf",fig,width=8.6,height=8.0,device=grDevices::cairo_pdf)
ggsave("figure_splice_model.png",fig,width=8.6,height=8.0,dpi=300)
e("[out] figure_splice_model.pdf figure_splice_model.png")
