#!/usr/bin/env Rscript
# the two candidate panel a designs, rendered standalone so they can be put side by side
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork)})
e<-function(...) cat(...,"\n",file=stderr())
INK<-"#1A1A1A"; BASE<-7
U1C<-"#08519C"; ACCC<-"#E8A33D"; GREY<-"#9AA0A6"; GTX<-"#5A6068"
CF<-read_tsv("acceptor_options_coefficients.tsv",show_col_types=FALSE)
PS<-read_tsv("acceptor_position.tsv",show_col_types=FALSE)
MT<-read_tsv("acceptor_options_metrics.tsv",show_col_types=FALSE)
M26<-"26  U1 x25 + acceptor scalar"; M50<-"50  U1 x25 + acceptor per bin"
th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-1),axis.line=element_line(color=INK,linewidth=.3),
  axis.title=element_text(size=BASE),
  plot.title=element_text(size=BASE+1,face="bold",hjust=0,margin=margin(b=5)),
  legend.position="none")

# option 1
opt1<-function(){
  U<-CF%>%filter(model==M26,!is.na(channel),channel=="u1_dens4")
  A<-CF$coefficient[which(CF$model==M26 & CF$channel=="acceptor_scalar")][1]
  stopifnot(nrow(U)==25,is.finite(A))
  jm<-which.min(U$coefficient); rng<-range(c(U$coefficient,A,0)); pad<-diff(rng)*0.16
  top<-ggplot(U,aes(bin_centre,coefficient))+
    annotate("rect",xmin=-Inf,xmax=0,ymin=-Inf,ymax=Inf,fill=GREY,alpha=.07)+
    geom_hline(yintercept=0,color=INK,linewidth=.35)+
    geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.35)+
    geom_col(width=88,fill=U1C)+
    annotate("text",x=U$bin_centre[jm],y=U$coefficient[jm],
             label=sprintf("%+.2f at %+d nt",U$coefficient[jm],U$bin_centre[jm]),
             size=2.1,color=U1C,vjust=1.5)+
    annotate("segment",x=760,xend=760,y=rng[1]-pad,yend=rng[2]+pad,color=GREY,linewidth=.3)+
    annotate("rect",xmin=880,xmax=1160,ymin=0,ymax=A,fill=ACCC)+
    annotate("text",x=1020,y=A,label=sprintf("%+.2f",A),size=2.2,color=ACCC,vjust=-0.6)+
    annotate("text",x=1020,y=rng[1]-pad*0.4,label="MaxEnt 3'ss\n(one scalar)",
             size=2.0,color=ACCC,vjust=1,lineheight=1.1)+
    annotate("text",x=-1950,y=rng[2]+pad*0.7,
             label="U1 duplex density (-dG >= 4) per 100 nt bin",size=2.2,color=U1C,hjust=0)+
    scale_x_continuous(breaks=seq(-2000,500,500),limits=c(-2100,1260),
                       expand=expansion(mult=c(0,0)))+
    scale_y_continuous(limits=c(rng[1]-pad*1.1,rng[2]+pad*1.1),expand=expansion(mult=c(0,0)))+
    labs(title="A   Fitted weights by position",x=NULL,y="ridge coefficient")+
    th+theme(axis.text.x=element_blank(),axis.ticks.x=element_blank(),
             axis.line.x=element_blank(),plot.margin=margin(4,4,0,4))
  D<-PS%>%mutate(class=factor(class,levels=c("real","competitor"),
                              labels=c("real 3' end","competitor")))
  bot<-ggplot(D,aes(bin_centre,fraction,color=class))+
    annotate("rect",xmin=-Inf,xmax=0,ymin=-Inf,ymax=Inf,fill=GREY,alpha=.07)+
    geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.35)+
    geom_step(linewidth=.5,direction="mid")+
    scale_color_manual(values=c(`real 3' end`=U1C,competitor=ACCC))+
    scale_x_continuous(breaks=seq(-2000,500,500),limits=c(-2100,1260),
                       expand=expansion(mult=c(0,0)))+
    scale_y_continuous(expand=expansion(mult=c(0,.16)))+
    annotate("text",x=-250,y=Inf,vjust=1.4,hjust=.5,size=2.0,color=U1C,label="real 3' end")+
    annotate("text",x=-1700,y=Inf,vjust=1.4,hjust=.5,size=2.0,color=ACCC,label="competitor")+
    labs(x="position relative to the cleavage site (nt)",
         y="where the winning\nacceptor sits")+
    th+theme(plot.margin=margin(0,4,4,4),
             axis.title.y=element_text(size=BASE-1.5,lineheight=1.05))
  top/bot+plot_layout(heights=c(2.5,1))
}

# option 2
opt2<-function(){
  H<-CF%>%filter(model==M50,!is.na(channel),channel %in% c("u1_dens4","me3_max"))%>%
    mutate(row=factor(channel,levels=c("u1_dens4","me3_max"),
                      labels=c("U1 duplex density (-dG >= 4)","MaxEnt 3' splice site (per bin)")))
  stopifnot(nrow(H)==50)
  lim<-max(abs(H$coefficient))
  d<-MT%>%filter(model==M50)
  e(sprintf("[opt2] heatmap drawn from the %d-feature model: within-gene %.4f, %+.4f vs the 26",
            50,d$winrate[1],d$delta_vs_26[1]))
  ggplot(H,aes(bin_centre,row,fill=coefficient))+
    geom_tile(width=96,height=.86,color="white",linewidth=.3)+
    geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.45)+
    scale_fill_gradient2(low="#B2182B",mid="white",high="#2166AC",midpoint=0,
                         limits=c(-lim,lim),name="ridge\ncoefficient")+
    scale_x_continuous(breaks=seq(-2000,500,500),expand=expansion(mult=c(.01,.01)))+
    labs(title="A   Fitted weights by position and channel",
         subtitle=sprintf(paste0("Drawn from a %d-feature model, %+.3f within gene against the ",
             "26-feature model: the 25 acceptor bins\nreplace the one scalar and cost ",
             "performance. Their weights are small enough that the shared scale\nleaves the ",
             "row nearly blank."),50,d$delta_vs_26[1]),
         x="position relative to the cleavage site (nt)",y=NULL)+
    th+theme(plot.subtitle=element_text(size=BASE-1.5,color=GTX,lineheight=1.2,
                                        margin=margin(b=4)),
             legend.position="right",legend.title=element_text(size=BASE-1.5,color=INK),
             legend.text=element_text(size=BASE-1.5),
             legend.key.width=unit(.45,"lines"),legend.key.height=unit(1.0,"lines"),
             axis.text.y=element_text(size=BASE-0.5))
}

p1<-opt1(); p2<-opt2()
ggsave("option1_panelA.pdf",p1,width=6.4,height=3.9,device=grDevices::cairo_pdf)
ggsave("option1_panelA.png",p1,width=6.4,height=3.9,dpi=300)
ggsave("option2_panelA.pdf",p2,width=6.4,height=3.0,device=grDevices::cairo_pdf)
ggsave("option2_panelA.png",p2,width=6.4,height=3.0,dpi=300)
e("[out] option1_panelA.pdf option1_panelA.png option2_panelA.pdf option2_panelA.png")
