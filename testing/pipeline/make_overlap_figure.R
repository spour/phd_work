#!/usr/bin/env Rscript
# a: upset-style plot of gene sets (rf/lr/cnn flagging internal>canonical) b
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr);library(readr);library(tidyr)})
A <- if(length(commandArgs(TRUE))>=1) commandArgs(TRUE)[1] else "."
rd<-function(f){p<-file.path(A,f); if(file.exists(p)) suppressMessages(read_tsv(p,show_col_types=FALSE)) else NULL}
RF<-"#0072B2"; LR<-"#E69F00"; CNN<-"#009E73"; NAVY<-"#103A5C"; RED<-"#B0453C"; GRY<-"#C9CCD1"; INK<-"#1A1A1A"; B<-14
th<-theme_classic(base_size=B)+theme(text=element_text(color=INK),
  plot.title=element_text(face="bold",size=B+1),plot.subtitle=element_text(size=B-4,color="grey35"),
  legend.position="top",legend.title=element_blank(),axis.text=element_text(color=INK,size=B-3))

# a: upset (top bar = intersection size
vn<-rd("model_overlap_venn.tsv")
pA<-ggplot()+theme_void()
if(!is.null(vn)){
  memb<-list("RF only"=c("RF"),"LR only"=c("LR"),"CNN only"=c("CNN"),
             "RF&LR"=c("RF","LR"),"RF&CNN"=c("RF","CNN"),"LR&CNN"=c("LR","CNN"),
             "all three"=c("RF","LR","CNN"))
  vn<-vn[order(-vn$n),]
  vn$region<-factor(vn$region,levels=vn$region)
  models<-c("RF","LR","CNN")
  mat<-do.call(rbind,lapply(seq_len(nrow(vn)),function(i){
    r<-as.character(vn$region[i]); data.frame(region=r,model=models,
      on=models %in% memb[[r]], x=i, stringsAsFactors=FALSE)}))
  mat$region<-factor(mat$region,levels=levels(vn$region))
  mat$model<-factor(mat$model,levels=rev(models))
  vn$x<-seq_len(nrow(vn))
  pTop<-ggplot(vn,aes(x,n))+geom_col(fill=NAVY,width=.7)+
    geom_text(aes(label=format(n,big.mark=",")),vjust=-.3,size=3.1)+
    scale_y_continuous(expand=expansion(c(0,.12)))+
    scale_x_continuous(limits=c(.5,nrow(vn)+.5),expand=c(0,0))+
    labs(title="A  Model overlap (internal site scored > canonical)",y="genes",x=NULL)+
    th+theme(axis.text.x=element_blank(),axis.ticks.x=element_blank(),axis.line.x=element_blank())
  con<-mat[mat$on,] %>% group_by(x) %>% summarise(ymin=min(as.integer(model)),ymax=max(as.integer(model)),.groups="drop")
  pBot<-ggplot(mat,aes(x,model))+
    geom_point(aes(color=on),size=4)+
    geom_segment(data=con,aes(x=x,xend=x,y=ymin,yend=ymax),inherit.aes=FALSE,color=INK,linewidth=.7)+
    scale_color_manual(values=c(`TRUE`=INK,`FALSE`=GRY),guide="none")+
    scale_x_continuous(limits=c(.5,nrow(vn)+.5),expand=c(0,0))+
    labs(x=NULL,y=NULL)+th+
    theme(axis.text.x=element_blank(),axis.ticks.x=element_blank(),axis.line=element_blank())
  pA<-wrap_elements(pTop/pBot+plot_layout(heights=c(3,1)))
}

# b: agreement
ag<-rd("model_overlap_agreement.tsv")
pB<- if(!is.null(ag)){
  ag$category<-factor(ag$category,levels=c("all 3 same site","2 of 3 same","all different"))
  ggplot(ag,aes(category,frac,fill=category))+geom_col(width=.62)+
    geom_text(aes(label=sprintf("%.0f%%\n(n=%s)",frac*100,format(n,big.mark=","))),vjust=-.3,size=3.2,lineheight=.9)+
    scale_fill_manual(values=c("all 3 same site"=CNN,"2 of 3 same"=LR,"all different"=RED),guide="none")+
    scale_y_continuous(limits=c(0,max(ag$frac)*1.25),labels=scales::percent,expand=expansion(c(0,0)))+
    labs(title="B  Same internal site?",subtitle="genes contested in all 3; same = within 250nt",
         x=NULL,y="share of genes")+th+theme(axis.text.x=element_text(angle=15,hjust=1))
} else ggplot()+theme_void()

# c: cdf of pairwise distance (fraction within x), 250nt ref
pdd<-rd("model_overlap_pairdist.tsv")
pC<- if(!is.null(pdd)){
  pdd$pair<-factor(pdd$pair,levels=c("RF-LR","RF-CNN","LR-CNN"))
  pdd$dist<-pmax(pdd$dist,1)
  cdf<-pdd %>% group_by(pair) %>% arrange(dist) %>%
    mutate(frac=row_number()/n()) %>% ungroup()
  w250<-pdd %>% group_by(pair) %>% summarise(w=mean(dist<=250),.groups="drop")
  sub<-sprintf("within 250nt: RF-LR %.0f%%, RF-CNN %.0f%%, LR-CNN %.0f%%",
    100*w250$w[w250$pair=="RF-LR"],100*w250$w[w250$pair=="RF-CNN"],100*w250$w[w250$pair=="LR-CNN"])
  ggplot(cdf,aes(dist,frac,color=pair))+
    geom_vline(xintercept=250,linetype=2,color=RED)+
    annotate("text",x=250,y=.97,label="250nt",hjust=-.1,size=3.2,color=RED)+
    geom_step(linewidth=1)+
    scale_color_manual(values=c("RF-LR"=RF,"RF-CNN"=CNN,"LR-CNN"=LR))+
    scale_x_continuous(trans="log10",labels=scales::label_number(scale_cut=scales::cut_short_scale()))+
    scale_y_continuous(labels=scales::percent,limits=c(0,1))+
    labs(title="C  Distance between models' internal picks",subtitle=sub,
         x="pairwise distance (nt, log)",y="cumulative fraction of genes")+th
} else ggplot()+theme_void()

fig<-pA/(pB|pC)+plot_layout(heights=c(1.2,1))
ggsave(file.path(A,"figure_overlap.pdf"),fig,width=11,height=11,device=cairo_pdf)
ggsave(file.path(A,"figure_overlap.png"),fig,width=11,height=11,dpi=300)
cat("[out] figure_overlap.pdf/png\n")