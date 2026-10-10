# genome-wide cpa identification: every model against the baseline
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(patchwork)})
INK<-"#1A1A1A"; GTX<-"#5A6068"; GREY<-"#8A8F98"; GRID<-"#E7E9EC"; PAPER<-"#FCFCFD"
ACC<-"#2A6DB0"; HI<-"#C0504D"; BASE<-9
M<-read.delim("master_model_metrics.tsv",stringsAsFactors=FALSE)
NG<-M$n_genes[1]; NP<-M$n_peaks[1]; BR<-M$pooled_auprc[M$model=="Chance"]
CH_W<-0.5; CH_P<-M$p_at_1[M$model=="Chance"]
SHORT<-c("Chance"="chance",
  "Acceptor 3'ss alone"="acceptor 3'ss",
  "Window GC alone [-500, 0)"="window GC",
  "GC dip depth alone"="GC dip depth",
  "Scalar splice model"="splice, scalars",
  "Positional splice (locked)"="splice, positional",
  "Full binned splice"="splice, binned",
  "Positional splice + dip depth"="splice + dip depth",
  "Positional splice + GC per bin"="splice + GC per bin",
  "Boosted tree on the 51 splice features"="boosted tree, splice",
  "Sequence CNN"="sequence CNN")
M<-M%>%mutate(lab=SHORT[model],
              kind=ifelse(model=="Chance","baseline",
                   ifelse(ladder_step=="no","aside",
                   ifelse(model=="Sequence CNN","cnn","ladder"))),
              nf=ifelse(n_features=="-","",n_features),
              lab=ifelse(nf=="",lab,sprintf("%s  (%s)",lab,nf)))
M<-M%>%arrange(within_gene_winrate)%>%mutate(lab=factor(lab,levels=lab))
COL<-c(baseline=GREY, aside=GREY, ladder=ACC, cnn=HI)
SHP<-c(baseline=21, aside=21, ladder=19, cnn=19)
th<-theme_classic(base_size=BASE)+
  theme(text=element_text(colour=INK),
        axis.line.y=element_blank(), axis.ticks.y=element_blank(),
        axis.line.x=element_line(colour=INK,linewidth=.3),
        axis.ticks.x=element_line(colour=INK,linewidth=.3),
        panel.grid.major.y=element_line(colour=GRID,linewidth=.3),
        plot.title=element_text(face="bold",size=BASE,margin=margin(b=4)),
        plot.background=element_rect(fill=PAPER,colour=NA),
        plot.margin=margin(4,7,4,4))
bar<-function(val,lo,hi,ref,ttl,xlab,ylabs,xlim){
  d<-M%>%mutate(v=.data[[val]],
                l=if(is.na(lo)) NA_real_ else .data[[lo]],
                h=if(is.na(hi)) NA_real_ else .data[[hi]])
  p<-ggplot(d,aes(v,lab))+
    geom_vline(xintercept=ref,linetype=2,colour=INK,linewidth=.32)+
    {if(!is.na(lo)) geom_segment(aes(x=l,xend=h,y=lab,yend=lab,colour=I(COL[kind])),
                                 linewidth=.55,na.rm=TRUE)}+
    geom_point(aes(colour=I(COL[kind]),shape=I(SHP[kind]),fill=I(PAPER)),size=1.85,stroke=.55)+
    scale_x_continuous(limits=xlim,expand=c(0,0))+
    labs(title=ttl,x=xlab,y=NULL)+th
  if(!ylabs) p<-p+theme(axis.text.y=element_blank())
  p
}
pA<-bar("within_gene_winrate","winrate_ci_lo","winrate_ci_hi",CH_W,
        "A   Within-gene win rate","win rate",TRUE,c(.46,.88))
pB<-bar("p_at_1",NA,NA,CH_P,"B   Precision at 1","P@1",FALSE,c(.25,.70))
pC<-bar("pooled_auroc",NA,NA,0.5,"C   Pooled AUROC, reference","AUROC",FALSE,c(.46,.90))
pD<-bar("pooled_auprc",NA,NA,BR,"D   Pooled AUPRC, reference","AUPRC",FALSE,c(.08,.56))
p<-(pA|pB|pC|pD)+plot_layout(widths=c(1.75,1,1,1))
ggsave("figure_performance.pdf",p,width=9.8,height=3.0,device=cairo_pdf)
ggsave("figure_performance.png",p,width=9.8,height=3.0,dpi=400,bg=PAPER)
cat(sprintf("[perf] %d models, %s genes, %s peaks, base rate %.4f\n",nrow(M),
            format(NG,big.mark=","),format(NP,big.mark=","),BR))
cat(sprintf("[perf] chance win rate %.4f, chance P@1 %.4f\n",CH_W,CH_P))
for (i in seq_len(nrow(M))) cat(sprintf("   %-24s win %.4f [%s, %s]  P@1 %.4f  AUROC %.4f  AUPRC %.4f  %s\n",
  M$lab[i],M$within_gene_winrate[i],
  ifelse(is.na(M$winrate_ci_lo[i]),"  -  ",sprintf("%.4f",M$winrate_ci_lo[i])),
  ifelse(is.na(M$winrate_ci_hi[i]),"  -  ",sprintf("%.4f",M$winrate_ci_hi[i])),
  M$p_at_1[i],M$pooled_auroc[i],M$pooled_auprc[i],M$kind[i]))
cat("[out] figure_performance.pdf / .png\n")
