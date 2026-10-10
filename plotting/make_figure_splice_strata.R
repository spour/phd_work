#!/usr/bin/env Rscript
# splice_arch (4 features) by competitor stratum
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork)})
INK<-"#1A1A1A"; BASE<-12; CAN<-"#0072B2"; GRY<-"#8A8F98"
th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-3),axis.title=element_text(size=BASE-2),
  axis.line=element_line(color=INK,linewidth=.4),axis.ticks=element_line(color=INK,linewidth=.4),
  plot.title=element_text(size=BASE,face="bold",hjust=0),
  plot.subtitle=element_text(size=BASE-3,color="#555555",lineheight=1.05),
  legend.position="none",panel.grid.major.y=element_line(color="grey92",linewidth=.3))
num<-c("n_peaks","n_genes","n_genes_ranked","value","lo","hi","chance","chance_perm","excess","exc_lo","exc_hi","base_rate")
D<-read_tsv("splice_arch_strata.tsv",show_col_types=FALSE,col_types=cols(.default=col_character())) %>%
  filter(section=="metric") %>% mutate(across(all_of(num),as.numeric))
lab<-D %>% distinct(stratum,n_peaks,n_genes) %>% arrange(n_peaks) %>%
  mutate(tag=sprintf("%s   %s peaks / %s genes",stratum,formatC(n_peaks,big.mark=",",format="d"),
                     formatC(n_genes,big.mark=",",format="d")))
D<-D %>% left_join(lab %>% select(stratum,tag),by="stratum") %>% mutate(tag=factor(tag,levels=lab$tag))

pan<-function(met,useexc,ttl,sub,xlab,ref,txt){
  d<-D %>% filter(metric==met)
  if(useexc) d<-d %>% mutate(value=excess,lo=exc_lo,hi=exc_hi)
  d$txt<-txt(D %>% filter(metric==met))
  rngx<-range(c(d$lo,d$hi,ref),na.rm=TRUE); pad<-diff(rngx)*0.62
  p<-ggplot(d,aes(value,tag))+
    {if(is.finite(ref)) geom_vline(xintercept=ref,linetype=2,color=GRY,linewidth=.4)}+
    geom_errorbar(aes(xmin=lo,xmax=hi),orientation="y",width=.24,linewidth=.5,color=INK)+
    geom_point(size=2.4,shape=21,fill=CAN,color=INK,stroke=.35)
  if(met=="AUPRC") p<-p+geom_point(aes(x=base_rate),size=2.0,shape=23,fill="white",color=GRY,stroke=.5)
  p+geom_text(aes(x=rngx[2]+pad*0.08,label=txt),hjust=0,size=BASE*0.24,color="#555555")+
    coord_cartesian(xlim=c(rngx[1],rngx[2]+pad),clip="off")+
    scale_x_continuous(expand=expansion(mult=c(.05,.05)))+
    labs(title=ttl,subtitle=sub,x=xlab,y=NULL)+th
}
p1<-pan("AUROC",FALSE,"a   AUROC","Held-out, pooled over folds; dashed = 0.5.","AUROC",0.5,
        function(d) sprintf("%.3f",d$value))
p2<-pan("AUPRC",FALSE,"b   AUPRC","Diamond = that stratum's base rate (canonical share of the pool).","AUPRC",NA,
        function(d) sprintf("%.3f vs %.3f base",d$value,d$base_rate))
p3<-pan("P@1",TRUE,"c   P@1, excess over that stratum's chance","Text = raw P@1 vs chance, mean(1 / n_i), permutation checked.","P@1 - chance",0,
        function(d) sprintf("%.3f vs %.3f",d$value,d$chance))
p4<-pan("MRR",TRUE,"d   MRR, excess over that stratum's chance","Text = raw MRR vs chance, mean(H_n / n), permutation checked.","MRR - chance",0,
        function(d) sprintf("%.3f vs %.3f",d$value,d$chance))

fig<-(p1|p2)/(p3|p4)+plot_annotation(
  title="SPLICE_arch (4 features) by competitor context",
  subtitle=paste0("spl_acc, spl_dpa, spl_strong5, spl_max5 only. 5-fold gene-split CV, standardized in training folds, ridge 1.0, pooled held-out predictions,\n",
                  "bootstrap CI over genes. Canonical set fixed at 6,583; only the competitor set varies. Rows ordered by peaks."),
  caption=paste0("Strata OVERLAP: intronic contains the three intron rows, exonic contains terminal exon, and unannotated + intronic sits inside both of its parents. This is a\n",
                 "descriptive breakdown, not a set of independent tests; no multiplicity correction is implied. P@1 and MRR chance depend on peaks per gene, so only the excess is\n",
                 "comparable across rows; chance was verified by within-gene label permutation (chance_perm in splice_arch_strata.tsv)."),
  theme=theme(plot.title=element_text(face="bold",size=BASE+3,color=INK),
              plot.subtitle=element_text(size=BASE-3,color="#555555"),
              plot.caption=element_text(size=BASE-5,color="#666666",hjust=0)))
ggsave("figure_splice_strata.pdf",fig,width=15.5,height=9.8,device=cairo_pdf)
ggsave("figure_splice_strata.png",fig,width=15.5,height=9.8,dpi=200)
cat("wrote figure_splice_strata.pdf / .png\n")
