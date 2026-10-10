#!/usr/bin/env Rscript
# the internal-exon signature: a strong 5'ss
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork);library(scales)})
args<-commandArgs(trailingOnly=TRUE)
INDIR<-if(length(args)>=1) args[1] else "."
PFX  <-if(length(args)>=2) args[2] else "fig3"
rd<-function(f){p<-file.path(INDIR,paste0(PFX,"_",f)); if(file.exists(p)) read_tsv(p,show_col_types=FALSE) else NULL}
INK<-"#1A1A1A"; BASE<-13; CAN<-"#0072B2"; COMP<-"#E69F00"
th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),axis.text=element_text(color=INK,size=BASE-4),
  axis.line=element_line(color=INK,linewidth=.4),plot.title=element_text(size=BASE,face="bold",hjust=0),
  plot.subtitle=element_text(size=BASE-5,color="#555555"),legend.position="top",legend.title=element_blank())

AU<-rd("downstream_auroc.tsv")
dn<-function(mm,letter,lab){
  D<-rd(paste0("downstream_",mm,".tsv")); if(is.null(D)) return(ggplot()+theme_void())
  D$cls<-factor(D$cls,levels=c("Canonical","Competitor"))
  au<-if(!is.null(AU)) AU$auroc[AU$feature==paste0("downstream_",mm)][1] else NA
  ggplot(D,aes(value,fill=cls,color=cls))+
    geom_density(alpha=.25,linewidth=.7)+
    scale_fill_manual(values=c(Canonical=CAN,Competitor=COMP))+
    scale_color_manual(values=c(Canonical=CAN,Competitor=COMP))+
    labs(title=letter,subtitle=sprintf("%s 70\u2013300nt downstream of anchor 3\u2032ss (AUROC %.3f)",lab,au),
         x="strongest downstream signal",y="density")+th
}
pA<-dn("donor","A","Donor 5\u2032ss")
pB<-dn("u1","B","U1 hybridization")

# c: fraction present
if(!is.null(AU)){
  FR<-AU%>%select(feature,frac_canon_present,frac_comp_present)%>%
    tidyr::pivot_longer(-feature,names_to="cls",values_to="frac")%>%
    mutate(cls=ifelse(grepl("canon",cls),"Canonical","Competitor"),
           feat=ifelse(grepl("donor",feature),"Donor 5\u2032ss","U1"))
  pC<-ggplot(FR,aes(feat,frac,fill=cls))+
    geom_col(position=position_dodge(.7),width=.62)+
    geom_text(aes(label=sprintf("%.0f%%",frac*100)),position=position_dodge(.7),vjust=-0.3,size=3)+
    scale_fill_manual(values=c(Canonical=CAN,Competitor=COMP))+
    scale_y_continuous(labels=percent_format(1),expand=expansion(c(0,.1)))+
    labs(title="C",subtitle="Fraction with a strong downstream site",x=NULL,y="fraction of peaks")+th
} else pC<-ggplot()+theme_void()

fig<-(pA|pB|pC)+plot_layout(guides="collect")+
  plot_annotation(title="Internal-exon signature: competitors carry a strong 5\u2032ss downstream of their acceptor",
    theme=theme(plot.title=element_text(face="bold",size=BASE+2,hjust=0)))&theme(legend.position="top")
ggsave(file.path(INDIR,"figure3_internal_exon.pdf"),fig,width=14,height=5,device=cairo_pdf)
ggsave(file.path(INDIR,"figure3_internal_exon.png"),fig,width=14,height=5,dpi=300)
cat("[out] figure3_internal_exon.pdf / .png\n")