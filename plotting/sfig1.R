#!/usr/bin/env Rscript
# supplementary figure 1: selection of constitutive polya sites from polya db v4
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(patchwork);library(scales)})
args<-commandArgs(trailingOnly=TRUE)
TSV<-if(length(args)>=1) args[1] else "constitutive_cpa.tsv"

INK<-"#1F2937"; MUT<-"#6B7280"; KEEP<-"#3E82B5"; ACC<-"#E8963A"; DROPC<-"#B0662A"
th<-theme_minimal(base_size=12)+
  theme(text=element_text(color=INK),
        plot.title=element_text(face="bold",size=13,hjust=0,margin=margin(t=6,b=2)),
        plot.subtitle=element_text(size=10,color=MUT,margin=margin(b=6)),
        axis.text=element_text(color=INK,size=9),
        panel.grid.minor=element_blank(),
        panel.grid.major.y=element_blank(),
        legend.position="none")

# numbers from the polya db v4 diagnostics (editable)
stage_nm<-c("all gene symbols","protein-coding","uniquely mapped","3\u2032UTR site (kept)")
stage_n <-c(31101, 17623, 16749, 15700)
drop_rsn<-c("\u2212 13,478 non-coding biotypes","\u2212 874 multi-mapped (PAS2gene\u22601)","\u2212 1,049 no 3\u2032UTR PAS")

biotype<-c("protein_coding","lncRNA","transcribed_pseudogene","pseudogene","snoRNA",
           "miRNA","tRNA","V_segment","snRNA","misc_RNA","ncRNA","V_segment_pseudogene","other")
biocnt <-c(17623,11278,803,607,286,146,116,74,47,24,20,16,61)

drop_lab<-c("Intron","Intergenic","CDS","5\u2032UTR","Downstream","Upstream",
            "3\u2032-most exon","Internal exon","5\u2032-most exon")
drop_n  <-c(1679,1385,388,270,199,142,28,4,1)
drop_term<-c(FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,TRUE,FALSE,FALSE)

# a: selection funnel
Wmax<-1.0; hh<-0.20; maxn<-max(stage_n)
fd<-data.frame(nm=stage_nm,n=stage_n,hw=stage_n/maxn*Wmax,y=-(seq_along(stage_n)-1))
bars<-do.call(rbind,lapply(seq_len(nrow(fd)),function(i)
  data.frame(id=i,x=c(-fd$hw[i],fd$hw[i],fd$hw[i],-fd$hw[i]),
             y=c(fd$y[i]-hh,fd$y[i]-hh,fd$y[i]+hh,fd$y[i]+hh))))
bars$fill<-factor(bars$id)
conn<-do.call(rbind,lapply(seq_len(nrow(fd)-1),function(i)
  data.frame(id=i,x=c(-fd$hw[i],fd$hw[i],fd$hw[i+1],-fd$hw[i+1]),
             y=c(fd$y[i]-hh,fd$y[i]-hh,fd$y[i+1]+hh,fd$y[i+1]+hh))))
fills<-c("#A9C9E0","#6FA6CC","#3E82B5","#1E5C8F")
ctext<-c("#123A57","white","white","white")
xlab_r<-Wmax+0.12
pA<-ggplot()+
  geom_polygon(data=conn,aes(x,y,group=id),fill="#E9EDF1",color=NA)+
  geom_polygon(data=bars,aes(x,y,group=id,fill=fill),color="white",linewidth=.4)+
  scale_fill_manual(values=fills)+
  guides(fill="none")+
  geom_text(data=fd,aes(0,y,label=comma(n)),color=ctext,fontface="bold",size=4)+
  geom_text(data=fd,aes(xlab_r,y,label=nm),hjust=0,size=3.4,color=INK)+
  geom_text(data=data.frame(y=(fd$y[-nrow(fd)]+fd$y[-1])/2,lab=drop_rsn),
            aes(xlab_r,y,label=lab),hjust=0,size=3,fontface="italic",color=DROPC)+
  annotate("text",x=0,y=fd$y[nrow(fd)]-0.42,
           label=sprintf("%.1f%% of protein-coding genes retained",100*15700/17623),
           size=3,color=MUT)+
  coord_cartesian(xlim=c(-Wmax-0.05,Wmax+1.75),ylim=c(min(fd$y)-0.55,0.30),clip="off")+
  labs(title="Selecting constitutive polyadenylation sites",
       subtitle="One dominant 3\u2032UTR site per gene (highest PSE\u00d7RPM), from 303,312 PAS in PolyA DB v4")+
  theme_void(base_size=12)+
  theme(plot.title=element_text(face="bold",size=13,hjust=0),
        plot.subtitle=element_text(size=10,color=MUT),
        plot.margin=margin(4,4,2,4))

# b: biotype composition (lollipop)
bt<-data.frame(biotype,n=biocnt)%>%arrange(desc(n))
bt$biotype<-factor(bt$biotype,levels=rev(bt$biotype))
bt$hl<-bt$biotype=="protein_coding"
pB<-ggplot(bt,aes(n,biotype))+
  geom_segment(aes(x=0,xend=n,yend=biotype,color=hl),linewidth=.9)+
  geom_point(aes(color=hl,size=hl))+
  geom_text(aes(label=comma(n)),hjust=-0.22,size=2.7,color=INK)+
  scale_color_manual(values=c(`TRUE`=ACC,`FALSE`="#9AA0A6"))+
  scale_size_manual(values=c(`TRUE`=3.3,`FALSE`=2.1))+
  scale_x_continuous(expand=expansion(mult=c(0,0.20)),labels=comma)+
  labs(title="Most gene symbols aren't protein-coding",
       subtitle="Distinct gene symbols by biotype",
       x="genes",y=NULL)+th
# c: dropped genes have no 3'utr site
dc<-data.frame(lab=drop_lab,n=drop_n,term=drop_term)%>%arrange(desc(n))
dc$lab<-factor(dc$lab,levels=rev(dc$lab))
dc$grp<-ifelse(dc$term,"terminal exon (recoverable)","cryptic / non-terminal")
pC<-ggplot(dc,aes(n,lab,fill=grp))+
  geom_col(width=.68)+
  geom_text(aes(label=comma(n)),hjust=-0.2,size=2.7,color=INK)+
  scale_fill_manual(values=c("cryptic / non-terminal"="#9AA0A6","terminal exon (recoverable)"=ACC),name=NULL)+
  scale_x_continuous(expand=expansion(mult=c(0,0.18)),labels=comma)+
  labs(title="Dropped genes lack a 3\u2032UTR site",
       subtitle="PAS location \u2014 nearly all cryptic",
       x="PAS in these genes",y=NULL)+
  th+theme(legend.position=c(.72,.28),legend.background=element_rect(fill="white",color=NA),
           legend.text=element_text(size=8),legend.key.size=unit(.8,"lines"))

pD<-NULL
if(file.exists(TSV)){
  dd<-tryCatch(suppressMessages(readr::read_tsv(TSV,show_col_types=FALSE)),error=function(e) NULL)
  if(!is.null(dd) && "PSExRPM"%in%names(dd)){
    v<-suppressWarnings(as.numeric(dd$PSExRPM)); v<-v[is.finite(v)&v>0]
    if(length(v)>=10){
      med<-median(v); q1<-as.numeric(quantile(v,.25)); q3<-as.numeric(quantile(v,.75))
      pD<-ggplot(data.frame(v=v),aes(v))+
        geom_histogram(aes(y=after_stat(density)),bins=46,fill="#CBDDEC",color="white",linewidth=.2)+
        geom_density(color=KEEP,linewidth=.9)+
        geom_vline(xintercept=med,color=DROPC,linewidth=.7)+
        geom_vline(xintercept=c(q1,q3),color=MUT,linetype=2,linewidth=.4)+
        scale_x_log10(labels=trans_format("log10",math_format(10^.x)))+
        annotate("text",x=med,y=Inf,label=sprintf("median %s",comma(round(med))),
                 hjust=-0.08,vjust=1.6,size=3,color=DROPC)+
        labs(title=sprintf("Selected sites well-supported (n=%s)",comma(length(v))),
             subtitle="PSE\u00d7RPM per gene (log; dashed = IQR)",
             x="PSE \u00d7 RPM",y="density")+th
    }
  }
}
if(is.null(pD))
  pD<-ggplot()+annotate("text",0,0,label=paste0("PSE\u00d7RPM panel:\nrun in the dir with ",TSV),size=3.2,color=MUT)+
    labs(title="Selected-site scores")+theme_void()+theme(plot.title=element_text(face="bold"))

fig<-pA/(pB|pC|pD)+plot_layout(heights=c(1.18,1))+
  plot_annotation(
    title="Supplementary Figure 1  |  Constitutive polyadenylation-site selection from PolyA DB v4",
    caption="Constitutive site = the highest PSE\u00d7RPM PAS in the 3\u2032UTR of each protein-coding gene (one per gene). PSE\u00d7RPM from the 3READS quantification.",
    theme=theme(plot.title=element_text(face="bold",size=14,hjust=0),
                plot.caption=element_text(size=8.5,color=MUT,hjust=0)))
ggsave("supp_fig1.pdf",fig,width=13,height=10,device=cairo_pdf,limitsize=FALSE)
ggsave("supp_fig1.png",fig,width=13,height=10,dpi=220,limitsize=FALSE)
cat("wrote supp_fig1.pdf / .png\n")