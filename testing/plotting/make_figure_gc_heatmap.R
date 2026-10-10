#!/usr/bin/env Rscript
# gc around the pa, canonical vs competitor
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork)})
INK<-"#1A1A1A"; BASE<-12; CAN<-"#0072B2"; COMP<-"#E69F00"; GRY<-"#8A8F98"
th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-3),axis.title=element_text(size=BASE-2),
  axis.line=element_line(color=INK,linewidth=.4),axis.ticks=element_line(color=INK,linewidth=.4),
  plot.title=element_text(size=BASE,face="bold",hjust=0),
  plot.subtitle=element_text(size=BASE-3,color="#555555",lineheight=1.05),
  legend.title=element_text(size=BASE-4),legend.text=element_text(size=BASE-4),
  legend.key.height=unit(.9,"lines"),legend.key.width=unit(.7,"lines"))

M<-read_tsv("gc_heatmap_matrix.tsv",show_col_types=FALSE)
P<-read_tsv("gc_profile.tsv",show_col_types=FALSE)
MU<-mean(M$gc,na.rm=TRUE); LIM<-0.16
BAND<-c(-50,0)
bandau<-P %>% filter(position>=BAND[1],position<=BAND[2])
best<-P %>% slice_max(abs(auroc-0.5),n=1)
bigd<-P %>% slice_max(abs(diff),n=1)
NR<-max(M$row)

heat<-function(g,ttl,sub){
  d<-M %>% filter(group==g)
  ggplot(d,aes(position,row,fill=gc))+geom_raster()+
    annotate("rect",xmin=BAND[1],xmax=BAND[2],ymin=0.5,ymax=NR+0.5,fill=NA,color=INK,linewidth=.35)+
    geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.4)+
    scale_fill_gradient2(low="#2166AC",mid="white",high="#B2182B",midpoint=MU,
                         limits=c(MU-LIM,MU+LIM),oob=scales::squish,
                         name=sprintf("GC fraction\n(50 nt window,\ncentred on %.2f)",MU))+
    scale_x_continuous(expand=c(0,0),breaks=c(-2000,-1500,-1000,-500,0,500))+
    scale_y_continuous(expand=c(0,0))+
    labs(title=ttl,subtitle=sub,x="position relative to pA (nt)",y="peak (sorted by anchor-to-pA distance)")+
    th+theme(legend.position="right")
}
pa<-heat("canonical","a   Canonical 3' ends","500 peaks, one per gene. Box = -50..0, where the per-window localiser peaked (AUROC 0.626).")
pb<-heat("competitor","b   Competitors","The same 500 genes, one competitor each. Rows sorted the same way, so a/b are row-comparable.")

lab<-sprintf("per-position AUROC: best %.3f at %+d nt; -50..0 band mean %.3f\nlargest gap %.4f GC at %+d nt (canonical %.3f vs competitor %.3f)",
             best$auroc,best$position,mean(bandau$auroc),abs(bigd$diff),bigd$position,bigd$mean_canonical,bigd$mean_competitor)
LG<-P %>% select(position,mean_canonical,mean_competitor) %>%
  tidyr::pivot_longer(-position,names_to="grp",values_to="gc") %>%
  mutate(grp=recode(grp,mean_canonical="canonical",mean_competitor="competitor"))
RB<-bind_rows(P %>% transmute(position,grp="canonical",lo=can_lo,hi=can_hi),
              P %>% transmute(position,grp="competitor",lo=comp_lo,hi=comp_hi))
pc<-ggplot()+
  annotate("rect",xmin=BAND[1],xmax=BAND[2],ymin=-Inf,ymax=Inf,fill="grey90",alpha=.8)+
  geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.4)+
  geom_ribbon(data=RB,aes(position,ymin=lo,ymax=hi,fill=grp),alpha=.25)+
  geom_line(data=LG,aes(position,gc,color=grp),linewidth=.8)+
  scale_color_manual(values=c(canonical=CAN,competitor=COMP),name=NULL)+
  scale_fill_manual(values=c(canonical=CAN,competitor=COMP),guide="none")+
  scale_x_continuous(expand=c(0,0),breaks=c(-2000,-1500,-1000,-500,0,500))+
  annotate("text",x=-1975,y=-Inf,label=lab,hjust=0,vjust=-0.4,size=BASE*0.24,color="#555555",lineheight=1.1)+
  labs(title="c   Mean GC per position, with the difference below",
       subtitle="Ribbons = 95% bootstrap over genes (the canonical / competitor pair is the resampling unit).",
       x=NULL,y="mean GC fraction")+th+
  theme(legend.position="bottom",axis.text.x=element_blank(),axis.ticks.x=element_blank(),
        plot.margin=margin(5,5,0,5))+coord_cartesian(clip="off")
pd<-ggplot(P,aes(position,diff))+
  annotate("rect",xmin=BAND[1],xmax=BAND[2],ymin=-Inf,ymax=Inf,fill="grey90",alpha=.8)+
  geom_hline(yintercept=0,color=GRY,linewidth=.4)+
  geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.4)+
  geom_ribbon(aes(ymin=diff_lo,ymax=diff_hi),fill=GRY,alpha=.3)+
  geom_line(linewidth=.7,color=INK)+
  scale_x_continuous(expand=c(0,0),breaks=c(-2000,-1500,-1000,-500,0,500))+
  labs(subtitle="canonical - competitor; below zero means the competitor window is GC-richer.",
       x="position relative to pA (nt)",y="GC difference")+th+theme(plot.margin=margin(0,5,5,5))

fig<-wrap_plots(pa,pb,ncol=2,guides="collect")/(pc/pd+plot_layout(heights=c(1.55,1)))+plot_layout(heights=c(1.5,1.35))+
  plot_annotation(
    title="GC composition around the pA: canonical 3' ends vs competitors",
    subtitle="pA-2000 .. pA+500, transcript orientation (minus strand reverse-complemented). Sliding GC, 50 nt window, 10 nt step, 246 columns, hg38.",
    caption=paste0("Rows in a and b are SORTED BY d_anchor_to_pa, the distance from the anchor 3' ss to the pA. The diagonal edge in both rasters is that sort key and is not a\n",
                   "result. 500 canonical and 500 competitor peaks, seed 42, gene-paired: each of 500 genes contributes its canonical peak and one of its own competitors, so a and b\n",
                   "share row genes. The two-class GC gap is small per bin, so panel c carries the claim; the rasters only show that it is not driven by a subset of rows."),
    theme=theme(plot.title=element_text(face="bold",size=BASE+3,color=INK),
                plot.subtitle=element_text(size=BASE-3,color="#555555"),
                plot.caption=element_text(size=BASE-5,color="#666666",hjust=0)))
ggsave("figure_gc_heatmap.pdf",fig,width=15,height=12.5,device=cairo_pdf)
ggsave("figure_gc_heatmap.png",fig,width=15,height=12.5,dpi=200)
cat(sprintf("wrote figure_gc_heatmap.pdf / .png  (mean GC %.4f, fill +/-%.2f)\n",MU,LIM))
