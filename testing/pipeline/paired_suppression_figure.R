#!/usr/bin/env Rscript
# (clean 3-row stack) row 1: a donor %competitor>true end by distance | b donor median diff
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(tidyr);library(readr);library(patchwork)})

col_u1<-"#E69F00"; col_me5<-"#C1651A"; col_3ss<-"#103A5C"; col_null<-"#8A8F98"; ink<-"#1A1A1A"
AGG<-Sys.getenv("AGG","pair")
theme_fig<-function(base=12) theme_minimal(base_size=base)+
  theme(plot.title=element_text(face="bold",size=base+1,color=ink),
        plot.subtitle=element_text(size=base-2,color="grey35",margin=margin(b=6)),
        axis.title=element_text(size=base-1,color=ink),
        axis.title.y=element_text(margin=margin(r=8)),
        axis.text=element_text(size=base-3,color="grey25"),
        panel.grid.minor=element_blank(),
        panel.grid.major=element_line(color="grey93",linewidth=.4),
        plot.margin=margin(10,16,8,10))
rd<-function(f) if(file.exists(f)) read_tsv(f,show_col_types=FALSE) else NULL

# donor data
u1<-rd("fivess_profile_u1_paired_bydist.tsv");  if(!is.null(u1)) u1$metric<-"U1 hybridization"
me5<-rd("fivess_profile_me5_paired_bydist.tsv");if(!is.null(me5)) me5$metric<-"5\u2032 splice-site (MaxEntScan)"
if(is.null(u1)) stop("missing fivess_profile_u1_paired_bydist.tsv")
D<-bind_rows(u1,me5)
if("agg" %in% names(D)) D<-D%>%filter(kind=="random"|agg==AGG)
donor_cols<-c("U1 hybridization"=col_u1,"5\u2032 splice-site (MaxEntScan)"=col_me5)

# a: donor %comp>true end by distance (shell)
sR<-D%>%filter(scheme=="shell",kind=="real")%>%mutate(mid=(d0+d1)/2)
sN<-D%>%filter(scheme=="shell",kind=="random")%>%mutate(mid=(d0+d1)/2)
pA<-ggplot()+
  geom_hline(yintercept=50,color="grey70",linewidth=.4,linetype="22")+
  geom_line(data=sN,aes(mid,frac_pos*100,group=metric),color=col_null,linewidth=.6,linetype="22")+
  geom_ribbon(data=sR,aes(mid,ymin=frac_lo*100,ymax=frac_hi*100,fill=metric),alpha=.18)+
  geom_line(data=sR,aes(mid,frac_pos*100,color=metric),linewidth=1)+
  geom_point(data=sR,aes(mid,frac_pos*100,color=metric),size=1.8)+
  scale_color_manual(values=donor_cols)+scale_fill_manual(values=donor_cols)+
  labs(title="A  Donor suppression is proximal",
       subtitle=sprintf("%% competitor > true end in upstream donor/U1 load (agg: %s); grey = random pair (50%%)",AGG),
       x="upstream distance from cleavage site (nt)",y="% competitor > true end",color=NULL,fill=NULL)+
  theme_fig()+theme(legend.position=c(.99,.99),legend.justification=c(1,1),
                    legend.background=element_rect(fill="white",color=NA),legend.text=element_text(size=8),
                    legend.key.size=unit(.9,"lines"))

# b: donor median diff by window (cumulative)
cR<-D%>%filter(scheme=="cumulative",kind=="real",   d1>=500)
cN<-D%>%filter(scheme=="cumulative",kind=="random", d1>=500)
pB<-ggplot()+
  geom_hline(yintercept=0,color="grey60",linewidth=.5)+
  geom_line(data=cN,aes(d1,median,group=metric),color=col_null,linewidth=.6,linetype="22")+
  geom_ribbon(data=cR,aes(d1,ymin=med_lo,ymax=med_hi,fill=metric),alpha=.18)+
  geom_line(data=cR,aes(d1,median,color=metric),linewidth=1)+
  geom_point(data=cR,aes(d1,median,color=metric),size=2)+
  scale_color_manual(values=donor_cols)+scale_fill_manual(values=donor_cols)+
  labs(title="B  Widening the window dilutes it",
       subtitle="cumulative 0\u2013X: median within-gene (competitor \u2212 true end) load",
       x="upstream window width (0 to X nt)",y="median (competitor \u2212 true end)\nload / 100nt",color=NULL)+
  theme_fig()+theme(legend.position="none")

# acceptor data
B3<-rd("threess_profile_paired_bydist.tsv")
if(!is.null(B3) && "agg" %in% names(B3)){
  a2R<-B3%>%filter(scheme=="shell",kind=="real",agg==AGG)%>%mutate(mid=(d0+d1)/2)
  a2N<-B3%>%filter(scheme=="shell",kind=="random")%>%mutate(mid=(d0+d1)/2)
  pA2<-ggplot()+
    geom_hline(yintercept=50,color="grey70",linewidth=.4,linetype="22")+
    geom_line(data=a2N,aes(mid,frac_pos*100),color=col_null,linewidth=.6,linetype="22")+
    geom_ribbon(data=a2R,aes(mid,ymin=frac_lo*100,ymax=frac_hi*100),fill=col_3ss,alpha=.18)+
    geom_line(data=a2R,aes(mid,frac_pos*100),color=col_3ss,linewidth=1)+
    geom_point(data=a2R,aes(mid,frac_pos*100),color=col_3ss,size=1.8)+
    labs(title="A2  Acceptor definition is proximal",
         subtitle=sprintf("%% genes where the TRUE END has a stronger upstream 3\u2032ss (agg: %s); grey = random (50%%)",AGG),
         x="upstream distance from cleavage site (nt)",y="% canonical > competitor")+
    theme_fig()
  b2R<-B3%>%filter(scheme=="cumulative",kind=="real",agg==AGG, d1>=500)
  b2N<-B3%>%filter(scheme=="cumulative",kind=="random", d1>=500)
  has_ci<-all(c("med_lo","med_hi") %in% names(b2R))
  pB2<-ggplot()+
    geom_hline(yintercept=0,color="grey60",linewidth=.5)+
    geom_line(data=b2N,aes(d1,median),color=col_null,linewidth=.6,linetype="22")+
    {if(has_ci) geom_ribbon(data=b2R,aes(d1,ymin=med_lo,ymax=med_hi),fill=col_3ss,alpha=.18)}+
    geom_line(data=b2R,aes(d1,median),color=col_3ss,linewidth=1)+
    geom_point(data=b2R,aes(d1,median),color=col_3ss,size=2)+
    labs(title="B2  Acceptor advantage vs window",
         subtitle="cumulative 0\u2013X: median within-gene (canonical \u2212 competitor) 3\u2032ss",
         x="upstream window width (0 to X nt)",y="median (canonical \u2212 competitor)\n3\u2032ss")+
    theme_fig()
  cAll<-B3%>%filter(scheme=="cumulative",kind=="real",agg %in% c("pair","mean","max","min"))
  cN2<-B3%>%filter(scheme=="cumulative",kind=="random")
  agg_cols<-c(max="#103A5C",pair="#2E7D9E",mean="#3E6B8B",min="#9CB4C7")
  agg_lab<-c(max="vs strongest competitor",pair="per competitor (each pair)",mean="vs mean competitor",min="vs weakest competitor")
  cAll$agg<-factor(cAll$agg,levels=c("max","pair","mean","min"))
  pC<-ggplot()+
    geom_hline(yintercept=50,color="grey70",linewidth=.4,linetype="22")+
    geom_line(data=cN2,aes(d1,frac_pos*100),color=col_null,linewidth=.6,linetype="22")+
    geom_line(data=cAll,aes(d1,frac_pos*100,color=agg),linewidth=1)+
    geom_point(data=cAll,aes(d1,frac_pos*100,color=agg),size=1.8)+
    scale_color_manual(values=agg_cols,labels=agg_lab,name=NULL)+
    labs(title="C  Acceptor defines the true end (all aggregations)",
         subtitle="% genes canonical > competitor upstream 3\u2032ss, by cumulative window",
         x="upstream window width (0 to X nt)",y="% canonical > competitor")+
    theme_fig()+theme(legend.position=c(.99,.02),legend.justification=c(1,0),
                      legend.background=element_rect(fill="white",color=NA),legend.text=element_text(size=8),
                      legend.key.size=unit(.9,"lines"))
} else { pA2<-ggplot()+theme_void(); pB2<-ggplot()+theme_void(); pC<-ggplot()+theme_void() }

fig<-(pA|pB)/(pA2|pB2)/pC+
  plot_layout(heights=c(1,1,1))+
  plot_annotation(
    title="Within-gene paired mechanism: competitors carry more upstream DONOR signal (suppression); true ends have the stronger upstream ACCEPTOR (definition)",
    caption="All within-gene paired (gene-position confound cancelled). A,B donor/U1: competitor > true end, proximal. A2,B2,C acceptor: canonical > competitor, proximal.",
    theme=theme(plot.title=element_text(face="bold",size=13,color=ink,hjust=.5,margin=margin(b=8)),
                plot.caption=element_text(size=9,color="grey40",hjust=0)))

ggsave("paired_suppression.pdf",fig,width=11,height=14,device=cairo_pdf)
ggsave("paired_suppression.png",fig,width=11,height=14,dpi=200)
cat("wrote paired_suppression.pdf / .png  (3-row stack, 11x14)\n")