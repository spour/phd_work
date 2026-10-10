# the rescue as a flow, not a stack of bar charts
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(tidyr);library(ggforce)})
INK<-"#16191C"; GTX<-"#5A6068"; PAPER<-"#FFFFFF"
NEU<-"#B6BDC4"; FAIL<-"#2A6DB0"; GOOD<-"#14395E"; MISS<-"#D9A441"; SPL<-"#E8A33D"
BASE<-8
G<-read.delim("rescue_rate_genes.tsv",stringsAsFactors=FALSE)
RK<-read.delim("rescue_ranks.tsv",stringsAsFactors=FALSE)
B<-G%>%filter(candidate_set=="full_shortlist")
N_ALL<-15560; N_RES<-8977; N_CON<-nrow(B)
N_REC<-sum(B$cnn); N_MISS<-N_CON-N_REC
N_OK0<-N_RES; N_OK1<-N_RES+N_REC
stopifnot(N_CON==6583)
sp1<-RK$rank_splice==1; cn1<-RK$rank_cnn==1; cb1<-RK$rank_combo==1
nA<-nrow(RK); fCON<-N_CON/N_ALL; fRES<-N_RES/N_ALL
TOT_CB<-fRES+mean(cb1)*fCON
TOT_OR<-fRES+mean(sp1|cn1)*fCON

# a: what each model reads
mag<-function(x) -2000+(x+250)/500*2500
yTOP<-0; yMAG<--1.55
WIDE<-tribble(~x,~lab,~col,
  -1750,"5' splice site",SPL, -1180,"5' splice site",SPL,
   -760,"3' splice site",FAIL, -430,"5' splice site",SPL)
CORE<-tribble(~x,~lab,
  -170,"UGUA",  -58,"AAUAAA",  56,"U-rich",  170,"GU-rich")
pA<-ggplot()+
  annotate("rect",xmin=-2000,xmax=500,ymin=yTOP-.30,ymax=yTOP+.30,fill=FAIL,alpha=.13)+
  annotate("segment",x=-2000,xend=500,y=yTOP,yend=yTOP,colour="#B9C0C7",linewidth=.9)+
  annotate("rect",xmin=-250,xmax=250,ymin=yTOP-.30,ymax=yTOP+.30,fill=NEU,alpha=.75)+
  geom_point(data=WIDE,aes(x,yTOP,fill=I(col)),shape=21,colour=PAPER,size=2.7,stroke=.5)+
  geom_text(data=WIDE%>%distinct(lab,.keep_all=TRUE),aes(x,yTOP+.44,label=lab),
            size=2.2,colour=GTX)+
  annotate("segment",x=0,xend=0,y=yTOP+.30,yend=yTOP+.80,colour=INK,linewidth=.45)+
  annotate("point",x=0,y=yTOP+.30,shape=25,size=1.6,fill=INK,colour=INK)+
  annotate("text",x=0,y=yTOP+.96,label="cleavage site",size=2.35,colour=INK)+
  annotate("text",x=-2000,y=yTOP+1.16,hjust=0,size=2.7,colour=INK,fontface="bold",
           label="extended models")+
  annotate("text",x=-2000,y=yTOP+.80,hjust=0,size=2.35,colour=GTX,label="2,500 nt")+
  annotate("segment",x=-250,xend=mag(-250),y=yTOP-.32,yend=yMAG+.34,colour="#C6CBD0",
           linewidth=.3,linetype=2)+
  annotate("segment",x=250,xend=mag(250),y=yTOP-.32,yend=yMAG+.34,colour="#C6CBD0",
           linewidth=.3,linetype=2)+
  annotate("rect",xmin=mag(-250),xmax=mag(250),ymin=yMAG-.30,ymax=yMAG+.30,fill=NEU,alpha=.55)+
  annotate("segment",x=mag(-250),xend=mag(250),y=yMAG,yend=yMAG,colour="#B9C0C7",linewidth=.9)+
  annotate("segment",x=mag(0),xend=mag(0),y=yMAG-.30,yend=yMAG+.30,colour=INK,linewidth=.4)+
  geom_point(data=CORE,aes(mag(x),yMAG),fill=GOOD,shape=21,colour=PAPER,size=2.7,stroke=.5)+
  geom_text(data=CORE,aes(mag(x),yMAG-.50,label=lab),size=2.2,colour=GTX)+
  annotate("text",x=-2000,y=yMAG+1.12,hjust=0,size=2.7,colour=INK,fontface="bold",
           label="baseline")+
  annotate("text",x=-2000,y=yMAG+.78,hjust=0,size=2.35,colour=GTX,label="500 nt")+
  annotate("text",x=-2030,y=yTOP,hjust=1,label="-2,000",size=2.0,colour=GTX)+
  annotate("text",x=530,y=yTOP,hjust=0,label="+500",size=2.0,colour=GTX)+
  annotate("text",x=mag(-250)-30,y=yMAG,hjust=1,label="-250",size=2.0,colour=GTX)+
  annotate("text",x=mag(250)+30,y=yMAG,hjust=0,label="+250",size=2.0,colour=GTX)+
  annotate("text",x=0,y=yMAG-1.00,size=2.1,colour=GTX,
           label="nt relative to the cleavage site")+
  scale_x_continuous(limits=c(-2300,780))+
  coord_cartesian(ylim=c(-2.60,1.86),expand=FALSE,clip="off")+
  labs(title="a   Sequence context read by each model")+
  theme_void(base_size=BASE)+
  theme(text=element_text(colour=INK),
        plot.title=element_text(face="bold",size=BASE+0.5,colour=INK,margin=margin(b=6)),
        plot.background=element_rect(fill=PAPER,colour=NA),plot.margin=margin(6,8,10,8))

# a: the flow, with attribution
N_UNI<-N_RES+nA
N_BOTH<-sum(sp1&cn1); N_CNNO<-sum(!sp1&cn1); N_SPLO<-sum(sp1&!cn1); N_NEI<-sum(!sp1&!cn1)
N_ANY<-N_BOTH+N_CNNO+N_SPLO
N_OKU<-N_RES+N_ANY
P_BASE<-N_RES/N_UNI; P_ORAC<-N_OKU/N_UNI; P_COMB<-(N_RES+sum(cb1))/N_UNI
stopifnot(N_BOTH+N_CNNO+N_SPLO+N_NEI==nA)
GAP<-620
nodes<-tribble(~id,~x,~lo,~fill,~lab,~n,~side,
  "all",  0, 0,                                    NEU,  "all genes",         N_UNI,  "l",
  "res",  1, 0,                                    NEU,  "baseline resolves", N_RES,  "r",
  "con",  1, N_RES+GAP,                            FAIL, "baseline fails",    nA,     "t",
  "keep", 2, 0,                                    NEU,  "already correct",   N_RES,  "n",
  "both", 2, N_RES+GAP,                            GOOD, "both models",       N_BOTH, "n",
  "cnno", 2, N_RES+GAP+N_BOTH+220,                 FAIL, "sequence CNN only", N_CNNO, "n",
  "splo", 2, N_RES+GAP+N_BOTH+N_CNNO+440,          SPL,  "splice only",       N_SPLO, "n",
  "nei",  2, N_RES+GAP+N_BOTH+N_CNNO+N_SPLO+660,   MISS, "neither",           N_NEI,  "n",
  "ok",   3, 0,                                    GOOD, "correct",           N_OKU,  "r",
  "bad",  3, N_OKU+GAP,                            MISS, "incorrect",         N_NEI,  "r")%>%
  mutate(hi=lo+n)
nd<-function(i) nodes[nodes$id==i,]
rib<-function(a,b,ao,bo,n,fill,alpha){
  A<-nd(a); B_<-nd(b)
  data.frame(x=c(A$x+.055,B_$x-.055,B_$x-.055,A$x+.055),
             y=c(A$lo+ao,B_$lo+bo,B_$lo+bo+n,A$lo+ao+n),
             fill=fill,alpha=alpha,g=paste0(a,b,ao))
}
ribs<-rbind(
  rib("all","res",0,0,N_RES,NEU,.42),
  rib("all","con",N_RES,0,nA,FAIL,.36),
  rib("res","keep",0,0,N_RES,NEU,.38),
  rib("con","both",0,0,N_BOTH,GOOD,.50),
  rib("con","cnno",N_BOTH,0,N_CNNO,FAIL,.46),
  rib("con","splo",N_BOTH+N_CNNO,0,N_SPLO,SPL,.60),
  rib("con","nei",N_BOTH+N_CNNO+N_SPLO,0,N_NEI,MISS,.50),
  rib("keep","ok",0,0,N_RES,NEU,.38),
  rib("both","ok",0,N_RES,N_BOTH,GOOD,.55),
  rib("cnno","ok",0,N_RES+N_BOTH,N_CNNO,FAIL,.50),
  rib("splo","ok",0,N_RES+N_BOTH+N_CNNO,N_SPLO,SPL,.62),
  rib("nei","bad",0,0,N_NEI,MISS,.50))
lab<-nodes%>%mutate(ym=(lo+hi)/2,
  txt=sprintf("%s\n%s   %.1f%%",lab,format(n,big.mark=","),100*n/N_UNI))
mid<-lab%>%filter(side=="n",id!="keep")%>%
  mutate(tx=sprintf("%s   %s   %.1f%%",lab,format(n,big.mark=","),100*n/N_UNI))
pB2<-ggplot()+
  geom_diagonal_wide(data=ribs,aes(x,y,group=g,fill=I(fill),alpha=I(alpha)),
                     strength=.72,colour=NA)+
  geom_rect(data=nodes,aes(xmin=x-.028,xmax=x+.028,ymin=lo,ymax=hi,fill=I(fill)),colour=NA)+
  geom_text(data=lab%>%filter(side=="l"),aes(x-.055,ym,label=txt),hjust=1,size=2.5,
            colour=INK,lineheight=1.06)+
  geom_text(data=lab%>%filter(side=="r"),aes(x+.055,ym,label=txt),hjust=0,size=2.5,
            colour=INK,lineheight=1.06)+
  geom_text(data=lab%>%filter(side=="t"),aes(x,hi+560,label=txt),hjust=.5,vjust=0,size=2.5,
            colour=INK,lineheight=1.06)+
  geom_text(data=mid,aes(x-.075,ym,label=tx),hjust=1,size=2.3,colour=INK)+
  annotate("text",x=-.34,y=N_UNI+1750,
           label=sprintf("P@1   %.3f   \u2192   %.3f",P_BASE,P_ORAC),
           size=3.5,colour=INK,fontface="bold",hjust=0)+
  coord_cartesian(xlim=c(-.46,3.40),ylim=c(-1200,N_UNI+3100),expand=FALSE,clip="off")+
  labs(title="b   Recovery of baseline-unresolved 3' ends")+
  theme_void(base_size=BASE)+
  theme(plot.title=element_text(face="bold",size=BASE+0.5,colour=INK,margin=margin(b=6)),
        plot.background=element_rect(fill=PAPER,colour=NA),plot.margin=margin(6,8,4,8))

# b: the mechanism, as curves
MK<-c(baseline="rank_baseline",splice="rank_splice",`sequence CNN`="rank_cnn",
      combined="rank_combo")
KS<-1:12
CU<-do.call(rbind,lapply(names(MK),function(m)
  data.frame(model=m,k=KS,f=sapply(KS,function(k) mean(RK[[MK[m]]]<=k)))))%>%
  mutate(model=factor(model,levels=names(MK)))
CO<-c(baseline=NEU,splice=SPL,`sequence CNN`=FAIL,combined=GOOD)
r1<-setNames(CU$f[CU$k==1],as.character(CU$model[CU$k==1]))
ends<-CU%>%filter(k==max(KS))%>%
  mutate(txt=sprintf("%s  %.0f%% at rank 1",model,100*r1[as.character(model)]),
         ylab=c(combined=.965,`sequence CNN`=.885,splice=.805,baseline=.725)[as.character(model)])
pB<-ggplot(CU,aes(k,f,colour=model))+
  geom_line(linewidth=.65)+
  geom_point(data=CU%>%filter(k==1),size=2.1)+
  geom_segment(data=ends,aes(x=max(KS),xend=max(KS)+.9,y=f,yend=ylab),
               linewidth=.25,linetype=3,show.legend=FALSE)+
  geom_text(data=ends,aes(x=max(KS)+1.1,y=ylab,label=txt),hjust=0,size=2.5,show.legend=FALSE)+
  scale_colour_manual(values=CO)+
  scale_x_continuous(breaks=c(1,3,6,9,12),limits=c(1,22))+
  scale_y_continuous(limits=c(0,1),breaks=seq(0,1,.25),
                     labels=scales::percent_format(1),expand=c(0,0))+
  labs(title="c   Rank of the annotated 3' end among candidate sites",
       x="rank threshold",y="genes at or above threshold")+
  theme_classic(base_size=BASE)+
  theme(text=element_text(colour=INK),legend.position="none",
        plot.title=element_text(face="bold",size=BASE+0.5,margin=margin(b=6)),
        axis.line=element_line(colour=INK,linewidth=.3),
        axis.ticks=element_line(colour=INK,linewidth=.3),
        axis.title=element_text(colour=GTX,size=BASE-1),
        plot.background=element_rect(fill=PAPER,colour=NA),plot.margin=margin(6,8,6,8))
CAP<-paste0(
 "(a) The baseline scores ten core polyadenylation motifs, the polyadenylation signal in three ",
 "variants, four downstream elements, UGUA, poly-U and NUDT, each scanned in 30 nt windows at ",
 "stride 10 across a 500 nt span. The extended models read 2,500 nt, [-2000, +500) in transcript ",
 "orientation, and add splice context and learned sequence. The magnified row is that 500 nt span; ",
 "element positions within it are schematic, while the spans and the motif list are not.\n",
 "(b) Genes are contested when the baseline ranks at least one competitor at or above the ",
 "annotated 3' end, which it cannot avoid: build_shortlist.py retains a competitor only if the ",
 "random forest, logistic regression and baseline CNN all score it at or above that site, so the ",
 "baseline is never first on a contested gene. The universe is the ",format(N_RES,big.mark=","),
 " genes it resolves plus the ",format(nA,big.mark=",")," contested genes whose candidates both ",
 "extended models score, ",format(N_UNI,big.mark=",")," in total; within it every ribbon closes. ",
 "Both models are leave-one-chromosome-out. Categories give which model places the annotated site ",
 "first. The ",sprintf("%.3f",P_ORAC)," assumes the better model is chosen per gene and is an ",
 "upper bound; a single combined ranking, the sum of the two scores z-scored within each gene, ",
 "gives ",sprintf("%.3f",P_COMB),".\n",

 "Scope. Across all ",format(N_CON,big.mark=",")," contested genes, scored against their full ",
 "candidate lists, the sequence CNN alone recovers ",sprintf("%.3f",N_REC/N_CON)," and lifts ",
 "precision at 1 over all ",format(N_ALL,big.mark=",")," genes from ",sprintf("%.3f",N_OK0/N_ALL),
 " to ",sprintf("%.3f",N_OK1/N_ALL),". The ",format(N_CON-nA,big.mark=",")," contested genes outside ",
 "(b) are neither harder nor unscored, the CNN ranking the annotated site first in 66.8% of them ",
 "against 68.1% elsewhere; they are excluded because every one of their competitors is an ",
 "annotated polyadenylation site or a reference terminal exon, leaving no contest once annotated ",
 "competitors are removed.")
wrapcap<-function(x,w=120) paste(unlist(lapply(strsplit(x,"\n")[[1]],
                   function(p) paste(strwrap(p,width=w),collapse="\n"))),collapse="\n")
library(patchwork)
p<-(pA/pB2)+plot_layout(heights=c(.78,1.5))+
  plot_annotation(caption=wrapcap(CAP),
   theme=theme(plot.caption=element_text(size=BASE-2.2,colour=GTX,hjust=0,lineheight=1.28,
                                         margin=margin(t=8)),
               plot.background=element_rect(fill=PAPER,colour=NA)))
ggsave("figure_rescue.pdf",p,width=7.2,height=6.6,device=cairo_pdf)
ggsave("figure_rescue.png",p,width=7.2,height=6.6,dpi=400,bg=PAPER)
cat(sprintf("[a] %s genes -> %s resolved + %s contested; CNN rescues %s, misses %s\n",
  format(N_ALL,big.mark=","),format(N_RES,big.mark=","),format(N_CON,big.mark=","),
  format(N_REC,big.mark=","),format(N_MISS,big.mark=",")))
cat(sprintf("[a] correct %s -> %s ; P@1 %.4f -> %.4f\n",format(N_OK0,big.mark=","),
  format(N_OK1,big.mark=","),N_OK0/N_ALL,N_OK1/N_ALL))
cat(sprintf("[b] rank<=1 baseline %.4f  splice %.4f  CNN %.4f  combined %.4f  (n=%s)\n",
  mean(RK$rank_baseline<=1),mean(RK$rank_splice<=1),mean(RK$rank_cnn<=1),
  mean(RK$rank_combo<=1),format(nrow(RK),big.mark=",")))
cat(sprintf("[b] rank<=3 baseline %.4f  splice %.4f  CNN %.4f  combined %.4f\n",
  mean(RK$rank_baseline<=3),mean(RK$rank_splice<=3),mean(RK$rank_cnn<=3),
  mean(RK$rank_combo<=3)))
cat(sprintf("[c] both %s (%.1f%%)  CNN only %s (%.1f%%)  splice only %s (%.1f%%)  neither %s (%.1f%%)\n",
  format(sum(sp1&cn1),big.mark=","),100*mean(sp1&cn1),format(sum(!sp1&cn1),big.mark=","),
  100*mean(!sp1&cn1),format(sum(sp1&!cn1),big.mark=","),100*mean(sp1&!cn1),
  format(sum(!sp1&!cn1),big.mark=","),100*mean(!sp1&!cn1)))
cat(sprintf("[c] genome-wide total: combined %.4f, either-model upper bound %.4f\n",TOT_CB,TOT_OR))
cat("[out] figure_rescue.pdf / .png\n")
