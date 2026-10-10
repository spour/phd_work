#!/usr/bin/env Rscript
# figure 3: what a model false positive is, vs a real 3' end
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(tidyr);library(readr);library(patchwork)})
rdp<-function(f){ if(!file.exists(f)) stop("missing: ",f); read_tsv(f,show_col_types=FALSE) }
INK<-"#1A1A1A"; BASE<-11
CANON<-"#0072B2"; COMP<-"#E69F00"; DEC<-"#9AA0A6"
KCOL<-c(canonical=CANON,competitor=COMP,decoy=DEC)
th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-3),axis.line=element_line(color=INK,linewidth=.4),
  axis.ticks=element_line(color=INK,linewidth=.4),
  plot.title=element_text(size=BASE+1,face="bold",hjust=0),
  plot.subtitle=element_text(size=BASE-3,color="#555555",lineheight=1.05),
  legend.title=element_blank(),legend.text=element_text(size=BASE-3),
  legend.key.height=unit(.75,"lines"),legend.key.width=unit(.75,"lines"))
NN<-rdp("fig3env_groupn.tsv"); nof<-setNames(NN$n_peaks,NN$group)
ncanon<-nof["FAILURE"]; ncomp<-nof["COMPETITOR"]; ndec<-nof["DECOY"]
fc<-function(n) formatC(n,big.mark=",",format="d")
onp<-function(lbl,x,y,hj=0.5,sz=BASE*0.27,col=INK,it=TRUE)
  annotate("text",x=x,y=y,label=lbl,size=sz,color=col,hjust=hj,fontface=if(it)"italic" else "plain")

# mirrored-violin helper: canonical (left), competitor (right) around a center
half<-function(vals,center,side,width=0.42,n=256){
  vals<-vals[is.finite(vals)]; d<-density(vals,n=n)
  w<-d$y/max(d$y)*width
  data.frame(x=c(center,center+side*w,center),y=c(d$x[1],d$x,d$x[length(d$x)]))
}
med_seg<-function(vals,center,side,width=0.42){
  m<-median(vals[is.finite(vals)]); data.frame(x0=center,x1=center+side*width*0.9,y=m)
}

# a: atlas overlap
A<-rdp("fig3env_panelA.tsv") %>% filter(group!="SUCCESS") %>%
  mutate(group=recode(group,FAILURE="canonical",COMPETITOR="competitor",DECOY="decoy"),
         group=factor(group,levels=c("canonical","competitor","decoy")))
cd<-rdp("fig3env_panelA_cmp.tsv") %>% filter(comparison=="COMPETITOR - DECOY")
xlabA<-c(canonical=sprintf("canonical\n(n=%s)",fc(ncanon)),competitor=sprintf("competitor\n(n=%s)",fc(ncomp)),
         decoy=sprintf("matched intronic\npositions (n=%s)",fc(ndec)))
pA<-ggplot(A,aes(group,any_atlas,fill=group))+
  geom_col(width=.68,color=INK,linewidth=.25)+
  geom_errorbar(aes(ymin=lo,ymax=hi),width=.2,linewidth=.4,color=INK)+
  geom_text(aes(y=any_atlas+.05,label=sprintf("%.0f%%",100*any_atlas)),size=BASE*0.3,color=INK)+
  scale_fill_manual(values=KCOL,guide="none")+scale_x_discrete(labels=xlabA)+
  scale_y_continuous(limits=c(0,1.16),breaks=seq(0,1,.25),expand=expansion(mult=c(0,0)))+
  annotate("segment",x=2,xend=3,y=1.05,yend=1.05,color=INK,linewidth=.4)+
  onp(sprintf("competitor - background  +%.2f [%.2f, %.2f]",cd$effect,cd$lo,cd$hi),2.5,1.11)+
  labs(title="a   Competitors rarely coincide with a real polyA site",
       subtitle="Fraction with any PolyASite / PolyA_DB site in the 500 nt window. Canonical is definitional.\nCompetitors overlap a real site 5.4x more than matched intronic positions, at minor usage (median 1%).",
       x=NULL,y="fraction with atlas site")+th

# b: fine compartment
LEV<-c("first exon","internal exon","last exon-3'UTR","first intron","internal intron","last intron","intergenic-other")
FILL<-c("first exon"="#A6CEE3","internal exon"="#4A90C2","last exon-3'UTR"="#0072B2",
        "first intron"="#FDD9A6","internal intron"="#FDAE6B","last intron"="#E6690B","intergenic-other"="#BDBDBD")
B<-rdp("fig3env_panelB_fine.tsv") %>% mutate(group=factor(group,levels=c("canonical","competitor")),class=factor(class,levels=LEV))
bc<-rdp("fig3env_panelB_fine_cmp.tsv")
xlabB<-c(canonical=sprintf("canonical\n(n=%s)",fc(ncanon)),competitor=sprintf("competitor\n(n=%s)",fc(ncomp)))
pB<-ggplot(B,aes(group,frac,fill=class))+geom_col(width=.68,color=INK,linewidth=.15)+
  scale_fill_manual(values=FILL,name=NULL,limits=LEV)+scale_x_discrete(labels=xlabB)+
  scale_y_continuous(limits=c(0,1.13),breaks=seq(0,1,.25),expand=expansion(mult=c(0,0)))+
  onp(sprintf("intronic:  competitor - canonical  +%.2f [%.2f, %.2f]",bc$effect,bc$lo,bc$hi),1.5,1.07)+
  labs(title="b   Competitors sit in introns, real ends in the terminal exon",
       subtitle="GENCODE compartment of the 500 nt window. Blue = exon, amber = intron. Competitors are 92%\nintronic (mostly internal / last introns); real ends are 89% terminal exon.",
       x=NULL,y="fraction of peaks")+th+theme(legend.position="right")

# kinetic data
KV<-rdp("fig3env_kinetic_vals.tsv"); KS<-rdp("fig3env_kinetic_stats.tsv")
gst<-function(m) KS %>% filter(metric==m)
pfmt<-function(m){ z<-abs(gst(m)$mw_z); if(z>37) "p < 1e-300" else sprintf("Mann-Whitney %s",gst(m)$mw_p) }

CM<-c("up_donor_dist","intron_len"); CX<-c(up_donor_dist=1,intron_len=2)
clab<-c(up_donor_dist="distance to\nupstream donor",intron_len="host intron\nlength")
polyC<-list(); segC<-list()
for(m in CM){ ctr<-CX[[m]]
  vc<-log10(pmin(pmax(KV$value[KV$metric==m & KV$group=="canonical"],30),40000))
  vk<-log10(pmin(pmax(KV$value[KV$metric==m & KV$group=="competitor"],30),40000))
  a<-half(vc,ctr,-1); a$group<-"canonical"; a$m<-m; b<-half(vk,ctr,1); b$group<-"competitor"; b$m<-m
  polyC[[length(polyC)+1]]<-rbind(a,b)
  s1<-med_seg(vc,ctr,-1); s1$group<-"canonical"; s2<-med_seg(vk,ctr,1); s2$group<-"competitor"
  segC[[length(segC)+1]]<-rbind(s1,s2) }
polyC<-bind_rows(polyC); segC<-bind_rows(segC)
ybr<-log10(c(50,100,500,1000,5000,20000)); ybl<-c("50","100","500","1,000","5,000","20,000")
pC<-ggplot(polyC,aes(x,y))+
  annotate("rect",xmin=-Inf,xmax=Inf,ymin=log10(63),ymax=log10(958),fill="#000000",alpha=.05)+
  annotate("segment",x=0.45,xend=2.55,y=log10(63),yend=log10(63),linetype=3,color=INK,linewidth=.3)+
  annotate("segment",x=0.45,xend=2.55,y=log10(958),yend=log10(958),linetype=3,color=INK,linewidth=.3)+
  onp("Tian 2007 window: 63-958 nt",1.5,log10(1120),0.5,BASE*0.24,"#555555")+
  geom_polygon(aes(group=interaction(m,group),fill=group),color=INK,linewidth=.2,alpha=.9)+
  geom_segment(data=segC,aes(x=x0,xend=x1,y=y,yend=y),color=INK,linewidth=.5,inherit.aes=FALSE)+
  scale_fill_manual(values=KCOL,guide="none")+
  scale_x_continuous(breaks=CX,labels=clab,limits=c(0.4,2.6))+
  scale_y_continuous(breaks=ybr,labels=ybl)+
  annotate("segment",x=0.65,xend=1.35,y=log10(14000),yend=log10(14000),color=INK,linewidth=.4)+
  annotate("segment",x=1.65,xend=2.35,y=log10(14000),yend=log10(14000),color=INK,linewidth=.4)+
  onp(sprintf("AUROC %.3f [%.3f, %.3f]\n%s",gst("up_donor_dist")$auroc,gst("up_donor_dist")$au_lo,gst("up_donor_dist")$au_hi,pfmt("up_donor_dist")),1,log10(30000),0.5,BASE*0.25)+
  onp(sprintf("AUROC %.3f [%.3f, %.3f]\n%s",gst("intron_len")$auroc,gst("intron_len")$au_lo,gst("intron_len")$au_hi,pfmt("intron_len")),2,log10(30000),0.5,BASE*0.25)+
  labs(title="c   The host intron is the same size, the donor the same distance",
       subtitle="Scan-derived host intron. Canonical (left) vs competitor (right). Competitors sit marginally closer to their\nupstream donor and in slightly shorter introns: the opposite of, or far weaker than, a kinetic-competition effect.",
       x=NULL,y="nucleotides (log scale)")+th

m<-"up_donor_score"
vc<-KV$value[KV$metric==m & KV$group=="canonical"]; vk<-KV$value[KV$metric==m & KV$group=="competitor"]
pd<-rbind(within(half(vc,1,-1),{group<-"canonical"}),within(half(vk,1,1),{group<-"competitor"}))
sd<-rbind(within(med_seg(vc,1,-1),{group<-"canonical"}),within(med_seg(vk,1,1),{group<-"competitor"}))
st<-gst(m)
pD<-ggplot(pd,aes(x,y))+
  geom_polygon(aes(group=group,fill=group),color=INK,linewidth=.2,alpha=.9)+
  geom_segment(data=sd,aes(x=x0,xend=x1,y=y,yend=y),color=INK,linewidth=.5,inherit.aes=FALSE)+
  scale_fill_manual(values=KCOL,guide="none")+
  scale_x_continuous(breaks=c(0.7,1.3),labels=c(sprintf("canonical\n(n=%s)",fc(ncanon)),sprintf("competitor\n(n=%s)",fc(ncomp))),limits=c(0.5,1.5))+
  coord_cartesian(ylim=c(8.4,13.5))+
  annotate("segment",x=0.65,xend=1.35,y=12.55,yend=12.55,color=INK,linewidth=.4)+
  onp(sprintf("within-gene paired, 500 perms: p = %.2f (n.s.)\nwin-rate %.3f [%.3f, %.3f]   AUROC %.3f [%.3f, %.3f]",
              st$perm_p,st$winrate,st$wr_lo,st$wr_hi,st$auroc,st$au_lo,st$au_hi),1,13.05,0.5,BASE*0.25)+
  labs(title="d   The upstream donor is not weaker",
       subtitle="MaxEnt 5'ss score of the scan-derived upstream donor (all >= 8.46 by definition). The Tian 2007\nprediction that competitors have weaker donors is not supported: the distributions are indistinguishable.",
       x=NULL,y="upstream donor score (MaxEnt 5'ss)")+th

design<-"AB\nCD"
fig<-pA+pB+pC+pD+plot_layout(design=design,heights=c(1,1.15))
ggsave("figure3_env.pdf",fig,width=14,height=11,device=cairo_pdf)
ggsave("figure3_env.png",fig,width=14,height=11,dpi=200)
cat("wrote figure3_env.pdf / .png\n")

relab<-function(d) d %>% filter(group!="SUCCESS") %>% mutate(group=recode(group,FAILURE="canonical",COMPETITOR="competitor"))
xl2<-c(canonical=sprintf("canonical\n(n=%s)",fc(ncanon)),competitor=sprintf("competitor\n(n=%s)",fc(ncomp)))
Cv<-relab(rdp("fig3env_panelD_vals.tsv")) %>% mutate(group=factor(group,levels=c("canonical","competitor")))
Cs<-relab(rdp("fig3env_panelD_stats.tsv")); auacc<-rdp("fig3env_panelD_cmp.tsv") %>% filter(comparison=="FAILURE > COMPETITOR")
pS1<-ggplot(Cv,aes(group,acc,fill=group))+geom_violin(scale="width",width=.75,color=INK,linewidth=.25,alpha=.9)+
  geom_point(data=Cs,aes(group,mean_acc),color=INK,size=1.2,inherit.aes=FALSE)+
  scale_fill_manual(values=KCOL,guide="none")+scale_x_discrete(labels=xl2)+coord_cartesian(ylim=c(-2,16))+
  onp(sprintf("AUROC %.3f [%.3f, %.3f]",auacc$effect,auacc$lo,auacc$hi),1.5,15,0.5)+
  labs(title="Real ends carry a stronger acceptor",x=NULL,y="anchor acceptor (MaxEnt 3'ss)")+th
Sw<-rdp("fig3env_panelE_sweep.tsv")
pS2<-ggplot(Sw,aes(factor(window_kb,levels=c(1,2,5,10,20)),win_rate,group=1))+
  geom_ribbon(aes(ymin=lo,ymax=hi),fill=CANON,alpha=.18)+geom_line(color=CANON,linewidth=.7)+geom_point(color=CANON,size=1.6)+
  geom_hline(yintercept=.5,linetype=2,color=INK,linewidth=.3)+coord_cartesian(ylim=c(.48,.66))+
  labs(title="Acceptors accumulate upstream (within-gene)",subtitle="up-minus-down strong 3'ss win-rate, 1-20 kb.",
       x="window (kb)",y="win-rate (real > competitor)")+th
extra<-c("dn_acc_dist","frac_pos","dn_acc_score")
elab<-c(dn_acc_dist="distance to downstream acceptor (nt, log)",frac_pos="position in intron (0 donor -> 1 acceptor)",dn_acc_score="downstream acceptor score")
plist<-lapply(extra,function(m){
  vc<-KV$value[KV$metric==m & KV$group=="canonical"]; vk<-KV$value[KV$metric==m & KV$group=="competitor"]
  tr<-if(grepl("dist",m)) function(z) log10(pmin(pmax(z,30),40000)) else function(z) z
  pp<-rbind(within(half(tr(vc),1,-1),{group<-"canonical"}),within(half(tr(vk),1,1),{group<-"competitor"}))
  ss<-rbind(within(med_seg(tr(vc),1,-1),{group<-"canonical"}),within(med_seg(tr(vk),1,1),{group<-"competitor"}))
  st<-gst(m)
  ggplot(pp,aes(x,y))+geom_polygon(aes(group=group,fill=group),color=INK,linewidth=.2,alpha=.9)+
    geom_segment(data=ss,aes(x=x0,xend=x1,y=y,yend=y),color=INK,linewidth=.5,inherit.aes=FALSE)+
    scale_fill_manual(values=KCOL,guide="none")+scale_x_continuous(breaks=c(0.7,1.3),labels=c("canonical","competitor"),limits=c(0.5,1.5))+
    labs(title=elab[[m]],subtitle=sprintf("AUROC %.3f [%.3f, %.3f]  win-rate %.3f (p=%s)",st$auroc,st$au_lo,st$au_hi,st$winrate,st$perm_p),x=NULL,y=NULL)+th+
    theme(plot.title=element_text(size=BASE-2,face="bold"),plot.subtitle=element_text(size=BASE-4))
})
figS<-(pS1+pS2)/(plist[[1]]+plist[[2]]+plist[[3]])+
  plot_annotation(title="Figure 3 - supplement",
    subtitle="Demoted acceptor / asymmetry evidence (real ends have a stronger acceptor and more upstream acceptors), and the\nkinetic quantities not shown in the main figure. All effects are small (AUROC 0.49-0.53).",
    theme=theme(plot.title=element_text(size=BASE+1,face="bold"),plot.subtitle=element_text(size=BASE-3,color="#555555")))
ggsave("figure3_env_supp.pdf",figS,width=13,height=8.5,device=cairo_pdf)
ggsave("figure3_env_supp.png",figS,width=13,height=8.5,dpi=200)
cat("wrote figure3_env_supp.pdf / .png\n")
