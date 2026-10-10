#!/usr/bin/env Rscript
# what lowers the model score, from the exhaustive 6-mer scan and the minimisation run
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(tidyr)
                               library(patchwork);library(scales);library(ggseqlogo)})
INK<-"#1A1A1A"; UPC<-"#08519C"; DNC<-"#B03A2E"; MOT<-"#E8A33D"; NUL<-"#8A8F98"; BASE<-7
th<-theme_classic(base_size=BASE,base_family="sans")+theme(
  plot.title=element_text(size=BASE+1,face="bold",colour=INK),
  plot.subtitle=element_text(size=BASE-0.5,colour=NUL),
  axis.text=element_text(colour=INK),axis.title=element_text(colour=INK),
  strip.background=element_blank(),strip.text=element_text(size=BASE-0.5,colour=INK),
  legend.key.size=unit(7,"pt"),legend.text=element_text(size=BASE-1),
  legend.title=element_text(size=BASE-1),plot.tag=element_text(size=BASE+2,face="bold"))
rd<-function(f) read.delim(f,check.names=FALSE)
KS<-rd("kmer_scan.tsv"); MC<-rd("evolve_min_contrast.tsv"); MS<-rd("evolve_min_summary.tsv")
EMO<-rd("evolved_motif_pwm.tsv"); EFI<-rd("evolved_motif_fit.tsv")
MT<-rd("evolve_min_traj.tsv"); UT<-rd("evolve_traj.tsv")
MK<-rd("evolve_min_kmers.tsv"); UK<-rd("evolve_kmers.tsv")
K1<-KS%>%filter(background=="real_canonical",position==-20)
stopifnot(nrow(K1)==4096)

# a  the two tails of the complete 6-mer space
fam<-function(k) ifelse(grepl("TGTA|TACA",k),"TGTA or TACA",
                 ifelse(nchar(gsub("[^C]","",k))>=3,"Three or more C",
                 ifelse(grepl("GAGA|AGAG",k),"GA repeat","Other")))
RP<-K1%>%arrange(delta_logit)%>%head(15)%>%mutate(end="Most repressive")
AC<-K1%>%arrange(desc(delta_logit))%>%head(15)%>%mutate(end="Most activating")
TL<-bind_rows(RP,AC)%>%
  mutate(kmer=factor(kmer,levels=c(rev(RP$kmer),AC$kmer[order(AC$delta_logit)])),f=fam(kmer),
         end=factor(end,levels=c("Most repressive","Most activating")))
stopifnot(nlevels(TL$kmer)==30)
pA<-ggplot(TL,aes(delta_logit,kmer,fill=f))+
  geom_vline(xintercept=0,colour=NUL,linewidth=.3)+
  geom_col(width=.72)+
  geom_point(aes(x=arrangement),shape=21,fill="white",colour=INK,size=1.1,stroke=.4,na.rm=TRUE)+
  facet_wrap(~end,ncol=1,scales="free_y")+
  scale_fill_manual(values=c(`TGTA or TACA`=UPC,`Three or more C`=DNC,
                             `GA repeat`="#7D3C98",Other=NUL),name=NULL)+
  labs(title="C runs lower the score most; TGTA raises it most",
       subtitle=paste("All 4,096 6-mers, real 3' end background at -20. Open point is the part",
                      "left after the exact composition control"),
       x="Change in logit",y=NULL)+th+
  theme(legend.position="bottom",legend.margin=margin(t=-4))+guides(fill=guide_legend(nrow=1))

# b  where the named elements sit in that space
NM<-c(CCCCCC="poly-C",GGGGGG="poly-G",GTAAGT="5'ss, U1 core",AATAAA="AATAAA, the PAS",
      TGTGTG="GU-rich element",ATTAAA="ATTAAA",TTTTTT="poly-T",TGTAAA="TGTA in context")
QQ<-quantile(K1$delta_logit,c(.05,.25,.5,.75,.95))
NP<-K1%>%filter(kmer %in% names(NM))%>%
  mutate(pct=sapply(delta_logit,function(v) 100*mean(K1$delta_logit<v)),
         lab=sprintf("%s, %s",kmer,NM[kmer]))%>%
  arrange(delta_logit)%>%mutate(lab=factor(lab,levels=lab))
stopifnot(nrow(NP)==length(NM))
pB<-ggplot(NP,aes(delta_logit,lab))+
  annotate("rect",xmin=QQ[1],xmax=QQ[5],ymin=-Inf,ymax=Inf,fill=NUL,alpha=.12)+
  annotate("rect",xmin=QQ[2],xmax=QQ[4],ymin=-Inf,ymax=Inf,fill=NUL,alpha=.16)+
  geom_vline(xintercept=QQ[3],colour=NUL,linetype="2222",linewidth=.35)+
  geom_segment(aes(x=QQ[3],xend=delta_logit,yend=lab),colour=DNC,linewidth=.4)+
  geom_point(colour=DNC,size=1.5)+
  geom_text(aes(label=sprintf("percentile %.1f",pct),
                hjust=ifelse(delta_logit>QQ[3],-0.28,1.28)),size=1.85,colour=INK)+
  scale_x_continuous(expand=expansion(mult=c(.30,.30)))+
  labs(title="The canonical polyadenylation signal is a median 6-mer to this model",
       subtitle="Named elements against all 4,096. Bands are the middle 50 and 90 percent, dashed line the median",
       x="Change in logit",y=NULL)+th

# c  both directions of evolution
ARM<-c(whole="Whole 2500 nt window",site="Cleavage site, -250 to 0")
Cp<-bind_rows(UT%>%mutate(dir="Maximise, from real competitors"),
              MT%>%mutate(dir="Minimise, from real 3' ends"))%>%
  filter(step>0)%>%mutate(arm=ARM[arm])
pC<-ggplot(Cp,aes(step,mean_gain,colour=dir,linetype=arm))+
  geom_hline(yintercept=0,colour=NUL,linewidth=.3)+
  geom_line(linewidth=.55)+
  scale_colour_manual(values=c(`Maximise, from real competitors`=UPC,
                               `Minimise, from real 3' ends`=DNC),name=NULL)+
  scale_linetype_manual(values=c(`Whole 2500 nt window`="22",
                               `Cleavage site, -250 to 0`="solid"),name=NULL)+
  labs(title="Repression is real but weaker than activation",
       subtitle="Same code path, sign flipped. Both directions stopped on budget, not convergence",
       x="Substitutions accepted",y="Change in logit")+th+
  theme(legend.position="right",legend.box="vertical",legend.spacing.y=unit(1,"pt"))

# d  what the two directions build
EL<-c(GGGG="GGGG",TGTA="TGTA",TTTTTT="poly-T",AATAAA="AATAAA")
cnt<-function(f,lab){k<-rd(f); sapply(names(EL),function(m)
  sum(k$n_evolved[grepl(m,k$kmer,fixed=TRUE)])/sum(k$n_evolved))}
Dp<-bind_rows(
  tibble(el=names(EL),v=cnt("evolve_kmers.tsv"),dir="Maximise"),
  tibble(el=names(EL),v=cnt("evolve_min_kmers.tsv"),dir="Minimise"))%>%
  mutate(el=factor(EL[el],levels=EL),dir=factor(dir,levels=c("Minimise","Maximise")))
pD<-ggplot(Dp,aes(v*100,el,fill=dir))+
  geom_col(position=position_dodge(width=.72),width=.66)+
  scale_fill_manual(values=c(Maximise=UPC,Minimise=DNC),name=NULL)+
  labs(title="The two directions build opposite elements",
       subtitle="Share of 6-mers in the evolved site windows containing each element",
       x="Percent of 6-mers in the window",y=NULL)+th+
  theme(legend.position=c(.78,.28),legend.background=element_blank())

LB<-c(acc_best="3'ss, MaxEnt optimum",acc_ppt="3'ss, poly-T tract",donor="5'ss, U1 optimum",
      AATAAA="AATAAA",evolved_6="Evolved 6-mer, CCCCCC",evolved_23="Evolved 23-mer")
UL<-c("Whole 2500 nt window","Cleavage site, -250 to 0","Evolved 23-mer","3'ss, MaxEnt optimum",
      "5'ss, U1 optimum","Evolved 6-mer, CCCCCC","AATAAA","3'ss, poly-T tract")
Ew<-MS%>%filter(control=="mono")%>%transmute(unit=ARM[arm],seq=evolved_gain,shuf=control_gain,ev=TRUE)
Em<-MC%>%filter(background=="real_canonical")%>%mutate(lab=LB[unit])%>%
  transmute(unit=lab,seq=delta_logit,shuf=mono_shuffle_mean,ev=grepl("Evolved",lab))
Ep<-bind_rows(Ew,Em)%>%mutate(unit=factor(unit,levels=rev(UL)),d=seq-shuf)
stopifnot(!any(is.na(Ep$unit)),nrow(Ep)==length(UL))
PL<-pseudo_log_trans(sigma=.02,base=10)
pE<-ggplot(Ep,aes(y=unit))+
  geom_vline(xintercept=0,colour=NUL,linewidth=.3)+
  geom_segment(aes(x=shuf,xend=seq,yend=unit,colour=ev),linewidth=.45)+
  geom_point(aes(x=shuf),shape=21,fill="white",colour=NUL,size=1.5,stroke=.45)+
  geom_point(aes(x=seq,colour=ev),size=1.7)+
  geom_text(aes(x=pmin(seq,shuf),label=ifelse(abs(d)<1e-9,"0, no other arrangement exists",
                                              sprintf("%+.2f",d)),colour=ev),
            hjust=1.12,size=1.85,show.legend=FALSE)+
  scale_x_continuous(trans=PL,breaks=c(-100,-10,-1,-.1,0),
                     labels=c("-100","-10","-1","-0.1","0"),limits=c(-260,.3))+
  scale_colour_manual(values=c(`TRUE`=DNC,`FALSE`=MOT),guide="none")+
  labs(title="Every real element sits at or above a shuffle of its own bases",
       subtitle="Filled point is the sequence, open point its mononucleotide shuffle, number is the difference",
       x="Change in logit, pseudo-log axis",y=NULL)+th

lmat<-function(d){m<-t(as.matrix(d[,c("A","C","G","T")])); rownames(m)<-c("A","C","G","T"); m}
MSET<-c(activating="Activating",repressive="Repressive")
MM<-lapply(names(MSET),function(k) lmat(EMO%>%filter(set==k)%>%arrange(offset)))
FT<-EFI%>%filter(chosen==1)
nm<-sapply(names(MSET),function(k){m<-MM[[match(k,names(MSET))]]; f<-FT%>%filter(set==k)
  sprintf("%s: %s, %.1f bits against a shuffled null of %.1f",MSET[k],
          paste(rownames(m)[apply(m,2,which.max)],collapse=""),f$relative_entropy[1],f$null_mean[1])})
names(MM)<-nm
stopifnot(nrow(FT)==2,all(sapply(MM,ncol)==FT$width[match(names(MSET),FT$set)]))
pMot<-ggseqlogo(MM,method="bits",ncol=1)+
  scale_y_continuous(limits=c(0,2.02),breaks=c(0,1,2),expand=c(0,0))+
  theme_classic(base_size=BASE)+
  theme(text=element_text(colour=INK),legend.position="none",strip.background=element_blank(),
        strip.text=element_text(size=BASE-1.5,colour=INK,hjust=0,margin=margin(b=1,t=3)),
        axis.text.x=element_blank(),axis.ticks.x=element_blank(),axis.line.x=element_blank(),
        axis.text.y=element_text(size=BASE-2,colour=NUL),axis.title.x=element_blank(),
        plot.title=element_text(size=BASE+1,face="bold",colour=INK),
        plot.subtitle=element_text(size=BASE-0.5,colour=NUL))+
  labs(y="bits",title="The elements themselves, once aligned",
       subtitle="De novo from the evolved windows, against their own base composition")

FIG<-(pA|(pB/pD/pMot))/(pC|pE)+plot_layout(heights=c(1.25,1))+
  plot_annotation(tag_levels="A",
    title="What lowers the model score, by enumeration and by evolution",
    caption=paste0(
"Two independent routes to the same answer, on the reported CNN.\n",
"A Every 6-mer implanted by the protocol the splice sites were given. C runs dominate the repressive tail and TGTA the activating tail. Because the scan is\n",
"  exhaustive, a k-mer's mononucleotide shuffles are the other members of its permutation class, so the composition control is exact rather than sampled;\n",
"  the four homopolymers have no other arrangement and carry no open point. B AATAAA sits at the 52nd percentile of all 4,096 and the U1 core at the 30th.\n",
"C Minimisation builds GGGG and removes TGTA and poly-T; maximisation does the reverse. Neither direction builds low-complexity sequence: entropy rises\n",
"  in both, 1.910 to 1.963 bits going down and 1.921 to 1.961 going up, and the longest homopolymer run shortens in both.\n",
"D Stacking the evolved windows by position gives 0.06 bits per column, because neither element has a fixed position and every replicate places one\n",
"  differently. Aligned, both are readable. The activating motif carries TGTAC, which panel A independently ranks at the top of all 4,096 6-mers, so\n",
"  evolution, enumeration and this alignment reach it three separate ways; it sits 2.7 null standard deviations above its shuffle, real but not\n",
"  overwhelming, against 8.1 for the repressive one. Relative entropy was still rising at the widest width searched, so both widths are floors.\n",
"E Minimisation reaches -62.9 over the whole window and -21.4 at the site, against +98.2 and +35.7 for maximisation from the mirror-image start.\n",
"F The same control as the activating figure, mirrored. Every real element sits at or above its own shuffle; the evolved windows and the evolved 23-mer\n",
"  fall below it. The most repressive 6-mer evolution found is CCCCCC, which the exhaustive scan independently ranks lowest of all 4,096, and which is\n",
"  its own mononucleotide shuffle, so no composition control for it exists."),
    theme=theme(plot.title=element_text(size=BASE+3,face="bold",colour=INK,family="sans"),
                plot.caption=element_text(size=BASE-1.6,colour=INK,hjust=0,family="sans"),
                plot.tag=element_text(size=BASE+2,face="bold")))
ggsave("figure_repression.pdf",FIG,width=11.4,height=11.0,device=cairo_pdf)
ggsave("figure_repression.png",FIG,width=11.4,height=11.0,dpi=200)
cat("[out] figure_repression.pdf figure_repression.png\n")
