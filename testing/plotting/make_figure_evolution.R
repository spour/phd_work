#!/usr/bin/env Rscript
# the positive control for the implantation series: what the cnn actually wants
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(tidyr)
                               library(patchwork);library(scales)})
INK<-"#1A1A1A"; EVO<-"#08519C"; EV2<-"#6BAED6"; MOT<-"#E8A33D"; NUL<-"#8A8F98"; BASE<-7
th<-theme_classic(base_size=BASE,base_family="sans")+theme(
  plot.title=element_text(size=BASE+1,face="bold",colour=INK),
  plot.subtitle=element_text(size=BASE-0.5,colour=NUL),
  axis.text=element_text(colour=INK),axis.title=element_text(colour=INK),
  strip.background=element_blank(),strip.text=element_text(size=BASE-0.5,colour=INK),
  legend.key.size=unit(7,"pt"),legend.text=element_text(size=BASE-1),
  legend.title=element_text(size=BASE-1),plot.tag=element_text(size=BASE+2,face="bold"))
rd<-function(f) read.delim(f,check.names=FALSE)

TR<-rd("evolve_traj.tsv"); CT<-rd("evolve_contrast.tsv")
SU<-rd("evolve_summary.tsv"); KM<-rd("evolve_kmers.tsv"); PW<-rd("evolve_pwm.tsv")
MOTIFS<-c("acc_best","acc_ppt","donor","AATAAA")
CEIL<-max(CT$delta_logit[CT$unit %in% MOTIFS])

# a  trajectory, against the ceiling every real motif ran into
ARM<-c(whole="Whole 2500 nt window",site="Cleavage site, -250 to 0")
Ap<-TR%>%filter(step>0)%>%mutate(arm=ARM[arm])
pA<-ggplot(Ap,aes(step,mean_gain,colour=arm))+
  geom_hline(yintercept=CEIL,linetype="2222",colour=MOT,linewidth=.4)+
  geom_line(linewidth=.6)+
  annotate("text",x=8,y=CEIL,label=sprintf("best real motif, %+.2f",CEIL),
           hjust=0,vjust=-0.6,size=2.05,colour=MOT)+
  scale_colour_manual(values=setNames(c(EVO,EV2),ARM),name=NULL)+
  scale_y_log10(breaks=c(.1,1,10,100),labels=c("0.1","1","10","100"))+
  annotation_logticks(sides="l",linewidth=.25,short=unit(1,"pt"),mid=unit(1.5,"pt"),long=unit(2.5,"pt"))+
  labs(title="Sequence can move this model by 100 logits; motifs moved it by 0.27",
       subtitle="Mean gain in logit over 64 held-out competitor windows. Neither arm has converged",
       x="Substitutions accepted",y="Gain in logit")+th+
  theme(legend.position=c(.68,.24),legend.background=element_blank())

# b  the contrast table
LB<-c(acc_best="3'ss, MaxEnt optimum",acc_ppt="3'ss, poly-T tract",donor="5'ss, U1 optimum",
      AATAAA="AATAAA",evolved_6="Evolved 6-mer",evolved_23="Evolved 23-mer")
BG<-c(real_canonical="Real 3' end background",real_competitor="Real competitor background")
Bp<-CT%>%mutate(unit=factor(LB[unit],levels=rev(LB)),bg=BG[background])%>%
  select(unit,bg,raw=delta_logit,comp=minus_composition)%>%
  pivot_longer(c(raw,comp),names_to="what",values_to="v")%>%
  mutate(what=factor(c(raw="Raw effect",comp="Minus its own base composition")[what],
                     levels=c("Raw effect","Minus its own base composition")),
         ev=grepl("Evolved",as.character(unit)))
PL<-pseudo_log_trans(sigma=.02,base=10)
pB<-ggplot(Bp,aes(v,unit,colour=ev))+
  geom_vline(xintercept=0,colour=NUL,linewidth=.3)+
  geom_segment(aes(x=0,xend=v,yend=unit),linewidth=.45)+geom_point(size=1.3)+
  geom_text(aes(label=sprintf("%+.2f",v)),hjust=ifelse(Bp$v>0,-0.25,1.25),size=1.85,show.legend=FALSE)+
  facet_grid(what~bg)+
  scale_x_continuous(trans=PL,breaks=c(-.1,0,.1,1,10),limits=c(-.42,34))+
  scale_colour_manual(values=c(`TRUE`=EVO,`FALSE`=MOT),guide="none")+
  labs(title="The model's own sequence is the only unit that beats its base composition",
       subtitle="Max over 60 positions, 256 real backgrounds. Pseudo-log x axis",
       x="Change in logit",y=NULL)+th

# c  every unit against a shuffle of its own bases
UL<-c("Whole 2500 nt window","Cleavage site, -250 to 0","Evolved 23-mer","Evolved 6-mer",
      "3'ss, poly-T tract","AATAAA","3'ss, MaxEnt optimum","5'ss, U1 optimum")
Cw<-SU%>%filter(control=="mono")%>%
  transmute(unit=ARM[arm],seq=evolved_gain,shuf=control_gain,ev=TRUE)
Cm<-CT%>%filter(background=="real_competitor")%>%mutate(lab=LB[unit])%>%
  transmute(unit=lab,seq=delta_logit,shuf=mono_shuffle_mean,ev=grepl("Evolved",lab))
Cp<-bind_rows(Cw,Cm)%>%mutate(unit=factor(unit,levels=rev(UL)),d=seq-shuf)
stopifnot(!any(is.na(Cp$unit)),nrow(Cp)==length(UL))
pC<-ggplot(Cp,aes(y=unit))+
  geom_vline(xintercept=0,colour=NUL,linewidth=.3)+
  geom_segment(aes(x=shuf,xend=seq,yend=unit,colour=ev),linewidth=.45)+
  geom_point(aes(x=shuf),shape=21,fill="white",colour=NUL,size=1.5,stroke=.45)+
  geom_point(aes(x=seq,colour=ev),size=1.7)+
  geom_text(aes(x=pmax(seq,shuf),label=sprintf("%+.2f",d),colour=ev),
            hjust=-0.3,size=1.9,show.legend=FALSE)+
  scale_x_continuous(trans=PL,breaks=c(0,.1,1,10,100),
                     labels=c("0","0.1","1","10","100"),limits=c(-.16,520))+
  scale_colour_manual(values=c(`TRUE`=EVO,`FALSE`=MOT),guide="none")+
  labs(title="Only the model's own sequences beat a shuffle of their own bases",
       subtitle=paste("Filled point is the sequence, open point its mononucleotide shuffle,",
                      "number is the difference"),
       x="Gain in logit, pseudo-log axis",y=NULL)+th

# d  what it built
Dp<-KM%>%filter(n_evolved>=20)%>%arrange(desc(n_evolved))%>%head(15)%>%
  mutate(kmer=factor(kmer,levels=rev(kmer)),
         fam=factor(ifelse(grepl("TGTA",kmer),"Contains TGTA, the CFIm element",
                    ifelse(grepl("CCGA|CGGT",kmer),"Contains CCGA or CGGT","Other")),
                    levels=c("Contains TGTA, the CFIm element","Contains CCGA or CGGT","Other")))
pD<-ggplot(Dp,aes(log2_vs_shuffled,kmer,fill=fam))+
  geom_col(width=.7)+
  scale_fill_manual(values=setNames(c(EVO,EV2,NUL),levels(Dp$fam)),name=NULL)+
  labs(title="Dispersed short motifs, one of them a real cleavage element",
       subtitle="Top 6-mers by count, against a shuffle of the same evolved windows",
       x="log2 enrichment over a shuffle holding composition",y=NULL)+th+
  theme(legend.position="bottom",legend.margin=margin(t=-4))+guides(fill=guide_legend(nrow=1))

pE<-ggplot(PW,aes(offset,ic))+
  geom_hline(yintercept=2,colour=NUL,linetype="2222",linewidth=.3)+
  geom_line(colour=EVO,linewidth=.4)+
  annotate("text",x=-248,y=1.85,hjust=0,size=2,colour=NUL,
           label=sprintf("a fixed base would sit at 2 bits; the maximum reached is %.2f",max(PW$ic)))+
  scale_y_continuous(limits=c(0,2.1),breaks=c(0,1,2))+
  labs(title="No position is fixed: the code is compositional in k-mers, not positional",
       subtitle="Information content per position across the 64 independently evolved site windows",
       x="Position relative to the cleavage site (nt)",y="Bits")+th

FIG<-(pA|pB)/(pC|pD)/pE+plot_layout(heights=c(1,1,.45))+
  plot_annotation(tag_levels="A",
    title="What the model wants, found by evolving sequence against it",
    caption=paste0(
"In-silico evolution against the reported CNN, as the positive control for five rounds of motif implantation that moved it by at most +0.27.\n",
"A Greedy hill climbing on the logit from 64 held-out real competitor windows. The whole-window arm proposes substitutions by gradient and scores\n",
"  all 192 exactly; the site arm scores every one of the 1,000 substitutions in [-250, 0] at each step. Both were stopped on budget, not convergence,\n",
"  and were still gaining 0.16 and 0.12 logits per substitution at the last step. B The same implantation protocol the motifs were given. The evolved\n",
"  23-mer TTCCGACGGAGGACCGACCGAGT gains +9.76, against +0.03 for the MaxEnt-optimal acceptor, which is stronger than any acceptor in the data.\n",
"C The control that decides the question, in the real competitor background evolution started from. Every evolved sequence beats a shuffle of its own\n",
"  bases, the two evolved windows in 100% of sequences; every real splice site falls below its own shuffle. The whole-window arm also beats a within-bin\n",
"  shuffle that holds the positional composition profile, by +96.6. So the model does read arrangement. It does not read the arrangements splice sites\n",
"  make. Controls are mononucleotide throughout: a dinucleotide shuffle returns the poly-T tract unchanged in 10 draws of 10 and the U1 optimum in 4.\n",
"D TGTA, the CFIm binding element, rises from 1.55 to 7.25 copies per window, 183 times what survival of the starting sequence would give. AATAAA is\n",
"  lost at the rate plain overwriting predicts. E Evolution raised sequence entropy from 1.921 to 1.961 bits and shortened the longest homopolymer\n",
"  run from 5.9 to 5.4 nt, so what it built is not low-complexity, and no position carries as much as 0.3 of the 2 bits a fixed base would."),
    theme=theme(plot.title=element_text(size=BASE+3,face="bold",colour=INK,family="sans"),
                plot.caption=element_text(size=BASE-1.6,colour=INK,hjust=0,family="sans"),
                plot.tag=element_text(size=BASE+2,face="bold")))
ggsave("figure_evolution.pdf",FIG,width=11.4,height=11.2,device=cairo_pdf)
ggsave("figure_evolution.png",FIG,width=11.4,height=11.2,dpi=200)
cat("[out] figure_evolution.pdf figure_evolution.png\n")
