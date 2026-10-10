#!/usr/bin/env Rscript
# what the model does to a 5' splice site when allowed to rewrite it
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(tidyr);library(readr)
                               library(patchwork);library(ggseqlogo)})
INK<-"#1A1A1A"; MOD<-"#08519C"; U1C<-"#E8A33D"; NUL<-"#8A8F98"; BASE<-7
th<-theme_classic(base_size=BASE,base_family="sans")+theme(
  plot.title=element_text(size=BASE+1,face="bold",colour=INK),
  plot.subtitle=element_text(size=BASE-0.5,colour=NUL),
  axis.text=element_text(colour=INK),axis.title=element_text(colour=INK),
  strip.background=element_blank(),strip.text=element_text(size=BASE-0.5,colour=INK),
  legend.key.size=unit(7,"pt"),legend.text=element_text(size=BASE-1),
  legend.title=element_text(size=BASE-1),plot.tag=element_text(size=BASE+2,face="bold"))
TR<-read_tsv("evolve_seeded_traj.tsv",show_col_types=FALSE)
FN<-read_tsv("evolve_seeded_final.tsv",show_col_types=FALSE)
PW<-read_tsv("evolve_seeded_pwm.tsv",show_col_types=FALSE)
UP<-read_tsv("u1_pwm.tsv",show_col_types=FALSE)
AL<-c(u1seed="Free to change all 9 positions",u1gt="GT pinned: still a legal donor")
S9<-TR%>%filter(arm %in% names(AL))
stopifnot(nrow(FN%>%filter(arm %in% names(AL)))==128)

# a  the model score rises while the duplex it was built from collapses
MX<-max(S9$step)
CAR<-S9%>%group_by(arm,rep)%>%complete(step=0:MX)%>%
  fill(delta_logit,u1_dg,.direction="down")%>%ungroup()
Ap<-CAR%>%pivot_longer(c(delta_logit,u1_dg),names_to="q",values_to="v")%>%
  group_by(arm,q,step)%>%summarise(m=mean(v),lo=quantile(v,.1),hi=quantile(v,.9),.groups="drop")%>%
  mutate(arm=AL[arm],q=factor(c(delta_logit="Model score, change in logit",
                                u1_dg="U1 duplex, -dG")[q],
         levels=c("Model score, change in logit","U1 duplex, -dG")))
pA<-ggplot(Ap,aes(step,m,colour=q,fill=q))+
  geom_hline(yintercept=0,colour=NUL,linewidth=.3)+
  geom_ribbon(aes(ymin=lo,ymax=hi),alpha=.16,colour=NA)+geom_line(linewidth=.65)+
  facet_grid(q~arm,scales="free_y",switch="y")+
  scale_colour_manual(values=c(MOD,U1C),guide="none")+
  scale_fill_manual(values=c(MOD,U1C),guide="none")+
  scale_x_continuous(breaks=seq(0,MX,2))+
  labs(title="Rewriting a splice site to please the model destroys the splice site",
       subtitle="64 replicates per arm, each on its own draw of backgrounds. Band is the 10th to 90th percentile",
       x="Substitutions accepted",y=NULL)+th+
  theme(strip.placement="outside",strip.text.y.left=element_text(angle=90,size=BASE-1))

# b  no replicate trades duplex for score
FS<-FN%>%filter(arm %in% names(AL))%>%mutate(arm=AL[arm])
Z0<-S9%>%filter(step==0)%>%summarise(m=mean(delta_logit))%>%pull(m)
SEED<-tibble(u1_dg=11.70,delta_logit=Z0)
pB<-ggplot(FS,aes(u1_dg,delta_logit,colour=arm))+
  geom_point(size=1.3,alpha=.75)+
  geom_point(data=SEED,aes(u1_dg,delta_logit),inherit.aes=FALSE,shape=8,size=2.4,
             colour=U1C,stroke=.7)+
  geom_text(data=SEED,aes(u1_dg,delta_logit,label="CAGGTGAGT, the duplex optimum"),
            inherit.aes=FALSE,hjust=1.1,vjust=-1.1,size=2.05,colour=U1C)+
  scale_colour_manual(values=c(MOD,"#6BAED6"),name=NULL)+
  labs(title="The endpoints keep none of the duplex they started with",
       subtitle="One point per replicate, at the sequence hill climbing stopped on",
       x="U1 duplex of the final 9-mer, -dG",y="Change in logit")+th+
  theme(legend.position=c(.62,.22),legend.background=element_blank())

# c  the logos: what it started as, what it became
lmat<-function(d){m<-t(as.matrix(d[,c("A","C","G","T")])); rownames(m)<-c("A","C","G","T"); m}
L<-list(lmat(UP%>%arrange(offset)),
        lmat(PW%>%filter(arm=="u1seed")%>%arrange(offset)),
        lmat(PW%>%filter(arm=="u1gt")%>%arrange(offset)),
        lmat(PW%>%filter(arm=="rand9")%>%arrange(offset)))
ic<-function(m){p<-m; 2+colSums(ifelse(p>0,p*log2(pmax(p,1e-12)),0))}
names(L)<-c(sprintf("Start: 115 near-optimal U1 9-mers, mean %.2f bits",mean(UP$ic)),
            sprintf("Evolved, free: mean %.2f bits",mean(PW$ic[PW$arm=="u1seed"])),
            sprintf("Evolved, GT pinned: mean %.2f bits",mean(PW$ic[PW$arm=="u1gt"])),
            sprintf("Evolved from random 9-mers: mean %.2f bits",mean(PW$ic[PW$arm=="rand9"])))
stopifnot(all(sapply(L,ncol)==9))
pC<-ggseqlogo(L,method="bits",ncol=1)+
  scale_y_continuous(limits=c(0,2.02),breaks=c(0,1,2),expand=c(0,0))+
  theme_classic(base_size=BASE)+
  theme(text=element_text(colour=INK),legend.position="none",strip.background=element_blank(),
        strip.text=element_text(size=BASE-1.5,colour=INK,hjust=0,margin=margin(b=1,t=3)),
        axis.text.x=element_blank(),axis.ticks.x=element_blank(),axis.line.x=element_blank(),
        axis.text.y=element_text(size=BASE-2,colour=NUL),axis.title.x=element_blank())+
  labs(y="bits",title="What it was, and what it becomes")+
  theme(plot.title=element_text(size=BASE+1,face="bold",colour=INK))

TOP<-FN%>%filter(arm=="u1gt")%>%count(sequence,sort=TRUE)%>%slice(1)
FIG<-(pA/(pB|pC))+plot_layout(heights=c(1,1.15))+
  plot_annotation(tag_levels="A",
    title="Evolving the 5' splice site itself",
    caption=paste0(
"Hill climbing on the model's own score, started from CAGGTGAGT, the 9-mer with the strongest U1 duplex of all 262,144. 64 replicates per arm; because greedy\n",
"search from one seed is deterministic, each replicate draws its own 64 held-out backgrounds, so the spread is real context variation and not injected noise.\n",
"A The model score rises from ",sprintf("%.3f",Z0)," to +0.400 while the duplex the seed was chosen for falls from 11.70 to 2.09. Both are measured at the fixed\n",
"  evolution position; the -0.006 the dose panel reports for this 9-mer is a maximum over 60 positions. The two curves separate from the first substitution,\n",
"  so the model is not repairing a splice site, it is leaving one. B No replicate ends anywhere near the duplex it started from.\n",
"C The starting matrix is a positional motif and so is the endpoint, but the endpoint's sharpness is an artefact of the shared seed, not a 9-mer motif in the\n",
"  model: the fourth row evolves 9-mers from RANDOM starts and lands at 0.06 bits per column, flat. The dominant endpoint is ",
TOP$sequence[1],",\n  reached by ",TOP$n[1]," of 64 replicates with GT pinned and 33 of 64 with all nine positions free. It contains GTGTAC, and TTGTAC and TGTACA are the two\n",
"  most activating 6-mers of all 4,096 in the exhaustive scan. The GT survives because TGTAC contains one, not because the model is scoring a donor.\n",
"  Pinning GT costs nothing: +0.397 against +0.400 free, so a sequence that is still a legal 5' splice site reaches the same score, by ceasing to be a good one."),
    theme=theme(plot.title=element_text(size=BASE+3,face="bold",colour=INK,family="sans"),
                plot.caption=element_text(size=BASE-1.6,colour=INK,hjust=0,family="sans"),
                plot.tag=element_text(size=BASE+2,face="bold")))
ggsave("figure_u1_rewrite.pdf",FIG,width=10.4,height=8.2,device=cairo_pdf)
ggsave("figure_u1_rewrite.png",FIG,width=10.4,height=8.2,dpi=200)
cat("[out] figure_u1_rewrite.pdf figure_u1_rewrite.png\n")
