#!/usr/bin/env Rscript
# calibrated architecture rules (maxent only)
suppressMessages({library(ggplot2);library(dplyr);library(readr);library(tidyr);library(patchwork);library(scales)})
INK<-"#1A1A1A"; BASE<-13; CAN<-"#0072B2"; COMP<-"#E69F00"; NUL<-"#8A8F98"; BR<-0.09308
th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-4),axis.line=element_line(color=INK,linewidth=.4),
  plot.title=element_text(size=BASE,face="bold"),plot.subtitle=element_text(size=BASE-5,color="#555555"),
  legend.title=element_text(size=BASE-6))
rdp<-function(f) read_tsv(f,show_col_types=FALSE)

# a: 2d grid
GR<-rdp("arch_grid2d.tsv"); RF<-rdp("arch_grid2d_refs.tsv")
lim<-max(abs(GR$p_real-BR))
rd<-RF%>%filter(axis=="donor"); ra<-RF%>%filter(axis=="acceptor")
pA<-ggplot(GR,aes(x_donor,y_acceptor))+
  geom_tile(aes(fill=p_real,alpha=n),color="grey85",linewidth=.1)+
  scale_fill_gradient2(low=COMP,mid="white",high=CAN,midpoint=BR,limits=c(BR-lim,BR+lim),
    labels=percent,name="P(real end)")+
  scale_alpha_continuous(range=c(0.35,1),trans="sqrt",name="n peaks")+
  geom_vline(data=rd,aes(xintercept=score),linetype=2,color=INK,linewidth=.35)+
  geom_hline(data=ra,aes(yintercept=score),linetype=2,color=INK,linewidth=.35)+
  geom_text(data=rd,aes(x=score,y=14.5,label=paste0(sensitivity,"%")),size=2.5,color=INK,angle=90,vjust=-0.3,hjust=1)+
  geom_text(data=ra,aes(x=12.3,y=score,label=paste0(sensitivity,"%")),size=2.5,color=INK,vjust=-0.3,hjust=1)+
  labs(title="A   Calibrated 2D rule: acceptor x intervening donor",
       subtitle="Fill = P(real end) vs 9.31% base; alpha = n (cells n>=200). Dashed = MaxEnt 50/75/90% sensitivity scores.\nReal ends live top-left: strong acceptor, weak intervening donor. Both axes are MaxEnt bits.",
       x="best intervening 5'ss in (anchor, pA)  (MaxEnt)",y="anchor 3'ss  (MaxEnt)")+th

# b: classes
CL<-rdp("arch_rules.tsv")
CL$class<-factor(CL$class,levels=c("terminal-like","ambiguous","internal-like"))
CL$absence_rule<-factor(CL$absence_rule,levels=c("absence>=8","absence>=4.13"),
                        labels=c("absence 5'ss>=8 (specific, primary)","absence 5'ss>=4.13 (90% sens, too strict)"))
pB<-ggplot(CL,aes(class,p_real,fill=class))+
  geom_col(width=.65,color=INK,linewidth=.2)+
  geom_hline(yintercept=BR,linetype=2,color=INK,linewidth=.4)+
  geom_errorbar(aes(ymin=ci_lo,ymax=ci_hi),width=.2,linewidth=.4)+
  geom_text(aes(label=sprintf("%.1f%%\n(n=%s)",100*p_real,format(n,big.mark=","))),vjust=-0.3,size=2.5,color=INK)+
  facet_wrap(~absence_rule)+
  scale_fill_manual(values=c("terminal-like"=CAN,"ambiguous"=NUL,"internal-like"=COMP),guide="none")+
  scale_y_continuous(labels=percent,expand=expansion(mult=c(0,.22)))+
  labs(title="B   Classes (rule 1)",subtitle="Dashed = 9.31% base rate. Primary uses the specific (>=8) absence cutoff; 4.13 is too strict (disqualifies mediocre-donor terminal exons).",
       x=NULL,y="P(real end)")+th+theme(axis.text.x=element_text(angle=18,hjust=1),
       strip.background=element_rect(fill="grey92",color=NA),strip.text=element_text(size=BASE-5,face="bold"))

# c: ladder top-1 accuracy
LD<-rdp("arch_ladder.tsv"); LD$rule<-factor(LD$rule,levels=rev(LD$rule))
chance<-LD$top1_acc[grepl("base",LD$rule)]
pC<-ggplot(LD,aes(top1_acc,rule))+
  geom_vline(xintercept=chance,linetype=2,color=NUL,linewidth=.4)+
  geom_errorbar(aes(xmin=top1_lo,xmax=top1_hi),width=.25,linewidth=.5,color=CAN)+
  geom_point(size=3,color=CAN)+
  geom_text(aes(label=sprintf(" %.2f",top1_acc)),hjust=0,size=2.7,color=INK)+
  scale_x_continuous(labels=percent,expand=expansion(mult=c(.05,.15)))+
  labs(title="C   Rule ladder: per-gene top-1 accuracy",
       subtitle=sprintf("Is the rule's top-ranked peak the real end? Bootstrap CI over genes (dashed = chance %.2f, verified by within-gene label\npermutation). The acceptor takes top-1 from 0.27 to 0.325; terminal-like@8 adds a further +0.019 to 0.344. Rule (e) is tautological.",chance),
       x="top-1 accuracy (real end ranked first)",y=NULL)+th

# d: absence sweep (extended)
SW<-rdp("arch_absence_sweep.tsv")%>%filter(!is.na(p_real))
SWl<-SW%>%transmute(cut=as.numeric(donor_absence_cutoff),`P(real end)`=p_real,`top-1 accuracy`=top1_acc)%>%
  pivot_longer(-cut,names_to="metric",values_to="v")
CH<-0.2695; ASY<-0.3251
pD<-ggplot(SWl,aes(cut,v,color=metric))+
  geom_hline(yintercept=CH,linetype=3,color=NUL,linewidth=.4)+
  geom_hline(yintercept=ASY,linetype=2,color=COMP,linewidth=.35)+
  geom_vline(xintercept=8,linetype=2,color=INK,linewidth=.5)+
  geom_vline(xintercept=4.13,linetype=3,color=NUL,linewidth=.4)+
  annotate("text",x=8,y=0.20,label="primary 8\n(functional donor)",size=2.6,color=INK,hjust=-0.05)+
  annotate("text",x=4.13,y=0.155,label="4.13 (wrong)",size=2.4,color=NUL,hjust=1.05)+
  annotate("text",x=16,y=ASY,label="acceptor-only 0.325",size=2.4,color=COMP,hjust=1,vjust=-0.5)+
  annotate("text",x=16,y=CH,label="chance 0.270",size=2.4,color=NUL,hjust=1,vjust=-0.5)+
  geom_line(linewidth=.8)+geom_point(size=2.4)+
  scale_color_manual(values=c("P(real end)"=CAN,"top-1 accuracy"=COMP),name=NULL)+
  labs(title="D   Donor-absence cutoff sweep (rule 1), extended",
       subtitle="terminal-like = acc>=8 & no 5'ss above the cutoff. Top-1 accuracy peaks at 8 then decays to the acceptor-only\nasymptote (0.325); pooled P(real) peaks near 6. The 90%-sensitivity value (4.13) is not the optimum.",
       x="donor-absence cutoff (MaxEnt 5'ss)",y="value (top-1 acc / P(real))")+th+theme(legend.position="bottom")

cap<-paste0(
 "Provenance: anchor_3ss_pos is the argmax MaxEnt 3'ss in the upstream window (not GENCODE), so the acceptor is not circular and rule (e) is near-tautological (99.8%). Top-1 baseline is chance\n",
 "0.270 (gene-averaged 1/n_peaks), verified by within-gene label permutation (0.270 [0.260,0.280], 1000 perms); NOT the 9.31% pooled rate. Tie rule for top-1: among peaks satisfying the binary\n",
 "terminal-like rule, the one with the highest anchor-acceptor MaxEnt is taken; if none qualify, the highest-acceptor peak is taken. Cutoff: primary donor-absence = 8, chosen by maximizing per-gene\n",
 "top-1 on this data, AND coinciding with the independently calibrated >=8 from the GENCODE positive control (50% sensitivity, median annotated donor 8.46) -- two independent procedures selecting the\n",
 "same threshold. Reconciliation, above chance 0.270: acceptor +0.056 (to 0.325), donor-absence@>=8 +0.019 (to 0.344). The donor changes the pick in 1,267 of 6,583 genes (19%) and is correct in 695 of\n",
 "them (54.9%, excess +0.049), netting +0.019 -- the same magnitude as the within-gene ordered-geometry effect (+0.047, p=0.002): one effect measured two ways. Held-out (choose the cutoff on half the\n",
 "genes, evaluate on the other half, 20 splits): mean gain +0.016 [+0.010,+0.023] vs +0.019 full-data -- selection bias negligible. Top-3 paired diff +0.005 [-0.004,+0.014], not significant. MaxEnt only; no U1, no SpliceAI.")
fig<-(pA|pB)/(pC|pD)+plot_layout(heights=c(1.15,1))+
  plot_annotation(title="Architecture rules for the real 3' end, calibrated on annotated splice sites (MaxEnt only)",
    caption=cap,
    theme=theme(plot.title=element_text(face="bold",size=BASE+1,color=INK),plot.caption=element_text(size=BASE-6,color="#666666",hjust=0)))
ggsave("figure_architecture_rules.pdf",fig,width=16,height=13,device=cairo_pdf)
ggsave("figure_architecture_rules.png",fig,width=16,height=13,dpi=200)
cat("[out] figure_architecture_rules.pdf/.png\n")
