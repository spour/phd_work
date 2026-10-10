#!/usr/bin/env Rscript
# figure3_v3_2 summary panel: descriptive stratification by three cached binary terms
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork)})
rdp<-function(f){ if(!file.exists(f)) stop("missing: ",f); read_tsv(f,show_col_types=FALSE) }
INK<-"#1A1A1A"; BASE<-13; CAN<-"#0072B2"; COMP<-"#E69F00"; DEC<-"#6A51A3"; NUL<-"#8A8F98"
th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-4),axis.line=element_line(color=INK,linewidth=.4),
  plot.title=element_text(size=BASE,face="bold",hjust=0),plot.subtitle=element_text(size=BASE-5,color="#555555"),
  legend.position="none")
G8<-rdp("v32_sum_grid8.tsv"); G4<-rdp("v32_sum_grid4.tsv"); WF<-rdp("v32_sum_waterfall.tsv")
WG<-rdp("v32_sum_withingene.tsv"); MT<-rdp("v32_sum_meta.tsv")
br<-MT$value[MT$key=="base_rate"]; NC<-MT$value[MT$key=="n_canonical"]; NK<-MT$value[MT$key=="n_competitor"]
phi<-MT$value[MT$key=="phi_donor_u1"]; pear<-MT$value[MT$key=="pearson_first5dn_dens525"]

# p1: 8-cell
G8<-G8%>%arrange(p_real)%>%mutate(label=factor(label,levels=label),fav=factor(favourable,levels=c(1,0),labels=c("all favourable","other")))
p1<-ggplot(G8,aes(p_real,label,fill=fav))+
  geom_vline(xintercept=br,linetype=2,color=INK,linewidth=.4)+
  geom_col(width=.68,color=INK,linewidth=.2)+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),width=.25,linewidth=.4)+
  geom_text(aes(x=ci_hi,label=sprintf("  %.1f%%  (n=%s)",100*p_real,format(n,big.mark=","))),hjust=0,size=2.6,color=INK)+
  scale_fill_manual(values=c("all favourable"=CAN,"other"=NUL))+
  scale_x_continuous(labels=scales::percent,expand=expansion(mult=c(.01,.30)))+
  labs(title="A   Eight cells: P(real end) per acceptor / donor / U1 combination",
       subtitle=sprintf("Strong acceptor = 3'ss MaxEnt >=8; donor = strong 5'ss in (anchor, pA); U1 = dens_5.25 above/below median. Pooled, sorted. Dashed = base rate %.1f%%.",100*br),
       x="P(real end | cell)",y=NULL)+th

# p2: 4-cell acceptor x donor
G4<-G4%>%arrange(p_real)%>%mutate(label=factor(label,levels=label),fav=factor(favourable,levels=c(1,0),labels=c("favourable","other")))
p2<-ggplot(G4,aes(p_real,label,fill=fav))+
  geom_vline(xintercept=br,linetype=2,color=INK,linewidth=.4)+
  geom_col(width=.6,color=INK,linewidth=.2)+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),width=.22,linewidth=.4)+
  geom_text(aes(x=ci_hi,label=sprintf("  %.1f%% (n=%s)",100*p_real,format(n,big.mark=","))),hjust=0,size=2.6,color=INK)+
  scale_fill_manual(values=c("favourable"=CAN,"other"=NUL))+
  scale_x_continuous(labels=scales::percent,expand=expansion(mult=c(.01,.32)))+
  labs(title="B   Acceptor x donor only (no U1)",subtitle="The same stratification dropping the U1 term.",
       x="P(real end | cell)",y=NULL)+th

# p3: waterfall
WF$step<-factor(WF$step,levels=WF$step)
p3<-ggplot(WF,aes(step,p_real,group=1))+
  geom_hline(yintercept=br,linetype=2,color=NUL,linewidth=.4)+
  geom_line(color=CAN,linewidth=.8)+geom_point(color=CAN,size=3)+
  geom_errorbar(aes(ymin=ci_lo,ymax=ci_hi),width=.15,linewidth=.4,color=CAN)+
  geom_text(aes(label=sprintf("%.1f%%\nn=%s",100*p_real,format(n,big.mark=","))),vjust=-0.5,size=2.6,color=INK)+
  scale_y_continuous(labels=scales::percent,expand=expansion(mult=c(.05,.25)))+
  labs(title="C   Waterfall (reading order)",subtitle="Base rate, then add each favourable term. P(real) and n surviving.",
       x=NULL,y="P(real end)")+th+theme(axis.text.x=element_text(angle=18,hjust=1,size=BASE-5))

# p4: within-gene
WG<-WG%>%mutate(cell=factor(cell,levels=cell))
p4<-ggplot(WG,aes(frac_real,cell))+
  geom_vline(xintercept=0.5,linetype=3,color=NUL)+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),width=.2,linewidth=.5,color=CAN)+
  geom_point(size=3.2,color=CAN)+
  geom_text(aes(x=ci_hi,label=sprintf("  %.3f (n=%s genes)",frac_real,format(n_genes,big.mark=","))),hjust=0,size=2.7,color=INK)+
  scale_x_continuous(expand=expansion(mult=c(.05,.45)))+
  labs(title="D   Within-gene: is the unique favourable peak the real end?",
       subtitle="Genes with EXACTLY ONE peak in the cell; fraction where that peak is the real end (bootstrap CI). Adding terms\nrestricts to different, larger gene sets and does not raise the fraction: within gene the acceptor term carries it.",
       x="P(the one favourable peak is the real end)",y=NULL)+th

cap<-paste0("Descriptive stratification of the SAME peaks (base rate ",sprintf("%.2f%%",100*br),", ",
  format(NC,big.mark=",")," canonical / ",format(NK,big.mark=",")," competitor), NOT a held-out classifier.\n",
  "U1 is included only as a compartment proxy: the matched intronic-decoy control showed pA-proximal U1 is not a site-level suppression mark\n",
  "(competitor vs matched intronic decoy, u1_excess win-rate 0.50), so the U1 term is not mechanistic. The donor and U1 terms are correlated\n",
  sprintf("(same GT-anchored 9-mer family: phi=%.2f, Pearson=%.2f on the continuous features), so they are not independent evidence.\n",phi,pear),
  "Binomial (Wilson) CIs in A-C; bootstrap-over-genes CIs in D.")
fig<-(p1/(p2|p3)/p4)+plot_layout(heights=c(1.25,1.0,0.85))+
  plot_annotation(title="Figure 3 (v3_2) summary: stratifying the real 3' end by acceptor, intervening donor, and U1",
    caption=cap,theme=theme(plot.title=element_text(face="bold",size=BASE+1,hjust=0,color=INK),
      plot.caption=element_text(size=BASE-5,color="#666666",hjust=0)))
ggsave("figure3_v3_2_summary.pdf",fig,width=14,height=15,device=cairo_pdf)
ggsave("figure3_v3_2_summary.png",fig,width=14,height=15,dpi=200)
cat("[out] figure3_v3_2_summary.pdf / .png\n")
