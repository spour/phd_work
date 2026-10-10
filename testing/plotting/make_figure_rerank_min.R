#!/usr/bin/env Rscript
# the minimal splicing reranker, one figure
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(tidyr);library(patchwork)})
INK<-"#1A1A1A"; BASE<-12; CAN<-"#0072B2"; COMP<-"#E69F00"; GRY<-"#8A8F98"; GRN<-"#1D9E75"
th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-3),axis.title=element_text(size=BASE-2),
  axis.line=element_line(color=INK,linewidth=.4),axis.ticks=element_line(color=INK,linewidth=.4),
  plot.title=element_text(size=BASE,face="bold",hjust=0),
  plot.subtitle=element_text(size=BASE-3,color="#555555",lineheight=1.05),
  legend.title=element_blank(),legend.text=element_text(size=BASE-3),
  legend.key.height=unit(.8,"lines"))
fc<-function(x) formatC(x,big.mark=",",format="d")
rd<-function(f) read_tsv(f,show_col_types=FALSE,comment="#")

MET<-rd("rerank_min_metrics.tsv")
g<-function(m,k) MET[[k]][MET$model==m]
HOLD<-g("final_hold","p1"); HLO<-g("final_hold","p1lo"); HHI<-g("final_hold","p1hi")
HCH<-g("final_hold","p1_chance"); HEX<-g("final_hold","p1_exc")
HEXLO<-g("final_hold","p1_exc_lo"); HEXHI<-g("final_hold","p1_exc_hi")
LOCO<-g("final_loco_all","p1"); LOCOCH<-g("final_loco_all","p1_chance")
NF<-MET$n_feat[1]
HEADLINE<-sprintf("holdout P@1 %.3f [%.3f, %.3f] vs chance %.3f, excess %+.3f [%+.3f, %+.3f]",
                  HOLD,HLO,HHI,HCH,HEX,HEXLO,HEXHI)

# a1 / a2 -- roc and pr, reranker vs rf baseline vs chance
CU<-rd("rerank_min_curves.tsv") %>% mutate(model=factor(model,levels=c("reranker","RF baseline")))
MODCOL<-c("reranker"=CAN,"RF baseline"=COMP)
BR<-g("final_loco_all","base_rate")
pa1<-ggplot(CU,aes(fpr,tpr,color=model))+
  geom_abline(slope=1,intercept=0,linetype=2,color=GRY,linewidth=.4)+
  geom_line(linewidth=.9)+scale_color_manual(values=MODCOL)+
  annotate("text",x=.97,y=.10,hjust=1,size=BASE*0.25,color="#555555",
           label=sprintf("AUROC  reranker %.3f [%.3f, %.3f]\n            RF baseline %.3f\n            chance 0.500",
                         g("final_loco_all","auroc"),g("final_loco_all","auroc_lo"),g("final_loco_all","auroc_hi"),
                         g("rf_baseline_all","auroc")))+
  labs(title="a   Ranking the true 3' end among its own competitors",
       subtitle="ROC over all contested genes, leave one chromosome out. Dashed = chance.\nThe RF sits below chance by construction: a competitor is listed only if it already outscored the true end.",
       x="false positive rate",y="true positive rate")+th
pa2<-ggplot(CU,aes(tpr,precision,color=model))+
  geom_hline(yintercept=BR,linetype=2,color=GRY,linewidth=.4)+
  geom_line(linewidth=.9)+scale_color_manual(values=MODCOL)+
  annotate("text",x=.97,y=.95,hjust=1,vjust=1,size=BASE*0.25,color="#555555",
           label=sprintf("AUPRC  reranker %.3f [%.3f, %.3f]\n            RF baseline %.3f\n            base rate %.3f (dashed)",
                         g("final_loco_all","auprc"),g("final_loco_all","auprc_lo"),g("final_loco_all","auprc_hi"),
                         g("rf_baseline_all","auprc"),BR))+
  labs(title="b   The same ranking at the base rate that matters",
       subtitle=sprintf("Precision-recall. One canonical per gene, so the base rate is %.3f.",BR),
       x="recall",y="precision")+th

# b -- loco lift per chromosome
CH<-rd("rerank_min_bychrom.tsv") %>%
  mutate(chrom=factor(chrom,levels=chrom[order(as.numeric(sub("chr","",sub("X","23",chrom))))]))
pb<-ggplot(CH,aes(p1,chrom))+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),orientation="y",width=.25,linewidth=.45,color=INK)+
  geom_point(aes(x=chance),size=1.9,shape=23,fill="white",color=GRY,stroke=.5)+
  geom_point(size=2.2,shape=21,fill=CAN,color=INK,stroke=.35)+
  geom_vline(xintercept=LOCO,linetype=2,color=CAN,linewidth=.4)+
  labs(title="c   Every chromosome held out in turn",
       subtitle=sprintf("Blue = P@1 for genes on that chromosome, fit on all others. Diamond = that chromosome's chance.\nPooled LOCO P@1 %.3f (dashed). The headline number is the gene holdout, quoted once above.",LOCO),
       x="P@1 (held out)",y=NULL)+th
# c -- contest difficulty
CO<-rd("rerank_min_bycontest.tsv") %>% mutate(bucket=factor(bucket,levels=c("1","2","3-4","5-8","9+")))
CL<-CO %>% select(bucket,n_genes,model=p1,chance) %>% pivot_longer(c(model,chance),names_to="src",values_to="v")
pc<-ggplot()+
  geom_col(data=CL,aes(bucket,v,fill=src),position=position_dodge(width=.7),width=.62,color=INK,linewidth=.2)+
  geom_errorbar(data=CO,aes(x=bucket,ymin=ci_lo,ymax=ci_hi),width=.14,linewidth=.45,color=INK,
                position=position_nudge(x=.175))+
  geom_text(data=CO,aes(bucket,ci_hi+.045,label=sprintf("%+.2f",lift)),size=BASE*0.26,color=INK)+
  geom_text(data=CO,aes(bucket,-.05,label=fc(n_genes)),size=BASE*0.22,color="#555555")+
  scale_fill_manual(values=c(model=CAN,chance=GRY),labels=c(chance="chance (1 / candidates)",model="reranker"))+
  scale_y_continuous(limits=c(-.08,.82),breaks=seq(0,.8,.2),expand=expansion(mult=0))+
  labs(title="d   The lift is concentrated in easy contests",
       subtitle="Correct rate by number of competitors.\nAbove bars = lift over chance, below = genes.",
       x="competitors in the gene",y="correct rate")+th

# d -- permutation null, matched protocol
PN<-rd("rerank_min_permnull.tsv")
pnraw<-readLines("rerank_min_permnull.tsv")
gv<-function(tag) as.numeric(sub(paste0("^#",tag,"\t"),"",grep(paste0("^#",tag,"\t"),pnraw,value=TRUE)))
OBS<-gv("observed"); PV<-gv("p"); NCH<-gv("chance")
pd<-ggplot(PN,aes(p1))+
  geom_histogram(bins=30,fill=GRY,color="white",linewidth=.2)+
  geom_vline(xintercept=OBS,color=CAN,linewidth=.9)+
  geom_vline(xintercept=NCH,linetype=2,color=INK,linewidth=.4)+
  annotate("text",x=OBS,y=Inf,vjust=1.6,hjust=1.05,size=BASE*0.25,color=CAN,
           label=sprintf("observed %.3f\np %s %.3f",OBS,ifelse(PV<=1/(nrow(PN)+1),"<","="),max(PV,1/(nrow(PN)+1))))+
  labs(title="e   Permutation null, same protocol",
       subtitle=sprintf("Canonical label shuffled within each gene, full LOCO refit, %d permutations.\nDashed = analytic chance %.3f.",nrow(PN),NCH),
       x="P@1 under the null",y="permutations")+th
OV<-rd("rerank_min_override.tsv") %>% mutate(outcome=factor(outcome,levels=rev(c(
  "both correct","model correct, RF wrong","RF correct, model wrong","both wrong"))))
pe<-ggplot(OV,aes(frac,outcome,fill=outcome))+geom_col(width=.62,color=INK,linewidth=.2)+
  geom_text(aes(label=sprintf("%.1f%%  (%s genes)",100*frac,fc(n_genes))),hjust=-0.06,size=BASE*0.25,color=INK)+
  scale_fill_manual(values=c("both correct"=GRN,"model correct, RF wrong"=CAN,
                             "RF correct, model wrong"=COMP,"both wrong"=GRY),guide="none")+
  scale_x_continuous(limits=c(0,max(OV$frac)*1.55),expand=expansion(mult=c(0,0)))+
  labs(title="f   What the override buys",
       subtitle="Model pick vs the RF pick, per gene, LOCO.\nThe RF is degenerate on contested genes by shortlist construction.",
       x="fraction of contested genes",y=NULL)+th
MG<-rd("rerank_min_margin.tsv") %>% filter(hit==0)
med<-median(MG$margin_frac); near<-mean(MG$margin_frac<=0.10); n25<-mean(MG$margin_frac<=0.25)
pf<-ggplot(MG,aes(margin_frac))+
  geom_histogram(bins=45,fill=COMP,color="white",linewidth=.2)+
  geom_vline(xintercept=med,linetype=2,color=INK,linewidth=.5)+
  annotate("text",x=-Inf,y=Inf,hjust=-0.05,vjust=1.4,size=BASE*0.25,color="#555555",
           label=sprintf("misses only, %s genes\nmedian gap %.2f of the range (dashed)\n%.1f%% within 10%% of the top, %.1f%% within 25%%",
                         fc(nrow(MG)),med,100*near,100*n25))+
  labs(title="g   Limitation: the failures are confident, not marginal",
       subtitle="Gap between the model's pick and the true 3' end as a fraction of the gene's score range, misses only.\nA near-tie defence does not hold: the median miss sits about half the range below the pick.",
       x="(top pick - true end) / within-gene score range",y="genes")+th
CF<-rd("rerank_min_coefs.tsv") %>% mutate(feature=reorder(feature,abs(beta_std)),
                                          sgn=ifelse(beta_std>0,"real end","competitor"))
pg<-ggplot(CF,aes(beta_std,feature,color=sgn))+
  geom_vline(xintercept=0,linetype=2,color=GRY,linewidth=.4)+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),orientation="y",width=.24,linewidth=.5)+
  geom_point(size=2.4)+
  geom_text(aes(x=ci_hi+0.03,label=sprintf("flip %.2f",sign_flip_frac)),hjust=0,size=BASE*0.22,color="#555555")+
  scale_color_manual(values=c("real end"=CAN,"competitor"=COMP),guide="none")+
  scale_x_continuous(expand=expansion(mult=c(.08,.28)))+
  labs(title="h   What the model actually uses",
       subtitle="Standardized logistic coefficients, dev fit, 95% CI bootstrapped over genes.\nBlue points to the real end, orange to the competitor; flip = resamples with the opposite sign.",
       x="standardized coefficient",y=NULL)+th
PA<-rd("rerank_min_path.tsv"); KS<-PA$k[PA$chosen==1][1]
PL<-bind_rows(PA %>% transmute(k,set="dev (LOCO)",p1=dev_p1,lo=dev_lo,hi=dev_hi),
              PA %>% transmute(k,set="untouched 20%",p1=hold_p1,lo=hold_lo,hi=hold_hi))
ph<-ggplot(PL,aes(k,p1,color=set,fill=set))+
  geom_hline(yintercept=PA$hold_chance[1],linetype=2,color=GRY,linewidth=.4)+
  geom_ribbon(aes(ymin=lo,ymax=hi),alpha=.16,color=NA)+geom_line(linewidth=.8)+geom_point(size=1.7)+
  geom_vline(xintercept=KS,linetype=3,color=INK,linewidth=.5)+
  annotate("point",x=KS,y=PA$hold_p1[PA$k==KS],size=3.6,shape=21,fill="white",color=INK,stroke=.7)+
  annotate("text",x=KS+0.15,y=min(PL$lo),hjust=0,vjust=0,size=BASE*0.25,color=INK,
           label=sprintf("chosen k = %d",KS))+
  scale_color_manual(values=c("dev (LOCO)"=GRY,"untouched 20%"=CAN))+
  scale_fill_manual(values=c("dev (LOCO)"=GRY,"untouched 20%"=CAN),guide="none")+
  scale_x_continuous(breaks=PA$k)+
  labs(title="j   Where the curve flattens",
       subtitle="Forward selection path. The holdout is flat from k = 2 to k = 12; 4 is the smallest k attaining the plateau.\nDashed = chance.",
       x="features in the model",y="P@1")+th

RC<-rd("rerank_robust_coefs.tsv") %>% filter(model %in% c("unrestricted","TERM-TERM")) %>%
  mutate(model=factor(ifelse(model=="TERM-TERM","TERM-TERM (compartment fixed)","unrestricted"),
                      levels=c("unrestricted","TERM-TERM (compartment fixed)")),
         feature=factor(feature,levels=rev(c("spl_upstream_max_me3","scan_n3ss_dn5k","spl_n_5ss_1k","spl_pas_prox_u1"))))
pi2<-ggplot(RC,aes(beta_std,feature,color=model))+
  geom_vline(xintercept=0,linetype=2,color=GRY,linewidth=.4)+
  geom_errorbar(aes(xmin=ci_lo,xmax=ci_hi),orientation="y",width=.22,linewidth=.5,
                position=position_dodge(width=.55))+
  geom_point(size=2.4,position=position_dodge(width=.55))+
  geom_text(data=RC %>% filter(model=="TERM-TERM (compartment fixed)"),
            aes(x=ci_hi+0.03,label=sprintf("flip %.2f",sign_flip_frac)),hjust=0,size=BASE*0.21,
            color="#555555",show.legend=FALSE)+
  scale_color_manual(values=c("unrestricted"=CAN,"TERM-TERM (compartment fixed)"="#7B3294"),name=NULL)+
  scale_x_continuous(expand=expansion(mult=c(.10,.30)))+
  labs(title="i   The acceptor is not a compartment detector",
       subtitle=paste0("Refit where canonical and competitor are BOTH terminal-exon (1,969 genes).\n",
                       "spl_upstream_max_me3 stays signed and CI-clean (+0.399 to +0.223, zero flips)\n",
                       "and the model still beats its chance rate (+0.083 excess, CI excludes zero).\n",
                       "scan_n3ss_dn5k collapses (-0.289 to -0.001, 50.3% flips): within 5 kb it reads\n",
                       "inside-the-gene-body, so it carries compartment. Density terms hold or strengthen."),
       x="standardized coefficient",y=NULL)+th

MS<-rd("rerank_min_misses.tsv")
GLEV<-c("correct pick","marginal miss","confident miss","competitor pool")
CO2<-MS %>% filter(section=="composition") %>%
  mutate(group=factor(group,levels=rev(GLEV)),
         key=factor(key,levels=c("first exon","internal exon","last exon-3'UTR",
                                 "first intron","internal intron","last intron","intergenic-other")),
         value=as.numeric(value))
CFILL<-c("first exon"="#A6CEE3","internal exon"="#4A90C2","last exon-3'UTR"="#0072B2",
         "first intron"="#FDD9A6","internal intron"="#FDAE6B","last intron"="#E6690B","intergenic-other"="#BDBDBD")
pj1<-ggplot(CO2,aes(value,group,fill=key))+geom_col(width=.66,color=INK,linewidth=.15,position=position_stack(reverse=TRUE))+
  scale_fill_manual(values=CFILL,name=NULL)+
  scale_x_continuous(expand=expansion(mult=c(0,0)),labels=scales::percent_format(accuracy=1))+
  labs(title="k   Where the wrong picks sit",
       subtitle=paste0("Compartment of the peak the model picked, by outcome. A dominant 3' end lies in the\n",
                       "terminal exon by what a terminal exon means, so this panel is descriptive; panel i is\n",
                       "where the architecture claim is tested. Wrong picks are intronic, but shifted toward\n",
                       "the terminal exon against the pool (0.16 vs 0.06), the compartment of real ends."),
       x="fraction of picks",y=NULL)+th
AT<-MS %>% filter(section=="atlas",key!="median_usage_if_matched") %>%
  mutate(group=factor(group,levels=rev(GLEV)),value=as.numeric(value),
         key=recode(key,matched_25nt="PolyASite match <= 25 nt",matched_usage_ge10="matched and usage >= 10%"))
pj2<-ggplot(AT,aes(value,group,fill=key))+
  geom_col(position=position_dodge(width=.72),width=.64,color=INK,linewidth=.15)+
  geom_text(aes(label=sprintf("%.2f",value)),position=position_dodge(width=.72),hjust=-0.15,size=BASE*0.22,color=INK)+
  scale_fill_manual(values=c("PolyASite match <= 25 nt"=GRN,"matched and usage >= 10%"=CAN),name=NULL)+
  scale_x_continuous(limits=c(0,1.12),expand=expansion(mult=c(0,0)))+
  labs(title="l   A resolution limit, not confusion",
       subtitle=paste0("The features encode whether a peak is a plausible 3' end. Confident misses are enriched\n",
                       "2.2x for an atlas match and 3.7x for usage >= 10% over the competitor pool: the model picks\n",
                       "peaks that look like real CPA sites, because many are, just not the dominant one.\n",
                       "Splice architecture cannot encode usage share; that is the boundary of this feature class."),
       x="fraction of picks",y=NULL)+th

fig<-(pa1|pa2|pb)/(pc|pd|pe)/(pf|pg|pi2)/(ph|pj1|pj2)+
  plot_layout(guides="collect")+
  plot_annotation(
    title=sprintf("A %d-feature splicing reranker picks the used 3' end inside its own shortlist",NF),
    subtitle=paste0(sprintf("Headline, from the untouched 20%% of genes, 4 features: %s.  ",HEADLINE),
                    "spl_upstream_max_me3 carries the model (standardized beta +0.40, about double the next feature,\n",
                    "tightest CI, zero sign flips in the bootstrap).\nLogistic regression, features standardized inside each fit, ridge 1.0, leave-one-chromosome-out. The holdout plateau is flat from k = 2 to k = 12 and 4 is the smallest k attaining it."),
    caption=paste0("Candidates are the shortlist rows of the 6,583 contested genes: one true 3' end per gene plus every intragenic region that outscored it under the baseline ensemble.\n",
                   "Panels c to i use LOCO over all contested genes, which is the per-chromosome protocol; the headline number above is the gene-holdout protocol and is the one to quote.\n",
                   "The two differ in which genes are held out, not in the model. Chance is the random floor, mean 1 / candidates per gene, and it moves with contest size, so lift is the comparable quantity.\n",
                   "The 2,167 confident misses are two populations and should not be summarised as one number: 758 land on an atlas site (281 of them used at 10% or more, near-miss alternative 3' ends,\n",
                   "median gap 1.00 of the score range) and 1,409 are unannotated, 92% intronic, backed with a median gap of 0.92. Both are confident; only the first is biologically defensible.\n",
                   "A prediction made before this panel, that misses would concentrate in annotated terminal-exon competitors, was wrong. The stratified analysis it came from measured where LIFT is small,\n",
                   "not where picks go wrong: terminal-exon competitors are 5.9% of the pool, so the model rarely picks one. About 1 in 8 confident misses is a genuinely used alternative site.\n",
                   "Restricting competitors to those with no PolyASite and no PolyA_DB annotation in the 500 nt window raises excess P@1 from +0.122 to +0.158 and tightens every coefficient:\n",
                   "against pure model artifacts the architecture signal is stronger, not weaker. The anchor is not a relabelled annotation either. 40.4% of canonical anchors sit within 2 nt of a\n",
                   "GENCODE acceptor and 59.6% do not (median distance 264 nt), versus 9.0% for competitors: the scan finds real acceptors preferentially, but most of the canonical signal comes\n",
                   "from acceptors GENCODE does not annotate at that position."),
    theme=theme(plot.title=element_text(face="bold",size=BASE+3,color=INK),
                plot.subtitle=element_text(size=BASE-3,color="#555555"),
                plot.caption=element_text(size=BASE-5,color="#666666",hjust=0)))&
  theme(legend.position="bottom")
ggsave("figure_rerank_min.pdf",fig,width=19,height=20,device=cairo_pdf)
ggsave("figure_rerank_min.png",fig,width=19,height=20,dpi=180)
cat(sprintf("wrote figure_rerank_min.pdf / .png  (%d features, holdout P@1 %.4f)\n",NF,HOLD))
