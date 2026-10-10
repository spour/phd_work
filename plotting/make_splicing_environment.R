#!/usr/bin/env Rscript
# the splicing environment of real 3' ends against their competitors
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork)})
rd<-function(f){ if(!file.exists(f)) stop("missing: ",f); read_tsv(f,show_col_types=FALSE) }
e<-function(...) cat(...,"\n",file=stderr())
INK<-"#1A1A1A"; BASE<-7
REAL<-"#08519C"; COMP<-"#E8A33D"; DEC<-"#7B5EA7"; GREY<-"#8A8F98"
LEV<-c("real","competitor","decoy")
LAB<-c(real="real 3' end",competitor="competitor",decoy="intronic decoy")
PAL<-setNames(c(REAL,COMP,DEC),LAB[LEV])
fc<-function(x) factor(x,levels=LEV,labels=LAB[LEV])
th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-1),axis.line=element_line(color=INK,linewidth=.3),
  axis.title=element_text(size=BASE),
  panel.grid.major.y=element_line(color="#E8EAED",linewidth=.2),
  plot.title=element_text(size=BASE+1,face="bold",hjust=0,margin=margin(b=4)),
  legend.title=element_blank(),legend.text=element_text(size=BASE-1),
  legend.key.height=unit(.7,"lines"),legend.key.width=unit(.9,"lines"),
  legend.margin=margin(0,0,0,0),legend.box.spacing=unit(2,"pt"),
  strip.background=element_blank(),strip.text=element_text(size=BASE-.5,color=INK))
nfmt<-function(x) format(as.numeric(x),big.mark=",")
ST<-rd("u1_5ss_stats.tsv")
sv<-function(k) suppressWarnings(as.numeric(ST$value[ST$key==k][1]))
WX<-rd("splice_env_wilcoxon.tsv"); wx<-function(k) WX$value[WX$key==k][1]
SS<-rd("splice_score_stats.tsv")
ssv<-function(k){ v<-SS$value[SS$key==k]; if(!length(v)) stop("missing stat: ",k)
                  suppressWarnings(as.numeric(v[1])) }
MG<-rd("splice_env_metagene.tsv")
DL<-rd("splice_env_gene_delta.tsv")
JD<-rd("splice_score_joint.tsv")
EX<-rd("splice_score_exceedance.tsv")
MEAN<-as.numeric(wx("mean_delta")); MED<-as.numeric(wx("median_delta"))
PW<-wx("p"); NGP<-as.numeric(wx("n_paired_genes"))
FBZ<-as.numeric(wx("frac_below_zero"))
NWX<-as.numeric(wx("n_genes"))
FOLD<-function(sc,t,h=2){ q<-EX%>%filter(class=="pooled",score==sc)
  f<-function(x) q$frac[which.min(abs(q$threshold-x))]
  (f(t-h)/f(t+h))^(1/(2*h)) }
SMOOTH<-FALSE
WIN<-as.character(wx("window"))
stopifnot(identical(WIN,"[-2000,+500]"), identical(sst_win<-SS$value[SS$key=="window"][1],WIN))
e("[read] one scan, window",WIN,": splice_env_metagene.tsv  splice_env_gene_delta.tsv",
  "splice_env_wilcoxon.tsv  splice_score_joint.tsv  splice_score_exceedance.tsv  splice_score_stats.tsv")
e("[win]  panels A, B, C and D all computed on",WIN,"-- asserted equal between the panel-B",
  "Wilcoxon file and the panel-C/D stats file")
e("[stat] Wilcoxon signed-rank V =",wx("V"),"on",wx("n_genes"),"genes, z =",wx("z"),
  ", p",PW,"(log10 p ~",wx("log10_p"),")")
e("[stat] window",WIN,"  mean per-gene delta",sprintf("%.2f /kb",MEAN),
  "  median",sprintf("%.2f /kb",MED),"  genes below 0:",FBZ)
e("[A]    panel A lines/ribbons:",if(SMOOTH) "3-bin rolling mean" else "raw")

# a: 5'ss density metagene
roll3<-function(x){ n<-length(x); sapply(seq_len(n),function(i){
  j<-max(1,i-1):min(n,i+1); mean(x[j]) }) }
panel_A<-function(){
  D<-MG%>%group_by(class)%>%arrange(offset,.by_group=TRUE)%>%
    mutate(y=if(SMOOTH) roll3(donor_per_kb) else donor_per_kb,
           lo=if(SMOOTH) roll3(donor_per_kb-se) else donor_per_kb-se,
           hi=if(SMOOTH) roll3(donor_per_kb+se) else donor_per_kb+se)%>%
    ungroup()%>%mutate(class=fc(class))
  pa<-MG%>%filter(offset==-50); v<-function(c) sprintf("%.2f",pa$donor_per_kb[pa$class==c])
  e("[A]    pA bin (-50 nt) raw: real",v("real"),"< competitor",v("competitor"),
    "< decoy",v("decoy"),"/kb")
  ggplot(D,aes(offset,y,color=class,fill=class))+
    annotate("rect",xmin=-Inf,xmax=0,ymin=-Inf,ymax=Inf,fill=GREY,alpha=.05)+
    geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.3)+
    geom_ribbon(aes(ymin=lo,ymax=hi),alpha=.18,color=NA,show.legend=FALSE)+
    geom_line(linewidth=.45)+
    scale_color_manual(values=PAL)+scale_fill_manual(values=PAL,guide="none")+
    scale_x_continuous(breaks=seq(-2000,500,500))+
    annotate("segment",x=-70,xend=-330,y=.552,yend=.60,color=GREY,linewidth=.25)+
    annotate("text",x=-350,y=.60,hjust=1,vjust=.4,size=2.0,color=INK,lineheight=1.05,
             label=paste0("-50 nt bin:  real ",v("real"),"  <  competitor ",v("competitor"),
                          "  <  decoy ",v("decoy")))+
    annotate("text",x=-1950,y=Inf,hjust=0,vjust=1.6,label="upstream of the poly(A) site",
             size=2.0,color="#555555")+
    labs(title="A   Strong 5' splice-site density flanking the poly(A) site",
         x="position relative to poly(A) site (nt)",
         y="strong 5' splice sites (MaxEnt >= 8) per kb")+
    th+theme(legend.position="bottom")
}

# b: within-gene difference
panel_B<-function(){
  bw<-diff(range(DL$delta_per_kb))/54
  ggplot(DL,aes(delta_per_kb))+
    geom_histogram(aes(y=after_stat(density)),binwidth=bw,fill=GREY,alpha=.45,color=NA)+
    geom_density(color=INK,linewidth=.45,adjust=1.1)+
    geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.35)+
    geom_vline(xintercept=MEAN,color=REAL,linewidth=.5)+
    annotate("text",x=MEAN,y=Inf,vjust=1.5,hjust=1.08,size=2.0,color=REAL,
             label=sprintf("mean %.1f",MEAN))+
    annotate("text",x=0,y=Inf,vjust=1.5,hjust=-0.12,size=2.0,color="#555555",label="no difference")+
    annotate("text",x=-Inf,y=Inf,hjust=-0.06,vjust=2.6,size=2.1,color=INK,lineheight=1.25,
      label=paste0("median ",sprintf("%.1f",MED)," per kb\n",
                   sprintf("%.0f%%",100*FBZ)," of genes below zero\n",
                   "paired Wilcoxon p ",PW))+
    scale_y_continuous(expand=expansion(mult=c(0,.14)))+
    labs(title="B   Within-gene difference in U1 site density",
         x="per gene: real - mean competitor, U1 sites (-dG >= 4) per kb",
         y="density of genes")+th
}

# c: redundancy of the two donor scores
XL_U1<-"U1:pre-mRNA duplex free energy (ViennaRNA duplexfold, U1 snRNA 5' end ACUUACCUG)"
YL_ME<-"5' splice-site score (MaxEntScan)"
panel_C<-function(){
  r<-ssv("r_all_pooled"); rg<-ssv("r_gt_pooled")
  n<-ssv("n_pos_all"); ng<-ssv("n_pos_gt")
  e(sprintf("[C]    HEADLINE GT-anchored r = %.3f over %s donor-dinucleotide positions",rg,nfmt(ng)))
  e(sprintf("[C]    context  all positions r = %.3f over %s positions (%.2f%% are GT-anchored)",
            r,nfmt(n),100*ng/n))
  G<-JD%>%filter(subset=="gt"); A<-JD%>%filter(subset=="all")
  ramp<-c("#F7FBFF","#C6DBEF","#6BAED6","#2171B5","#08306B")
  main<-ggplot(G,aes(u1_dg,me5,fill=count))+geom_tile()+
    scale_fill_gradientn(colours=ramp,trans="log10",name="positions",
                         breaks=c(1e1,1e2,1e3,1e4,1e5,1e6),
                         labels=c("10","100","1k","10k","100k","1M"))+
    scale_x_continuous(breaks=seq(-4,12,2),expand=expansion(mult=c(.01,.01)))+
    scale_y_continuous(breaks=seq(-50,10,10),expand=expansion(mult=c(.36,.06)))+
    annotate("text",x=-Inf,y=-Inf,hjust=-0.06,vjust=-0.55,size=2.3,color=INK,lineheight=1.3,
             label=paste0("r = ",sprintf("%.2f",rg)," at GT-anchored positions\n",
                          "n = ",nfmt(ng)," donor dinucleotides"))+
    labs(title="C   Correlation between U1 duplex free energy and MaxEntScan 5' splice-site score",
         x=XL_U1,y=YL_ME)+
    th+theme(legend.position="right",legend.key.width=unit(.45,"lines"),
             legend.key.height=unit(1.0,"lines"),
             legend.title=element_text(size=BASE-1.5,color=INK))
  ins<-ggplot(A,aes(u1_dg,me5,fill=count))+geom_tile(show.legend=FALSE)+
    scale_fill_gradientn(colours=ramp,trans="log10")+
    scale_x_continuous(breaks=c(-4,0,4,8,12),expand=expansion(mult=c(.01,.01)))+
    scale_y_continuous(breaks=c(-40,-20,0),expand=expansion(mult=c(.02,.02)))+
    labs(x=NULL,y=NULL,
         title=paste0("all scanned positions, r = ",sprintf("%.2f",r),"\n",
                      "diluted by the ",sprintf("%.0f%%",100*(1-ng/n)),
                      " of positions with no GT,\nwhere neither score is informative"))+
    theme_classic(base_size=BASE-2)+
    theme(axis.text=element_text(color=INK,size=BASE-3),
          axis.line=element_line(color=INK,linewidth=.25),
          plot.title=element_text(size=BASE-2.2,color="#555555",hjust=0,lineheight=1.2,
                                  margin=margin(b=2)),
          plot.background=element_rect(fill="white",color=GREY,linewidth=.2),
          plot.margin=margin(3,4,3,3))
  main+inset_element(ins,left=.555,bottom=.02,right=.995,top=.345,align_to="panel")
}

# d: threshold selection
SCLAB<-c(u1_dg="U1:pre-mRNA duplex free energy, -dG (kcal/mol)",
         me5="5' splice-site score (MaxEntScan, bits)")
panel_D<-function(){
  D<-EX%>%filter(class=="pooled",frac>=1e-6)%>%
    mutate(score=factor(score,levels=names(SCLAB),labels=SCLAB))
  pc<-function(x) if (x>=0.01) sprintf("%.1f%%",100*x) else sprintf("%.2f%%",100*x)
  MK<-bind_rows(
    data.frame(sc=names(SCLAB)[1],threshold=c(4,7),
               frac=c(ssv("exceed_u1_4"),ssv("exceed_u1_7")),
               tag=sprintf("-dG >= %g",c(4,7))),
    data.frame(sc=names(SCLAB)[2],threshold=8,frac=ssv("exceed_me5_8"),
               tag="MaxEnt >= 8"))%>%
    mutate(score=factor(sc,levels=names(SCLAB),labels=SCLAB))
  MK$fold<-mapply(FOLD,MK$sc,MK$threshold)
  ANN<-MK%>%group_by(score)%>%
    summarise(txt=paste(sprintf("%s   %s of positions",tag,sapply(frac,pc)),collapse="\n"),
              .groups="drop")
  e("[D]    marked thresholds: ",paste(sprintf("%s -> %s (tail decays %.1fx per unit score)",
        MK$tag,sapply(MK$frac,pc),MK$fold),collapse="   "))
  ggplot(D,aes(threshold,frac))+
    geom_segment(data=MK,aes(x=threshold,xend=threshold,y=1e-6,yend=frac),
                 linetype=2,color=COMP,linewidth=.3)+
    geom_line(linewidth=.5,color=INK)+
    geom_point(data=MK,size=1.2,color=COMP)+
    geom_text(data=ANN,aes(x=Inf,y=Inf,label=txt),inherit.aes=FALSE,
              hjust=1.04,vjust=1.4,size=2.1,color=INK,lineheight=1.35)+
    facet_wrap(~score,scales="free_x",strip.position="bottom")+
    scale_y_log10(breaks=c(1,.1,.01,1e-3,1e-4,1e-5),
                  labels=function(x) format(x,scientific=FALSE,drop0trailing=TRUE,trim=TRUE),
                  expand=expansion(mult=c(.03,.22)))+
    labs(title="D   Score distributions and threshold selection",
         x=NULL,y="fraction of positions >= threshold")+
    th+theme(strip.placement="outside",
             strip.text=element_text(size=BASE,color=INK,margin=margin(t=3)),
             panel.spacing=unit(10,"pt"))
}

CAP<-paste0("n = ",nfmt(sv("n_real"))," real 3' ends, ",nfmt(sv("n_competitor")),
  " competitors and ",nfmt(sv("n_decoy"))," intronic decoys in ",nfmt(NGP),
  " paired genes. Definitions, windows and thresholds: thesis_section_u1landscape.txt.")

fig<-(panel_A()/panel_B()/panel_C()/panel_D())+plot_layout(heights=c(1,.88,1.05,.85))+
  plot_annotation(
    title="Real 3' ends occupy splice-donor-depleted terminal-exon sequence relative to intragenic competitors",
    subtitle="Two marks of one donor axis at the same sites, and the calibration of both",
    caption=CAP,
    theme=theme(plot.title=element_text(size=BASE+2,face="bold",color=INK),
                plot.subtitle=element_text(size=BASE,color="#555555"),
                plot.caption=element_text(size=BASE-1.4,color="#555555",hjust=0)))

ggsave("figure_splicing_environment.pdf",fig,width=6.8,height=12.6,device=grDevices::cairo_pdf)
ggsave("figure_splicing_environment.png",fig,width=6.8,height=12.6,dpi=300)
e("[out]  figure_splicing_environment.pdf  figure_splicing_environment.png")

pa<-MG%>%filter(offset==-50); pv<-function(c) sprintf("%.2f",pa$donor_per_kb[pa$class==c])
pse<-function(c) sprintf("%.3f",pa$se[pa$class==c])
mrange<-function(lo,hi,c){ q<-MG%>%filter(offset>lo,offset<hi,class==c)
  if(!nrow(q)) stop("mrange: no rows for class ",c); sprintf("%.2f",mean(q$donor_per_kb)) }
pc<-function(x) if (x>=0.01) sprintf("%.2f%%",100*x) else sprintf("%.3f%%",100*x)
NPOS<-ssv("n_pos_all"); NGT<-ssv("n_pos_gt")
NSITE<-sum(sapply(LEV,function(c) ssv(paste0("n_sites_",c))))
FN<-rd("analysis_set_funnel.tsv")
lab<-function(x) format(as.numeric(x),big.mark=",",trim=TRUE)
NBINS<-nrow(MG)/length(LEV)
e("[sec]  building thesis_section_u1landscape.txt")

W<-function(...) paste(strwrap(paste0(...),width=98),collapse="\n")
SEC<-paste(c(
"METHODS",
"",
"Site sets",
"",
W("Candidate cleavage sites were taken from cpa_peaks_final.parquet, the peak table behind the ",
  "rest of this work (70,723 peaks in total). All analyses in this figure use the subset flagged ",
  "in_analysis_set: ",nfmt(sv("n_real"))," real 3' ends and ",nfmt(sv("n_competitor")),
  " intragenic competitors, ",nfmt(as.numeric(sv("n_real"))+as.numeric(sv("n_competitor"))),
  " peaks in ",nfmt(NGP)," genes. A real 3' end is a peak with is_canonical set; the analysis ",
  "set carries exactly one per gene, and every gene in it carries at least one competitor, so ",
  "the gene count and the real-end count coincide."),
"",
W("Competitors are the peaks the baseline model calls that survive three exclusions. The column ",
  "that records the result, in_analysis_set in cpa_peaks_final.parquet, is written by ",
  "task_build_final_parquet.py, but that script only copies a mask it loads; the exclusions ",
  "themselves are applied in task_rerank_corr.py and task_upstream_blocks.py. Counts below are ",
  "reconstructed from the raw annotation caches by trace_analysis_set.py, which asserts that its ",
  "reconstruction equals the shipped column exactly, and are drawn in the funnel figure."),
"",
W("Starting from all ",lab(FN$n_peaks[1])," called peaks in ",lab(FN$n_genes[1])," genes (",
  lab(FN$n_canonical[1])," canonical, one per gene, and ",lab(FN$n_competitor[1]),
  " competitors), the exclusions act on the competitor pool only; canonical peaks are never ",
  "removed by a filter, only by losing their gene. First, competitors annotated as real ",
  "alternative 3' ends are dropped: a competitor is annotated if the nearest same-strand ",
  "PolyASite 2.0 cluster representative site (atlas.clusters.2.0.GRCh38.96.tsv) or the nearest ",
  "same-strand PolyA_DB v4 site (polyadb_v4/HumanPas/hg38.PAS.main.tsv) lies within 500 nt. This ",
  "removes ",lab(FN$n_removed[2])," competitors. Second, competitors lying in reference terminal ",
  "exons are dropped: the 500 nt strand-oriented window immediately upstream of the peak must not ",
  "overlap, by even one nucleotide, the last exon of any GENCODE v49 transcript (single-exon ",
  "transcripts count as terminal, and the last exons of all transcripts are merged per strand ",
  "before testing). This removes a further ",lab(FN$n_removed[3])," competitors."),
"",
W("[VERIFY: these two exclusions are written as a single conjunction in task_rerank_corr.py ",
  "(SURV = COMP & ~APA & ~TERMEX), so the code does not order them. The counts above attribute ",
  "the ",lab(5294)," competitors that carry both flags to the annotation filter, which is listed ",
  "first; reported separately the flags remove ",lab(18119)," and ",lab(6370)," competitors and ",
  lab(19195)," in union. Any ordering language in the text is a presentational choice, not a ",
  "property of the pipeline.]"),
"",
W("A gene is then kept only if it still holds exactly one canonical peak and at least one ",
  "competitor. ",lab(FN$n_genes[3]-FN$n_genes[4])," genes lose every competitor at this point and ",
  "leave with their canonical peaks, which is why ",lab(FN$n_removed[4]),
  " canonical peaks disappear while the competitor count does not change. Third, competitors at ",
  "or downstream of the gene's canonical site in transcript coordinates are dropped, on the ",
  "grounds that transcription terminates at the used poly(A) site so sequence past it was never a ",
  "candidate for premature cleavage; this removes ",lab(FN$n_removed[5])," competitors, and the ",
  "gene rule is then re-applied, costing a further ",lab(FN$n_removed[6])," genes and their ",
  "canonical peaks."),
"",
W("What remains is the analysis set: ",lab(FN$n_peaks[6])," peaks in ",lab(FN$n_genes[6]),
  " genes, ",lab(FN$n_canonical[6])," real 3' ends and ",lab(FN$n_competitor[6]),
  " competitors. Every gene contributes exactly one real 3' end and at least one competitor, so ",
  "the gene count and the real-end count coincide, and every within-gene comparison in this work ",
  "is defined on every gene of the set."),
"",
W("Intronic decoys come from decoy_set.npz, built by task_b16.py against GENCODE v49 ",
  "(../train/data/annotation/gencode.v49.annotation.gtf). Within each gene, candidate positions ",
  "are taken every 25 nt through every annotated intron, excluding any position within 1,000 nt ",
  "of a poly(A) atlas entry (../train/data/atlas.chr.bed), and capped at 600 candidates per gene ",
  "by random subsampling. Each competitor is then matched to 5 decoys, without replacement ",
  "within a gene, minimising a weighted squared distance on three standardised coordinates: ",
  "distance to the gene's canonical poly(A) site (weight 1), distance to the nearest annotated ",
  "3' splice site (weight 2) and distance to the nearest annotated 5' splice site (weight 2), ",
  "with a hard cap of 500 nt on the difference in distance-to-canonical-poly(A). This yields ",
  "138,758 decoys, of which 134,969 fall in analysis-set genes; ",nfmt(sv("n_decoy")),
  " of those were drawn without replacement at seed 42 for this figure. The decoys are therefore ",
  "matched on intron location and on splice-site context, and they bound the confound that ",
  "competitors sit inside introns while real 3' ends do not."),
"",
"Sequence",
"",
W("Sequence is read from hg38 (../train/data/hg38.fa) through a random-access FASTA index. ",
  "Fetches are transcript-oriented: offset 0 is the peak centre, negative offsets are upstream ",
  "along the transcript, and minus-strand intervals are reverse-complemented on read. Sequence ",
  "is upper-cased on read, so the genome's soft-masking of repeats is discarded and repeat-derived ",
  "positions are scored like any other."),
"",
W("Every panel uses the same window, [-2,000, +500] nt about the peak centre, 2,501 offsets ",
  "per site, and every number in this figure comes from one scan of it. Fetches are padded by ",
  "40 nt on each side so that a full 9-mer exists at every offset in the window, and the padding ",
  "is then discarded. This is also the window the GC metagene and the CNN input use, so the ",
  "measurements here are on the same footing as the rest of this work."),
"",
"Scoring",
"",
W("U1 duplex free energy. Every 9-mer of the 4^9 = 262,144-member 9-mer space was scored once, ",
  "before any analysis, by build_u1_lookup.py, as the free energy of duplex formation between the ",
  "9-mer read as RNA and the U1 snRNA 5' end 5'-ACUUACCUG-3' (Mount 1982; Patel and Steitz 2003), ",
  "computed with the ViennaRNA Python bindings as RNA.duplexfold(U1, nonamer).energy. The ",
  "canonical, unmodified U1 sequence is used because the nearest-neighbour parameters do not model ",
  "the pseudouridines U1 carries in vivo. The cached table is keyed on the DNA 9-mer. Following ",
  "the feature pipeline (scan_u1 in scripts/extract_splicing_features.py) the sign is inverted and ",
  "scores are reported as -dG in kcal/mol, so that larger values mean stronger predicted U1 ",
  "binding. Over the full 9-mer space -dG runs from -3.90 to 11.70. Every 9-mer has a value: this ",
  "score has no sequence prerequisite."),
"",
W("MaxEntScan 5' splice site. Scores come from a dense 4^9 lookup that reproduces maxentpy's ",
  "score5 exactly. The 9-mer carries the donor GT dinucleotide at 0-indexed positions 3 and 4, ",
  "that is the fourth and fifth bases (scan_me5, following Yeo and Burge 2004); the 9-mer ",
  "therefore spans the last 3 exonic and the first 6 intronic nucleotides of the donor. Over the ",
  "full 9-mer space score5 runs from -49.17 to 11.81. In the feature pipeline scan_me5 returns a ",
  "value only where that GT is present and NaN otherwise. The scans behind this figure score every ",
  "9-mer instead, which changes nothing about the strong-donor measurements: no 9-mer without the ",
  "GT reaches 8 bits anywhere in the 4^9 space (the maximum over the 245,760 non-GT 9-mers is ",
  "4.16), so the set of positions above threshold is identical either way."),
"",
W("The GT-anchored subset used in panel C is defined on exactly that criterion: a scanned position ",
  "is GT-anchored when its 9-mer has G at 0-indexed position 3 and T at position 4. Each 9-mer ",
  "beginning at sequence index k is assigned to offset k+3, the G of the GT, so that a donor score ",
  "and a U1 score at the same reported offset describe the same 9-mer."),
"",
W("MaxEntScan 3' splice site is not used in any panel of this figure. Where it is used elsewhere ",
  "in this work it is scored by maxentpy's score3 on 23-mers carrying the acceptor AG at 0-indexed ",
  "positions 18 and 19, spanning the last 20 intronic and the first 3 exonic nucleotides ",
  "(scan_me3, following Yeo and Burge 2004)."),
"",
"Aggregation",
"",
W("Panel A is a metagene of strong 5' splice sites: positions with MaxEnt score5 >= 8 are counted ",
  "in ",NBINS," bins of 100 nt across the window and expressed per kb of scanned sequence, separately ",
  "for each class, with ribbons of +/-1 standard error across sites within the class."),
"",
W("Panel B reduces each site to one scalar, the number of U1 sites (-dG >= 4) per kb over the ",
  "same window, and then each gene to one difference: the value at that gene's real 3' end minus ",
  "the mean over that gene's competitors. Because both terms come from the same gene, this ",
  "difference is free of the between-gene and intron-location differences that the pooled ",
  "comparisons carry."),
"",
W("Panels C and D are computed over every 9-mer position of every site's window, ",
  nfmt(NPOS)," positions in total across the ",nfmt(NSITE)," sites, of which ",nfmt(NGT),
  " (",sprintf("%.2f%%",100*NGT/NPOS),") carry the GT donor dinucleotide. A density is hits per ",
  "kb of scanned positions, so the denominator of the exceedance curve is scanned positions, not ",
  "GT dinucleotides."),
"",
W("Neither panel bins for its reported statistics. The Pearson correlation is ",
  "accumulated from exact running sums over the raw scores, and the exceedance curves are read ",
  "from histograms whose edges are placed so that 4, 7 and 8 fall exactly on an edge, which makes ",
  "P(score >= t) exact at each threshold in use. The heat maps in panel C are binned to 0.5 ",
  "kcal/mol by 1 bit for display only."),
"",
"Thresholds",
"",
W("Three cuts appear in this figure: U1 -dG >= 4, U1 -dG >= 7 and MaxEnt score5 >= 8. Over the ",
  nfmt(NPOS)," scanned positions they admit ",pc(ssv("exceed_u1_4")),", ",pc(ssv("exceed_u1_7")),
  " and ",pc(ssv("exceed_me5_8"))," of positions respectively; over the ",nfmt(NGT),
  " GT-anchored positions they admit ",pc(ssv("exceed_gt_u1_4")),", ",pc(ssv("exceed_gt_u1_7")),
  " and ",pc(ssv("exceed_gt_me5_8")),". Each cut sits inside a smoothly decaying tail rather than ",
  "at a break in the distribution: across the two units of score either side of each one the ",
  "exceedance falls by a steady factor of ",sprintf("%.1f",FOLD("u1_dg",4)),", ",
  sprintf("%.1f",FOLD("u1_dg",7))," and ",sprintf("%.1f",FOLD("me5",8)),
  " per unit of score. They are conventions on a continuum."),
"",
W("The U1 cuts are the pipeline's own. -dG >= 4 is the cutoff the feature pipeline has always ",
  "used, and -dG >= 7 is the threshold that minimises the false-discovery rate against a ",
  "dinucleotide-shuffle null in the calibration of u1_threshold_calibrate.py. That calibration is ",
  "worth quoting rather than hiding: no threshold in [3, 9] reaches an FDR of 0.10 against that ",
  "null, the minimum attained being 0.84 at -dG = 7, so the U1 9-mer score is composition-dominated ",
  "throughout its range. A U1 site in this work is a position of favourable base composition for ",
  "U1 hybridization, not a detected motif, and nothing in this figure is claimed as motif discovery. ",
  "The exceedance measured here, ",pc(ssv("exceed_u1_4"))," at -dG >= 4, sits ",
  sprintf("%.2f",abs(100*ssv("exceed_u1_4")-9.30))," percentage points from the 9.30% that ",
  "calibration found independently over its own peak windows."),
"",
"RESULTS",
"",
W("Real 3' ends sit in sequence that is depleted of strong splice donors, and the depletion is ",
  "confined to the upstream side. In the 100 nt bin immediately upstream of the site the density ",
  "of MaxEnt >= 8 donors is ",pv("real")," per kb at real 3' ends against ",pv("competitor"),
  " at competitors and ",pv("decoy")," at intronic decoys (+/-",pse("real"),", ",pse("competitor"),
  " and ",pse("decoy")," SE; panel A). Averaged over the whole 500 nt upstream of the site the ",
  "ordering is the same, ",mrange(-500,0,"real")," against ",mrange(-500,0,"competitor")," and ",
  mrange(-500,0,"decoy"),". Downstream of the site the three classes converge: averaged over ",
  "+1 to +500 nt they read ",mrange(0,501,"real"),", ",mrange(0,501,"competitor")," and ",
  mrange(0,501,"decoy")," per kb, a residual spread of ",
  sprintf("%.2f",max(sapply(LEV,function(c) as.numeric(mrange(0,501,c))))-
                 min(sapply(LEV,function(c) as.numeric(mrange(0,501,c))))),
  " per kb against the ",
  sprintf("%.2f",as.numeric(mrange(-500,0,"decoy"))-as.numeric(mrange(-500,0,"real"))),
  " per kb that separates real ends from decoys upstream. The bin at -50 nt is the minimum of ",
  "the real-end profile."),
"",
W("The decoy arm is what makes this a statement about 3' ends rather than about introns. ",
  "Competitors lie inside introns and real 3' ends do not, so any pooled contrast between them ",
  "confounds the two. The decoys are intronic and matched to the competitors on distance to the ",
  "canonical poly(A) site and on splice-site context, and they show no depletion at all; the ",
  "depletion tracks the real 3' end, not the intronic location."),
"",
W("The same contrast survives inside individual genes, where gene identity, expression and ",
  "intron architecture are held fixed by construction. Taking each gene's real 3' end minus the ",
  "mean of its own competitors, the real end carries ",sprintf("%.1f",abs(MEAN)),
  " fewer U1 sites per kb over the window (median ",sprintf("%.1f",abs(MED)),"; ",
  sprintf("%.0f%%",100*FBZ)," of ",nfmt(NGP)," genes below zero; two-sided Wilcoxon signed-rank ",
  "on the per-gene differences, V = ",wx("V"),", n = ",nfmt(NWX)," after dropping genes whose ",
  "difference is exactly zero, z = ",wx("z"),", p ",PW,", log10 p about ",wx("log10_p"),
  "; panel B). The effect is a shift of the whole distribution rather than a separation: the ",
  "distribution of per-gene differences is unimodal and straddles zero, with ",
  sprintf("%.0f%%",100*(1-FBZ))," of genes at or above zero. This is a population-level property ",
  "of where real 3' ends sit, not a per-site discriminator, and no classification metric is ",
  "reported for it."),
"",
W("The two donor measurements are largely one measurement. At GT-anchored positions, where a ",
  "donor score is defined, U1 duplex free energy and the MaxEntScan 5' splice-site score correlate ",
  "at Pearson r = ",sprintf("%.3f",ssv("r_gt_pooled"))," over ",nfmt(NGT),
  " positions (panel C), and the value is the same in each class separately (",
  sprintf("%.3f",ssv("r_gt_real")),", ",sprintf("%.3f",ssv("r_gt_competitor")),", ",
  sprintf("%.3f",ssv("r_gt_decoy"))," for real ends, competitors and decoys). Across all ",
  nfmt(NPOS)," scanned positions the correlation falls to ",sprintf("%.3f",ssv("r_all_pooled")),
  ", but that number is diluted rather than informative: only ",
  sprintf("%.2f%%",100*NGT/NPOS)," of positions carry a GT, and at the remaining ",
  sprintf("%.1f%%",100*(1-NGT/NPOS))," no donor exists for either score to measure. The ",
  "thermodynamic and the motif-based read of the donor are therefore treated throughout as a ",
  "single donor axis, and agreement between them is not counted as independent evidence."),
"",
W("What this figure does not contain. It reports no model, no classifier and no ranking metric: ",
  "every quantity is a measured density, a per-gene difference or a correlation. It makes no motif ",
  "claim, for the calibration reason given above. Cleavage-site base composition, including the ",
  "AT-rich dip and its depth, and terminal-exon architecture, including donor-flanked and ",
  "internal-exon-like classification, are measured in separate figures and are not defined or ",
  "shown here."),
""),collapse="\n")
writeLines(SEC,"thesis_section_u1landscape.txt")
e("")
e(SEC)
e("")
NV<-length(grep("[VERIFY",strsplit(SEC,"\n")[[1]],fixed=TRUE))
e("[sec]  [VERIFY] placeholders left in:",NV,"paragraphs")
e("[chk]  panel C headline = GT-anchored r",sprintf("%.3f",ssv("r_gt_pooled")),
  "; all-positions r",sprintf("%.3f",ssv("r_all_pooled")),"is inset context only")
PAT<-paste0("RNA","hybrid")
FILES<-c("make_splicing_environment.R","splice_score_redundancy.py","thesis_section_u1landscape.txt")
HITS<-unlist(lapply(FILES,function(f){ L<-readLines(f); paste0(f,":",grep(PAT,L)) }))
HITS<-HITS[!grepl(":$",HITS)]
e("[chk]  '",PAT,"' occurrences across ",paste(FILES,collapse=", "),": ",length(HITS),
  if(length(HITS)) paste0("  (",paste(HITS,collapse=" "),")") else "  (none: the lookup is ViennaRNA duplexfold)")
e("[out]  thesis_section_u1landscape.txt")
