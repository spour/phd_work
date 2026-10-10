#!/usr/bin/env Rscript
# figure 1: what the cnn discriminates
source("cnn_fig_common.R")
if(is.null(SB)) stop("shuffle_benchmark.tsv is missing; run shuffle_benchmark.py first")
SC<-rd("cnn_training_schema.tsv"); sv<-function(k) SC$value[SC$key==k][1]
svn<-function(k) as.numeric(sv(k))
ACCC<-"#7FA9C9"

LAY<-data.frame(
  x=1:9,
  lab=c("input\n4 x 2,500","conv 32 x 19\nexponential","max-pool 10","conv 64 x 7\nReLU",
        "max-pool 5","conv 96 x 5\nReLU","max-pool 2","flatten 96 x 25\n(position kept)",
        "dense 64\n-> logit"),
  kind=c("io","exp","pool","conv","pool","conv","pool","flat","io"))
pA1<-ggplot(LAY,aes(x,0))+
  geom_tile(aes(fill=kind),width=.86,height=1,color=INK,linewidth=.25)+
  geom_text(aes(label=lab),size=1.75,color=INK,lineheight=1.05)+
  geom_segment(data=data.frame(x=1:8+.44,xend=2:9-.44),aes(x=x,xend=xend,y=0,yend=0),
               inherit.aes=FALSE,color=INK,linewidth=.25,
               arrow=grid::arrow(length=unit(.045,"in"),type="closed"))+
  scale_fill_manual(values=c(io="#EDF1F5",exp="#CFE0EF",pool="#F5F5F5",conv="#E4EDE6",
                             flat="#F2E9DC"),guide="none")+
  coord_cartesian(xlim=c(.5,9.5),ylim=c(-.7,.7),expand=FALSE)+theme_void()

MB<-paste(
 paste0("Task           real 3' end (",sv("real_label"),") vs intragenic competitor (",
        sv("competitor_label"),"), scored within each gene"),
 paste0("Input          [-",nf(svn("window_up")),", +",nf(svn("window_dn")),
        "] nt around the site in transcript orientation, one-hot, length ",nf(svn("window_len"))),
 paste0("Split          ",sv("split_rule"),"; no gene spans two splits"),
 paste0("               train ",nf(svn("n_train"))," / val ",nf(svn("n_val")),
        " / test ",nf(svn("n_test"))," peaks;  test = ",nf(svn("n_test_genes")),
        " genes, base rate ",sprintf("%.3f",svn("base_rate_test"))),
 paste0("Loss           BCEWithLogits, pos_weight ",sprintf("%.2f",svn("pos_weight")),
        " (= competitors / real ends in train)"),
 paste0("Stopping       ",sv("early_stop"),", the metric used throughout"),
 paste0("Calibration    Platt (scale and bias) on val, temperature ",
        sprintf("%.1f",svn("temperature")),
        ". Bias-free temperature scaling does not converge here: pos_weight"),
 paste0("               offsets the logits, and a pure scale cannot undo an offset. ",
        "Ranking metrics are unaffected either way."),
 sep="\n")
pA2<-ggplot()+annotate("text",x=0,y=1,hjust=0,vjust=1,size=1.95,color=INK,lineheight=1.42,
                       label=MB,family="mono")+
  coord_cartesian(xlim=c(0,1),ylim=c(0,1))+theme_void()
pA<-wrap_elements(full=(pA1/pA2)+plot_layout(heights=c(.62,1))) + labs(tag="A")

# model series for the bar panels (c)
LV<-c("sequence CNN","splice features\n(acceptor/donor/U1, no GC)","acceptor 3'ss alone","window GC alone")
PAL<-setNames(c(REAL,SPL,ACCC,GREY),LV)
mk<-function(metric) data.frame(
  model=factor(LV,levels=LV),
  value=c(gv("full",metric),spv(metric),gv("acceptor_alone",metric),gv("window_gc_alone",metric)),
  lo   =c(gv("full",metric,"ci_lo"),spv(metric,col="ci_lo"),
          gv("acceptor_alone",metric,"ci_lo"),gv("window_gc_alone",metric,"ci_lo")),
  hi   =c(gv("full",metric,"ci_hi"),spv(metric,col="ci_hi"),
          gv("acceptor_alone",metric,"ci_hi"),gv("window_gc_alone",metric,"ci_hi")),
  metric=metric)

if (is.null(CV)) stop("curves_roc_pr.tsv is missing; run score_for_curves.py first")
CVd <- CV %>% mutate(model = factor(model, levels = CURVE_LV))
au <- vapply(CURVE_LV, function(m) cvv(m, "auroc"), 0)
ap <- vapply(CURVE_LV, function(m) cvv(m, "auprc"), 0)
lab_roc <- setNames(sprintf("%s  (AUROC %.3f)", CURVE_LV, au), CURVE_LV)
lab_pr  <- setNames(sprintf("%s  (AUPRC %.3f)", CURVE_LV, ap), CURVE_LV)
e("[fig1B] curves from curves_roc_pr.tsv; AUROC ",
  paste(sprintf("%s %.3f", CURVE_LV, au), collapse="  "))
e("[fig1B] AUPRC ", paste(sprintf("%s %.3f", CURVE_LV, ap), collapse="  "))

thc <- th + theme(legend.position = "bottom", legend.direction = "horizontal",
                  legend.text = element_text(size = BASE - 2.2),
                  legend.key.width = unit(1.1, "lines"),
                  legend.margin = margin(t = -2),
                  panel.grid.major.x = element_line(color = "#E8EAED", linewidth = .2),
                  strip.text = element_text(size = BASE, face = "bold", colour = INK))
SUBB <- paste0(
  "Pooled over all ", nf(NPEAK), " held-out test peaks, shown for comparability with other studies. Pooled metrics have no 0.5 null on this design:\n",
  "each gene contributes one real end against about nine competitors, so the within-gene metric in C is the headline. The acceptor's pooled\n",
  "AUPRC (", sprintf("%.3f", ap[["acceptor 3'ss"]]), ") slightly exceeds the splice model's (", sprintf("%.3f", ap[["splice features (no GC)"]]),
  ") even though the splice model wins within gene; pooled and within-gene rankings need not agree.")
pB1 <- ggplot(CVd, aes(fpr, tpr, colour = model)) +
  geom_abline(slope = 1, intercept = 0, linetype = "22", linewidth = .3, colour = INK) +
  geom_line(linewidth = .55) +
  facet_wrap(~"Receiver operating characteristic") +
  scale_colour_manual(values = CURVE_PAL, labels = lab_roc, name = NULL) +
  guides(colour = guide_legend(nrow = 2, byrow = TRUE)) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, .25), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, .25), expand = c(0, 0)) +
  labs(title = "Pooled discrimination (reference)", subtitle = SUBB,
       x = "false positive rate", y = "true positive rate", tag = "B") + thc
pB2 <- ggplot(CVd, aes(recall, precision, colour = model)) +
  geom_hline(yintercept = BASER, linetype = "22", linewidth = .3, colour = INK) +
  annotate("text", x = .03, y = BASER, label = sprintf("base rate %.3f", BASER),
           hjust = 0, vjust = -0.6, size = 1.9, colour = INK) +
  geom_line(linewidth = .55) +
  facet_wrap(~"Precision-recall") +
  scale_colour_manual(values = CURVE_PAL, labels = lab_pr, name = NULL) +
  guides(colour = guide_legend(nrow = 2, byrow = TRUE)) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, .25), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, .25), expand = c(0, 0)) +
  labs(title = NULL, subtitle = NULL, x = "recall", y = "precision") + thc
pB <- pB1 + pB2 + plot_layout(widths = c(1, 1))

Cd<-rbind(mk("winrate"),mk("p_at_1"))
Cd$metric<-factor(ifelse(Cd$metric=="winrate","within-gene win-rate  P(real > competitor)",
                         "P@1  real end is the gene's top site"),
                  levels=c("within-gene win-rate  P(real > competitor)","P@1  real end is the gene's top site"))
REFC<-data.frame(metric=factor(rep("within-gene win-rate  P(real > competitor)",2),levels=levels(Cd$metric)),
                 yint=c(0.5,ACC_REF),lab=c("chance 0.50",paste0("acceptor 3'ss reference ",sprintf("%.3f",ACC_REF))))
pC<-ggplot(Cd,aes(model,value,fill=model))+
  geom_col(width=.66)+
  geom_errorbar(aes(ymin=lo,ymax=hi),width=.16,linewidth=.28,color=INK)+
  geom_hline(data=REFC,aes(yintercept=yint),linetype="22",linewidth=.3,color=INK)+
  geom_text(data=REFC,aes(x=.55,y=yint,label=lab),inherit.aes=FALSE,hjust=0,vjust=-0.5,
            size=1.8,color=INK)+
  geom_text(aes(y=hi,label=sprintf("%.3f",value)),vjust=-0.75,size=1.9,color=INK)+
  facet_wrap(~metric,scales="free_y")+
  scale_fill_manual(values=PAL,guide="none")+
  scale_y_continuous(expand=expansion(mult=c(0,.20)))+
  labs(title="Within-gene discrimination by model",
       subtitle=paste0("Gene-averaged over ",nf(NGENE)," test genes, ties counted as 0.5, bootstrap 95% CIs over genes. The splice model is fitted on ",NSEL,
                       " forward-selected\nfeatures (",paste(gsub("^spl_","",SEL_COLS),collapse=", "),
                       "), each computed in its own search window by the pipeline's featurizer, not on a matched\n[-2,000, +500] vector. Note that window GC alone (",
                       sprintf("%.3f",gv("window_gc_alone")),") outscores the splice model here while sitting near chance pooled (AUROC ",
                       sprintf("%.3f",gv("window_gc_alone","auroc")),"): within a gene,\nGC contrast between a real end and its own competitors is informative even though its absolute level is not."),
       x=NULL,y=NULL,tag="C")+th+theme(axis.text.x=element_text(size=BASE-2.2,lineheight=.9))

DL<-c("sequence CNN","splice features\n(acceptor/donor/U1, no GC)")
Dd<-do.call(rbind,lapply(seq_along(c("cnn","splice_selected")),function(i){
  m<-c("cnn","splice_selected")[i]
  data.frame(model=factor(DL[i],levels=DL),
             seqtype=factor(c("real sequence","dinucleotide-shuffled"),
                            levels=c("real sequence","dinucleotide-shuffled")),
             value=c(sbv(m,"real"),sbv(m,"shuffled")),
             lo=c(sbv(m,"real_lo"),sbv(m,"shuffled_lo")),
             hi=c(sbv(m,"real_hi"),sbv(m,"shuffled_hi")))}))
ANN<-data.frame(model=factor(DL,levels=DL),
  lab=vapply(c("cnn","splice_selected"),function(m)
      sprintf("drop %.3f\n%.0f%% of the lift lost",sbv(m,"drop"),sbv(m,"pct_of_lift_lost")),""),
  y=vapply(c("cnn","splice_selected"),function(m) sbv(m,"real"),0))
e("[fig1D] CNN real ",sprintf("%.4f",sbv("cnn","real"))," -> shuffled ",sprintf("%.4f",sbv("cnn","shuffled")),
  " (",sprintf("%.0f%%",sbv("cnn","pct_of_lift_lost"))," of the lift lost)")
e("[fig1D] splice real ",sprintf("%.4f",sbv("splice_selected","real"))," -> shuffled ",
  sprintf("%.4f",sbv("splice_selected","shuffled"))," (",
  sprintf("%.0f%%",sbv("splice_selected","pct_of_lift_lost"))," of the lift lost)")
pD<-ggplot(Dd,aes(seqtype,value,fill=seqtype))+
  geom_hline(yintercept=.5,linetype="22",linewidth=.3,color=INK)+
  geom_col(width=.6)+
  geom_errorbar(aes(ymin=lo,ymax=hi),width=.14,linewidth=.28,color=INK)+
  geom_text(aes(y=hi,label=sprintf("%.3f",value)),vjust=-0.75,size=1.9,color=INK)+
  geom_text(data=ANN,aes(x=1.5,y=y*1.055,label=lab),inherit.aes=FALSE,size=1.9,
            color=INK,lineheight=1.15)+
  facet_wrap(~model)+
  scale_fill_manual(values=setNames(c(REAL,COMP),levels(Dd$seqtype)),guide="none")+
  scale_y_continuous(expand=expansion(mult=c(0,.22)))+
  labs(title="Effect of dinucleotide shuffling on within-gene discrimination",
       subtitle=paste0("Both models are scored on THE SAME shuffled sequences (",nf(svn("n_test")),
                       " test sites x ",SB$k_shuffles[1],
                       " Altschul-Erikson shuffles), which preserve mononucleotide and\ndinucleotide frequency exactly, so GC and CpG are unchanged and only order is destroyed. The splice model's features are ",
                       "re-derived\nfrom the shuffled sequence, not carried over. Dashed line is chance."),
       x=NULL,y="within-gene win-rate",tag="D")+th

CAP<-paste(strwrap(paste0(
  "Sites are the ",nf(svn("n_train")+svn("n_val")+svn("n_test"))," analysis-set peaks: ",sv("real_label")," real 3' ends and their intragenic competitors (",sv("competitor_label"),"). ",
  "Test set ",nf(NPEAK)," peaks in ",nf(NGENE)," genes, base rate ",sprintf("%.3f",BASER),"; ",sv("split_rule"),". ",
  "The splice comparator uses no GC or base-content column: its pool is the 26 spl_* features (MaxEnt 3' and 5' splice sites, U1 hybridisation, branchpoint, ",
  "polypyrimidine tract, U2AF, donor counts and distances), forward-selected on an inner validation split of TRAINING genes only, stopping when a feature added less than 0.005. ",
  "Each of those features is computed in its own search window by the pipeline's featurizer, so B, C and D are matched on the split, the test genes and the metric, not on the input window. ",
  "The acceptor 3'ss within-gene reference of ",sprintf("%.3f",ACC_REF)," is the project's published value. ",
  "Panel D drop percentages are expressed against each model's own lift above chance."),
  width=152),collapse="\n")

fig<-(pA/pB/pC/pD)+plot_layout(heights=c(1.02,1.0,1.0,1.0))+
  plot_annotation(
    title="A sequence CNN discriminates real 3' ends from intragenic competitors within gene, and two-thirds of the discrimination requires sequence order",
    subtitle="Architecture and training schema, pooled and within-gene discrimination against a splice-biology comparator, and the matched dinucleotide-shuffle control.",
    caption=CAP,
    theme=theme(plot.title=element_text(size=BASE+2,face="bold",color=INK),
                plot.subtitle=element_text(size=BASE,color="#555555",lineheight=1.2),
                plot.caption=element_text(size=BASE-1.7,color="#555555",hjust=0,lineheight=1.25),
                plot.tag=element_text(size=BASE+2,face="bold",color=INK)))
ggsave("figure_cnn_performance.pdf",fig,width=9.2,height=11.8,device=grDevices::cairo_pdf)
ggsave("figure_cnn_performance.png",fig,width=9.2,height=11.8,dpi=300)
e("[out] figure_cnn_performance.pdf figure_cnn_performance.png")
