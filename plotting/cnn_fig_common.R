# shared theme, palette and tsv readers for the two cnn figures
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(tidyr);library(patchwork)})
rd<-function(f){ if(!file.exists(f)) stop("missing: ",f); read_tsv(f,show_col_types=FALSE) }
e<-function(...) cat(...,"\n",file=stderr())
INK<-"#1A1A1A"; BASE<-7
REAL<-"#08519C"
COMP<-"#E8A33D"
SPL <-"#2E7D5B"
GREY<-"#8A8F98"
th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-1),axis.line=element_line(color=INK,linewidth=.3),
  axis.title=element_text(size=BASE),
  panel.grid.major.y=element_line(color="#E8EAED",linewidth=.2),
  plot.title=element_text(size=BASE+1,face="bold",hjust=0),
  plot.subtitle=element_text(size=BASE-1.2,color="#555555",lineheight=1.2),
  legend.title=element_blank(),legend.text=element_text(size=BASE-1),
  legend.key.height=unit(.7,"lines"),legend.key.width=unit(.9,"lines"),
  legend.margin=margin(0,0,0,0),legend.box.spacing=unit(2,"pt"),
  strip.background=element_blank(),strip.text=element_text(size=BASE-1,color=INK,face="bold"))
nf<-function(x) format(as.numeric(x),big.mark=",")

M  <-rd("cnn_metrics.tsv")
SP <-rd("splice_feature_metrics.tsv")
OCC<-rd("occlusion_profile_reference.tsv")
gv <-function(s,m="winrate",col="value")
  suppressWarnings(as.numeric(M[[col]][M$score==s & M$metric==m][1]))
spv<-function(m,s="splice_selected",col="value")
  suppressWarnings(as.numeric(SP[[col]][SP$score==s & SP$metric==m][1]))
spx<-function(metric) as.character(SP$value[SP$score=="selection" & SP$metric==metric][1])
NGENE<-as.numeric(M$n_genes[1]); NPEAK<-as.numeric(M$n_peaks[1])
BASER<-as.numeric(M$base_rate[1])
ACC_REF<-0.606
SEL_COLS<-strsplit(spx("selected_columns"),"\\|")[[1]]
NSEL<-length(SEL_COLS)

CVF<-"curves_roc_pr.tsv"
CV<-if(file.exists(CVF)) rd(CVF) else NULL
CURVE_LV<-c("sequence CNN","splice features (no GC)","acceptor 3'ss","window GC")
CURVE_PAL<-setNames(c(REAL,SPL,"#7FA9C9",GREY),CURVE_LV)
cvv<-function(mod,col) if(is.null(CV)) NA_real_ else CV[[col]][CV$model==mod][1]

U1_RECONCILIATION<-paste0(
  "Two findings point the same way from different analyses. FROM THE SCALAR SPLICE ANALYSIS: U1-favourable density (-dG >= 4) and strong-5'ss density are correlated compositional reads (r = 0.22 to 0.44), and positioned U1 (-dG >= 7) reaches within-gene 0.587, below the acceptor's 0.606. FROM THE CNN: about 31% of the within-gene lift survives dinucleotide shuffling (composition), and no positioned donor or U1 motif is recovered (occlusion, implantation and in-silico mutagenesis are all null for GT/GU). The CNN does not identify WHICH composition its surviving third reads; it shows only that a compositional component exists and that no positioned donor or U1 motif is present. ")

SBF<-"shuffle_benchmark.tsv"
SB<-NULL
if(file.exists(SBF)){
  SB<-suppressWarnings(read_tsv(SBF,col_types=cols(.default=col_character())))%>%
      filter(model!="check")%>%
      mutate(across(-model,as.numeric))
  SBCHK<-suppressWarnings(read_tsv(SBF,col_types=cols(.default=col_character())))%>%
      filter(model=="check")
}
sbv<-function(mod,col) if(is.null(SB)) NA_real_ else SB[[col]][SB$model==mod][1]
