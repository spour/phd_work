#!/usr/bin/env Rscript
# does a per-site count of strong upstream 5'ss, in a tight window
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork)})
rd<-function(f){ if(!file.exists(f)) stop("missing: ",f); read_tsv(f,show_col_types=FALSE) }
e<-function(...) cat(...,"\n",file=stderr())
INK<-"#1A1A1A"; BASE<-7
REAL<-"#08519C"; COMP<-"#E8A33D"; DEC<-"#7B5EA7"; GREY<-"#8A8F98"; GRN<-"#2E7D5B"
LEV<-c("real","competitor","decoy")
LAB<-c(real="real 3' end",competitor="competitor",decoy="intronic decoy")
PAL<-setNames(c(REAL,COMP,DEC),LAB[LEV])
fc<-function(x) factor(x,levels=LEV,labels=LAB[LEV])
th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-1),axis.line=element_line(color=INK,linewidth=.3),
  axis.title=element_text(size=BASE),
  plot.title=element_text(size=BASE+1,face="bold",hjust=0),
  plot.subtitle=element_text(size=BASE-1.2,color="#555555",lineheight=1.2),
  legend.title=element_blank(),legend.text=element_text(size=BASE-1),
  legend.key.height=unit(.7,"lines"),legend.key.width=unit(.9,"lines"),
  legend.margin=margin(0,0,0,0),legend.box.spacing=unit(2,"pt"),
  strip.background=element_blank(),strip.text=element_text(size=BASE,color=INK,face="bold"))
nfmt<-function(x) format(as.numeric(x),big.mark=",")
ST<-rd("upstream_count_stats.tsv")
sv<-function(k) suppressWarnings(as.numeric(ST$value[ST$key==k][1]))
ss<-function(k) ST$value[ST$key==k][1]
slo<-function(k) ST$lo[ST$key==k][1]; shi<-function(k) ST$hi[ST$key==k][1]
D<-rd("upstream_count_dist.tsv"); SW<-rd("upstream_count_sweep.tsv"); EC<-rd("upstream_maxent_ecdf.tsv")
WD<-sv("default_window"); WP<-sv("chosen_window")
REF_A<-sv("ref_architecture"); REF_B<-sv("ref_acceptor")
e("[read] upstream_count_dist.tsv upstream_count_sweep.tsv upstream_maxent_ecdf.tsv upstream_count_stats.tsv")
e("[n]    real",nfmt(sv("n_real")),"  competitor",nfmt(sv("n_competitor")),"  decoy",nfmt(sv("n_decoy")),
  "  paired genes",nfmt(sv("n_paired_genes")))

panel_A<-function(){
  B<-D%>%filter(bucket!=">=1")%>%mutate(class=fc(class),
      bucket=factor(bucket,levels=c("0","1","2",">=3")),
      window=factor(window,levels=c(WD,WP),
        labels=c(paste0("[-",WD,",0]  tight (default)"),paste0("[-",WP,",0]  sweep peak"))))
  e("[A]    frac >=1 at [-",WD,",0]: real ",sprintf("%.4f",sv(paste0("frac_ge1_default_real"))),
    " competitor ",sprintf("%.4f",sv("frac_ge1_default_competitor")),
    " decoy ",sprintf("%.4f",sv("frac_ge1_default_decoy")),sep="")
  e("[A]    frac >=1 at [-",WP,",0]: real ",sprintf("%.4f",sv("frac_ge1_real")),
    " competitor ",sprintf("%.4f",sv("frac_ge1_competitor")),
    " decoy ",sprintf("%.4f",sv("frac_ge1_decoy")),sep="")
  main<-ggplot(B,aes(bucket,frac,fill=class))+
    geom_col(position=position_dodge(width=.8),width=.7,color=INK,linewidth=.12)+
    geom_errorbar(aes(ymin=ci_lo,ymax=ci_hi),position=position_dodge(width=.8),
                  width=.16,linewidth=.25,color=INK)+
    facet_wrap(~window)+scale_fill_manual(values=PAL)+
    scale_y_continuous(expand=expansion(mult=c(0,.05)))+
    labs(title="A   How many strong upstream donors a site carries",
         subtitle=paste0("Fraction with >=1 strong (MaxEnt>=8) donor: ",
           sprintf("%.2f",sv("frac_ge1_default_real")),"/",sprintf("%.2f",sv("frac_ge1_default_competitor")),
           "/",sprintf("%.2f",sv("frac_ge1_default_decoy"))," at ",WD," nt and ",
           sprintf("%.2f",sv("frac_ge1_real")),"/",sprintf("%.2f",sv("frac_ge1_competitor")),"/",
           sprintf("%.2f",sv("frac_ge1_decoy"))," at ",WP," nt (real/competitor/decoy).\nWithin-gene win-rate ",
           sprintf("%.3f",sv("winrate_default"))," at ",WD," nt, where ",
           sprintf("%.0f",100*sv("tied_default")),"% of within-gene pairs TIE, against ",
           sprintf("%.3f",sv("winrate_chosen"))," at ",WP," nt (",
           sprintf("%.0f",100*sv("tied_chosen")),"% tied). Among the pairs that are not tied,\ncompetitor > real in ",
           sprintf("%.0f",100*sv("asym_default")),"% at ",WD," nt and ",
           sprintf("%.0f",100*sv("asym_chosen")),"% at ",WP,
           " nt. Inset: the graded max MaxEnt5 in the tight window separates where the count only ties."),
         x="strong upstream donors in the window",y="fraction of sites (Wilson 95%)")+
    th+theme(legend.position="bottom")
  ins<-ggplot(EC%>%filter(window==WD)%>%mutate(class=fc(class)),aes(maxent,ecdf,color=class))+
    geom_line(linewidth=.4)+geom_vline(xintercept=8,linetype=2,color=INK,linewidth=.25)+
    scale_color_manual(values=PAL,guide="none")+
    labs(x=paste0("max MaxEnt5 in [-",WD,",0]"),y="ECDF")+theme_classic(base_size=BASE-2)+
    theme(axis.text=element_text(color=INK,size=BASE-2.5),axis.title=element_text(size=BASE-2.2),
          axis.line=element_line(color=INK,linewidth=.25),
          plot.background=element_rect(fill="white",color=GREY,linewidth=.2))
  main+inset_element(ins,left=.58,bottom=.46,right=.99,top=1.0,align_to="panel")
}

panel_B<-function(){
  S8<-SW%>%filter(scalar=="count_me8")
  e("[B]    count_me8 sweep win-rate: ",paste(sprintf("w%d %.3f",S8$window,S8$winrate),collapse="  "),sep="")
  ggplot(S8,aes(window))+
    geom_hline(yintercept=.5,color=GREY,linewidth=.25)+
    geom_hline(yintercept=REF_A,linetype="22",color=GREY,linewidth=.35)+
    geom_hline(yintercept=REF_B,linetype="22",color=INK,linewidth=.35)+
    annotate("text",x=100,y=REF_B+.004,hjust=0,size=1.9,color=INK,
             label=paste0("acceptor ",sprintf("%.3f",REF_B)," (the discriminator)"))+
    annotate("text",x=100,y=REF_A+.004,hjust=0,size=1.9,color="#555555",
             label=paste0("intervening-donor architecture ",sprintf("%.3f",REF_A))) +
    geom_ribbon(aes(ymin=ci_lo,ymax=ci_hi),alpha=.15,fill=INK)+
    geom_line(aes(y=winrate),linewidth=.6,color=INK)+geom_point(aes(y=winrate),size=.9,color=INK)+
    geom_ribbon(aes(ymin=asym_lo,ymax=asym_hi),alpha=.12,fill=GRN)+
    geom_line(aes(y=asym),linewidth=.5,color=GRN,linetype="41")+
    geom_point(aes(y=asym),size=.8,color=GRN)+
    annotate("text",x=2000,y=SW$asym[SW$scalar=="count_me8"&SW$window==2000]-.008,hjust=1,size=1.9,
             color=GRN,label="discordant pairs: competitor > real")+
    annotate("text",x=2000,y=SW$winrate[SW$scalar=="count_me8"&SW$window==2000]-.009,hjust=1,size=1.9,
             color=INK,label="win-rate (ties 0.5)")+
    scale_x_continuous(trans="log10",breaks=c(100,200,300,500,1000,2000))+
    labs(title="B   Window sweep: the metagene was not diluting",
         subtitle=paste0("Win-rate rises monotonically with window width, ",
           sprintf("%.3f",S8$winrate[S8$window==100])," at 100 nt to ",
           sprintf("%.3f",S8$winrate[S8$window==2000]),
           " at 2 kb, so a tighter window is WORSE, not sharper. The\ndirectional asymmetry is flat near ",
           sprintf("%.2f",mean(S8$asym)),
           ": the sign of the effect is stable, only the number of untied pairs changes. Colour is the statistic, not a class."),
         x="upstream window width w, [-w,0] (nt, log)",y="within-gene P(competitor > real)")+th
}

panel_C<-function(){
  S<-SW%>%mutate(scalar=factor(scalar,levels=c("count_me8","count_me65","max_me5"),
        labels=c("count, MaxEnt>=8","count, MaxEnt>=6.5","max MaxEnt5 (graded)")))
  bw<-ss("best_scalar")
  e("[C]    best scalar ",bw," win-rate ",sprintf("%.4f",sv("best_winrate")),
    " [",slo("best_winrate"),",",shi("best_winrate"),"]",sep="")
  ggplot(S,aes(window,winrate,color=scalar,fill=scalar))+
    geom_hline(yintercept=REF_A,linetype="22",color=GREY,linewidth=.35)+
    geom_hline(yintercept=REF_B,linetype="22",color=INK,linewidth=.35)+
    geom_ribbon(aes(ymin=ci_lo,ymax=ci_hi),alpha=.12,color=NA)+
    geom_line(linewidth=.55)+geom_point(size=.9)+
    scale_color_manual(values=c("count, MaxEnt>=8"=INK,"count, MaxEnt>=6.5"=GREY,
                                "max MaxEnt5 (graded)"=GRN))+
    scale_fill_manual(values=c("count, MaxEnt>=8"=INK,"count, MaxEnt>=6.5"=GREY,
                               "max MaxEnt5 (graded)"=GRN),guide="none")+
    scale_x_continuous(trans="log10",breaks=c(100,200,300,500,1000,2000))+
    labs(title="C   Presence/absence against a graded score",
         subtitle=paste0("The graded max beats both counts at every width and peaks at ",
           sprintf("%.3f",sv("best_winrate"))," (",bw,
           "), because it never ties. Lowering\nthe threshold to 6.5 helps the count only by untying it. All three stay under the acceptor's ",
           sprintf("%.3f",REF_B),". Colour is the scalar."),
         x="upstream window width w, [-w,0] (nt, log)",y="within-gene P(competitor > real)")+
    th+theme(legend.position="bottom")
}

VERD<-if(sv("clears_0534")==1)
  paste0("VERDICT: the count clears the 0.534 architecture win-rate only in WIDE windows (",
    sprintf("%.3f",sv("winrate_chosen"))," at ",WP," nt); in the tight ",WD,
    " nt window it does not (",sprintf("%.3f",sv("winrate_default")),
    "). A tight per-site count is not a sharper read of the metagene.") else
  paste0("VERDICT: the count does not clear the 0.534 architecture win-rate at any width; the metagene already sufficed.")
CAP<-paste(strwrap(paste0(
  "n = ",nfmt(sv("n_real"))," real 3' ends, ",nfmt(sv("n_competitor"))," competitors and ",
  nfmt(sv("n_decoy"))," intronic decoys (seed 42), in the ",nfmt(sv("n_paired_genes")),
  " genes of the cpa_peaks_final.parquet analysis set. ",
  "THE CONFOUND, PLAINLY: competitors sit inside introns and real 3' ends do not, so the pooled fractions in A mix the site contrast with intron location; ",
  "the decoy arm shows how much of it that accounts for, and every win-rate here is within-gene, which the confound cannot reach. ",
  VERD," Even the best scalar on this figure (",sprintf("%.3f",sv("best_winrate")),
  ") sits under the acceptor's ",sprintf("%.3f",REF_B),
  ", which is the discriminator; the donor is supporting architecture, not a predictor. ",
  "Win-rate is the gene-averaged P(competitor > real), ties 0.5, bootstrapped over paired genes; bars are Wilson 95%."),
  width=135),collapse="\n")

ann<-function() plot_annotation(
  title="A tight per-site count of upstream donors does not sharpen the metagene",
  subtitle="Strong 5'ss counted in [-w,0] upstream of the site, real 3' ends against competitors and intronic decoys",
  caption=CAP,
  theme=theme(plot.title=element_text(size=BASE+3,face="bold",color=INK),
              plot.subtitle=element_text(size=BASE,color="#555555"),
              plot.caption=element_text(size=BASE-1.4,color="#555555",hjust=0,lineheight=1.25)))

fig<-(panel_A()/panel_B()/panel_C())+plot_layout(heights=c(1.15,1,1))+ann()
ggsave("figure_upstream_donor_count.pdf",fig,width=6.8,height=10.2,device=grDevices::cairo_pdf)
ggsave("figure_upstream_donor_count.png",fig,width=6.8,height=10.2,dpi=300)
e("[out]  figure_upstream_donor_count.pdf  figure_upstream_donor_count.png")
