#!/usr/bin/env Rscript
# the donor 5'ss as terminal-exon architecture
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
  plot.title=element_text(size=BASE+1,face="bold",hjust=0),
  plot.subtitle=element_text(size=BASE-1.2,color="#555555",lineheight=1.2),
  legend.title=element_blank(),legend.text=element_text(size=BASE-1),
  legend.key.height=unit(.7,"lines"),legend.key.width=unit(.9,"lines"),
  legend.margin=margin(0,0,0,0),legend.box.spacing=unit(2,"pt"),
  strip.background=element_blank(),strip.text=element_text(size=BASE,color=INK,face="bold"))
nfmt<-function(x) format(x,big.mark=",")
ST<-rd("u1_5ss_stats.tsv"); sv<-function(k) ST$value[ST$key==k][1]

MG<-rd("v32_F_metagene_analysisset.tsv")%>%mutate(class=recode(group,canonical="real"))
NR<-rd("nearest5ss.tsv")
DF<-rd("donor_flanked.tsv")
N<-MG%>%group_by(class)%>%summarise(n=max(n),.groups="drop")
gn<-function(c) nfmt(N$n[N$class==c])
e("[read] v32_F_metagene_analysisset.tsv  nearest5ss.tsv  donor_flanked.tsv  u1_5ss_stats.tsv")
e("[read] fivess_profile_me5_long.tsv NOT used (different classes/design; see header)")
e("[read] u1_5ss_sitescan.npz NOT opened; nothing recomputed")
e("[n]    real",gn("real"),"  competitor",gn("competitor"),"  decoy",gn("decoy"))

panel_A<-function(){
  D<-MG%>%mutate(class=fc(class))
  pa<-MG%>%filter(offset==-50)
  v<-function(c) sprintf("%.2f",pa$donor_per_kb[pa$class==c])
  e("[A]    pA bin (-50 nt) strong-5'ss density: real",v("real"),"< competitor",v("competitor"),
    "< decoy",v("decoy"),"sites/kb")
  ggplot(D,aes(offset,donor_per_kb,color=class,fill=class))+
    annotate("rect",xmin=-Inf,xmax=0,ymin=-Inf,ymax=Inf,fill=GREY,alpha=.06)+
    geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.3)+
    geom_ribbon(aes(ymin=donor_per_kb-se,ymax=donor_per_kb+se),alpha=.16,color=NA)+
    geom_line(linewidth=.55)+
    scale_color_manual(values=PAL)+scale_fill_manual(values=PAL,guide="none")+
    scale_x_continuous(breaks=seq(-2000,2000,500))+
    annotate("text",x=-1950,y=Inf,hjust=0,vjust=1.6,label="upstream of pA (the exon)",size=2.0,color="#555555")+
    labs(title="A   Strong 5'ss density around the pA",
         subtitle=paste0("At the pA bin (-50 nt) real ",v("real")," < competitor ",v("competitor"),
           " < decoy ",v("decoy")," sites/kb. Pooled: the decoy arm controls intron\nlocation. The separation is upstream-only and converges downstream of the pA."),
         x="position relative to pA (nt)",y="strong 5'ss sites / kb (MaxEnt >=8)")+
    th+theme(legend.position="bottom")
}

panel_B<-function(){
  D<-NR%>%mutate(class=fc(class),
       side=factor(side,levels=c("upstream","downstream"),
                   labels=c("upstream of the site","downstream of the site")))
  m<-function(s,c) sv(paste0("D_median_",s,"_",c))
  e("[B]    median nearest strong 5'ss: upstream",m("upstream","real"),"/",m("upstream","competitor"),
    "/",m("upstream","decoy"),"nt   downstream",m("downstream","real"),"/",
    m("downstream","competitor"),"/",m("downstream","decoy"),"nt (censored at 2 kb)")
  ggplot(D,aes(distance,ecdf,color=class))+geom_line(linewidth=.55)+facet_wrap(~side)+
    scale_color_manual(values=PAL)+scale_x_continuous(breaks=seq(0,2000,500))+
    labs(title="B   Distance to the nearest strong 5'ss",
         subtitle=paste0("Median upstream ",m("upstream","real")," / ",m("upstream","competitor"),
           " / ",m("upstream","decoy")," nt against downstream ",m("downstream","real"),
           " / ",m("downstream","competitor")," / ",m("downstream","decoy"),
           " nt, identical to within 1 nt.\nThe donor-free stretch is the exon body, upstream. Downstream the three classes are indistinguishable: past a real\n3' end there is no transcript to define. Censored at 2 kb, so the curves stop below 1."),
         x="distance to nearest 5'ss with MaxEnt >=8 (nt)",y="ECDF")+
    th+theme(legend.position="bottom")
}

# c (optional)
panel_C<-function(){
  D<-DF%>%filter(property %in% c("acceptor","donor_flanked_intervening"))%>%
    mutate(class=fc(class),
           property=factor(property,levels=c("acceptor","donor_flanked_intervening"),
             labels=c("strong upstream 3'ss\n(shared by all three)",
                      "acceptor AND intervening donor\n(internal-exon-like)")))
  e("[C]    acceptor",sprintf("%.3f / %.3f / %.3f",sv("E_acceptor_real"),sv("E_acceptor_competitor"),
      sv("E_acceptor_decoy")),"  internal-exon-like",
    sprintf("%.3f / %.3f / %.3f",sv("E_donor_flanked_intervening_real"),
      sv("E_donor_flanked_intervening_competitor"),sv("E_donor_flanked_intervening_decoy")),
    "  within-gene win-rate",sprintf("%.3f",sv("E_winrate_intervening")))
  ggplot(D,aes(property,frac,fill=class))+
    geom_col(position=position_dodge(width=.78),width=.68,color=INK,linewidth=.15)+
    geom_errorbar(aes(ymin=ci_lo,ymax=ci_hi),position=position_dodge(width=.78),
                  width=.16,linewidth=.3,color=INK)+
    geom_text(aes(label=sprintf("%.2f",frac)),position=position_dodge(width=.78),
              vjust=-1.5,size=1.9,color=INK)+
    scale_fill_manual(values=PAL)+
    scale_y_continuous(limits=c(0,1.05),breaks=seq(0,1,.25),expand=expansion(mult=c(0,.02)))+
    labs(title="C   What the donor adds to the acceptor",
         subtitle=paste0("A strong upstream 3'ss is near-universal in all three classes (",
           sprintf("%.2f",sv("E_acceptor_real"))," / ",sprintf("%.2f",sv("E_acceptor_competitor")),
           " / ",sprintf("%.2f",sv("E_acceptor_decoy")),"), so it barely separates them here.\nWhat the real end lacks is the intervening donor that would make the acceptor-to-pA span an internal exon: ",
           sprintf("%.3f",sv("E_donor_flanked_intervening_real"))," < ",
           sprintf("%.3f",sv("E_donor_flanked_intervening_competitor"))," < ",
           sprintf("%.3f",sv("E_donor_flanked_intervening_decoy")),
           ",\nwithin-gene win-rate ",sprintf("%.3f",sv("E_winrate_intervening")),
           ". The same property defined past the pA is null (",sprintf("%.3f",sv("E_winrate_dn300")),")."),
         x=NULL,y="fraction of sites (Wilson 95%)")+
    th+theme(legend.position="bottom")
}

CAP<-paste(strwrap(paste0(
  "n = ",gn("real")," real 3' ends, ",gn("competitor")," competitors and ",gn("decoy"),
  " intronic decoys (seed 42), in the ",nfmt(sv("n_paired_genes")),
  " genes of the cpa_peaks_final.parquet analysis set; competitors are baseline-model false positives after APA-annotated peaks, ",
  "reference terminal exons and at-or-downstream-of-canonical peaks are removed. ",
  "THE CONFOUND, PLAINLY: competitors sit inside introns and real 3' ends do not, so the pooled panels mix the site contrast with intron location; ",
  "the decoy arm is there to show how much of the separation that accounts for. ",
  "The donor is SUPPORTING evidence, not the discriminator: within-gene win-rate ",
  sprintf("%.3f",sv("E_winrate_intervening")),", under the acceptor's 0.606, which is the headline discriminator. ",
  "Read this figure as architecture (the real 3' end is a terminal exon: acceptor-defined, no donor between the acceptor and the pA), not as a predictor. ",
  "Ribbons 1 SE, bars Wilson 95%."),width=135),collapse="\n")
ann<-function() plot_annotation(
  title="The real 3' end is a terminal exon: no donor between its acceptor and the pA",
  subtitle="Strong 5'ss (MaxEnt >=8) at real 3' ends, competitors and intronic decoys. 'Downstream' is past the pA, where the transcript has ended.",
  caption=CAP,
  theme=theme(plot.title=element_text(size=BASE+3,face="bold",color=INK),
              plot.subtitle=element_text(size=BASE,color="#555555"),
              plot.caption=element_text(size=BASE-1.4,color="#555555",hjust=0,lineheight=1.25)))

AB<-(panel_A()/panel_B())+ann()
ggsave("figure_5ss_terminal_exon.pdf",AB,width=6.6,height=7.4,device=grDevices::cairo_pdf)
ggsave("figure_5ss_terminal_exon.png",AB,width=6.6,height=7.4,dpi=300)
ABC<-(panel_A()/panel_B()/panel_C())+plot_layout(heights=c(1,1,1.05))+ann()
ggsave("figure_5ss_terminal_exon_ABC.pdf",ABC,width=6.6,height=10.4,device=grDevices::cairo_pdf)
PNG<-Sys.getenv("ABC_PNG"); if(nzchar(PNG)) ggsave(PNG,ABC,width=6.6,height=10.4,dpi=300)
e("[out]  figure_5ss_terminal_exon.pdf  figure_5ss_terminal_exon.png  figure_5ss_terminal_exon_ABC.pdf")
