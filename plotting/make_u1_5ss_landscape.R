#!/usr/bin/env Rscript
# u1 and the donor 5'ss around competitors and real 3' ends
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork)})
rd<-function(f){ if(!file.exists(f)) stop("missing: ",f); read_tsv(f,show_col_types=FALSE) }
INK<-"#1A1A1A"; BASE<-7
REAL<-"#08519C"; COMP<-"#E8A33D"; DEC<-"#7B5EA7"; GREY<-"#8A8F98"
PAL<-c(real=REAL,competitor=COMP,decoy=DEC)
LEV<-c("real","competitor","decoy")
LAB<-c(real="real 3' end",competitor="competitor",decoy="intronic decoy")
th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-1),axis.line=element_line(color=INK,linewidth=.3),
  axis.title=element_text(size=BASE),
  plot.title=element_text(size=BASE+1,face="bold",hjust=0),
  plot.subtitle=element_text(size=BASE-1.2,color="#555555",lineheight=1.15),
  plot.caption=element_text(size=BASE-1.4,color="#555555",hjust=0,lineheight=1.2),
  legend.title=element_blank(),legend.text=element_text(size=BASE-1),
  legend.key.height=unit(.7,"lines"),legend.key.width=unit(.9,"lines"),
  legend.margin=margin(0,0,0,0),legend.box.spacing=unit(2,"pt"),
  strip.background=element_blank(),strip.text=element_text(size=BASE,color=INK,face="bold"))
fc<-function(x) factor(x,levels=LEV,labels=LAB[LEV])
ST<-rd("u1_5ss_stats.tsv"); sv<-function(k) ST$value[ST$key==k][1]
slo<-function(k) ST$lo[ST$key==k][1]; shi<-function(k) ST$hi[ST$key==k][1]
nfmt<-function(x) format(x,big.mark=",")
UPSHADE<-function() annotate("rect",xmin=-Inf,xmax=0,ymin=-Inf,ymax=Inf,fill=GREY,alpha=.06)
ZERO<-function() geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.3)

# a: landscape
panel_A<-function(){
  D<-rd("u1_metagene.tsv"); D$class<-fc(D$class)
  ord<-sprintf("real %.2f < competitor %.2f < decoy %.2f",
               sv("pA_u1_real"),sv("pA_u1_competitor"),sv("pA_u1_decoy"))
  ggplot(D,aes(offset,u1_dg,color=class,fill=class))+UPSHADE()+ZERO()+
    geom_ribbon(aes(ymin=u1_dg-se,ymax=u1_dg+se),alpha=.16,color=NA)+geom_line(linewidth=.55)+
    scale_color_manual(values=setNames(PAL,LAB[LEV]))+
    scale_fill_manual(values=setNames(PAL,LAB[LEV]),guide="none")+
    scale_x_continuous(breaks=seq(-2000,2000,1000))+
    labs(title="A   U1 landscape around the site",
         subtitle=paste0("Mean U1 -dG per 100 nt bin. At the pA bin (-50 nt) ",ord,
           " kcal/mol.\nPooled, so mostly intron location; the separation is upstream only and panel B is the control."),
         x="position relative to site (nt)",y="U1 -dG (kcal/mol)")+
    th+theme(legend.position="bottom")
}

# b: within-gene
panel_B<-function(){
  D<-rd("u1_withingene.tsv")
  L<-D%>%filter(series!="difference")%>%mutate(series=factor(series,levels=c("real","competitor"),
        labels=c("real 3' end","competitor")))
  Q<-D%>%filter(series=="difference")
  wr<-sv("B_winrate"); dd<-sv("B_upstream_diff")
  main<-ggplot(L,aes(offset,value,color=series,fill=series))+UPSHADE()+ZERO()+
    geom_hline(yintercept=0,color=GREY,linewidth=.25)+
    geom_ribbon(aes(ymin=value-se,ymax=value+se),alpha=.16,color=NA)+geom_line(linewidth=.55)+
    scale_color_manual(values=c("real 3' end"=REAL,"competitor"=COMP))+
    scale_fill_manual(values=c("real 3' end"=REAL,"competitor"=COMP),guide="none")+
    scale_x_continuous(breaks=seq(-500,500,250))+
    labs(title="B   Same contrast inside the gene",
         subtitle=paste0("Gene-centred U1 -dG, 25 nt bins, ",nfmt(sv("n_paired_genes")),
           " paired genes. The competitor sits above its own gene's real end\nupstream: integrated -500..0 difference ",
           sprintf("%+.3f",dd)," kcal/mol [",sprintf("%+.3f",slo("B_upstream_diff")),", ",
           sprintf("%+.3f",shi("B_upstream_diff")),"], win-rate ",sprintf("%.3f",wr),
           ". Survives, and is modest."),
         x="position relative to site (nt)",y="U1 -dG, gene-centred")+
    th+theme(legend.position="bottom")
  ins<-ggplot(Q,aes(offset,value))+
    annotate("rect",xmin=-Inf,xmax=0,ymin=-Inf,ymax=Inf,fill=GREY,alpha=.06)+
    geom_hline(yintercept=0,color=GREY,linewidth=.25)+
    geom_ribbon(aes(ymin=value-se,ymax=value+se),alpha=.2,fill=INK)+
    geom_line(linewidth=.4,color=INK)+
    scale_x_continuous(breaks=c(-500,0,500))+
    labs(x=NULL,y="comp - real")+theme_classic(base_size=BASE-2)+
    theme(axis.text=element_text(color=INK,size=BASE-2.5),axis.title=element_text(size=BASE-2),
          axis.line=element_line(color=INK,linewidth=.25),plot.background=element_rect(fill="white",color=GREY,linewidth=.2))
  main+inset_element(ins,left=.60,bottom=.62,right=1.0,top=1.0,align_to="panel")
}

# c: distribution
panel_C<-function(){
  D<-rd("u1_scalar.tsv")
  E4<-D%>%filter(kind=="ecdf",thr=="4")%>%mutate(class=fc(class))
  E7<-D%>%filter(kind=="ecdf",thr=="7")%>%mutate(class=fc(class))
  H<-D%>%filter(kind=="delta")
  w4<-sv("C_winrate_thr4"); w7<-sv("C_winrate_thr7")
  main<-ggplot()+
    geom_step(data=E7,aes(x,value,color=class),linewidth=.35,alpha=.40,linetype="22")+
    geom_step(data=E4,aes(x,value,color=class),linewidth=.55)+
    scale_color_manual(values=setNames(PAL,LAB[LEV]))+
    scale_x_continuous(trans="log1p",breaks=c(0,3,10,30,100,300))+
    labs(title="C   U1 density at the site",
         subtitle=paste0("Strong-U1 density per kb over [-500,+250]. Solid: U1-favorable density (-dG>=4), win-rate ",
           sprintf("%.3f",w4),". Dashed:\npositioned U1 sites (-dG>=7), win-rate ",sprintf("%.3f",w7),
           ", ~10x rarer, so its ECDF sits left. Both above 0.5, both modest."),
         x="strong-U1 sites per kb (log1p)",y="ECDF")+
    th+theme(legend.position="bottom")
  ins<-ggplot(H,aes(x,value))+
    geom_col(width=diff(range(H$x))/45,fill=GREY,color=NA)+
    geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.3)+
    labs(x="real - mean competitor (/kb)",y="genes")+theme_classic(base_size=BASE-2)+
    theme(axis.text=element_text(color=INK,size=BASE-2.5),axis.title=element_text(size=BASE-2.2),
          axis.line=element_line(color=INK,linewidth=.25),plot.background=element_rect(fill="white",color=GREY,linewidth=.2))
  main+inset_element(ins,left=.50,bottom=.02,right=1.0,top=.47,align_to="panel")
}

# d: nearest strong donor
panel_D<-function(){
  D<-rd("nearest5ss.tsv")%>%mutate(class=fc(class),
        side=factor(side,levels=c("upstream","downstream"),
                    labels=c("upstream of the site","downstream of the site")))
  md<-function(s,c) sv(paste0("D_median_",s,"_",c))
  ggplot(D,aes(distance,ecdf,color=class))+geom_line(linewidth=.55)+
    facet_wrap(~side)+
    scale_color_manual(values=setNames(PAL,LAB[LEV]))+
    scale_x_continuous(breaks=seq(0,2000,500))+
    labs(title="D   Distance to the nearest strong 5'ss",
         subtitle=paste0("Median upstream ",md("upstream","real"),
           " / ",md("upstream","competitor")," / ",md("upstream","decoy"),
           " nt (real / competitor / decoy); downstream ",md("downstream","real"),
           " / ",md("downstream","competitor"),"\n/ ",md("downstream","decoy"),
           " nt. Real ends carry the donor-free stretch UPSTREAM, in the exon body. Downstream of the pA the three classes\nare indistinguishable: past a real 3' end there is no transcript to define."),
         x="distance to nearest 5'ss with MaxEnt >=8 (nt)",y="ECDF")+
    th+theme(legend.position="bottom")
}

panel_E<-function(){
  D<-rd("donor_flanked.tsv")%>%
    filter(property %in% c("acceptor","intervening_donor","donor_flanked_intervening","donor_flanked_dn300"))%>%
    mutate(class=fc(class),
           property=factor(property,
             levels=c("acceptor","intervening_donor","donor_flanked_intervening","donor_flanked_dn300"),
             labels=c("strong upstream\n3'ss","donor between\nanchor and pA",
                      "both\n(internal-exon-like)","acceptor + donor\npast the pA (null)")))
  ggplot(D,aes(property,frac,fill=class))+
    geom_col(position=position_dodge(width=.78),width=.7,color=INK,linewidth=.15)+
    geom_errorbar(aes(ymin=ci_lo,ymax=ci_hi),position=position_dodge(width=.78),
                  width=.18,linewidth=.3,color=INK)+
    scale_fill_manual(values=setNames(PAL,LAB[LEV]))+
    scale_y_continuous(limits=c(0,1),expand=expansion(mult=c(0,.02)))+
    labs(title="E   Terminal exon vs internal exon",
         subtitle=paste0("Real ends are acceptor-defined (",sprintf("%.2f",sv("E_acceptor_real")),
           " carry a strong upstream 3'ss) and least often carry the\nintervening donor that would make the span an internal exon: ",
           sprintf("%.3f",sv("E_donor_flanked_intervening_real")),"  vs  ",
           sprintf("%.3f",sv("E_donor_flanked_intervening_competitor")),"  vs  ",
           sprintf("%.3f",sv("E_donor_flanked_intervening_decoy")),",\nwithin-gene win-rate ",
           sprintf("%.3f",sv("E_winrate_intervening")),
           ". Past the pA the same property is null (win-rate ",sprintf("%.3f",sv("E_winrate_dn300")),")."),
         x=NULL,y="fraction of sites (Wilson 95%)")+
    th+theme(legend.position="bottom")
}

panel_F<-function(){
  D<-rd("u1_vs_5ss.tsv"); r<-D$r[1]; r7<-sv("F_pearson_r_thr7")
  ggplot(D,aes(x,y,fill=count))+geom_tile()+
    scale_fill_gradient(low="#EEF2F7",high="#12355B",trans="log10",name="sites")+
    labs(title="F   U1 and the 5'ss are one axis, weakly",
         subtitle=paste0("Same sites, same [-500,+250] window, all three classes pooled.\nPearson r = ",
           sprintf("%.2f",r)," for the U1-favorable field, ",sprintf("%.2f",r7),
           " for positioned U1 sites (-dG>=7).\nU1 is not a second, independent donor signal; it is a blurred read of the same one."),
         x="U1 sites (-dG>=4) per kb",y="strong 5'ss (MaxEnt>=8) in the window")+
    th+theme(legend.position="right",legend.key.width=unit(.5,"lines"),
             legend.key.height=unit(1.1,"lines"),legend.title=element_text(size=BASE-1))
}

CAP<-paste0("n = ",nfmt(sv("n_real"))," real 3' ends, ",nfmt(sv("n_competitor")),
  " competitors and ",nfmt(sv("n_decoy")),
  " intronic decoys (seed 42), all in the ",nfmt(sv("n_paired_genes")),
  " genes of the cpa_peaks_final.parquet analysis set; competitors are baseline-model false positives after APA-annotated peaks,\n",
  "reference terminal exons and at-or-downstream-of-canonical peaks are removed. THE CONFOUND, PLAINLY: competitors sit inside introns and real 3' ends do not, so every pooled panel (A, C, D) mixes the\n",
  "site contrast with intron location. Only the within-gene panels are confound-free, and there the effect is small: win-rate ",
  sprintf("%.3f",sv("B_winrate"))," (U1 upstream), ",sprintf("%.3f",sv("C_winrate_thr4")),
  " (U1 density), ",sprintf("%.3f",sv("E_winrate_intervening")),
  " (internal-exon architecture), against 0.606 for the acceptor.\n",
  "Win-rate is the gene-averaged P(competitor > real), ties 0.5, bootstrapped over genes. U1 and the 5'ss are treated as one donor measurement (F). Ribbons and bars are 1 SE / Wilson 95%.")

fig<-(panel_A()|panel_B())/(panel_C()|panel_D())/(panel_E()|panel_F())+
  plot_annotation(
    title="U1 and the donor 5'ss describe terminal-exon architecture, they do not predict the 3' end",
    subtitle="Landscape, distribution and mechanistic property, each pooled view paired with its within-gene or decoy control",
    caption=CAP,
    theme=theme(plot.title=element_text(size=BASE+3,face="bold",color=INK),
                plot.subtitle=element_text(size=BASE,color="#555555"),
                plot.caption=element_text(size=BASE-1.4,color="#555555",hjust=0,lineheight=1.25)))

ggsave("figure_u1_5ss_landscape.pdf",fig,width=9.6,height=10.6,device=grDevices::cairo_pdf)
ggsave("figure_u1_5ss_landscape.png",fig,width=9.6,height=10.6,dpi=300)
ggsave("figure_u1_landscape_ABC.pdf",(panel_A()/panel_B()/panel_C()),width=5.2,height=10.6,device=grDevices::cairo_pdf)
ggsave("figure_5ss_architecture_DEF.pdf",(panel_D()/panel_E()/panel_F()),width=5.2,height=10.6,device=grDevices::cairo_pdf)
cat("[out] figure_u1_5ss_landscape.pdf figure_u1_5ss_landscape.png",
    "figure_u1_landscape_ABC.pdf figure_5ss_architecture_DEF.pdf\n")
