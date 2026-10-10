#!/usr/bin/env Rscript
# supplementary: distance from the anchor 3'ss to the nearest strong 5'ss, by class
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr)})
args<-commandArgs(trailingOnly=TRUE)
INDIR<-if(length(args)>=1) args[1] else "."
PFX  <-if(length(args)>=2) args[2] else "fig3"
need<-function(p){ if(!file.exists(p)) stop("missing required input: ",p,call.=FALSE); p }
rd<-function(f) read_tsv(need(file.path(INDIR,paste0(PFX,"_",f))),show_col_types=FALSE)

INK<-"#1A1A1A"; BASE<-13; CAN<-"#0072B2"; COMP<-"#E69F00"
th<-theme_classic(base_size=BASE)+
  theme(text=element_text(color=INK),axis.text=element_text(color=INK,size=BASE-4),
        axis.line=element_line(color=INK,linewidth=.4),
        plot.title=element_text(size=BASE,face="bold",hjust=0),
        plot.subtitle=element_text(size=BASE-5,color="#555555"),
        legend.title=element_blank(),legend.text=element_text(size=BASE-5),
        legend.key.height=unit(.8,"lines"),
        plot.caption=element_text(size=BASE-6,color="#666666",hjust=0))

F5 <-rd("first5_from_anchor.tsv")
IEL<-rd("internal_exon_like.tsv")
CC <-rd("class_counts.tsv")
F5$cls<-factor(F5$cls,levels=c("Canonical","Competitor"))
ncan <-as.numeric(CC$n[CC$cls=="Canonical"][1])
ncomp<-as.numeric(CC$n[CC$cls=="Competitor"][1])
iel<-function(c,col) IEL[[col]][IEL$cls==c][1]
fc<-iel("Canonical","frac_internal_exon_like"); fk<-iel("Competitor","frac_internal_exon_like")
nF5<-nrow(F5); ncanF<-sum(F5$cls=="Canonical"); ncompF<-sum(F5$cls=="Competitor")

sub<-sprintf(paste0(
  "Competitors carry a strong 5′ss at internal-exon distance more often than real ends: %.1f%% vs %.1f%% (%.2fx). The effect is modest and\n",
  "most sites carry none there: %.1f%% of competitors have no strong 5′ss at 50-300nt. Only %s of %s peaks have a scorable nearest 5′ss,\n",
  "and the scorable rate itself differs by class (%.0f%% of real ends, %.0f%% of competitors), so the panel conditions on a class-correlated quantity."),
  100*fk,100*fc,fk/fc,100*(1-fk),format(nF5,big.mark=","),
  format(ncan+ncomp,big.mark=","),100*ncanF/ncan,100*ncompF/ncomp)

p<-ggplot(F5,aes(value,fill=cls,color=cls))+
  annotate("rect",xmin=50,xmax=300,ymin=-Inf,ymax=Inf,fill="grey70",alpha=.16)+
  annotate("text",x=175,y=Inf,vjust=1.6,label="internal-exon length",size=2.9,color="#555555")+
  geom_density(alpha=.30,linewidth=.7)+
  scale_fill_manual(values=c(Canonical=CAN,Competitor=COMP))+
  scale_color_manual(values=c(Canonical=CAN,Competitor=COMP))+
  coord_cartesian(xlim=c(0,800))+
  labs(title="S1   Distance from the anchor 3′ss to the nearest strong 5′ss",
       subtitle=sub,
       x="distance from anchor 3′ss to nearest strong 5′ss (nt)",y="density",
       caption=sprintf(paste0("Wilson 95%% CI on the 50-300nt fraction: real ends %.4f [%.4f, %.4f], n = %s; competitors %.4f [%.4f, %.4f], n = %s. ",
                "Strong 5′ss is MaxEnt >= 8; distance is scored in both directions from the anchor."),
                fc,iel("Canonical","lo"),iel("Canonical","hi"),format(iel("Canonical","n"),big.mark=","),
                fk,iel("Competitor","lo"),iel("Competitor","hi"),format(iel("Competitor","n"),big.mark=",")))+
  th+theme(legend.position="top")

ggsave(file.path(INDIR,"figure3_supp_position.pdf"),p,width=11,height=6,device=cairo_pdf)
ggsave(file.path(INDIR,"figure3_supp_position.png"),p,width=11,height=6,dpi=300)
cat(sprintf("[S1] 50-300nt: real end %.4f, competitor %.4f (%.3fx); %d of %d peaks scorable\n",
            fc,fk,fk/fc,nF5,ncan+ncomp))
cat("[out] figure3_supp_position.pdf / .png\n")
