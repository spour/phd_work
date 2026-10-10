#!/usr/bin/env Rscript
# the three synthetic 3' splice site units on one dose axis
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(tidyr);library(readr)
                               library(patchwork)})
INK<-"#1A1A1A"; NUL<-"#8A8F98"; GTX<-"#5A6068"; FAINT<-"#E3E6E9"; BASE<-7.6
PAL<-c(acc_ppt="#B03A2E", acc_noC="#E8A33D", acc_best="#08519C")
LAB<-c(acc_ppt="TTTTTTTTTTTTTCAGG   T-rich tract + AG, hand-built",
       acc_noC="TTTTTTTTTGTGTGTTTTAGGTT   T-rich, score3 14.38",
       acc_best="TTTCCTCTCTCTCCCCGCAGGTG   MaxEnt optimum, 43% C")
ORD<-c("acc_ppt","acc_noC","acc_best")
th<-theme_classic(base_size=BASE,base_family="sans")+theme(
  plot.title=element_text(size=BASE+1.2,face="bold",colour=INK),
  axis.text=element_text(colour=INK),axis.title=element_text(colour=GTX,size=BASE-0.4),
  legend.key.size=unit(9,"pt"),legend.text=element_text(size=BASE-0.9),
  legend.title=element_blank(),plot.tag=element_text(size=BASE+3,face="bold"))
rd<-function(f) read_tsv(f,show_col_types=FALSE,progress=FALSE)
FILES<-c("gia_dose_matched_ppt.tsv","gia_dose_matched_all.tsv","gia_dose_best.tsv")
FILES<-FILES[file.exists(FILES)]
G<-bind_rows(lapply(FILES,function(f) rd(f)%>%mutate(src=f)))%>%
   filter(background=="real_canonical", site %in% ORD)
PREF<-G%>%group_by(site)%>%summarise(src=src[which.max(n_bg)][1],.groups="drop")
G<-G%>%inner_join(PREF,by=c("site","src"))
B<-G%>%distinct(site,seq_id,arm,n_copies,best_delta,best_se,n_bg)
MOT<-B%>%filter(arm=="motif")%>%transmute(site,n=n_copies,raw=best_delta,se=best_se,n_bg)
MON<-B%>%filter(arm=="mono")%>%group_by(site,n=n_copies)%>%
     summarise(c=mean(best_delta),csd=sd(best_delta),k=n(),.groups="drop")
stopifnot(all(MON$k==10))
D<-MOT%>%left_join(MON,by=c("site","n"))%>%
   mutate(dif=raw-c, dse=sqrt(se^2+(csd/sqrt(10))^2),
          s=factor(LAB[site],levels=unname(LAB[ORD])))
NB<-D%>%group_by(site)%>%summarise(n_bg=max(n_bg),.groups="drop")
LONG<-D%>%select(site,s,n,raw,se,dif,dse)%>%
  pivot_longer(c(raw,dif),names_to="k",values_to="v")%>%
  mutate(e=ifelse(k=="raw",se,dse),
         k=factor(c(raw="Raw effect of implanting",
                    dif="Minus the unit's OWN base composition")[k],
                  levels=c("Raw effect of implanting","Minus the unit's OWN base composition")))
ENDS<-LONG%>%filter(n %in% c(1,7))
p<-ggplot(LONG,aes(n,v,colour=s,fill=s))+
  geom_hline(yintercept=0,linetype="2222",colour=NUL,linewidth=.4)+
  geom_ribbon(aes(ymin=v-e,ymax=v+e),alpha=.16,colour=NA)+
  geom_line(linewidth=.7)+geom_point(size=1.5)+
  geom_text(data=ENDS,aes(label=sprintf("%+.3f",v)),
            vjust=ifelse(ENDS$v>0,-1.0,1.9),hjust=ifelse(ENDS$n==1,0.15,0.85),
            size=2.2,show.legend=FALSE)+
  facet_wrap(~k,nrow=1)+
  scale_colour_manual(values=setNames(unname(PAL[ORD]),unname(LAB[ORD])))+
  scale_fill_manual(values=setNames(unname(PAL[ORD]),unname(LAB[ORD])))+
  scale_x_continuous(breaks=1:7)+
  scale_y_continuous(breaks=seq(-1,0.5,.25))+
  labs(title="A T-rich tract with an AG moves the model; an acceptor does not",
       x="Tandem copies implanted",y="Change in model score")+
  th+theme(legend.position="bottom",legend.direction="vertical",
           legend.margin=margin(t=-2),
           strip.background=element_blank(),
           strip.text=element_text(size=BASE-0.4,face="bold",colour=INK,hjust=0),
           panel.spacing.x=unit(11,"pt"))
CAP<-paste0(
"Each unit implanted at 60 positions in real held-out 3' end windows and reported at the position most favourable to it, so a null is not an artifact\n",
"of placement. The dashed control is the same construct with its bases shuffled, ten draws per copy number; the right panel is the difference, and the\n",
"band is its standard error. Backgrounds per unit: ",
paste(sprintf("%s %s", NB$site, format(NB$n_bg,big.mark=",")), collapse="; "),".\n",
"READING. The two T-rich units carry the large raw effects at one copy, +0.204 and +0.281, and BOTH sit below a shuffle of their own bases at every\n",
"dose. Adding more of them drives the score down, to -0.538 and -0.779 at seven copies. The MaxEnt-optimal acceptor, the only one of the three that is\n",
"a strong acceptor by its own measure at score3 17.17, has the smallest raw effect and the ONLY positive composition-corrected one, +0.032 at seven\n",
"copies, which reaches z = +6.0 when measured on all 14,198 held-out windows.\n",
"So the boost belongs to the bases and survives reordering them, and what survives the reordering is not the tract but the acceptor, at a fortieth of\n",
"the size. The dose collapse is the tell: a model reading acceptors would not fall as acceptors are added.")
FIG<-p+plot_annotation(caption=CAP,
  theme=theme(plot.caption=element_text(size=BASE-1.9,colour=GTX,hjust=0,lineheight=1.3,
                                        family="sans",margin=margin(t=8))))
ggsave("figure_acceptor_dose.pdf",FIG,width=9.6,height=5.4,device=cairo_pdf)
ggsave("figure_acceptor_dose.png",FIG,width=9.6,height=5.4,dpi=220)
cat("[out] figure_acceptor_dose.pdf figure_acceptor_dose.png\n")
for (u in ORD) { d<-D%>%filter(site==.env$u)
  cat(sprintf("  %-9s n_bg %6d  raw n1 %+.3f n7 %+.3f   minus comp n1 %+.3f n7 %+.3f\n",
      u, max(d$n_bg), d$raw[d$n==1], d$raw[d$n==7], d$dif[d$n==1], d$dif[d$n==7])) }
