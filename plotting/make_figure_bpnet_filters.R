#!/usr/bin/env Rscript
# every first-layer filter of the bpnet variant
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(tidyr)
                               library(patchwork);library(ggseqlogo)})
e<-function(...) cat(...,"\n",file=stderr())
INK<-"#1A1A1A"; BASE<-7; BLUE<-"#08519C"; ORNG<-"#E8A33D"; GREY<-"#8A8F98"; GTX<-"#5A6068"

F<-read_tsv("bpnet_filters_all.tsv",show_col_types=FALSE)
L<-read_tsv("bpnet_logos_all.tsv",show_col_types=FALSE)
O<-read_tsv("cnn_filters_matched.tsv",show_col_types=FALSE)
V<-read_tsv("bpnet_verdict.tsv",col_types=cols(.default=col_character()))
nv<-function(k) V$value[V$key==k][1]
NFIL<-nrow(F); W<-F$width[1]
ORD<-F%>%arrange(desc(ic_per_col))%>%pull(filter)

th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-1),axis.line=element_line(color=INK,linewidth=.3),
  axis.title=element_text(size=BASE),plot.title=element_text(size=BASE+1,face="bold",hjust=0,
  margin=margin(b=4)),legend.position="none")

# a: logo grid, all filters
mats<-lapply(ORD,function(f){
  m<-L%>%filter(filter==f)%>%select(position,base,prob)%>%
     pivot_wider(names_from=position,values_from=prob)%>%as.data.frame()
  rn<-m$base; m<-as.matrix(m[,-1]); rownames(m)<-rn; m[c("A","C","G","T"),,drop=FALSE]})
names(mats)<-sapply(ORD,function(f){
  r<-F[F$filter==f,]
  sprintf("%s   %.2f b/col   %s sites   %s%s", sub("filter","filter ",f), r$ic_per_col,
          format(r$n_sites,big.mark=","),
          ifelse(r$best_match=="none","no reference match",r$best_match),
          ifelse(r$best_match=="none","",sprintf(" (r = %.2f)",r$r)))})
pA<-ggseqlogo(mats,method="bits",ncol=4)+
  labs(title=sprintf("A   All %d first-layer filters of the BPNet variant, ordered by information content",NFIL))+
  theme_classic(base_size=BASE)+
  theme(text=element_text(color=INK),legend.position="none",
        strip.background=element_blank(),
        strip.text=element_text(size=BASE-1.6,color=GTX,hjust=0,margin=margin(b=1,t=3)),
        axis.text.x=element_blank(),axis.ticks.x=element_blank(),
        axis.text.y=element_text(size=BASE-2.6,color=INK),
        axis.line=element_line(color=INK,linewidth=.25),
        plot.title=element_text(size=BASE+1,face="bold",hjust=0,margin=margin(b=6)))

# b: ic comparison
CMP<-bind_rows(
  data.frame(model=sprintf("BPNet variant (%d filters, %d nt)",NFIL,W),ic=F$ic_per_col),
  data.frame(model="Reported CNN (32 filters, 19 nt)",ic=O$ic_per_col))
MED<-CMP%>%group_by(model)%>%summarise(m=median(ic),.groups="drop")
pB<-ggplot(CMP,aes(ic,model,color=I(ifelse(grepl("BPNet",model),BLUE,GREY))))+
  geom_point(position=position_jitter(height=.14,seed=1),size=1.4,alpha=.85)+
  geom_point(data=MED,aes(m,model),shape=124,size=6,color=INK)+
  geom_text(data=MED,aes(m,model,label=sprintf("median %.3f",m)),vjust=-2.6,size=2.1,color=INK)+
  coord_cartesian(xlim=c(0.28,0.98))+scale_x_continuous(breaks=seq(.3,1,.1))+
  labs(title="B   Information content per filter, count-matched",
       x="Information content (bits per column, maximum 2.0)",y=NULL)+th

# c: are the matches motifs or at composition?
FC<-bind_rows(F%>%transmute(frac_AT,r,best_match,ic_per_col,filter,mdl="BPNet variant"),
              O%>%transmute(frac_AT,r,best_match,ic_per_col,filter,mdl="Reported CNN"))%>%
    mutate(matched=best_match!="none")
pC<-ggplot(FC,aes(frac_AT,r,color=I(ifelse(matched,BLUE,GREY)),
                  shape=I(ifelse(mdl=="BPNet variant",16,1))))+
  geom_vline(xintercept=.5,linetype=2,color=GREY,linewidth=.35)+
  geom_point(size=1.6,alpha=.9)+
  geom_text(data=FC%>%group_by(mdl)%>%filter(ic_per_col>=quantile(ic_per_col,.94))%>%ungroup(),
            aes(label=sub("filter","",filter)),size=1.9,hjust=-.5,color=INK)+
  annotate("text",x=.44,y=.30,hjust=0,size=2.0,color=GTX,
           label="filled = BPNet, open = reported;  blue = passes the null")+
  coord_cartesian(xlim=c(.42,.95))+scale_x_continuous(breaks=seq(.45,.9,.1))+
  labs(title="C   Reference match against A+T composition",
       x="A+T share of the filter's PWM mass",
       y="Correlation with its best reference motif")+th

CAP<-paste0(
 "BPNet variant, the model that ranks +", nv("winrate_delta"), " above the reported CNN ",
 "(within-gene ", nv("winrate_new"), " against ", nv("winrate_reported"), ").\n",
 "Panel A: every filter is thresholded at its own 99.9th activation percentile, which admits the ",
 "same share of positions in both models, so information content is comparable across ",
 "architectures and is reported per column because the widths differ.\n",
 "Panel B: the variant's filters are NOT sharper. Median ", sprintf("%.3f",median(F$ic_per_col)),
 " against ", sprintf("%.3f",median(O$ic_per_col)), " bits per column, although its single ",
 "cleanest filter reaches ", sprintf("%.3f",max(F$ic_per_col)), " against ",
 sprintf("%.3f",max(O$ic_per_col)), ". ", sum(F$best_match!="none"), " of ", NFIL,
 " filters pass the random-PWM null against ", sum(O$best_match!="none"), " of ", nrow(O),
 ".\n",
 "Panel C: the reference panel is AT-rich by construction, since AATAAA is A5T1 and the ",
 "acceptor reference is a pyrimidine tract, so an AT-rich filter correlates with it without ",
 "encoding a motif. Filters that pass the random-PWM null carry a higher A+T share than those ",
 "that do not, and the exception is the pure polypyrimidine tract at the top of panel A.\n",
 "Both models are shown count-matched throughout; the reported model's published information ",
 "content in filter_matches_reference.tsv is the 30%-of-maximum estimate and is not comparable ",
 "with these numbers.")

FIG<-pA/(pB|pC)+plot_layout(heights=c(2.6,1))+
  plot_annotation(title="First-layer filters of the BPNet variant",caption=CAP,
    theme=theme(plot.title=element_text(size=BASE+3,face="bold",color=INK,hjust=0,
                                        margin=margin(b=6)),
                plot.caption=element_text(size=BASE-2,color=GTX,hjust=0,lineheight=1.3,
                                          margin=margin(t=8))))
ggsave("figure_bpnet_filters.pdf",FIG,width=11.5,height=15.5,device=grDevices::cairo_pdf,
       limitsize=FALSE)
ggsave("figure_bpnet_filters.png",FIG,width=11.5,height=15.5,dpi=210,limitsize=FALSE)
e("[filters]",NFIL,"filters of width",W,"| IC",sprintf("%.3f-%.3f (median %.3f)",
  min(F$ic_per_col),max(F$ic_per_col),median(F$ic_per_col)),"bits/col")
e("[filters] reported model count-matched: IC",sprintf("%.3f-%.3f (median %.3f)",
  min(O$ic_per_col),max(O$ic_per_col),median(O$ic_per_col)),"| matched",
  sum(O$best_match!="none"),"/",nrow(O))
e("[filters] matching a reference:",sum(F$best_match!="none"),"/",NFIL,
  "| A+T share matched",sprintf("%.3f",mean(F$frac_AT[F$best_match!="none"])),
  "vs unmatched",sprintf("%.3f",mean(F$frac_AT[F$best_match=="none"])))
e("[out] figure_bpnet_filters.pdf figure_bpnet_filters.png")
