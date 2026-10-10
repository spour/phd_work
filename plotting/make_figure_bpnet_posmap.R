#!/usr/bin/env Rscript
# where the bpnet variant reads, and how that differs from the reported model
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(tidyr)
                               library(patchwork);library(ggseqlogo)})
e<-function(...) cat(...,"\n",file=stderr())
INK<-"#1A1A1A"; BASE<-7; BLUE<-"#08519C"; ORNG<-"#E8A33D"; GREY<-"#8A8F98"; GTX<-"#5A6068"

PM<-read_tsv("bpnet_posmap.tsv",show_col_types=FALSE)
BP<-read_tsv("bpnet_profile.tsv",show_col_types=FALSE)
OC<-read_tsv("occlusion_multiscale.tsv",show_col_types=FALSE)%>%filter(width==100)
FL<-read_tsv("bpnet_filters_all.tsv",show_col_types=FALSE)
LG<-read_tsv("bpnet_logos_all.tsv",show_col_types=FALSE)
RFN<-376; RFO<-378

th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-1),axis.line=element_line(color=INK,linewidth=.3),
  axis.title=element_text(size=BASE),plot.title=element_text(size=BASE+1,face="bold",hjust=0,
  margin=margin(b=4)),legend.position="none")

# a: profile overlay, as a fraction of each model's own lift
PR<-bind_rows(
  BP%>%transmute(model="BPNet variant",centre,frac=d_frac_of_lift),
  OC%>%transmute(model="Reported CNN",centre,frac=d_winrate/(intact_winrate-0.5)))
stat<-PR%>%group_by(model)%>%summarise(peak=centre[which.max(frac)],
  cen=sum(centre*pmax(frac,0))/sum(pmax(frac,0)),.groups="drop")
pA<-ggplot(PR,aes(centre,frac,color=I(ifelse(model=="BPNet variant",BLUE,GREY))))+
  geom_hline(yintercept=0,linetype=2,color=GREY,linewidth=.35)+
  geom_line(linewidth=.6)+
  geom_point(data=stat,aes(peak,0),shape=17,size=1.8)+
  annotate("text",x=-1980,y=c(.070,.062),hjust=0,size=2.2,color=c(BLUE,GREY),
           label=sprintf("%s   peak %+.0f nt, centroid %+.0f nt",stat$model,stat$peak,stat$cen))+
  scale_x_continuous(breaks=seq(-2000,500,500))+
  labs(title="A   Occlusion positional profile, both models",
       x="Position relative to the cleavage site (nt)",
       y="Win-rate drop, as a fraction of the model's own lift over 0.5")+th

# b: full filter x position map
FO<-FL%>%arrange(desc(ic_per_col))%>%pull(filter)
PM2<-PM%>%mutate(filter=factor(filter,levels=rev(FO)),
                 centre=(bin_start+bin_end)/2)
LIM<-max(abs(PM2$contrib))
lab<-FL%>%mutate(txt=sprintf("%s  %.2f  %s",sub("filter","f",filter),ic_per_col,
                 ifelse(best_match=="none","-",sub("_.*","",best_match))))%>%
     select(filter,txt)
pB<-ggplot(PM2,aes(centre,filter,fill=contrib))+geom_tile()+
  scale_fill_gradient2(low=ORNG,mid="white",high=BLUE,midpoint=0,limits=c(-LIM,LIM))+
  scale_y_discrete(labels=setNames(lab$txt,lab$filter))+
  scale_x_continuous(breaks=seq(-2000,500,500),expand=c(0,0))+
  labs(title="B   Filter by position contribution, every filter",
       x="Position relative to the cleavage site (nt)",y=NULL)+
  th+theme(axis.text.y=element_text(size=BASE-3),axis.ticks.y=element_blank(),
           axis.line.y=element_blank())

# c: strongest filters, with logos as row labels
TOPN<-10
top<-PM%>%group_by(filter)%>%summarise(m=max(abs(contrib)),.groups="drop")%>%
     arrange(desc(m))%>%slice_head(n=TOPN)%>%pull(filter)
PT<-PM%>%filter(filter %in% top)%>%
    mutate(filter=factor(filter,levels=rev(top)),centre=(bin_start+bin_end)/2)
LIM2<-max(abs(PT$contrib))
pC<-ggplot(PT,aes(centre,filter,fill=contrib))+geom_tile()+
  scale_fill_gradient2(low=ORNG,mid="white",high=BLUE,midpoint=0,limits=c(-LIM2,LIM2),
                       name="contribution")+
  scale_y_discrete(labels=setNames(sub("filter","filter ",top),top))+
  scale_x_continuous(breaks=seq(-2000,500,500),expand=c(0,0))+
  labs(title="C   Strongest filters, with their logos",
       x="Position relative to the cleavage site (nt)",y=NULL)+
  th+theme(legend.position="right",legend.key.width=unit(6,"pt"),
           legend.key.height=unit(16,"pt"),legend.title=element_text(size=BASE-2),
           legend.text=element_text(size=BASE-2))
mk<-function(f){
  m<-LG%>%filter(filter==f)%>%select(position,base,prob)%>%
     pivot_wider(names_from=position,values_from=prob)%>%as.data.frame()
  rn<-m$base; m<-as.matrix(m[,-1]); rownames(m)<-rn; m[c("A","C","G","T"),,drop=FALSE]}
mats<-lapply(top,mk)
names(mats)<-sapply(top,function(f){r<-FL[FL$filter==f,]
  sprintf("%s  %.2f b/col  %s",sub("filter","filter ",f),r$ic_per_col,
          ifelse(r$best_match=="none","no match",r$best_match))})
pL<-ggseqlogo(mats,method="bits",ncol=1)+
  labs(title=" ")+theme_classic(base_size=BASE)+
  theme(text=element_text(color=INK),legend.position="none",strip.background=element_blank(),
        strip.text=element_text(size=BASE-2.4,color=GTX,hjust=0,margin=margin(b=0,t=2)),
        axis.text=element_blank(),axis.ticks=element_blank(),axis.line=element_blank(),
        axis.title=element_blank(),
        plot.title=element_text(size=BASE+1,face="bold",hjust=0,margin=margin(b=4)))

CAP<-paste0(
 "BPNet variant against the reported CNN. Both read a window of [-2,000, +500] and one output ",
 "bin of each sees almost the same amount of sequence: ",RFN," nt for the variant (25 nt first ",
 "layer, dilations 2 to 64, 100 nt pool) against ",RFO," nt for the reported model, matched on ",
 "purpose so a difference here is not a difference in how much they can see.\n",
 "Panel A: each profile is divided by its own model's lift over 0.5, because the variant starts ",
 "from a higher win-rate and raw drops would not be comparable. Windows are 100 nt on a 50 nt ",
 "grid, ",nrow(BP)," of them, identical for both models.\n",
 "Panels B and C: one filter is silenced in one 100 nt band at a time and the model is re-scored. ",
 "Positive, blue, means the silencing costs the real 3' end more than it costs its competitors, ",
 "so the filter argues for the real end at that position; orange is the opposite. ",
 PM$n_sites[1]," sites in ",PM$n_genes[1]," test genes, ",nrow(FL)," filters x 25 bands.\n",
 "Row labels in panel B give the filter, its count-matched information content in bits per ",
 "column, and its best reference match where it passes the random-PWM null.")

FIG<-pA/pB/(pL|pC)+plot_layout(heights=c(0.8,2.1,1.5))+
  plot_annotation(title="Where the BPNet variant reads, and how that differs from the reported model",
    caption=CAP,
    theme=theme(plot.title=element_text(size=BASE+3,face="bold",color=INK,hjust=0,
                                        margin=margin(b=6)),
                plot.caption=element_text(size=BASE-2,color=GTX,hjust=0,lineheight=1.3,
                                          margin=margin(t=8))))
ggsave("figure_bpnet_posmap.pdf",FIG,width=11.0,height=13.5,device=grDevices::cairo_pdf,
       limitsize=FALSE)
ggsave("figure_bpnet_posmap.png",FIG,width=11.0,height=13.5,dpi=220,limitsize=FALSE)
e("[profile] peak/centroid:",paste(sprintf("%s %+.0f/%+.0f",stat$model,stat$peak,stat$cen),
                                   collapse="  "))
e("[posmap] contrib range",sprintf("%+.4f to %+.4f",min(PM$contrib),max(PM$contrib)),
  "| strongest filters:",paste(sub("filter","",top),collapse=","))
e("[out] figure_bpnet_posmap.pdf figure_bpnet_posmap.png")
