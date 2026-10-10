# the dose table against four negatives, drawn
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(patchwork)})
INK<-"#1A1A1A"; BASE<-13; CAN<-"#0072B2"; COMP<-"#E69F00"; NUL<-"#8A8F98"; GRID<-"#E3E5E8"
D<-read.delim("dose_null_panel.tsv",stringsAsFactors=FALSE)
ORD<-c("model_me3","robust_agdep","maxent_optimum","t_rich_noC","u1_donor")
SHOW<-list(model_me3=c(1,7),robust_agdep=c(1,7),maxent_optimum=c(1,7),
           t_rich_noC=c(1,7),u1_donor=c(1,7,14))
AGPOS<-list(model_me3=c(19,20),robust_agdep=c(19,20),maxent_optimum=c(19,20),
            t_rich_noC=c(19,20),u1_donor=c(4,5))
CW<-1.06; X0<-1.0
XDESC<-30; XN<-58; XLEN<-65; XRAW<-76; XSH<-85; XDG<-94; XSF<-103; XZ<-110; XCEIL<-119; XSTRIP<-122; SW<-0.92
LIM<-2
sq<-function(v) pmax(pmin(v,LIM),-LIM)
rows<-do.call(rbind,lapply(ORD,function(s) data.frame(name=s,n=SHOW[[s]],stringsAsFactors=FALSE)))
rows<-rows%>%left_join(D,by=c("name","n"="n_copies"))%>%mutate(y=-seq_len(n()))
gl<-do.call(rbind,lapply(seq_len(nrow(rows)),function(i){
  ch<-strsplit(rows$unit[i],"")[[1]]; k<-AGPOS[[rows$name[i]]]
  data.frame(x=X0+(seq_along(ch)-1)*CW,y=rows$y[i],lab=ch,hi=seq_along(ch)%in%k)}))
ymid<-rows%>%group_by(name)%>%summarise(y=mean(y),.groups="drop")
st<-D%>%filter(name%in%ORD)%>%inner_join(ymid,by="name")%>%
  mutate(x=XSTRIP+(n_copies-1)*SW)
hdr<-data.frame(x=c(X0,XDESC,XN,XLEN,XRAW,XSH,XDG,XSF,XZ,XCEIL,XSTRIP),
  lab=c("implanted unit","what it is","n","nt","raw","- shifts","- derang","- shuffle","z","past ceiling","every dose"),
  h=c(0,0,1,1,1,1,1,1,1,1,0))
sep<-rows$y[!duplicated(rows$name)][-1]+0.5
NB<-format(D$n_bg[1],big.mark=",")
pT<-ggplot()+
  geom_hline(yintercept=-0.45,colour=INK,linewidth=.45)+
  geom_hline(yintercept=sep,colour=GRID,linewidth=.35)+
  geom_text(data=hdr,aes(x,0,label=lab,hjust=h),size=3.0,colour=NUL,fontface="bold")+
  geom_text(data=gl,aes(x,y,label=lab,colour=hi),family="mono",size=2.8,show.legend=FALSE)+
  scale_colour_manual(values=c(`FALSE`=INK,`TRUE`=CAN))+
  geom_text(data=rows,aes(XDESC,y,label=description),hjust=0,size=3.1,colour=INK)+
  geom_text(data=rows,aes(XN,y,label=n),hjust=1,size=3.1,colour=INK)+
  geom_text(data=rows,aes(XLEN,y,label=insert_len),hjust=1,size=3.1,colour=NUL)+
  geom_tile(data=rows,aes(XRAW-2.6,y,fill=sq(raw)),width=8.4,height=.78)+
  geom_tile(data=rows,aes(XSH-2.6,y,fill=sq(minus_shift)),width=8.4,height=.78)+
  geom_tile(data=rows,aes(XDG-2.6,y,fill=sq(minus_derange)),width=8.4,height=.78)+
  geom_tile(data=rows,aes(XSF-2.6,y,fill=sq(minus_shuffle)),width=8.4,height=.78)+
  geom_text(data=rows,aes(XRAW,y,label=sprintf("%+.3f",raw)),hjust=1,size=3.1,colour=INK)+
  geom_text(data=rows,aes(XSH,y,label=sprintf("%+.3f",minus_shift)),hjust=1,size=3.1,colour=INK)+
  geom_text(data=rows,aes(XDG,y,label=sprintf("%+.3f",minus_derange)),hjust=1,size=3.1,
            fontface="bold",colour=INK)+
  geom_text(data=rows,aes(XSF,y,label=sprintf("%+.3f",minus_shuffle)),hjust=1,size=3.1,colour=INK)+
  geom_text(data=rows,aes(XZ,y,label=sprintf("%+.1f",z_vs_derange)),hjust=1,size=3.0,
            colour=ifelse(abs(rows$z_vs_derange)>=2,INK,NUL))+
  geom_text(data=rows,aes(XCEIL,y,label=sprintf("%.1f%%",pct_past_observed_max)),hjust=1,size=3.0,
            colour=ifelse(rows$pct_past_observed_max>=5,COMP,NUL))+
  geom_tile(data=st,aes(x,y,fill=sq(minus_derange)),width=SW*.86,height=.62,
            colour="white",linewidth=.18)+
  scale_fill_gradient2(low=COMP,mid="#FFFFFF",high=CAN,midpoint=0,limits=c(-LIM,LIM))+
  coord_cartesian(xlim=c(0,XSTRIP+14*SW+2),ylim=c(-nrow(rows)-.7,1.0),expand=FALSE,clip="off")+
  labs(title="The splice sites, against four negatives",
       subtitle=sprintf("%s real 3' end windows, fixed position -230, no maximum taken. Blue the model scores higher, orange lower.",NB))+
  theme_void(base_size=BASE)+
  theme(plot.title=element_text(face="bold",colour=INK,size=BASE,margin=margin(b=1)),
        plot.subtitle=element_text(colour=NUL,size=BASE-4,margin=margin(b=5)),
        legend.position="none",plot.margin=margin(4,6,4,6))
# the off-scale row, kept apart because it does not share the axis
F7<-D%>%filter(name=="free_max")
fl<-data.frame(x=X0+(seq_len(nchar(F7$unit[1]))-1)*CW,y=0,
               lab=strsplit(F7$unit[1],"")[[1]])
pF<-ggplot()+
  geom_text(data=fl,aes(x,y,label=lab),family="mono",size=2.8,colour=NUL)+
  geom_text(aes(XDESC,0,label="no constraint, not a splice site, score3 -24.79"),hjust=0,
            size=3.1,colour=NUL)+
  geom_text(aes(XN,0,label="1"),hjust=1,size=3.1,colour=NUL)+
  geom_text(aes(XLEN,0,label="23"),hjust=1,size=3.1,colour=NUL)+
  geom_text(aes(XRAW,0,label=sprintf("%+.3f",F7$raw[F7$n_copies==1])),hjust=1,size=3.1,colour=NUL)+
  geom_text(aes(XSH,0,label=sprintf("%+.3f",F7$minus_shift[F7$n_copies==1])),hjust=1,size=3.1,colour=NUL)+
  geom_text(aes(XDG,0,label=sprintf("%+.3f",F7$minus_derange[F7$n_copies==1])),hjust=1,size=3.1,colour=NUL)+
  geom_text(aes(XSF,0,label=sprintf("%+.3f",F7$minus_shuffle[F7$n_copies==1])),hjust=1,size=3.1,colour=NUL)+
  geom_text(aes(XZ,0,label=sprintf("%+.1f",F7$z_vs_derange[F7$n_copies==1])),hjust=1,size=3.0,colour=NUL)+
  geom_text(aes(XCEIL,0,label=sprintf("%.1f%%",F7$pct_past_observed_max[F7$n_copies==1])),hjust=1,size=3.0,colour=COMP)+
  geom_text(aes(XSTRIP,0,label=sprintf("%.0f%% at seven copies",
                F7$pct_past_observed_max[F7$n_copies==7])),hjust=0,size=2.9,colour=COMP)+
  coord_cartesian(xlim=c(0,XSTRIP+14*SW+2),ylim=c(-0.55,0.55),expand=FALSE,clip="off")+
  labs(title="Off the scale, for contrast",
       subtitle=sprintf("The 23-mer the model wants when nothing constrains it. Its logits leave the range the model produces on real sequence (ceiling %+.2f), so its size is not an effect size.",D$observed_max[1]))+
  theme_void(base_size=BASE)+
  theme(plot.title=element_text(face="bold",colour=NUL,size=BASE,margin=margin(b=1)),
        plot.subtitle=element_text(colour=NUL,size=BASE-4,margin=margin(b=5)),
        plot.margin=margin(4,6,4,6))
p<-pT/pF+plot_layout(heights=c(nrow(rows),1.5))+
  plot_annotation(
    title="Every negative agrees where the effect is a motif, and only there",
    subtitle="Shifts and derangements substitute every base; the shuffle permutes the bases it already has. Agreement between them means arrangement, disagreement means composition.",
    caption=paste0("- shifts subtracts the mean of the three cyclic shifts A to C to G to T to A. - derang subtracts the mean of all nine derangements of ACGT. - shuffle subtracts the mean of 50 permutations of the insert.\n",
    "z is against the nine derangements. Past ceiling is the share of windows whose logit exceeds the highest value the model emits on any held-out sequence; above a few percent the delta stops being an effect size.\n",
    "Fill is capped at plus or minus 2 so the small rows stay readable. Strip shows every dose on the - derang scale, 1 to 7 for the 3' splice sites and 1 to 14 for the donor. Blue marks the invariant AG or GT."),
    theme=theme(plot.title=element_text(face="bold",colour=INK,size=BASE+2),
                plot.subtitle=element_text(colour=NUL,size=BASE-3.5,margin=margin(b=6)),
                plot.caption=element_text(colour=NUL,size=BASE-4.5,hjust=0,margin=margin(t=8))))
ggsave("figure_dose_null.pdf",p,width=14.4,height=6.4,device=cairo_pdf)
ggsave("figure_dose_null.png",p,width=14.4,height=6.4,dpi=300)
cat("[out] figure_dose_null.pdf / .png\n")
