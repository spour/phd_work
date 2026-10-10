# the dose table, drawn
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(tidyr);library(patchwork)})
INK<-"#1A1A1A"; BASE<-13; CAN<-"#0072B2"; COMP<-"#E69F00"; NUL<-"#8A8F98"; GRID<-"#E3E5E8"
D<-read.delim("dose_table.tsv",stringsAsFactors=FALSE)
SHOW<-list(acc_best=c(1,7),acc_noC=c(1,7),donor=c(1,7,14))
ORD<-c("acc_best","acc_noC","donor")
AGPOS<-list(acc_best=c(19,20),acc_noC=c(19,20),donor=c(4,5))
CW<-1.06; X0<-1.0
XDESC<-30; XN<-56; XLEN<-63; XRAW<-73.5; XMC<-82.5; XZ<-89; XSTRIP<-92.5; SW<-0.92
LIM<-1.2
sq<-function(v) pmax(pmin(v,LIM),-LIM)
panel<-function(bg){
  d<-D[D$background==bg,]
  rows<-do.call(rbind,lapply(ORD,function(s) data.frame(site=s,n=SHOW[[s]],stringsAsFactors=FALSE)))
  rows<-rows %>% left_join(d,by=c("site","n"="n_copies")) %>% mutate(y=-seq_len(n()))
  gl<-do.call(rbind,lapply(seq_len(nrow(rows)),function(i){
    s<-rows$sequence[i]; ch<-strsplit(s,"")[[1]]; k<-AGPOS[[rows$site[i]]]
    data.frame(x=X0+(seq_along(ch)-1)*CW,y=rows$y[i],lab=ch,
               hi=seq_along(ch)%in%k,stringsAsFactors=FALSE)}))
  ymid<-rows %>% group_by(site) %>% summarise(y=mean(y),.groups="drop")
  st<-d %>% inner_join(ymid,by="site") %>% mutate(x=XSTRIP+(n_copies-1)*SW)
  hdr<-data.frame(x=c(X0,XDESC,XN,XLEN,XRAW,XMC,XZ,XSTRIP),
                  lab=c("implanted unit","what it is","n","nt","raw","- comp","z","dose 1 to n"),
                  h=c(0,0,1,1,1,1,1,0))
  ggplot()+
    geom_hline(yintercept=-0.45,colour=INK,linewidth=.45)+
    geom_hline(yintercept=rows$y[!duplicated(rows$site)][-1]+0.5,colour=GRID,linewidth=.35)+
    geom_text(data=hdr,aes(x,0,label=lab,hjust=h),size=3.05,colour=NUL,fontface="bold")+
    geom_text(data=gl,aes(x,y,label=lab,colour=hi),family="mono",size=2.85,show.legend=FALSE)+
    scale_colour_manual(values=c(`FALSE`=INK,`TRUE`=CAN))+
    geom_text(data=rows,aes(XDESC,y,label=description),hjust=0,size=3.15,colour=INK)+
    geom_text(data=rows,aes(XN,y,label=n),hjust=1,size=3.15,colour=INK)+
    geom_text(data=rows,aes(XLEN,y,label=insert_len),hjust=1,size=3.15,colour=NUL)+
    geom_tile(data=rows,aes(XRAW-2.6,y,fill=sq(raw)),width=8.4,height=.78)+
    geom_tile(data=rows,aes(XMC-2.6,y,fill=sq(minus_comp)),width=8.4,height=.78)+
    geom_text(data=rows,aes(XRAW,y,label=sprintf("%+.3f",raw)),hjust=1,size=3.15,colour=INK)+
    geom_text(data=rows,aes(XMC,y,label=sprintf("%+.3f",minus_comp)),hjust=1,size=3.15,
              fontface="bold",colour=INK)+
    geom_text(data=rows,aes(XZ,y,label=sprintf("%+.1f",z)),hjust=1,size=3.0,
              colour=ifelse(abs(rows$z)>=2,INK,NUL))+
    geom_tile(data=st,aes(x,y,fill=sq(minus_comp)),width=SW*.86,height=.62,colour="white",linewidth=.18)+
    scale_fill_gradient2(low=COMP,mid="#FFFFFF",high=CAN,midpoint=0,limits=c(-LIM,LIM),
                         name="delta logit",breaks=c(-1,0,1))+
    coord_cartesian(xlim=c(0,XSTRIP+14*SW),ylim=c(-nrow(rows)-.7,1.0),expand=FALSE,clip="off")+
    labs(title=d$bg_label[1],subtitle=sprintf("%s windows; blue the model scores higher, orange lower",
         format(d$n_bg[1],big.mark=",")))+
    theme_void(base_size=BASE)+
    theme(plot.title=element_text(face="bold",colour=INK,size=BASE,margin=margin(b=1)),
          plot.subtitle=element_text(colour=NUL,size=BASE-4,margin=margin(b=5)),
          legend.position="none",plot.margin=margin(4,6,4,6))
}
p<-panel("real_canonical")/panel("real_competitor")+
  plot_annotation(
    title="Matching composition removes the acceptor gain",
    subtitle="Best of 60 upstream positions. Minus comp subtracts the mean of 10 mononucleotide shuffles of the same insert, so length and base composition are matched.",
    caption=paste0("raw = intact unit minus unmodified background. - comp = intact unit minus its own shuffles. z from the two standard errors combined.\n",
               "Strip shows every dose, 1 to 7 for the acceptors and 1 to 14 for the donor, on the - comp scale. Blue marks the invariant AG or GT."),
    theme=theme(plot.title=element_text(face="bold",colour=INK,size=BASE+2),
                plot.subtitle=element_text(colour=NUL,size=BASE-3.5,margin=margin(b=6)),
                plot.caption=element_text(colour=NUL,size=BASE-4.5,hjust=0,margin=margin(t=8))))
ggsave("figure_dose_table.pdf",p,width=10.2,height=5.6,device=cairo_pdf)
ggsave("figure_dose_table.png",p,width=10.2,height=5.6,dpi=300)
cat("[out] figure_dose_table.pdf / .png\n")
