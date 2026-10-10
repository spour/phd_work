# figure 3.2, which genes each result is computed on, and how models are scored
suppressPackageStartupMessages({library(ggplot2);library(patchwork)})
INK<-"#1A1A1A"; GTX<-"#5A6068"; GREY<-"#8A8F98"; PAPER<-"#FCFCFD"
CAN<-"#0072B2"; COMP<-"#E69F00"; LIGHT<-"#E4E7EB"; MID<-"#A9B0BA"
TS<-3.0
TH<-theme_void(base_size=10)+theme(
  plot.title=element_text(face="bold",colour=INK,size=11,margin=margin(b=4)),
  plot.margin=margin(6,8,6,8), plot.background=element_rect(fill="white",colour=NA))
fmt<-function(n) format(n,big.mark=",")

# counts, from the tables
F<-read.delim("../analysis_set_funnel.tsv",check.names=FALSE)
W<-read.delim("../cnn_wholegenome_p1.tsv",check.names=FALSE)
n_all<-W$n_genes[W$scope=="genome-wide"][1]
n_con<-W$n_genes[W$scope=="contested"][1]
n_unc<-n_all-n_con
st<-function(s,col) F[[col]][F$step==s]
n_cand<-st("all peaks","n_peaks")
rm_apa<-st("competitors: APA-annotated removed","n_removed")
rm_te <-st("competitors: reference terminal exon removed","n_removed")
rm_g1 <-st("genes without a surviving competitor dropped","n_removed")
rm_dn <-st("competitors: at or downstream of the canonical removed","n_removed")
rm_g2 <-st("analysis set","n_removed")
n_ana <-st("analysis set","n_genes"); p_ana<-st("analysis set","n_peaks")
c_ana <-st("analysis set","n_competitor")
stopifnot(n_all==15560, n_con==6583, n_unc==8977, n_cand==70723, n_ana==4847,
          p_ana==48394, c_ana==43547, n_unc+n_ana==13824)

# a  gene sets
box<-function(x,y,w,h,head,sub=NULL,fill=PAPER,border=GREY,col=INK){
  l<-list(annotate("rect",xmin=x-w/2,xmax=x+w/2,ymin=y-h/2,ymax=y+h/2,fill=fill,colour=border,linewidth=0.35))
  if (is.null(sub)) l<-c(l,annotate("text",x=x,y=y,label=head,fontface="bold",colour=col,size=TS+0.2))
  else l<-c(l,annotate("text",x=x,y=y+h*0.2,label=head,fontface="bold",colour=col,size=TS+0.2),
                annotate("text",x=x,y=y-h*0.2,label=sub,colour=GTX,size=TS-0.3,lineheight=0.95))
  l}
arr<-function(x,y,xend,yend) annotate("segment",x=x,y=y,xend=xend,yend=yend,colour=INK,linewidth=0.35,
                                      arrow=arrow(length=unit(1.5,"mm"),type="closed"))
tag<-function(x,y,s,hj=0.5) annotate("text",x=x,y=y,label=s,colour=CAN,size=TS-0.3,fontface="italic",hjust=hj)
ys<-c(64,57.5,51,44.5,38)
side<-function(y,s) list(annotate("segment",x=50,xend=53,y=y,yend=y,colour=GREY,linewidth=0.35),
                         annotate("text",x=54,y=y,label=s,colour=GTX,size=TS-0.5,hjust=0))
pA<-ggplot()+
  box(40,95,44,8,paste(fmt(n_all),"genes with an annotated 3\u2032 end"))+
  arr(26,91,14,82.3)+arr(50,91,50,82.3)+
  box(14,76,26,12,paste(fmt(n_unc),"uncontested"),"the baseline already ranks\nthe annotated end first")+
  box(50,76,34,12,paste(fmt(n_con),"contested"),
      paste0("a competitor scores at or above\nthe annotated end; ",fmt(n_cand)," candidates"))+
  arr(50,70,50,31.3)+
  side(ys[1],paste0("\u2212",fmt(rm_apa)," competitors at an annotated poly(A) site"))+
  side(ys[2],paste0("\u2212",fmt(rm_te)," competitors in a reference terminal exon"))+
  side(ys[3],paste0("\u2212",fmt(rm_g1)," genes left with no competitor"))+
  side(ys[4],paste0("\u2212",fmt(rm_dn)," competitors at or downstream of the annotated end"))+
  side(ys[5],paste0("\u2212",fmt(rm_g2)," genes left with no competitor"))+
  box(50,25,34,12,paste(fmt(n_ana),"genes: analysis set"),
      paste0(fmt(p_ana)," candidates: one annotated\nend and ",fmt(c_ana)," competitors"),border=INK)+
  arr(14,70,14,31.3)+arr(33,25,27.2,25)+
  box(14,25,26,12,paste(fmt(n_unc+n_ana),"genes"),
      paste0(fmt(n_unc)," uncontested\n+ ",fmt(n_ana)," analysis set"))+
  coord_cartesian(xlim=c(0,100),ylim=c(15,100),clip="off")+
  ggtitle("A   Which genes each result is computed on")+TH

# b  within-gene ranking
X2<-2.2
g1<-data.frame(g=1,s=c(2.4,1.6,1.1,0.5),cls=c("real","comp","comp","comp"))
g2<-data.frame(g=X2,s=c(2.2,1.8,1.2,0.9,0.3),cls=c("comp","real","comp","comp","comp"))
G<-rbind(g1,g2)
pB<-ggplot(G)+
  annotate("segment",x=c(1,X2),xend=c(1,X2),y=0.1,yend=2.7,colour=LIGHT,linewidth=2.2,lineend="round")+
  geom_point(aes(x=g,y=s,colour=cls),size=3.2)+
  scale_colour_manual(values=c(real=CAN,comp=COMP),guide="none")+
  annotate("text",x=c(1.1,X2+0.1),y=c(2.4,1.8),label=c("rank 1","rank 2"),hjust=0,colour=CAN,size=TS-0.3)+
  annotate("point",x=0.62,y=3.25,colour=CAN,size=2.4)+
  annotate("text",x=0.72,y=3.25,label="annotated end",hjust=0,colour=GTX,size=TS-0.4)+
  annotate("point",x=1.75,y=3.25,colour=COMP,size=2.4)+
  annotate("text",x=1.85,y=3.25,label="competitor",hjust=0,colour=GTX,size=TS-0.4)+
  annotate("text",x=c(1,X2),y=-0.25,label=c("gene 1","gene 2"),colour=INK,size=TS,fontface="bold")+
  annotate("text",x=c(1,X2),y=-0.55,label=c("P@1 = 1\nwin-rate 3/3 = 1.00","P@1 = 0\nwin-rate 3/4 = 0.75"),
           colour=GTX,size=TS-0.4,lineheight=0.95,vjust=1)+
  annotate("text",x=0.5,y=1.4,label="model score",angle=90,colour=GTX,size=TS-0.3)+
  coord_cartesian(xlim=c(0.45,2.9),ylim=c(-1.4,3.4),clip="off")+
  ggtitle("B   Within-gene ranking")+TH

# c  out-of-fold scoring
chr<-c(1:22,"X"); k<-7
cw<-1; cx<-seq_along(chr)
CH<-data.frame(x=cx,lab=chr,fill=ifelse(cx==k,INK,LIGHT))
fx<-c(train=27658,val=6538,test=14198); gx<-c(train=2781,val=443,test=1623); fr<-cumsum(c(0,fx))/sum(fx)*23+0.5
pC<-ggplot()+
  annotate("text",x=0.5,y=4.45,hjust=0,label="leave-one-chromosome-out, 23 folds",colour=INK,size=TS,fontface="bold")+
  geom_rect(data=CH,aes(xmin=x-0.45,xmax=x+0.45,ymin=3.2,ymax=3.8,fill=I(fill)),colour=GREY,linewidth=0.2)+
  geom_text(data=CH[CH$lab %in% c("1","7","12","22","X"),],aes(x=x,y=3.0,label=lab),colour=GTX,size=TS-0.6)+
  annotate("text",x=k,y=4.0,label="held out",colour=INK,size=TS-0.4)+
  annotate("rect",xmin=0.55,xmax=17,ymin=2.0,ymax=2.5,fill=LIGHT,colour=GREY,linewidth=0.2)+
  annotate("rect",xmin=17,xmax=23.45,ymin=2.0,ymax=2.5,fill=MID,colour=GREY,linewidth=0.2)+
  annotate("text",x=8.8,y=2.25,label="training genes, 75%",colour=INK,size=TS-0.4)+
  annotate("text",x=20.2,y=2.25,label="tuning, 25%",colour=INK,size=TS-0.4)+
  annotate("text",x=0.5,y=0.35,hjust=0,label="fixed split by gene",colour=INK,size=TS,fontface="bold")+
  annotate("rect",xmin=fr[1],xmax=fr[2],ymin=-1.0,ymax=-0.2,fill=LIGHT,colour=GREY,linewidth=0.2)+
  annotate("rect",xmin=fr[2],xmax=fr[3],ymin=-1.0,ymax=-0.2,fill=MID,colour=GREY,linewidth=0.2)+
  annotate("rect",xmin=fr[3],xmax=fr[4],ymin=-1.0,ymax=-0.2,fill=INK,colour=GREY,linewidth=0.2)+
  annotate("text",x=(fr[1]+fr[2])/2,y=-0.6,label=paste0("training\n",fmt(gx[1])," genes"),colour=INK,size=TS-0.5)+
  annotate("text",x=(fr[2]+fr[3])/2,y=-0.6,label=paste0("validation\n",fmt(gx[2])),colour=INK,size=TS-0.5)+
  annotate("text",x=(fr[3]+fr[4])/2,y=-0.6,label=paste0("test\n",fmt(gx[3])," genes"),colour="white",size=TS-0.5)+
  coord_cartesian(xlim=c(0.3,23.6),ylim=c(-1.3,4.6),clip="off")+
  ggtitle("C   How every candidate is scored")+TH

fig<-pA/(pB+pC+plot_layout(widths=c(0.85,1.15)))+plot_layout(heights=c(1.25,1))
dir.create("out",showWarnings=FALSE)
ggsave("out/fig3_02_genesets.pdf",fig,width=180,height=190,units="mm",device=cairo_pdf)
ggsave("out/fig3_02_genesets.png",fig,width=180,height=190,units="mm",dpi=300,bg="white")
cat("[out] out/fig3_02_genesets.{pdf,png}\n")
