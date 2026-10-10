# the network as a diagram
suppressPackageStartupMessages({library(ggplot2);library(ggforce);library(dplyr)})
INK<-"#1A1A1A"; GTX<-"#5A6068"; GREY<-"#8A8F98"; PAPER<-"#FCFCFD"
SLATE<-"#BCD3E8"; SLATE_B<-"#DFE9F3"
PALE <-"#EDE9E3"; PALE_B <-"#F6F4F0"
LW<-0.30; LWB<-0.20; BASE<-9

# dimensions and receptive field, derived
L0<-2500
ops<-list(list(k=19,s=1),list(k=10,s=10),list(k=7,s=1),list(k=5,s=5),
          list(k=5,s=1),list(k=2,s=2))
rf<-1; jump<-1
for (o in ops){ rf<-rf+(o$k-1)*jump; jump<-jump*o$s }
RF<-rf; STRIDE<-jump
stopifnot(RF==378, STRIDE==100, L0/STRIDE==25)

ST<-tribble(
  ~id, ~lab,          ~len,  ~ch, ~fill,  ~sub,
  1,   "input",       2500,   4,  PALE,   "one-hot DNA",
  2,   "conv 19",     2500,  32,  SLATE,  "BatchNorm, exp",
  3,   "max pool 10",  250,  32,  SLATE,  "",
  4,   "conv 7",       250,  64,  SLATE,  "BatchNorm, ReLU",
  5,   "max pool 5",    50,  64,  SLATE,  "",
  6,   "conv 5",        50,  96,  SLATE,  "BatchNorm, ReLU",
  7,   "max pool 2",    25,  96,  SLATE,  "position kept"
)
ST<-ST%>%mutate(w=2.6+9.4*(len/max(len))^(1/3), h=2.6+7.4*sqrt(ch/96))
gapx<-4.8
ST$x1<-NA; ST$x2<-NA; cur<-0
for (i in seq_len(nrow(ST))){ ST$x1[i]<-cur; ST$x2[i]<-cur+ST$w[i]; cur<-cur+ST$w[i]+gapx }
ST<-ST%>%mutate(xm=(x1+x2)/2, y1=-h/2, y2=h/2)
LB<-min(ST$y1)-0.9
DEPTH<-4; DX<-0.42; DY<-0.34
BFILL<-c(setNames(SLATE_B,SLATE), setNames(PALE_B,PALE))
back<-do.call(rbind,lapply(seq_len(nrow(ST)),function(i)
  do.call(rbind,lapply(DEPTH:1,function(d) data.frame(
    x1=ST$x1[i]+d*DX, x2=ST$x2[i]+d*DX, y1=ST$y1[i]+d*DY, y2=ST$y2[i]+d*DY,
    fill=unname(BFILL[ST$fill[i]]))))))
headx<-cur+3.0
fcx<-c(headx+5.6, headx+13.0, headx+19.6)
nodes<-rbind(
  data.frame(x=fcx[1], y=seq(-3.4,3.4,length.out=11), r=0.30, fill=SLATE_B),
  data.frame(x=fcx[2], y=seq(-2.2,2.2,length.out=7),  r=0.34, fill=SLATE),
  data.frame(x=fcx[3], y=0, r=0.62, fill=INK))
edges<-rbind(
  expand.grid(x=fcx[1], xe=fcx[2], y=seq(-3.4,3.4,length.out=11), ye=seq(-2.2,2.2,length.out=7)),
  expand.grid(x=fcx[2], xe=fcx[3], y=seq(-2.2,2.2,length.out=7),  ye=0))
XMAX<-fcx[3]+4.6
XR<-c(-1.5,XMAX); YR<-c(min(ST$y1)-4.4,6.6)
arr<-data.frame(x=head(ST$x2,-1)+DEPTH*DX+0.55, xe=tail(ST$x1,-1)-0.45, y=0, ye=0)
set.seed(7)
NB<-26L; BCOL<-c(A="#3D9A50",C="#2C6FB5",G="#E8A33D",T="#C6413B")
oh<-expand.grid(j=seq_len(NB),row=1:4)%>%
  mutate(base=names(BCOL)[row],
         cw=(ST$x2[1]-ST$x1[1]-0.7)/NB, rh=(ST$y2[1]-ST$y1[1]-0.7)/4,
         x=ST$x1[1]+0.35+(j-0.5)*cw, y=ST$y2[1]-0.35-(row-0.5)*rh)
hot<-sample(1:4,NB,replace=TRUE)
oh<-oh%>%mutate(on=row==hot[j])
p<-ggplot()+
  geom_rect(data=back,aes(xmin=x1,xmax=x2,ymin=y1,ymax=y2,fill=I(fill)),colour=INK,linewidth=LWB)+
  geom_segment(data=edges,aes(x,y,xend=xe,yend=ye),colour=GTX,linewidth=.10,alpha=.30)+
  geom_circle(data=nodes,aes(x0=x,y0=y,r=r,fill=I(fill)),colour=INK,linewidth=LWB)+
  geom_rect(data=ST,aes(xmin=x1,xmax=x2,ymin=y1,ymax=y2,fill=I(fill)),colour=INK,linewidth=LW)+
  geom_tile(data=oh,aes(x,y,width=cw*.82,height=rh*.72,
                        fill=I(ifelse(on,BCOL[base],"#FFFFFF")),alpha=I(ifelse(on,1,.22))))+
  geom_segment(data=arr,aes(x,y,xend=xe,yend=ye),colour=GTX,linewidth=.32,
               arrow=arrow(length=unit(3.4,"pt"),type="closed"))+
  geom_text(data=ST,aes(xm,LB,label=lab),size=2.65,colour=INK,fontface="bold",vjust=1)+
  geom_text(data=ST%>%filter(sub!=""),aes(xm,LB-1.2,label=sub),size=2.3,colour=GTX,vjust=1)+
  geom_text(data=ST,aes(xm,LB-2.4,label=sprintf("%s x %s",ch,format(len,big.mark=",",trim=TRUE))),
            size=2.5,colour=INK,vjust=1)+
  geom_segment(aes(x=ST$x2[7]+DEPTH*DX+0.55,xend=fcx[1]-1.25,y=0,yend=0),colour=GTX,linewidth=.32,
               arrow=arrow(length=unit(3.4,"pt"),type="closed"))+
  geom_text(aes(x=ST$x2[7]+DEPTH*DX+0.6,y=LB,label="flatten"),size=2.65,colour=INK,fontface="bold",vjust=1,hjust=0)+
  geom_text(aes(x=ST$x2[7]+DEPTH*DX+0.6,y=LB-1.2,label="2,400"),size=2.5,colour=INK,vjust=1,hjust=0)+
  geom_text(aes(x=fcx[1],y=LB,label="dense 64"),size=2.65,colour=INK,fontface="bold",vjust=1)+
  geom_text(aes(x=fcx[1],y=LB-1.2,label="ReLU, Dropout 0.4"),size=2.3,colour=GTX,vjust=1)+
  geom_text(aes(x=fcx[3],y=LB,label="logit"),size=2.65,colour=INK,fontface="bold",vjust=1)+

  annotate("text",x=-1.0,y=6.0,hjust=0,size=2.55,colour=GTX,
           label=sprintf("one output bin sees %d nt of sequence",RF))+
  coord_fixed(xlim=XR,ylim=YR,expand=FALSE,clip="off")+
  theme_void(base_size=BASE)+
  theme(plot.background=element_rect(fill=PAPER,colour=NA),plot.margin=margin(6,8,6,8))
W<-9.4; H<-W*diff(YR)/diff(XR)+0.10
ggsave("figure_architecture.pdf",p,width=W,height=H,device=cairo_pdf)
ggsave("figure_architecture.png",p,width=W,height=H,dpi=400,bg=PAPER)
cat(sprintf("[size] data aspect %.2f, saved %.2f x %.2f in\n",diff(XR)/diff(YR),W,H))
cat(sprintf("[arch] receptive field %d nt, stride %d nt, %d bins\n",RF,STRIDE,L0/STRIDE))
cat("[out] figure_architecture.pdf / .png\n")
