# figure 3.14, precision at 1 against chance on the genes the baseline gets wrong
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr)})
setwd("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3")
source("R/_harvest.R")
INK<-"#1A1A1A"; GTX<-"#5A6068"; GREY<-"#8A8F98"; CAN<-"#0B55A0"; SPL<-"#E8A33D"; COMB<-"#4A4E69"
set.seed(1)
boot<-function(v,B=2000){ n<-length(v); b<-replicate(B,mean(v[sample.int(n,n,TRUE)])); c(mean(v),quantile(b,c(.025,.975))) }

W<-read.delim("../wg_per_gene.tsv"); Wc<-W%>%filter(uncontested==0)
R<-read.delim("../rescue_ranks.tsv")
stopifnot(nrow(Wc)==6583, nrow(R)==4847)
rows<-list(
  c(set="Contested genes\n(n = 6,583)", model="chance",       boot(1/Wc$n_candidates)),
  c(set="Contested genes\n(n = 6,583)", model="splice model", boot(Wc$splice)),
  c(set="Contested genes\n(n = 6,583)", model="sequence CNN", boot(Wc$cnn)),
  c(set="Analysis set\n(n = 4,847)",    model="chance",       boot(1/R$n_candidates)),
  c(set="Analysis set\n(n = 4,847)",    model="splice model", boot(as.numeric(R$rank_splice==1))),
  c(set="Analysis set\n(n = 4,847)",    model="sequence CNN", boot(as.numeric(R$rank_cnn==1))),
  c(set="Analysis set\n(n = 4,847)",    model="combined",     boot(as.numeric(R$rank_combo==1))))
P<-as.data.frame(do.call(rbind,rows)); names(P)<-c("set","model","p","lo","hi")
P<-P%>%mutate(across(c(p,lo,hi),as.numeric),
              model=factor(model,levels=c("chance","splice model","sequence CNN","combined")),
              set=factor(set,levels=unique(set)))
write.table(P%>%mutate(set=gsub("\n"," ",set)),"data/fig3_11_p_at_1.tsv",sep="\t",quote=FALSE,row.names=FALSE)
print(P%>%mutate(set=gsub("\n"," ",set)),digits=3)
pA<-ggplot(P,aes(model,p,fill=model))+
  geom_col(width=0.7,colour=INK,linewidth=0.25)+
  geom_errorbar(aes(ymin=lo,ymax=hi),width=0.18,linewidth=0.35,colour=INK)+
  facet_grid(.~set,scales="free_x",space="free_x")+
  scale_fill_manual(values=c(chance="white","splice model"=SPL,"sequence CNN"=CAN,combined=COMB),guide="none")+
  scale_y_continuous(limits=c(0,0.8),expand=c(0,0))+
  labs(x=NULL,y="precision at 1",title="A   Annotated end ranked first on contested genes")+
  theme_classic(base_size=10)+
  theme(plot.title=element_text(face="bold",size=11,colour=INK),strip.background=element_blank(),
        strip.text=element_text(colour=INK,size=9),axis.text=element_text(colour=GTX),
        axis.text.x=element_text(size=8.5))

# b: sankey of all 15,560 genes: all -> uncontested / contested -> which model ranks the
WG<-read.delim("../wg_per_gene.tsv"); stopifnot(nrow(WG)==15560)
ct<-WG%>%filter(uncontested==0)
N<-c(unc=sum(WG$uncontested==1),both=sum(ct$cnn==1&ct$splice==1),net=sum(ct$cnn==1&ct$splice==0),spl=sum(ct$cnn==0&ct$splice==1),none=sum(ct$cnn==0&ct$splice==0))
stopifnot(sum(N)==15560,N["unc"]==8977,sum(N[-1])==6583); print(N)
fmt<-function(x) format(x,big.mark=",")
GAP<-700; X<-c(0,1,2); BW<-0.2
node<-function(sizes,gap=GAP){ top<-0; o<-list(); for(k in names(sizes)){ o[[k]]<-c(top,top+sizes[[k]]); top<-top+sizes[[k]]+gap }; o }
tot<-sum(N)+4*GAP
n1<-list(all=c((tot-15560)/2,(tot-15560)/2+15560))
s2<-c(unc=N[["unc"]],con=sum(N[-1])); n2<-node(s2,gap=tot-sum(s2))
n3<-node(as.list(N))
rib<-function(x0,x1,a0,a1,b0,b1,id,fill){ t<-seq(0,1,length.out=60); sg<-3*t^2-2*t^3; xs<-x0+(x1-x0)*t
  data.frame(x=c(xs,rev(xs)),y=c(a0+(b0-a0)*sg,rev(a1+(b1-a1)*sg)),id=id,fill=fill) }
R2<-rbind(
  rib(X[1]+BW,X[2]-BW,n1$all[1],n1$all[1]+s2[["unc"]],n2$unc[1],n2$unc[2],"a","unc"),
  rib(X[1]+BW,X[2]-BW,n1$all[1]+s2[["unc"]],n1$all[2],n2$con[1],n2$con[2],"b","con"),
  rib(X[2]+BW,X[3]-BW,n2$unc[1],n2$unc[2],n3$unc[1],n3$unc[2],"c","unc"))
off<-n2$con[1]
for(k in c("both","net","spl","none")){ R2<-rbind(R2,rib(X[2]+BW,X[3]-BW,off,off+N[[k]],n3[[k]][1],n3[[k]][2],k,k)); off<-off+N[[k]] }
ND<-rbind(data.frame(x=X[1],y0=n1$all[1],y1=n1$all[2],fill="all",lab=paste0("All genes\n",fmt(15560))),
          data.frame(x=X[2],y0=n2$unc[1],y1=n2$unc[2],fill="unc",lab=paste0("Uncontested\n",fmt(N[["unc"]]))),
          data.frame(x=X[2],y0=n2$con[1],y1=n2$con[2],fill="con",lab=paste0("Contested\n",fmt(sum(N[-1])))),
          data.frame(x=X[3],y0=n3$unc[1],y1=n3$unc[2],fill="unc",lab=paste0("Baseline ranks end first  ",fmt(N[["unc"]]))),
          data.frame(x=X[3],y0=n3$both[1],y1=n3$both[2],fill="both",lab=paste0("Network and splice model  ",fmt(N[["both"]]))),
          data.frame(x=X[3],y0=n3$net[1],y1=n3$net[2],fill="net",lab=paste0("Network only  ",fmt(N[["net"]]))),
          data.frame(x=X[3],y0=n3$spl[1],y1=n3$spl[2],fill="spl",lab=paste0("Splice model only  ",fmt(N[["spl"]]))),
          data.frame(x=X[3],y0=n3$none[1],y1=n3$none[2],fill="none",lab=paste0("Neither  ",fmt(N[["none"]]))))
FILL<-c(all="#4A4E69",unc="#C9CDD3",con="#E69F00",both=COMB,net=CAN,spl=SPL,none="#E4E6EA")
fl<-ggplot()+geom_polygon(data=R2,aes(x,y,group=id,fill=fill),alpha=0.55)+
  geom_rect(data=ND,aes(xmin=x-BW,xmax=x+BW,ymin=y0,ymax=y1,fill=fill),colour=INK,linewidth=0.25)+
  geom_text(data=ND[1:3,],aes(x=x,y=(y0+y1)/2,label=lab),size=3.1,colour=c("white",INK,INK),lineheight=0.95,fontface="bold")+
  geom_text(data=ND[4:8,],aes(x=x+BW+0.03,y=(y0+y1)/2,label=lab),size=3.1,colour=INK,hjust=0)+
  annotate("text",x=X,y=-700,label=c("Annotated 3' end","Chapter 2 baseline","Annotated end ranked first by"),size=3.1,colour=GTX,hjust=c(0.5,0.5,0))+
  scale_fill_manual(values=FILL,guide="none")+scale_y_reverse()+scale_x_continuous(limits=c(-0.25,3.35))+
  labs(title="B   Fate of all 15,560 genes")+theme_void(base_size=10)+theme(plot.title=element_text(face="bold",size=11,colour=INK))
fig<-pA/fl+plot_layout(heights=c(0.8,1))
ggsave("out/fig3_14_recovery.pdf",fig,width=7.4,height=8.2,device=cairo_pdf)
ggsave("out/fig3_14_recovery.png",fig,width=7.4,height=8.2,dpi=300,bg="white")
cat("[out] out/fig3_14_recovery.{pdf,png}\n")
