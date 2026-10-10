# which properties of the window the network represents and relies on
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr)})
setwd("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3")
INK<-"#1A1A1A"; GTX<-"#5A6068"; CAN<-"#0072B2"; NUL<-"#8A8F98"
D<-read.delim("data/concept_erasure_within.tsv",check.names=FALSE)
ord<-c("A fraction, -300 to 0","G fraction, -300 to 0","C fraction, -300 to 0","U fraction, -300 to 0",
       "UGUA count, -300 to 0","poly(A) signal, -35 to -10","GU dinucleotides, +1 to +100",
       "U1 sites per kb, -500 to 0","strong 5' splice sites, -500 to 0","best 3' splice site, -500 to 0","best 3' splice site, -2,000 to -126",
       "annotated 5' splice site within 1 kb upstream","annotated 3' splice site within 1 kb upstream")
lab<-c("A content","G content","C content","U content","UGUA count","Poly(A) signal present","GU downstream",
       "U1 sites","Strong 5' splice sites (MaxEnt)","Best 3' splice site within 500 nt (MaxEnt)","Best 3' splice site within 2 kb (MaxEnt)",
       "Annotated 5' splice site upstream","Annotated 3' splice site upstream")
grp<-c(rep("Base composition",4),rep("Cleavage elements",3),rep("Splice-site scores",4),rep("Annotated splice sites",2))
stopifnot(all(D$concept%in%ord))
D<-D%>%mutate(row=factor(lab[match(concept,ord)],levels=rev(lab)),group=factor(grp[match(concept,ord)],levels=unique(grp)),
   layer=factor(layer,levels=c("convolutional output (2,400)","dense layer (64)"),labels=c("Convolutional output","Dense layer")))
th<-theme_classic(base_size=10)+theme(plot.title=element_text(face="bold",size=11,colour=INK),axis.text=element_text(colour=GTX),strip.background=element_blank(),
      strip.text.y=element_blank(),legend.title=element_blank(),legend.key.size=unit(4,"mm"),panel.spacing.y=unit(3,"mm"))
A<-D%>%filter(layer=="Convolutional output")%>%mutate(r2=pmax(r2_heldout,0))
pA<-ggplot(A,aes(r2,row))+geom_segment(aes(x=0,xend=r2,yend=row),linewidth=0.9,colour=CAN)+geom_point(size=2.6,colour=CAN)+
  facet_grid(group~.,scales="free_y",space="free_y")+scale_x_continuous(limits=c(0,1),breaks=c(0,0.5,1))+
  labs(x=expression("decoded from the network ("*R^2*")"),y=NULL,title="A   Represented")+th+theme(axis.text.y=element_text(colour=INK))
pB<-ggplot(D,aes(100*delta_vs_control,row,colour=layer,shape=layer))+geom_vline(xintercept=0,colour=GTX,linewidth=0.3)+
  geom_errorbar(aes(xmin=100*delta_lo,xmax=100*delta_hi),orientation="y",width=0.3,linewidth=0.4,position=position_dodge(width=0.6))+
  geom_point(size=2.3,position=position_dodge(width=0.6))+facet_grid(group~.,scales="free_y",space="free_y")+
  scale_colour_manual(values=c("Convolutional output"=CAN,"Dense layer"="#4A4E69"))+scale_shape_manual(values=c(16,17))+
  labs(x="change in P@1 when erased (percentage points)",y=NULL,title="B   Relied on")+th+theme(axis.text.y=element_blank(),axis.ticks.y=element_blank(),legend.position="bottom")
fig<-pA+pB+plot_layout(widths=c(0.8,1.2))
ggsave("out/fig3_concepts.pdf",fig,width=7.6,height=5.0,device=cairo_pdf)
ggsave("out/fig3_concepts.png",fig,width=7.6,height=5.0,dpi=300,bg="white")
cat("[out] out/fig3_concepts.{pdf,png}\n")
