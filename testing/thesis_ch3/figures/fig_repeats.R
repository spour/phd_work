# (v2, no bar charts) -- repeat elements at competitor sites: what they
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr)})
setwd("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3")
INK<-"#1A1A1A"; GTX<-"#5A6068"; GREY<-"#8A8F98"; REAL<-"#08519C"; COMP<-"#E8A33D"; REP<-"#CFC6E3"; REPD<-"#7B5EA7"
TS<-2.7
TH<-theme_classic(base_size=8.5)+theme(plot.title=element_text(face="bold",size=10.5,colour=INK),axis.text=element_text(colour=GTX),
      legend.key.size=unit(3,"mm"),legend.text=element_text(size=7.5),legend.title=element_blank())
THV<-theme_void(base_size=8.5)+theme(plot.title=element_text(face="bold",size=10.5,colour=INK),plot.margin=margin(4,6,4,6))

# a  what was measured (schematic)
seg<-function(x1,x2,y,lab,fill,h=0.45) list(annotate("rect",xmin=x1,xmax=x2,ymin=y-h/2,ymax=y+h/2,fill=fill,colour=INK,linewidth=0.25),
                                            annotate("text",x=(x1+x2)/2,y=y,label=lab,size=TS-0.4,colour=INK))
pA<-ggplot()+
  annotate("segment",x=2,xend=98,y=6,yend=6,colour=GREY,linewidth=0.5)+
  annotate("rect",xmin=2,xmax=9,ymin=5.7,ymax=6.3,fill="#D9DDE3",colour=INK,linewidth=0.25)+
  annotate("rect",xmin=86,xmax=98,ymin=5.7,ymax=6.3,fill="#D9DDE3",colour=INK,linewidth=0.25)+
  annotate("segment",x=97,xend=97,y=6.3,yend=7.2,colour=REAL,linewidth=0.8)+annotate("point",x=97,y=7.2,colour=REAL,size=2.6)+
  annotate("text",x=97,y=7.65,label="real 3′ end",colour=REAL,size=TS,fontface="bold",hjust=0.9)+
  annotate("rect",xmin=36,xmax=62,ymin=5.75,ymax=6.25,fill=REP,colour=REPD,linewidth=0.4)+
  annotate("segment",x=40,xend=58,y=6,yend=6,colour=REPD,linewidth=0.5,arrow=arrow(length=unit(1.5,"mm"),type="closed"))+
  annotate("text",x=35,y=6.6,hjust=1,label="repeat element (RepeatMasker);\narrow, its own 5′→3′ direction",colour=REPD,size=TS-0.3,lineheight=0.9)+
  annotate("segment",x=57,xend=57,y=6.25,yend=7.2,colour=COMP,linewidth=0.8)+annotate("point",x=57,y=7.2,colour=COMP,size=2.6)+
  annotate("text",x=57,y=7.65,label="competitor site",colour=COMP,size=TS,fontface="bold")+
  annotate("segment",x=2,xend=14,y=7.4,yend=7.4,colour=INK,linewidth=0.4,arrow=arrow(length=unit(1.5,"mm"),type="closed"))+
  annotate("text",x=2,y=7.8,label="gene, 5′→3′",hjust=0,colour=INK,size=TS-0.3)+
  annotate("text",x=83,y=5.15,hjust=0,colour=GTX,size=TS-0.4,lineheight=0.95,
           label="sense: element on the gene's strand\nantisense: element on the opposite strand")+
  annotate("segment",x=36,xend=20,y=5.75,yend=4.4,colour=GREY,linewidth=0.3,linetype="22")+
  annotate("segment",x=62,xend=80,y=5.75,yend=4.4,colour=GREY,linewidth=0.3,linetype="22")+
  annotate("rect",xmin=20,xmax=80,ymin=3.9,ymax=4.4,fill=REP,colour=REPD,linewidth=0.4)+
  annotate("text",x=19,y=4.15,label="0\n5′ end",hjust=1,size=TS-0.4,colour=GTX,lineheight=0.9)+
  annotate("text",x=81,y=4.15,label="1\n3′ end",hjust=0,size=TS-0.4,colour=GTX,lineheight=0.9)+
  annotate("segment",x=68.5,xend=68.5,y=3.8,yend=4.5,colour=COMP,linewidth=1.1)+
  annotate("text",x=68.5,y=3.45,label="position of the site along the element (here 0.8)",colour=COMP,size=TS-0.3)+
  annotate("text",x=2,y=2.2,label="LTR",hjust=0,fontface="bold",size=TS,colour=INK)+
  seg(10,24,2.2,"U3",REP)+seg(24,30,2.2,"R",REPD)+seg(30,34,2.2,"U5",REP)+
  annotate("text",x=27,y=1.55,label="own poly(A) signal in R",size=TS-0.5,colour=REPD)+
  annotate("text",x=38,y=2.2,label="Alu",hjust=0,fontface="bold",size=TS,colour=INK)+
  seg(44,55,2.2,"left arm",REP)+seg(55,57.5,2.2,"",REPD)+seg(57.5,69,2.2,"right arm",REP)+seg(69,72,2.2,"",REPD)+
  annotate("text",x=56.2,y=1.55,label="A-rich linker",size=TS-0.5,colour=REPD)+annotate("text",x=70.5,y=1.55,label="A-rich tail",size=TS-0.5,colour=REPD)+
  annotate("text",x=75,y=2.2,label="L1",hjust=0,fontface="bold",size=TS,colour=INK)+
  annotate("rect",xmin=79,xmax=86,ymin=1.975,ymax=2.425,fill="white",colour=REPD,linewidth=0.3,linetype="22")+
  seg(86,97,2.2,"ORF2  3′",REP)+
  annotate("text",x=88,y=1.55,label="most copies lack the 5′ part",size=TS-0.5,colour=REPD)+
  annotate("text",x=50,y=0.85,label="elements drawn 5′ to 3′, not to scale",size=TS-0.5,colour=GTX)+
  coord_cartesian(xlim=c(0,100),ylim=c(0.6,8),clip="off")+ggtitle("A   What was measured")+THV

# b  orientation: sites against all element sequence
CO<-read.delim("data/repeat_strand_composition.tsv",check.names=FALSE)
SP<-read.delim("data/repeat_site_positions.tsv")
ori<-bind_rows(lapply(c("L1","LTR","Alu"),function(f){
  a<-SP$n[SP$family==f & SP$orientation=="antisense"]; s_<-SP$n[SP$family==f & SP$orientation=="sense"]
  ba<-CO$bases[CO$sequence==paste0(f,", antisense")]; bs<-CO$bases[CO$sequence==paste0(f,", sense")]
  p<-a/(a+s_); se<-sqrt(p*(1-p)/(a+s_))
  tibble(element=f,what=c("competitor sites","all of this element's sequence"),frac=c(p,ba/(ba+bs)),lo=c(p-1.96*se,NA),hi=c(p+1.96*se,NA),n=a+s_)}))
ori$element<-factor(ori$element,levels=c("Alu","LTR","L1"))
pB<-ggplot(ori,aes(frac,element))+
  geom_vline(xintercept=0.5,colour=GREY,linetype="22",linewidth=0.3)+
  geom_line(aes(group=element),colour=GREY,linewidth=0.6)+
  geom_errorbar(aes(xmin=lo,xmax=hi),orientation="y",width=0.15,colour=COMP,linewidth=0.4,na.rm=TRUE)+
  geom_point(aes(colour=what),size=3)+
  scale_colour_manual(values=c("competitor sites"=COMP,"all of this element's sequence"=INK))+
  scale_x_continuous(limits=c(0,1),labels=function(x) paste0(round(100*x),"%"))+
  annotate("text",x=0.02,y=3.45,label="mostly sense",hjust=0,size=TS-0.3,colour=GTX)+annotate("text",x=0.98,y=3.45,label="mostly antisense",hjust=1,size=TS-0.3,colour=GTX)+
  labs(x="fraction antisense to the gene",y=NULL,title="B   Orientation")+TH+theme(legend.position="bottom")
# c  what each orientation puts on the gene's strand
tile<-function(df,title){ ggplot(df,aes(col,row))+geom_tile(aes(fill=scaled),colour="white",linewidth=0.8)+
  geom_text(aes(label=lab),size=TS,colour=INK)+scale_fill_gradient(low="#F4F6F9",high="#E8A33D",guide="none")+
  scale_x_discrete(position="top",labels=function(x) gsub("\\|","\n",x))+labs(x=NULL,y=NULL,title=title)+
  theme_minimal(base_size=8.5)+theme(panel.grid=element_blank(),plot.title=element_text(face="bold",size=10.5,colour=INK),
        axis.text.x.top=element_text(size=7.5,colour=INK,lineheight=0.9),axis.text.y=element_text(size=8,colour=INK)) }
ROWS<-c("repeat-free intron","L1, sense","L1, antisense","LTR, sense","LTR, antisense","Alu, sense","Alu, antisense")
cc<-CO%>%mutate(row=ifelse(sequence=="none","repeat-free intron",sequence))%>%filter(row %in% ROWS)
C<-bind_rows(tibble(row=cc$row,col="U content",v=cc$T,lab=paste0(round(100*cc$T),"%")),
             tibble(row=cc$row,col="UUUUU|per kb",v=cc$TTTTT_per_kb,lab=sprintf("%.1f",cc$TTTTT_per_kb)),
             tibble(row=cc$row,col="AAUAAA or AUUAAA|per kb",v=cc$PAS_per_kb,lab=sprintf("%.1f",cc$PAS_per_kb)))%>%
   group_by(col)%>%mutate(scaled=(v-min(v))/(max(v)-min(v)))%>%ungroup()%>%
   mutate(row=factor(row,levels=rev(ROWS)),col=factor(col,levels=c("U content","UUUUU|per kb","AAUAAA or AUUAAA|per kb")))
pC<-tile(C,"C   What each orientation puts on the gene's strand")
# d  where in the element (enrichment heatmap)
H<-bind_rows(lapply(seq_len(nrow(SP)),function(k){ s_<-as.numeric(strsplit(SP$site_deciles[k]," ")[[1]]); b<-as.numeric(strsplit(SP$background_deciles[k]," ")[[1]])
  tibble(row=paste0(SP$family[k],", ",SP$orientation[k]," (n = ",format(SP$n[k],big.mark=","),")"),bin=0:9,e=log2((s_+1e-3)/b))}))
H$row<-factor(H$row,levels=rev(unique(H$row[order(match(sub(" \\(.*","",H$row),c("LTR, sense","LTR, antisense","Alu, sense","Alu, antisense","L1, sense","L1, antisense")))])))
pD<-ggplot(H,aes(bin+0.5,row,fill=pmax(pmin(e,2.5),-2.5)))+geom_tile(colour="white",linewidth=0.6)+
  scale_fill_gradient2(low="#3B6FB6",mid="#F4F6F9",high="#C0392B",midpoint=0,limits=c(-2.5,2.5),name="sites relative to\nrandom positions",
                       breaks=c(-2,0,2),labels=c("4\u00d7 fewer","same","4\u00d7 more"))+
  scale_x_continuous(breaks=c(0,5,10),labels=c("5\u2032 end","middle","3\u2032 end"),expand=c(0,0))+
  labs(x="position along the element",y=NULL,title="D   Where in the element the site lies")+TH+theme(legend.position="right",legend.title=element_text(size=7.5))
PA<-read.delim("data/figr_pas.tsv"); ST<-read.delim("data/figr_structure.tsv"); NW<-read.delim("data/figr_network.tsv",na.strings="NA")
SH<-read.delim("data/repeat_shuffle.tsv"); SH$group<-sub("competitor, ","",SH$group)
RD<-read.delim("data/repeat_deep.tsv",header=FALSE,fill=TRUE,col.names=paste0("V",1:6))
Tfr<-RD%>%filter(V1=="elements",V3=="T fraction, +1 to +40")%>%transmute(group=sub("competitor, ","",V2),v=as.numeric(V4))
G<-c("real 3' end","no repeat","L1 antisense","L1 sense","LTR","Alu","DNA transposon")
E<-bind_rows(
  PA%>%filter(grepl("AAUAAA",feature))%>%transmute(group,col="AAUAAA or AUUAAA|at -35 to -10",v=value,lab=paste0(round(100*value),"%")),
  Tfr%>%transmute(group,col="U content|+1 to +40",v,lab=paste0(round(100*v),"%")),
  ST%>%transmute(group,col="signal|unpaired",v=pas_unpaired,lab=sprintf("%.2f",pas_unpaired)),
  NW%>%filter(!is.na(centred_logit))%>%transmute(group,col="network score vs|gene's competitors",v=centred_logit,lab=sprintf("%+.2f",centred_logit)),
  SH%>%transmute(group,col="change when \u00b1300 nt|is shuffled",v=delta_logit,lab=sprintf("%+.2f",delta_logit)))%>%
  filter(group %in% G)%>%group_by(col)%>%mutate(scaled=(v-min(v))/(max(v)-min(v)))%>%ungroup()%>%
  mutate(row=factor(recode(group,"real 3' end"="real 3\u2032 end","no repeat"="competitor, no repeat","L1 antisense"="competitor, L1 antisense",
         "L1 sense"="competitor, L1 sense","LTR"="competitor, LTR","Alu"="competitor, Alu","DNA transposon"="competitor, DNA transposon"),
         levels=rev(c("real 3\u2032 end","competitor, no repeat","competitor, L1 antisense","competitor, L1 sense","competitor, LTR","competitor, Alu","competitor, DNA transposon"))),
         col=factor(col,levels=c("AAUAAA or AUUAAA|at -35 to -10","U content|+1 to +40","signal|unpaired","network score vs|gene's competitors","change when \u00b1300 nt|is shuffled")))
E<-bind_rows(E,tibble(group="real 3' end",col=factor("network score vs|gene's competitors",levels=levels(E$col)),v=NA,lab="\u2013",scaled=NA,
                     row=factor("real 3\u2032 end",levels=levels(E$row))))
pE<-tile(E,"E   The sites and the network's response")+scale_fill_gradient(low="#F4F6F9",high="#E8A33D",na.value="white",guide="none")
fig<-pA/(pB|pC)/pD/pE+plot_layout(heights=c(1.2,0.9,0.85,0.95))
ggsave("out/fig_repeats.pdf",fig,width=10.4,height=13,device=cairo_pdf)
ggsave("out/fig_repeats.png",fig,width=10.4,height=13,dpi=250,bg="white")
cat("[out] out/fig_repeats.{pdf,png}\n")
