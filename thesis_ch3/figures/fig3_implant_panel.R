# response of the network to the measured binding motif of every rna-binding protein
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(ggrepel)})
setwd("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3")
INK<-"#1A1A1A"; GTX<-"#5A6068"; CAN<-"#0072B2"; COMP<-"#E69F00"; NUL<-"#B8BDC4"
D<-read.delim("data/implant_cisbp.tsv",check.names=FALSE)%>%filter(!grepl("^[0-9.]+$",protein),!grepl("^(AC|AL|AP|ENSG)[0-9]",protein))
B<-D%>%group_by(protein)%>%slice_max(abs(diff),n=1,with_ties=FALSE)%>%ungroup()%>%arrange(diff)%>%mutate(rank=row_number())
fam<-function(p) ifelse(grepl("^HNRNPA|^DAZAP1$|^MSI",p),"hnRNP A/B family",ifelse(grepl("^SRSF|^TRA2",p),"SR proteins",
      ifelse(p%in%c("NUDT21","CSTF2","CSTF2T","PABPN1","CPEB1","CPEB2","CPEB4"),"cleavage and polyadenylation factors",
      ifelse(p%in%c("U2AF2","SF1","SNRNP70","SNRPA","SNRPB2","SF3B4","SF3B6","PUF60","RBM22","RNPC3","ZCRB1"),"spliceosome components","other"))))
B$family<-factor(fam(B$protein),levels=c("SR proteins","spliceosome components","cleavage and polyadenylation factors","hnRNP A/B family","other"))
lab<-B%>%filter(family!="other"&(abs(diff)>=0.04|protein%in%c("NUDT21","CSTF2","U2AF2","SF1","PABPN1","SRSF1"))|rank<=1|rank>max(rank)-1)%>%mutate(l=paste0(protein,"  ",consensus))
COL<-c("SR proteins"="#0072B2","spliceosome components"="#009E73","cleavage and polyadenylation factors"="#CC79A7","hnRNP A/B family"="#D55E00","other"=NUL)
p<-ggplot(B,aes(rank,diff))+geom_hline(yintercept=0,colour=GTX,linewidth=0.3)+
  geom_segment(aes(xend=rank,y=0,yend=diff,colour=family),linewidth=0.55,alpha=0.9)+
  geom_point(data=B%>%filter(family!="other"),aes(colour=family),size=1.6)+
  geom_text_repel(data=lab,aes(label=l,colour=family),size=2.55,max.overlaps=Inf,min.segment.length=0,segment.size=0.2,segment.colour=GTX,
                  nudge_y=ifelse(lab$diff>0,0.035,-0.035),direction="both",force=4,box.padding=0.3,show.legend=FALSE,seed=1)+
  scale_colour_manual(values=COL,name=NULL)+scale_y_continuous(limits=c(-0.2,0.17))+scale_x_continuous(expand=expansion(mult=0.03))+
  labs(x=paste0(nrow(B)," RNA-binding proteins, ranked"),y="change in the network's score (logit)")+theme_classic(base_size=10)+
  theme(axis.text=element_text(colour=GTX),axis.text.x=element_blank(),axis.ticks.x=element_blank(),legend.position="bottom",legend.key.size=unit(4,"mm"))+
  guides(colour=guide_legend(nrow=2,override.aes=list(size=2.5,linewidth=1)))
ggsave("out/fig3_implant_panel.pdf",p,width=7.8,height=6.0,device=cairo_pdf); ggsave("out/fig3_implant_panel.png",p,width=7.8,height=6.0,dpi=300,bg="white")
write.table(B,"data/implant_cisbp_best.tsv",sep="\t",quote=FALSE,row.names=FALSE); cat(nrow(B),"proteins\n[out] out/fig3_implant_panel\n")
