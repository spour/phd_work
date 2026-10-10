#!/usr/bin/env Rscript
# figure s3, the 8-gene architecture panel
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(tidyr);library(readr)})
args<-commandArgs(trailingOnly=TRUE)
INDIR<-if(length(args)>=1) args[1] else "."
need<-function(p){ if(!file.exists(p)) stop("missing required input: ",p,call.=FALSE); p }
rdp<-function(f) read_tsv(need(file.path(INDIR,f)),show_col_types=FALSE)

INK<-"#1A1A1A"; BASE<-13; NUL<-"#8A8F98"
CLS<-c("Competitor","Real end")
FEATS<-c("3′ss (MaxEnt ≥ 8)","5′ss (MaxEnt ≥ 8)")
# colour = feature
# figure 3, so a reader cannot mistake a colour here for a class
FCOL<-setNames(c("#008B8B","#C2258F"),FEATS)
SHP <-setNames(c(24,21),FEATS)
U1FILL<-"#8FA8C8"
NCOL<-4

GEN<-rdp("exemplar_genes.tsv"); TRK<-rdp("exemplar_tracks.tsv"); U1<-rdp("exemplar_u1.tsv")

# alphabetical
# gradient the sample does not have and would read as if the genes were ranked
GEN<-GEN%>%arrange(gene)%>%
     mutate(slot=row_number(),
            grow=((slot-1)%/%NCOL)+1, gcol=((slot-1)%%NCOL)+1,
            head=gene)
key<-GEN%>%select(gene,grow,gcol,head,contrast)

SLIM<-ceiling(max(TRK$score,na.rm=TRUE))
ULIM<-max(U1$u1_density,na.rm=TRUE)
u2y<-function(u) u/ULIM*SLIM

MARK<-TRK%>%inner_join(key,by="gene")%>%
      mutate(feat=factor(ifelse(kind=="acceptor",FEATS[1],FEATS[2]),levels=FEATS),
             class=factor(class,levels=CLS),yv=score)
AREA<-U1%>%inner_join(key,by="gene")%>%
      mutate(yv=u2y(u1_density),class=factor(class,levels=CLS))

NROW<-max(GEN$grow)
mkrow<-function(grow,class) factor(sprintf("%d|%s",grow,class),
        levels=as.vector(t(outer(1:NROW,CLS,function(a,b) sprintf("%d|%s",a,b)))))
MARK$frow<-mkrow(MARK$grow,MARK$class); AREA$frow<-mkrow(AREA$grow,AREA$class)
LANEMAP<-setNames(sub("^[0-9]+[|]","",levels(MARK$frow)),levels(MARK$frow))

HEAD<-key%>%mutate(class=CLS[1],frow=mkrow(grow,class))
SCOR<-key%>%crossing(class=CLS)%>%
      mutate(frow=mkrow(grow,class),xs=-500,xe=0,xm=-250,txt="scored window")

message(sprintf("[S3] %d genes in %d rows x %d cols",nrow(GEN),NROW,NCOL))
message("[S3] panel order (top-left first, reading across):")
for(i in seq_len(nrow(GEN)))
  message(sprintf("[S3]   r%dc%d  %-10s p%-5.1f contrast %+7.2f",
          GEN$grow[i],GEN$gcol[i],GEN$gene[i],GEN$actual_pct[i],GEN$contrast[i]))
for(f in FEATS) message(sprintf("[S3] colour %-20s %s",f,FCOL[f]))
message(sprintf("[S3] shaded U1 layer %s; no U1 stems",U1FILL))
message(sprintf("[S3] shared axes: MaxEnt 0..%.0f bits, U1 0..%.0f sites/kb",SLIM,ULIM))
for(f in FEATS) message(sprintf("[S3] stems, %-20s %d",f,sum(MARK$feat==f)))

p<-ggplot()+
  geom_area(data=AREA,aes(pos,yv),fill=U1FILL,alpha=.45,color=NA)+
  geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.35)+
  geom_segment(data=MARK,aes(x=pos,xend=pos,y=0,yend=yv,color=feat),linewidth=.45)+
  geom_point(data=MARK,aes(pos,yv,shape=feat,color=feat,fill=feat),size=2.0,stroke=.35)+
  geom_segment(data=SCOR,aes(x=xs,xend=xe,y=-Inf,yend=-Inf),color=INK,linewidth=.7)+
  geom_text(data=SCOR,aes(x=xm,y=-Inf,label=txt),vjust=-0.55,size=2.3,color="#555555")+
  geom_text(data=HEAD,aes(x=-2000,y=Inf,label=head),fontface=2,
            hjust=0,vjust=1.35,size=3.4,color=INK)+
  scale_color_manual(values=FCOL,name="feature")+
  scale_fill_manual(values=FCOL,name="feature")+
  scale_shape_manual(values=SHP,name="feature")+
  scale_x_continuous(breaks=c(-2000,-1500,-1000,-500,0),
                     labels=c("-2 kb","-1.5 kb","-1 kb","-500","pA"),
                     expand=expansion(mult=.03))+
  scale_y_continuous(limits=c(0,SLIM),expand=expansion(mult=c(.02,.28)),
                     name="MaxEnt score (bits)",
                     sec.axis=sec_axis(~ . /SLIM*ULIM,name="U1 density (sites per kb)"))+
  facet_grid(frow~gcol,switch="y",
             labeller=labeller(frow=LANEMAP,gcol=function(x) rep("",length(x))))+
  labs(title="Splicing signal architecture at 20 genes drawn at random",
       subtitle=paste0("Figure S3, supporting Figure 3 F, G and J. Each gene shows its competitor (top lane) and its real 3' end (bottom lane) over the\n",
                       "2 kb upstream of that peak's own pA. Colour gives the feature and is the same in both lanes, so class is read by lane position."),
       caption=paste0("Splicing signals across 20 genes drawn at random (seed 42) from those with a canonical peak, at least one competitor, and their anchor 3'ss\n",
                      "within the window. Competitor (top) and real 3' end (bottom), 2 kb upstream of the pA. Teal = strong 3'ss, magenta = strong 5'ss (MaxEnt >= 8);\n",
                      "shading = U1 density. The within-gene U1 enrichment (Fig 3F/J, +18 sites/kb, win-rate 0.62) is an aggregate tendency; this panel shows the\n",
                      "underlying architecture, not a per-gene effect. Terminal-exon length does not differ within gene (median diff -19 nt)."),
       x="position relative to that peak's pA site")+
  theme_classic(base_size=BASE)+
  theme(text=element_text(color=INK),
        axis.text=element_text(color=INK,size=BASE-4),
        axis.text.y=element_text(size=BASE-5),
        axis.title.y=element_text(size=BASE-3),
        axis.line=element_line(color=INK,linewidth=.35),
        plot.title=element_text(size=BASE+3,face="bold",hjust=0),
        plot.subtitle=element_text(size=BASE-3,color="#555555"),
        plot.caption=element_text(size=BASE-4,color="#666666",hjust=0),
        strip.background=element_blank(),
        strip.placement="outside",
        strip.text.y.left=element_text(size=BASE-3,color=INK,angle=0),
        strip.text.x=element_blank(),
        panel.spacing.x=unit(1.4,"lines"),
        panel.spacing.y=unit(0.35,"lines"),
        panel.background=element_rect(fill="grey98",color=NA),
        legend.position="top",legend.box="horizontal",
        legend.text=element_text(size=BASE-3),legend.title=element_text(size=BASE-3))

ggsave(file.path(INDIR,"figS3_exemplar_tracks.pdf"),p,width=19,height=16,device=cairo_pdf)
ggsave(file.path(INDIR,"figS3_exemplar_tracks.png"),p,width=19,height=16,dpi=300)
cat("[out] figS3_exemplar_tracks.pdf / .png\n")
