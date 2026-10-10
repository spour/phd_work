#!/usr/bin/env Rscript
# splice-reranking figure (11 panels, 4x3 with one blank)
suppressPackageStartupMessages({
  library(ggplot2); library(ggpubr); library(dplyr); library(readr); library(tidyr); library(scales)
})
PFX <- if (length(commandArgs(trailingOnly=TRUE))>=1) commandArgs(trailingOnly=TRUE)[1] else "rerank"
P <- function(s) paste0(PFX,"_",s)

col_rf<-"#0072B2"; col_splice<-"#7B5EA7"; col_supp<-"#0072B2"; col_grey<-"#8A8F98"
col_teal<-"#3E8E7E"; col_train<-"#7A9E7E"
theme_pub<-function(base=11) theme_pubr(base_size=base)+
  theme(plot.title=element_text(face="bold",size=base+1,margin=margin(b=1)),
        plot.subtitle=element_text(size=base-1.5,color="grey35",margin=margin(b=3)),
        legend.position="none",plot.margin=margin(6,8,6,6))
rd<-function(p) if(file.exists(p)) suppressMessages(read_tsv(p,show_col_types=FALSE)) else NULL
getm<-function(df,k){ if(is.null(df)||!"metric"%in%names(df)) return(NA); v<-df$value[df$metric==k]; if(length(v)) as.numeric(v[1]) else NA }
nlab<-function(n) paste0("n = ",format(round(n),big.mark=","))
ppr <-function(p){ if(is.na(p)) "" else if(p<1e-3) "p < 0.001" else paste0("p = ",formatC(p,format="g",digits=2)) }
ph<-function(t) ggplot()+annotate("text",0,0,label=paste0(t,"\n(run the flag)"),size=3,color=col_grey)+
  labs(title=t)+theme_void()+theme(plot.title=element_text(face="bold"))

pretty_name<-c(
  spl_upstream_max_me3="strongest upstream 3\u2032ss",spl_u1_suppression_up="U1/donor density (up)",
  spl_u1_suppression_down="U1/donor density (down)",spl_pas_prox_u1="U1 near polyA signal",
  spl_comp_dn_3ss="competing 3\u2032ss (down)",spl_comp_up_3ss="competing 3\u2032ss (up)",
  spl_interv_me5_max="strongest intervening 5\u2032ss",spl_term_3ss="terminal 3\u2032ss strength",
  spl_term_ppt="terminal polypyrimidine",spl_upstream_mean_u1="mean upstream U1",
  spl_u2af35_ag="U2AF35 (3\u2032 AG)",spl_sum_5ss="total donor strength",
  spl_mean_top3_5ss="mean top-3 donors",spl_n_5ss_1k="donor count / 1kb",
  spl_n_5ss_500="donor count / 500nt",spl_dist_to_max_3ss="distance to strongest 3\u2032ss",
  spl_dist_interv_me5="distance to intervening 5\u2032ss",spl_local_me5="local 5\u2032ss strength",
  spl_dn5ss_has_strong="strong downstream donor",spl_dist_nearest_5ss="distance to nearest donor")
relab<-function(f) ifelse(f%in%names(pretty_name),pretty_name[f],gsub("^spl_","",f))

mt <- rd(P("metrics.tsv"))
N_contested <- getm(mt,"n_contested_genes")
loco_floor <- getm(mt,"loco_p1_random_floor")
pm <- rd(P("permnull_metrics.tsv")); p_perm <- getm(pm,"p_value")

# a: per-chromosome loco
bc<-rd(P("genomewide_bychrom.tsv"))
if(!is.null(bc)){
  ord<-paste0("chr",c(1:22,"X","Y")); bc$chrom<-factor(bc$chrom,levels=ord[ord%in%bc$chrom])
  w<-bc$n/sum(bc$n); gwm<-sum(bc$p1*w); gwf<-sum(bc$floor*w); Ntot<-sum(bc$n)
  pA<-ggplot(bc,aes(chrom))+
    geom_segment(aes(xend=chrom,y=floor,yend=p1),color="#DADEE3",linewidth=1.3)+
    geom_point(aes(y=floor),color=col_grey,size=1.7)+
    geom_errorbar(aes(ymin=ci_lo,ymax=ci_hi),width=.3,color=col_splice,linewidth=.4)+
    geom_point(aes(y=p1),color=col_splice,size=2.2)+
    geom_hline(yintercept=gwm,linetype=2,color=col_splice,linewidth=.5)+
    scale_y_continuous(labels=percent_format(1),limits=c(0,max(bc$ci_hi)*1.1))+
    labs(title="Reranker lift on every held-out chromosome",
         subtitle=sprintf("LOCO splice %.3f vs floor %.3f; %s; %s",gwm,gwf,nlab(Ntot),ppr(p_perm)),
         x=NULL,y="P@1 (RF+splice)")+
    theme_pub()+theme(axis.text.x=element_text(angle=45,hjust=1,size=8))
} else pA<-ph("per-chromosome LOCO")

# d: atlas recovery
at<-rd(P("atlas.tsv"))
if(!is.null(at) && "on_atlas_rate"%in%names(at)){
  at$which<-factor(at$which,levels=c("RF would pick","reranker moves to"))
  ng<-if("n_changed"%in%names(at)) max(at$n_changed,na.rm=TRUE) else NA
  pD<-ggplot(at,aes(which,on_atlas_rate,fill=which))+
    geom_col(width=.62)+
    geom_text(aes(label=percent(on_atlas_rate,accuracy=1)),vjust=-.4,size=3.3)+
    scale_fill_manual(values=c("RF would pick"=col_rf,"reranker moves to"=col_splice),guide="none")+
    scale_y_continuous(labels=percent_format(1),expand=expansion(c(0,.14)))+
    labs(title="Reranker picks land on real polyA sites",
         subtitle=sprintf("gene pick within 50 nt of a polyA-atlas site%s",
                          if(!is.na(ng)) paste0("; ",nlab(ng)) else ""),
         x=NULL,y="on atlas")+theme_pub()
} else pD<-ph("atlas recovery")

# e: atlas usage-weighting -- clean density of the paired log2 usage ratio
au<-rd(P("atlas_usage.tsv")); aum<-rd(P("atlas_usage_metrics.tsv"))
if(!is.null(au) && all(c("rf_usage","rerank_usage","overrode")%in%names(au))){
  sub<-au%>%filter(overrode%in%c(TRUE,"True","TRUE"),
                   !is.na(rf_usage),!is.na(rerank_usage),rf_usage>0,rerank_usage>0)%>%
       mutate(lr=log2(rerank_usage/rf_usage))
  if(nrow(sub)>=10){
    psign<-getm(aum,"p_sign"); fr<-getm(aum,"frac_rerank_higher_usage"); med<-median(sub$lr)
    pE<-ggplot(sub,aes(lr))+
      geom_density(fill=col_splice,color=col_splice,alpha=.30,linewidth=.8,adjust=1.1)+
      geom_vline(xintercept=0,linetype=2,color=col_grey,linewidth=.5)+
      geom_vline(xintercept=med,color=col_teal,linewidth=.8)+
      annotate("text",x=med,y=Inf,label=sprintf("median %+.2f",med),color=col_teal,
               hjust=-0.08,vjust=1.6,size=2.8)+
      scale_y_continuous(expand=expansion(c(0,.05)))+
      labs(title="Overrides move to more-used polyA sites",
           subtitle=sprintf("log2(reranker / RF site usage), overrides on atlas; %s%s",nlab(nrow(sub)),
                            if(!is.na(psign)) paste0("; ",ppr(psign)) else ""),
           x="log2(reranker usage / RF usage)",y="density")+theme_pub()
  } else pE<-ph("atlas usage (few overrides on atlas)")
} else pE<-ph("atlas usage (no usage column)")

# f: override outcome
pg<-rd(P("override_pergene.tsv"))
if(!is.null(pg)){
  ov<-pg%>%filter(overrode%in%c(TRUE,"True","TRUE"))
  brk<-function(s){s<-as.numeric(s);c(`on true site`=mean(s==0,na.rm=TRUE),upstream=mean(s<0,na.rm=TRUE),downstream=mean(s>0,na.rm=TRUE))}
  D2<-rbind(data.frame(who="reranker",outcome=names(brk(ov$spl_signed)),frac=as.numeric(brk(ov$spl_signed))),
            data.frame(who="RF argmax",outcome=names(brk(ov$rf_signed)),frac=as.numeric(brk(ov$rf_signed))))
  D2$who<-factor(D2$who,levels=c("RF argmax","reranker"))
  D2$outcome<-factor(D2$outcome,levels=c("on true site","upstream","downstream"))
  pF<-ggplot(D2,aes(who,frac,fill=outcome))+geom_col(width=.6)+
    geom_text(data=subset(D2,frac>=.05),aes(label=sprintf("%.0f%%",frac*100)),position=position_stack(vjust=.5),color="white",size=3)+
    scale_fill_manual(values=c("on true site"=col_splice,"upstream"=col_teal,"downstream"=col_rf))+
    scale_y_continuous(labels=percent_format(1),expand=expansion(c(0,.02)))+
    labs(title="Override outcome",subtitle=sprintf("%s where the pick changed",nlab(nrow(ov))),
         x=NULL,y="share of overridden genes",fill=NULL)+theme_pub()+theme(legend.position="top")
} else pF<-ph("override outcome")

im<-rd(P("importance.tsv"))
if(!is.null(im)){
  im<-im%>%filter(p1_drop>0)%>%arrange(desc(p1_drop))%>%slice_head(n=10)%>%
    mutate(feature=relab(feature),
           dir=ifelse(direction<0,"marks competitor (\u2212)","marks true end (+)"),
           feature=factor(feature,levels=rev(feature)))
  subG<-if(!is.na(N_contested)) sprintf("%s; permutation drop \u00b1 sd",nlab(N_contested)) else "permutation drop \u00b1 sd"
  pG<-ggplot(im,aes(feature,p1_drop,fill=dir))+
    geom_col(width=.72)+
    geom_errorbar(aes(ymin=pmax(0,p1_drop-p1_drop_sd),ymax=p1_drop+p1_drop_sd),width=.25,linewidth=.3)+
    coord_flip()+
    scale_fill_manual(values=c("marks true end (+)"=col_splice,"marks competitor (\u2212)"=col_supp))+
    scale_y_continuous(expand=expansion(c(0,.10)))+
    labs(title="Feature importance",subtitle=subG,x=NULL,y="\u0394 P@1 when shuffled",fill=NULL)+theme_pub()+
    theme(legend.position=c(.62,.16),legend.text=element_text(size=8),
          legend.background=element_rect(fill="white",color=NA),legend.key.size=unit(.85,"lines"))
} else pG<-ph("importance")

svp<-function(){
  VL<-rd(P("violins_long.tsv")); VS<-rd(P("violins_stats.tsv"))
  if(is.null(VL)) return(ph("feature violins"))
  VL<-do.call(rbind,lapply(split(VL,VL$feature),function(d){
    lo<-quantile(d$value,.005,na.rm=TRUE);hi<-quantile(d$value,.995,na.rm=TRUE)
    d$value<-pmin(pmax(d$value,lo),hi);d}))
  lab<-setNames(relab(VS$feature),VS$feature)
  poly<-list();seg<-list();k<-1;hw<-.42
  for(f in unique(VL$feature)){
    sub<-VL[VL$feature==f,]
    dens<-lapply(c("canonical","competitor"),function(g){
      v<-sub$value[sub$group==g];v<-v[is.finite(v)];if(length(v)<5)return(NULL)
      d<-density(v,n=128);list(g=g,vx=d$x,vy=d$y,q=quantile(v,c(.25,.5,.75)))})
    dens<-Filter(Negate(is.null),dens);if(!length(dens))next
    md<-max(sapply(dens,function(z)max(z$vy)))
    for(z in dens){sd<-z$vy/md*hw;side<-if(z$g=="canonical")-1 else 1;xx<-side*sd;n<-length(xx)
      poly[[k]]<-data.frame(facet=lab[[f]],grp=z$g,x=c(xx,0,0),y=c(z$vx,z$vx[n],z$vx[1]),gid=paste(f,z$g))
      seg[[k]]<-data.frame(facet=lab[[f]],grp=z$g,xpos=side*.11,ymin=z$q[1],ymed=z$q[2],ymax=z$q[3]);k<-k+1}}
  PP<-do.call(rbind,poly);SS<-do.call(rbind,seg)
  PP$grp<-factor(PP$grp,levels=c("canonical","competitor"),labels=c("true end","competitor"))
  SS$grp<-factor(SS$grp,levels=c("canonical","competitor"),labels=c("true end","competitor"))
  ann<-VS;ann$facet<-relab(ann$feature);ann$txt<-sprintf("r=%+.2f",ann$rank_biserial)
  ggplot()+
    geom_polygon(data=PP,aes(x,y,fill=grp,group=gid),color=NA,alpha=.85)+
    geom_segment(data=SS,aes(x=xpos,xend=xpos,y=ymin,yend=ymax),color="black",linewidth=.5)+
    geom_point(data=SS,aes(x=xpos,y=ymed),shape=21,fill="white",color="black",size=1.6)+
    facet_wrap(~facet,scales="free_y")+
    geom_text(data=ann,aes(x=0,y=Inf,label=txt),inherit.aes=FALSE,vjust=1.4,size=2.6,color="#555555")+
    scale_fill_manual(values=c("true end"=col_splice,"competitor"=col_train),name=NULL)+
    scale_x_continuous(limits=c(-.55,.55),breaks=NULL)+
    labs(title="Feature distributions: true ends vs competitors",
         subtitle=sprintf("%s; rank-biserial r; all p < 0.001 (Mann-Whitney)",
                          if(!is.na(N_contested)) nlab(N_contested) else ""),
         x=NULL,y="feature value")+
    theme_pub()+theme(strip.text=element_text(face="bold",size=9),legend.position="top")
}
pH<-svp()

cb<-rd(P("by_contest.tsv"))
if(!is.null(cb)){
  cb$bucket<-factor(cb$bucket,levels=c("1","2","3-4","5-8","9+"))
  EL<-rbind(data.frame(bucket=cb$bucket,method="floor",rate=cb$floor),
            data.frame(bucket=cb$bucket,method="RF + splice",rate=cb$p1))
  EL$method<-factor(EL$method,levels=c("floor","RF + splice"))
  pI<-ggplot(EL,aes(bucket,rate,fill=method))+
    geom_col(width=.66,position=position_dodge(.72))+
    geom_text(aes(label=sprintf("%.2f",rate)),position=position_dodge(.72),vjust=-.35,size=2.6)+
    geom_text(data=cb,aes(x=bucket,y=-0.03,label=paste0("n=",format(n_genes,big.mark=","))),
              inherit.aes=FALSE,size=2.4,color="grey40")+
    scale_fill_manual(values=c("floor"=col_grey,"RF + splice"=col_splice))+
    scale_y_continuous(limits=c(-0.06,max(EL$rate)*1.18),expand=expansion(c(0,0)),labels=percent_format(1))+
    labs(title="Correct rate by contest difficulty",
         subtitle=sprintf("%s across buckets",if(!is.na(N_contested)) nlab(N_contested) else "contested genes"),
         x="competitors per gene",y="P@1",fill=NULL)+
    theme_pub()+theme(legend.position=c(.75,.92),legend.direction="horizontal",
                      legend.background=element_rect(fill="white",color=NA))
} else pI<-ph("by contest")

nA<-rd(P("permnull.tsv"))
if(!is.null(nA)){
  fl<-getm(pm,"floor"); if(is.na(fl)) fl<-loco_floor
  if("p1"%in%names(nA)){ realv<-nA$p1[nA$kind=="real"][1]; nulv<-nA$p1[nA$kind=="null"] }
  else { realv<-nA$boost[nA$kind=="real"][1]+fl; nulv<-nA$boost[nA$kind=="null"]+fl }
  loco<-getm(pm,"loco_boost")
  pJ<-ggplot(data.frame(v=nulv),aes(v))+
    geom_histogram(bins=max(15,min(40,round(length(nulv)/8))),fill=col_grey,color="white",linewidth=.15,alpha=.9)+
    geom_vline(xintercept=realv,color=col_splice,linewidth=1.1)+
    annotate("text",x=realv,y=Inf,label=sprintf("real %.3f",realv),color=col_splice,hjust=1.1,vjust=1.8,size=3,fontface="bold")+
    labs(title="Permutation null (label-shuffled)",
         subtitle=sprintf("%s shuffled-label fits; %s",format(length(nulv),big.mark=","),ppr(p_perm)),
         x="reranker P@1 (fixed split)",y="count")+theme_pub()
} else pJ<-ph("permutation null")

mg<-rd(P("margin.tsv"))
if(!is.null(mg) && "margin"%in%names(mg)){
  mg$grp<-ifelse(mg$margin>0,"canonical on top","competitor on top")
  fr<-mean(mg$margin>0,na.rm=TRUE)
  pK<-ggplot(mg,aes(margin,fill=grp))+
    geom_histogram(bins=60,color=NA,alpha=.85,position="identity")+
    geom_vline(xintercept=0,linetype=2,color="grey20",linewidth=.5)+
    scale_fill_manual(values=c("canonical on top"=col_splice,"competitor on top"=col_grey),name=NULL)+
    scale_y_continuous(expand=expansion(c(0,.05)))+
    labs(title="When it's right, it's decisive; misses are near-ties",
         subtitle=sprintf("reranker score: canonical \u2212 best competitor; %s",nlab(nrow(mg))),
         x="score margin",y="genes")+
    theme_pub()+theme(legend.position=c(.02,.95),legend.justification=c(0,1),
                      legend.text=element_text(size=8),legend.key.size=unit(.85,"lines"),
                      legend.background=element_rect(fill="white",color=NA))
} else pK<-ph("score margin")

sp<-rd(P("selection_path.tsv")); sf<-rd(P("selected_features.tsv"))
if(!is.null(sp)){
  kept<-if(!is.null(sf)) nrow(sf) else NA
  full_b<-max(sp$boost,na.rm=TRUE)
  pSel<-ggplot(sp,aes(k,boost))+
    geom_hline(yintercept=full_b,linetype=2,color=col_grey,linewidth=.5)+
    geom_line(color=col_splice,linewidth=.9)+geom_point(color=col_splice,size=1.6)+
    {if(!is.na(kept)) geom_point(data=subset(sp,k==kept),aes(k,boost),color=col_teal,size=3.4)}+
    {if(!is.na(kept)) geom_vline(xintercept=kept,linetype=3,color=col_teal,linewidth=.5)}+
    scale_y_continuous(labels=percent_format(0.1))+
    labs(title="Only a few features carry the boost",
         subtitle=sprintf("LOCO boost vs # features added by importance%s",
                          if(!is.na(kept)) sprintf("; kept %d",kept) else ""),
         x="features kept (ranked by importance)",y="P@1 boost over floor")+theme_pub()
} else pSel<-ph("selection curve")

pdp<-rd(P("pdp.tsv"))
if(!is.null(pdp)){
  pdp$flab<-factor(relab(pdp$feature),levels=unique(relab(pdp$feature)))
  pPDP<-ggplot(pdp,aes(x,pd))+
    geom_line(color=col_splice,linewidth=1)+
    facet_wrap(~flab,scales="free_x",ncol=2)+
    scale_y_continuous(labels=percent_format(1))+
    labs(title="How the reranker uses each feature",
         subtitle="partial dependence: P(canonical) vs feature (others held at data)",
         x="feature value",y="P(canonical)")+
    theme_pub()+theme(strip.text=element_text(face="bold",size=8),
                      panel.spacing=unit(.5,"lines"),axis.text=element_text(size=7))
} else pPDP<-ph("partial dependence")

ex<-rd(P("examples.tsv"))
if(!is.null(ex) && all(c("rel_kb","spl_upstream_max_me3")%in%names(ex))){
  ex$gene<-factor(ex$gene,levels=unique(ex$gene))
  istrue<-function(x) x%in%c(TRUE,"True","TRUE")
  ex$role<-ifelse(istrue(ex$is_rerank_pick),"reranker pick = true 3\u2032 end",
            ifelse(istrue(ex$is_rf_pick),"RF's pick","other candidate site"))
  ex$role<-factor(ex$role,levels=c("reranker pick = true 3\u2032 end","RF's pick","other candidate site"))
  pEX<-ggplot(ex,aes(rel_kb,spl_upstream_max_me3,color=role,size=role))+
    geom_vline(xintercept=0,linetype=3,color=col_grey,linewidth=.4)+
    geom_point()+
    facet_wrap(~gene,ncol=1,scales="free_x")+
    scale_color_manual(values=c("reranker pick = true 3\u2032 end"=col_splice,
                                "RF's pick"=col_rf,"other candidate site"="#C2C7CE"),name=NULL)+
    scale_size_manual(values=c("reranker pick = true 3\u2032 end"=3.9,"RF's pick"=3.9,
                               "other candidate site"=1.8),guide="none")+
    labs(title="Worked examples: reranker moves to the strong-acceptor true end",
         subtitle="per gene: each candidate site placed by position and splice-acceptor strength",
         x="position relative to true 3\u2032 end (kb)",
         y="acceptor strength (MaxEnt 3\u2032ss)")+
    theme_pub()+theme(strip.text=element_text(face="bold",size=8.5),
                      panel.spacing=unit(.45,"lines"),
                      legend.position="top",legend.text=element_text(size=8))
} else pEX<-ph("worked examples")

fig<-ggarrange(pA,pI,pJ, pD,pE,pF, pG,pH,pPDP, pSel,pEX,pK, ncol=3,nrow=4,
               labels=c("A","B","C","D","E","F","G","H","I","J","K","L"),font.label=list(size=14,face="bold"))
fig<-annotate_figure(fig,top=text_grob("Splicing-informed reranking of the CPA candidate-selection pipeline",face="bold",size=13))
ggsave("figure_reranking.pdf",fig,width=15,height=17.5,device=cairo_pdf,limitsize=FALSE)
ggsave("figure_reranking.png",fig,width=15,height=17.5,dpi=200,limitsize=FALSE)
cat("wrote figure_reranking.pdf / .png\n")

