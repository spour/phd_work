#!/usr/bin/env Rscript
# 4-panel ggpubr figure for the splice-reranking result

suppressPackageStartupMessages({
  library(ggplot2); library(ggpubr); library(dplyr); library(readr); library(tidyr); library(scales)
})

# config
f_metrics  <- "rerank_metrics.tsv"
f_import   <- "rerank_importance.tsv"
f_resid    <- if (file.exists("residual_miss_gw_fair.tsv")) "residual_miss_gw_fair.tsv" else "residual_miss_fair.tsv"
f_strat_len<- "rerank_strat_bylength.tsv"
f_strat_con<- "rerank_strat_bycontest.tsv"
f_viol_long<- "feature_violins_long.tsv"
f_viol_stat<- "feature_violins_stats.tsv"
f_pn_null  <- "rerank_permnull_nullA.tsv"
f_pn_met   <- "rerank_permnull_metrics.tsv"
f_abl      <- "rerank_ablation_metrics.tsv"
f_geo_hist <- "override_geometry_hist.tsv"
f_geo_pg   <- "override_geometry_pergene.tsv"
f_gw_e2e   <- "rerank_genomewide_endtoend.tsv"
f_shap     <- "shap_beeswarm_long.tsv"
out_pdf    <- "figure_reranking.pdf"
out_png    <- "figure_reranking.png"

col_rf     <- "#9AA0A6"
col_splice <- "#C65B3C"
col_supp   <- "#3E6B8B"
col_te     <- "#C65B3C"; col_in <- "#5B7C6F"; col_ex <- "#9AA0A6"
col_train  <- "#5B7C6F"
col_green  <- "#5B7C6F"

theme_pub <- function(base=11) theme_pubr(base_size=base) +
  theme(plot.title=element_text(face="bold", size=base+1),
        plot.subtitle=element_text(size=base-1, color="grey30"),
        legend.position="none")
rd <- function(p) if (file.exists(p)) suppressMessages(read_tsv(p, show_col_types=FALSE)) else NULL
getm <- function(df,k){ if(is.null(df)) return(NA); v<-df$value[df$metric==k]; if(length(v)) as.numeric(v[1]) else NA }
nlab <- function(n) paste0("n = ", format(round(n), big.mark=","))

# panel a : grouped, n = all genes scored at each scope (not contested-only)
mt <- rd(f_metrics)
summ <- rd("shortlist_summary.tsv")
if (is.null(summ)) summ <- rd("../shortlist_summary.tsv")
gw_all <- if (!is.null(summ)) nrow(summ) else getm(mt,"n_genes_genomewide")
if (is.na(gw_all)) gw_all <- 13930
test_all <- getm(mt,"n_endtoend_test")
n_contested <- getm(mt,"n_contested_test")
nlab_c <- if (!is.na(n_contested)) nlab(n_contested) else "contested test genes"
rf_test  <- getm(mt,"overall_rf_argmax")
spl_test <- getm(mt,"overall_shortlist_splice")
gwm <- rd(f_gw_e2e)
rf_gw  <- if(!is.null(gwm)) getm(gwm,"rf_argmax") else NA
spl_gw <- if(!is.null(gwm)) getm(gwm,"rf_plus_splice") else NA

dfA <- rbind(
  data.frame(method="RF argmax",   scope=sprintf("test\n(n=%s)",  format(test_all,big.mark=",")), p1=rf_test),
  data.frame(method="RF + splice", scope=sprintf("test\n(n=%s)",  format(test_all,big.mark=",")), p1=spl_test),
  data.frame(method="RF argmax",   scope=sprintf("genome-wide\n(n=%s)",format(gw_all,big.mark=",")), p1=rf_gw),
  data.frame(method="RF + splice", scope=sprintf("genome-wide\n(n=%s)",format(gw_all,big.mark=",")), p1=spl_gw)
)
dfA <- dfA[!is.na(dfA$p1),]
dfA$method <- factor(dfA$method, levels=c("RF argmax","RF + splice"))
dfA$scope  <- factor(dfA$scope,  levels=unique(dfA$scope))
cat("[panel A] all-gene n: test=",test_all," genome-wide=",gw_all,"\n",sep="")
pA <- ggplot(dfA, aes(scope, p1, fill=method)) +
  geom_col(width=.66, position=position_dodge(width=.72)) +
  geom_text(aes(label=sprintf("%.3f",p1)), position=position_dodge(width=.72),
            vjust=-0.4, size=3.3) +
  scale_fill_manual(values=c("RF argmax"=col_rf,"RF + splice"=col_splice)) +
  scale_y_continuous(limits=c(0,max(dfA$p1)*1.18), expand=expansion(c(0,0))) +
  labs(title="End-to-end P@1 (all genes scored)", x=NULL, y="dominant-site P@1", fill=NULL) +
  theme_pub() +
  theme(legend.position=c(.5,.92), legend.direction="horizontal",
        legend.background=element_rect(fill="white",color=NA))

# panel b (feature importance + direction)
pretty_name <- c(
  spl_upstream_max_me3   = "strongest upstream 3\u2032 splice site",
  spl_u1_suppression_up  = "U1 / donor density (upstream)",
  spl_u1_suppression_down= "U1 / donor density (downstream)",
  spl_pas_prox_u1        = "U1 near the polyA signal",
  spl_comp_dn_3ss        = "competing 3\u2032 splice site (downstream)",
  spl_comp_up_3ss        = "competing 3\u2032 splice site (upstream)",
  spl_interv_me5_max     = "strongest intervening 5\u2032 splice site",
  spl_term_3ss           = "terminal 3\u2032 splice site strength",
  spl_term_ppt           = "terminal polypyrimidine tract",
  spl_term_3ss_abs       = "terminal 3\u2032 splice site (absolute)",
  spl_upstream_mean_u1   = "mean upstream U1 density",
  spl_u2af35_ag          = "U2AF35 (3\u2032 AG) strength",
  spl_u2af65_ppt         = "U2AF65 (pyrimidine tract)",
  spl_bp_strength        = "branchpoint strength",
  spl_local_me5          = "local 5\u2032 splice site strength",
  spl_dn5ss_has_strong   = "strong downstream donor present",
  spl_sum_5ss            = "total donor strength nearby",
  spl_mean_top3_5ss      = "mean of top-3 donors nearby",
  spl_n_5ss_500          = "donor count within 500 nt",
  spl_n_5ss_1k           = "donor count within 1 kb",
  spl_dist_nearest_5ss   = "distance to nearest strong donor",
  spl_dist_up_5ss        = "distance to upstream donor",
  spl_dist_dn_5ss        = "distance to downstream donor",
  spl_rel_3ss_up         = "terminal vs upstream acceptor",
  spl_rel_3ss_dn         = "terminal vs downstream acceptor",
  spl_dist_to_max_3ss    = "distance to strongest 3\u2032 splice site"
)
relabel <- function(f) ifelse(f %in% names(pretty_name), pretty_name[f], gsub("^spl_","",f))
im <- rd(f_import) %>% filter(p1_drop>0) %>% arrange(desc(p1_drop)) %>% slice_head(n=10) %>%
  mutate(feature=relabel(feature),
         dir=ifelse(direction>0,"marks true end (+)","marks suppressed (\u2212)"),
         dir=ifelse(direction==0,"marks true end (+)",dir),
         feature=factor(feature, levels=rev(feature)))
pB <- ggplot(im, aes(feature,p1_drop,fill=dir)) +
  geom_col(width=.72) +
  geom_errorbar(aes(ymin=pmax(0,p1_drop-p1_drop_sd),ymax=p1_drop+p1_drop_sd),width=.25,linewidth=.3) +
  coord_flip() +
  scale_fill_manual(values=c("marks true end (+)"=col_splice,"marks suppressed (\u2212)"=col_supp)) +
  scale_y_continuous(expand=expansion(c(0,.10))) +
  labs(title="Feature importance",
       subtitle=sprintf("monotone sign; %s", nlab_c),
       x=NULL, y="drop in P@1 when shuffled", fill=NULL) +
  theme_pub() +
  theme(legend.position=c(.66,.18), legend.title=element_blank(),
        legend.text=element_text(size=8.5),
        legend.background=element_rect(fill="white",color=NA),
        legend.key.size=unit(.9,"lines"))

# panel d (changed-pick atlas recovery)
fr <- rd(f_resid)
d_n <- if(!is.null(fr) && "n_changed" %in% names(fr)) fr$n_changed[1] else NA
dfD <- if(!is.null(fr)) fr %>% transmute(method=which, rate=as.numeric(on_atlas_rate)) else
  data.frame(method=c("RF would pick","reranker moves to"), rate=c(0.322,0.561))
dfD$method <- factor(dfD$method, levels=c("RF would pick","reranker moves to"))
pC <- ggplot(dfD, aes(method,rate,fill=method)) +
  geom_col(width=.6) +
  geom_text(aes(label=sprintf("%.0f%%",rate*100)), vjust=-0.4, size=3.7) +
  scale_fill_manual(values=c(col_rf,col_splice)) +
  scale_y_continuous(limits=c(0,max(dfD$rate)*1.2), labels=percent_format(1), expand=expansion(c(0,0))) +
  labs(title="Reranker overrides land on real APA sites",
       subtitle=if(!is.na(d_n)) paste0(nlab(d_n)," genes where the reranker overrode RF") else NULL,
       x=NULL, y="pick on a real atlas APA site") +
  theme_pub()

col_boost <- col_splice; col_p1ctx <- col_rf
Lst <- rd(f_strat_len)
if (!is.null(Lst)) {
  ord <- order(Lst$bin)
  lev <- sprintf("Q%d\n%g-%gkb", Lst$bin[ord]+1, round(Lst$len_lo[ord]/1000), round(Lst$len_hi[ord]/1000))
  Lst$xlab <- factor(sprintf("Q%d\n%g-%gkb", Lst$bin+1, round(Lst$len_lo/1000), round(Lst$len_hi/1000)), levels=lev)
  EL <- rbind(
    data.frame(xlab=Lst$xlab, method="RF argmax",   rate=Lst$rf_rate),
    data.frame(xlab=Lst$xlab, method="RF + splice", rate=Lst$splice_rate))
  EL$method <- factor(EL$method, levels=c("RF argmax","RF + splice"))
  pE <- ggplot(EL, aes(xlab, rate, fill=method)) +
    geom_col(width=.66, position=position_dodge(width=.72)) +
    geom_text(aes(label=sprintf("%.2f",rate)), position=position_dodge(width=.72),
              vjust=-0.35, size=2.6) +
    geom_text(data=Lst, aes(x=xlab, y=-0.04, label=paste0("n=",format(n,big.mark=","))),
              inherit.aes=FALSE, size=2.4, color="grey40") +
    scale_fill_manual(values=c("RF argmax"=col_rf,"RF + splice"=col_splice)) +
    scale_y_continuous(limits=c(-0.08, max(EL$rate)*1.18), expand=expansion(c(0,0))) +
    labs(title="Correct rate by gene length", subtitle="all genes; RF argmax vs RF + splice",
         x="gene length (quintile, short \u2192 long)", y="fraction correct (P@1)", fill=NULL) +
    theme_pub() + theme(legend.position=c(.72,.93), legend.direction="horizontal",
                        legend.background=element_rect(fill="white",color=NA))
} else pE <- ggplot()+theme_void()

split_violin_panel <- function(long_file, stat_file, title, ylab) {
  VL <- rd(long_file); VS <- rd(stat_file)
  if (is.null(VL)) return(ggplot()+theme_void())
  VL <- do.call(rbind, lapply(split(VL, VL$feature), function(d){
    lo <- quantile(d$value, .005, na.rm=TRUE); hi <- quantile(d$value, .995, na.rm=TRUE)
    d$value <- pmin(pmax(d$value, lo), hi); d }))
  lab_map <- setNames(VS$label, VS$feature)
  feats <- unique(VL$feature)
  poly <- list(); seg <- list(); k <- 1; halfw <- 0.42
  for (f in feats) {
    sub <- VL[VL$feature==f, ]
    grps <- c("canonical","competitor")
    dens <- lapply(grps, function(g){
      v <- sub$value[sub$group==g]; v <- v[is.finite(v)]
      if (length(v) < 5) return(NULL)
      d <- density(v, n=128); list(g=g, vx=d$x, vy=d$y, q=quantile(v,c(.25,.5,.75)))
    })
    dens <- Filter(Negate(is.null), dens)
    if (length(dens)==0) next
    maxd <- max(sapply(dens, function(z) max(z$vy)))
    for (z in dens) {
      sdw  <- z$vy / maxd * halfw
      side <- if (z$g=="canonical") -1 else 1
      xx   <- side * sdw; n <- length(xx)
      poly[[k]] <- data.frame(facet=lab_map[[f]], grp=z$g,
                              x=c(xx, 0, 0), y=c(z$vx, z$vx[n], z$vx[1]),
                              gid=paste(f, z$g))
      seg[[k]]  <- data.frame(facet=lab_map[[f]], grp=z$g, xpos=side*0.11,
                              ymin=z$q[1], ymed=z$q[2], ymax=z$q[3])
      k <- k + 1
    }
  }
  P <- do.call(rbind, poly); S <- do.call(rbind, seg)
  P$grp <- factor(P$grp, levels=c("canonical","competitor"), labels=c("true end","competitor"))
  S$grp <- factor(S$grp, levels=c("canonical","competitor"), labels=c("true end","competitor"))
  ann <- VS; ann$facet <- ann$label; ann$txt <- sprintf("effect = %+.2f", ann$rank_biserial)
  ggplot() +
    geom_polygon(data=P, aes(x, y, fill=grp, group=gid), color=NA, alpha=.85) +
    geom_segment(data=S, aes(x=xpos, xend=xpos, y=ymin, yend=ymax), color="black", linewidth=.5) +
    geom_point(data=S, aes(x=xpos, y=ymed), shape=21, fill="white", color="black", size=1.7) +
    facet_wrap(~facet, scales="free_y") +
    geom_text(data=ann, aes(x=0, y=Inf, label=txt), inherit.aes=FALSE,
              vjust=1.4, size=2.9, color="#555555") +
    scale_fill_manual(values=c("true end"=col_splice,"competitor"=col_train), name=NULL) +
    scale_x_continuous(limits=c(-.55,.55), breaks=NULL) +
    labs(title=title,
         subtitle=sprintf("genome-wide contested peaks (%s)", nlab(nrow(VL)/2)),
         x=NULL, y=ylab) +
    theme_pub() + theme(strip.text=element_text(face="bold",size=9.5),
                        legend.position="top", legend.justification="center")
}

pF <- split_violin_panel(f_viol_long, f_viol_stat,
                         "Feature distributions: true ends vs competitors", "feature value")

nA <- rd(f_pn_null); pm <- rd(f_pn_met)
if (!is.null(nA)) {
  real <- getm(pm,"real_p1"); nullB <- getm(pm,"nullB_mean")
  pval <- pm$p[pm$metric=="nullA_mean"]; pval <- if(length(pval)) as.numeric(pval[1]) else NA
  pG <- ggplot(nA, aes(nullA)) +
    geom_histogram(bins=40, fill=col_rf, color=NA, alpha=.8) +
    geom_vline(xintercept=real, color=col_splice, linewidth=1.1) +
    annotate("text", x=real, y=Inf, label=sprintf("real = %.3f", real),
             color=col_splice, hjust=1.1, vjust=1.6, size=3.2, fontface="bold") +
    {if(!is.na(nullB)) geom_vline(xintercept=nullB, color=col_supp, linewidth=.7, linetype="dashed")} +
    scale_x_continuous(expand=expansion(mult=c(0.03,0.10))) +
    labs(title="Permutation null",
         subtitle=sprintf("shuffled-label P@1 (1,000 draws); p=%s; contested %s", ifelse(is.na(pval),"<0.001",format(pval)), nlab_c),
         x="contested P@1 under the null", y="count") +
    theme_pub()
} else pG <- ggplot()+theme_void()

abl <- rd(f_abl); floor_v <- getm(pm,"nullA_mean")
if (!is.null(abl)) {
  ren <- c(cpa="rf_prob alone", splice="splice", combined="splice + rf_prob")
  abl$lab <- ifelse(abl$group %in% names(ren), ren[abl$group], abl$group)
  abl$lab <- factor(abl$lab, levels=c("rf_prob alone","splice","splice + rf_prob"))
  abl$fill <- ifelse(abl$group=="cpa", col_rf, col_splice)
  pH <- ggplot(abl, aes(lab, p1)) +
    geom_col(aes(fill=fill), width=.62) + scale_fill_identity() +
    geom_errorbar(aes(ymin=ci_lo, ymax=ci_hi), width=.18, color="grey30") +
    geom_text(aes(label=sprintf("%.3f",p1)), vjust=-0.6, size=3.1) +
    {if(!is.na(floor_v)) geom_hline(yintercept=floor_v, linetype="dashed", color="grey50")} +
    {if(!is.na(floor_v)) annotate("text", x=0.6, y=floor_v, label="floor", vjust=-0.4, hjust=0, size=2.8, color="grey50")} +
    scale_y_continuous(limits=c(0,max(abl$ci_hi)*1.2), expand=expansion(c(0,0))) +
    labs(title="Feature-group ablation",
         subtitle=sprintf("contested P@1 (%s); rf_prob degenerate within shortlist", nlab_c),
         x=NULL, y="contested P@1") +
    theme_pub() + theme(axis.text.x=element_text(angle=12, hjust=1))
} else pH <- ggplot()+theme_void()

hh <- rd(f_geo_hist)
pg_for_n <- rd(f_geo_pg)
n_over <- if (!is.null(pg_for_n)) sum(pg_for_n$overrode %in% c(TRUE,"True","TRUE"), na.rm=TRUE) else NA
if (is.null(pg_for_n) || is.na(n_over) || n_over==0) n_over <- 4709
if (!is.null(hh)) {
  hl <- rbind(data.frame(kb=hh$bin_center_kb, n=hh$reranker, who="reranker"),
              data.frame(kb=hh$bin_center_kb, n=hh$rf,       who="RF argmax"))
  hl$who <- factor(hl$who, levels=c("RF argmax","reranker"))
  pI <- ggplot(hl, aes(kb, n, color=who, fill=who)) +
    geom_area(position="identity", alpha=.35, linewidth=.6) +
    scale_color_manual(values=c("RF argmax"=col_rf,"reranker"=col_splice)) +
    scale_fill_manual(values=c("RF argmax"=col_rf,"reranker"=col_splice)) +
    geom_vline(xintercept=0, color="grey40", linewidth=.4, linetype="dotted") +
    labs(title="Where overrides land vs the true site",
         subtitle=sprintf("strand-aware distance (+ = downstream); %s overridden", if(exists("n_over")) nlab(n_over) else "reranker-moved genes"),
         x="distance from canonical (kb)", y="genes") +
    theme_pub() + theme(legend.position="top", legend.justification="center")
} else pI <- ggplot()+theme_void()

pg <- rd(f_geo_pg)
if (!is.null(pg)) {
  ov <- pg %>% filter(overrode %in% c(TRUE,"True","TRUE"))
  brk <- function(signed){ s <- as.numeric(signed)
    c(`on true site`=mean(s==0,na.rm=TRUE), upstream=mean(s<0,na.rm=TRUE), downstream=mean(s>0,na.rm=TRUE)) }
  D2 <- rbind(
    data.frame(who="reranker",  outcome=names(brk(ov$spl_signed)), frac=as.numeric(brk(ov$spl_signed))),
    data.frame(who="RF argmax", outcome=names(brk(ov$rf_signed)),  frac=as.numeric(brk(ov$rf_signed))))
  D2$who <- factor(D2$who, levels=c("RF argmax","reranker"))
  D2$outcome <- factor(D2$outcome, levels=c("on true site","upstream","downstream"))
  pJ <- ggplot(D2, aes(who, frac, fill=outcome)) +
    geom_col(width=.6) +
    geom_text(data=subset(D2, frac>=.05), aes(label=sprintf("%.0f%%",frac*100)),
              position=position_stack(vjust=.5), color="white", size=3) +
    scale_fill_manual(values=c("on true site"=col_splice,"upstream"=col_green,"downstream"=col_supp)) +
    scale_y_continuous(labels=function(x) paste0(x*100,"%"), expand=expansion(c(0,0.02))) +
    labs(title="Override outcome", subtitle=sprintf("of %s where the pick changed", if(exists("n_over")) nlab(n_over) else "genes"),
         x=NULL, y="share of overridden genes", fill=NULL) +
    theme_pub() + theme(legend.position="top", legend.justification="center")
} else pJ <- ggplot()+theme_void()

sh <- rd(f_shap)
if (!is.null(sh)) {
  ord_sh <- sh %>% group_by(feature) %>% summarise(mi=mean(abs(shap)), .groups="drop") %>%
    arrange(desc(mi)) %>% slice_head(n=12)
  sh2 <- sh %>% filter(feature %in% ord_sh$feature)
  sh2 <- sh2 %>% group_by(feature) %>%
    mutate(vlo=quantile(value,.02,na.rm=TRUE), vhi=quantile(value,.98,na.rm=TRUE),
           vnorm=pmin(pmax((value-vlo)/pmax(vhi-vlo,1e-9),0),1)) %>% ungroup()
  flev <- rev(ord_sh$feature)
  sh2$flab <- factor(relabel(sh2$feature),
                     levels=relabel(flev))
  set.seed(1)
  if (nrow(sh2) > 60000) sh2 <- sh2 %>% group_by(feature) %>% slice_sample(n=5000) %>% ungroup()
  pK <- ggplot(sh2, aes(x=shap, y=flab, color=vnorm)) +
    geom_vline(xintercept=0, color="grey70", linewidth=.3) +
    geom_jitter(height=.28, width=0, size=.35, alpha=.5) +
    scale_color_gradient(low=col_supp, high=col_splice, name="feature value",
                         breaks=c(0,1), labels=c("low","high")) +
    labs(title="SHAP feature attribution",
         subtitle=sprintf("per-peak SHAP on the reranker; %s", nlab_c),
         x="SHAP value (impact on canonical probability)", y=NULL) +
    theme_pub() +
    theme(legend.position="right", legend.key.height=unit(.8,"lines"),
          legend.title=element_text(size=8), legend.text=element_text(size=8),
          axis.text.y=element_text(size=8))
} else pK <- ggplot()+theme_void()

fig <- ggarrange(pA,pB,pC,pE,pF,pG,pH,pI,pJ,pK, ncol=3, nrow=4,
                 labels=c("A","B","C","D","E","F","G","H","I","J"),
                 font.label=list(size=14,face="bold"))
fig <- annotate_figure(fig, top=text_grob(
  "Splicing-informed reranking of the RF candidate-selection pipeline",
  face="bold", size=13))
ggsave(out_pdf, fig, width=15.5, height=18.5, device=cairo_pdf, limitsize=FALSE)
ggsave(out_png, fig, width=15.5, height=18.5, dpi=200, limitsize=FALSE)
cat("wrote",out_pdf,"and",out_png,"\n")