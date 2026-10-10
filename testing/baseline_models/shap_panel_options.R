#!/usr/bin/env Rscript
# render four styles of the shap panel from the same data, side by side

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(readr); library(tidyr); library(ggpubr)
})
have_ridges <- requireNamespace("ggridges", quietly=TRUE)
have_quasi  <- requireNamespace("ggbeeswarm", quietly=TRUE)

f_shap <- "shap_beeswarm_long.tsv"
col_splice <- "#C65B3C"; col_supp <- "#3E6B8B"; col_rf <- "#9AA0A6"; col_green <- "#5B7C6F"

pretty_name <- c(
  spl_upstream_max_me3="strongest upstream 3\u2032 splice site",
  spl_u1_suppression_up="U1 / donor density (upstream)",
  spl_pas_prox_u1="U1 near the polyA signal",
  spl_comp_dn_3ss="competing 3\u2032 splice site (downstream)",
  spl_comp_up_3ss="competing 3\u2032 splice site (upstream)",
  spl_interv_me5_max="strongest intervening 5\u2032 splice site",
  spl_term_3ss="terminal 3\u2032 splice site strength",
  spl_term_ppt="terminal polypyrimidine tract",
  spl_u2af35_ag="U2AF35 (3\u2032 AG) strength",
  spl_dist_to_max_3ss="distance to strongest 3\u2032 splice site",
  spl_sum_5ss="total donor strength nearby",
  spl_mean_top3_5ss="mean of top-3 donors nearby",
  spl_upstream_mean_u1="mean upstream U1 density",
  spl_dist_nearest_5ss="distance to nearest strong donor")
relab <- function(f) ifelse(f %in% names(pretty_name), pretty_name[f], gsub("^spl_","",f))

theme_pub <- function(base=11) theme_pubr(base_size=base) +
  theme(plot.title=element_text(face="bold", size=base+1),
        plot.subtitle=element_text(size=base-1, color="grey30"),
        axis.text.y=element_text(size=8))

sh <- read_tsv(f_shap, show_col_types=FALSE)
ord <- sh %>% group_by(feature) %>% summarise(mi=mean(abs(shap)), .groups="drop") %>%
  arrange(desc(mi)) %>% slice_head(n=12)
sh <- sh %>% filter(feature %in% ord$feature) %>%
  group_by(feature) %>%
  mutate(vlo=quantile(value,.02,na.rm=TRUE), vhi=quantile(value,.98,na.rm=TRUE),
         vnorm=pmin(pmax((value-vlo)/pmax(vhi-vlo,1e-9),0),1)) %>% ungroup()
flev <- rev(ord$feature)
sh$flab <- factor(relab(sh$feature), levels=relab(flev))

# option 1: ridgeline (density colored by mean feature value)
if (have_ridges) {
  library(ggridges)
  ridge_fill <- sh %>% group_by(flab) %>% summarise(mv=mean(vnorm), .groups="drop")
  shR <- sh %>% left_join(ridge_fill, by="flab")
  p1 <- ggplot(shR, aes(x=shap, y=flab, fill=mv)) +
    geom_density_ridges(scale=2.2, rel_min_height=.01, color="white", linewidth=.2, alpha=.9) +
    geom_vline(xintercept=0, color="grey60", linewidth=.3) +
    scale_fill_gradient(low=col_supp, high=col_splice, name="mean\nfeature value",
                        breaks=c(0,1), labels=c("low","high")) +
    labs(title="Option 1: ridgeline", subtitle="SHAP density per feature",
         x="SHAP value", y=NULL) + theme_pub()
} else {
  p1 <- ggplot()+annotate("text",x=0,y=0,label="install ggridges for Option 1")+theme_void()
}

# option 2: thinned beeswarm (jitter, small alpha)
set.seed(1)
sh_thin <- sh %>% group_by(feature) %>%
  group_modify(~ slice_sample(.x, n = min(1500L, nrow(.x)))) %>% ungroup()
sh_thin$flab <- factor(relab(sh_thin$feature), levels=relab(flev))
p2 <- ggplot(sh_thin, aes(x=shap, y=flab, color=vnorm)) +
  geom_vline(xintercept=0, color="grey70", linewidth=.3) +
  geom_jitter(height=.18, width=0, size=.25, alpha=.35) +
  scale_color_gradient(low=col_supp, high=col_splice, name="feature value",
                       breaks=c(0,1), labels=c("low","high")) +
  labs(title="Option 2: thinned jitter", subtitle="1,500 pts/feature, small/alpha",
       x="SHAP value", y=NULL) + theme_pub()

# option 3: quasirandom beeswarm (proper density positioning)
if (have_quasi) {
  library(ggbeeswarm)
  set.seed(1)
  sh_q <- sh %>% group_by(feature) %>%
    group_modify(~ slice_sample(.x, n = min(2000L, nrow(.x)))) %>% ungroup()
  sh_q$flab <- factor(relab(sh_q$feature), levels=relab(flev))
  p3 <- ggplot(sh_q, aes(x=shap, y=flab, color=vnorm)) +
    geom_vline(xintercept=0, color="grey70", linewidth=.3) +
    geom_quasirandom(groupOnX=FALSE, size=.3, alpha=.5, width=.32) +
    scale_color_gradient(low=col_supp, high=col_splice, name="feature value",
                         breaks=c(0,1), labels=c("low","high")) +
    labs(title="Option 3: quasirandom beeswarm", subtitle="density-positioned (real SHAP look)",
         x="SHAP value", y=NULL) + theme_pub()
} else {
  p3 <- ggplot()+annotate("text",x=0,y=0,label="install ggbeeswarm for Option 3")+theme_void()
}

# option 4: dumbbell (low-value mean shap vs high-value mean shap)
db <- sh %>% group_by(feature, flab) %>%
  mutate(tert=ntile(value,3)) %>%
  filter(tert %in% c(1,3)) %>%
  group_by(flab, tert) %>% summarise(m=mean(shap), .groups="drop") %>%
  mutate(grp=ifelse(tert==1,"low value","high value"))
db_w <- db %>% select(flab,grp,m) %>% pivot_wider(names_from=grp, values_from=m)
p4 <- ggplot(db_w, aes(y=flab)) +
  geom_vline(xintercept=0, color="grey70", linewidth=.3) +
  geom_segment(aes(x=`low value`, xend=`high value`, yend=flab), color="grey60", linewidth=.6) +
  geom_point(aes(x=`low value`), color=col_supp, size=2.4) +
  geom_point(aes(x=`high value`), color=col_splice, size=2.4) +
  labs(title="Option 4: low\u2192high dumbbell",
       subtitle="mean SHAP at low (blue) vs high (orange) feature value",
       x="mean SHAP value", y=NULL) + theme_pub()

fig <- ggarrange(p1,p2,p3,p4, ncol=2, nrow=2, labels=c("1","2","3","4"))
ggsave("shap_options.png", fig, width=15, height=11, dpi=200)
ggsave("shap_options.pdf", fig, width=15, height=11, device=cairo_pdf)
cat("wrote shap_options.png / .pdf\n")
cat("ggridges available:", have_ridges, " | ggbeeswarm available:", have_quasi, "\n")
if (!have_ridges) cat("  -> install.packages('ggridges') for Option 1\n")
if (!have_quasi)  cat("  -> install.packages('ggbeeswarm') for Option 3\n")
