#!/usr/bin/env Rscript
# distance-resolved within-gene suppression

suppressPackageStartupMessages({ library(ggplot2); library(dplyr); library(tidyr)
  library(readr); library(patchwork) })

col_u1  <- "#C65B3C"; col_me5 <- "#3E6B8B"; col_null <- "#8A8F98"; ink <- "#222222"
theme_fig <- function(base=12) theme_minimal(base_size=base) +
  theme(plot.title=element_text(face="bold",size=base+1,color=ink),
        plot.subtitle=element_text(size=base-1,color="grey35",margin=margin(b=8)),
        axis.title=element_text(size=base-1,color=ink),
        axis.text=element_text(size=base-2,color="grey25"),
        panel.grid.minor=element_blank(),
        panel.grid.major=element_line(color="grey93",linewidth=.4),
        plot.margin=margin(12,14,10,12))

load_dist <- function(prefix,label){
  f <- paste0(prefix,"_paired_bydist.tsv")
  if(!file.exists(f)){ message("missing ",f); return(NULL) }
  read_tsv(f,show_col_types=FALSE) %>% mutate(metric=label)
}
u1  <- load_dist("fivess_profile_u1","U1 hybridization")
me5 <- load_dist("fivess_profile_me5","5\u2032 splice-site (MaxEntScan)")
if(is.null(u1)) stop("run the profiles with --paired (updated) first")
D <- bind_rows(u1,me5)

# panel a: per-shell % competitor>true end vs distance, real (with ci ribbon)
# plus the random-within-gene-pair null (grey) for reference
shellR <- D %>% filter(scheme=="shell", kind=="real") %>% mutate(mid=(d0+d1)/2)
shellN <- D %>% filter(scheme=="shell", kind=="random") %>% mutate(mid=(d0+d1)/2)

pA <- ggplot() +
  geom_line(data=shellN, aes(mid, frac_pos*100, group=metric),
            color=col_null, linewidth=.7, linetype="22") +
  annotate("text", x=max(shellR$mid)*0.98, y=mean(shellN$frac_pos*100),
           label="random within-gene pair", hjust=1, vjust=-0.6, color=col_null, size=3) +
  geom_ribbon(data=shellR, aes(mid, ymin=frac_lo*100, ymax=frac_hi*100, fill=metric),
              alpha=.18, color=NA) +
  geom_line(data=shellR, aes(mid, frac_pos*100, color=metric), linewidth=1.1) +
  geom_point(data=shellR, aes(mid, frac_pos*100, color=metric), size=2.3) +
  scale_color_manual(values=c("U1 hybridization"=col_u1,"5\u2032 splice-site (MaxEntScan)"=col_me5)) +
  scale_fill_manual(values=c("U1 hybridization"=col_u1,"5\u2032 splice-site (MaxEntScan)"=col_me5)) +
  labs(title="Suppression signal is proximal",
       subtitle="% of genes where a competitor carries more upstream donor load than the true end\n(shaded = 95% CI; grey = random within-gene pair)",
       x="upstream distance from cleavage site (nt)",
       y="% genes  competitor > true end", color=NULL, fill=NULL) +
  theme_fig() + theme(legend.position=c(.98,.98), legend.justification=c(1,1),
                      legend.background=element_rect(fill="white",color=NA),
                      legend.text=element_text(size=9))

# panel b: cumulative window median (comp-canon), real vs null, with bootstrap-ish
# show real vs null lines, null near 0)
cumR <- D %>% filter(scheme=="cumulative", kind=="real")
cumN <- D %>% filter(scheme=="cumulative", kind=="random")
pB <- ggplot() +
  geom_hline(yintercept=0, color="grey60", linewidth=.5) +
  geom_line(data=cumN, aes(d1, median, group=metric), color=col_null, linewidth=.7, linetype="22") +
  geom_ribbon(data=cumR, aes(d1, ymin=med_lo, ymax=med_hi, fill=metric), alpha=.18, color=NA) +
  geom_line(data=cumR, aes(d1, median, color=metric), linewidth=1.1) +
  geom_point(data=cumR, aes(d1, median, color=metric), size=2.5) +
  annotate("text", x=max(cumR$d1)*0.98, y=0, label="random pair \u2248 0",
           hjust=1, vjust=-0.6, color=col_null, size=3) +
  scale_color_manual(values=c("U1 hybridization"=col_u1,"5\u2032 splice-site (MaxEntScan)"=col_me5)) +
  scale_fill_manual(values=c("U1 hybridization"=col_u1,"5\u2032 splice-site (MaxEntScan)"=col_me5)) +
  labs(title="Widening the window dilutes the effect",
       subtitle="cumulative 0\u2013X window: median within-gene (competitor \u2212 true end) load (shaded = 95% bootstrap CI)",
       x="upstream window width (0 to X nt)",
       y="median (competitor \u2212 true end)\ntotal load / 100nt", color=NULL) +
  theme_fig() + theme(legend.position="none")

fig <- (pA | pB) +
  plot_layout(widths=c(1.15,1)) +
  plot_annotation(
    tag_levels="A",
    title="Within-gene, competitors carry more upstream splice-donor signal than true 3\u2032 ends \u2014 strongest proximally",
    caption="Within-gene paired comparison (gene-position confound cancelled). Real curves rise above the random-within-gene-pair null near the site and decay toward it with distance: the telescripting-suppression signature.",
    theme=theme(
      plot.title=element_text(face="bold", size=14, color=ink, hjust=0.5,
                              margin=margin(b=6)),
      plot.caption=element_text(size=8.5, color="grey40", hjust=0)))  &
  theme(plot.tag=element_text(size=15, face="bold", color=ink))

ggsave("paired_suppression.pdf", fig, width=13, height=5.4, device=cairo_pdf)
ggsave("paired_suppression.png", fig, width=13, height=5.4, dpi=220)
cat("wrote paired_suppression.pdf / .png\n")