#!/usr/bin/env Rscript
# endogenous retroviral elements among cryptic poly(a) sites

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
  library(readr); library(scales); library(forcats); library(ggrepel)
})
args <- commandArgs(trailingOnly=TRUE)
DIR  <- if(length(args)>=1) args[1] else "tenull"
STEM <- if(length(args)>=2) args[2] else "figure_te"

M <- read_tsv(file.path(DIR,"sites.tsv"), show_col_types=FALSE, progress=FALSE)
F <- read_tsv(file.path(DIR,"families.tsv"), show_col_types=FALSE,
              progress=FALSE)

# style
INK <- "#1A1917"; MUT <- "#7A736A"; PAPER <- "#FBFAF7"
UP   <- "#8C2F1B"
DOWN <- "#2E5E6E"
FLAT <- "#B9B2A6"
SENSE <- "#8C2F1B"; ANTI <- "#C4BCAF"
BASE <- 13

th <- theme_minimal(base_size=BASE)+
  theme(text=element_text(colour=INK),
        plot.background=element_rect(fill=PAPER, colour=NA),
        panel.background=element_rect(fill=PAPER, colour=NA),
        panel.grid.minor=element_blank(),
        panel.grid.major=element_line(linewidth=0.22, colour="#E8E3DA"),
        axis.text=element_text(colour=INK, size=BASE-4),
        axis.title=element_text(size=BASE-3, colour=MUT),
        axis.ticks=element_blank(),
        plot.title=element_text(size=BASE+1, face="bold", hjust=0,
                                margin=margin(b=3)),
        plot.title.position="plot",
        plot.subtitle=element_text(size=BASE-4, colour=MUT,
                                   margin=margin(b=9), lineheight=1.3),
        legend.position="none",
        strip.text=element_text(face="bold", size=BASE-4, colour=INK),
        plot.margin=margin(10,14,10,12))

fmtp <- function(p){
  ifelse(p < 1e-300, "P < 10⁻³⁰⁰",
  ifelse(p < 0.001, sprintf("P = %.0e", p), sprintf("P = %.3f", p)))
}
sig <- function(p) ifelse(p<1e-4,"****",ifelse(p<1e-3,"***",
                   ifelse(p<0.01,"**",ifelse(p<0.05,"*","ns"))))

C  <- M |> filter(group=="competing site")
B  <- M |> filter(group=="matched random")
te_c <- mean(C$is_te); te_b <- mean(B$is_te)
ct <- matrix(c(sum(C$is_te==1), sum(C$is_te==0),
               sum(B$is_te==1), sum(B$is_te==0)), nrow=2, byrow=TRUE)
ft <- fisher.test(ct)

# a  volcano: every family at once
# effect on one axis, evidence on the other, point area for how many competing
# sites the family carries
ga <- F |>
  filter(n_competing + n_null >= 100, is.finite(ratio), ratio > 0) |>
  mutate(l2 = log2(ratio),
         negl10 = -log10(pmax(p, 1e-300)),
         dir = case_when(q < 0.05 & l2 >  log2(1.25) ~ "enriched",
                         q < 0.05 & l2 < -log2(1.25) ~ "depleted",
                         TRUE ~ "no difference"),
         lab = ifelse(q < 0.05 & abs(l2) > log2(1.25), family, NA))

pa <- ggplot(ga, aes(l2, negl10))+
  annotate("rect", xmin=-log2(1.25), xmax=log2(1.25), ymin=-Inf, ymax=Inf,
           fill="#EFEAE0", colour=NA)+
  geom_hline(yintercept=-log10(0.05), linetype=3, linewidth=0.4,
             colour=MUT)+
  geom_vline(xintercept=0, linewidth=0.4, colour=INK)+
  geom_point(aes(size=n_competing, fill=dir), shape=21, colour="white",
             stroke=0.5, alpha=0.95)+
  geom_text_repel(aes(label=lab, colour=dir), size=3.1, fontface="bold",
                  seed=1, box.padding=0.45, max.overlaps=20,
                  segment.colour=MUT, segment.size=0.25, na.rm=TRUE)+
  scale_fill_manual(values=c(enriched=UP, depleted=DOWN,
                             `no difference`=FLAT))+
  scale_colour_manual(values=c(enriched=UP, depleted=DOWN,
                               `no difference`=FLAT))+
  scale_size_area(max_size=11)+
  scale_x_continuous(breaks=log2(c(0.1,0.25,0.5,1,2,4)),
                     labels=c("0.1\u00d7","0.25\u00d7","0.5\u00d7","1\u00d7",
                              "2\u00d7","4\u00d7"))+
  scale_y_continuous(expand=expansion(mult=c(0.02,0.10)))+
  labs(title="a   Every element family, against matched random positions",
       subtitle=sprintf(paste0("competing sites lie in element sequence less ",
                               "often than matched positions do\n(%.1f%% ",
                               "against %.1f%%, odds ratio %.2f, %s). the ",
                               "families move in\nopposite directions. point ",
                               "area is competing sites carried"),
                        100*te_c, 100*te_b, ft$estimate, fmtp(ft$p.value)),
       x="competing sites relative to matched random positions",
       y=expression(-log[10]~italic(P)))+th

gb <- F |>
  filter(n_competing >= 100, is.finite(ratio), is.finite(sense_ratio)) |>
  mutate(anti_ratio = (pct_competing - pct_competing_sense) /
                      pmax(pct_null - pct_null_sense, 1e-9)) |>
  filter(is.finite(anti_ratio), anti_ratio > 0) |>
  mutate(erv = grepl("^ERV", family),
         gap = sense_ratio / pmax(anti_ratio, 1e-9))

pb <- ggplot(gb, aes(x=anti_ratio, xend=sense_ratio,
                     y=fct_reorder(family, sense_ratio)))+
  geom_vline(xintercept=1, linewidth=0.4, colour=INK)+
  geom_segment(aes(yend=fct_reorder(family, sense_ratio),
                   colour=erv), linewidth=1.5, alpha=0.35,
               lineend="round")+
  geom_point(aes(x=anti_ratio), size=3, colour=ANTI)+
  geom_point(aes(x=sense_ratio, colour=erv), size=3.6)+
  geom_text(aes(x=pmax(sense_ratio, anti_ratio),
                label=sprintf("%.1f\u00d7", sense_ratio), colour=erv),
            hjust=-0.35, size=3.0, fontface="bold")+
  scale_colour_manual(values=c(`TRUE`=UP, `FALSE`="#9A948A"))+
  scale_x_log10(breaks=c(0.25,0.5,1,2,4),
                labels=c("0.25\u00d7","0.5\u00d7","1\u00d7","2\u00d7","4\u00d7"),
                expand=expansion(mult=c(0.05,0.18)))+
  labs(title="b   Split by the element's orientation in the transcript",
       subtitle=paste0("pale point is the element pointing against the ",
                       "transcript, solid point with it.\nthe two strands of ",
                       "an element have nearly the same base composition, so ",
                       "a\nseparation here is architecture. retroviral ",
                       "families are marked"),
       x="competing sites relative to matched random positions", y=NULL)+th

ervs <- c("ERV1","ERVK","ERVL","ERVL-MaLR")
ervs <- ervs[ervs %in% F$family]

mk <- function(fam){
  cs <- sum(C$fam==fam & C$sense==1); ca <- sum(C$fam==fam & C$sense==0)
  bs <- sum(B$fam==fam & B$sense==1); ba <- sum(B$fam==fam & B$sense==0)
  ps <- fisher.test(matrix(c(cs, nrow(C)-cs, bs, nrow(B)-bs), 2, byrow=TRUE))
  pan <- fisher.test(matrix(c(ca, nrow(C)-ca, ba, nrow(B)-ba), 2, byrow=TRUE))
  tibble(family=fam,
         orientation=rep(c("sense","antisense"), each=2),
         set=rep(c("competing site","matched random"), 2),
         pct=c(100*cs/nrow(C), 100*bs/nrow(B),
               100*ca/nrow(C), 100*ba/nrow(B)),
         p=c(ps$p.value, ps$p.value, pan$p.value, pan$p.value))
}
gc_ <- bind_rows(lapply(ervs, mk)) |>
  mutate(orientation=factor(orientation, c("sense","antisense")),
         set=factor(set, c("competing site","matched random")),
         family=factor(family, ervs))

bar <- gc_ |> group_by(family, orientation) |>
  summarise(top=max(pct), p=first(p), .groups="drop") |>
  mutate(lab=paste0(sig(p), "  ", fmtp(p)))

pc <- ggplot(gc_, aes(orientation, pct, fill=set))+
  geom_col(position=position_dodge(width=0.72), width=0.62)+
  geom_segment(data=bar, aes(x=as.numeric(orientation)-0.18,
                             xend=as.numeric(orientation)+0.18,
                             y=top*1.13, yend=top*1.13),
               inherit.aes=FALSE, linewidth=0.35, colour=INK)+
  geom_text(data=bar, aes(x=orientation, y=top*1.18, label=lab),
            inherit.aes=FALSE, size=2.7, colour=INK, vjust=0)+
  facet_wrap(~family, nrow=1, scales="free_y")+
  scale_fill_manual(values=c(`competing site`=SENSE,
                             `matched random`=ANTI))+
  scale_y_continuous(labels=function(x) paste0(x,"%"),
                     expand=expansion(mult=c(0,0.30)))+
  labs(title="c   Retroviral families, in the percentages the ratios are built from",
       subtitle=paste0("dark bar is competing sites, pale bar matched random ",
                       "positions. every family\nseparates in sense ",
                       "orientation and none does against it. Fisher exact, ",
                       "two-sided"),
       x=NULL, y="% of positions")+th+
  theme(legend.position="bottom", legend.title=element_blank(),
        legend.text=element_text(size=BASE-4),
        legend.margin=margin(t=-6), legend.key.size=unit(0.8,"lines"))

fig <- (pa | pb) / pc +
  plot_layout(heights=c(1.25, 1)) &
  theme(plot.background=element_rect(fill=PAPER, colour=NA))

ggsave(paste0(STEM,".pdf"), fig, width=15, height=11.5, device=cairo_pdf,
       bg=PAPER, limitsize=FALSE)
ggsave(paste0(STEM,".png"), fig, width=15, height=11.5, dpi=450, bg=PAPER,
       limitsize=FALSE)
cat("[out] ", STEM, ".pdf / .png\n", sep="")

cat(sprintf("\nany TE: competing %.1f%%, matched random %.1f%%, OR %.3f, P = %.3g\n",
            100*te_c, 100*te_b, ft$estimate, ft$p.value))
cat("\nfamilies, sense orientation:\n")
gb |> arrange(desc(sense_ratio)) |>
  select(family, pct_competing_sense, pct_null_sense, sense_ratio,
         anti_ratio, q) |>
  as.data.frame() |> print(row.names=FALSE, digits=3)