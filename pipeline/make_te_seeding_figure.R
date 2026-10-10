#!/usr/bin/env Rscript
# how specific retrotransposon families seed cryptic poly(a) architecture

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
  library(readr); library(scales); library(forcats)
})
args  <- commandArgs(trailingOnly=TRUE)
INDIR <- if(length(args)>=1) args[1] else "seed"
STEM  <- if(length(args)>=2) args[2] else "figure_te_seeding"
rd <- function(f, req=TRUE){
  p <- file.path(INDIR,f)
  if(!file.exists(p)){ if(req) stop("missing ",p); return(NULL) }
  read_tsv(p, show_col_types=FALSE, progress=FALSE)
}

INK <- "#141414"; MUT <- "#5A5A5A"; BASE <- 13
SENSE <- "#08519C"; ANTI <- "#C6C6C6"
BASECOL <- c(A="#3B9E4A", C="#2C6FBB", G="#E8A33D", T="#C4342B")

th <- theme_classic(base_size=BASE)+
  theme(text=element_text(colour=INK),
        axis.text=element_text(colour=INK, size=BASE-4),
        axis.title=element_text(size=BASE-2),
        axis.line=element_line(colour=INK, linewidth=0.45),
        axis.ticks=element_line(colour=INK, linewidth=0.45),
        plot.title=element_text(size=BASE, face="bold", hjust=0,
                                margin=margin(b=3)),
        plot.title.position="plot",
        plot.subtitle=element_text(size=BASE-4, colour=MUT,
                                   margin=margin(b=5), lineheight=1.15),
        legend.position="bottom", legend.title=element_blank(),
        legend.text=element_text(size=BASE-4),
        legend.margin=margin(t=-4), legend.key.size=unit(0.75,"lines"),
        strip.background=element_blank(),
        strip.text=element_text(face="bold", size=BASE-4, margin=margin(b=2)),
        panel.grid.major=element_line(linewidth=0.2, colour="grey94"),
        plot.margin=margin(5,9,5,7))

SM <- rd("family_summary.tsv")
DN <- rd("hexamer_density.tsv")
CS <- rd("consensus.tsv")
PKS <- rd("density_peaks.tsv")
AG <- rd("age.tsv", req=FALSE)

# a  rate and asymmetry
ga <- SM |>
  select(family, orientation, arch_per_kb) |>
  pivot_wider(names_from=orientation, values_from=arch_per_kb) |>
  mutate(ratio = sense / pmax(antisense, 1e-9)) |>
  arrange(desc(ratio)) |>
  mutate(family = fct_inorder(family))
gal <- SM |> mutate(family = factor(family, levels=levels(ga$family)),
                    orientation = factor(orientation,
                                         levels=c("sense","antisense")))
pa <- ggplot(gal, aes(arch_per_kb, fct_rev(family), fill=orientation))+
  geom_col(position=position_dodge(width=0.75), width=0.68)+
  geom_text(data=ga, aes(x=sense, y=fct_rev(family),
                         label=sprintf("%.1f\u00d7", ratio)),
            inherit.aes=FALSE, hjust=-0.25, size=3.0, fontface="bold",
            colour=INK)+
  scale_fill_manual(values=c(sense=SENSE, antisense=ANTI))+
  scale_x_continuous(expand=expansion(mult=c(0, 0.18)))+
  labs(title="a   Complete architectures per kilobase of element",
       subtitle=paste0("labels give sense over antisense. base composition is ",
                       "nearly strand-symmetric,\nso the asymmetry is ",
                       "structural"),
       x="architectures per kb", y=NULL)+th

# b  positional density along the consensus
# smoothed lightly, since single consensus positions are noisy at these copy
# numbers, and the question is where the density concentrates rather than the
gb <- DN |> filter(orientation == "sense") |>
  group_by(family) |> arrange(cons_pos) |>
  mutate(sm = stats::filter(per_1000, rep(1/9, 9), sides=2) |> as.numeric()) |>
  ungroup() |> filter(!is.na(sm))
ann <- PKS |> group_by(family) |> slice_max(per_1000, n=2) |> ungroup()
pb <- ggplot(gb, aes(cons_pos, sm))+
  geom_area(fill=SENSE, alpha=0.30)+
  geom_line(colour=SENSE, linewidth=0.6)+
  geom_point(data=ann, aes(cons_pos, per_1000), colour="#B2182B", size=1.6,
             inherit.aes=FALSE)+
  facet_wrap(~family, scales="free", ncol=3)+
  labs(title="b   Where in the element the hexamers arise",
       subtitle=paste0("hexamers per 1,000 aligned copies, along the family ",
                       "consensus. red marks the\ntwo strongest positions per ",
                       "family"),
       x="position in the element consensus (nt)",
       y="hexamers per 1,000 copies")+th

gc_ <- PKS |>
  filter(!is.na(consensus_hexamer), nchar(consensus_hexamer) == 6) |>
  group_by(family) |> slice_max(per_1000, n=2) |> ungroup() |>
  mutate(lab = paste0(family, "  @", cons_pos)) |>
  arrange(substitutions_from_PAS, desc(per_1000)) |>
  mutate(lab = fct_inorder(lab))
gc_long <- gc_ |>
  mutate(ctx = consensus_context) |>
  separate_rows(ctx, sep="") |> filter(ctx != "") |>
  group_by(lab) |> mutate(i = row_number() - 4) |> ungroup()
pc <- ggplot(gc_long, aes(i, fct_rev(lab), label=ctx, fill=ctx))+
  annotate("rect", xmin=-0.5, xmax=5.5, ymin=-Inf, ymax=Inf,
           fill="#FFF4CE", colour=NA)+
  geom_tile(colour="white", linewidth=0.6, alpha=0.85)+
  geom_text(size=3.0, fontface="bold", colour="white")+
  geom_text(data=gc_, aes(x=9.2, y=fct_rev(lab),
                          label=sprintf("%d sub", substitutions_from_PAS)),
            inherit.aes=FALSE, hjust=0, size=2.9, colour=MUT)+
  scale_fill_manual(values=BASECOL, guide="none")+
  scale_x_continuous(breaks=c(-3,0,5,8), labels=c("-3","1","6","9"),
                     limits=c(-4, 12))+
  labs(title="c   The element consensus where those hexamers arise",
       subtitle=paste0("shaded box marks the six positions; the label gives ",
                       "how many substitutions\nseparate the consensus from ",
                       "AATAAA or ATTAAA"),
       x="position relative to the hexamer", y=NULL)+th

pd_ <- NULL
if(!is.null(AG) && nrow(AG)){
  gd <- AG |> mutate(family = factor(family, levels=levels(ga$family)))
  pd_ <- ggplot(gd, aes(median_div, arch_per_kb, colour=family,
                        group=family))+
    geom_line(linewidth=0.9)+
    geom_point(size=2.2)+
    scale_colour_brewer(palette="Dark2")+
    labs(title="d   Older copies seed no less than younger ones",
         subtitle=paste0("copies split into quartiles of divergence from the ",
                         "family consensus,\nthe standard proxy for insertion ",
                         "age"),
         x="divergence from consensus (%)",
         y="architectures per kb")+th+
    theme(legend.position="right", legend.text=element_text(size=BASE-5))
}

row1 <- (pa | pc) + plot_layout(widths=c(1, 1.1))
row2 <- pb
parts <- Filter(Negate(is.null), list(row1, row2, pd_))
hts <- c(1, 1.35, if(!is.null(pd_)) 0.85)
fig <- Reduce(`/`, parts) + plot_layout(heights=hts)

HH <- 5.2 * length(parts) + 1
ggsave(paste0(STEM,".pdf"), fig, width=14, height=HH, device=cairo_pdf,
       limitsize=FALSE)
ggsave(paste0(STEM,".png"), fig, width=14, height=HH, dpi=400, limitsize=FALSE)
cat("[out] ", STEM, ".pdf / .png\n", sep="")

cat("\nfamily summary:\n")
SM |> as.data.frame() |> print(row.names=FALSE)
cat("\ndensity peaks and their consensus:\n")
PKS |> as.data.frame() |> print(row.names=FALSE)