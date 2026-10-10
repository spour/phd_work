#!/usr/bin/env Rscript
# figure 1.8 top: the input window each model reads, drawn to scale bottom

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(dplyr)
  library(tidyr)
})

args <- commandArgs(trailingOnly=TRUE)
STEM <- if(length(args)>=1) args[1] else "fig1_8_model_timeline"

INK <- "#141414"; MUT <- "#5A5A5A"; BASE <- 11

# up-to-date model data
M <- tribble(
  ~model,          ~year, ~win_lo, ~win_hi, ~arch,                    
  "polyadq",        1999,    -60,      60,  "Two PWMs",               
  "PolyA_SVM",      2006,   -100,     100,  "SVM, 15 cis-elements",   
  "POLYAR",         2010,   -100,     100,  "Linear discriminant",    
  "Dragon PolyA",   2012,   -200,     200,  "ANN/RF, physicochemical",
  "Omni-PolyA",     2017,   -100,     100,  "Ensemble, GA-tuned",     
  "DeepPolyA",      2018,   -100,     100,  "CNN",                    
  "DeeReCT-PolyA",  2018,   -103,     103,  "CNN",                    
  "DeepPASTA",      2019,   -100,     100,  "CNN + LSTM, +structure", 
  "APARENT",        2019,   -102,     102,  "CNN",                    
  "SANPolyA",       2020,   -103,     103,  "CNN + Self-Attention",   
  "Poly(A)-DG",     2020,   -103,     103,  "CNN",                    
  "DeepPASS",       2021,   -100,     100,  "CNN",                    
  "APARENT2",       2022,   -100,     100,  "Residual CNN"           
) |> 
  arrange(year) |> 
  mutate(
    model = factor(model, levels = model),
    kind = ifelse(grepl("CNN|LSTM", arch), "CNN",
           ifelse(grepl("ANN", arch), "Neural Net", "Classical")),
    span = win_hi - win_lo
  )

p1 <- ggplot(M, aes(y=model)) +
  annotate("rect", xmin=-2500, xmax=2500, ymin=-Inf, ymax=Inf, fill="#FDF3E7", colour=NA, alpha=0.5) +
  annotate("rect", xmin=-500, xmax=500, ymin=-Inf, ymax=Inf, fill="white", colour=NA) +
  geom_vline(xintercept=0, linetype=2, colour="#D4AF37", linewidth=0.6) +
  
  geom_segment(aes(x=win_lo, xend=win_hi, yend=model, colour=kind),
               linewidth=3, lineend="round") +
  geom_text(aes(x=win_hi, label=paste0("  ", span, " nt")),
            hjust=0, size=3, colour=MUT) +
  
  annotate("text", x=0, y=length(levels(M$model)) + 1, size=3.5, colour="#D4AF37",
           label="cleavage site", fontface="italic", vjust=0) +
  annotate("text", x=-2000, y=1, size=3, colour="#B07A3C",
           label="beyond every model's input") +
  
  scale_colour_manual(values=c("CNN"="#08519C", "Neural Net"="#4F9BBF", "Classical"="#9A9A9A")) +
  scale_x_continuous(limits=c(-2600, 1200),
                     breaks=c(-2000,-1000,-500,0,500,1000),
                     labels=c("-2000","-1000","-500","0","+500","+1000")) +
  scale_y_discrete(limits=rev(levels(M$model))) +
  labs(title="a   Sequence footprint read by each model",
       subtitle="Drawn to scale relative to the cleavage site. Models are blind to long-range determinants beyond their window.",
       x="position relative to the cleavage site (nt)", y=NULL) + 
  theme_classic(base_size=BASE) +
  theme(text=element_text(colour=INK),
        axis.text.y=element_text(colour=INK, face="bold", size=BASE-1),
        axis.line.y=element_blank(),
        axis.ticks.y=element_blank(),
        plot.title=element_text(face="bold", margin=margin(b=4)),
        plot.subtitle=element_text(colour=MUT, margin=margin(b=10)),
        legend.position="none",
        plot.margin=margin(10,15,20,15)) +
  coord_cartesian(clip="off")

positions <- c(0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 0.75, -0.75, 1.25, -1.25, 1.75, -1.75, 0.4)

M_timeline <- M |> 
  mutate(
    position = positions[1:n()],
    direction = sign(position),
    text_position = position + (0.15 * direction)
  )

p2 <- ggplot(M_timeline, aes(x=year, y=position, color=kind)) +
  geom_hline(yintercept=0, color="black", linewidth=0.5) +
  
  geom_segment(aes(xend=year, yend=0), color="grey40", linewidth=0.4) +
  
  geom_point(size=4) +
  
  geom_text(aes(y=text_position, label=model), 
            size=3.5, fontface="bold", show.legend=FALSE,
            vjust=ifelse(M_timeline$direction > 0, 0, 1)) +
  
  geom_text(aes(y=text_position + (0.12 * direction), label=arch), 
            size=2.8, color=MUT, show.legend=FALSE,
            vjust=ifelse(M_timeline$direction > 0, 0, 1)) +
  
  scale_color_manual(values=c("CNN"="#08519C", "Neural Net"="#4F9BBF", "Classical"="#9A9A9A"),
                     name="Architecture Category:") +
  scale_x_continuous(breaks=seq(1998, 2024, by=2), limits=c(1998, 2024)) +
  scale_y_continuous(limits=c(-2.2, 2.2)) +
  labs(title="b   Timeline of computational PAS prediction",
       x="Publication Year", y=NULL) +
  theme_classic(base_size=BASE) +
  theme(axis.line.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        axis.line.x=element_blank(),
        axis.text.x=element_text(size=BASE, face="bold", color=INK),
        plot.title=element_text(face="bold", margin=margin(b=15)),
        legend.position="bottom",
        legend.title=element_text(face="bold"),
        plot.margin=margin(20,15,10,15))

fig <- p1 / p2 + plot_layout(heights = c(1, 1.2))

ggsave(paste0(STEM,".pdf"), fig, width=11, height=10, device=cairo_pdf)
ggsave(paste0(STEM,".png"), fig, width=11, height=10, dpi=400, bg="white")

cat("[out] ", STEM, ".pdf / .png\n", sep="")