#!/usr/bin/env Rscript
# corroboration of canonical vs competitor peaks against polyasite 2.0 and polya_db v4
suppressWarnings(suppressMessages({library(ggplot2); library(patchwork)}))
INK<-"#1A1A1A"; CAN<-"#0072B2"; COMP<-"#E69F00"; BASE<-13
th<-theme_classic(base_size=BASE)+theme(
  plot.title=element_text(face="bold",colour=INK),
  plot.subtitle=element_text(colour=INK),
  axis.text=element_text(colour=INK), axis.title=element_text(colour=INK),
  strip.background=element_blank(), strip.text=element_text(face="bold",colour=INK),
  legend.position="top", legend.title=element_blank())
cols<-c(canonical=CAN, competitor=COMP)

ov<-read.delim("polya_corroboration_overlap.tsv")
ov$class<-factor(ov$class, levels=c("canonical","competitor"))
pA<-ggplot(ov, aes(full_window_nt, frac_matched, colour=class))+
  geom_line(linewidth=0.9)+geom_point(size=2.2)+
  facet_wrap(~db)+scale_colour_manual(values=cols)+
  scale_y_continuous(limits=c(0,1))+
  labs(title="Overlap with annotated poly(A) sites",
       subtitle="Fraction of peaks with a same-strand database site inside the window",
       x="window width (nt)", y="fraction matched")+th

st<-read.delim("polya_corroboration_strength.tsv")
st$class<-factor(st$class, levels=c("canonical","competitor"))
st$panel<-paste0(st$db, ": ", st$metric)
pB<-ggplot(st, aes(class, value, fill=class))+
  geom_violin(colour=NA, alpha=0.55, scale="width")+
  geom_boxplot(width=0.16, outlier.shape=NA, colour=INK, fill="white", linewidth=0.4)+
  facet_wrap(~panel, scales="free_y", ncol=4)+
  scale_fill_manual(values=cols)+
  labs(title="Strength of the matched poly(A) site",
       subtitle="Matched inside the 500-nt window (+/-250 nt)",
       x=NULL, y="value")+th+theme(axis.text.x=element_text(angle=20,hjust=1))

fig<-pA/pB+plot_layout(heights=c(1,1.35))+plot_annotation(theme=theme(plot.background=element_rect(fill="white",colour=NA)))
ggsave("polya_corroboration.pdf", fig, width=11, height=9, device=cairo_pdf)
ggsave("polya_corroboration.png", fig, width=11, height=9, dpi=150, bg="white")
cat("wrote polya_corroboration.pdf, polya_corroboration.png\n")
