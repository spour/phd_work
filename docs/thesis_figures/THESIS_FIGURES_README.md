# Thesis figures: what each shows, how it was made, how to remake it

Covers Chapters 2 and 3 (Chapter 1 excluded; Chapter 4 has no figures).
Text sources: `thesis_ch2/chapter2/chapter2.md`, `thesis_ch3/chapter3/chapter3.md`.
Each entry gives the figure's question, its data, its method, the files and commands
needed to rebuild it, and its key numbers. Unresolved items are marked **OPEN**.

## 0. Conventions and shared inputs

**Paths**
- `PUB` = `/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub`
- `CLEAN` = `.../reanal/clean` (parent of `PUB`)
- `T` = `PUB/thesis_ch3`

**Software**
- R: use the full path `/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/r/4.5.0/bin/Rscript`, because `module load` silently does nothing in a batch shell.
- R packages: ggplot2 4.0.3, patchwork, cairo_pdf.
- Python: `source ~/splice_env2/bin/activate` (maxentpy 0.0.3, PyTorch, pysam).
  - Chapter 2 models used Python 3.10, PyTorch 2.6.0, scikit-learn 1.2.1.
  - Chapter 3 analyses deliberately use no sklearn and no scipy; AUROC, logistic/IRLS and Wilcoxon are written inline.

**Genome and annotation**
- hg38 (UCSC): `CLEAN/train/data/hg38.fa`
- GENCODE v49: `CLEAN/train/data/annotation/gencode.v49.annotation.gtf`

**Poly(A) site databases**
- PolyA_DB: `CLEAN/train/data/polyadb_v4/HumanPas/hg38.PAS.main.tsv`
  - **OPEN:** the text and the selection script say v3; the directory, figure title and table columns (ENCODE.PB, GTEx.ONT) say v4.
- PolyASite 2.0 (GRCh38, Ensembl 96): `CLEAN/train/data/atlas.clusters.2.0.GRCh38.96.tsv`, 569,005 clusters, also as `atlas.chr.bed`. Source: polyasite.unibas.ch.

**Central candidate table: `PUB/cpa_peaks_final.parquet`**
- 70,723 candidates in 6,583 contested genes: one constitutive (real) 3′ end per gene plus 64,140 competitors.
- Flags: `is_canonical`, `flag_apa_annotated`, `flag_reference_terminal_exon`, `flag_downstream_of_canonical`, `in_analysis_set`.
- Documented in `PUB/cpa_peaks_final.README.md`; built by `PUB/task_build_final_parquet.py`.

**Gene sets. Always state the denominator.**

| Set | Genes | Candidates | Definition |
|---|---|---|---|
| Scanned genome-wide | 15,560 | – | Genes with a constitutive site that were scanned (Ch2 also quotes 15,568 and 15,559; **OPEN**) |
| Uncontested | 8,977 | – | No window scored at or above the constitutive site by all three Chapter 2 models |
| Contested (full shortlist) | 6,583 | 70,723 | At least one competitor |
| **Analysis set** | **4,847** | **48,394** (4,847 real + 43,547 competitors) | Competitors after the Figure 2.4 filters |

**Per-parameter dossier for Chapter 2:** `CLEAN/METHODS_DIGEST.md` (1,681 lines, every parameter with `path:line`). Its internal figure numbers map as 0.3 → 2.1, 0.1/0.2 → 2.2, 0.4 → 2.3, 0.5 → 2.5, 0.6/0.7 → 2.6/2.7.

**Not in git.** Nothing under `reanal/` is tracked. `figure1.png` and `figure2.png` differ from git HEAD, so the images in the docx may be from an older render.
## How this document was verified, and how to rebuild it

Every figure below is shown in its Nature-style render, followed by the code that makes
it, copied from disk at build time rather than retyped.

**Images are the Nature-style renders; the thesis originals are kept.** The 19 coded
figures were re-rendered by `PUB/nature_figs/make_nature_figs.R` (reproduced in the
appendix). It runs each figure's own script unchanged, with `ggsave` and `write.table`
switched off, takes the finished plot object and changes presentation only: bold
lowercase panel letters, no descriptive titles, subtitles or captions, 7 pt Helvetica
(6 pt ticks), no gridlines, plain facet labels, 0.25 mm axes, 183 mm wide. Data, panels and in-panel values are the same
as in the originals. Four figures also get a small layout fix, listed in the script's
`FIX` table: 2.1 drops its "retained" footnote, 2.9 drops the site counts from its
facet strips, 3.4 wraps its two bottom strips and 3.12 angles its x labels. The
originals, as they appear in the thesis docx, are in `figures_original/`; full-size PDFs
and PNGs are in `PUB/nature_figs/`. Panel letters in this document are lowercase to match.
**The per-panel n that the old subtitles printed now belongs in each figure legend.**

**The code was checked against the figure, not just cited.** (This check was done on the
original renders in `figures_original/`.) Each coded figure was
re-rendered from its current script into a sandbox that redirects every write, with the
real outputs made read-only so a stray write would fail rather than overwrite a thesis
figure. The re-render was then compared pixel by pixel with the image in the thesis:

- **18 of 19 coded figures are pixel-identical.** For these the script in this document
  is provably the script that made the figure.
- **Figure 2.9 differs in 0.08% of pixels**, confined to a few curves in panel a; panels
  b, c and d are identical. The current script reproduces the current
  `figure_overlap_shared.png` exactly, so the image embedded in the thesis docx is an
  older render that predates the last edit to `overlap_shared_recompute.py`. Re-render
  it before submission so the thesis image matches the code.
- Figures 2.2, 2.6, 2.7 and 2.8 have no code: they are drawings, gel photographs and
  sequencing traces.

**Legends and printed numbers: Chapter 3 Figures 3.1 to 3.7 and Table 3.1 are audited,
the rest are not.** Pixel-identity proves the code made the image; it says nothing about
whether the legend and the bullets below agree with what is drawn. For 3.1 to 3.7 and
Table 3.1 every panel letter, printed number and colour claim was checked against the
source tables. The result: **no numeric error in the chapter text or in Table 3.1**, and
the corrections that were needed are now folded into the entries below. Four of them were
re-verified independently here against `hypothesis_curve.tsv`, `score_separation_sites.tsv`,
`pas_controlled.tsv` and `cnn_data.npz`.

Chapter 3 Figures 3.8 to 3.14 are now audited the same way, with the same result: **every
headline number traced to a plotted table reproduced exactly.** The problems found were in
wording, sign conventions and three counts, and are folded into the entries below; the load-
bearing ones were re-verified here against `mechanism.tsv`, `wg_per_gene.tsv`,
`implant_cisbp_best.tsv` and `cnn_posmap_occlusion.py`.

Still unaudited in this sense: **all of Chapter 2.** Treat those entries as careful notes
rather than verified fact. Known disagreements are flagged **OPEN** throughout.

To rebuild this document after editing the prose:

```bash
cd pub/thesis_figures_readme
python3 build_readme.py            # refresh images, code blocks and code_manifest.tsv
python3 build_readme.py --check    # exit 1 if the markdown is stale
```

`build_readme.py` only rewrites the regions between `

<!-- BEGIN AUTO:... -->` and
`<!-- END AUTO:... -->

`; prose outside them is never touched, so the two can be edited
independently. Code is read from the live scripts, so a block cannot drift from the
script it documents. `code_manifest.tsv` lists every embedded file with its line count
and SHA-256 prefix, and `code/` holds a copy of each one.

**Two traps that affect any rebuild**

- `make_figure1.R` and `make_figure2.R` resolve each input through a `have()` helper
  that returns `NA` when the file is absent, and each panel is guarded by
  `if (!is.na(...))`. A rebuild on an incomplete tree therefore **drops panels silently
  and still exits 0**. Check the panel count against the legend, not the exit status.
- Twelve of the Chapter 3 plotting scripts begin with a hardcoded
  `setwd("/scratch/spour98/.../pub/thesis_ch3")`. They run from any directory, but they
  break the moment the tree is moved or copied. Edit that line before running them
  elsewhere.

## CHAPTER 2: Local sequence does not distinguish constitutive 3′ ends from competitor sites

### Shared Chapter 2 pipeline (Figures 2.2 to 2.5 and 2.9)

**Positive windows**
- 500 nt centred on the constitutive cleavage site, transcript orientation.
- `PUB/data/Dominant_CPA_baseline_total.bed`, 15,794 rows (`.noY.bed` 15,785).

**Negative windows**
- 500 nt genomic windows excluding annotated pA sites; about 40 per positive (realised 36.7); 577,065 windows in total.
- Two groups in roughly 1:6 ratio, with and without a candidate PAS: `PUB/data/negatives_{w,no}_pa_clean.bed`.

**Splits**
- Production split: train chr1, 2, 4-12, 14-17; validation chr3, 13; test chr18-22, X. chrY excluded.
  - `PUB/scripts/make_splits.py --fold 40 --decoy_per_total 7 --seed 42`, then `make_prod_split.sh`.
- Whole-gene scoring used a 5-fold chromosome cross-fit, so every gene is scored by models that never saw its chromosome: `scripts/assign_folds.py`, realised in `PUB/folds/fold_assignments.tsv`.

**Features** (`PUB/scripts/featurization.py`, driven by `featurize_all.slurm`)
- 48 bins of 30 nt at stride 10.
- 15 feature families per bin = 720 features: CA and UA dinucleotides, UGUA, U-rich and GU-rich k-mers, PWM log-odds for the PAS and related hexamers.
- Background composition `--bg 0.3 0.2 0.2 0.3`.
- PWM list lives at `/scratch/spour98/identify_model/pwm_paths`, outside this tree. **OPEN:** copy it in.
- FASTA extraction: `PUB/scripts/extract_fastas.sh`.

**Models** (`PUB/scripts/train_{rf,lr,cnn}.py`, CNN defined in `cpa_cnn.py`; SLURM arrays `train_*_array.slurm`, orchestrated by `submit_all.sh`; logs in `PUB/logs/`)
- **LR:** L1 penalty, C = 0.1, saga, max_iter 2000, balanced class weights.
- **RF:** 30,000 trees, log-loss criterion, min_split 5, min_leaf 4, balanced class weights, seed 42; sigmoid (Platt) calibration with `cv="prefit"` on the validation set.
- **CNN:** about 661k parameters.
  - Three convolutions, 128/256/256 filters, kernels 8/8/4, each with BatchNorm and ReLU, max pooling between layers, then adaptive max pooling.
  - Two dense layers, dropout 0.3.
  - AdamW, lr 1e-3 with cosine schedule and 5% warm-up, weight decay 1e-4, grad clip 1.0, label smoothing 0.1, batch 256, up to 40 epochs, early stop after 3 epochs without validation AUPRC improvement.
  - The production CNN in fact trained on CPU (`PUB/logs/cnn_46303739_5.out` reports "Device: cpu").
- Decision threshold per model: the value in 0.05 to 0.95, step 0.01, maximising validation F1.
- Stored in `PUB/models/{fold0..4,prod}/{rf,lr,cnn}/`. **Each RF joblib is about 15.7 GB, so exclude them from any archive.**

**Whole-gene scanning**
- `PUB/scripts/score_genes.py` via `score_array.slurm` (that file holds the full command).
- Outputs `PUB/scans/{fold*,prod}/per_gene/*.parquet` (prod covers 20,012 genes); `merge_fold_scans.sh` symlinks the cross-fit scans into `scans/crossfit/per_gene` (19,955 genes).

**Competitor calling** (`PUB/scripts/build_shortlist.py`)
- Keeps every window that RF, LR **and** CNN each score at or above that gene's own constitutive window, then collapses them by non-maximum suppression (`--merge_dist 250`).
- Recorded invocation: `--per_gene_dir scans/crossfit/per_gene --canonical_bed data/Dominant_CPA_baseline_total.bed --aparent_scan_dir ../train/gene_scans/per_gene --win 500 --out shortlist_crossfit`.
- Output `PUB/shortlist_crossfit_windows.parquet`, then `cpa_peaks_final.parquet`.

**APARENT2** (Linder et al. 2022), used as model-independent evidence
- `CLEAN/scripts/score_genes_aparent2.py`, run by `/scratch/spour98/containers/run_aparent2_scan.sh` inside `/scratch/spour98/containers/aparent2.sif` (8.3 GB; recipe `aparent2.def`).
- Weights: `aparent_all_libs_resnet_no_clinvar_wt_ep_5_var_batch_size_inference_mode_no_drop.h5`.
- 205 nt input, cut index 70, library index 11, cut radius 5.
- Scores written into `CLEAN/train/gene_scans/per_gene/*.parquet`; run log `CLEAN/logs/aparent2_scan_summary.json`.
### Figure 2.1: Selection of constitutive CPA sites

<!-- BEGIN AUTO:2.1:image -->
![Figure 2.1](figures/fig2_1.png)
<!-- END AUTO:2.1:image -->

**Shows**
- A funnel of PolyA_DB gene symbols through each filter.
- Gene symbols by biotype.
- Where the pA sites of genes dropped for lacking a 3′ UTR site actually lie.
- Distribution of PSE x RPM for the retained sites (log10 axis; median and IQR marked).

**Why it exists:** it defines one real 3′ end per protein-coding gene, the positive class for everything that follows.

**What was done**
- Start: 303,312 pA sites over 31,101 gene symbols.
- Removed 13,478 non-coding genes, 874 genes whose sites map to more than one gene, and 1,049 genes with no 3′ UTR site.
- Kept, per gene, the 3′ UTR site with the highest PSE x RPM (percentage of samples expressing the site x mean reads per million). No expression threshold.
- Result: one site for each of 15,700 genes, 89.1% of protein-coding genes, median PSE x RPM 413.

**How**
- Data: `PUB/scripts/select_constitutive_cpa.py --pas hg38.PAS.main.tsv`; defaults are source 3reads, region 3′ UTR, protein_coding, multi-mapping sites dropped. No `--ucsc_refgene` was passed, so the longest-isoform step was approximated.
- Output: `PUB/constitutive_cpa.tsv` and `.bed` (15,700 rows).
- Plot: `cd PUB && Rscript sfig1.R constitutive_cpa.tsv` writes `supp_fig1.{pdf,png}` (thesis image7). Panels a-c have hard-coded counts; only panel d reads data.

**OPEN**
- The funnel ends at 15,700 genes but the positive BED has 15,794 rows; the 94-row difference is unexplained, and the chapter also quotes 15,785, 15,568 and 15,559 in places.
- PolyA_DB version (v3 vs v4).
- `PUB/figureS1_constitutive_selection.png` is a different render with an unknown generator.

<!-- BEGIN AUTO:2.1:code -->
**Code.** `sfig1.R`, 139 lines, sha256 `e7591fa0ffb8`. Reproduced verbatim; also in `code/sfig1.R`.

```r
#!/usr/bin/env Rscript
# make_supp1.R -- Supplementary Figure 1: selection of constitutive polyA sites
# from PolyA DB v4 (hg38.PAS.main.tsv), and why the positive set is ~15.7k not 20k.
#   A  selection funnel (all gene symbols -> protein-coding -> unique -> 3'UTR site)
#   B  biotype composition of all distinct gene symbols (most are non-coding)
#   C  the dropped genes have no 3'UTR site: their PAS are cryptic/non-terminal
#   D  PSExRPM of the selected constitutive sites (real, read from the output TSV)
#   Rscript make_supp1.R [constitutive_cpa.tsv]
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(patchwork);library(scales)})
args<-commandArgs(trailingOnly=TRUE)
TSV<-if(length(args)>=1) args[1] else "constitutive_cpa.tsv"

INK<-"#1F2937"; MUT<-"#6B7280"; KEEP<-"#3E82B5"; ACC<-"#E8963A"; DROPC<-"#B0662A"
th<-theme_minimal(base_size=12)+
  theme(text=element_text(color=INK),
        plot.title=element_text(face="bold",size=13,hjust=0,margin=margin(t=6,b=2)),
        plot.subtitle=element_text(size=10,color=MUT,margin=margin(b=6)),
        axis.text=element_text(color=INK,size=9),
        panel.grid.minor=element_blank(),
        panel.grid.major.y=element_blank(),
        legend.position="none")

## ---- numbers from the PolyA DB v4 diagnostics (editable) ----
stage_nm<-c("all gene symbols","protein-coding","uniquely mapped","3\u2032UTR site (kept)")
stage_n <-c(31101, 17623, 16749, 15700)
drop_rsn<-c("\u2212 13,478 non-coding biotypes","\u2212 874 multi-mapped (PAS2gene\u22601)","\u2212 1,049 no 3\u2032UTR PAS")

biotype<-c("protein_coding","lncRNA","transcribed_pseudogene","pseudogene","snoRNA",
           "miRNA","tRNA","V_segment","snRNA","misc_RNA","ncRNA","V_segment_pseudogene","other")
biocnt <-c(17623,11278,803,607,286,146,116,74,47,24,20,16,61)

drop_lab<-c("Intron","Intergenic","CDS","5\u2032UTR","Downstream","Upstream",
            "3\u2032-most exon","Internal exon","5\u2032-most exon")
drop_n  <-c(1679,1385,388,270,199,142,28,4,1)
drop_term<-c(FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,TRUE,FALSE,FALSE)  # terminal-exon (recoverable)

## ===== A: selection funnel =====
Wmax<-1.0; hh<-0.20; maxn<-max(stage_n)
fd<-data.frame(nm=stage_nm,n=stage_n,hw=stage_n/maxn*Wmax,y=-(seq_along(stage_n)-1))
# bars
bars<-do.call(rbind,lapply(seq_len(nrow(fd)),function(i)
  data.frame(id=i,x=c(-fd$hw[i],fd$hw[i],fd$hw[i],-fd$hw[i]),
             y=c(fd$y[i]-hh,fd$y[i]-hh,fd$y[i]+hh,fd$y[i]+hh))))
bars$fill<-factor(bars$id)
# connectors (funnel walls) between consecutive stages
conn<-do.call(rbind,lapply(seq_len(nrow(fd)-1),function(i)
  data.frame(id=i,x=c(-fd$hw[i],fd$hw[i],fd$hw[i+1],-fd$hw[i+1]),
             y=c(fd$y[i]-hh,fd$y[i]-hh,fd$y[i+1]+hh,fd$y[i+1]+hh))))
fills<-c("#A9C9E0","#6FA6CC","#3E82B5","#1E5C8F")
ctext<-c("#123A57","white","white","white")
xlab_r<-Wmax+0.12
pA<-ggplot()+
  geom_polygon(data=conn,aes(x,y,group=id),fill="#E9EDF1",color=NA)+
  geom_polygon(data=bars,aes(x,y,group=id,fill=fill),color="white",linewidth=.4)+
  scale_fill_manual(values=fills)+
  guides(fill="none")+
  # count inside each bar
  geom_text(data=fd,aes(0,y,label=comma(n)),color=ctext,fontface="bold",size=4)+
  # stage name in the right column
  geom_text(data=fd,aes(xlab_r,y,label=nm),hjust=0,size=3.4,color=INK)+
  # drop labels in the gaps
  geom_text(data=data.frame(y=(fd$y[-nrow(fd)]+fd$y[-1])/2,lab=drop_rsn),
            aes(xlab_r,y,label=lab),hjust=0,size=3,fontface="italic",color=DROPC)+
  # retained fraction of the protein-coding pool
  annotate("text",x=0,y=fd$y[nrow(fd)]-0.42,
           label=sprintf("%.1f%% of protein-coding genes retained",100*15700/17623),
           size=3,color=MUT)+
  coord_cartesian(xlim=c(-Wmax-0.05,Wmax+1.75),ylim=c(min(fd$y)-0.55,0.30),clip="off")+
  labs(title="Selecting constitutive polyadenylation sites",
       subtitle="One dominant 3\u2032UTR site per gene (highest PSE\u00d7RPM), from 303,312 PAS in PolyA DB v4")+
  theme_void(base_size=12)+
  theme(plot.title=element_text(face="bold",size=13,hjust=0),
        plot.subtitle=element_text(size=10,color=MUT),
        plot.margin=margin(4,4,2,4))

## ===== B: biotype composition (lollipop) =====
bt<-data.frame(biotype,n=biocnt)%>%arrange(desc(n))
bt$biotype<-factor(bt$biotype,levels=rev(bt$biotype))
bt$hl<-bt$biotype=="protein_coding"
pB<-ggplot(bt,aes(n,biotype))+
  geom_segment(aes(x=0,xend=n,yend=biotype,color=hl),linewidth=.9)+
  geom_point(aes(color=hl,size=hl))+
  geom_text(aes(label=comma(n)),hjust=-0.22,size=2.7,color=INK)+
  scale_color_manual(values=c(`TRUE`=ACC,`FALSE`="#9AA0A6"))+
  scale_size_manual(values=c(`TRUE`=3.3,`FALSE`=2.1))+
  scale_x_continuous(expand=expansion(mult=c(0,0.20)),labels=comma)+
  labs(title="Most gene symbols aren't protein-coding",
       subtitle="Distinct gene symbols by biotype",
       x="genes",y=NULL)+th
## ===== C: dropped genes have no 3'UTR site =====
dc<-data.frame(lab=drop_lab,n=drop_n,term=drop_term)%>%arrange(desc(n))
dc$lab<-factor(dc$lab,levels=rev(dc$lab))
dc$grp<-ifelse(dc$term,"terminal exon (recoverable)","cryptic / non-terminal")
pC<-ggplot(dc,aes(n,lab,fill=grp))+
  geom_col(width=.68)+
  geom_text(aes(label=comma(n)),hjust=-0.2,size=2.7,color=INK)+
  scale_fill_manual(values=c("cryptic / non-terminal"="#9AA0A6","terminal exon (recoverable)"=ACC),name=NULL)+
  scale_x_continuous(expand=expansion(mult=c(0,0.18)),labels=comma)+
  labs(title="Dropped genes lack a 3\u2032UTR site",
       subtitle="PAS location \u2014 nearly all cryptic",
       x="PAS in these genes",y=NULL)+
  th+theme(legend.position=c(.72,.28),legend.background=element_rect(fill="white",color=NA),
           legend.text=element_text(size=8),legend.key.size=unit(.8,"lines"))

## ===== D: PSExRPM of the selected sites (real data) =====
pD<-NULL
if(file.exists(TSV)){
  dd<-tryCatch(suppressMessages(readr::read_tsv(TSV,show_col_types=FALSE)),error=function(e) NULL)
  if(!is.null(dd) && "PSExRPM"%in%names(dd)){
    v<-suppressWarnings(as.numeric(dd$PSExRPM)); v<-v[is.finite(v)&v>0]
    if(length(v)>=10){
      med<-median(v); q1<-as.numeric(quantile(v,.25)); q3<-as.numeric(quantile(v,.75))
      pD<-ggplot(data.frame(v=v),aes(v))+
        geom_histogram(aes(y=after_stat(density)),bins=46,fill="#CBDDEC",color="white",linewidth=.2)+
        geom_density(color=KEEP,linewidth=.9)+
        geom_vline(xintercept=med,color=DROPC,linewidth=.7)+
        geom_vline(xintercept=c(q1,q3),color=MUT,linetype=2,linewidth=.4)+
        scale_x_log10(labels=trans_format("log10",math_format(10^.x)))+
        annotate("text",x=med,y=Inf,label=sprintf("median %s",comma(round(med))),
                 hjust=-0.08,vjust=1.6,size=3,color=DROPC)+
        labs(title=sprintf("Selected sites well-supported (n=%s)",comma(length(v))),
             subtitle="PSE\u00d7RPM per gene (log; dashed = IQR)",
             x="PSE \u00d7 RPM",y="density")+th
    }
  }
}
if(is.null(pD))
  pD<-ggplot()+annotate("text",0,0,label=paste0("PSE\u00d7RPM panel:\nrun in the dir with ",TSV),size=3.2,color=MUT)+
    labs(title="Selected-site scores")+theme_void()+theme(plot.title=element_text(face="bold"))

fig<-pA/(pB|pC|pD)+plot_layout(heights=c(1.18,1))+
  plot_annotation(
    title="Supplementary Figure 1  |  Constitutive polyadenylation-site selection from PolyA DB v4",
    caption="Constitutive site = the highest PSE\u00d7RPM PAS in the 3\u2032UTR of each protein-coding gene (one per gene). PSE\u00d7RPM from the 3READS quantification.",
    theme=theme(plot.title=element_text(face="bold",size=14,hjust=0),
                plot.caption=element_text(size=8.5,color=MUT,hjust=0)))
ggsave("supp_fig1.pdf",fig,width=13,height=10,device=cairo_pdf,limitsize=FALSE)
ggsave("supp_fig1.png",fig,width=13,height=10,dpi=220,limitsize=FALSE)
cat("wrote supp_fig1.pdf / .png\n")
```
<!-- END AUTO:2.1:code -->

### Figure 2.2: Windows, features and models (schematic)

<!-- BEGIN AUTO:2.2:image -->
![Figure 2.2](figures/fig2_2a.png)

![Figure 2.2](figures/fig2_2b.png)
<!-- END AUTO:2.2:image -->

**Shows** the 500 nt window and its 48 bins of 30 nt carrying 15 feature families; LR and RF reading the feature matrix while the CNN reads one-hot sequence; each model returning a probability.

**How made:** hand-drawn. No script, and the source file (PowerPoint, Illustrator or similar) was not found on the cluster. **OPEN.** Content is the shared pipeline above (thesis images image8 and image9).

<!-- BEGIN AUTO:2.2:code -->
*No code: this figure is a drawing, a photograph of a gel, or a sequencing trace panel. See the figure's entry above for what is known about its origin.*
<!-- END AUTO:2.2:code -->

### Figure 2.3: Performance of the three models

<!-- BEGIN AUTO:2.3:image -->
![Figure 2.3](figures/fig2_3.png)
<!-- END AUTO:2.3:image -->

**Shows**
- **a.** AUROC, AUPRC and F1 on the test set.
- **b.** Pairwise Pearson correlation of model scores over every window of the scanned genes.
- **c.** Fraction of genes whose constitutive site lies within 250 nt of one of the K highest-scoring windows, per model and for the best-of-three rank.
- **d.** Every window of *ERP27* (chr12:14,913,288-14,938,537, minus strand; 24,750 windows) by distance from the gene's 5′ end.
- **e.** Scores at constitutive sites, at PolyASite 2.0 clusters more than 250 nt away, and at six random windows per gene.
- **f.** Mean per-gene Jaccard index of the peak sets called by each pair of models.
- **g.** Score against site expression for PolyASite clusters, split by terminal-exon / intragenic / other, in 12 equal-count bins.

**Key numbers**

| Model | AUROC | AUPRC | F1 |
|---|---|---|---|
| CNN | 0.991 | 0.923 | 0.865 |
| RF | 0.987 | 0.841 | 0.772 |
| LR | 0.983 | 0.827 | 0.758 |

- Test set 104,031 windows: 2,935 positive, 101,096 negative.
- F1 thresholds chosen on validation: RF 0.29, LR 0.93, CNN 0.37.
- Best-of-three rank: the constitutive site is the top window in 47.5% of genes, in the top 10 in 68.9%. So in about half of genes some other position wins.
- Mean per-gene Jaccard between model peak sets: 0.07 to 0.17, i.e. the three models agree on performance but not on which positions they like.

**How**
- Data: `PUB/scripts/make_figure1_data.py` (v3.0), arguments `--prod_dir --crossfit_dir --canonical_bed --atlas_bed --prod_rf --prod_lr --prod_cnn --erp27 ERP27 --vic 250`. **OPEN:** the exact invocation and its log were not found.
- Inputs: `PUB/scans/prod/per_gene`, `PUB/scans/crossfit/per_gene`, `PUB/data/Dominant_CPA_baseline_total.bed`, `CLEAN/train/data/atlas.chr.bed`.
- Outputs by panel:

| Panel | File |
|---|---|
| counts | `fig1_counts.tsv` |
| b | `fig1_corr.tsv` |
| c | `fig1_topk.tsv` |
| d | `fig1_traces.tsv` |
| e | `fig1_scoredist.tsv` (41 MB) |
| f | `fig1_jaccard.tsv` |
| g | `figG_binned.tsv`, `figG_raw.tsv`, `figG_group_counts.tsv`, `figG_corr.tsv` |
| thresholds | `fig1_thresholds.tsv` |

- Plot: `Rscript PUB/scripts/make_figure1.R .` writes `figure1.{pdf,png}`, 15 x 21 in at 400 dpi (thesis image10).
- Legends: `PUB/figure1_legends.md`, thesis version `PUB/figure1_legend_thesis.md`.

**OPEN, and it matters:** panel a's nine numbers are hard-coded at `make_figure1.R:83-84` and five disagree with `models/prod/*/metadata.json`.

| Model | Metric | In the figure | In metadata |
|---|---|---|---|
| RF | AUPRC | 0.851 | 0.841 |
| RF | F1 | 0.781 | 0.772 |
| LR | AUPRC | 0.840 | 0.827 |
| LR | F1 | 0.761 | 0.758 |
| CNN | F1 | 0.872 | 0.865 |

The chapter text quotes the metadata values, so the figure should be re-rendered from metadata before submission.

<!-- BEGIN AUTO:2.3:code -->
**Code.** `scripts/make_figure1.R`, 227 lines, sha256 `6410fd64f528`. Reproduced verbatim; also in `code/make_figure1.R`.

```r
#!/usr/bin/env Rscript
# make_figure1.R
# Builds one combined figure (panels A-G) in a single layout, shared theme/palette.
# theme_classic, font >= 17, colorblind-safe, one hue per model.
#   Rscript make_figure1.R [indir]
#
# Figure 1 (figure1.pdf/png):
#   A metrics [prod,locked]  B corr [prod]  C top-K recovery [crossfit, within-250nt]
#   D ERP27 traces [prod]    E score-by-class [prod, Canonical/Atlas/Background]
#   F Jaccard [prod, vicinity threshold]
# Figure G (appended below A-F in figure1.pdf/png):
#   model score vs PolyASite expression as binned lines (mean +/- 95% CI per
#   expression bin), colored by site group (Terminal/Intragenic/Other), faceted
#   by model. Per-group Spearman(score, expr) is written alongside by the data step.
#
# Every data panel reads its TSV if present, else draws a labelled placeholder.
#
# NOTE ON THRESHOLDS
#   Peaks are called PER GENE, using the score at that gene's own 3' end as the
#   cut. There is no single global threshold, so no horizontal threshold line is
#   drawn in panel D or E -- one would misrepresent the calling rule. The
#   fig1_thresholds.tsv values are still read and printed to stderr for the
#   record, but they are not plotted.

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
  library(readr);  library(scales)
})
args  <- commandArgs(trailingOnly = TRUE)
INDIR <- if (length(args) >= 1) args[1] else "."
have  <- function(f){ p <- file.path(INDIR,f); if (file.exists(p)) p else NA_character_ }

## ============================ shared theme + palette ============================
MODELS <- c("RF","LR","CNN")
MCOL   <- c(RF="#0072B2", LR="#E69F00", CNN="#009E73")
CLSCOL <- c(Canonical="#0072B2", Atlas="#5FA8D3", Background="#C9CCD1")
TCOL   <- c(TE="#0072B2", EX="#56B4E9", IN="#E69F00", DS="#009E73",
            AE="#CC79A7", AI="#D55E00", AU="#8A8F98", IG="#B0B4BA")
INK    <- "#1A1A1A"; BASE <- 17
th <- theme_classic(base_size = BASE) +
  theme(text=element_text(color=INK), axis.text=element_text(color=INK,size=BASE-3),
        axis.title=element_text(size=BASE-1), axis.line=element_line(color=INK,linewidth=0.5),
        axis.ticks=element_line(color=INK,linewidth=0.5),
        plot.title=element_text(size=BASE+2,face="bold",hjust=0),
        plot.subtitle=element_text(size=BASE-4,color="#555555"),
        legend.position="top", legend.title=element_blank(),
        legend.key.size=unit(0.9,"lines"), legend.text=element_text(size=BASE-3),
        strip.background=element_blank(), strip.text=element_text(face="bold",size=BASE-3),
        plot.margin=margin(6,8,6,8))
placeholder <- function(title, msg="awaiting scan output"){
  ggplot()+annotate("text",0,0.15,label=msg,size=4.4,color="#8A8F98")+
    annotate("text",0,-0.05,label="(fills in when data lands)",size=3.4,color="#B0B4BA")+
    lims(x=c(-1,1),y=c(-1,1))+labs(title=title)+th+
    theme(axis.line=element_blank(),axis.text=element_blank(),
          axis.ticks=element_blank(),axis.title=element_blank())
}

## ---- thresholds: read for the record only; NOT plotted (per-gene calling) ----
THRESH <- c(RF=0.932, LR=0.980, CNN=0.889)
fT <- have("fig1_thresholds.tsv")
if(!is.na(fT)){
  td <- read_tsv(fT, show_col_types=FALSE)
  THRESH <- setNames(td$threshold, td$model)[MODELS]
}

## ---- per-panel sample sizes (annotate each subtitle with n) ----
NCOUNT <- list()
fN <- have("fig1_counts.tsv")
if(!is.na(fN)){
  nc <- read_tsv(fN, show_col_types=FALSE)
  for(i in seq_len(nrow(nc))) NCOUNT[[nc$panel[i]]] <- list(n=nc$n[i], unit=nc$unit[i])
}
nsub <- function(panel, base){
  if(!is.null(NCOUNT[[panel]])){
    u <- NCOUNT[[panel]]$unit; nn <- format(NCOUNT[[panel]]$n, big.mark=",")
    paste0(base, "  (n = ", nn, " ", u, ")")
  } else base
}

## ================================ FIGURE 1 ================================

## ---- A: metrics (prod, locked) ----
metrics <- tribble(~model,~AUROC,~AUPRC,~F1,
  "RF",0.987,0.851,0.781, "LR",0.983,0.840,0.761, "CNN",0.991,0.923,0.872) |>
  pivot_longer(-model,names_to="metric",values_to="value") |>
  mutate(model=factor(model,levels=MODELS),
         metric=factor(metric,levels=c("AUROC","AUPRC","F1")))
pA <- ggplot(metrics,aes(metric,value,fill=model))+
  geom_col(position=position_dodge(width=0.80),width=0.66)+
  geom_text(aes(label=sprintf("%.3f",value)),position=position_dodge(width=0.80),
            vjust=-0.5,size=2.6,color=INK)+
  scale_fill_manual(values=MCOL)+
  scale_y_continuous(limits=c(0,1.08),breaks=seq(0,1,.25),expand=expansion(mult=c(0,0.02)))+
  labs(title="A",subtitle=nsub("A","Held-out test performance"),x=NULL,y="Score")+th

## ---- B: correlation (prod) ----
fB <- have("fig1_corr.tsv")
if(!is.na(fB)){
  d <- read_tsv(fB,show_col_types=FALSE)
  if(!all(c("model_x","model_y","r") %in% names(d))){
    m <- as.matrix(d[,-1]); rownames(m) <- d[[1]]
    d <- as.data.frame(as.table(m)); names(d) <- c("model_x","model_y","r")
  }
  d <- mutate(d, model_x=factor(model_x,levels=MODELS), model_y=factor(model_y,levels=rev(MODELS)))
  pB <- ggplot(d,aes(model_x,model_y,fill=r))+
    geom_tile(color="white",linewidth=1.2)+
    geom_text(aes(label=sprintf("%.2f",r)),size=4.2,color=INK)+
    scale_fill_gradient(low="#F2F6FA",high="#0072B2",limits=c(0,1),name="r")+
    coord_equal()+labs(title="B",subtitle=nsub("B","Per-window score correlation"),x=NULL,y=NULL)+
    th+theme(legend.position="right",legend.title=element_text(size=BASE-4),panel.grid=element_blank())
} else pB <- placeholder("B","per-window correlation")

## ---- C: top-K recovery (crossfit, within-250nt); axis capped to data ----
fC <- have("fig1_topk.tsv")
if(!is.na(fC)){
  d <- read_tsv(fC,show_col_types=FALSE) |> mutate(model=factor(model,levels=c(MODELS,"Combined")))
  ymax <- min(1, max(d$recovery)*1.15)
  pC <- ggplot(d,aes(K,recovery,color=model))+
    geom_line(linewidth=0.9)+geom_point(size=1.8)+
    scale_color_manual(values=c(MCOL,Combined=INK))+
    scale_y_continuous(labels=percent_format(accuracy=1),limits=c(0,ymax),
                       expand=expansion(mult=c(0,0.02)))+
    scale_x_continuous(breaks=pretty(d$K))+
    labs(title="C",subtitle=nsub("C","Top-K canonical recovery"),x="K (top windows)",y="Recovery")+th
} else pC <- placeholder("C","top-K recovery")

## ---- D: ERP27 traces (prod) ----
# No threshold line: peaks are called per gene from that gene's own 3'-end
# score, so a single horizontal cut across the trace would be wrong.
fD <- have("fig1_traces.tsv")
if(!is.na(fD)){
  d <- read_tsv(fD,show_col_types=FALSE) |> mutate(model=factor(model,levels=MODELS))
  pD <- ggplot(d,aes(pos,prob))+
    geom_line(aes(color=model),linewidth=0.6,show.legend=FALSE)+
    facet_grid(model~.,scales="free_y")+scale_color_manual(values=MCOL)+
    scale_y_continuous(breaks=c(0,0.5,1))+
    labs(title="D",subtitle=nsub("D","Per-gene score traces (ERP27)"),x="Position in gene (nt)",y="Score")+
    th+theme(strip.text=element_text(face="bold",size=BASE-4))
} else pD <- placeholder("D","ERP27 score traces")

## ---- E: score by class (prod, Canonical/Atlas/Background) ----
# No threshold line, same reason as D.
fE <- have("fig1_scoredist.tsv")
if(!is.na(fE)){
  d <- read_tsv(fE,show_col_types=FALSE)
  present <- intersect(names(CLSCOL), unique(d$class))
  d <- d |> mutate(model=factor(model,levels=MODELS), class=factor(class,levels=present))
  pE <- ggplot(d,aes(class,score,fill=class))+
    geom_violin(color=NA,alpha=0.9,scale="width")+
    facet_wrap(~model,nrow=1)+scale_fill_manual(values=CLSCOL[present],guide="none")+
    scale_y_continuous(limits=c(0,1))+
    labs(title="E",subtitle=nsub("E","Score by class"),x=NULL,y="Score")+
    th+theme(axis.text.x=element_text(angle=30,hjust=1),strip.text=element_text(face="bold",size=BASE-4))
} else pE <- placeholder("E","score distributions")

## ---- F: Jaccard (prod) ----
fF <- have("fig1_jaccard.tsv")
if(!is.na(fF)){
  d <- read_tsv(fF,show_col_types=FALSE)
  if(!"pair" %in% names(d)) names(d)[1] <- "pair"
  if(!"jaccard" %in% names(d)) names(d)[2] <- "jaccard"
  d <- mutate(d, pair=reorder(pair,jaccard))
  pF <- ggplot(d,aes(jaccard,pair))+
    geom_col(fill="#0072B2",width=0.62)+
    geom_text(aes(label=sprintf("%.2f",jaccard)),hjust=-0.2,size=3.4,color=INK)+
    scale_x_continuous(limits=c(0,max(d$jaccard)*1.25),expand=expansion(mult=c(0,0.02)))+
    labs(title="F",subtitle=nsub("F","Peak-set overlap (Jaccard)"),x="Jaccard",y=NULL)+th
} else pF <- placeholder("F","pairwise Jaccard")

## ---- G: score vs expression, binned lines by site group ----
GCOL <- c(Terminal="#0072B2", Intragenic="#E69F00", Other="#B0B4BA")
fg <- have("figG_binned.tsv")
if(!is.na(fg)){
  bl <- read_tsv(fg, show_col_types=FALSE) |>
    mutate(model=factor(model,levels=MODELS),
           group=factor(group, levels=c("Terminal","Intragenic","Other")))
  gcl <- setNames(levels(bl$group), levels(bl$group))
  gcf <- have("figG_group_counts.tsv")
  if(!is.na(gcf)){
    gc <- read_tsv(gcf, show_col_types=FALSE)
    for(g in names(gcl)){ k <- gc$n[gc$group==g]
      if(length(k)) gcl[g] <- sprintf("%s (n=%s)", g, format(k, big.mark=",")) }
  }
  ann <- NULL
  fgc <- have("figG_corr.tsv")
  if(!is.na(fgc)){
    xleft <- min(bl$med_tpm, na.rm=TRUE)
    ann <- read_tsv(fgc, show_col_types=FALSE) |>
      filter(!is.na(spearman)) |>
      mutate(model=factor(model, levels=MODELS),
             group=factor(group, levels=c("Terminal","Intragenic","Other"))) |>
      arrange(model, group) |>
      group_by(model) |>
      mutate(x=xleft, y=0.20 - 0.070*(as.integer(group)-1),
             lab=sprintf("%s rho=%+.2f", substr(as.character(group),1,4), spearman)) |>
      ungroup()
  }
  pG <- ggplot(bl, aes(med_tpm, mean_score, color=group, fill=group)) +
    geom_ribbon(aes(ymin=lo, ymax=hi), alpha=0.15, color=NA) +
    geom_line(linewidth=1.0) + geom_point(size=1.6) +
    facet_wrap(~model, nrow=1) +
    scale_x_continuous(trans="log10", breaks=c(0.1,1,10,100),
                       labels=c("0.1","1","10","100")) +
    scale_y_continuous(limits=c(0,1)) +
    scale_color_manual(values=GCOL, labels=gcl, name="Site group") +
    scale_fill_manual(values=GCOL, guide="none") +
    labs(title="G", subtitle=nsub("G","Model score vs expression (mean +/- 95% CI per expression bin; rho = Spearman score vs expr)"),
         x="Average expression (TPM, log scale)", y="Mean model score") +
    th + theme(legend.position="right", legend.title=element_text(size=BASE-4))
  if(!is.null(ann))
    pG <- pG + geom_text(data=ann, aes(x=x, y=y, label=lab, color=group),
                         hjust=0, size=3.0, show.legend=FALSE, inherit.aes=FALSE)
} else pG <- placeholder("G","score vs expression by site group")

## ================================ ASSEMBLE (A-G) ================================
# top six panels in a 3x2 grid, then G spanning the full width below.
top <- pA+pB+pC+pD+pE+pF+plot_layout(design="AB\nCD\nEF",heights=c(1,1,1))
fig <- top / pG + plot_layout(heights=c(3, 1.15))
ggsave(file.path(INDIR,"figure1.pdf"),fig,width=15,height=21,device=cairo_pdf,
       limitsize=FALSE)
ggsave(file.path(INDIR,"figure1.png"),fig,width=15,height=21,dpi=400,
       limitsize=FALSE)
cat("[out] figure1.pdf, figure1.png (panels A-G)\n")
cat("peaks are called per gene from that gene's own 3'-end score;",
    "no global threshold line is drawn.\n")
cat("thresholds in fig1_thresholds.tsv (recorded, not plotted):",
    paste(sprintf("%s=%.3f",MODELS,THRESH[MODELS]),collapse=" "),"\n")
```
<!-- END AUTO:2.3:code -->

### Figure 2.4: Derivation of the analysis set

<!-- BEGIN AUTO:2.4:image -->
![Figure 2.4](figures/fig2_4.png)
<!-- END AUTO:2.4:image -->

**Shows** how many peaks survive each filter. Every filter acts on competitors only; constitutive sites are never removed.

**Why:** it strips out competitors that are already known pA sites, or that cannot be sites of premature cleavage, leaving the set used for the rest of the thesis.

**Filters in order**

| Step | Removed | Remaining |
|---|---|---|
| All candidates | - | 70,723 peaks = 6,583 constitutive + 64,140 competitors |
| Competitor within 500 nt of a PolyASite 2.0 cluster or PolyA_DB site, same strand | 18,119 | 46,021 competitors |
| Competitor whose upstream 500 nt overlaps a GENCODE v49 last exon by >= 1 nt | 1,076 | - |
| Genes left without a competitor | 1,660 genes | 4,923 genes |
| Competitor at or downstream of the constitutive site (transcription ends at the used site) | 1,398 | 43,547 competitors |
| Gene rule re-applied | 76 genes | **4,847 genes, 48,394 candidates** |

- Competitors per gene: median 3, range 1 to 897. The top 10% of genes hold 62% of competitors; Spearman rho with gene length is 0.42. **OPEN:** no saved table for these three numbers, though they are recomputable from the parquet.

**How**
- `PUB/trace_analysis_set.py` rebuilds the mask from `atlas_peak_cache.npz` (`build_atlas_cache.py`), `polyadb_peak_cache.npz` (`build_polyadb_cache.py`) and `termexon_gc_cache.npz` (`task_termexon_flag.py`). It asserts the result equals `corr_rowmask.npz` (from `task_rerank_corr.py`), `upstream_rowmask.npz` (from `task_upstream_blocks.py`) and the parquet's `in_analysis_set`, then writes `analysis_set_funnel.tsv`.
- Plot: `cd PUB && Rscript make_analysis_set_funnel.R` writes `figure_analysis_set_funnel.{pdf,png}` (thesis image11).
- Legend: `PUB/THESIS_figure_analysis_set_funnel.txt`.
- Note the exclusion used PolyA_DB **v4** plus PolyASite 2.0 within 500 nt.

<!-- BEGIN AUTO:2.4:code -->
**Code.** `make_analysis_set_funnel.R`, 111 lines, sha256 `8d980bcfa891`. Reproduced verbatim; also in `code/make_analysis_set_funnel.R`.

```r
#!/usr/bin/env Rscript
# make_analysis_set_funnel.R -- how cpa_peaks_final.parquet's in_analysis_set is built.
#
# PLOTTING ONLY. Every count is read from analysis_set_funnel.tsv, written by
# trace_analysis_set.py, which rebuilds the mask from the raw annotation caches and asserts
# it equals corr_rowmask.npz, upstream_rowmask.npz and the shipped column.
#
# VERTICAL ATTRITION FUNNEL. Two earlier designs were discarded. A Sankey put five removed
# branches on ribbons that cross the trunk and each other. A waterfall drew each row as the
# population entering a stage with the next loss peeled off its right end, which made every
# bar mean something different from its own row label. Here a bar is one thing only: the
# peaks that survive that stage. Losses live in the gaps, as the notch between two bars.
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr)})
e<-function(...) cat(...,"\n",file=stderr())
INK<-"#1A1A1A"; BASE<-8
REAL<-"#08519C"; COMP<-"#E8A33D"; BAR<-"#59677A"; GREY<-"#9AA0A6"; GTX<-"#5A6068"
F<-read_tsv("analysis_set_funnel.tsv",show_col_types=FALSE)
stopifnot(nrow(F)==6, F$n_peaks==F$n_canonical+F$n_competitor)
lab<-function(x) format(as.numeric(x),big.mark=",",trim=TRUE)
TOT<-F$n_peaks[1]

STAGE<-c("All called peaks",
         "Annotated alternative 3' ends removed",
         "Reference terminal-exon peaks removed",
         "Genes without a surviving competitor dropped",
         "Peaks at or downstream of the canonical removed",
         "Analysis set")
CUT<-c("APA-annotated",
       "reference terminal exon",
       "canonical peaks (gene lost all competitors)",
       "at or downstream of the canonical",
       "canonical peaks (gene rule re-applied)")
F$k<-seq_len(nrow(F)); F$y<--F$k; F$stage<-STAGE
H<-0.30; PAD<-TOT*0.012

# bars: stages 1-5 are one neutral block, the last is split into its two populations
B<-bind_rows(
  F[1:5,]%>%transmute(y,x0=0,x1=n_peaks,seg="stage"),
  data.frame(y=F$y[6],x0=0,x1=F$n_canonical[6],seg="real"),
  data.frame(y=F$y[6],x0=F$n_canonical[6],x1=F$n_peaks[6],seg="competitor"))%>%
  mutate(seg=factor(seg,levels=c("stage","real","competitor")))
# losses: the notch between bar k-1 and bar k, drawn in the gap it belongs to
L<-data.frame(y=F$y[-1]+0.5, x0=F$n_peaks[-1], x1=F$n_peaks[-nrow(F)],
              n=F$n_removed[-1], cut=CUT)%>%
  mutate(txt=paste0("−",lab(n),"  ",cut))
stopifnot(all(L$x1-L$x0==L$n))

p<-ggplot()+
  geom_rect(data=L,aes(xmin=x0,xmax=x1,ymin=y-0.5+H,ymax=y+0.5-H),
            fill=GREY,alpha=.30,color=NA)+
  geom_rect(data=B,aes(xmin=x0,xmax=x1,ymin=y-H,ymax=y+H,fill=seg),
            color="white",linewidth=.3)+
  scale_fill_manual(values=c(stage=BAR,real=REAL,competitor=COMP),guide="none")+
  # peaks surviving each stage, at the end of its bar
  geom_text(data=F,aes(x=n_peaks+PAD,y=y,label=paste0(lab(n_peaks)," peaks, ",lab(n_genes)," genes")),
            hjust=0,vjust=0.5,size=2.5,color=INK)+
  # what the step above removed, in the gap, just past the notch
  geom_text(data=L,aes(x=x1+PAD,y=y,label=txt),hjust=0,vjust=0.5,size=2.4,color=GTX)+
  # the final composition, on the bar itself
  annotate("text",x=F$n_canonical[6]/2,y=F$y[6],hjust=.5,vjust=.5,size=2.4,color="white",
           fontface="bold",label=lab(F$n_canonical[6]))+
  annotate("text",x=F$n_canonical[6]+(F$n_competitor[6]-F$n_canonical[6])/2,y=F$y[6],
           hjust=.5,vjust=.5,size=2.4,color="white",fontface="bold",label=lab(F$n_competitor[6]))+
  annotate("text",x=F$n_canonical[6]/2,y=F$y[6]-H-0.13,hjust=.5,vjust=1,size=2.3,color=REAL,
           label="real 3' ends")+
  annotate("text",x=F$n_canonical[6]+(F$n_competitor[6]-F$n_canonical[6])/2,y=F$y[6]-H-0.13,
           hjust=.5,vjust=1,size=2.3,color="#A5751D",label="competitors")+
  scale_x_continuous(breaks=seq(0,70000,10000),
                     labels=function(x) format(x/1000,trim=TRUE),
                     expand=expansion(mult=c(.004,.24)))+
  scale_y_continuous(breaks=F$y,labels=F$stage,
                     limits=c(min(F$y)-0.95,max(F$y)+0.55),expand=expansion(mult=c(0,0)))+
  coord_cartesian(clip="off")+
  labs(title="Construction of the analysis set by successive filtering of candidate peaks",
       subtitle="Bar length is the number of peaks surviving that stage; the shaded notch between two bars is what the step removed",
       x="peaks (thousands)",y=NULL,
       caption=paste0(
"Filters act on the competitor pool, but two of the five steps remove canonical peaks rather than competitors: a gene that runs out of\n",
"competitors is dropped and its canonical peak leaves with it, so the canonical count falls from ",lab(F$n_canonical[1])," to ",
lab(F$n_canonical[4])," at step 4 and to ",lab(F$n_canonical[6])," at step 6.\n",
"A peak is APA-annotated when the nearest same-strand PolyASite 2.0 cluster or PolyA_DB v4 site lies within 500 nt; a peak is in a reference\n",
"terminal exon when its 500 nt upstream window overlaps a GENCODE v49 last exon by at least 1 nt. Counts are reconstructed from those\n",
"annotation caches by trace_analysis_set.py and asserted equal to the in_analysis_set column of cpa_peaks_final.parquet."))+
  theme_classic(base_size=BASE)+
  theme(text=element_text(color=INK),
        axis.text.x=element_text(color=INK,size=BASE-1),
        axis.text.y=element_text(color=INK,size=BASE-0.5,hjust=1),
        axis.ticks.y=element_blank(),axis.line.y=element_blank(),
        axis.line.x=element_line(color=INK,linewidth=.3),
        axis.title.x=element_text(size=BASE,margin=margin(t=4)),
        plot.title=element_text(size=BASE+2.5,face="bold",hjust=0),
        plot.subtitle=element_text(size=BASE-0.5,color=GTX,margin=margin(b=8)),
        plot.caption=element_text(size=BASE-2,color=GTX,hjust=0,lineheight=1.3),
        plot.title.position="plot", plot.caption.position="plot",
        plot.margin=margin(8,10,6,6),
        panel.grid.major.x=element_line(color="#ECEEF1",linewidth=.25))

ggsave("figure_analysis_set_funnel.pdf",p,width=9.0,height=4.5,device=grDevices::cairo_pdf)
ggsave("figure_analysis_set_funnel.png",p,width=9.0,height=4.5,dpi=300)

e("[design] vertical attrition funnel: one left-aligned bar per stage, length = peaks surviving,")
e("         shared 0..",lab(TOT)," count axis. Not a waterfall (no peeled slices) and not a Sankey (no ribbons).")
e("[stages]")
for(i in seq_len(nrow(F)))
  e(sprintf("   %d  %-48s %8s peaks  %6s genes",i,F$stage[i],lab(F$n_peaks[i]),lab(F$n_genes[i])))
e("[removed labels drawn in the gaps]")
for(i in seq_len(nrow(L))) e("  ",L$txt[i])
e("[final]  bar 6 is a single stacked bar split by class:",lab(F$n_canonical[6]),
  "real 3' ends (",REAL,") +",lab(F$n_competitor[6]),"competitors (",COMP,") =",lab(F$n_peaks[6]))
e("[text]   the two canonical-removing steps are stated in the caption, not drawn as tracks")
e("[out]    figure_analysis_set_funnel.pdf  figure_analysis_set_funnel.png")
```
<!-- END AUTO:2.4:code -->

### Figure 2.5: Competitors carry the same sequence elements as constitutive sites

<!-- BEGIN AUTO:2.5:image -->
![Figure 2.5](figures/fig2_5.png)
<!-- END AUTO:2.5:image -->

**Shows**
- **a.** Spread between the three models' peak centres for 43,547 competitors.
- **b.** APARENT2 rank percentile of the constitutive site among its gene's competitors, by competitor count (4,332 genes).
- **c.** PAS hexamers per site from -100 to +100 nt, real against shuffled.
- **D, E.** APARENT2 ROC and precision-recall for constitutive sites against competitors.
- **f.** As C for PAS hexamers, UGUA, U and GU/UG, each on its own scale.

**Why:** the competitors were *defined* by the three models, so they must be characterised with evidence those models did not generate. APARENT2 was trained on reporter cleavage data, and the shuffled-sequence null needs no model at all.

**Key numbers**
- The three models place a competitor within a median of 5 nt of each other; under 50 nt for 80.4%.
- APARENT2 AUROC 0.478 (95% CI 0.470-0.485), AUPRC 0.134 against a chance value of 0.134. It does not separate the classes.
- Mean within-gene rank percentile of the constitutive site 0.477 against 0.5 expected, and flat in the number of competitors.
- PAS hexamer density peaks 23 nt upstream in both classes, at about 8x shuffled.
- A hexamer lies in the 50 nt upstream of 96.1% of constitutive sites and 98.8% of competitors (37.9% and 35.3% after shuffling).
- U-rich and GU-rich downstream elements: same positions and densities in both classes.
- UGUA is the only element that differs: 0.84 per constitutive site against 0.68 per competitor. This matters because the Chapter 3 network responds to UGUA.

**How**
- PAS hexamers = AATAAA plus its 12 most frequent single-substitution variants, listed at `featurize_shortlist_cpa.py:52-54`. UGUA = TGTA, U-rich = T, GU-rich = GT or TG.
- Null: 20 dinucleotide-preserving (Altschul-Erickson) shuffles per sequence. Profiles smoothed with a 9 nt moving average.
- AUROC from the Mann-Whitney U statistic, AUPRC as average precision, 95% CIs from 1,000 bootstrap resamples of genes (rng seed 0). Rank-percentile ties counted as one half.
- Densities: `PUB/scripts/featurize_shortlist_cpa.py --shortlist shortlist_crossfit_windows.parquet --peaks cpa_peaks_final.parquet --reference_fasta ../train/data/hg38.fa --split test --contested_only --n_shuffle 20 --out shortlist_cpa` writes `shortlist_cpa.parquet` and `shortlist_cpa_density.tsv`.
- Statistics: `PUB/scripts/figure2_stats.py --shortlist shortlist_crossfit_windows.parquet --cpa shortlist_cpa.parquet --peaks cpa_peaks_final.parquet --cpa_bg_auroc 0.991 --boot 1000 --boot_pr 200 --out .` writes `fig2_dispersion.tsv` (a), `fig2_rank_binned.tsv` (b), `fig2_roc.tsv` (d), `fig2_pr.tsv` (e), `fig2_aparent.tsv`, `fig2_counts.tsv`.
- Plot: `Rscript PUB/scripts/make_figure2.R` writes `figure2.{pdf,png}`, 18 x 11 in (thesis image12). Panels c and f use the 9-point moving average.
- Legend and caveats: `PUB/THESIS_figure2_ccpa_grammar.txt`.

**OPEN:** panels d and e use 5,898 constitutive and 38,186 competitor sites, every site with an APARENT2 score, not the 4,847 / 43,547 analysis set.

<!-- BEGIN AUTO:2.5:code -->
**Code.** `scripts/make_figure2.R`, 233 lines, sha256 `2722d546811d`. Reproduced verbatim; also in `code/make_figure2.R`.

```r
#!/usr/bin/env Rscript
# make_figure2.R — Figure 2: competitor CPA (cCPA) sites are sequence-competent
# and CPA grammar cannot distinguish them from canonical 3' ends.
#
# Panel order, reading left to right, top to bottom:
#   A  inter-model peak convergence
#   B  APARENT2 within-gene rank, by contest size
#   C  PAS density, real vs dinucleotide null
#   D  APARENT2 ROC
#   E  APARENT2 precision-recall
#   F  positional motif density
#   Rscript make_figure2.R [indir]
#
# Panel subtitles are gone: descriptions and n values belong in the figure
# legend text, not stacked above every panel. Counts print to the console after
# the run so the caption can quote them.
#
# What this figure does and does not show
#   Every panel is descriptive sequence composition. That the sites look like
#   real 3' ends is expected — they were selected by sequence models for scoring
#   at or above the gene's own canonical site. Sequence competence here is a
#   property of construction, not independent evidence of use.
#
# Panel B is neither an ROC nor a histogram of per-gene ranks. A gene with n
# cCPA sites can only produce ranks on a lattice of n+1 values, so a raw
# histogram spikes at 0.5 and 1.0 for arithmetic reasons. Binning by contest
# size shows the statistic sits near 0.5 at every size.
#
# E matters more than D: canonical sites are ~13% of the set, so the no-skill
# AUPRC is 0.134, not 0.5.

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
  library(readr); library(scales)
})
args <- commandArgs(trailingOnly=TRUE)
INDIR <- if(length(args)>=1) args[1] else "."
have <- function(f){p<-file.path(INDIR,f); if(file.exists(p)) p else NA_character_}

INK<-"#1A1A1A"; BASE<-16
CANON<-"#0072B2"; CC<-"#E69F00"
CCLAB <- "cCPA site"

relab <- function(x) dplyr::recode(x,
                                   `Competitor` = CCLAB,
                                   `Model false positive` = CCLAB,
                                   .default = x)

## title flush to the panel; legends at the bottom so nothing sits between the
## panel letter and the plotting area
th <- theme_classic(base_size=BASE)+
  theme(text=element_text(color=INK),
        axis.text=element_text(color=INK,size=BASE-3),
        axis.title=element_text(size=BASE-2),
        axis.line=element_line(color=INK,linewidth=0.5),
        plot.title=element_text(size=BASE+2,face="bold",hjust=0,
                                margin=margin(b=1)),
        plot.title.position="plot",
        plot.subtitle=element_blank(),
        legend.position="bottom",
        legend.title=element_blank(),
        legend.text=element_text(size=BASE-4),
        legend.margin=margin(t=-4,b=0),
        legend.box.margin=margin(t=-2),
        legend.key.size=unit(0.8,"lines"),
        strip.background=element_blank(),
        strip.text=element_text(face="bold",size=BASE-4,
                                margin=margin(t=0,b=2)),
        plot.margin=margin(2,8,3,6))
ph <- function(t,m="awaiting data") ggplot()+annotate("text",0,0,label=m,size=4,color="#8A8F98")+
  lims(x=c(-1,1),y=c(-1,1))+labs(title=t)+th+
  theme(axis.line=element_blank(),axis.text=element_blank(),
        axis.ticks=element_blank(),axis.title=element_blank())

## fig2_counts.tsv is keyed by the OLD panel letters from figure2_stats.py:
##   A dispersion, B motif density, C rank, D pas-null
## Map them onto the new positions so the printed caption counts are right.
OLD2NEW <- c(A="A", C="B", D="C", B="F")
NC <- list()
fn <- have("fig2_counts.tsv")
if(!is.na(fn)){
  nc <- read_tsv(fn, show_col_types=FALSE)
  for(i in seq_len(nrow(nc))){
    k <- OLD2NEW[[nc$panel[i]]]
    if(!is.null(k)) NC[[k]] <- list(n=nc$n[i], unit=nc$unit[i])
  }
}

AD <- NULL
fap <- have("fig2_aparent.tsv")
if(!is.na(fap)) AD <- read_tsv(fap, show_col_types=FALSE)
getm <- function(k){
  if(is.null(AD)) return(NULL)
  r <- AD[AD$kind==k,]
  if(nrow(r)==0) NULL else as.list(r[1,])
}

## ---- A: inter-model peak convergence ----
fa <- have("fig2_dispersion.tsv")
if(!is.na(fa)){
  d <- read_tsv(fa,show_col_types=FALSE)
  med <- median(d$spread_nt); f50 <- mean(d$spread_nt<=50)
  XMAX <- 60; ntail <- mean(d$spread_nt > XMAX)
  pA <- ggplot(d, aes(spread_nt))+
    geom_histogram(binwidth=2,fill=CC,color=NA,boundary=0)+
    geom_vline(xintercept=med,linetype=2,color=INK,linewidth=0.5)+
    annotate("text",x=med+2,y=Inf,vjust=1.5,hjust=0,size=3.2,color=INK,
             label=paste0("median ",med," nt"))+
    annotate("text",x=XMAX,y=Inf,vjust=3.2,hjust=1,size=3.0,color=INK,
             label=paste0(percent(f50,accuracy=0.1)," within 50 nt"))+
    annotate("text",x=XMAX,y=Inf,vjust=5.0,hjust=1,size=3.0,color=INK,
             label=paste0(percent(ntail,accuracy=0.1)," beyond ",XMAX," nt"))+
    coord_cartesian(xlim=c(-1,XMAX+1))+
    scale_y_continuous(expand=expansion(mult=c(0,0.20)))+
    labs(title="A",x="Spread across the three model peaks (nt)",y="cCPA sites")+th
} else pA <- ph("A","inter-model spread")

## ---- B: within-gene rank by contest size ----
fcb <- have("fig2_rank_binned.tsv")
if(!is.na(fcb)){
  d <- read_tsv(fcb,show_col_types=FALSE) |> arrange(lo) |>
    mutate(bin=factor(bin, levels=bin))
  w <- getm("within_gene_rank")
  rlab <- if(is.null(w)) "" else sprintf("overall %.3f [%.3f, %.3f]", w$value, w$lo, w$hi)
  pB <- ggplot(d, aes(bin, mean))+
    geom_hline(yintercept=0.5, linetype=2, color=INK, linewidth=0.5)+
    geom_errorbar(aes(ymin=ci_lo, ymax=ci_hi), width=0.16, linewidth=0.6, color=CC)+
    geom_point(size=2.6, color=CC)+
    geom_text(aes(label=comma(n_genes), y=ci_hi), vjust=-0.9, size=2.6, color="#555555")+
    annotate("text", x=-Inf, y=0.5, vjust=-0.6, hjust=-0.10, size=2.9, color=INK,
             label="no preference")+
    annotate("text", x=Inf, y=-Inf, vjust=-0.8, hjust=1.05, size=2.9, color=INK,
             label=rlab)+
    coord_cartesian(ylim=c(0.30,0.66), clip="off")+
    scale_y_continuous(breaks=seq(0.3,0.65,0.1))+
    labs(title="B",x="cCPA sites in the gene", y="Rank of canonical site")+th
} else pB <- ph("B","within-gene rank")

## ---- C: PAS density, real vs dinucleotide null ----
fd <- have("shortlist_cpa_density.tsv")
if(is.na(fd)) fd <- have("fig2_density.tsv")
if(!is.na(fd)){
  d <- read_tsv(fd,show_col_types=FALSE) |>
    filter(motif %in% c("PAS","PAS (dinuc null)")) |>
    mutate(src=ifelse(motif=="PAS","Real","Dinucleotide null"),
           cls=relab(cls),
           cls=factor(cls,levels=c("Canonical",CCLAB)),
           src=factor(src,levels=c("Real","Dinucleotide null")))
  d <- d |> group_by(cls,src) |> arrange(pos) |>
    mutate(density=stats::filter(density,rep(1/9,9),sides=2) |> as.numeric()) |>
    ungroup() |> filter(!is.na(density))
  pC <- ggplot(d,aes(pos,density,color=cls,linetype=src))+
    geom_vline(xintercept=0,linetype=3,color="#B0B4BA",linewidth=0.4)+
    geom_line(linewidth=0.8)+
    scale_color_manual(values=setNames(c(CANON,CC), c("Canonical",CCLAB)),name=NULL)+
    scale_linetype_manual(values=c(Real="solid",`Dinucleotide null`="22"),name=NULL)+
    guides(color=guide_legend(order=1,nrow=1),linetype=guide_legend(order=2,nrow=1))+
    labs(title="C",x="Position relative to site (nt)",y="PAS density per peak")+th
} else pC <- ph("C","dinucleotide null")

## ---- D: ROC ----
fe <- have("fig2_roc.tsv")
if(!is.na(fe)){
  d <- read_tsv(fe,show_col_types=FALSE)
  ro <- getm("pooled_auroc")
  lab <- if(is.null(ro)) "" else
    sprintf("AUROC %.3f\n[%.3f, %.3f]\nchance 0.500", ro$value, ro$lo, ro$hi)
  pD <- ggplot(d, aes(fpr, tpr))+
    geom_abline(slope=1,intercept=0,linetype=2,color=INK,linewidth=0.5)+
    geom_line(color=CC,linewidth=1.0)+
    annotate("text", x=0.97, y=0.05, hjust=1, vjust=0, size=3.0, color=INK,
             lineheight=1.05, label=lab)+
    coord_equal(xlim=c(0,1), ylim=c(0,1))+
    scale_x_continuous(breaks=seq(0,1,0.25))+
    scale_y_continuous(breaks=seq(0,1,0.25))+
    labs(title="D",x="False positive rate", y="True positive rate")+th
} else pD <- ph("D","ROC")

## ---- E: precision-recall ----
ff <- have("fig2_pr.tsv")
if(!is.na(ff)){
  d <- read_tsv(ff,show_col_types=FALSE)
  bl <- if("baseline" %in% names(d)) d$baseline[1] else NA_real_
  pr <- getm("pooled_auprc")
  lab <- if(is.null(pr)) "" else
    sprintf("AUPRC %.3f\n[%.3f, %.3f]\nchance %.3f", pr$value, pr$lo, pr$hi, pr$baseline)
  pE <- ggplot(d, aes(recall, precision))+
    { if(is.finite(bl)) geom_hline(yintercept=bl,linetype=2,color=INK,linewidth=0.5) } +
    geom_line(color=CC,linewidth=1.0)+
    annotate("text", x=0.97, y=0.97, hjust=1, vjust=1, size=3.0, color=INK,
             lineheight=1.05, label=lab)+
    { if(is.finite(bl)) annotate("text", x=0.02, y=bl, hjust=0, vjust=-0.5,
                                 size=2.8, color=INK, label="chance") } +
    coord_equal(xlim=c(0,1), ylim=c(0,1))+
    scale_x_continuous(breaks=seq(0,1,0.25))+
    scale_y_continuous(breaks=seq(0,1,0.25))+
    labs(title="E",x="Recall", y="Precision")+th
} else pE <- ph("E","precision-recall")

## ---- F: positional motif density ----
fb <- have("shortlist_cpa_density.tsv")
if(is.na(fb)) fb <- have("fig2_density.tsv")
if(!is.na(fb)){
  d <- read_tsv(fb,show_col_types=FALSE) |>
    filter(motif %in% c("PAS","UGUA","U-rich","GU-rich")) |>
    mutate(motif=factor(motif,levels=c("PAS","UGUA","U-rich","GU-rich")),
           cls=relab(cls),
           cls=factor(cls,levels=c("Canonical",CCLAB)))
  d <- d |> group_by(cls,motif) |> arrange(pos) |>
    mutate(density=stats::filter(density,rep(1/9,9),sides=2) |> as.numeric()) |>
    ungroup() |> filter(!is.na(density))
  pF <- ggplot(d,aes(pos,density,color=cls))+
    geom_vline(xintercept=0,linetype=3,color="#B0B4BA",linewidth=0.4)+
    geom_line(linewidth=0.8)+
    facet_wrap(~motif,nrow=2,scales="free_y")+
    scale_color_manual(values=setNames(c(CANON,CC), c("Canonical",CCLAB)))+
    labs(title="F",x="Position relative to site (nt)",y="Motif density per peak")+th
} else pF <- ph("F","motif density")

## ---- assemble: 3 x 2, alphabetical left to right, top to bottom ----
fig <- (pA | pB | pC) / (pD | pE | pF)
ggsave(file.path(INDIR,"figure2.pdf"),fig,width=18,height=11,device=cairo_pdf)
ggsave(file.path(INDIR,"figure2.png"),fig,width=18,height=11,dpi=400)
cat("[out] figure2.pdf, figure2.png\n")

## counts for the caption, since they are no longer on the panels
if(length(NC)){
  cat("\npanel n (for the figure legend):\n")
  for(p in sort(names(NC)))
    cat(sprintf("  %s  %s %s\n", p, comma(NC[[p]]$n), NC[[p]]$unit))
}
cat("  D  38,186 cCPA + 5,898 canonical (APARENT2 available)\n")
cat("  E  38,186 cCPA + 5,898 canonical (APARENT2 available)\n")
```
<!-- END AUTO:2.5:code -->

### Figure 2.6: Reporter assay design (schematic)

<!-- BEGIN AUTO:2.6:image -->
![Figure 2.6](figures/fig2_6a.png)

![Figure 2.6](figures/fig2_6b.png)
<!-- END AUTO:2.6:image -->

**Shows** the candidate 500 nt window cloned downstream of eGFP and upstream of the vector's own poly(A) site; RT from an anchored oligo(dT) primer plus PCR with a forward primer in the reporter gives about 500 bp if the cloned site is used and about 1 kb if the vector's site is used; then the workflow of clone, transfect, harvest RNA at 24 h, RT-PCR, gel, Sanger.

**How made:** hand-drawn, embedded in `PUB/SP_thesis_0826.docx` only (images image18, image19). No source file found. **OPEN.**

<!-- BEGIN AUTO:2.6:code -->
*No code: this figure is a drawing, a photograph of a gel, or a sequencing trace panel. See the figure's entry above for what is known about its origin.*
<!-- END AUTO:2.6:code -->

### Figure 2.7: Gels of the reporter RT-PCR products

<!-- BEGIN AUTO:2.7:image -->
![Figure 2.7](figures/fig2_7_top.png)

![Figure 2.7](figures/fig2_7_bottom.png)
<!-- END AUTO:2.7:image -->

**Shows**
- **Top gel** (image13): ladder, untransfected cells, negative control, positive control, and six competitor constructs named by gene: *PREX2*, *AFF2*, *BARX2*, *CTNNA2*, *ATP9B*, *TRAF3IP1* (constructs CR_1 to CR_6).
- **Bottom gel** (image14): 24 constructs named by plasmid number and grouped by source: positive control, negative control, predicted competitors, U1 AMO sites, immune intronic pA sites, leukaemia intronic pA sites, synthetic designs. The 24 plasmid IDs on the gel match the appendix table exactly.

**Result as stated:** all six top-gel competitors give a predominant product of about 500 bp, the size of the positive control, so the cloned site is used. Two of the six also give the roughly 1 kb vector product, which is the only product of the negative control.

**OPEN**
- Which two constructs give both products: the legend says *AFF2* and *BARX2*, the gel reads more like *BARX2* and *CTNNA2*.
- How many of the bottom-gel competitor constructs gave the short product (by eye, seven of eight).
- The assay is not quantified anywhere.
- No raw gel image files were found on the cluster; only the docx-embedded images exist.
- Cell line inconsistency: the figure legends say HEK293, the Methods say HEK293T and HeLa.

<!-- BEGIN AUTO:2.7:code -->
*No code: this figure is a drawing, a photograph of a gel, or a sequencing trace panel. See the figure's entry above for what is known about its origin.*
<!-- END AUTO:2.7:code -->

### Figure 2.8: Sanger sequencing of the *PREX2* and *AFF2* products

<!-- BEGIN AUTO:2.8:image -->
![Figure 2.8](figures/fig2_8a.png)

![Figure 2.8](figures/fig2_8b.png)
<!-- END AUTO:2.8:image -->

**Shows**, per construct: the position of the cloned site within the gene (exons drawn), the Phred quality at each base with a line at Q20, and the alignment of the read to hg38 with the AAUAAA marked. The transcripts end in a poly(A) tail beginning downstream of the AAUAAA of the cloned sequence.

**Genes:** *PREX2* chr8:67,952,046-68,237,032 (+); *AFF2* chrX:148,500,617-149,000,663 (+).

**How made:** **OPEN.** No plotting code was found, and no `.ab1` or `.seq` trace files exist on the cluster. Images image15 and image16 in the docx are the only copies. To remake this panel the traces must be recovered from the lab records.

<!-- BEGIN AUTO:2.8:code -->
*No code: this figure is a drawing, a photograph of a gel, or a sequencing trace panel. See the figure's entry above for what is known about its origin.*
<!-- END AUTO:2.8:code -->

### Table 2.1: Competitor sites tested in the reporter assay

- 18 rows: gene, hg38 coordinates of the 500 nt window, strand, score.
- **Source resolved:** the 18 rows match `CLEAN/train/data/Cryptic_CPA_sites.hg38.tested_experimental.bed` exactly, itself a subset of `/scratch/spour98/remake_model/updated_HumanGeneEnds/files/Cryptic_CPA_sites.hg38.bed` (40,537 sites). Column 5 is the score, also carried as `paper_prob` in `CLEAN/train/cryptic_scored.tsv`.
- **Which model produced "Score":** the earlier-generation baseline random forest of Shkurin, Pour and Hughes, "Known sequence features can explain half of all human gene ends" (per `updated_HumanGeneEnds/README.md:83`), not the RF/LR/CNN consensus of this chapter. This settles the [VERIFY] note in the table caption.
- **OPEN:** four sites in the table (*SHLD2*, *MYLK*, *KIF13A*, *TBL1X*) have no construct in the appendix and appear on neither gel. Either drop them or say they were designed but not assayed.

### Reporter assay: plasmids, inserts, primers, selection rationale

Everything recoverable has been extracted from the thesis appendix (`thesis_ch2/src/thesis.md`, appendix table) into two machine-readable files, so the sequences never need to be retyped:

- `thesis_figures_readme/reporter_constructs.tsv` - one row per construct and per primer: construct name, plasmid number, category, hg38 window, the original source label including every selection metric, insert length, full insert sequence.
- `thesis_figures_readme/reporter_constructs.fa` - the same inserts as FASTA, headed `name|plasmid|window`.

**Primers.** Two are recorded, both in the TSV: a forward primer in eGFP and a reverse/RT anchored oligo(dT) primer. Each carries an Illumina Nextera adapter prefix (Read 1 and Read 2 respectively) ahead of the priming sequence, which means these constructs were also designed to be sequenced on an Illumina platform, not only gel-read. **OPEN:** no separate genomic-DNA reverse primer (used for the gDNA control PCR in the Methods) is recorded, and there is no order sheet.

**Backbone.** pTH13195, an eGFP reporter, obtained from IDT per the text. Inserts were cloned between FseI and EcoRI. Some constructs were cloned in-house, others synthesised by Bio Basic. **OPEN:** no backbone sequence or plasmid map exists on the cluster, and no record of which constructs were synthesised rather than cloned.

**Construct inventory, 30 inserts in 7 groups** (full detail in the TSV)

| Group | Plasmids | n | Insert length | Why these were chosen |
|---|---|---|---|---|
| Positive control | pTH17606 (ACTB), pTH17607 (GAPDH) | 2 | 484 nt | Annotated, heavily used 3′ ends; set the "cleaved at the cloned site" band size |
| Negative control | pTH17608 (CTCF region), pTH17609 (GAPDH region) | 2 | 484 nt | Genomic windows with no pA site; set the "vector site only" band |
| Predicted competitors, bottom gel | pTH17612 MYDGF, 17613 ZSWIM5, 17615 FAM9B, 17616 CDH13, 17617 RHOJ, 17618 RYR1, 17620 CACNA2D3, 17621 DCAF17 | 8 | 484 nt | Intronic windows the baseline RF scored as highly as the gene's real end; scores are the Table 2.1 column (0.694 to 0.775) |
| Predicted competitors, top gel | CR_1 PREX2, CR_2 AFF2, CR_3 BARX2, CR_4 CTNNA2, CR_5 ATP9B, CR_6 TRAF3IP1 | 6 | 500-501 nt | The highest-scoring competitors in Table 2.1 (0.827 to 0.855). **OPEN: these six have no pTH number recorded** |
| U1 AMO-activated sites | pTH17623, 17625, 17626 | 3 | 484 nt | Intronic sites that appear when U1 snRNP is inhibited (Oh et al. 2017). Selected on `maxlog2fc` (5.7 to 11.7) and `maxtreat`; both values are preserved in the label column of the TSV |
| Immune intronic pA sites | pTH17628 C1orf228, 17629 NF1, 17630 ANKRD28, 17631 FCRL5, 17633 PRDM1 | 5 | 484 nt | Intronic pA sites used in immune cells / CLL (Singh and Lee 2018). Selected on a composite `score` (23.3 to 37.5) built from max and mean expression, log2FC, fraction of samples on, and CV; all retained in the label column |
| Leukaemia intronic pA sites | pTH17636 RAPGEF3, 17637 MAP4K4 | 2 | 484 nt | As above but with CLL versus normal B cell and versus tissue contrasts (`log2FC_NormB`, `log2FC_Tissue`), composite score 22.8 to 23.7 |
| Synthetic designs | pTH17640 (design_1), pTH17643 (design_4) | 2 | 484 nt | Designed sequences, no genomic coordinates. **OPEN: the design criteria are not recorded anywhere** |

- The selection-metric strings in the label column are the **only** surviving record of why each non-control construct was picked. **OPEN:** the script that computed `maxlog2fc`, `maxCLL`, `frac_on`, `cv` and `score` and selected these inserts was not found; without it the selection cannot be reproduced, only reported.
- **OPEN:** plasmid numbers pTH17610, 17611, 17614, 17619, 17622, 17624, 17627, 17632, 17634, 17635, 17638, 17639, 17641, 17642 appear nowhere. The numbering implies constructs that were planned and dropped, or cloned and not recorded.

**Protocols** (written up in `chapter2.md` section 2.4.9; reagents and catalogue numbers in Table 2.2 there)

- *Cloning.* Insert and vector digested with FseI and EcoRI; vector dephosphorylated with Antarctic phosphatase; ligated with T4 DNA ligase (all New England Biolabs). Transformed into Subcloning Efficiency DH5-alpha.
- *Transfection.* HEK293T and HeLa, Lipofectamine 2000. Plasmid DNA and Lipofectamine diluted separately in Opti-MEM at 37 C, combined, held 5 min at room temperature, added to cells. Reporter expression induced with doxycycline. **OPEN: dose and induction time.** Harvest after one day.
- *Nucleic acid extraction.* AllPrep DNA/RNA Mini Kit (QIAGEN), giving RNA and genomic DNA from the same well.
- *Reverse transcription.* SuperScript III with an anchored oligo(dT) primer. RNA 8.5 uL + primer 0.5 uL at 100 uM + dNTPs 1 uL at 10 mM, 80 C for 5 min, brought to 42 C; add 10 uL of 4 uL 5x First-Strand Buffer, 1 uL 0.1 M DTT, 1 uL RNaseOUT, 1 uL SuperScript III, 3 uL water; then 42 C, 48 C and 55 C for 15 min each and 80 C for 5 min.
- *PCR.* Phusion High-Fidelity polymerase in 10 uL Phusion GC Buffer with 0.5 M betaine and 10% DMSO; 30 cycles of 98 C, 68 C, 72 C. **OPEN: the time at each step.** Forward primer upstream of the cloned sequence. The genomic-DNA control uses the same forward primer with a reverse primer downstream of the vector's pA site.
- *Readout.* Agarose gel (UltraPure agarose, ethidium bromide), bands excised and purified with the QIAquick Gel Extraction Kit, then Sanger sequenced.

**What is missing for the wet-lab package** (searched `/scratch/spour98` and `/home/spour98`; none of these exist on the cluster and they are presumably in lab records or an ELN)
- plasmid maps or full backbone sequence (`.gb`, `.dna`)
- Sanger traces (`.ab1`)
- raw gel images
- primer order sheets
- a written protocol document, as opposed to the Methods prose
- the insert-selection script
### Figure 2.9: Overlap of competitor sites with published sets of poly(A) sites

<!-- BEGIN AUTO:2.9:image -->
![Figure 2.9](figures/fig2_9.png)
<!-- END AUTO:2.9:image -->

This is the heatmap figure the committee asked about, so it is documented in more detail than the rest.

**Shows**, over five sets of sites restricted to the 334 protein-coding genes in which *every* set has at least one site (356 constitutive, 4,027 competitors, 573 immune/leukaemia 3′-seq sites, 1,306 U1 AMO sites, 552 PCF11-dependent sites):
- **a.** Cumulative distribution of the distance from each site of one set to the nearest site of each other set in the same gene, log10 axis, dashed line at 50 nt.
- **b.** The heatmap: fraction of the sites in each row lying within 50 nt of a site in each column.
- **c.** Percentage of sites with AAUAAA or AUUAAA in the 60 nt upstream, against the same positions displaced by 5 kb, with the ratio printed.
- **d.** RF (calibrated probability) and CNN (percentile among all 70,723 peaks) scores at the sites of each set, taken as the best-scoring window containing the site.

**Why:** it asks whether the competitors are simply the cryptic or intronic pA sites that other labs have already reported. Mostly they are not.

**Key numbers**
- Pairwise agreement between published datasets is itself only 16 to 24% within these genes, which sets the ceiling for what agreement with the competitors could look like.
- AAUAAA or AUUAAA in the 60 nt upstream: 95.5% of competitors, 86.0% of constitutive sites, 80.1% of immune/leukaemia sites, 66.3% of PCF11-dependent sites, 13.9% of U1 AMO sites.
- From the older unrestricted version of this analysis, quoted in section 2.2.7: of 43,547 competitors, 370 lie within 50 nt of any published site: 273 of 39,970 U1 AMO sites (0.68%, 3.9x chance, P < 0.005), 83 of 39,972 control-morpholino sites (0.21%, 1.2x, P = 0.065), 19 of 5,451 immune/leukaemia sites (0.35%, 2.1x, P = 0.015). More than 99% of competitors match nothing published.

**Matching rules**
- A competitor and a published site count as the same site within 50 nt; same strand where the data are stranded, either strand for the unstranded morpholino data.
- Chance expectation: mean over 200 displacements of the competitors by a distance drawn uniformly between 10 and 100 kb. The smallest P value this procedure can return is 0.005. No multiple-testing correction.
- Published signal was merged into peaks within 25 nt; for the morpholino data the 40,000 strongest peaks per condition were kept.
- hg19 coordinates were lifted to hg38 with `liftOver -minMatch=0.95`.

**Published datasets used, with accessions and local files**

| Set in the figure | Publication | Accession | Local files | Size |
|---|---|---|---|---|
| Immune (part of "3′-seq atlas") | Singh et al. 2018 | GSE111310 | `PUB/external/geo/GSE111310/GSE111310_ipa_tpm.txt.gz` | 370 KB |
| Leukaemia (part of "3′-seq atlas") | Lee et al. 2018 | GSE111793 | `PUB/external/geo/GSE111793/GSE111793_ipa_tpm.txt.gz` | 357 KB |
| U1 AMO, treatment | Oh et al. 2017 | GSE103252 | `.../GSE103252_U1AMO_15nmol_odT_selection.polyA_sites.bw` | 2.0 MB |
| U1 AMO, control morpholino | Oh et al. 2017 | GSE103252 | `.../GSE103252_cAMO_15nmol_odT_selection.polyA_sites.bw` | 1.9 MB |
| PCF11-dependent | Kamieniarz-Gdula et al. 2019 (Proudfoot lab) | GSE123105 | `.../GSE123105/3pmrnaseq/GSM3536480_LUCsi_mergedReplicates_{F,R}.bw` (siLUC control) and `GSM3536484_PCFsi_mergedReplicates_{F,R}.bw` (siPCF11) | ~0.5-0.6 MB each |

- The Singh and Lee sets are **pooled** into the one "3′-seq atlas" row. Per `datasets.yaml` the malignant samples are myeloma (Singh) and CLL (Lee), plus a shared cell-line panel (SRP029953), so the chapter's label "immune cells and leukaemia" is incomplete. Cite both papers.
- Oh et al. data are HeLa and **unstranded** (strand recorded as ".").
- PCF11 sites are defined as enriched at least 3-fold in the siLUC control over siPCF11, i.e. **sites lost on PCF11 knockdown**. HeLa nuclear QuantSeq REV. Wang 2019 is cited only for the direction of the effect. The raw archive `GSE123105_RAW.tar` is 4.8 GB and should stay out of any archive.
- LiftOver inputs: chain `PUB/external/chain/hg19ToHg38.over.chain.gz` (228 KB), binary `PUB/external/bin/liftOver` (24.8 MB).
- Gene set for the comparison: `updated_HumanGeneEnds/reanal/score_genome/pc_genes.bed` (GENCODE v43, 693 KB).
- Download record: `PUB/external/manifest.tsv` and `verify_report.tsv`; fetch scripts `PUB/scripts/00_download_and_liftover.sh`, `00b_verify_and_repair.sh`, `01_fetch_gse135140.sh`, and `PUB/fetch_gsm.sh` for GSE123105.

**Downloaded but not used in this figure** (keep out of the archive unless asked): So et al. 2019 GSE135140 (2.3 GB), GSE125397, GSE250342 (81 MB), GSE262383 (47 MB), GSE275154 (86 MB), plus the earlier July-pipeline intermediates `PUB/external/lifted/{immune,cll}_hg38.tsv` and `PUB/external/peaks/oh2017_*`.

**Rebuild, all run from `PUB`** (also written up in `PUB/THESIS_figure_overlap_shared.txt`)
1. `python3 hiconf_sets.py --config hiconf.yaml --peaks cpa_peaks_final.parquet --outdir hiconf_out`
   - Settings: tolerance 50 nt, min_reads 2, enrich_ratio 3.0, merge_within 25 nt; liftOver called internally at `-minMatch=0.95`.
   - Writes `hiconf_out/sites/{atlas_3seq,oh_u1amo,pcf11_dep,singh_ipa,lee_ipa}.tsv`, 0.1 to 0.5 MB each. **These five files are the "bed files used for the overlaps".**
   - **OPEN:** no log, and the script was edited on 20 Aug after its outputs were written on 17 Aug, so the current code may not reproduce them exactly. Re-run and compare before relying on it.
2. `python3 overlap_model_scores.py` (log `overlap_model_scores.log`) writes `overlap_scores_sites.tsv` (5.4 MB), `overlap_shared_genes.tsv` (the 334 genes) and `overlap_scores_summary.tsv`.
3. `python3 overlap_rf_dense.py` writes `overlap_rf_dense.tsv` (251 KB).
4. `python3 overlap_shared_recompute.py` writes `shared_distances.tsv` (a), `shared_containment.tsv` (b) and `shared_hexamer.tsv` (c).
5. `Rscript make_figure_overlap_shared.R` writes `figure_overlap_shared.{pdf,png}` (thesis image17). Notes in `PUB/figure_overlap_shared_notes.txt`.

- Competitor and constitutive inputs both come from `cpa_peaks_final.parquet` via `is_canonical`.
- Scores in panel d: RF from `CLEAN/rerank_genomewide_scored.parquet` (73,801 windows; **OPEN:** may not be the prod RF), CNN from `PUB/cnn_wg_scores.npz`.
- The older, unrestricted version of the analysis is `make_figure_overlap.R` to `figure_overlap_filtered.*`, with data from `task_overlap_accounting.py` and `scripts/03_simple_overlap.py`, notes in `figure_overlap_notes.md`. **That version is the source of the 370 / 273 / 83 / 19 numbers in the text**, so both versions are needed to support the chapter as written.

**OPEN, and both should be fixed before submission**
1. `overlap_model_scores.py:107-109` does **not** filter on `in_analysis_set`, so this figure uses all 64,140 competitors in all 6,583 genes, not the 4,847-gene analysis set. This is the discrepancy already flagged as [VERIFY] in the chapter, and the flag is correct. The 4,027 competitors in the 334 shared genes are from the unfiltered set.
2. The U1 AMO sites carry strand ".", and `has_pas()` treats any strand that is not "-" as "+". About half the U1 AMO hexamer tests are therefore likely run on the wrong strand, which probably explains the strikingly low 13.9% in panel c; the older `hiconf_out/hexamer.tsv` reports 32.5% for the same data. Re-run panel c strand-agnostically.
3. `overlap_rf_dense.tsv` has 6,813 rows where 6,814 were expected; one site is missing.
4. Steps 1 and 4 have no sbatch file and no log.

<!-- BEGIN AUTO:2.9:code -->
**Code.** `make_figure_overlap_shared.R`, 131 lines, sha256 `25bd9ddb3c2b`. Reproduced verbatim; also in `code/make_figure_overlap_shared.R`.

```r
#!/usr/bin/env Rscript
# make_figure_overlap_shared.R -- the site-overlap figure, on the genes every dataset covers,
# with what the baseline models make of each set. Reads only TSVs; computes nothing but ECDFs.
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(tidyr);library(readr)
                               library(patchwork);library(scales)})
INK<-"#1A1A1A"; NUL<-"#8A8F98"; GTX<-"#5A6068"; FAINT<-"#E3E6E9"; BASE<-7.4
PAL<-c(canonical="#08519C", model_fp="#8A8F98", atlas_3seq="#E8A33D",
       pcf11_dep="#7D3C98", oh_u1amo="#B03A2E")
LAB<-c(canonical="Canonical CPA", model_fp="Internal competitors",
       atlas_3seq="3'-seq IpA atlas", pcf11_dep="PCF11-dependent CPA", oh_u1amo="U1 AMO")
ORD<-c("canonical","model_fp","atlas_3seq","pcf11_dep","oh_u1amo")
fset<-function(x) factor(LAB[x],levels=unname(LAB[ORD]))
th<-theme_minimal(base_size=BASE,base_family="sans")+theme(
  panel.grid.minor=element_blank(),
  panel.grid.major=element_line(colour=FAINT,linewidth=.3),
  plot.title=element_text(size=BASE+1.4,face="bold",colour=INK),
  plot.subtitle=element_text(size=BASE-0.6,colour=GTX,lineheight=1.15),
  axis.text=element_text(colour=INK),axis.title=element_text(colour=GTX,size=BASE-0.4),
  strip.text=element_text(size=BASE-0.5,colour=INK,face="bold",hjust=0),
  legend.key.size=unit(8,"pt"),legend.text=element_text(size=BASE-0.8),
  legend.title=element_text(size=BASE-0.8,colour=GTX),
  plot.tag=element_text(size=BASE+3,face="bold",colour=INK))
rd<-function(f) read_tsv(f,show_col_types=FALSE,progress=FALSE)
DI<-rd("shared_distances.tsv"); CO<-rd("shared_containment.tsv")
HX<-rd("shared_hexamer.tsv");   SI<-rd("overlap_scores_sites.tsv"); RFD<-rd("overlap_rf_dense.tsv")
NG<-nrow(rd("overlap_shared_genes.tsv"))
NS<-SI%>%filter(in_shared==1)%>%count(set)
stopifnot(setequal(NS$set,ORD))
sub<-function(x) sprintf("%s  (%s sites)",LAB[x],comma(NS$n[match(x,NS$set)]))

# ---- A  nearest-neighbour distance -----------------------------------------------
Ap<-DI%>%mutate(src=factor(sub(set_a),levels=sub(ORD)),tgt=fset(set_b))%>%
  group_by(src,tgt)%>%arrange(dist,.by_group=TRUE)%>%
  mutate(f=row_number()/n())%>%ungroup()%>%
  mutate(dist=pmax(dist,1))
pA<-ggplot(Ap,aes(dist,f,colour=tgt))+
  geom_vline(xintercept=50,linetype="22",colour=NUL,linewidth=.35)+
  geom_step(linewidth=.55)+
  facet_wrap(~src,nrow=1)+
  scale_colour_manual(values=setNames(unname(PAL[ORD]),unname(LAB[ORD])),name="nearest site in")+
  scale_x_log10(breaks=c(1,1e2,1e4,1e6),labels=c("1","100","10K","1M"))+
  scale_y_continuous(labels=percent_format(accuracy=1),limits=c(0,1))+
  labs(title="A   Nearest-neighbour distance between site sets",

       x="distance to nearest site (nt, log scale)",y="cumulative fraction of source-set sites")+th+
  theme(legend.position="right",panel.spacing.x=unit(7,"pt"))

# ---- B  containment --------------------------------------------------------------
Bp<-CO%>%mutate(a=fset(set_a),b=fset(set_b))
pB<-ggplot(Bp,aes(b,a,fill=containment))+
  geom_tile(colour="white",linewidth=1.1)+
  geom_text(aes(label=sprintf("%.3f",containment),
                colour=I(ifelse(containment>.14,"white",INK))),size=2.25)+
  scale_fill_gradient(low="#F4F7FA",high="#08519C",limits=c(0,max(Bp$containment)),
                      name="containment",labels=number_format(accuracy=.01))+
  scale_x_discrete()+scale_y_discrete(limits=rev)+
  labs(title="B   Pairwise containment at 50 nt",

       x=NULL,y=NULL)+th+
  theme(panel.grid.major=element_blank(),
        axis.text.x=element_text(angle=20,hjust=1,size=BASE-1),
        axis.text.y=element_text(size=BASE-1),legend.position="right")

# ---- C  canonical PAS ------------------------------------------------------------
Cp<-HX%>%mutate(s=fset(set))%>%
  select(s,set,Observed=pct_obs,`Displaced 5 kb`=pct_rand,fold)%>%
  pivot_longer(c(Observed,`Displaced 5 kb`),names_to="w",values_to="v")%>%
  mutate(w=factor(w,levels=c("Displaced 5 kb","Observed")))
pC<-ggplot(Cp,aes(v,s,fill=w))+
  geom_col(position=position_dodge(width=.72),width=.62)+
  geom_text(data=Cp%>%filter(w=="Observed"),
            aes(label=sprintf("%.1f%%  %.1f-fold",v,fold)),
            hjust=-0.12,size=2.2,colour=INK,position=position_nudge(y=.17))+
  scale_fill_manual(values=c(`Observed`="#B03A2E",`Displaced 5 kb`=FAINT),name=NULL)+
  scale_x_continuous(limits=c(0,132),breaks=c(0,25,50,75,100),
                     labels=function(x) paste0(x,"%"))+
  scale_y_discrete(limits=rev)+
  labs(title="C   Canonical polyadenylation signal at site centres",

       x="sites carrying a canonical PAS",y=NULL)+th+
  theme(legend.position="bottom",legend.margin=margin(t=-4),
        panel.grid.major.y=element_blank())

# ---- D  what the baselines make of each set --------------------------------------
# The forest comes from the dense per-nucleotide genome scan, so every site has a score; the
# 73,801-window table used elsewhere covers only 14 to 23% of these sites.
DR<-RFD%>%transmute(set,v=rf_dense_max,model="Random forest, probability")
DC<-SI%>%filter(in_shared==1,!is.na(cnn_pct))%>%
    transmute(set,v=cnn_pct/100,model="CNN, percentile of all windows")
Dp<-bind_rows(DR,DC)%>%mutate(s=fset(set),model=factor(model,
      levels=c("Random forest, probability","CNN, percentile of all windows")))
MED<-Dp%>%group_by(model,s,set)%>%summarise(m=median(v),n=n(),.groups="drop")
pD<-ggplot(Dp,aes(v,s,fill=set,colour=set))+
  geom_violin(scale="width",width=.82,alpha=.30,linewidth=.28)+
  geom_boxplot(width=.16,outlier.shape=NA,alpha=1,linewidth=.3,fill="white",colour=INK)+
  geom_text(data=MED,aes(x=1.02,y=s,label=sprintf("%.2f",m)),hjust=0,size=2.15,colour=INK,
            inherit.aes=FALSE)+
  facet_wrap(~model,nrow=1)+
  scale_fill_manual(values=PAL,guide="none")+scale_colour_manual(values=PAL,guide="none")+
  scale_x_continuous(limits=c(0,1.14),breaks=c(0,.25,.5,.75,1))+
  scale_y_discrete(limits=rev)+
  labs(title="D   Baseline model scores at the sites of each set",

       x="score",y=NULL)+th+theme(panel.grid.major.y=element_blank())

# House style: no prose under a panel heading. The detail those subtitles carried now lives in
# figure_overlap_shared_notes.txt, and the check is an assertion rather than a habit.
stopifnot(all(sapply(list(pA,pB,pC,pD), function(g) is.null(g$labels$subtitle))))
FIG<-pA/(pB|pC)/pD+plot_layout(heights=c(1,1.05,1.05))+
  plot_annotation(
    title="Concordance of published 3' end datasets with each other and with model calls, on commonly covered genes",
    caption=paste0(
"All panels restricted to the ",comma(NG)," protein-coding genes in which ALL five sets place at least one site. Low overlap between two datasets otherwise has two\n",
"explanations, real disagreement and different gene coverage, and the unrestricted panels cannot separate them. Sites per set after restriction are in panel A.\n",
"D The forest is read from the dense per-nucleotide genome scan (score_genome/out_scores), so all ",comma(nrow(RFD))," sites carry a score; the 73,801-window table\n",
"  used elsewhere covers only 14 to 23% of them and would have compared one model on a seventh of the sites against the other on all of them. The CNN is\n",
"  leave-one-chromosome-out and so out of fold. LOGISTIC REGRESSION IS ABSENT: fig1_scoredist.tsv has LR scores but no coordinates, the per-gene parquet scans\n",
"  that produced them are gone from the tree, and no lr_model.joblib survives, so LR cannot be placed on a genomic site without retraining it.\n",
"  ON THE COMPETITOR ROW. The internal competitors were defined as sites the forest ranks ABOVE the canonical site OF THE SAME GENE, so their low marginal CNN\n",
"  score is not a contradiction of how they were selected. The selection is a WITHIN-GENE statement and panel D is a marginal one, pooled over genes, and the two\n",
"  need not agree: measured within gene on this scope, the best competitor still outscores the canonical site in 71.9% of genes under the forest, while the CNN\n",
"  reverses that to 44.6%. Over all 6,583 genes the out-of-fold CNN places the canonical site first in 57.7%. That reversal is what the CNN was trained to do,\n",
"  so neither model's verdict on this row is independent of how the row was defined; the three external sets are the honest test, and both models rank U1 AMO\n",
"  lowest of them."),
    theme=theme(plot.title=element_text(size=BASE+5,face="bold",colour=INK,family="sans",
                                        margin=margin(b=3)),
                plot.caption=element_text(size=BASE-1.8,colour=GTX,hjust=0,lineheight=1.25,
                                          family="sans")))
ggsave("figure_overlap_shared.pdf",FIG,width=11.6,height=12.2,device=cairo_pdf)
ggsave("figure_overlap_shared.png",FIG,width=11.6,height=12.2,dpi=200)
cat("[out] figure_overlap_shared.pdf figure_overlap_shared.png\n")
```
<!-- END AUTO:2.9:code -->

## CHAPTER 3: Wider sequence context, and the splicing environment upstream of a 3′ end

14 figures (3.1 to 3.14) and one table. Every figure traces to an existing script with existing inputs.

**The chapter's spine (the user's hypothesis):** a stronger 3′ splice site upstream of a candidate makes it look more like a real gene end; a stronger 5′ splice site upstream makes it look more like a competitor. Figure 3.5 is the direct test.

### Shared Chapter 3 inputs

**Network dataset: `PUB/cnn_data.npz`** (written by `PUB/cnn_data.py`)
- 48,394 windows x 2,500 nt, uint8, the analysis set.
- Sequence spans -2,000 to +500 nt around the candidate in transcript orientation. On the minus strand the genomic interval is [center-499, center+2001).
- Fields `y`, `gene`, `split`, `loco`, `gene_holdout`.
- **Fixed split:** test = chr1, 8, 15, 21 plus the seed-42 20% gene holdout; validation = chr3, 18; train = the rest. Sizes: train 2,781 genes / 27,658 candidates; validation 443 / 6,538; test 1,623 / 14,198 (1,623 real ends, 12,575 competitors).
- **LOCO:** 23 folds, one per chromosome, with 25% of the remaining genes held out inside each fold for tuning.

**Per-peak MaxEnt table: `PUB/fig3_ucube_meta.tsv`** (70,723 rows)
- Columns include `acc` (highest MaxEntScan score3 over -2,000 to -126 nt), `interv_donor` (highest score5 between that acceptor and the candidate), `u1_dw_up`, `u1_dw_dn`.
- Written by `PUB/scripts/figure3_data.py`.
- **OPEN, and the largest reproducibility gap in the chapter:** the invocation of `figure3_data.py` (`--shortlist`, `--reference_fasta`, `--u1_lookup`, `--me5_lookup`) was never recorded, so the MaxEnt half of the chapter cannot be rebuilt from scratch. Treat `fig3_ucube_meta.tsv` as a primary input and archive it. Related: `fig3_new_positional_features.tsv` defines the 70,723-row order used by the caches; `shortlist_feat_all.parquet` holds the baseline `rf_prob`.

**Splice-site and U1 scoring**
- MaxEntScan through **maxentpy 0.0.3**. score3 computed live on 23-mers with AG at 0-based positions 18-19; score5 read from a dense 4^9 lookup `CLEAN/train/splice_train/data/me5_lookup.npy`, documented as byte-identical to maxentpy score5.
- U1 duplex -dG from a 4^9 lookup `CLEAN/train/splice_train/data/u1_lookup.npy`, built with ViennaRNA 2.6.4 `duplexfold` against ACUUACCUG. **OPEN:** its generator `build_u1_lookup.py` is cited but missing.
- Thresholds used throughout: score3 >= 6.5, score5 >= 8, U1 -dG >= 4 (and >= 7 for a strict set).

**Matched intronic decoys: `PUB/decoy_set.npz`** (built by `PUB/task_b16.py`)
- Candidate positions every 25 nt in GENCODE v49 introns, at least 1 kb from any PolyASite cluster, at most 600 per gene, giving a pool of 138,758.
- 5 decoys per competitor, matched on a weighted distance: weight 1 for distance to the gene's real end, weight 2 each for distance to the nearest 3′ and 5′ splice site.
- 30,000 drawn at seed 42 for the figures.

**Network checkpoints**
- `PUB/acc_arch_base_s42.pt` to `s46.pt`: the five 2,500 nt networks, from `PUB/acceptor_arch_train.py`. Used by Figs 3.10b, 3.12, 3.13 and section 3.2.7.
- `PUB/cnn_model.pt` (seed 42, from `cnn_train.py`) is weight-identical to `acc_arch_base_s42.pt`. Used by Fig 3.11.
- Architecture: `PUB/cnn_model_def.py`. 32 first-layer filters of width 19 with exponential activation; max pooling by 10, 5 then 2, reducing 2,500 nt to 25 positions each with a 378 nt receptive field at 100 nt stride; flattened to 2,400 values; one 64-unit dense layer; one output.
- Training: BCE with `pos_weight` = negatives/positives, Adam lr 1e-3, weight decay 1e-6, batch 128, dropout 0.4, up to 60 epochs, patience 10 on inner-holdout win-rate, seed 42.
- The 8-seed comparison (seeds 42-49, with and without the acceptor score) is `PUB/acc_fu.sbatch`, job 58016143, writing `acceptor_arch_train.tsv`.
- **OPEN: no LOCO checkpoints were saved**, only scores (`cnn_loco_scores.npz` from `cnn_loco_train.py --ksh 3`, job 57877643; `cnn_wg_scores.npz` from `cnn_wg_loco.py`, job 57904993). Rebuilding either means retraining on a GPU, so archive the `.npz` score files.

**Statistics:** `T/py/stats_ch3.py` writes `T/data/stats_ch3.tsv`, which holds every Wilcoxon, McNemar, Welch and bootstrap result cited in the chapter, with a row ID per claim. It defines the `wilcoxon()` other scripts `exec`. Run by hand, no sbatch.

**How the regrouped figures are built.** `T/R/fig3_regroup.R` uses `T/R/_harvest.R` to re-run a `PUB/` figure script inside a sandbox where `ggsave` and file-writing `writeLines` are no-ops and the working directory is `PUB`. It takes the resulting panel objects, strips their in-figure text and retitles them. One run from `T` rebuilds Figures 3.3, 3.4, 3.6, 3.7, 3.8, 3.9 and 3.11; log `T/logs/regroup.log`. The harvested `PUB/` scripts must find all of their own inputs or they abort.

**All R scripts in `T/R` assume the working directory is `T`.**

**External motif data:** CISBP-RNA 2.00 (Ray et al. 2013, cisbp-rna.ccbr.utoronto.ca), as `CLEAN/cisbp_rna.meme` (544 matrices named like `M00002_2.00_TRIM71`), downloaded as `CLEAN/Homo_sapiens_2026_06_25_4:21_pm.zip` with `CLEAN/cisbp_rna/` alongside.

**SpliceAI is not used in any chapter figure.** `PUB/sa_peak_scalars_final.tsv` (from the bigWigs at `/gbdb/hg38/bbi/spliceAi/wildtype/`, local copy `PUB/spliceai_bw/`, 3.4 GB) is read only by `pas_controlled.py` and `splice_model_expand.py`, for rows the chapter does not report. The user rejected SpliceAI for interpretation on the grounds that its 10 kb context encodes exon-intron structure.

**Figure-number to file mapping** (note `T/README.md` is stale and uses the old numbering)

| Thesis | Image in `T/out/` | Script in `T/R/` |
|---|---|---|
| 3.1 | fig3_01_hypothesis.png | fig3_01_hypothesis.R |
| 3.2 | fig3_02_genesets.png | fig3_02_genesets.R |
| 3.3 | fig3_03_splice_environment.png | fig3_regroup.R |
| 3.4 | fig3_04_donor_scores.png | fig3_regroup.R |
| 3.5 | fig3_hypothesis_dose.png | fig3_hypothesis_dose.R |
| 3.6 | fig3_05_splice_model.png | fig3_regroup.R |
| 3.7 | fig3_06_splice_model_performance.png | fig3_regroup.R |
| 3.8 | fig3_07_architecture.png | fig3_regroup.R |
| 3.9 | fig3_08_cnn_vs_splice.png | fig3_regroup.R |
| 3.10 | fig3_09_robustness.png | fig3_09_robustness.R |
| 3.11 | fig3_12_interpretation.png | fig3_regroup.R |
| 3.12 | fig3_13_mechanism.png | fig3_13_mechanism.R |
| 3.13 | fig3_implant_panel.png | fig3_implant_panel.R |
| 3.14 | fig3_14_recovery.png | fig3_14_recovery.R |

**Built but deliberately not in the chapter:** `fig3_10_exon_context`, `fig3_11_content`, `fig3_u1amo`, `fig3_concepts`, `fig3_deep`, `fig_repeats`; the whole U1 AMO derepression analysis (So 2019 GSE135140 and the replication set) was removed because the competitor peak after U1 AMO is under 1% of the real-end peak; the repeat and exon-context analyses were cut as having no clear story. The earlier long version of the chapter is `T/chapter3_v1/`. Do not reintroduce these without asking.
### Figure 3.1: The comparison, as a schematic

<!-- BEGIN AUTO:3.1:image -->
![Figure 3.1](figures/fig3_1.png)
<!-- END AUTO:3.1:image -->

**Shows**
- **a.** A gene model with the annotated 3′ end in the terminal exon (blue) and intronic competitors (orange). Competitors are the sites scored at or above the annotated end by all three Chapter 2 models.
- **b.** The 2,500 nt window the Chapter 3 models read, -2,000 to +500 nt, drawn for a real end and for a competitor as the hypothesis predicts: donor-like sites capable of pairing with U1, U1 snRNP bound, and cleavage not used.

**How:** `cd T && Rscript R/fig3_01_hypothesis.R`. Drawn only, no data. **OPEN:** the script also builds an unused panel c with stale figure references; dead code.

<!-- BEGIN AUTO:3.1:code -->
**Code.** `thesis_ch3/R/fig3_01_hypothesis.R`, 104 lines, sha256 `aadf86ab9d21`. Reproduced verbatim; also in `code/fig3_01_hypothesis.R`.

```r
# fig3_01_hypothesis.R -- Figure 3.1, the chapter's hypothesis as a cartoon.
#
# A  one gene: a used 3' end in the terminal exon, competitors in introns that carry a complete
#    local poly(A) signal (Chapter 2)
# B  the 2,500 nt window the models read, [-2000, +500) about the site, for each class:
#    a real end sits after an acceptor in a terminal exon with no donor; a competitor sits in
#    intron, near donors and donor-like sites that bind U1, which suppresses cleavage
# C  the three predictions the chapter tests, each pointed at the figure that tests it
#
# Positions in A and B are schematic. Nothing here is data.
suppressPackageStartupMessages({library(ggplot2);library(patchwork)})
INK<-"#1A1A1A"; GTX<-"#5A6068"; GREY<-"#8A8F98"; PAPER<-"#FCFCFD"
CAN<-"#0072B2"; COMP<-"#E69F00"; EXON<-"#D9DDE3"; U1C<-"#6A5ACD"
TS<-3.1   # text size (mm) for labels
tt<-function(s) ggtitle(s)
TH<-theme_void(base_size=10)+theme(
  plot.title=element_text(face="bold",colour=INK,size=11,margin=margin(b=4)),
  plot.margin=margin(6,8,6,8), plot.background=element_rect(fill="white",colour=NA))

# ---------------------------------------------------------------- A  gene
ex<-data.frame(x1=c(2,20,40,60,84), x2=c(8,25,45,65,98))
comp<-c(12,31,34,52,72,77)
pA<-ggplot()+
  annotate("segment",x=2,xend=98,y=0,yend=0,colour=GREY,linewidth=0.5)+
  geom_rect(data=ex,aes(xmin=x1,xmax=x2,ymin=-1,ymax=1),fill=EXON,colour=INK,linewidth=0.3)+
  annotate("segment",x=comp,xend=comp,y=0,yend=2.2,colour=COMP,linewidth=0.6)+
  annotate("point",x=comp,y=2.2,colour=COMP,size=2.4)+
  annotate("segment",x=97,xend=97,y=1,yend=3.2,colour=CAN,linewidth=0.8)+
  annotate("point",x=97,y=3.2,colour=CAN,size=3.4)+
  annotate("text",x=97,y=4.3,label="used 3′ end",colour=CAN,size=TS,fontface="bold",hjust=0.8)+
  annotate("text",x=44,y=3.6,colour=COMP,size=TS,fontface="bold",
           label="competitors")+
  annotate("text",x=91,y=-1.9,label="terminal exon",colour=GTX,size=TS-0.3)+
  annotate("text",x=0,y=0,label="5′",colour=GTX,size=TS,hjust=1)+
  annotate("text",x=100,y=0,label="3′",colour=GTX,size=TS,hjust=0)+
  coord_cartesian(xlim=c(-2,102),ylim=c(-2.4,4.8),clip="off")+
  tt("A   One used 3′ end and its intragenic competitors")+TH

# ---------------------------------------------------------------- B  the two windows
yR<-2.2; yC<-0         # row heights
u1R<-c(-1800,-1450,-1150)                             # few donor-like sites upstream of a real end
u1C<-c(-1250,-1000,-760,-520,-380,-240,-120,160,380)  # many around a competitor
pB<-ggplot()+
  # real end row: intron, acceptor, terminal exon, cleavage, downstream (not in the mRNA)
  annotate("segment",x=-2000,xend=-700,y=yR,yend=yR,colour=GREY,linewidth=0.5)+
  annotate("rect",xmin=-700,xmax=0,ymin=yR-0.3,ymax=yR+0.3,fill=EXON,colour=INK,linewidth=0.3)+
  annotate("segment",x=0,xend=500,y=yR,yend=yR,colour=GREY,linewidth=0.4,linetype="22")+
  annotate("segment",x=-700,xend=-700,y=yR+0.3,yend=yR+0.75,colour=INK,linewidth=0.5)+
  annotate("text",x=-700,y=yR+0.98,label="acceptor (3′ ss)",colour=INK,size=TS-0.3)+
  annotate("segment",x=u1R,xend=u1R,y=yR-0.18,yend=yR+0.18,colour=U1C,linewidth=0.6)+
  annotate("segment",x=0,xend=0,y=yR+0.3,yend=yR+0.85,colour=CAN,linewidth=0.8)+
  annotate("point",x=0,y=yR+0.85,colour=CAN,size=3)+
  annotate("text",x=-350,y=yR-0.62,label="terminal exon",colour=GTX,size=TS-0.3)+
  annotate("text",x=530,y=yR,label="real 3′ end",colour=CAN,size=TS,fontface="bold",hjust=0)+
  # competitor row: intron throughout, an upstream exon ending in a donor, many donor-like sites
  annotate("segment",x=-2000,xend=500,y=yC,yend=yC,colour=GREY,linewidth=0.5)+
  annotate("rect",xmin=-1750,xmax=-1500,ymin=yC-0.3,ymax=yC+0.3,fill=EXON,colour=INK,linewidth=0.3)+
  annotate("segment",x=-1500,xend=-1500,y=yC+0.3,yend=yC+0.75,colour=INK,linewidth=0.5)+
  annotate("text",x=-1500,y=yC+0.98,label="donor (5′ ss)",colour=INK,size=TS-0.3)+
  annotate("segment",x=u1C,xend=u1C,y=yC-0.18,yend=yC+0.18,colour=U1C,linewidth=0.6)+
  annotate("point",x=c(-1500,-380),y=yC-0.55,colour=U1C,size=5.2,alpha=0.9)+
  annotate("text",x=c(-1500,-380),y=yC-0.55,label="U1",colour="white",size=2.3,fontface="bold")+
  annotate("segment",x=0,xend=0,y=yC+0.3,yend=yC+0.85,colour=COMP,linewidth=0.8)+
  annotate("point",x=0,y=yC+0.85,colour=COMP,size=3)+
  annotate("point",x=0,y=yC+0.85,shape=4,colour=INK,size=3.6,stroke=1)+
  annotate("text",x=530,y=yC,label="competitor",colour=COMP,size=TS,fontface="bold",hjust=0)+
  # legend for the ticks, and the window
  annotate("segment",x=-2000,xend=-1960,y=-1.6,yend=-1.6,colour=U1C,linewidth=0.6)+
  annotate("segment",x=-1980,xend=-1980,y=-1.78,yend=-1.42,colour=U1C,linewidth=0.6)+
  annotate("text",x=-1930,y=-1.6,label="donor-like site",colour=GTX,size=TS-0.3,hjust=0)+
  annotate("segment",x=-2000,xend=500,y=-2.15,yend=-2.15,colour=INK,linewidth=0.3,
           arrow=arrow(ends="both",length=unit(1.6,"mm"),type="closed"))+
  annotate("text",x=-750,y=-2.45,label="2,500 nt window",
           colour=INK,size=TS-0.3)+
  scale_x_continuous(breaks=c(-2000,-1500,-1000,-500,0,500))+
  coord_cartesian(xlim=c(-2050,850),ylim=c(-2.6,3.4),clip="off")+
  labs(x="position relative to the site (nt)")+
  tt("B   Hypothesised splicing environment of each site")+TH+
  theme(axis.text.x=element_text(colour=GTX,size=8),axis.ticks.x=element_line(colour=GREY),
        axis.ticks.length.x=unit(1.2,"mm"),
        axis.title.x=element_text(colour=GTX,size=8,margin=margin(t=3)))

# ---------------------------------------------------------------- C  predictions
P<-data.frame(
  x=c(1,2,3),
  head=c("Prediction 1","Prediction 2","Prediction 3"),
  body=c("Fewer strong donors and\nU1 sites upstream of\na used 3′ end",
         "A strong upstream acceptor\nmarks a used 3′ end",
         "A network reading the same\n2,500 nt learns these\nsignals, and others"),
  test=c("Figures 3.3, 3.4","Figures 3.5, 3.6","Figures 3.7 to 3.11"))
W<-0.46
pC<-ggplot(P)+
  geom_rect(aes(xmin=x-W,xmax=x+W,ymin=0,ymax=1),fill=PAPER,colour=GREY,linewidth=0.3)+
  geom_text(aes(x=x-W+0.04,y=0.84,label=head),hjust=0,fontface="bold",colour=INK,size=TS)+
  geom_text(aes(x=x-W+0.04,y=0.5,label=body),hjust=0,colour=INK,size=TS-0.2,lineheight=0.95)+
  geom_text(aes(x=x-W+0.04,y=0.15,label=test),hjust=0,colour=CAN,size=TS-0.3,fontface="italic")+
  coord_cartesian(xlim=c(0.5,3.5),ylim=c(0,1))+
  tt("C   Three predictions, and where each is tested")+TH

fig<-pA/pB+plot_layout(heights=c(0.9,1.7))
dir.create("out",showWarnings=FALSE)
ggsave("out/fig3_01_hypothesis.pdf",fig,width=180,height=140,units="mm",device=cairo_pdf)
ggsave("out/fig3_01_hypothesis.png",fig,width=180,height=140,units="mm",dpi=300,bg="white")
cat("[out] out/fig3_01_hypothesis.{pdf,png}\n")
```
<!-- END AUTO:3.1:code -->

### Figure 3.2: Gene sets and how performance is measured

<!-- BEGIN AUTO:3.2:image -->
![Figure 3.2](figures/fig3_2.png)
<!-- END AUTO:3.2:image -->

**Shows**
- **a.** The funnel of gene sets: 15,560 scanned, 8,977 uncontested, 6,583 contested (70,723 candidates), 4,847 analysis set (48,394 candidates, 43,547 competitors), and a 13,824-gene union box. Two gene-removal steps (-1,660 and -76 genes left with no competitor) appear only in the figure, never in the text.
- **b.** The two within-gene measures on two example genes: P@1 (is the real end ranked first?) and win-rate (what fraction of a gene's competitors score below the real end?).
- **c.** The two evaluation schemes: 23-fold leave-one-chromosome-out with a 25% inner gene holdout, and the fixed split by gene.

**Why:** the chapter reports several P@1 values that are not comparable unless the gene set and the scheme are named, so the figure fixes the vocabulary.

**Key numbers.** None of these is printed in the figure; all are text-only in section 3.2.1.
- Chance P@1 on the analysis set is 0.287 (the mean over genes of 1/n).
- Competitors per gene 1 to 897, median 3; the top 10% of genes hold 61.9% of all competitors; Spearman rho against gene length 0.42. Recomputed from `cnn_data.npz`: min 1, max 897, median 3, top decile 0.6189.

**How:** `cd T && Rscript R/fig3_02_genesets.R`, reading `PUB/analysis_set_funnel.tsv` and `PUB/cnn_wholegenome_p1.tsv`. The script asserts 15,560 / 6,583 / 8,977 / 70,723 / 4,847 / 48,394 / 43,547 / 13,824. Panels b and c are drawn and the split sizes in c are hard-coded. **OPEN:** the three competitor-distribution numbers have no saved table.

<!-- BEGIN AUTO:3.2:code -->
**Code.** `thesis_ch3/R/fig3_02_genesets.R`, 117 lines, sha256 `82e0f4a8d3da`. Reproduced verbatim; also in `code/fig3_02_genesets.R`.

```r
# fig3_02_genesets.R -- Figure 3.2, which genes each result is computed on, and how models are scored.
#
# A  gene sets. Counts are READ from ../analysis_set_funnel.tsv and ../cnn_wholegenome_p1.tsv and
#    asserted, so the diagram cannot drift from the tables.
# B  within-gene ranking: precision at 1 and win-rate on two toy genes
# C  out-of-fold scoring: leave-one-chromosome-out with a 25% gene holdout for tuning, and the one
#    fixed split used for interpretation and robustness
suppressPackageStartupMessages({library(ggplot2);library(patchwork)})
INK<-"#1A1A1A"; GTX<-"#5A6068"; GREY<-"#8A8F98"; PAPER<-"#FCFCFD"
CAN<-"#0072B2"; COMP<-"#E69F00"; LIGHT<-"#E4E7EB"; MID<-"#A9B0BA"
TS<-3.0
TH<-theme_void(base_size=10)+theme(
  plot.title=element_text(face="bold",colour=INK,size=11,margin=margin(b=4)),
  plot.margin=margin(6,8,6,8), plot.background=element_rect(fill="white",colour=NA))
fmt<-function(n) format(n,big.mark=",")

# ---------------------------------------------------------------- counts, from the tables
F<-read.delim("../analysis_set_funnel.tsv",check.names=FALSE)
W<-read.delim("../cnn_wholegenome_p1.tsv",check.names=FALSE)
n_all<-W$n_genes[W$scope=="genome-wide"][1]
n_con<-W$n_genes[W$scope=="contested"][1]
n_unc<-n_all-n_con
st<-function(s,col) F[[col]][F$step==s]
n_cand<-st("all peaks","n_peaks")
rm_apa<-st("competitors: APA-annotated removed","n_removed")
rm_te <-st("competitors: reference terminal exon removed","n_removed")
rm_g1 <-st("genes without a surviving competitor dropped","n_removed")
rm_dn <-st("competitors: at or downstream of the canonical removed","n_removed")
rm_g2 <-st("analysis set","n_removed")
n_ana <-st("analysis set","n_genes"); p_ana<-st("analysis set","n_peaks")
c_ana <-st("analysis set","n_competitor")
stopifnot(n_all==15560, n_con==6583, n_unc==8977, n_cand==70723, n_ana==4847,
          p_ana==48394, c_ana==43547, n_unc+n_ana==13824)

# ---------------------------------------------------------------- A  gene sets
box<-function(x,y,w,h,head,sub=NULL,fill=PAPER,border=GREY,col=INK){
  l<-list(annotate("rect",xmin=x-w/2,xmax=x+w/2,ymin=y-h/2,ymax=y+h/2,fill=fill,colour=border,linewidth=0.35))
  if (is.null(sub)) l<-c(l,annotate("text",x=x,y=y,label=head,fontface="bold",colour=col,size=TS+0.2))
  else l<-c(l,annotate("text",x=x,y=y+h*0.2,label=head,fontface="bold",colour=col,size=TS+0.2),
                annotate("text",x=x,y=y-h*0.2,label=sub,colour=GTX,size=TS-0.3,lineheight=0.95))
  l}
arr<-function(x,y,xend,yend) annotate("segment",x=x,y=y,xend=xend,yend=yend,colour=INK,linewidth=0.35,
                                      arrow=arrow(length=unit(1.5,"mm"),type="closed"))
tag<-function(x,y,s,hj=0.5) annotate("text",x=x,y=y,label=s,colour=CAN,size=TS-0.3,fontface="italic",hjust=hj)
ys<-c(64,57.5,51,44.5,38)
side<-function(y,s) list(annotate("segment",x=50,xend=53,y=y,yend=y,colour=GREY,linewidth=0.35),
                         annotate("text",x=54,y=y,label=s,colour=GTX,size=TS-0.5,hjust=0))
pA<-ggplot()+
  box(40,95,44,8,paste(fmt(n_all),"genes with an annotated 3\u2032 end"))+
  arr(26,91,14,82.3)+arr(50,91,50,82.3)+
  box(14,76,26,12,paste(fmt(n_unc),"uncontested"),"the baseline already ranks\nthe annotated end first")+
  box(50,76,34,12,paste(fmt(n_con),"contested"),
      paste0("a competitor scores at or above\nthe annotated end; ",fmt(n_cand)," candidates"))+
  arr(50,70,50,31.3)+
  side(ys[1],paste0("\u2212",fmt(rm_apa)," competitors at an annotated poly(A) site"))+
  side(ys[2],paste0("\u2212",fmt(rm_te)," competitors in a reference terminal exon"))+
  side(ys[3],paste0("\u2212",fmt(rm_g1)," genes left with no competitor"))+
  side(ys[4],paste0("\u2212",fmt(rm_dn)," competitors at or downstream of the annotated end"))+
  side(ys[5],paste0("\u2212",fmt(rm_g2)," genes left with no competitor"))+
  box(50,25,34,12,paste(fmt(n_ana),"genes: analysis set"),
      paste0(fmt(p_ana)," candidates: one annotated\nend and ",fmt(c_ana)," competitors"),border=INK)+
  arr(14,70,14,31.3)+arr(33,25,27.2,25)+
  box(14,25,26,12,paste(fmt(n_unc+n_ana),"genes"),
      paste0(fmt(n_unc)," uncontested\n+ ",fmt(n_ana)," analysis set"))+
  coord_cartesian(xlim=c(0,100),ylim=c(15,100),clip="off")+
  ggtitle("A   Which genes each result is computed on")+TH

# ---------------------------------------------------------------- B  within-gene ranking
X2<-2.2
g1<-data.frame(g=1,s=c(2.4,1.6,1.1,0.5),cls=c("real","comp","comp","comp"))
g2<-data.frame(g=X2,s=c(2.2,1.8,1.2,0.9,0.3),cls=c("comp","real","comp","comp","comp"))
G<-rbind(g1,g2)
pB<-ggplot(G)+
  annotate("segment",x=c(1,X2),xend=c(1,X2),y=0.1,yend=2.7,colour=LIGHT,linewidth=2.2,lineend="round")+
  geom_point(aes(x=g,y=s,colour=cls),size=3.2)+
  scale_colour_manual(values=c(real=CAN,comp=COMP),guide="none")+
  annotate("text",x=c(1.1,X2+0.1),y=c(2.4,1.8),label=c("rank 1","rank 2"),hjust=0,colour=CAN,size=TS-0.3)+
  annotate("point",x=0.62,y=3.25,colour=CAN,size=2.4)+
  annotate("text",x=0.72,y=3.25,label="annotated end",hjust=0,colour=GTX,size=TS-0.4)+
  annotate("point",x=1.75,y=3.25,colour=COMP,size=2.4)+
  annotate("text",x=1.85,y=3.25,label="competitor",hjust=0,colour=GTX,size=TS-0.4)+
  annotate("text",x=c(1,X2),y=-0.25,label=c("gene 1","gene 2"),colour=INK,size=TS,fontface="bold")+
  annotate("text",x=c(1,X2),y=-0.55,label=c("P@1 = 1\nwin-rate 3/3 = 1.00","P@1 = 0\nwin-rate 3/4 = 0.75"),
           colour=GTX,size=TS-0.4,lineheight=0.95,vjust=1)+
  annotate("text",x=0.5,y=1.4,label="model score",angle=90,colour=GTX,size=TS-0.3)+
  coord_cartesian(xlim=c(0.45,2.9),ylim=c(-1.4,3.4),clip="off")+
  ggtitle("B   Within-gene ranking")+TH

# ---------------------------------------------------------------- C  out-of-fold scoring
chr<-c(1:22,"X"); k<-7
cw<-1; cx<-seq_along(chr)
CH<-data.frame(x=cx,lab=chr,fill=ifelse(cx==k,INK,LIGHT))
fx<-c(train=27658,val=6538,test=14198); gx<-c(train=2781,val=443,test=1623); fr<-cumsum(c(0,fx))/sum(fx)*23+0.5
pC<-ggplot()+
  annotate("text",x=0.5,y=4.45,hjust=0,label="leave-one-chromosome-out, 23 folds",colour=INK,size=TS,fontface="bold")+
  geom_rect(data=CH,aes(xmin=x-0.45,xmax=x+0.45,ymin=3.2,ymax=3.8,fill=I(fill)),colour=GREY,linewidth=0.2)+
  geom_text(data=CH[CH$lab %in% c("1","7","12","22","X"),],aes(x=x,y=3.0,label=lab),colour=GTX,size=TS-0.6)+
  annotate("text",x=k,y=4.0,label="held out",colour=INK,size=TS-0.4)+
  annotate("rect",xmin=0.55,xmax=17,ymin=2.0,ymax=2.5,fill=LIGHT,colour=GREY,linewidth=0.2)+
  annotate("rect",xmin=17,xmax=23.45,ymin=2.0,ymax=2.5,fill=MID,colour=GREY,linewidth=0.2)+
  annotate("text",x=8.8,y=2.25,label="training genes, 75%",colour=INK,size=TS-0.4)+
  annotate("text",x=20.2,y=2.25,label="tuning, 25%",colour=INK,size=TS-0.4)+
  annotate("text",x=0.5,y=0.35,hjust=0,label="fixed split by gene",colour=INK,size=TS,fontface="bold")+
  annotate("rect",xmin=fr[1],xmax=fr[2],ymin=-1.0,ymax=-0.2,fill=LIGHT,colour=GREY,linewidth=0.2)+
  annotate("rect",xmin=fr[2],xmax=fr[3],ymin=-1.0,ymax=-0.2,fill=MID,colour=GREY,linewidth=0.2)+
  annotate("rect",xmin=fr[3],xmax=fr[4],ymin=-1.0,ymax=-0.2,fill=INK,colour=GREY,linewidth=0.2)+
  annotate("text",x=(fr[1]+fr[2])/2,y=-0.6,label=paste0("training\n",fmt(gx[1])," genes"),colour=INK,size=TS-0.5)+
  annotate("text",x=(fr[2]+fr[3])/2,y=-0.6,label=paste0("validation\n",fmt(gx[2])),colour=INK,size=TS-0.5)+
  annotate("text",x=(fr[3]+fr[4])/2,y=-0.6,label=paste0("test\n",fmt(gx[3])," genes"),colour="white",size=TS-0.5)+
  coord_cartesian(xlim=c(0.3,23.6),ylim=c(-1.3,4.6),clip="off")+
  ggtitle("C   How every candidate is scored")+TH

fig<-pA/(pB+pC+plot_layout(widths=c(0.85,1.15)))+plot_layout(heights=c(1.25,1))
dir.create("out",showWarnings=FALSE)
ggsave("out/fig3_02_genesets.pdf",fig,width=180,height=190,units="mm",device=cairo_pdf)
ggsave("out/fig3_02_genesets.png",fig,width=180,height=190,units="mm",dpi=300,bg="white")
cat("[out] out/fig3_02_genesets.{pdf,png}\n")
```
<!-- END AUTO:3.2:code -->

### Figure 3.3: The splicing environment upstream of real ends and competitors

<!-- BEGIN AUTO:3.3:image -->
![Figure 3.3](figures/fig3_3.png)
<!-- END AUTO:3.3:image -->

**Shows**, for 4,847 real ends, 43,547 competitors and 30,000 matched intronic decoys:
- **a.** Density of strong 5′ splice sites (score5 >= 8) per kb in 100 nt bins across the -2,000 to +500 nt window, ribbons +/- 1 SEM.
- **b.** Per-gene difference in U1 site density (-dG >= 4) between the real end and the gene's competitors, for 4,847 genes.
- **c.** Cumulative distribution of the highest score3 among AG-anchored 23-mers between -2,000 and -126 nt.
- **d.** Where that best AG sits, for sites whose best score3 is >= 6.5 (99% of sites), as a kernel density.

**Why:** this is the descriptive core of the hypothesis. Real ends have *fewer* strong donors upstream and a *stronger* acceptor upstream than their own competitors.

**Key numbers** (all verified against the tables)
- Strong donors per kb averaged over the 500 nt upstream: 0.643 real, 0.813 competitor, 0.875 decoy. Downstream they are equal: 0.81 / 0.83 / 0.82, so the depletion is specific to the upstream window.
- Within gene: -0.24 donors per kb [0.20, 0.28], z = -10.8, r = 0.18.
- U1 sites: -10.9 per kb at real ends, lower in 67% of genes, z = -25.4. Panel b plots all 4,847 genes; the Wilcoxon n of 4,836 in the text is after dropping 11 zero deltas. Both are right, but the pairing invites an examiner's question.
- Best upstream acceptor, median score3: 10.35 real, 9.72 competitor, 9.74 decoy. Within gene +0.58, higher in 62% of genes, z = 21.1.
- An acceptor within 1 kb upstream: 51.8% of real ends against 46.7% of competitors, z = 6.7.
- The acceptor score alone ranks the real end first in 38.5% of genes. **OPEN:** no saved table; it recomputes to 0.3848 from `fig3_ucube_meta.tsv`.

**How**
- Panels a and b are harvested from `PUB/make_splicing_environment.R`, reading `splice_env_metagene.tsv`, `splice_env_gene_delta.tsv`, `splice_env_wilcoxon.tsv` and `u1_5ss_stats.tsv`, produced by `PUB/splice_score_redundancy.py` via `PUB/splice_score.sbatch` (24 CPUs). **OPEN:** five candidate jobs (57734587, 57734618, 57734654, 57734959, 57736963) and no record of which wrote the final tables.
- Panels c and d are drawn inline in `fig3_regroup.R` from `PUB/acceptor_position_scan.tsv` (the script asserts 78,394 rows), produced by `PUB/acceptor_position_scan.py`, log `acc_pos_58007546.out`, no sbatch.
- Bootstrap 1,000 resamples. The within-gene statistic in the text comes from `stats_ch3.py` row A1, which rescans `cnn_data.npz` independently of panel a and agrees.
- Rebuild: `cd T && Rscript R/fig3_regroup.R`

<!-- BEGIN AUTO:3.3:code -->
**Code.** `thesis_ch3/R/fig3_regroup.R`, lines 24 to 53 (the block that saves `fig3_03_splice_environment`), plus the shared preamble. The whole file (120 lines, sha256 11f48036ce52) is in the appendix and in `code/`.

Shared preamble:

```r
# fig3_regroup.R -- Figures 3.3 to 3.8 and 3.10, regrouped from the panels pub/'s scripts build.
#
# Each old figure's script is run in a sandbox (R/_harvest.R) for its panel objects. Panels are
# relettered and retitled for the new numbering. In-figure text is limited to labels that NAME
# something (series, reference lines, categories, schematic parts); statistics, sentences,
# sequences and numeric call-outs are removed here and live in the legend or the text.
#
#   3.3  <- make_splicing_environment.R A, B  + new C, D (../acceptor_position_scan.tsv)
#   3.4  <- make_splicing_environment.R C, D     3.7  <- make_figure_architecture.R, make_figure_rescue.R a
#   3.5  <- make_splice_model_final.R   A, B     3.8  <- make_figure_loco.R A, B, D
#   3.6  <- make_splice_model_final.R   C, D     3.12 <- make_figure_cnn_interpretability.R
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr)})
setwd("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3")
source("R/_harvest.R")
dir.create("out", showWarnings = FALSE)
INK <- "#1A1A1A"; GTX <- "#5A6068"
TT <- theme(plot.title = element_text(face = "bold", size = 11, colour = INK),
            plot.subtitle = element_blank(), plot.caption = element_blank())
save2 <- function(fig, name, w, h, dpi = 300) {
  ggsave(file.path("out", paste0(name, ".pdf")), fig, width = w, height = h, device = cairo_pdf, limitsize = FALSE)
  ggsave(file.path("out", paste0(name, ".png")), fig, width = w, height = h, dpi = dpi, bg = "white", limitsize = FALSE)
  cat("[out]", name, "\n")
}
```

Block for this figure:

```r
# ---------------------------------------------------------------- 3.3 donor and acceptor environment
E <- harvest("make_splicing_environment.R")
drop_leader <- function(p) { p$layers <- Filter(function(l) !(inherits(l$geom, "GeomSegment") &&
    is.data.frame(l$data) && isTRUE(all(l$data$xend == -330))), p$layers); p }
pA0 <- E$panel_A();
a <- edit_text(drop_leader(pA0), drop = c("-50 nt bin", "upstream of the poly")) +
  labs(title = "A   Strong 5′ splice-site density") + TT
b <- edit_text(E$panel_B(), drop = c("^mean ", "no difference", "median -12")) +
  labs(title = "B   Within-gene difference in U1 site density") + TT
REAL <- "#08519C"; COMP <- "#E8A33D"; DEC <- "#7B5EA7"
AP <- read.delim("../acceptor_position_scan.tsv") %>%
  mutate(class = factor(class, levels = c("real", "competitor", "decoy"),
                        labels = c("real 3′ end", "competitor", "intronic decoy")))
stopifnot(nrow(AP) == 4847 + 43547 + 30000)
CLS <- c("real 3′ end" = REAL, competitor = COMP, "intronic decoy" = DEC)
TH3 <- theme_classic(base_size = 7.5) + theme(axis.text = element_text(colour = GTX), legend.position = "none",
        panel.grid.major.y = element_line(colour = "#E8EAED", linewidth = .2))
c3 <- ggplot(AP, aes(max_me3, colour = class)) +
  stat_ecdf(linewidth = 0.6) + scale_colour_manual(values = CLS) +
  coord_cartesian(xlim = c(4, 16)) +
  labs(x = "strongest upstream 3′ splice-site score (MaxEntScan score3)", y = "cumulative fraction of sites",
       title = "C   Strongest upstream 3′ splice site") + TH3 + TT
d3 <- ggplot(AP %>% filter(max_me3 >= 6.5), aes(position, colour = class)) +
  geom_density(aes(y = after_stat(density) * 1000), linewidth = 0.6, adjust = 0.6, key_glyph = "path") + scale_colour_manual(values = CLS, name = NULL) +
  scale_x_continuous(limits = c(-2000, -126), breaks = c(-2000, -1500, -1000, -500)) +
  labs(x = "position of the strongest upstream 3′ splice site (nt)", y = "density (per kb)",
       title = "D   Position of that site") + TH3 + TT +
  theme(legend.position = "bottom")
save2((a / b / (c3 | d3)) + plot_layout(heights = c(1, 0.8, 0.9)), "fig3_03_splice_environment", 6.8, 9.6)
```
<!-- END AUTO:3.3:code -->

### Figure 3.4: Donor scores, thresholds and background rates

<!-- BEGIN AUTO:3.4:image -->
![Figure 3.4](figures/fig3_4.png)
<!-- END AUTO:3.4:image -->

**Shows**
- **a.** Joint distribution of U1 duplex -dG and MaxEnt score5 over all 10,723,361 GT-anchored 9-mer positions, with an inset over all 196,063,340 scanned positions.
- **b.** Fraction of scanned positions at or above each score, marking -dG >= 4, -dG >= 7 and score5 >= 8.
- **c.** For thresholds 2 to 14 in steps of 0.5, sensitivity at annotated GENCODE splice sites against the rate of calls per kb in competitor windows. Open circles mark score3 >= 6.5 and score5 >= 8; triangles mark U1 >= 4 and >= 7.

**Why:** it justifies the thresholds used everywhere else, by showing what each one costs in false calls. Calibrated (composition-corrected) features were tried and **lowered** performance, so the chapter places thresholds on this curve instead of recalibrating.

**Key numbers**
- score5 and U1 -dG correlate at r = 0.692 over GT-anchored positions, but only 0.356 over all positions.
- Exceedance: -dG >= 4 at 9.40% of positions, -dG >= 7 at 0.59%, score5 >= 8 at 0.084%.
- Acceptor at score3 >= 6.5: sensitivity 0.822, 3.03 calls per kb of intron, and at least one call in 98.5% of competitor windows. So the threshold is permissive, which is why binarising it fails.
- Donor at score5 >= 8: sensitivity 0.633, 0.82 calls per kb.
- U1 -dG >= 4: 94.7% of annotated donors pass, but so do 94 positions per kb.
- Composition null (text only): 9.2% of real positions pass U1 >= 4 against 8.4% after dinucleotide-preserving shuffling, i.e. 92% of the signal is composition; at least 85% at every threshold from 1 to 10.

**How**
- Panels a and b harvested from `make_splicing_environment.R`, reading `splice_score_joint.tsv`, `splice_score_exceedance.tsv`, `splice_score_stats.tsv` (same producer as Fig 3.3).
- Panel c from `T/data/maxent_calibration.tsv` (`T/py/maxent_calibration.py`; sensitivity from `PUB/exon_ref_scores.tsv` via `PUB/scripts/exon_reference.py`, background from 2,000 competitor windows at seed 0, 99,710 acceptors and 99,710 donors) and `T/data/u1_calibration.tsv` (`T/py/u1_calibration.py`, needs pysam, 20,000 GENCODE v49 donors, seed 0).
- Composition null: `T/py/u1_null_analysis.py` writes `T/data/u1_null_analysis.tsv` (Altschul-Erickson shuffle, seed 0, 120,597,848 positions).
- Rebuild: `cd T && Rscript R/fig3_regroup.R`. **OPEN:** none of the calibration or null scripts has an sbatch file or log; they were run by hand.

<!-- BEGIN AUTO:3.4:code -->
**Code.** `thesis_ch3/R/fig3_regroup.R`, lines 54 to 79 (the block that saves `fig3_04_donor_scores`), plus the shared preamble. The whole file (120 lines, sha256 11f48036ce52) is in the appendix and in `code/`.

Shared preamble:

```r
# fig3_regroup.R -- Figures 3.3 to 3.8 and 3.10, regrouped from the panels pub/'s scripts build.
#
# Each old figure's script is run in a sandbox (R/_harvest.R) for its panel objects. Panels are
# relettered and retitled for the new numbering. In-figure text is limited to labels that NAME
# something (series, reference lines, categories, schematic parts); statistics, sentences,
# sequences and numeric call-outs are removed here and live in the legend or the text.
#
#   3.3  <- make_splicing_environment.R A, B  + new C, D (../acceptor_position_scan.tsv)
#   3.4  <- make_splicing_environment.R C, D     3.7  <- make_figure_architecture.R, make_figure_rescue.R a
#   3.5  <- make_splice_model_final.R   A, B     3.8  <- make_figure_loco.R A, B, D
#   3.6  <- make_splice_model_final.R   C, D     3.12 <- make_figure_cnn_interpretability.R
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr)})
setwd("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3")
source("R/_harvest.R")
dir.create("out", showWarnings = FALSE)
INK <- "#1A1A1A"; GTX <- "#5A6068"
TT <- theme(plot.title = element_text(face = "bold", size = 11, colour = INK),
            plot.subtitle = element_blank(), plot.caption = element_blank())
save2 <- function(fig, name, w, h, dpi = 300) {
  ggsave(file.path("out", paste0(name, ".pdf")), fig, width = w, height = h, device = cairo_pdf, limitsize = FALSE)
  ggsave(file.path("out", paste0(name, ".png")), fig, width = w, height = h, dpi = dpi, bg = "white", limitsize = FALSE)
  cat("[out]", name, "\n")
}
```

Block for this figure:

```r
# ---------------------------------------------------------------- 3.4 donor scores
pc <- E$panel_C()
pc[[1]] <- edit_text(pc[[1]], drop = "r = 0.69") + labs(title = "A   U1 duplex free energy against MaxEntScan 5′ score") + TT
if (length(pc) > 1) pc[[2]] <- pc[[2]] + labs(title = NULL, subtitle = NULL, caption = NULL)
d <- edit_text(E$panel_D(), drop = "<mapped txt>") + labs(title = "B   Score distributions and thresholds") + TT
# C: MaxEntScan sensitivity at annotated sites against background calls in intronic competitor windows
MC <- read.delim("data/maxent_calibration.tsv") %>% mutate(site = factor(site, levels = c("acceptor", "donor"),
        labels = c("3\u2032 splice site (score3)", "5\u2032 splice site (score5)")))
U1C <- read.delim("data/u1_calibration.tsv")
USED <- MC %>% filter((grepl("^3", site) & threshold == 6.5) | (grepl("^5", site) & threshold == 8))
SC <- c("3\u2032 splice site (score3)" = "#E8A33D", "5\u2032 splice site (score5)" = "#08519C")
c4 <- ggplot(MC %>% filter(per_kb_background > 0), aes(per_kb_background, sensitivity_annotated, colour = site)) +
  geom_path(linewidth = 0.6) + geom_point(size = 1) +
  geom_point(data = USED, size = 3, shape = 21, fill = "white", stroke = 1) +
  geom_point(data = U1C, aes(background_per_kb, sensitivity_annotated_donors), inherit.aes = FALSE,
             shape = 24, size = 2.6, fill = "#7B5EA7", colour = INK, stroke = 0.4) +
  geom_text(data = U1C, aes(background_per_kb, sensitivity_annotated_donors, label = paste0("U1 \u2265 ", threshold)), inherit.aes = FALSE,
            hjust = 1.2, vjust = 0.3, size = 2.4, colour = "#7B5EA7") +
  scale_x_log10(breaks = c(0.01, 0.1, 1, 10, 100), labels = c("0.01", "0.1", "1", "10", "100")) +
  scale_colour_manual(values = SC, name = NULL) +
  labs(x = "background calls per kb of intronic sequence", y = "fraction of annotated sites passing",
       title = "C   Calibration of the thresholds") +
  theme_classic(base_size = 7.5) + TT + theme(axis.text = element_text(colour = GTX), legend.position = c(0.3, 0.85),
        legend.key.size = unit(3, "mm"), legend.text = element_text(size = 7))
save2((pc / ((d | c4) + plot_layout(widths = c(1.9, 1)))) + plot_layout(heights = c(1.1, 0.9)), "fig3_04_donor_scores", 8.2, 8.0)
```
<!-- END AUTO:3.4:code -->

### Figure 3.5: The hypothesis tested directly, by splice-site strength

<!-- BEGIN AUTO:3.5:image -->
![Figure 3.5](figures/fig3_5.png)
<!-- END AUTO:3.5:image -->

**This is the chapter's key figure.**

**Shows**, on the analysis set (4,847 genes, 48,394 candidates):
- **a.** Observed over expected number of real 3′ ends, grouped by the best upstream score3 (blue) or by the best score5 between that acceptor and the candidate (orange), on a log2 axis, with 95% bootstrap ribbons over genes. Bins are one unit over the body of each range, with open end bins (acceptor < 6 and >= 13; donor < 4, 4-6, one-unit 6-10, >= 10). Groups under 200 candidates are dropped.
- **b.** The same ratio over both scores jointly, as a heat map. Cells under 200 candidates are dropped; all 42 cells of the 6 x 7 grid survive. Fill is a diverging blue-orange scale on observed/expected, blue above 1. Note this re-uses panel a's colours for a different meaning, and the chapter caption does not define the scale.

**The estimand:** expected = the sum over the group of 1/n, where n is the number of candidates in that gene. A ratio of 1 means the group holds its fair share of real ends. This avoids arbitrary binning of genes and keeps the within-gene structure.

**Key numbers**
- Acceptor: 0.43 at score3 6-7 rising to 1.62 at >= 13. Monotone.
- Donor: 1.14, 1.22, 1.07, 1.11 below score5 8, then 0.92, 0.91, 0.72 at >= 10. The trend reverses the acceptor's but is **not** monotone: the 4-6 bin (1.22) is the peak and 6-7 (1.07) dips below 7-8 (1.11). The chapter text says only "the relationship was reversed", which is right; do not upgrade it to monotone.
- Joint conditional-logit model (softmax within gene, Newton-Raphson, Wald intervals): acceptor OR per SD 1.474 [1.424, 1.527], P = 3.5e-106; donor OR 0.930 [0.900, 0.960], P = 7.8e-6; interaction 0.989, P = 0.50, i.e. **no interaction**, the two effects are independent.
- Grid extremes: 0.447 (score3 7-8 with score5 >= 10) to 1.887 (score3 12-13 with score5 4-6).
- By fifths of score: 0.67 to 1.48 for the acceptor, 1.22 to 0.83 for the donor.

**How**
- `T/py/hypothesis_grid.py` writes `T/data/hypothesis_curve.tsv` and `hypothesis_grid.tsv` from `PUB/cnn_data.npz` and `PUB/fig3_ucube_meta.tsv` (`acc`, `interv_donor`). 1,000 bootstrap resamples over genes, seed 0. Acceptor bins: < 6, then one-unit bins to >= 13. Donor bins: < 4, 4-6, one-unit 6-10, then >= 10.
- `T/py/hypothesis_dose.py` writes `T/data/hypothesis_stats.tsv` and `hypothesis_dose.tsv` (the odds ratios and the fifths; 500 bootstrap resamples). It also reads `T/data/u1_replication_per_site.npz` and `oh2017_per_site.npz` for usage rows the chapter does not report.
- Rebuild: `python T/py/hypothesis_grid.py && python T/py/hypothesis_dose.py && cd T && Rscript R/fig3_hypothesis_dose.R`

<!-- BEGIN AUTO:3.5:code -->
**Code.** `thesis_ch3/R/fig3_hypothesis_dose.R`, 24 lines, sha256 `fb62ec98c729`. Reproduced verbatim; also in `code/fig3_hypothesis_dose.R`.

```r
# fig3_hypothesis_dose.R -- frequency of real 3' ends by MaxEntScan score of the upstream splice sites.
# No computation: reads data/hypothesis_curve.tsv and data/hypothesis_grid.tsv (py/hypothesis_grid.py).
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr)})
setwd("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3")
INK<-"#1A1A1A"; GTX<-"#5A6068"; CAN<-"#0072B2"; COMP<-"#E69F00"; ACC<-"#0072B2"; DON<-"#D55E00"
th<-theme_classic(base_size=10)+theme(plot.title=element_text(face="bold",size=11,colour=INK),axis.text=element_text(colour=GTX),legend.key.size=unit(4,"mm"))
C<-read.delim("data/hypothesis_curve.tsv",check.names=FALSE)%>%filter(candidates>=200)%>%
  mutate(score=factor(score,levels=c("3' splice site","5' splice site"),labels=c("Upstream 3' splice site","Upstream 5' splice site")))
pA<-ggplot(C,aes(mid,observed_over_expected,colour=score,fill=score))+geom_hline(yintercept=1,colour=GTX,linetype="dashed",linewidth=0.3)+
  geom_ribbon(aes(ymin=lo,ymax=hi),alpha=0.18,colour=NA)+geom_line(linewidth=1)+geom_point(size=2.4)+
  scale_colour_manual(values=c(ACC,DON))+scale_fill_manual(values=c(ACC,DON))+scale_y_continuous(trans="log2",breaks=c(0.5,0.75,1,1.5,2),limits=c(0.28,2.1))+
  scale_x_continuous(breaks=seq(4,14,2))+
  labs(x="MaxEntScan score",y="real 3' ends, observed / expected",title="A   Each splice site")+th+theme(legend.title=element_blank(),legend.position="bottom")
G<-read.delim("data/hypothesis_grid.tsv",check.names=FALSE)
ax<-unique(G[order(G$acc_mid),"acc_bin"]); dx<-unique(G[order(G$don_mid),"don_bin"])
G<-G%>%mutate(a=factor(acc_bin,levels=ax),d=factor(don_bin,levels=dx),l=log2(observed_over_expected))
L<-max(abs(G$l))
pB<-ggplot(G,aes(a,d,fill=l))+geom_tile(colour="white",linewidth=0.6)+
  scale_fill_gradient2(low=COMP,mid="#F7F7F7",high=CAN,midpoint=0,limits=c(-L,L),breaks=log2(c(0.5,1,2)),labels=c("0.5","1","2"),name="observed /\nexpected")+
  labs(x="3' splice site score",y="5' splice site score",title="B   Both splice sites")+th+
  theme(axis.line=element_blank(),axis.ticks=element_blank(),axis.text=element_text(colour=INK,size=8.5),legend.position="right",legend.key.height=unit(7,"mm"),legend.key.width=unit(3.5,"mm"))+coord_fixed()
fig<-pA+pB+plot_layout(widths=c(1,1.15))
ggsave("out/fig3_hypothesis_dose.pdf",fig,width=7.8,height=4.0,device=cairo_pdf); ggsave("out/fig3_hypothesis_dose.png",fig,width=7.8,height=4.0,dpi=300,bg="white")
cat("[out] out/fig3_hypothesis_dose\n")
```
<!-- END AUTO:3.5:code -->

### Table 3.1: The same test with the poly(A) signal held fixed

**Shows**, for genes where the real end and at least one competitor both carry AAUAAA starting 10 to 35 nt upstream and within 3 nt of each other (2,313 genes, 10,022 competitors): the difference in each splice measure between the real end and the mean over that gene's retained competitors, as a median over genes, with a two-sided Wilcoxon signed-rank test and r = |z|/sqrt(n).

**Why:** it answers the obvious objection that the splicing differences are really poly(A) signal differences. Holding AAUAAA fixed at the same position, the splicing differences persist while the cleavage-element differences do not.

**Key numbers:** acceptor +0.50 (z = 12.2, r = 0.25, P = 3.2e-34) and the rest as tabulated. Subset sizes: canonical-PAS subset 3,857 genes / 27,114 competitors; strict same-position subset 2,313 / 10,022.

**How:** `python T/py/pas_controlled.py` writes `T/data/pas_controlled.tsv`; the chapter table is the "strict AAUAAA, same position" rows, **typed into the markdown by hand**. All seven rows reproduce the TSV exactly at the stated rounding.

**Two wording items to settle.** The row label "U1 score upstream, distance-weighted" renames the TSV's `U1 sites upstream` (`spl[:,1]`, `pas_controlled.py:71`); confirm the feature really is distance-weighted. And the UGUA row pairs z = +4.4 with `frac_real_higher` = 0.384, i.e. the real end is higher in only 38% of genes, which the chapter reads as "marginally more frequent" at real ends. Possible, but it reads oddly; justify it in a sentence or drop the claim. Inputs: `cnn_data.npz`, `cnn_loco_scores.npz`, `cnn_wg_scores.npz`, `fig3_ucube_meta.tsv`, `sa_peak_scalars_final.tsv`, `T/data/stress_context.npz` (from `stress_test.py`), `T/data/repeat_sites.tsv` (from `repeat_context.py`), and hg38 via pysam.

### Figure 3.6: The 26-feature splice model

<!-- BEGIN AUTO:3.6:image -->
![Figure 3.6](figures/fig3_6.png)
<!-- END AUTO:3.6:image -->

**Shows**
- **a.** The features: 25 bins of 100 nt each contributing its density of U1 sites (-dG >= 4) per kb, plus a 26th feature, the best MaxEnt score3 between -2,000 and -126 nt.
- **b.** The fitted coefficients.

**Method:** features are centred within gene and scaled, then fitted by ridge logistic regression (IRLS), seed 42, 25% inner holdout, lambda grid 0.1 to 1e4. The chosen lambda was 0.1, **the lowest value in the grid**, which is worth noting.

**Why centred within gene:** the question is which candidate in a gene is the real one, so only within-gene contrasts should carry weight.

**Key numbers:** the U1 coefficient is most negative near the site, -0.23 in the bin [-200, -100); the acceptor coefficient is +0.30. Sign-consistent with Figure 3.5.

**How:** `PUB/splice_model_26.py` (no sbatch) using `PUB/binned_common.py`, `PUB/binned_profiles.npz` (from `binned_profile_scan.py`) and `analysis_columns.npz` (from `cache_analysis_columns.py`), writing `splice_model_26_coefficients.tsv` and `splice_model_26_metrics.tsv`. Panel a is a schematic. Rebuild: `cd T && Rscript R/fig3_regroup.R` (the harvest also needs `recall_at_k.tsv`, `score_separation.tsv` and `score_separation_sites.tsv` present).

<!-- BEGIN AUTO:3.6:code -->
**Code.** `thesis_ch3/R/fig3_regroup.R`, lines 80 to 94 (the block that saves `fig3_05_splice_model`), plus the shared preamble. This block also builds `fig3_06_splice_model_performance`. The whole file (120 lines, sha256 11f48036ce52) is in the appendix and in `code/`.

Shared preamble:

```r
# fig3_regroup.R -- Figures 3.3 to 3.8 and 3.10, regrouped from the panels pub/'s scripts build.
#
# Each old figure's script is run in a sandbox (R/_harvest.R) for its panel objects. Panels are
# relettered and retitled for the new numbering. In-figure text is limited to labels that NAME
# something (series, reference lines, categories, schematic parts); statistics, sentences,
# sequences and numeric call-outs are removed here and live in the legend or the text.
#
#   3.3  <- make_splicing_environment.R A, B  + new C, D (../acceptor_position_scan.tsv)
#   3.4  <- make_splicing_environment.R C, D     3.7  <- make_figure_architecture.R, make_figure_rescue.R a
#   3.5  <- make_splice_model_final.R   A, B     3.8  <- make_figure_loco.R A, B, D
#   3.6  <- make_splice_model_final.R   C, D     3.12 <- make_figure_cnn_interpretability.R
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr)})
setwd("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3")
source("R/_harvest.R")
dir.create("out", showWarnings = FALSE)
INK <- "#1A1A1A"; GTX <- "#5A6068"
TT <- theme(plot.title = element_text(face = "bold", size = 11, colour = INK),
            plot.subtitle = element_blank(), plot.caption = element_blank())
save2 <- function(fig, name, w, h, dpi = 300) {
  ggsave(file.path("out", paste0(name, ".pdf")), fig, width = w, height = h, device = cairo_pdf, limitsize = FALSE)
  ggsave(file.path("out", paste0(name, ".png")), fig, width = w, height = h, dpi = dpi, bg = "white", limitsize = FALSE)
  cat("[out]", name, "\n")
}
```

Block for this figure:

```r
# ---------------------------------------------------------------- 3.5, 3.6 splice model
S <- harvest("make_splice_model_final.R")
s5a <- edit_text(S$pA, drop = "Within-gene evaluation",
                 relab = c("Sequence window" = "25 bins of 100 nt",
                           "U1 duplex density" = "U1 site density per bin",
                           "Maximum MaxEnt" = "strongest upstream 3′ splice-site score",
                           "Ridge \\(L2\\)" = "ridge logistic regression"))
s5b <- edit_text(S$pB, drop = c("-0.23 at", "^\\+0.30$"), relab = c("MaxEnt 3'ss" = "acceptor"))
save2((s5a + labs(title = "A   Features") + TT) / (s5b + labs(title = "B   Fitted coefficients by position") + TT) +
      plot_layout(heights = c(1.15, 1)), "fig3_05_splice_model", 7.4, 6.6)
s6c <- edit_text(S$pC, drop = "<mapped lab>", relab = c("random" = "chance"))
s6d <- edit_text(S$pD, drop = c("<mapped sprintf", "Win-rate 0.699"))
save2((s6c + labs(title = "A   Within-gene recall at rank k") + TT) | (s6d + labs(title = "B   Gene-centred scores") + TT),
      "fig3_06_splice_model_performance", 7.8, 3.6)
```
<!-- END AUTO:3.6:code -->

### Figure 3.7: Within-gene performance of the splice model

<!-- BEGIN AUTO:3.7:image -->
![Figure 3.7](figures/fig3_7.png)
<!-- END AUTO:3.7:image -->

**Shows**
- **a.** Fraction of genes whose real end ranks in the top k, against chance (the mean over genes of min(k, n)/n).
- **b.** Model scores after subtracting each gene's mean, as violins with boxes.

**Key numbers:** P@1 0.465 [0.451, 0.479] against chance 0.287; top-3 0.798 against 0.693; win-rate 0.699 [0.689, 0.710]; 75% of real ends score above the mean of their own competitors, median margin +0.48, z = 37.1, r = 0.53. So 26 features of splicing context alone recover the real end in nearly half of genes.

**Careful: +0.48 and panel b are different quantities.** The +0.48 margin is the real end minus the mean of that gene's *competitors*. Panel b plots score minus the gene mean over *all* candidates, whose medians are +0.33 (real) and +0.00 (competitor), verified from `score_separation_sites.tsv`. The caption is correct; it is the chapter's cross-reference "median margin of +0.48 (Figure 3.7b)" that misleads. Write "cf. Figure 3.7b", or give both numbers.

**How:** panel a from `PUB/recall_at_k.tsv` (`PUB/recall_at_k.py`, 23-fold LOCO with lambda retuned per fold on a 25% gene holdout, 2,000 bootstrap resamples, seed 42); panel b from `PUB/score_separation.tsv` and `score_separation_sites.tsv` (`PUB/score_separation.py`). Rebuild: `cd T && Rscript R/fig3_regroup.R`.

**Text-only results in section 3.2.4** (no figure, deliberately)
- 176 positional features reach P@1 0.494 (McNemar P = 1.3e-7); ranking loss 0.481; MLP 0.470 / 0.470; convolution over bins 0.512 (P = 1.9e-15). Scripts `T/py/splice_positional_features.py` (writes `T/data/splice_positional.npz`) and `T/py/splice_model_boost.py` (writes `T/data/splice_model_boost.tsv`, log `T/logs/splice_model_boost.log`); penalty chosen per fold from 1, 10, 100, 1000 x 0.1, three seeds averaged.
- A two-feature "arrangement" model gives 0.392 and 0.396: `T/py/splice_model_expand.py` to `T/data/splice_model_expand.tsv`, five-fold CV by gene, seed 7, ridge 0.01. **OPEN:** this is the only P@1 in the section that is not LOCO, so it is not strictly comparable.

<!-- BEGIN AUTO:3.7:code -->
**Code.** `thesis_ch3/R/fig3_regroup.R`, lines 80 to 94 (the block that saves `fig3_06_splice_model_performance`), plus the shared preamble. This block also builds `fig3_05_splice_model`. The whole file (120 lines, sha256 11f48036ce52) is in the appendix and in `code/`.

Shared preamble:

```r
# fig3_regroup.R -- Figures 3.3 to 3.8 and 3.10, regrouped from the panels pub/'s scripts build.
#
# Each old figure's script is run in a sandbox (R/_harvest.R) for its panel objects. Panels are
# relettered and retitled for the new numbering. In-figure text is limited to labels that NAME
# something (series, reference lines, categories, schematic parts); statistics, sentences,
# sequences and numeric call-outs are removed here and live in the legend or the text.
#
#   3.3  <- make_splicing_environment.R A, B  + new C, D (../acceptor_position_scan.tsv)
#   3.4  <- make_splicing_environment.R C, D     3.7  <- make_figure_architecture.R, make_figure_rescue.R a
#   3.5  <- make_splice_model_final.R   A, B     3.8  <- make_figure_loco.R A, B, D
#   3.6  <- make_splice_model_final.R   C, D     3.12 <- make_figure_cnn_interpretability.R
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr)})
setwd("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3")
source("R/_harvest.R")
dir.create("out", showWarnings = FALSE)
INK <- "#1A1A1A"; GTX <- "#5A6068"
TT <- theme(plot.title = element_text(face = "bold", size = 11, colour = INK),
            plot.subtitle = element_blank(), plot.caption = element_blank())
save2 <- function(fig, name, w, h, dpi = 300) {
  ggsave(file.path("out", paste0(name, ".pdf")), fig, width = w, height = h, device = cairo_pdf, limitsize = FALSE)
  ggsave(file.path("out", paste0(name, ".png")), fig, width = w, height = h, dpi = dpi, bg = "white", limitsize = FALSE)
  cat("[out]", name, "\n")
}
```

Block for this figure:

```r
# ---------------------------------------------------------------- 3.5, 3.6 splice model
S <- harvest("make_splice_model_final.R")
s5a <- edit_text(S$pA, drop = "Within-gene evaluation",
                 relab = c("Sequence window" = "25 bins of 100 nt",
                           "U1 duplex density" = "U1 site density per bin",
                           "Maximum MaxEnt" = "strongest upstream 3′ splice-site score",
                           "Ridge \\(L2\\)" = "ridge logistic regression"))
s5b <- edit_text(S$pB, drop = c("-0.23 at", "^\\+0.30$"), relab = c("MaxEnt 3'ss" = "acceptor"))
save2((s5a + labs(title = "A   Features") + TT) / (s5b + labs(title = "B   Fitted coefficients by position") + TT) +
      plot_layout(heights = c(1.15, 1)), "fig3_05_splice_model", 7.4, 6.6)
s6c <- edit_text(S$pC, drop = "<mapped lab>", relab = c("random" = "chance"))
s6d <- edit_text(S$pD, drop = c("<mapped sprintf", "Win-rate 0.699"))
save2((s6c + labs(title = "A   Within-gene recall at rank k") + TT) | (s6d + labs(title = "B   Gene-centred scores") + TT),
      "fig3_06_splice_model_performance", 7.8, 3.6)
```
<!-- END AUTO:3.7:code -->

### Figure 3.8: What each model reads, and the network architecture

<!-- BEGIN AUTO:3.8:image -->
![Figure 3.8](figures/fig3_8.png)
<!-- END AUTO:3.8:image -->

**Shows**
- **a.** The sequence available to each model: the Chapter 2 baseline scores ten poly(A) elements in 30 nt windows at 10 nt stride across 500 nt; the Chapter 3 models read 2,500 nt of raw sequence.
- **b.** The architecture, with the dimensions computed from the layer parameters rather than asserted: 32 filters of width 19 with exponential activation, pooling by 10, 5 and 2 to 25 positions, each with a 378 nt receptive field at 100 nt stride, flattened to 2,400 values, one 64-unit dense layer, one output.

**OPEN**
- **The acceptor/donor colours are flipped between this figure and Figures 3.11 and 3.12.** Here the 3′ splice site is blue and the 5′ splice site orange (`make_figure_rescue.R:5,33`); in Figure 3.11b and throughout Figure 3.12 the acceptor is orange and the donor blue. Panel a is the only place in the chapter where the acceptor is blue, and neither legend explains the dot colours. Recolour it or add a clause.
- The legend says the baseline scores **ten** elements, but the magnified bar shows **four** labelled dots (`make_figure_rescue.R:36`, `CORE` has four entries). "Element positions are schematic" does not cover "four of ten shown".

**How:** panel a is the drawn `pA` of `PUB/make_figure_rescue.R` (which also reads `rescue_rate_genes.tsv` and `rescue_ranks.tsv` and asserts 6,583); panel b is `PUB/make_figure_architecture.R`, which asserts receptive field 378, stride 100 and 25 positions, mirroring `PUB/cnn_model_def.py`. Rebuild: `cd T && Rscript R/fig3_regroup.R`.

<!-- BEGIN AUTO:3.8:code -->
**Code.** `thesis_ch3/R/fig3_regroup.R`, lines 95 to 102 (the block that saves `fig3_07_architecture`), plus the shared preamble. The whole file (120 lines, sha256 11f48036ce52) is in the appendix and in `code/`.

Shared preamble:

```r
# fig3_regroup.R -- Figures 3.3 to 3.8 and 3.10, regrouped from the panels pub/'s scripts build.
#
# Each old figure's script is run in a sandbox (R/_harvest.R) for its panel objects. Panels are
# relettered and retitled for the new numbering. In-figure text is limited to labels that NAME
# something (series, reference lines, categories, schematic parts); statistics, sentences,
# sequences and numeric call-outs are removed here and live in the legend or the text.
#
#   3.3  <- make_splicing_environment.R A, B  + new C, D (../acceptor_position_scan.tsv)
#   3.4  <- make_splicing_environment.R C, D     3.7  <- make_figure_architecture.R, make_figure_rescue.R a
#   3.5  <- make_splice_model_final.R   A, B     3.8  <- make_figure_loco.R A, B, D
#   3.6  <- make_splice_model_final.R   C, D     3.12 <- make_figure_cnn_interpretability.R
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr)})
setwd("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3")
source("R/_harvest.R")
dir.create("out", showWarnings = FALSE)
INK <- "#1A1A1A"; GTX <- "#5A6068"
TT <- theme(plot.title = element_text(face = "bold", size = 11, colour = INK),
            plot.subtitle = element_blank(), plot.caption = element_blank())
save2 <- function(fig, name, w, h, dpi = 300) {
  ggsave(file.path("out", paste0(name, ".pdf")), fig, width = w, height = h, device = cairo_pdf, limitsize = FALSE)
  ggsave(file.path("out", paste0(name, ".png")), fig, width = w, height = h, dpi = dpi, bg = "white", limitsize = FALSE)
  cat("[out]", name, "\n")
}
```

Block for this figure:

```r
# ---------------------------------------------------------------- 3.7 windows and architecture
Ar <- harvest("make_figure_architecture.R")
R  <- harvest("make_figure_rescue.R")
arch <- edit_text(Ar$p, drop = "one output bin sees")
save2((R$pA + labs(title = "A   Sequence read by the baseline and the extended models") + TT) /
      (arch + labs(title = "B   Network architecture") + TT + theme(plot.background = element_rect(fill = "white", colour = NA))) +
      plot_layout(heights = c(1, 0.72)), "fig3_07_architecture", 9.4, 4.8)
```
<!-- END AUTO:3.8:code -->

### Figure 3.9: Sequence network against the splice-feature model

<!-- BEGIN AUTO:3.9:image -->
![Figure 3.9](figures/fig3_9.png)
<!-- END AUTO:3.9:image -->

**Shows**, blue network and orange splice model, on identical LOCO folds:
- **a.** ROC and precision-recall pooled over all candidates, base rate 0.100.
- **b.** Fraction of genes whose real end ranks in the top k, with 95% bootstrap bands over genes.
- **c.** P@1 in hexagonal bins of gene length against number of candidates, for the baseline, the splice model and the network side by side. Hexagons drawn where at least 12 genes fall (20 bins).
  - **The legend's "15,560 genes" is wrong here: 15,475 are plotted.** `make_figure_loco.R:138` filters `!is.na(gene_length), gene_length > 0` first, and 85 of the 15,560 rows in `wg_per_gene.tsv` have no GENCODE span (verified). Figure 3.14 panel b does use all 15,560. Add "(15,475 with a GENCODE span)" or drop the count.

**Why:** the network reading raw sequence beats the hand-built splice model, and panel c shows where: both degrade with more candidates and longer genes, but the network degrades more slowly.

**Key numbers**
- Analysis set P@1: network 0.681 [0.669, 0.695], splice model 0.465, chance 0.287.
- Win-rate: 0.847 against 0.699.
- AUROC / AUPRC: 0.815 / 0.407 against 0.661 / 0.162; base rate 0.1002.
- Per-chromosome network win-rate 0.820 to 0.897.
- Paired: the network alone wins 1,515 genes, the splice model alone 457 (chi2 = 567, P = 3e-125); win-rate difference 0.148, z = 22.6, r = 0.43 (`stats_ch3` rows C1 and C3).

**How:** harvested from `PUB/make_figure_loco.R` (panels `pA1`, `pA2`, `pB`, `pD`), reading `figLC_curves.tsv`, `figLC_summary.tsv`, `figLC_recall.tsv`, `wg_per_gene.tsv`, `wg_by_length.tsv`, `loco_shuffle.tsv`, `cnn_wholegenome_p1.tsv`.

| Output | Script | Job |
|---|---|---|
| `cnn_loco_scores.npz` | `cnn_loco_train.py` via `cnn_loco.sbatch` | 57877643 |
| `figLC_*.tsv`, `loco_shuffle.tsv` | `loco_figure_data.py` via `loco_figure.sbatch` (splice-model LOCO refit, seed 42, 2,000 bootstrap resamples) | 57884326 |
| `cnn_wg_scores.npz`, `cnn_wholegenome_p1.tsv` | `cnn_wg_loco.py` | 57904993 |
| `wg_per_gene.tsv`, `wg_by_length.tsv` | `wg_by_length.py` via `wg_by_length.sbatch` (gene length from GENCODE v49) | 57906502 |

Rebuild: `cd T && Rscript R/fig3_regroup.R`. **OPEN:** the header of `make_figure_loco.R` describes panel d incorrectly.

<!-- BEGIN AUTO:3.9:code -->
**Code.** `thesis_ch3/R/fig3_regroup.R`, lines 103 to 113 (the block that saves `fig3_08_cnn_vs_splice`), plus the shared preamble. The whole file (120 lines, sha256 11f48036ce52) is in the appendix and in `code/`.

Shared preamble:

```r
# fig3_regroup.R -- Figures 3.3 to 3.8 and 3.10, regrouped from the panels pub/'s scripts build.
#
# Each old figure's script is run in a sandbox (R/_harvest.R) for its panel objects. Panels are
# relettered and retitled for the new numbering. In-figure text is limited to labels that NAME
# something (series, reference lines, categories, schematic parts); statistics, sentences,
# sequences and numeric call-outs are removed here and live in the legend or the text.
#
#   3.3  <- make_splicing_environment.R A, B  + new C, D (../acceptor_position_scan.tsv)
#   3.4  <- make_splicing_environment.R C, D     3.7  <- make_figure_architecture.R, make_figure_rescue.R a
#   3.5  <- make_splice_model_final.R   A, B     3.8  <- make_figure_loco.R A, B, D
#   3.6  <- make_splice_model_final.R   C, D     3.12 <- make_figure_cnn_interpretability.R
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr)})
setwd("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3")
source("R/_harvest.R")
dir.create("out", showWarnings = FALSE)
INK <- "#1A1A1A"; GTX <- "#5A6068"
TT <- theme(plot.title = element_text(face = "bold", size = 11, colour = INK),
            plot.subtitle = element_blank(), plot.caption = element_blank())
save2 <- function(fig, name, w, h, dpi = 300) {
  ggsave(file.path("out", paste0(name, ".pdf")), fig, width = w, height = h, device = cairo_pdf, limitsize = FALSE)
  ggsave(file.path("out", paste0(name, ".png")), fig, width = w, height = h, dpi = dpi, bg = "white", limitsize = FALSE)
  cat("[out]", name, "\n")
}
```

Block for this figure:

```r
# ---------------------------------------------------------------- 3.8 network vs splice model
L <- harvest("make_figure_loco.R")
a1 <- edit_text(L$pA1, relab = c("^Sequence CNN" = "Sequence CNN", "^Splice model" = "Splice model"))
a2 <- edit_text(L$pA2, drop = "^AUPRC", relab = c("base rate" = "base rate"))
b8 <- edit_text(L$pB, drop = "recall@1", relab = c("^Splice model" = "Splice model"))
d8 <- edit_text(L$pD, drop = "<mapped lab>")
top <- ((a1 + labs(title = "A   Pooled ROC and precision-recall") + TT) | (a2 + TT) |
        (b8 + labs(title = "B   Within-gene recall at rank k") + TT)) + plot_layout(widths = c(0.78, 0.78, 1.44))
bot <- d8 + labs(title = "C   Annotated end ranked first, by gene length and number of candidates") + TT
save2((top / bot) + plot_layout(heights = c(1, 0.95)), "fig3_08_cnn_vs_splice", 10.6, 7.4)
```
<!-- END AUTO:3.9:code -->

### Figure 3.10: Robustness

<!-- BEGIN AUTO:3.10:image -->
![Figure 3.10](figures/fig3_10.png)
<!-- END AUTO:3.10:image -->

**Shows** (panels a, b and d on the fixed split of 1,623 held-out genes / 14,198 candidates; panel c on LOCO):
- **a.** P@1 as the window is shortened from the upstream end, with an open circle for a symmetric -250 to +250 window. Points are the mean of five seeds, bars the range.
- **b.** P@1 of the five networks, **not refitted**, on held-out windows where strong 3′ splice sites, strong 5′ splice sites or both were modified (filled) or the same number of random positions were modified (open). Left, bases permuted; right, resampled from the window's composition. Sites overlapping -40 to +40 nt were left alone.
- **c.** Within-gene win-rate on real sequence against dinucleotide-preserving shuffles (3 replicates).
- **d.** Within-gene win-rate across eight seeds, sequence alone against sequence plus the MaxEnt acceptor score, with a dashed line for the acceptor score alone.

**Why:** A shows the long upstream context is doing real work; B asks whether the specific splice sites carry it; C bounds how much is composition; D asks whether the hand-built acceptor feature adds anything to raw sequence.

**Key numbers**
- Context sweep P@1: 0.479 (-100), 0.518 (-250), 0.570 (-500), 0.629 (-1000), 0.653 (-1500), 0.649 (-2000); symmetric -250 to +250 gives 0.506. Performance saturates near -1,500.
- Shuffling: win-rate 0.847 to 0.629, and 0.699 to 0.544 for the splice model.
- Seeds: 0.839 +/- 0.009 with the acceptor score against 0.832 +/- 0.013 without, Welch P = 0.22, i.e. **the hand-built acceptor feature adds nothing** to raw sequence. The acceptor score alone gives win-rate 0.620, P@1 0.397.
- Chance P@1 on this split is 0.292 (hard-coded in the script).

**How**
- Panel a: `T/data/context_sweep.tsv` from `T/py/context_sweep.py` via `context_sweep.sbatch`, job 61822760 (GPU). Upstream starts -100, -250, -500, -1000, -1500, -2000, all to +500, plus -250 to +250; seeds 42-46.
- Panel b: `T/data/splice_scramble.tsv` (job 61824148) and `T/data/splice_resample.tsv` (job 61825548) from `T/py/splice_scramble.py` in permute and resample modes; up to 50 re-modifications, per-window seed 1000 + index, the five `acc_arch_base_s4{2..6}.pt` checkpoints. Tests use `T/data/scramble_logits.npz` from `T/py/patching.py`, job 61827403.
- Panel c: `PUB/loco_shuffle.tsv`.
- Panel d: `PUB/acceptor_arch_train.tsv` (8 seeds, job 58016143) plus `T/data/acceptor_alone_test.tsv` for the dashed line. **OPEN: no script writes `acceptor_alone_test.tsv`.**
- Rebuild: `cd T && Rscript R/fig3_09_robustness.R`

**NOTE.** Panel c's y-axis reads "Within-gene win-rate" and panel d's "within-gene win-rate" (c is harvested, so the capitalisation differs). Three numbers here cannot be checked against a plotted table: the panel b P-values ("all P >= 0.12", from `scramble_logits.npz`), the chance line 0.292 (hardcoded as `CH<-0.292`), and the Figure 3.9 McNemar and Wilcoxon statistics (text-only, from `stats_ch3`).

<!-- BEGIN AUTO:3.10:code -->
**Code.** `thesis_ch3/R/fig3_09_robustness.R`, 74 lines, sha256 `feddef78cc3f`. Reproduced verbatim; also in `code/fig3_09_robustness.R`.

```r
# fig3_09_robustness.R -- Figure 3.9, what the network's ranking depends on.
#
# A  context length: one network retrained at seven windows        data/context_sweep.tsv
# B  strong splice sites scrambled in place vs matched random scrambles   data/splice_scramble.tsv
# C  dinucleotide shuffle of the whole window (LOCO)               make_figure_loco.R panel C, harvested
# D  seed spread, with and without the MaxEnt acceptor score as an extra input
#                                                                   ../acceptor_arch_train.tsv
# A, B and D are on the fixed gene split (1,623 held-out genes); C is leave-one-chromosome-out.
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr)})
setwd("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3")
source("R/_harvest.R")
INK<-"#1A1A1A"; GTX<-"#5A6068"; GREY<-"#8A8F98"; CAN<-"#0B55A0"; LIGHT<-"#C9D6E6"
TH<-theme_classic(base_size=10)+theme(plot.title=element_text(face="bold",size=11,colour=INK),
  axis.text=element_text(colour=GTX),plot.subtitle=element_blank(),plot.caption=element_blank())
CH<-0.292   # chance P@1 on the 1,623 held-out genes (mean 1/candidates)
ACC<-read.delim("data/acceptor_alone_test.tsv"); acc_wr<-ACC$value[ACC$metric=="win_rate"]

# A
D<-read.delim("data/context_sweep.tsv")
S<-D%>%group_by(context,start,end)%>%summarise(m=mean(test_p_at_1),lo=min(test_p_at_1),hi=max(test_p_at_1),.groups="drop")%>%mutate(up=-start)
U<-S%>%filter(end==500); B5<-S%>%filter(context=="span500")
pA<-ggplot(U,aes(up,m))+
  geom_hline(yintercept=CH,linetype="22",colour=GREY)+
  annotate("text",x=2000,y=CH+0.012,label="chance",hjust=1,colour=GREY,size=2.8)+
  geom_line(colour=CAN,linewidth=0.7)+geom_errorbar(aes(ymin=lo,ymax=hi),width=45,colour=CAN,linewidth=0.35)+
  geom_point(colour=CAN,size=1.9)+
  geom_errorbar(data=B5,aes(x=330,ymin=lo,ymax=hi),width=45,colour=INK,linewidth=0.35)+
  geom_point(data=B5,aes(x=330,y=m),shape=21,fill="white",colour=INK,size=2.2,stroke=0.6)+
  scale_x_continuous(breaks=c(100,250,500,1000,1500,2000))+scale_y_continuous(limits=c(0.25,0.7))+
  labs(x="upstream sequence read (nt), downstream to +500",y="precision at 1",title="A   Window length")+TH

# B
SC<-rbind(read.delim("data/splice_scramble.tsv")%>%mutate(mode="bases permuted"),
          read.delim("data/splice_resample.tsv")%>%mutate(mode="bases resampled"))
lv<-c("acc","don","both")
B<-SC%>%filter(arm!="intact")%>%mutate(target=sub("ctrl_","",arm),kind=ifelse(grepl("^ctrl",arm),"random positions","splice sites"))%>%
  group_by(mode,target,kind)%>%summarise(m=mean(test_p_at_1),lo=min(test_p_at_1),hi=max(test_p_at_1),.groups="drop")%>%
  mutate(target=factor(target,levels=lv,labels=c("acceptors","donors","both")),kind=factor(kind,levels=c("splice sites","random positions")))
I0<-SC%>%filter(arm=="intact",mode=="bases permuted")%>%summarise(m=mean(test_p_at_1))
pB<-ggplot(B,aes(target,m,shape=kind,fill=kind))+
  geom_hline(yintercept=I0$m,colour=CAN,linewidth=0.4)+
  annotate("text",x=-Inf,y=I0$m+0.014,label="  intact",hjust=0,colour=CAN,size=2.8)+
  geom_hline(yintercept=CH,linetype="22",colour=GREY)+
  annotate("text",x=-Inf,y=CH+0.014,label="  chance",hjust=0,colour=GREY,size=2.8)+
  geom_errorbar(aes(ymin=lo,ymax=hi),position=position_dodge(0.45),width=0.15,linewidth=0.35,colour=INK)+
  geom_point(position=position_dodge(0.45),size=2.4,colour=INK,stroke=0.6)+
  scale_shape_manual(values=c("splice sites"=21,"random positions"=21),name=NULL)+
  scale_fill_manual(values=c("splice sites"=CAN,"random positions"="white"),name=NULL)+
  scale_y_continuous(limits=c(0.25,0.7))+
  facet_wrap(~mode)+
  labs(x="sites modified",y="precision at 1",title="B   Strong splice sites modified in place")+TH+
  theme(strip.background=element_blank(),strip.text=element_text(size=9,colour=INK))+
  theme(legend.position=c(0.98,0.35),legend.justification=c(1,0.5),legend.key.size=unit(3,"mm"),legend.text=element_text(size=8))

# C
L<-harvest("make_figure_loco.R")
pC<-edit_text(L$pC,drop=c("<mapped lab>","filled = real"))+labs(title="C   Dinucleotide shuffle",subtitle=NULL,caption=NULL)+TH

# D
AT<-read.delim("../acceptor_arch_train.tsv")%>%mutate(variant=factor(variant,levels=c("base","aux"),
      labels=c("sequence only","sequence +\nacceptor score")))
M<-AT%>%group_by(variant)%>%summarise(m=mean(test_within_gene),.groups="drop")
pD<-ggplot(AT,aes(variant,test_within_gene))+
  geom_hline(yintercept=acc_wr,linetype="22",colour=GREY)+
  annotate("text",x=-Inf,y=acc_wr+0.01,label="  acceptor score alone",hjust=0,colour=GREY,size=2.8)+
  geom_point(position=position_jitter(width=0.08,seed=1),colour=CAN,size=1.8,alpha=0.85)+
  geom_errorbar(data=M,aes(y=m,ymin=m,ymax=m),width=0.4,colour=INK,linewidth=0.6)+
  scale_y_continuous(limits=c(0.58,0.88))+
  labs(x=NULL,y="within-gene win-rate",title="D   Eight seeds per network")+TH

fig<-(pA|pB)/(pC|pD)
ggsave("out/fig3_09_robustness.pdf",fig,width=8.6,height=7,device=cairo_pdf)
ggsave("out/fig3_09_robustness.png",fig,width=8.6,height=7,dpi=300,bg="white")
cat("[out] out/fig3_09_robustness.{pdf,png}\n")
```
<!-- END AUTO:3.10:code -->

### Figure 3.11: First-layer filters and splice-site implantation

<!-- BEGIN AUTO:3.11:image -->
![Figure 3.11](figures/fig3_11.png)
<!-- END AUTO:3.11:image -->

**Shows**, for the seed-42 network on the held-out genes (1,623 real ends, 12,575 competitors):
- **a.** A logo for each of the 32 first-layer filters, estimated from sites activating above 30% of that filter's maximum, with the contribution of each filter in each 100 nt band.
- **b.** Median predicted probability of the 12,575 competitor windows after writing tandem copies of a 3′ splice site (score3 10.28) or a 5′ splice site CAGGTGAGT (score5 10.67) at -230 nt, with 95% percentile bootstrap bars (2,000 resamples), the unmodified median as a dashed line, and a grey band for 20 mononucleotide shuffles and the nine base derangements.

**Why:** B is the causal version of Figure 3.5. Writing an acceptor upstream pushes competitors toward "real"; writing a donor pushes them away.

**Key numbers:** baseline median 0.62505 [0.6157, 0.6336], which the legend and text round to 0.63; acceptor x1 0.799 [0.792, 0.805] and x4 0.985; donor x1 0.593 [0.584, 0.604] and x8 0.331. Quoting 0.625 would be more faithful and costs nothing.

**OPEN, all in the legend rather than the figure**
- **The contribution is defined with the wrong sign.** The legend says "the mean change in logit on silencing that filter". The code computes `d = intact - occluded` (`cnn_posmap_occlusion.py:115`, verified), i.e. the **drop**, which is the negative of the "change on silencing". The colour sentence (blue = raises real ends) is right, and Methods 3.4.8 already words it correctly as "the mean drop in logit". Copy the Methods wording into the legend.
- **The grey band is not a range.** The legend calls it the "range of 20 mononucleotide shuffles and the nine base derangements". The code plots each control's mean +/- SD/sqrt(n) and takes the envelope of those two intervals, which is much narrower than the range across draws. Reword to "the two control means with their standard errors".
- **Panel a's logos and heatmap come from different samples.** The logos and their "N sites" counts use a random 4,000 of the 14,198 held-out windows (`cnn_interpret2.py:75`, `filter_pwms(n_seq=4000)`), while the occlusion heatmap beside them uses all 14,198. So filter 27's "24,668 sites" is a count over a 28% subsample. One clause in the legend fixes it.
- **The cross-reference to Section 3.4.8 is dangling**, and the two logos come from different recipes: the donor logo is exhaustive over all 262,144 9-mers by U1 duplex energy (n = 115), while the acceptor logo is Metropolis-Hastings sampled **against the network itself** over 23-mers with the AG pinned and score3 >= 10 (n = 668). That asymmetry is the point of 3.2.9 and deserves a sentence.

**The 10.86 label trap, resolved.** CAGGTGAGT scores 10.67 by maxentpy; 10.86 is CAGGTAAGT's score. **The figure carries no score label at all** on panel b (the facets read only "3' splice site" and "5' splice site"), because `fig3_regroup.R` strips the text layer. The stale `score5 10.86` survives only in the harvested source, `make_figure_cnn_interpretability.R` (visible in the appendix listing). The chapter legend's 10.67 is correct and nothing in the figure contradicts it, so there is nothing to fix in the figure. Leave the source string alone unless the pub/ script is ever used directly.

**How:** harvested from `PUB/make_figure_cnn_interpretability.R`.

| Input | Producer | Job |
|---|---|---|
| `cnnmain_posmap.tsv`, `cnnmain_logos.tsv`, `cnnmain_notes.tsv` | `figure_cnn_main_data.py`, collecting `cnn_posmap_occlusion.py` and `cnn_interpret2.py --legacy --model cnn_model.pt` | - |
| `panelB_prob_ci.tsv` | `panelB_prob_ci.py` via `pbci.sbatch` (20 shuffles, 9 derangements, 2,000 bootstrap resamples, offset -230, seed 42) | 58078652 / 58078806 |
| `gia_dose_matched_all.tsv` | `acceptor_all_numbers.py` | - |
| `gia_dose_evolved.tsv` | `gia_dose_evolved.py` | 57973862 |
| `me3_pwm.tsv` | `me3_pwm.py` (seed 42) | - |
| `me3_pwm_noc.tsv` | presumably `me3_pwm.py --no_c`; **OPEN: no recorded run** |  |
| `u1_pwm.tsv` | `u1_pwm.py` | - |
| `canon_acc_pwm*.tsv` | `model_acc_pwm.py` via `canpwm.sbatch` | 58079964 / 58080167 / 58083124 |

- The -230 nt offset was an unrecorded default that happens to sit at the response peak (see Figure 3.12a).
- Rebuild: `cd T && Rscript R/fig3_regroup.R`

**Text-only results in section 3.2.7**
- A MaxEnt-optimal acceptor gives -0.138 (z = -0.5, P = 0.66), while a sequence optimised against the network itself gives +1.880 (z = 15.4): `PUB/acceptor_agdep_final.tsv` from `acceptor_agdep_search.py` via `agdep.sbatch` (500 permutations, 3 restarts). A pyrimidine-tract acceptor gives +0.830 (`PUB/acceptor_floor_sweep.tsv`, arm `pyr80_noG`). The network responds to U-richness, not to MaxEnt score.
- Probing plus LEACE concept erasure (`T/py/concept_erasure.py` in "within" mode, jobs 61848415 and 61848542, writing `T/data/concept_erasure_within.tsv`): the network encodes base composition and U1-site density, not MaxEnt scores, PAS or annotated splice sites. Concepts must be centred within class, or erasure merely removes the label.
- Native-element removal and saturation mutagenesis (`T/py/interpret_deep.py`, job 61852055, writing `T/data/deep_strength.tsv`, `deep_necessity.tsv`, `deep_ism_*.tsv`; 200 sites per group, seed 0): removing native UGUA costs 0.23 to 0.25 logit, the largest single effect; the poly(A) signal costs 0.04, the smallest. Top ISM 6-mers contain UGUA; the worst are YAG intron ends.
- **OPEN:** the chapter says "mutating the AG to AC cost only 0.034", but in `acceptor_agdep_final.tsv` (row `prev_model_me3`) intact scores 1.8803 and the AC mutant 1.9147, so the mutant is 0.034 **higher**. Rewording is free: it strengthens the chapter's own argument that the network is not reading the AG as a splice-site signal.

<!-- BEGIN AUTO:3.11:code -->
**Code.** `thesis_ch3/R/fig3_regroup.R`, lines 114 to 120 (the block that saves `fig3_12_interpretation`), plus the shared preamble. The whole file (120 lines, sha256 11f48036ce52) is in the appendix and in `code/`.

Shared preamble:

```r
# fig3_regroup.R -- Figures 3.3 to 3.8 and 3.10, regrouped from the panels pub/'s scripts build.
#
# Each old figure's script is run in a sandbox (R/_harvest.R) for its panel objects. Panels are
# relettered and retitled for the new numbering. In-figure text is limited to labels that NAME
# something (series, reference lines, categories, schematic parts); statistics, sentences,
# sequences and numeric call-outs are removed here and live in the legend or the text.
#
#   3.3  <- make_splicing_environment.R A, B  + new C, D (../acceptor_position_scan.tsv)
#   3.4  <- make_splicing_environment.R C, D     3.7  <- make_figure_architecture.R, make_figure_rescue.R a
#   3.5  <- make_splice_model_final.R   A, B     3.8  <- make_figure_loco.R A, B, D
#   3.6  <- make_splice_model_final.R   C, D     3.12 <- make_figure_cnn_interpretability.R
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr)})
setwd("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3")
source("R/_harvest.R")
dir.create("out", showWarnings = FALSE)
INK <- "#1A1A1A"; GTX <- "#5A6068"
TT <- theme(plot.title = element_text(face = "bold", size = 11, colour = INK),
            plot.subtitle = element_blank(), plot.caption = element_blank())
save2 <- function(fig, name, w, h, dpi = 300) {
  ggsave(file.path("out", paste0(name, ".pdf")), fig, width = w, height = h, device = cairo_pdf, limitsize = FALSE)
  ggsave(file.path("out", paste0(name, ".png")), fig, width = w, height = h, dpi = dpi, bg = "white", limitsize = FALSE)
  cat("[out]", name, "\n")
}
```

Block for this figure:

```r
# ---------------------------------------------------------------- 3.12 interpretation
I <- harvest("make_figure_cnn_interpretability.R")
pa <- I$pA; pa[[2]] <- pa[[2]] + labs(title = "A   First-layer filters and their positional contribution") + TT
pb <- I$pB
pb[[2]] <- edit_text(pb[[2]], drop = c("untouched competitor", "<mapped txt>")) +
  labs(title = "B   Splice-site implantation into competitor windows") + TT
save2(pa / pb + plot_layout(heights = c(4.3, 1.35)), "fig3_12_interpretation", 9.8, 13.6, dpi = 220)
```
<!-- END AUTO:3.11:code -->

### Figure 3.12: In silico tests of splicing and cleavage signals

<!-- BEGIN AUTO:3.12:image -->
![Figure 3.12](figures/fig3_12.png)
<!-- END AUTO:3.12:image -->

**Shows**, using the five networks not refitted, on held-out genes:
- **a.** Effect of implanting the 3′ splice site (1 or 4 tandem copies) or the 5′ splice site CAGGTGAGT (1 or 8 copies) at each offset across the 12,575 competitor windows, minus the effect of three mononucleotide shuffles, with -230 nt marked.
- **b.** Effect of replacing the annotated terminal-exon 3′ splice site (23 nt = 20 intronic + 3 exonic; or 50 nt = 47 + 3) in real-end windows with bases drawn from that window's composition, minus the same replacement at three random positions, by distance.
- **c.** The same for the nearest annotated upstream 5′ splice site (9 nt = 3 exonic + 6 intronic; or 20 nt = 3 + 17) in 2,223 competitor windows.

**Why:** A gives the positional profile of the implantation effect; b and c are the knockout direction, on *annotated* sites in their native context.

**Key numbers**
- 50 nt acceptor knockout: -0.086 logit overall [0.069, 0.102]; 0.155 within 250 nt; 0.035 beyond 1 kb, so the effect is distance-dependent.
- Donor knockout: +0.030 overall, +0.061 at 250-500 nt upstream, i.e. the opposite sign, as predicted.
- P@1 falls 0.730 to 0.713 on acceptor knockout.
- The full logit range across these manipulations is 2.57 (`stats_ch3` rows Hu, Hc, Hd).

**How:** panel a from `T/data/implant_position.tsv` (`T/py/implant_position.py`, job 61829002; offsets -1,900 to +400 every 100 nt plus -230, 3 shuffles at seed 0, 500 bootstrap resamples). Panels b and c from `T/data/figm_knockout.tsv`, written by `T/py/fig_mech_data.py` from `T/data/mechanism.tsv` (`T/py/mechanism.py`, jobs 61829888 and 61829944, the later probably final; GENCODE v49 protein-coding transcripts, seed 0, 1,000 bootstrap resamples).

Rebuild: `sbatch T/py/implant_position.sbatch; sbatch T/py/mechanism.sbatch; python T/py/fig_mech_data.py; cd T && Rscript R/fig3_13_mechanism.R`

**OPEN**
- **The 50 nt acceptor replacement uses 1,003 windows, not 1,010** (verified in `mechanism.tsv`: 23-mer `n 1010`, 50-mer `n 1003`). The 7 lost sites are all in the -2000 to -1000 bin, where the 47 nt intronic reach runs off the window (349 -> 342). Both the legend and section 3.2.8 attach 1,010 to the 50 nt result. `mechanism.tsv` is itself inconsistent: its P@1 row is labelled `genes 1010` while reporting the 50-mer.
- `fig3_13_mechanism.R` builds five panels and saves only a|b|c; the two unused panels still require `T/data/figm_architecture.tsv` and `figm_scan.tsv` to exist.

<!-- BEGIN AUTO:3.12:code -->
**Code.** `thesis_ch3/R/fig3_13_mechanism.R`, 40 lines, sha256 `3ccdcab2254d`. Reproduced verbatim; also in `code/fig3_13_mechanism.R`.

```r
# fig3_13_mechanism.R -- Figure 3.13, in silico tests of mechanisms coupling splicing and 3' end formation.
# IN data/implant_position.tsv (py/implant_position.py), data/figm_{knockout,architecture,scan}.tsv (py/mechanism.py)
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr)})
setwd("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3")
INK<-"#1A1A1A"; GTX<-"#5A6068"; GREY<-"#8A8F98"; ACC<-"#E8A33D"; DON<-"#08519C"
TH<-theme_classic(base_size=8.5)+theme(plot.title=element_text(face="bold",size=10.5,colour=INK),axis.text=element_text(colour=GTX),
      legend.key.size=unit(3,"mm"),legend.text=element_text(size=7.5),legend.title=element_blank())
P<-read.delim("data/implant_position.tsv")%>%mutate(lab=paste0(ifelse(construct=="acceptor","3′ splice site","5′ splice site")," ×",copies))
pA<-ggplot(P,aes(offset,diff,colour=lab))+geom_hline(yintercept=0,colour=GREY,linewidth=0.3)+
  geom_vline(xintercept=-230,linetype="22",colour=GREY,linewidth=0.3)+
  geom_ribbon(aes(ymin=diff_lo,ymax=diff_hi,fill=lab),alpha=0.25,colour=NA)+geom_line(linewidth=0.6)+geom_point(size=0.9)+
  scale_colour_manual(values=c("3′ splice site ×1"="#F2C58A","3′ splice site ×4"=ACC,"5′ splice site ×1"="#8FB0D3","5′ splice site ×8"=DON))+
  scale_fill_manual(values=c("3′ splice site ×1"="#F2C58A","3′ splice site ×4"=ACC,"5′ splice site ×1"="#8FB0D3","5′ splice site ×8"=DON),guide="none")+
  labs(x="offset of the implanted sequence (nt)",y="change in competitor logit\n(construct minus shuffled control)",title="A   Implantation by position")+TH+theme(legend.position="bottom")+guides(colour=guide_legend(nrow=2))
K<-read.delim("data/figm_knockout.tsv")%>%filter(bin!="all")%>%
  mutate(bin=factor(sub("site at ","",sub(" nt","",bin)),levels=c("-2000 to -1000","-1000 to -500","-500 to -250","-250 to -60")),
         size=factor(size,levels=c("9 nt","20 nt","23 nt","50 nt")))
kplot<-function(e,ttl,col){ ggplot(K%>%filter(experiment==e),aes(bin,diff,fill=size))+geom_hline(yintercept=0,colour=GREY,linewidth=0.3)+
  geom_col(position=position_dodge(0.75),width=0.7,colour=INK,linewidth=0.2)+
  geom_errorbar(aes(ymin=lo,ymax=hi),position=position_dodge(0.75),width=0.15,linewidth=0.3)+
  scale_fill_manual(values=col)+labs(x="position of the annotated site (nt)",y="change in logit\n(knockout minus control)",title=ttl)+TH+
  theme(legend.position="bottom",axis.text.x=element_text(size=7)) }
pB<-kplot("terminal-exon 3' splice site","B   Terminal-exon 3′ splice site removed",c("23 nt"="#F2C58A","50 nt"=ACC))
pC<-kplot("upstream 5' splice site","C   Upstream 5′ splice site removed",c("9 nt"="#8FB0D3","20 nt"=DON))
Ar<-read.delim("data/figm_architecture.tsv")
pD<-ggplot(Ar,aes(factor(donor_offset),interaction,fill=arrangement))+geom_hline(yintercept=0,colour=GREY,linewidth=0.3)+
  geom_col(width=0.65,colour=INK,linewidth=0.2)+
  scale_fill_manual(values=c("internal exon"="#C9D6E6","intron, then terminal exon"="#6C7A89"))+
  labs(x="offset of the 5′ splice site; 3′ splice site at -300 (nt)",y="interaction (both - each alone)",title="D   Exon architecture")+
  TH+theme(legend.position="bottom")
S<-read.delim("data/figm_scan.tsv")%>%mutate(element=recode(element,"TTGTAC (UGUA)"="TTGTAC (UGUA)","AATAAA (PAS)"="AATAAA"))
pE<-ggplot(S,aes(offset,diff,colour=element))+geom_hline(yintercept=0,colour=GREY,linewidth=0.3)+
  geom_vline(xintercept=0,linetype="22",colour=INK,linewidth=0.3)+geom_line(linewidth=0.6)+geom_point(size=0.8)+
  scale_colour_manual(values=c("TTGTAC (UGUA)"="#2E7D32","AATAAA"="#8E44AD"))+
  labs(x="offset of the implanted 6-mer (nt)",y="change in competitor logit\n(6-mer minus shuffled control)",title="E   Core polyadenylation elements")+
  TH+theme(legend.position="bottom")
fig<-(pA|pB|pC)
ggsave("out/fig3_13_mechanism.pdf",fig,width=10.4,height=3.4,device=cairo_pdf)
ggsave("out/fig3_13_mechanism.png",fig,width=10.4,height=3.4,dpi=300,bg="white")
cat("[out] out/fig3_13_mechanism.{pdf,png}\n")
```
<!-- END AUTO:3.12:code -->

### Figure 3.13: RNA-binding protein motif screen

<!-- BEGIN AUTO:3.13:image -->
![Figure 3.13](figures/fig3_13.png)
<!-- END AUTO:3.13:image -->

**Shows**, for each of 217 human proteins in CISBP-RNA, the change in logit when sequences sampled from its motif are written into 3,000 held-out competitor windows, minus the change for permutations of the same bases, averaged over the five networks. Motifs were written at -1,000, -300, -200, -100 and +50 nt, and each protein is shown at its largest-magnitude motif and offset. Proteins are ranked by effect, labelled with protein and consensus, coloured by family.

**Why:** an unbiased, database-sourced replacement for hand-picked motifs. The user's rule is that sequences come from databases, never from hand, and that provenance is always recorded.

**Key numbers:** NUDT21 (CFIm25, the UGUA reader) +0.075, sixth overall; SRSF7 +0.092; U1A +0.065; U1-70K +0.060; CstF64 -0.007; U2AF65 and PTB +0.015.

**How:** `T/py/implant_cisbp.py` via `implant_cisbp.sbatch`, job 61851077, reading `CLEAN/cisbp_rna.meme` (544 matrices), trimming to columns of at least 0.25 bits, dropping matrices under 4 columns, 3,000 windows at seed 0, 2 permutations as control, 500 bootstrap resamples. It writes `T/data/implant_cisbp.tsv` (234 names); `T/R/fig3_implant_panel.R` strips "2.00" and clone or ENSG IDs, leaving 217, and as a side effect writes `T/data/implant_cisbp_best.tsv`.

Rebuild: `sbatch T/py/implant_cisbp.sbatch; cd T && Rscript R/fig3_implant_panel.R`

**OPEN**
- **212, not 214.** The text says "214 proteins changed the score"; 212 of the 217 have a CI excluding zero (verified) and no code produces 214.
- **"by at most 0.15" is exceeded.** DAZ4 is -0.1548. Say 0.16, or name DAZ4.
- **"the motifs that lowered it most were those of the hnRNP A/B family" is not what the figure shows.** The four most negative are DAZ4 -0.155, HNRNPA0 -0.124, HNRNPA1L2 -0.118, HNRNPA3 -0.106; ranks 1, 5 (DAZ1), 7 (NOVA2) and 8 (SART3) are not hnRNP A/B, and DAZ4 is labelled in grey at the far left where a reader will see it. The underlying argument survives, because DAZ4, DAZ1, SART3 and MSI1 all carry UAG or UUAG. One sentence fixes it: "the motifs that lowered it most all contain UAG, and include the hnRNP A/B family".
- **MSI1 and DAZAP1 are miscoloured as hnRNP A/B.** `fig3_implant_panel.R:9` assigns the family by the regex `^HNRNPA|^DAZAP1$|^MSI`. Musashi-1 is an RRM protein of the Musashi family; DAZAP1 is also outside the A/B family. A future MSI2 would be absorbed silently.
- **The top-ranked protein overall, MKRN1 at +0.1155, is never mentioned**, so both extremes of the figure are unmentioned grey "other" proteins.
- **NUDT21's consensus is given two ways:** the text says UGUA, the figure label and the data say AAUGUAA. The value (+0.0749) and rank (6 of 217) are right.
- **The text and the figure use different protein names** (U1A / U1-70K / CFIm25 / CstF64 / U2AF65 against SNRPA / SNRNP70 / NUDT21 / CSTF2 / U2AF2) with no bridge in the legend. U1A is +0.0646, which the text rounds to +0.07; +0.06 is closer.
- Nothing in the code checks that the proteins are human; that rests on the contents of the `.meme` file.

<!-- BEGIN AUTO:3.13:code -->
**Code.** `thesis_ch3/R/fig3_implant_panel.R`, 25 lines, sha256 `ee1b9349168a`. Reproduced verbatim; also in `code/fig3_implant_panel.R`.

```r
# fig3_implant_panel.R -- response of the network to the measured binding motif of every RNA-binding
# protein in CISBP-RNA. No computation: reads data/implant_cisbp.tsv (py/implant_cisbp.py).
# Per protein, the motif and offset with the largest absolute effect; proteins ranked by that effect.
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
```
<!-- END AUTO:3.13:code -->

### Figure 3.14: How often the annotated 3′ end is ranked first

<!-- BEGIN AUTO:3.14:image -->
![Figure 3.14](figures/fig3_14.png)
<!-- END AUTO:3.14:image -->

**Shows**
- **a.** P@1 per model on the full shortlist of contested genes (6,583 genes, 70,723 candidates) and on the analysis set (4,847 genes, 48,394 candidates), against chance (mean of 1/n), with 95% bootstrap intervals over genes (2,000 resamples). "Combined" is the sum of the two models' within-gene standardised scores. All scores are LOCO.
- **b.** A Sankey flow of all 15,560 genes: uncontested, then contested split by which model ranks the annotated end first. Band heights are proportional to gene counts.

**Key numbers**
- Analysis set: network 0.681, splice model 0.463, combined 0.694 [0.681, 0.707], chance 0.287. Combined over network alone: McNemar 447 against 384, chi2 = 4.6, P = 0.031, so the gain is real but small.
- Contested 6,583: network 0.577, splice model 0.403, chance 0.270; difference 17.5 points, P = 1e-108.
- Genome-wide 15,560: network 0.821, splice model 0.747, baseline 0.577.
- Flow asserts 8,977 uncontested and 1,882 / 1,918 / 769 / 2,014 in the contested branches.
- **Do not headline the 0.921 figure:** it is a per-gene oracle, the best of the two models chosen after the fact.

**How:** `cd T && Rscript R/fig3_14_recovery.R` (not `fig3_regroup.R`); 2,000 bootstrap resamples, `set.seed(1)`; it also writes `T/data/fig3_11_p_at_1.tsv`. The contested panel uses `PUB/wg_per_gene.tsv` rows with `uncontested == 0`; the analysis-set panel uses `PUB/rescue_ranks.tsv` (from `PUB/rescue_ranks.py`, built from `splice26_loco_scores.npz`, `cnn_wg_scores.npz` and `shortlist_feat_all.parquet` `rf_prob`). `splice26_loco_scores.npz` comes from `splice26_loco.py` via `sploco.sbatch`, job 58094193.

**OPEN:** `rescue_ranks.py` ranks with `(v > t).sum() + 1`, which breaks ties in the model's favour, contrary to Methods 3.4.10. The effect is probably negligible but should be stated or fixed.

<!-- BEGIN AUTO:3.14:code -->
**Code.** `thesis_ch3/R/fig3_14_recovery.R`, 83 lines, sha256 `1f05cf1433c1`. Reproduced verbatim; also in `code/fig3_14_recovery.R`.

```r
# fig3_14_recovery.R -- Figure 3.14, precision at 1 against chance on the genes the baseline gets wrong.
#
# A  P@1 per model with 95% gene-bootstrap intervals (2,000 resamples), on two gene sets:
#      contested, 6,583 genes, full candidate sets     ../wg_per_gene.tsv (uncontested == 0)
#      analysis set, 4,847 genes                       ../rescue_ranks.tsv (adds the combined ranking)
#    chance is mean(1 / candidates) over the same genes
# B  all 15,560 genes per model: uncontested, contested ranked first, contested not ranked first (../wg_per_gene.tsv)
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr)})
setwd("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3")
source("R/_harvest.R")
INK<-"#1A1A1A"; GTX<-"#5A6068"; GREY<-"#8A8F98"; CAN<-"#0B55A0"; SPL<-"#E8A33D"; COMB<-"#4A4E69"
set.seed(1)
boot<-function(v,B=2000){ n<-length(v); b<-replicate(B,mean(v[sample.int(n,n,TRUE)])); c(mean(v),quantile(b,c(.025,.975))) }

W<-read.delim("../wg_per_gene.tsv"); Wc<-W%>%filter(uncontested==0)
R<-read.delim("../rescue_ranks.tsv")
stopifnot(nrow(Wc)==6583, nrow(R)==4847)
rows<-list(
  c(set="Contested genes\n(n = 6,583)", model="chance",       boot(1/Wc$n_candidates)),
  c(set="Contested genes\n(n = 6,583)", model="splice model", boot(Wc$splice)),
  c(set="Contested genes\n(n = 6,583)", model="sequence CNN", boot(Wc$cnn)),
  c(set="Analysis set\n(n = 4,847)",    model="chance",       boot(1/R$n_candidates)),
  c(set="Analysis set\n(n = 4,847)",    model="splice model", boot(as.numeric(R$rank_splice==1))),
  c(set="Analysis set\n(n = 4,847)",    model="sequence CNN", boot(as.numeric(R$rank_cnn==1))),
  c(set="Analysis set\n(n = 4,847)",    model="combined",     boot(as.numeric(R$rank_combo==1))))
P<-as.data.frame(do.call(rbind,rows)); names(P)<-c("set","model","p","lo","hi")
P<-P%>%mutate(across(c(p,lo,hi),as.numeric),
              model=factor(model,levels=c("chance","splice model","sequence CNN","combined")),
              set=factor(set,levels=unique(set)))
write.table(P%>%mutate(set=gsub("\n"," ",set)),"data/fig3_11_p_at_1.tsv",sep="\t",quote=FALSE,row.names=FALSE)
print(P%>%mutate(set=gsub("\n"," ",set)),digits=3)
pA<-ggplot(P,aes(model,p,fill=model))+
  geom_col(width=0.7,colour=INK,linewidth=0.25)+
  geom_errorbar(aes(ymin=lo,ymax=hi),width=0.18,linewidth=0.35,colour=INK)+
  facet_grid(.~set,scales="free_x",space="free_x")+
  scale_fill_manual(values=c(chance="white","splice model"=SPL,"sequence CNN"=CAN,combined=COMB),guide="none")+
  scale_y_continuous(limits=c(0,0.8),expand=c(0,0))+
  labs(x=NULL,y="precision at 1",title="A   Annotated end ranked first on contested genes")+
  theme_classic(base_size=10)+
  theme(plot.title=element_text(face="bold",size=11,colour=INK),strip.background=element_blank(),
        strip.text=element_text(colour=INK,size=9),axis.text=element_text(colour=GTX),
        axis.text.x=element_text(size=8.5))

# B: Sankey of all 15,560 genes: all -> uncontested / contested -> which model ranks the annotated end first
WG<-read.delim("../wg_per_gene.tsv"); stopifnot(nrow(WG)==15560)
ct<-WG%>%filter(uncontested==0)
N<-c(unc=sum(WG$uncontested==1),both=sum(ct$cnn==1&ct$splice==1),net=sum(ct$cnn==1&ct$splice==0),spl=sum(ct$cnn==0&ct$splice==1),none=sum(ct$cnn==0&ct$splice==0))
stopifnot(sum(N)==15560,N["unc"]==8977,sum(N[-1])==6583); print(N)
fmt<-function(x) format(x,big.mark=",")
GAP<-700; X<-c(0,1,2); BW<-0.2
node<-function(sizes,gap=GAP){ top<-0; o<-list(); for(k in names(sizes)){ o[[k]]<-c(top,top+sizes[[k]]); top<-top+sizes[[k]]+gap }; o }
tot<-sum(N)+4*GAP
n1<-list(all=c((tot-15560)/2,(tot-15560)/2+15560))
s2<-c(unc=N[["unc"]],con=sum(N[-1])); n2<-node(s2,gap=tot-sum(s2))
n3<-node(as.list(N))
rib<-function(x0,x1,a0,a1,b0,b1,id,fill){ t<-seq(0,1,length.out=60); sg<-3*t^2-2*t^3; xs<-x0+(x1-x0)*t
  data.frame(x=c(xs,rev(xs)),y=c(a0+(b0-a0)*sg,rev(a1+(b1-a1)*sg)),id=id,fill=fill) }
R2<-rbind(
  rib(X[1]+BW,X[2]-BW,n1$all[1],n1$all[1]+s2[["unc"]],n2$unc[1],n2$unc[2],"a","unc"),
  rib(X[1]+BW,X[2]-BW,n1$all[1]+s2[["unc"]],n1$all[2],n2$con[1],n2$con[2],"b","con"),
  rib(X[2]+BW,X[3]-BW,n2$unc[1],n2$unc[2],n3$unc[1],n3$unc[2],"c","unc"))
off<-n2$con[1]
for(k in c("both","net","spl","none")){ R2<-rbind(R2,rib(X[2]+BW,X[3]-BW,off,off+N[[k]],n3[[k]][1],n3[[k]][2],k,k)); off<-off+N[[k]] }
ND<-rbind(data.frame(x=X[1],y0=n1$all[1],y1=n1$all[2],fill="all",lab=paste0("All genes\n",fmt(15560))),
          data.frame(x=X[2],y0=n2$unc[1],y1=n2$unc[2],fill="unc",lab=paste0("Uncontested\n",fmt(N[["unc"]]))),
          data.frame(x=X[2],y0=n2$con[1],y1=n2$con[2],fill="con",lab=paste0("Contested\n",fmt(sum(N[-1])))),
          data.frame(x=X[3],y0=n3$unc[1],y1=n3$unc[2],fill="unc",lab=paste0("Baseline ranks end first  ",fmt(N[["unc"]]))),
          data.frame(x=X[3],y0=n3$both[1],y1=n3$both[2],fill="both",lab=paste0("Network and splice model  ",fmt(N[["both"]]))),
          data.frame(x=X[3],y0=n3$net[1],y1=n3$net[2],fill="net",lab=paste0("Network only  ",fmt(N[["net"]]))),
          data.frame(x=X[3],y0=n3$spl[1],y1=n3$spl[2],fill="spl",lab=paste0("Splice model only  ",fmt(N[["spl"]]))),
          data.frame(x=X[3],y0=n3$none[1],y1=n3$none[2],fill="none",lab=paste0("Neither  ",fmt(N[["none"]]))))
FILL<-c(all="#4A4E69",unc="#C9CDD3",con="#E69F00",both=COMB,net=CAN,spl=SPL,none="#E4E6EA")
fl<-ggplot()+geom_polygon(data=R2,aes(x,y,group=id,fill=fill),alpha=0.55)+
  geom_rect(data=ND,aes(xmin=x-BW,xmax=x+BW,ymin=y0,ymax=y1,fill=fill),colour=INK,linewidth=0.25)+
  geom_text(data=ND[1:3,],aes(x=x,y=(y0+y1)/2,label=lab),size=3.1,colour=c("white",INK,INK),lineheight=0.95,fontface="bold")+
  geom_text(data=ND[4:8,],aes(x=x+BW+0.03,y=(y0+y1)/2,label=lab),size=3.1,colour=INK,hjust=0)+
  annotate("text",x=X,y=-700,label=c("Annotated 3' end","Chapter 2 baseline","Annotated end ranked first by"),size=3.1,colour=GTX,hjust=c(0.5,0.5,0))+
  scale_fill_manual(values=FILL,guide="none")+scale_y_reverse()+scale_x_continuous(limits=c(-0.25,3.35))+
  labs(title="B   Fate of all 15,560 genes")+theme_void(base_size=10)+theme(plot.title=element_text(face="bold",size=11,colour=INK))
fig<-pA/fl+plot_layout(heights=c(0.8,1))
ggsave("out/fig3_14_recovery.pdf",fig,width=7.4,height=8.2,device=cairo_pdf)
ggsave("out/fig3_14_recovery.png",fig,width=7.4,height=8.2,dpi=300,bg="white")
cat("[out] out/fig3_14_recovery.{pdf,png}\n")
```
<!-- END AUTO:3.14:code -->

## Appendix: shared panel sources

<!-- BEGIN AUTO:appendix:code -->
Reproduced once each, because several figures draw panels from them. Copies are in `code/`.

### `thesis_ch3/R/fig3_regroup.R`

Assembles Figures 3.3, 3.4, 3.6, 3.7, 3.8, 3.9 and 3.11 from harvested panels. 120 lines, sha256 `11f48036ce52`.

```r
# fig3_regroup.R -- Figures 3.3 to 3.8 and 3.10, regrouped from the panels pub/'s scripts build.
#
# Each old figure's script is run in a sandbox (R/_harvest.R) for its panel objects. Panels are
# relettered and retitled for the new numbering. In-figure text is limited to labels that NAME
# something (series, reference lines, categories, schematic parts); statistics, sentences,
# sequences and numeric call-outs are removed here and live in the legend or the text.
#
#   3.3  <- make_splicing_environment.R A, B  + new C, D (../acceptor_position_scan.tsv)
#   3.4  <- make_splicing_environment.R C, D     3.7  <- make_figure_architecture.R, make_figure_rescue.R a
#   3.5  <- make_splice_model_final.R   A, B     3.8  <- make_figure_loco.R A, B, D
#   3.6  <- make_splice_model_final.R   C, D     3.12 <- make_figure_cnn_interpretability.R
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(dplyr)})
setwd("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3")
source("R/_harvest.R")
dir.create("out", showWarnings = FALSE)
INK <- "#1A1A1A"; GTX <- "#5A6068"
TT <- theme(plot.title = element_text(face = "bold", size = 11, colour = INK),
            plot.subtitle = element_blank(), plot.caption = element_blank())
save2 <- function(fig, name, w, h, dpi = 300) {
  ggsave(file.path("out", paste0(name, ".pdf")), fig, width = w, height = h, device = cairo_pdf, limitsize = FALSE)
  ggsave(file.path("out", paste0(name, ".png")), fig, width = w, height = h, dpi = dpi, bg = "white", limitsize = FALSE)
  cat("[out]", name, "\n")
}
# ---------------------------------------------------------------- 3.3 donor and acceptor environment
E <- harvest("make_splicing_environment.R")
drop_leader <- function(p) { p$layers <- Filter(function(l) !(inherits(l$geom, "GeomSegment") &&
    is.data.frame(l$data) && isTRUE(all(l$data$xend == -330))), p$layers); p }
pA0 <- E$panel_A();
a <- edit_text(drop_leader(pA0), drop = c("-50 nt bin", "upstream of the poly")) +
  labs(title = "A   Strong 5′ splice-site density") + TT
b <- edit_text(E$panel_B(), drop = c("^mean ", "no difference", "median -12")) +
  labs(title = "B   Within-gene difference in U1 site density") + TT
REAL <- "#08519C"; COMP <- "#E8A33D"; DEC <- "#7B5EA7"
AP <- read.delim("../acceptor_position_scan.tsv") %>%
  mutate(class = factor(class, levels = c("real", "competitor", "decoy"),
                        labels = c("real 3′ end", "competitor", "intronic decoy")))
stopifnot(nrow(AP) == 4847 + 43547 + 30000)
CLS <- c("real 3′ end" = REAL, competitor = COMP, "intronic decoy" = DEC)
TH3 <- theme_classic(base_size = 7.5) + theme(axis.text = element_text(colour = GTX), legend.position = "none",
        panel.grid.major.y = element_line(colour = "#E8EAED", linewidth = .2))
c3 <- ggplot(AP, aes(max_me3, colour = class)) +
  stat_ecdf(linewidth = 0.6) + scale_colour_manual(values = CLS) +
  coord_cartesian(xlim = c(4, 16)) +
  labs(x = "strongest upstream 3′ splice-site score (MaxEntScan score3)", y = "cumulative fraction of sites",
       title = "C   Strongest upstream 3′ splice site") + TH3 + TT
d3 <- ggplot(AP %>% filter(max_me3 >= 6.5), aes(position, colour = class)) +
  geom_density(aes(y = after_stat(density) * 1000), linewidth = 0.6, adjust = 0.6, key_glyph = "path") + scale_colour_manual(values = CLS, name = NULL) +
  scale_x_continuous(limits = c(-2000, -126), breaks = c(-2000, -1500, -1000, -500)) +
  labs(x = "position of the strongest upstream 3′ splice site (nt)", y = "density (per kb)",
       title = "D   Position of that site") + TH3 + TT +
  theme(legend.position = "bottom")
save2((a / b / (c3 | d3)) + plot_layout(heights = c(1, 0.8, 0.9)), "fig3_03_splice_environment", 6.8, 9.6)

# ---------------------------------------------------------------- 3.4 donor scores
pc <- E$panel_C()
pc[[1]] <- edit_text(pc[[1]], drop = "r = 0.69") + labs(title = "A   U1 duplex free energy against MaxEntScan 5′ score") + TT
if (length(pc) > 1) pc[[2]] <- pc[[2]] + labs(title = NULL, subtitle = NULL, caption = NULL)
d <- edit_text(E$panel_D(), drop = "<mapped txt>") + labs(title = "B   Score distributions and thresholds") + TT
# C: MaxEntScan sensitivity at annotated sites against background calls in intronic competitor windows
MC <- read.delim("data/maxent_calibration.tsv") %>% mutate(site = factor(site, levels = c("acceptor", "donor"),
        labels = c("3\u2032 splice site (score3)", "5\u2032 splice site (score5)")))
U1C <- read.delim("data/u1_calibration.tsv")
USED <- MC %>% filter((grepl("^3", site) & threshold == 6.5) | (grepl("^5", site) & threshold == 8))
SC <- c("3\u2032 splice site (score3)" = "#E8A33D", "5\u2032 splice site (score5)" = "#08519C")
c4 <- ggplot(MC %>% filter(per_kb_background > 0), aes(per_kb_background, sensitivity_annotated, colour = site)) +
  geom_path(linewidth = 0.6) + geom_point(size = 1) +
  geom_point(data = USED, size = 3, shape = 21, fill = "white", stroke = 1) +
  geom_point(data = U1C, aes(background_per_kb, sensitivity_annotated_donors), inherit.aes = FALSE,
             shape = 24, size = 2.6, fill = "#7B5EA7", colour = INK, stroke = 0.4) +
  geom_text(data = U1C, aes(background_per_kb, sensitivity_annotated_donors, label = paste0("U1 \u2265 ", threshold)), inherit.aes = FALSE,
            hjust = 1.2, vjust = 0.3, size = 2.4, colour = "#7B5EA7") +
  scale_x_log10(breaks = c(0.01, 0.1, 1, 10, 100), labels = c("0.01", "0.1", "1", "10", "100")) +
  scale_colour_manual(values = SC, name = NULL) +
  labs(x = "background calls per kb of intronic sequence", y = "fraction of annotated sites passing",
       title = "C   Calibration of the thresholds") +
  theme_classic(base_size = 7.5) + TT + theme(axis.text = element_text(colour = GTX), legend.position = c(0.3, 0.85),
        legend.key.size = unit(3, "mm"), legend.text = element_text(size = 7))
save2((pc / ((d | c4) + plot_layout(widths = c(1.9, 1)))) + plot_layout(heights = c(1.1, 0.9)), "fig3_04_donor_scores", 8.2, 8.0)

# ---------------------------------------------------------------- 3.5, 3.6 splice model
S <- harvest("make_splice_model_final.R")
s5a <- edit_text(S$pA, drop = "Within-gene evaluation",
                 relab = c("Sequence window" = "25 bins of 100 nt",
                           "U1 duplex density" = "U1 site density per bin",
                           "Maximum MaxEnt" = "strongest upstream 3′ splice-site score",
                           "Ridge \\(L2\\)" = "ridge logistic regression"))
s5b <- edit_text(S$pB, drop = c("-0.23 at", "^\\+0.30$"), relab = c("MaxEnt 3'ss" = "acceptor"))
save2((s5a + labs(title = "A   Features") + TT) / (s5b + labs(title = "B   Fitted coefficients by position") + TT) +
      plot_layout(heights = c(1.15, 1)), "fig3_05_splice_model", 7.4, 6.6)
s6c <- edit_text(S$pC, drop = "<mapped lab>", relab = c("random" = "chance"))
s6d <- edit_text(S$pD, drop = c("<mapped sprintf", "Win-rate 0.699"))
save2((s6c + labs(title = "A   Within-gene recall at rank k") + TT) | (s6d + labs(title = "B   Gene-centred scores") + TT),
      "fig3_06_splice_model_performance", 7.8, 3.6)

# ---------------------------------------------------------------- 3.7 windows and architecture
Ar <- harvest("make_figure_architecture.R")
R  <- harvest("make_figure_rescue.R")
arch <- edit_text(Ar$p, drop = "one output bin sees")
save2((R$pA + labs(title = "A   Sequence read by the baseline and the extended models") + TT) /
      (arch + labs(title = "B   Network architecture") + TT + theme(plot.background = element_rect(fill = "white", colour = NA))) +
      plot_layout(heights = c(1, 0.72)), "fig3_07_architecture", 9.4, 4.8)

# ---------------------------------------------------------------- 3.8 network vs splice model
L <- harvest("make_figure_loco.R")
a1 <- edit_text(L$pA1, relab = c("^Sequence CNN" = "Sequence CNN", "^Splice model" = "Splice model"))
a2 <- edit_text(L$pA2, drop = "^AUPRC", relab = c("base rate" = "base rate"))
b8 <- edit_text(L$pB, drop = "recall@1", relab = c("^Splice model" = "Splice model"))
d8 <- edit_text(L$pD, drop = "<mapped lab>")
top <- ((a1 + labs(title = "A   Pooled ROC and precision-recall") + TT) | (a2 + TT) |
        (b8 + labs(title = "B   Within-gene recall at rank k") + TT)) + plot_layout(widths = c(0.78, 0.78, 1.44))
bot <- d8 + labs(title = "C   Annotated end ranked first, by gene length and number of candidates") + TT
save2((top / bot) + plot_layout(heights = c(1, 0.95)), "fig3_08_cnn_vs_splice", 10.6, 7.4)

# ---------------------------------------------------------------- 3.12 interpretation
I <- harvest("make_figure_cnn_interpretability.R")
pa <- I$pA; pa[[2]] <- pa[[2]] + labs(title = "A   First-layer filters and their positional contribution") + TT
pb <- I$pB
pb[[2]] <- edit_text(pb[[2]], drop = c("untouched competitor", "<mapped txt>")) +
  labs(title = "B   Splice-site implantation into competitor windows") + TT
save2(pa / pb + plot_layout(heights = c(4.3, 1.35)), "fig3_12_interpretation", 9.8, 13.6, dpi = 220)
```

### `thesis_ch3/R/_harvest.R`

Runs a pub/ script for its panel objects without letting it write; also `retitle` and `edit_text`. 36 lines, sha256 `8a8756dbc681`.

```r
# _harvest.R -- run one of pub/'s figure scripts for its panel objects without letting it write
# anything: ggsave and file-writing writeLines are no-ops inside the sandbox, and the script runs
# with pub/ as the working directory so its relative inputs resolve. pub/ scripts are not edited.
PUB <- normalizePath("..")
harvest <- function(script) {
  env <- new.env(parent = globalenv())
  env$ggsave <- function(...) invisible(NULL)
  env$writeLines <- function(text, con = stdout(), ...)
    if (is.character(con)) invisible(NULL) else base::writeLines(text, con, ...)
  old <- setwd(PUB); on.exit(setwd(old))
  suppressMessages(suppressWarnings(sys.source(file.path(PUB, script), envir = env)))
  env
}
# strip a panel's baked-in title and set a new one; subtitle and caption dropped unless kept
retitle <- function(p, title, keep_sub = FALSE)
  p + labs(title = title, subtitle = if (keep_sub) p$labels$subtitle else NULL, caption = NULL)

# text layers are matched on their label: a fixed label, a label column, or the mapping expression
labtext <- function(l) {
  lab <- c(l$aes_params$label, if (is.data.frame(l$data) && "label" %in% names(l$data)) as.character(l$data$label))
  if (is.null(lab) && !is.null(l$mapping$label)) lab <- paste0("<mapped ", rlang::quo_text(l$mapping$label), ">")
  paste(lab, collapse = " | ")
}
is_text <- function(l) inherits(l$geom, c("GeomText", "GeomLabel"))
# drop: regexes; a text layer whose label matches any is removed. relab: named vector regex -> new label
edit_text <- function(p, drop = character(), relab = character()) {
  keep <- rep(TRUE, length(p$layers)); hit <- setNames(rep(0L, length(c(drop, names(relab)))), c(drop, names(relab)))
  for (i in seq_along(p$layers)) { l <- p$layers[[i]]; if (!is_text(l)) next
    t <- labtext(l)
    for (d in drop) if (grepl(d, t)) { keep[i] <- FALSE; hit[d] <- hit[d] + 1L }
    for (r in names(relab)) if (keep[i] && grepl(r, t)) { l$aes_params$label <- relab[[r]]; p$layers[[i]] <- l; hit[r] <- hit[r] + 1L }
  }
  miss <- names(hit)[hit == 0]; if (length(miss)) stop("edit_text: no layer matched ", paste(miss, collapse = "; "))
  p$layers <- p$layers[keep]; p
}
```

### `make_splicing_environment.R`

Panels harvested for Figures 3.3 (a, b) and 3.4 (a, b). 486 lines, sha256 `9b58ff0006de`.

```r
#!/usr/bin/env Rscript
# make_splicing_environment.R -- the splicing environment of real 3' ends against their
# competitors, read on both marks of the same donor axis, with the two panels that
# justify the measurements.
#
# MEASUREMENT FIGURE. No model performance, no win-rates, no CNN anywhere. The win-rate
# annotation that used to sit on panel B and the "acceptor within-gene 0.606" line that
# used to sit in the caption are both discrimination metrics and have been removed.
#
# PLOTTING ONLY. Sources:
# ONE WINDOW, [-2000, +500], AND ONE SCAN. All four panels read files written by a single
# run of splice_score_redundancy.py, so they cannot disagree about what the window is:
#   A  splice_env_metagene.tsv      strong-5'ss density per 100 nt bin, per class, +/-1 SE
#   B  splice_env_gene_delta.tsv + splice_env_wilcoxon.tsv   per-gene real minus mean
#      competitor, U1 sites per kb, and the paired signed-rank on those differences
#   C,D splice_score_joint.tsv, splice_score_exceedance.tsv, splice_score_stats.tsv
# This supersedes the earlier arrangement, in which panel A came from panelF_standalone.py
# on +/-2 kb and panel B from u1_within_gene_test.py on [-1000, +500]. Those numbers are not
# relabelled here, they are recomputed; the panel A values in particular change, because a
# metagene bin mean over [-2000, +500] is not the same estimator as one over +/-2 kb.
#   u1_5ss_stats.tsv is read only for the per-class n.
#
# All methodological prose has moved out of the figure and into the full Methods and
# Results section thesis_section_u1landscape.txt, which this script writes.
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork)})
rd<-function(f){ if(!file.exists(f)) stop("missing: ",f); read_tsv(f,show_col_types=FALSE) }
e<-function(...) cat(...,"\n",file=stderr())
INK<-"#1A1A1A"; BASE<-7
REAL<-"#08519C"; COMP<-"#E8A33D"; DEC<-"#7B5EA7"; GREY<-"#8A8F98"
LEV<-c("real","competitor","decoy")
LAB<-c(real="real 3' end",competitor="competitor",decoy="intronic decoy")
PAL<-setNames(c(REAL,COMP,DEC),LAB[LEV])
fc<-function(x) factor(x,levels=LEV,labels=LAB[LEV])
th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-1),axis.line=element_line(color=INK,linewidth=.3),
  axis.title=element_text(size=BASE),
  panel.grid.major.y=element_line(color="#E8EAED",linewidth=.2),
  plot.title=element_text(size=BASE+1,face="bold",hjust=0,margin=margin(b=4)),
  legend.title=element_blank(),legend.text=element_text(size=BASE-1),
  legend.key.height=unit(.7,"lines"),legend.key.width=unit(.9,"lines"),
  legend.margin=margin(0,0,0,0),legend.box.spacing=unit(2,"pt"),
  strip.background=element_blank(),strip.text=element_text(size=BASE-.5,color=INK))
nfmt<-function(x) format(as.numeric(x),big.mark=",")
ST<-rd("u1_5ss_stats.tsv")
sv<-function(k) suppressWarnings(as.numeric(ST$value[ST$key==k][1]))
WX<-rd("splice_env_wilcoxon.tsv"); wx<-function(k) WX$value[WX$key==k][1]
SS<-rd("splice_score_stats.tsv")
ssv<-function(k){ v<-SS$value[SS$key==k]; if(!length(v)) stop("missing stat: ",k)
                  suppressWarnings(as.numeric(v[1])) }
MG<-rd("splice_env_metagene.tsv")           # class is already real/competitor/decoy
DL<-rd("splice_env_gene_delta.tsv")
JD<-rd("splice_score_joint.tsv")
EX<-rd("splice_score_exceedance.tsv")
MEAN<-as.numeric(wx("mean_delta")); MED<-as.numeric(wx("median_delta"))
PW<-wx("p"); NGP<-as.numeric(wx("n_paired_genes"))
FBZ<-as.numeric(wx("frac_below_zero"))
NWX<-as.numeric(wx("n_genes"))                       # Wilcoxon n, after zero deltas dropped
# Decay of the tail across each cut, so "no step defines the threshold" is measured, not
# asserted: fold change in P(score >= t) per unit of score over the two units either side.
FOLD<-function(sc,t,h=2){ q<-EX%>%filter(class=="pooled",score==sc)
  f<-function(x) q$frac[which.min(abs(q$threshold-x))]
  (f(t-h)/f(t+h))^(1/(2*h)) }
# Panel A is RAW. A 3-bin rolling mean reads smoother, but it lifts the minimum from
# 0.54 to 0.61, so the quoted pA-bin value would no longer be on the curve a reader
# checks. Honesty beats smoothness here; the 1 SE ribbons carry the noise.
SMOOTH<-FALSE
WIN<-as.character(wx("window"))
stopifnot(identical(WIN,"[-2000,+500]"), identical(sst_win<-SS$value[SS$key=="window"][1],WIN))
e("[read] one scan, window",WIN,": splice_env_metagene.tsv  splice_env_gene_delta.tsv",
  "splice_env_wilcoxon.tsv  splice_score_joint.tsv  splice_score_exceedance.tsv  splice_score_stats.tsv")
e("[win]  panels A, B, C and D all computed on",WIN,"-- asserted equal between the panel-B",
  "Wilcoxon file and the panel-C/D stats file")
e("[stat] Wilcoxon signed-rank V =",wx("V"),"on",wx("n_genes"),"genes, z =",wx("z"),
  ", p",PW,"(log10 p ~",wx("log10_p"),")")
e("[stat] window",WIN,"  mean per-gene delta",sprintf("%.2f /kb",MEAN),
  "  median",sprintf("%.2f /kb",MED),"  genes below 0:",FBZ)
e("[A]    panel A lines/ribbons:",if(SMOOTH) "3-bin rolling mean" else "raw")

## ===== A: 5'ss density metagene =====
roll3<-function(x){ n<-length(x); sapply(seq_len(n),function(i){
  j<-max(1,i-1):min(n,i+1); mean(x[j]) }) }
panel_A<-function(){
  D<-MG%>%group_by(class)%>%arrange(offset,.by_group=TRUE)%>%
    mutate(y=if(SMOOTH) roll3(donor_per_kb) else donor_per_kb,
           lo=if(SMOOTH) roll3(donor_per_kb-se) else donor_per_kb-se,
           hi=if(SMOOTH) roll3(donor_per_kb+se) else donor_per_kb+se)%>%
    ungroup()%>%mutate(class=fc(class))
  pa<-MG%>%filter(offset==-50); v<-function(c) sprintf("%.2f",pa$donor_per_kb[pa$class==c])
  e("[A]    pA bin (-50 nt) raw: real",v("real"),"< competitor",v("competitor"),
    "< decoy",v("decoy"),"/kb")
  ggplot(D,aes(offset,y,color=class,fill=class))+
    annotate("rect",xmin=-Inf,xmax=0,ymin=-Inf,ymax=Inf,fill=GREY,alpha=.05)+
    geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.3)+
    geom_ribbon(aes(ymin=lo,ymax=hi),alpha=.18,color=NA,show.legend=FALSE)+
    geom_line(linewidth=.45)+
    scale_color_manual(values=PAL)+scale_fill_manual(values=PAL,guide="none")+
    scale_x_continuous(breaks=seq(-2000,500,500))+
    annotate("segment",x=-70,xend=-330,y=.552,yend=.60,color=GREY,linewidth=.25)+
    annotate("text",x=-350,y=.60,hjust=1,vjust=.4,size=2.0,color=INK,lineheight=1.05,
             label=paste0("-50 nt bin:  real ",v("real"),"  <  competitor ",v("competitor"),
                          "  <  decoy ",v("decoy")))+
    annotate("text",x=-1950,y=Inf,hjust=0,vjust=1.6,label="upstream of the poly(A) site",
             size=2.0,color="#555555")+
    labs(title="A   Strong 5' splice-site density flanking the poly(A) site",
         x="position relative to poly(A) site (nt)",
         y="strong 5' splice sites (MaxEnt >= 8) per kb")+
    th+theme(legend.position="bottom")
}

## ===== B: within-gene difference =====
panel_B<-function(){
  bw<-diff(range(DL$delta_per_kb))/54
  ggplot(DL,aes(delta_per_kb))+
    geom_histogram(aes(y=after_stat(density)),binwidth=bw,fill=GREY,alpha=.45,color=NA)+
    geom_density(color=INK,linewidth=.45,adjust=1.1)+
    geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.35)+
    geom_vline(xintercept=MEAN,color=REAL,linewidth=.5)+
    annotate("text",x=MEAN,y=Inf,vjust=1.5,hjust=1.08,size=2.0,color=REAL,
             label=sprintf("mean %.1f",MEAN))+
    annotate("text",x=0,y=Inf,vjust=1.5,hjust=-0.12,size=2.0,color="#555555",label="no difference")+
    annotate("text",x=-Inf,y=Inf,hjust=-0.06,vjust=2.6,size=2.1,color=INK,lineheight=1.25,
      label=paste0("median ",sprintf("%.1f",MED)," per kb\n",
                   sprintf("%.0f%%",100*FBZ)," of genes below zero\n",
                   "paired Wilcoxon p ",PW))+
    scale_y_continuous(expand=expansion(mult=c(0,.14)))+
    labs(title="B   Within-gene difference in U1 site density",
         x="per gene: real - mean competitor, U1 sites (-dG >= 4) per kb",
         y="density of genes")+th
}

## ===== C: redundancy of the two donor scores =====
# The main map is the GT-anchored population, because that is where a donor score means
# anything: MaxEntScan score5 is only defined on a GT 9-mer in the featurizer (scan_me5
# returns NaN elsewhere), and no non-GT 9-mer in the whole 4^9 space reaches 8 bits
# (maximum 4.16). The all-positions r is shown as an inset for context only.
XL_U1<-"U1:pre-mRNA duplex free energy (ViennaRNA duplexfold, U1 snRNA 5' end ACUUACCUG)"
YL_ME<-"5' splice-site score (MaxEntScan)"
panel_C<-function(){
  r<-ssv("r_all_pooled"); rg<-ssv("r_gt_pooled")
  n<-ssv("n_pos_all"); ng<-ssv("n_pos_gt")
  e(sprintf("[C]    HEADLINE GT-anchored r = %.3f over %s donor-dinucleotide positions",rg,nfmt(ng)))
  e(sprintf("[C]    context  all positions r = %.3f over %s positions (%.2f%% are GT-anchored)",
            r,nfmt(n),100*ng/n))
  G<-JD%>%filter(subset=="gt"); A<-JD%>%filter(subset=="all")
  ramp<-c("#F7FBFF","#C6DBEF","#6BAED6","#2171B5","#08306B")
  main<-ggplot(G,aes(u1_dg,me5,fill=count))+geom_tile()+
    scale_fill_gradientn(colours=ramp,trans="log10",name="positions",
                         breaks=c(1e1,1e2,1e3,1e4,1e5,1e6),
                         labels=c("10","100","1k","10k","100k","1M"))+
    scale_x_continuous(breaks=seq(-4,12,2),expand=expansion(mult=c(.01,.01)))+
    scale_y_continuous(breaks=seq(-50,10,10),expand=expansion(mult=c(.36,.06)))+
    annotate("text",x=-Inf,y=-Inf,hjust=-0.06,vjust=-0.55,size=2.3,color=INK,lineheight=1.3,
             label=paste0("r = ",sprintf("%.2f",rg)," at GT-anchored positions\n",
                          "n = ",nfmt(ng)," donor dinucleotides"))+
    labs(title="C   Correlation between U1 duplex free energy and MaxEntScan 5' splice-site score",
         x=XL_U1,y=YL_ME)+
    th+theme(legend.position="right",legend.key.width=unit(.45,"lines"),
             legend.key.height=unit(1.0,"lines"),
             legend.title=element_text(size=BASE-1.5,color=INK))
  ins<-ggplot(A,aes(u1_dg,me5,fill=count))+geom_tile(show.legend=FALSE)+
    scale_fill_gradientn(colours=ramp,trans="log10")+
    scale_x_continuous(breaks=c(-4,0,4,8,12),expand=expansion(mult=c(.01,.01)))+
    scale_y_continuous(breaks=c(-40,-20,0),expand=expansion(mult=c(.02,.02)))+
    labs(x=NULL,y=NULL,
         title=paste0("all scanned positions, r = ",sprintf("%.2f",r),"\n",
                      "diluted by the ",sprintf("%.0f%%",100*(1-ng/n)),
                      " of positions with no GT,\nwhere neither score is informative"))+
    theme_classic(base_size=BASE-2)+
    theme(axis.text=element_text(color=INK,size=BASE-3),
          axis.line=element_line(color=INK,linewidth=.25),
          plot.title=element_text(size=BASE-2.2,color="#555555",hjust=0,lineheight=1.2,
                                  margin=margin(b=2)),
          plot.background=element_rect(fill="white",color=GREY,linewidth=.2),
          plot.margin=margin(3,4,3,3))
  main+inset_element(ins,left=.555,bottom=.02,right=.995,top=.345,align_to="panel")
}

## ===== D: threshold selection =====
SCLAB<-c(u1_dg="U1:pre-mRNA duplex free energy, -dG (kcal/mol)",
         me5="5' splice-site score (MaxEntScan, bits)")
panel_D<-function(){
  D<-EX%>%filter(class=="pooled",frac>=1e-6)%>%
    mutate(score=factor(score,levels=names(SCLAB),labels=SCLAB))
  pc<-function(x) if (x>=0.01) sprintf("%.1f%%",100*x) else sprintf("%.2f%%",100*x)
  MK<-bind_rows(
    data.frame(sc=names(SCLAB)[1],threshold=c(4,7),
               frac=c(ssv("exceed_u1_4"),ssv("exceed_u1_7")),
               tag=sprintf("-dG >= %g",c(4,7))),
    data.frame(sc=names(SCLAB)[2],threshold=8,frac=ssv("exceed_me5_8"),
               tag="MaxEnt >= 8"))%>%
    mutate(score=factor(sc,levels=names(SCLAB),labels=SCLAB))
  MK$fold<-mapply(FOLD,MK$sc,MK$threshold)
  ANN<-MK%>%group_by(score)%>%
    summarise(txt=paste(sprintf("%s   %s of positions",tag,sapply(frac,pc)),collapse="\n"),
              .groups="drop")
  e("[D]    marked thresholds: ",paste(sprintf("%s -> %s (tail decays %.1fx per unit score)",
        MK$tag,sapply(MK$frac,pc),MK$fold),collapse="   "))
  ggplot(D,aes(threshold,frac))+
    geom_segment(data=MK,aes(x=threshold,xend=threshold,y=1e-6,yend=frac),
                 linetype=2,color=COMP,linewidth=.3)+
    geom_line(linewidth=.5,color=INK)+
    geom_point(data=MK,size=1.2,color=COMP)+
    geom_text(data=ANN,aes(x=Inf,y=Inf,label=txt),inherit.aes=FALSE,
              hjust=1.04,vjust=1.4,size=2.1,color=INK,lineheight=1.35)+
    facet_wrap(~score,scales="free_x",strip.position="bottom")+
    scale_y_log10(breaks=c(1,.1,.01,1e-3,1e-4,1e-5),
                  labels=function(x) format(x,scientific=FALSE,drop0trailing=TRUE,trim=TRUE),
                  expand=expansion(mult=c(.03,.22)))+
    labs(title="D   Score distributions and threshold selection",
         x=NULL,y="fraction of positions >= threshold")+
    th+theme(strip.placement="outside",
             strip.text=element_text(size=BASE,color=INK,margin=margin(t=3)),
             panel.spacing=unit(10,"pt"))
}

CAP<-paste0("n = ",nfmt(sv("n_real"))," real 3' ends, ",nfmt(sv("n_competitor")),
  " competitors and ",nfmt(sv("n_decoy"))," intronic decoys in ",nfmt(NGP),
  " paired genes. Definitions, windows and thresholds: thesis_section_u1landscape.txt.")

fig<-(panel_A()/panel_B()/panel_C()/panel_D())+plot_layout(heights=c(1,.88,1.05,.85))+
  plot_annotation(
    title="Real 3' ends occupy splice-donor-depleted terminal-exon sequence relative to intragenic competitors",
    subtitle="Two marks of one donor axis at the same sites, and the calibration of both",
    caption=CAP,
    theme=theme(plot.title=element_text(size=BASE+2,face="bold",color=INK),
                plot.subtitle=element_text(size=BASE,color="#555555"),
                plot.caption=element_text(size=BASE-1.4,color="#555555",hjust=0)))

ggsave("figure_splicing_environment.pdf",fig,width=6.8,height=12.6,device=grDevices::cairo_pdf)
ggsave("figure_splicing_environment.png",fig,width=6.8,height=12.6,dpi=300)
e("[out]  figure_splicing_environment.pdf  figure_splicing_environment.png")

## ===== the Methods + Results section this figure replaces its own prose with =====
# Every number below is read from a file on disk. Anything that could not be confirmed by
# reading the code or the data is written as a flagged [VERIFY: ...] placeholder rather
# than filled in from memory.
pa<-MG%>%filter(offset==-50); pv<-function(c) sprintf("%.2f",pa$donor_per_kb[pa$class==c])
pse<-function(c) sprintf("%.3f",pa$se[pa$class==c])
mrange<-function(lo,hi,c){ q<-MG%>%filter(offset>lo,offset<hi,class==c)
  if(!nrow(q)) stop("mrange: no rows for class ",c); sprintf("%.2f",mean(q$donor_per_kb)) }
pc<-function(x) if (x>=0.01) sprintf("%.2f%%",100*x) else sprintf("%.3f%%",100*x)
NPOS<-ssv("n_pos_all"); NGT<-ssv("n_pos_gt")
NSITE<-sum(sapply(LEV,function(c) ssv(paste0("n_sites_",c))))
FN<-rd("analysis_set_funnel.tsv")           # trace_analysis_set.py, asserted against the parquet
lab<-function(x) format(as.numeric(x),big.mark=",",trim=TRUE)
NBINS<-nrow(MG)/length(LEV)                 # metagene bins, from the file itself
e("[sec]  building thesis_section_u1landscape.txt")

W<-function(...) paste(strwrap(paste0(...),width=98),collapse="\n")
SEC<-paste(c(
"METHODS",
"",
"Site sets",
"",
W("Candidate cleavage sites were taken from cpa_peaks_final.parquet, the peak table behind the ",
  "rest of this work (70,723 peaks in total). All analyses in this figure use the subset flagged ",
  "in_analysis_set: ",nfmt(sv("n_real"))," real 3' ends and ",nfmt(sv("n_competitor")),
  " intragenic competitors, ",nfmt(as.numeric(sv("n_real"))+as.numeric(sv("n_competitor"))),
  " peaks in ",nfmt(NGP)," genes. A real 3' end is a peak with is_canonical set; the analysis ",
  "set carries exactly one per gene, and every gene in it carries at least one competitor, so ",
  "the gene count and the real-end count coincide."),
"",
W("Competitors are the peaks the baseline model calls that survive three exclusions. The column ",
  "that records the result, in_analysis_set in cpa_peaks_final.parquet, is written by ",
  "task_build_final_parquet.py, but that script only copies a mask it loads; the exclusions ",
  "themselves are applied in task_rerank_corr.py and task_upstream_blocks.py. Counts below are ",
  "reconstructed from the raw annotation caches by trace_analysis_set.py, which asserts that its ",
  "reconstruction equals the shipped column exactly, and are drawn in the funnel figure."),
"",
W("Starting from all ",lab(FN$n_peaks[1])," called peaks in ",lab(FN$n_genes[1])," genes (",
  lab(FN$n_canonical[1])," canonical, one per gene, and ",lab(FN$n_competitor[1]),
  " competitors), the exclusions act on the competitor pool only; canonical peaks are never ",
  "removed by a filter, only by losing their gene. First, competitors annotated as real ",
  "alternative 3' ends are dropped: a competitor is annotated if the nearest same-strand ",
  "PolyASite 2.0 cluster representative site (atlas.clusters.2.0.GRCh38.96.tsv) or the nearest ",
  "same-strand PolyA_DB v4 site (polyadb_v4/HumanPas/hg38.PAS.main.tsv) lies within 500 nt. This ",
  "removes ",lab(FN$n_removed[2])," competitors. Second, competitors lying in reference terminal ",
  "exons are dropped: the 500 nt strand-oriented window immediately upstream of the peak must not ",
  "overlap, by even one nucleotide, the last exon of any GENCODE v49 transcript (single-exon ",
  "transcripts count as terminal, and the last exons of all transcripts are merged per strand ",
  "before testing). This removes a further ",lab(FN$n_removed[3])," competitors."),
"",
W("[VERIFY: these two exclusions are written as a single conjunction in task_rerank_corr.py ",
  "(SURV = COMP & ~APA & ~TERMEX), so the code does not order them. The counts above attribute ",
  "the ",lab(5294)," competitors that carry both flags to the annotation filter, which is listed ",
  "first; reported separately the flags remove ",lab(18119)," and ",lab(6370)," competitors and ",
  lab(19195)," in union. Any ordering language in the text is a presentational choice, not a ",
  "property of the pipeline.]"),
"",
W("A gene is then kept only if it still holds exactly one canonical peak and at least one ",
  "competitor. ",lab(FN$n_genes[3]-FN$n_genes[4])," genes lose every competitor at this point and ",
  "leave with their canonical peaks, which is why ",lab(FN$n_removed[4]),
  " canonical peaks disappear while the competitor count does not change. Third, competitors at ",
  "or downstream of the gene's canonical site in transcript coordinates are dropped, on the ",
  "grounds that transcription terminates at the used poly(A) site so sequence past it was never a ",
  "candidate for premature cleavage; this removes ",lab(FN$n_removed[5])," competitors, and the ",
  "gene rule is then re-applied, costing a further ",lab(FN$n_removed[6])," genes and their ",
  "canonical peaks."),
"",
W("What remains is the analysis set: ",lab(FN$n_peaks[6])," peaks in ",lab(FN$n_genes[6]),
  " genes, ",lab(FN$n_canonical[6])," real 3' ends and ",lab(FN$n_competitor[6]),
  " competitors. Every gene contributes exactly one real 3' end and at least one competitor, so ",
  "the gene count and the real-end count coincide, and every within-gene comparison in this work ",
  "is defined on every gene of the set."),
"",
W("Intronic decoys come from decoy_set.npz, built by task_b16.py against GENCODE v49 ",
  "(../train/data/annotation/gencode.v49.annotation.gtf). Within each gene, candidate positions ",
  "are taken every 25 nt through every annotated intron, excluding any position within 1,000 nt ",
  "of a poly(A) atlas entry (../train/data/atlas.chr.bed), and capped at 600 candidates per gene ",
  "by random subsampling. Each competitor is then matched to 5 decoys, without replacement ",
  "within a gene, minimising a weighted squared distance on three standardised coordinates: ",
  "distance to the gene's canonical poly(A) site (weight 1), distance to the nearest annotated ",
  "3' splice site (weight 2) and distance to the nearest annotated 5' splice site (weight 2), ",
  "with a hard cap of 500 nt on the difference in distance-to-canonical-poly(A). This yields ",
  "138,758 decoys, of which 134,969 fall in analysis-set genes; ",nfmt(sv("n_decoy")),
  " of those were drawn without replacement at seed 42 for this figure. The decoys are therefore ",
  "matched on intron location and on splice-site context, and they bound the confound that ",
  "competitors sit inside introns while real 3' ends do not."),
"",
"Sequence",
"",
W("Sequence is read from hg38 (../train/data/hg38.fa) through a random-access FASTA index. ",
  "Fetches are transcript-oriented: offset 0 is the peak centre, negative offsets are upstream ",
  "along the transcript, and minus-strand intervals are reverse-complemented on read. Sequence ",
  "is upper-cased on read, so the genome's soft-masking of repeats is discarded and repeat-derived ",
  "positions are scored like any other."),
"",
W("Every panel uses the same window, [-2,000, +500] nt about the peak centre, 2,501 offsets ",
  "per site, and every number in this figure comes from one scan of it. Fetches are padded by ",
  "40 nt on each side so that a full 9-mer exists at every offset in the window, and the padding ",
  "is then discarded. This is also the window the GC metagene and the CNN input use, so the ",
  "measurements here are on the same footing as the rest of this work."),
"",
"Scoring",
"",
W("U1 duplex free energy. Every 9-mer of the 4^9 = 262,144-member 9-mer space was scored once, ",
  "before any analysis, by build_u1_lookup.py, as the free energy of duplex formation between the ",
  "9-mer read as RNA and the U1 snRNA 5' end 5'-ACUUACCUG-3' (Mount 1982; Patel and Steitz 2003), ",
  "computed with the ViennaRNA Python bindings as RNA.duplexfold(U1, nonamer).energy. The ",
  "canonical, unmodified U1 sequence is used because the nearest-neighbour parameters do not model ",
  "the pseudouridines U1 carries in vivo. The cached table is keyed on the DNA 9-mer. Following ",
  "the feature pipeline (scan_u1 in scripts/extract_splicing_features.py) the sign is inverted and ",
  "scores are reported as -dG in kcal/mol, so that larger values mean stronger predicted U1 ",
  "binding. Over the full 9-mer space -dG runs from -3.90 to 11.70. Every 9-mer has a value: this ",
  "score has no sequence prerequisite."),
"",
W("MaxEntScan 5' splice site. Scores come from a dense 4^9 lookup that reproduces maxentpy's ",
  "score5 exactly. The 9-mer carries the donor GT dinucleotide at 0-indexed positions 3 and 4, ",
  "that is the fourth and fifth bases (scan_me5, following Yeo and Burge 2004); the 9-mer ",
  "therefore spans the last 3 exonic and the first 6 intronic nucleotides of the donor. Over the ",
  "full 9-mer space score5 runs from -49.17 to 11.81. In the feature pipeline scan_me5 returns a ",
  "value only where that GT is present and NaN otherwise. The scans behind this figure score every ",
  "9-mer instead, which changes nothing about the strong-donor measurements: no 9-mer without the ",
  "GT reaches 8 bits anywhere in the 4^9 space (the maximum over the 245,760 non-GT 9-mers is ",
  "4.16), so the set of positions above threshold is identical either way."),
"",
W("The GT-anchored subset used in panel C is defined on exactly that criterion: a scanned position ",
  "is GT-anchored when its 9-mer has G at 0-indexed position 3 and T at position 4. Each 9-mer ",
  "beginning at sequence index k is assigned to offset k+3, the G of the GT, so that a donor score ",
  "and a U1 score at the same reported offset describe the same 9-mer."),
"",
W("MaxEntScan 3' splice site is not used in any panel of this figure. Where it is used elsewhere ",
  "in this work it is scored by maxentpy's score3 on 23-mers carrying the acceptor AG at 0-indexed ",
  "positions 18 and 19, spanning the last 20 intronic and the first 3 exonic nucleotides ",
  "(scan_me3, following Yeo and Burge 2004)."),
"",
"Aggregation",
"",
W("Panel A is a metagene of strong 5' splice sites: positions with MaxEnt score5 >= 8 are counted ",
  "in ",NBINS," bins of 100 nt across the window and expressed per kb of scanned sequence, separately ",
  "for each class, with ribbons of +/-1 standard error across sites within the class."),
"",
W("Panel B reduces each site to one scalar, the number of U1 sites (-dG >= 4) per kb over the ",
  "same window, and then each gene to one difference: the value at that gene's real 3' end minus ",
  "the mean over that gene's competitors. Because both terms come from the same gene, this ",
  "difference is free of the between-gene and intron-location differences that the pooled ",
  "comparisons carry."),
"",
W("Panels C and D are computed over every 9-mer position of every site's window, ",
  nfmt(NPOS)," positions in total across the ",nfmt(NSITE)," sites, of which ",nfmt(NGT),
  " (",sprintf("%.2f%%",100*NGT/NPOS),") carry the GT donor dinucleotide. A density is hits per ",
  "kb of scanned positions, so the denominator of the exceedance curve is scanned positions, not ",
  "GT dinucleotides."),
"",
W("Neither panel bins for its reported statistics. The Pearson correlation is ",
  "accumulated from exact running sums over the raw scores, and the exceedance curves are read ",
  "from histograms whose edges are placed so that 4, 7 and 8 fall exactly on an edge, which makes ",
  "P(score >= t) exact at each threshold in use. The heat maps in panel C are binned to 0.5 ",
  "kcal/mol by 1 bit for display only."),
"",
"Thresholds",
"",
W("Three cuts appear in this figure: U1 -dG >= 4, U1 -dG >= 7 and MaxEnt score5 >= 8. Over the ",
  nfmt(NPOS)," scanned positions they admit ",pc(ssv("exceed_u1_4")),", ",pc(ssv("exceed_u1_7")),
  " and ",pc(ssv("exceed_me5_8"))," of positions respectively; over the ",nfmt(NGT),
  " GT-anchored positions they admit ",pc(ssv("exceed_gt_u1_4")),", ",pc(ssv("exceed_gt_u1_7")),
  " and ",pc(ssv("exceed_gt_me5_8")),". Each cut sits inside a smoothly decaying tail rather than ",
  "at a break in the distribution: across the two units of score either side of each one the ",
  "exceedance falls by a steady factor of ",sprintf("%.1f",FOLD("u1_dg",4)),", ",
  sprintf("%.1f",FOLD("u1_dg",7))," and ",sprintf("%.1f",FOLD("me5",8)),
  " per unit of score. They are conventions on a continuum."),
"",
W("The U1 cuts are the pipeline's own. -dG >= 4 is the cutoff the feature pipeline has always ",
  "used, and -dG >= 7 is the threshold that minimises the false-discovery rate against a ",
  "dinucleotide-shuffle null in the calibration of u1_threshold_calibrate.py. That calibration is ",
  "worth quoting rather than hiding: no threshold in [3, 9] reaches an FDR of 0.10 against that ",
  "null, the minimum attained being 0.84 at -dG = 7, so the U1 9-mer score is composition-dominated ",
  "throughout its range. A U1 site in this work is a position of favourable base composition for ",
  "U1 hybridization, not a detected motif, and nothing in this figure is claimed as motif discovery. ",
  "The exceedance measured here, ",pc(ssv("exceed_u1_4"))," at -dG >= 4, sits ",
  sprintf("%.2f",abs(100*ssv("exceed_u1_4")-9.30))," percentage points from the 9.30% that ",
  "calibration found independently over its own peak windows."),
"",
"RESULTS",
"",
W("Real 3' ends sit in sequence that is depleted of strong splice donors, and the depletion is ",
  "confined to the upstream side. In the 100 nt bin immediately upstream of the site the density ",
  "of MaxEnt >= 8 donors is ",pv("real")," per kb at real 3' ends against ",pv("competitor"),
  " at competitors and ",pv("decoy")," at intronic decoys (+/-",pse("real"),", ",pse("competitor"),
  " and ",pse("decoy")," SE; panel A). Averaged over the whole 500 nt upstream of the site the ",
  "ordering is the same, ",mrange(-500,0,"real")," against ",mrange(-500,0,"competitor")," and ",
  mrange(-500,0,"decoy"),". Downstream of the site the three classes converge: averaged over ",
  "+1 to +500 nt they read ",mrange(0,501,"real"),", ",mrange(0,501,"competitor")," and ",
  mrange(0,501,"decoy")," per kb, a residual spread of ",
  sprintf("%.2f",max(sapply(LEV,function(c) as.numeric(mrange(0,501,c))))-
                 min(sapply(LEV,function(c) as.numeric(mrange(0,501,c))))),
  " per kb against the ",
  sprintf("%.2f",as.numeric(mrange(-500,0,"decoy"))-as.numeric(mrange(-500,0,"real"))),
  " per kb that separates real ends from decoys upstream. The bin at -50 nt is the minimum of ",
  "the real-end profile."),
"",
W("The decoy arm is what makes this a statement about 3' ends rather than about introns. ",
  "Competitors lie inside introns and real 3' ends do not, so any pooled contrast between them ",
  "confounds the two. The decoys are intronic and matched to the competitors on distance to the ",
  "canonical poly(A) site and on splice-site context, and they show no depletion at all; the ",
  "depletion tracks the real 3' end, not the intronic location."),
"",
W("The same contrast survives inside individual genes, where gene identity, expression and ",
  "intron architecture are held fixed by construction. Taking each gene's real 3' end minus the ",
  "mean of its own competitors, the real end carries ",sprintf("%.1f",abs(MEAN)),
  " fewer U1 sites per kb over the window (median ",sprintf("%.1f",abs(MED)),"; ",
  sprintf("%.0f%%",100*FBZ)," of ",nfmt(NGP)," genes below zero; two-sided Wilcoxon signed-rank ",
  "on the per-gene differences, V = ",wx("V"),", n = ",nfmt(NWX)," after dropping genes whose ",
  "difference is exactly zero, z = ",wx("z"),", p ",PW,", log10 p about ",wx("log10_p"),
  "; panel B). The effect is a shift of the whole distribution rather than a separation: the ",
  "distribution of per-gene differences is unimodal and straddles zero, with ",
  sprintf("%.0f%%",100*(1-FBZ))," of genes at or above zero. This is a population-level property ",
  "of where real 3' ends sit, not a per-site discriminator, and no classification metric is ",
  "reported for it."),
"",
W("The two donor measurements are largely one measurement. At GT-anchored positions, where a ",
  "donor score is defined, U1 duplex free energy and the MaxEntScan 5' splice-site score correlate ",
  "at Pearson r = ",sprintf("%.3f",ssv("r_gt_pooled"))," over ",nfmt(NGT),
  " positions (panel C), and the value is the same in each class separately (",
  sprintf("%.3f",ssv("r_gt_real")),", ",sprintf("%.3f",ssv("r_gt_competitor")),", ",
  sprintf("%.3f",ssv("r_gt_decoy"))," for real ends, competitors and decoys). Across all ",
  nfmt(NPOS)," scanned positions the correlation falls to ",sprintf("%.3f",ssv("r_all_pooled")),
  ", but that number is diluted rather than informative: only ",
  sprintf("%.2f%%",100*NGT/NPOS)," of positions carry a GT, and at the remaining ",
  sprintf("%.1f%%",100*(1-NGT/NPOS))," no donor exists for either score to measure. The ",
  "thermodynamic and the motif-based read of the donor are therefore treated throughout as a ",
  "single donor axis, and agreement between them is not counted as independent evidence."),
"",
W("What this figure does not contain. It reports no model, no classifier and no ranking metric: ",
  "every quantity is a measured density, a per-gene difference or a correlation. It makes no motif ",
  "claim, for the calibration reason given above. Cleavage-site base composition, including the ",
  "AT-rich dip and its depth, and terminal-exon architecture, including donor-flanked and ",
  "internal-exon-like classification, are measured in separate figures and are not defined or ",
  "shown here."),
""),collapse="\n")
writeLines(SEC,"thesis_section_u1landscape.txt")
e("")
e(SEC)
e("")
NV<-length(grep("[VERIFY",strsplit(SEC,"\n")[[1]],fixed=TRUE))
e("[sec]  [VERIFY] placeholders left in:",NV,"paragraphs")
e("[chk]  panel C headline = GT-anchored r",sprintf("%.3f",ssv("r_gt_pooled")),
  "; all-positions r",sprintf("%.3f",ssv("r_all_pooled")),"is inset context only")
# built from parts so this check cannot match its own source line
PAT<-paste0("RNA","hybrid")
FILES<-c("make_splicing_environment.R","splice_score_redundancy.py","thesis_section_u1landscape.txt")
HITS<-unlist(lapply(FILES,function(f){ L<-readLines(f); paste0(f,":",grep(PAT,L)) }))
HITS<-HITS[!grepl(":$",HITS)]
e("[chk]  '",PAT,"' occurrences across ",paste(FILES,collapse=", "),": ",length(HITS),
  if(length(HITS)) paste0("  (",paste(HITS,collapse=" "),")") else "  (none: the lookup is ViennaRNA duplexfold)")
e("[out]  thesis_section_u1landscape.txt")
```

### `make_splice_model_final.R`

Panels harvested for Figures 3.6 and 3.7. 197 lines, sha256 `6fc53bcfa288`.

```r
#!/usr/bin/env Rscript
# make_splice_model_final.R -- the three finished splice-model panels composed into one figure.
#
#   A  what goes in: the window, the two feature channels, the estimator.  Drawn, not plotted.
#   B  what the model learnt: ridge coefficients by position, acceptor off the positional axis.
#   C  what it achieves: within-gene recall@k on the 4,847 analysis-set genes, LOCO out of fold.
#
# LAYOUT. A and B share one x scale, limits and breaks, so the schematic window sits directly
# above the coefficient profile and the figure reads top to bottom as window -> features ->
# weight at each position. patchwork aligns the panel regions, so the alignment survives the
# differing y-axis label widths. Every label in A is placed INSIDE that shared x range for
# this reason; nothing hangs off to the right. C is a curve rather than a positional track, so
# it sits centred and narrower at the bottom rather than being stretched to full width.
#
# Panels are assembled, not redesigned. Nothing is refitted: A is fixed coordinates, B reads
# splice_model_26_coefficients.tsv and C reads recall_at_k.tsv, and each panel's key number is
# asserted before the figure is written.
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(patchwork)})
e<-function(...) cat(...,"\n",file=stderr())
INK<-"#1A1A1A"; BASE<-7
U1C<-"#08519C"; ACCC<-"#E8A33D"; GREY<-"#9AA0A6"; GTX<-"#5A6068"
X0<--2000; X1<-500; BW<-100
XLO<--2150; XHI<-1280                       # the shared x range for A and B
XBRK<-seq(-2000,500,500)
CF<-read_tsv("splice_model_26_coefficients.tsv",show_col_types=FALSE)
RK<-read_tsv("recall_at_k.tsv",show_col_types=FALSE)
MT<-read_tsv("splice_model_26_metrics.tsv",show_col_types=FALSE)
mv<-function(k,c="value") MT[[c]][MT$metric==k][1]
base_th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-1),axis.line=element_line(color=INK,linewidth=.3),
  axis.title=element_text(size=BASE),
  plot.title=element_text(size=BASE+1,face="bold",hjust=0,margin=margin(b=4)),
  legend.position="none")

## ================= A: schematic =================
ed<-seq(X0,X1,by=BW)
yW<-3.40; hW<-0.26; yU<-2.25; yA<-1.30; hC<-0.24
pA<-ggplot()+
  annotate("rect",xmin=X0,xmax=0,ymin=yW-hW,ymax=yW+hW,fill="#EDEFF2",color=NA)+
  annotate("rect",xmin=0,xmax=X1,ymin=yW-hW,ymax=yW+hW,fill="#F7F8FA",color=NA)+
  annotate("segment",x=ed,xend=ed,y=yW-hW,yend=yW+hW,color="white",linewidth=.35)+
  annotate("rect",xmin=X0,xmax=X1,ymin=yW-hW,ymax=yW+hW,fill=NA,color=INK,linewidth=.35)+
  annotate("segment",x=0,xend=0,y=yW-hW-0.16,yend=yW+hW+0.14,linetype=2,color=INK,linewidth=.4)+
  annotate("text",x=-60,y=yW+hW+0.20,label="cleavage site",size=2.2,color=INK,hjust=1,vjust=0)+
  annotate("text",x=X0,y=yW+hW+0.20,label="Sequence window, 25 bins of 100 nt",
           size=2.1,color=GTX,hjust=0,vjust=0)+
  annotate("rect",xmin=X0,xmax=X1,ymin=yU-hC,ymax=yU+hC,fill=U1C,alpha=.85,color=NA)+
  annotate("segment",x=ed,xend=ed,y=yU-hC,yend=yU+hC,color="white",linewidth=.3)+
  annotate("text",x=X0,y=yU+hC+0.15,hjust=0,vjust=0,size=2.1,color=U1C,
           label="U1 duplex density (-dG >= 4) per bin, 25 values")+
  annotate("rect",xmin=X0,xmax=-126,ymin=yA-hC,ymax=yA+hC,fill=ACCC,alpha=.30,color=NA)+
  annotate("rect",xmin=-1120,xmax=-940,ymin=yA-hC,ymax=yA+hC,fill=ACCC,color=NA)+
  annotate("text",x=X0,y=yA+hC+0.15,hjust=0,vjust=0,size=2.1,color=ACCC,
           label="Maximum MaxEnt 3' splice-site score over [-2,000, -126], 1 value")+
  annotate("segment",x=380,xend=380,y=yW-hW-0.03,yend=yU+hC+0.03,color=GTX,linewidth=.3,
           arrow=grid::arrow(length=unit(.045,"in"),type="closed"))+
  annotate("segment",x=380,xend=380,y=yU-hC-0.03,yend=yA+hC+0.03,color=GTX,linewidth=.3,
           arrow=grid::arrow(length=unit(.045,"in"),type="closed"))+
  annotate("segment",x=-750,xend=-750,y=yA-hC-0.04,yend=0.74,color=GTX,linewidth=.3,
           arrow=grid::arrow(length=unit(.045,"in"),type="closed"))+
  annotate("rect",xmin=-1720,xmax=220,ymin=0.20,ymax=0.72,fill="white",color=INK,linewidth=.4)+
  annotate("text",x=-750,y=0.575,label="26 features",size=2.5,color=INK,fontface="bold")+
  annotate("text",x=-750,y=0.345,size=2.05,color=INK,
           label="Ridge (L2) logistic regression, features centred within gene")+
  annotate("segment",x=-750,xend=-750,y=0.18,yend=-0.02,color=GTX,linewidth=.3,
           arrow=grid::arrow(length=unit(.045,"in"),type="closed"))+
  annotate("text",x=-750,y=-0.10,vjust=1,size=2.05,color=INK,
           label="Within-gene evaluation: rank of the real 3' end among its gene's candidates")+
  scale_x_continuous(breaks=XBRK,limits=c(XLO,XHI),expand=expansion(mult=c(0,0)))+
  scale_y_continuous(limits=c(-0.42,3.95),expand=expansion(mult=c(0,0)))+
  labs(title="A   Feature construction and model specification",x=NULL,y=NULL)+
  base_th+theme(axis.text=element_blank(),axis.ticks=element_blank(),
                axis.line=element_blank(),plot.margin=margin(4,4,0,4))

## ================= B: coefficient profile =================
U<-CF%>%filter(!is.na(channel),channel=="u1_dens4")
A<-CF$coefficient[which(CF$channel=="acceptor")][1]
stopifnot(nrow(U)==25,is.finite(A))
jm<-which.min(U$coefficient)
stopifnot(abs(U$coefficient[jm]+0.2273)<5e-4, U$bin_centre[jm]==-150, abs(A-0.2959)<5e-4)
e(sprintf("[B] U1 minimum %+.4f at %+d nt (expected -0.2273 at -150); acceptor %+.4f (expected +0.2959)",
          U$coefficient[jm],U$bin_centre[jm],A))
rng<-range(c(U$coefficient,A,0)); pad<-diff(rng)*0.17
pB<-ggplot(U,aes(bin_centre,coefficient))+
  annotate("rect",xmin=-Inf,xmax=0,ymin=-Inf,ymax=Inf,fill=GREY,alpha=.07)+
  geom_hline(yintercept=0,color=INK,linewidth=.35)+
  geom_vline(xintercept=0,linetype=2,color=INK,linewidth=.35)+
  geom_col(width=88,fill=U1C)+
  annotate("text",x=U$bin_centre[jm],y=U$coefficient[jm],vjust=1.6,size=2.1,color=U1C,
           label=sprintf("%+.2f at %+d nt",U$coefficient[jm],U$bin_centre[jm]))+
  annotate("segment",x=700,xend=700,y=rng[1]-pad,yend=rng[2]+pad,color=GREY,linewidth=.3)+
  annotate("rect",xmin=850,xmax=1130,ymin=0,ymax=A,fill=ACCC)+
  annotate("text",x=990,y=A,label=sprintf("%+.2f",A),size=2.2,color=ACCC,vjust=-0.6)+
  annotate("text",x=990,y=rng[1]-pad*0.45,label="MaxEnt 3'ss\nscalar, not positional",
           size=2.0,color=ACCC,vjust=1,lineheight=1.1)+
  scale_x_continuous(breaks=XBRK,limits=c(XLO,XHI),expand=expansion(mult=c(0,0)))+
  scale_y_continuous(limits=c(rng[1]-pad*1.2,rng[2]+pad*1.2),expand=expansion(mult=c(0,0)))+
  labs(title="B   Ridge regression coefficients by position",
       x="position relative to the cleavage site (nt)",y="ridge coefficient")+
  base_th+theme(plot.margin=margin(2,4,4,4))

## ================= C: recall@k =================
NG<-RK$n_genes[1]; KM<-max(RK$k); g<-function(k,c="recall") RK[[c]][RK$k==k][1]
stopifnot(NG==4847)
e(sprintf("[C] %d analysis-set genes, recall@1 %.3f (chance %.3f), recall@3 %.3f (chance %.3f)",
          NG,g(1),g(1,"chance"),g(3),g(3,"chance")))
AN<-RK%>%filter(k %in% c(1,3))%>%
  mutate(lab=sprintf("recall@%d   %.3f\nrandom       %.3f",k,recall,chance))
pC<-ggplot(RK,aes(k))+
  geom_ribbon(aes(ymin=chance,ymax=recall),fill=U1C,alpha=.13)+
  geom_line(aes(y=chance),linetype=2,color=GREY,linewidth=.6)+
  geom_point(aes(y=chance),color=GREY,size=1.2)+
  geom_line(aes(y=recall),color=U1C,linewidth=.85)+
  geom_point(aes(y=recall),color=U1C,size=1.7)+
  geom_point(data=RK%>%filter(k==1),aes(y=recall),size=3.2,shape=21,fill=U1C,
             color="white",stroke=.8)+
  geom_text(data=AN,aes(y=recall,label=lab),hjust=-0.10,vjust=1.35,size=2.2,color=INK,
            lineheight=1.25)+
  annotate("text",x=KM+0.4,y=g(KM,"chance"),label="random\nranking",hjust=1,vjust=3.1,
           size=2.2,color=GREY,lineheight=1.1)+
  scale_x_continuous(breaks=1:KM,limits=c(1,KM+0.55),expand=expansion(mult=c(.02,.02)))+
  scale_y_continuous(limits=c(0.20,1),labels=scales::percent_format(accuracy=1),
                     expand=expansion(mult=c(0,.02)))+
  labs(title="C   Within-gene recall as a function of rank threshold",
       x="rank threshold k",y="genes with the true 3' end at rank <= k")+
  base_th+theme(panel.grid.major.y=element_line(color="#EEF0F3",linewidth=.25),
                plot.margin=margin(8,4,4,4))

## ================= D: within-gene score separation =================
SS<-read_tsv("score_separation.tsv",show_col_types=FALSE)
ssv<-function(k){ v<-SS$value[SS$key==k]; if(!length(v)) stop("missing: ",k); v[1] }
SD<-read_tsv("score_separation_sites.tsv",show_col_types=FALSE)
stopifnot(as.numeric(ssv("n_genes"))==NG, ssv("oof")=="leave-one-chromosome-out")
# PROOF that C and D share one scoring: recompute recall@1 from panel D's own scores and
# check it against panel C's file. Gene-centring does not change within-gene ranking, so
# ranks from the centred scores are the ranks panel C used.
CHK<-SD%>%group_by(gene)%>%
  summarise(top1=as.numeric(centred_score[class=="true"][1] >
                            max(centred_score[class=="competitor"])),.groups="drop")
R1D<-mean(CHK$top1); R1C<-g(1)
stopifnot(nrow(CHK)==NG, abs(R1D-R1C)<1e-6)
e(sprintf("[consistency] recall@1 recomputed from panel D's scores = %.4f, panel C's file = %.4f",
          R1D,R1C))
e(sprintf("[consistency] win-rate %.4f and the Wilcoxon (V = %s, z = %s, p %s) come from the",
          as.numeric(ssv("winrate_loco")),ssv("wilcoxon_V"),ssv("wilcoxon_z"),ssv("wilcoxon_p")))
e("[consistency] same LOCO scores over the same",NG,"genes; the held-out 0.472 is not used here")
WRD<-as.numeric(ssv("winrate_loco")); WLO<-as.numeric(ssv("winrate_lo"))
WHI<-as.numeric(ssv("winrate_hi")); MED<-as.numeric(ssv("median_delta"))
FP<-as.numeric(ssv("frac_positive"))
SD2<-SD%>%mutate(class=factor(class,levels=c("true","competitor"),
                              labels=c("true 3' end","competitor")))
QD<-SD2%>%group_by(class)%>%summarise(m=median(centred_score),.groups="drop")
YL<-quantile(SD2$centred_score,c(.002,.998)); YTOP<-YL[2]+0.62*diff(YL)
pD<-ggplot(SD2,aes(class,centred_score,fill=class,color=class))+
  geom_hline(yintercept=0,linetype=2,color=GREY,linewidth=.4)+
  geom_violin(width=.86,linewidth=.35,alpha=.45,trim=TRUE)+
  geom_boxplot(width=.13,outlier.shape=NA,linewidth=.35,fill="white",color=INK)+
  scale_fill_manual(values=c(`true 3' end`=U1C,competitor=ACCC))+
  scale_color_manual(values=c(`true 3' end`=U1C,competitor=ACCC))+
  geom_text(data=QD,aes(x=class,y=m,label=sprintf("%+.2f",m)),inherit.aes=FALSE,
            hjust=-0.55,size=2.1,color=INK)+
  annotate("text",x=0.52,y=YTOP,hjust=0,vjust=1.1,size=2.0,color=INK,lineheight=1.35,
    label=paste0("Win-rate ",sprintf("%.3f",WRD)," [",sprintf("%.3f",WLO),", ",
      sprintf("%.3f",WHI),"]\nPaired Wilcoxon on the per-gene difference\nV = ",
      format(as.numeric(ssv("wilcoxon_V")),big.mark=","),", n = ",format(NG,big.mark=","),
      ", z = ",sprintf("%+.1f",as.numeric(ssv("wilcoxon_z"))),", p ",ssv("wilcoxon_p"),
      "\nmedian difference ",sprintf("%+.2f",MED),", positive in ",
      sprintf("%.0f%%",100*FP)," of genes"))+
  coord_cartesian(ylim=c(YL[1],YTOP))+
  labs(title="D   Score distributions at real 3' ends and competitors",
       x=NULL,y="gene-centred model score")+
  base_th+theme(axis.text.x=element_text(size=BASE),
                panel.grid.major.y=element_line(color="#EEF0F3",linewidth=.25),
                plot.margin=margin(8,4,4,4))

CAP<-paste(strwrap(paste0("n = ",format(NG,big.mark=",")," genes, each with one real 3' end ",
  "and at least one competitor. Panels C and D are leave-one-chromosome-out over those genes ",
  "and share one scoring; C's shaded band is the gap to random ranking, D's scores are ",
  "centred within gene."),width=118),collapse="\n")
fig<-(pA/pB/(pC|pD))+plot_layout(heights=c(1.20,1.0,1.05))+
  plot_annotation(
    title="Positional U1 duplex density and upstream 3' splice-site strength discriminate real 3' ends from intragenic competitors",
    subtitle="Ridge logistic model, 26 features: U1 duplex density (-dG >= 4) in 25 bins of 100 nt across [-2,000, +500], and the maximum upstream MaxEnt 3' splice-site score",
    caption=CAP,
    theme=theme(plot.title=element_text(size=BASE+2.5,face="bold",color=INK),
                plot.subtitle=element_text(size=BASE,color=GTX,margin=margin(b=6)),
                plot.caption=element_text(size=BASE-1.6,color=GTX,hjust=0,lineheight=1.25)))
ggsave("figure_splice_model_final.pdf",fig,width=7.8,height=8.6,device=grDevices::cairo_pdf)
ggsave("figure_splice_model_final.png",fig,width=7.8,height=8.6,dpi=300)
e("[layout] A top full width, B full width sharing A's x limits",XLO,"to",XHI,"so the window",
  "aligns with the coefficient profile, and C and D side by side in a bottom row")
BAN<-c("CNN","shuffl","scalar splice","dinucleotide","whole genome","whole-genome")
SRC<-c(CAP,"An acceptor score and positional U1 density place the 3' end within its gene",
       "U1 duplex density (-dG >= 4) in 25 bins of 100 nt across [-2,000, +500] plus the upstream MaxEnt 3' splice-site score: 26 features")
e("[chk] banned terms in the drawn figure:",sum(sapply(BAN,function(b) sum(grepl(b,SRC)))))
e("[chk] panel headings carry a letter and a short title only, no subtitles")
e("[out] figure_splice_model_final.pdf figure_splice_model_final.png")
```

### `make_figure_architecture.R`

Panel b of Figure 3.8. 111 lines, sha256 `f8288552cd0f`.

```r
# make_figure_architecture.R -- the network as a diagram.
#
# The architecture is cnn_model_def.py. Every dimension and the receptive field are COMPUTED here
# from the layer parameters rather than typed, so the diagram cannot drift from the model:
#
#   c1  Conv1d(4, 32, 19, pad 9), BatchNorm, exp(clamp 8)   the layer whose filters are readable
#       max_pool 10
#   c2  Conv1d(32, 64, 7, pad 3), BatchNorm, ReLU
#       max_pool 5
#   c3  Conv1d(64, 96, 5, pad 2), BatchNorm, ReLU
#       max_pool 2                                          25 positional bins, position retained
#   fc1 Linear(96 x 25, 64), ReLU, Dropout 0.4
#   fc2 Linear(64, 1)
suppressPackageStartupMessages({library(ggplot2);library(ggforce);library(dplyr)})
INK<-"#1A1A1A"; GTX<-"#5A6068"; GREY<-"#8A8F98"; PAPER<-"#FCFCFD"
# One colour for every computed block. Colour previously encoded "layer 1 is interpretable" vs
# "layers 2 and 3", which is an interpretive claim and belongs in the text, not in the fill.
# soft fills with a thin ink outline, so the stack reads as outlined cards rather than flat slabs
SLATE<-"#BCD3E8"; SLATE_B<-"#DFE9F3"        # computed blocks, front and the copies behind
PALE <-"#EDE9E3"; PALE_B <-"#F6F4F0"        # input, a different material because it is data
LW<-0.30; LWB<-0.20; BASE<-9

# ---- dimensions and receptive field, derived ----
L0<-2500
ops<-list(list(k=19,s=1),list(k=10,s=10),list(k=7,s=1),list(k=5,s=5),
          list(k=5,s=1),list(k=2,s=2))
rf<-1; jump<-1
for (o in ops){ rf<-rf+(o$k-1)*jump; jump<-jump*o$s }
RF<-rf; STRIDE<-jump
stopifnot(RF==378, STRIDE==100, L0/STRIDE==25)

ST<-tribble(
  ~id, ~lab,          ~len,  ~ch, ~fill,  ~sub,
  1,   "input",       2500,   4,  PALE,   "one-hot DNA",
  2,   "conv 19",     2500,  32,  SLATE,  "BatchNorm, exp",
  3,   "max pool 10",  250,  32,  SLATE,  "",
  4,   "conv 7",       250,  64,  SLATE,  "BatchNorm, ReLU",
  5,   "max pool 5",    50,  64,  SLATE,  "",
  6,   "conv 5",        50,  96,  SLATE,  "BatchNorm, ReLU",
  7,   "max pool 2",    25,  96,  SLATE,  "position kept"
)
# width carries sequence length on a log scale, height carries channel count
ST<-ST%>%mutate(w=2.6+9.4*(len/max(len))^(1/3), h=2.6+7.4*sqrt(ch/96))
gapx<-4.8   # must clear the DEPTH offset copies and still leave a readable arrow
ST$x1<-NA; ST$x2<-NA; cur<-0
for (i in seq_len(nrow(ST))){ ST$x1[i]<-cur; ST$x2[i]<-cur+ST$w[i]; cur<-cur+ST$w[i]+gapx }
ST<-ST%>%mutate(xm=(x1+x2)/2, y1=-h/2, y2=h/2)
LB<-min(ST$y1)-0.9   # one baseline for every label, so they do not step with block height
DEPTH<-4; DX<-0.42; DY<-0.34            # offset copies behind each slab suggest channel depth
BFILL<-c(setNames(SLATE_B,SLATE), setNames(PALE_B,PALE))
back<-do.call(rbind,lapply(seq_len(nrow(ST)),function(i)
  do.call(rbind,lapply(DEPTH:1,function(d) data.frame(
    x1=ST$x1[i]+d*DX, x2=ST$x2[i]+d*DX, y1=ST$y1[i]+d*DY, y2=ST$y2[i]+d*DY,
    fill=unname(BFILL[ST$fill[i]]))))))
# the dense head, drawn as nodes rather than a slab because position is gone by then
headx<-cur+3.0
fcx<-c(headx+5.6, headx+13.0, headx+19.6)   # room for the flatten label between pool 2 and dense
nodes<-rbind(
  data.frame(x=fcx[1], y=seq(-3.4,3.4,length.out=11), r=0.30, fill=SLATE_B),
  data.frame(x=fcx[2], y=seq(-2.2,2.2,length.out=7),  r=0.34, fill=SLATE),
  data.frame(x=fcx[3], y=0, r=0.62, fill=INK))
edges<-rbind(
  expand.grid(x=fcx[1], xe=fcx[2], y=seq(-3.4,3.4,length.out=11), ye=seq(-2.2,2.2,length.out=7)),
  expand.grid(x=fcx[2], xe=fcx[3], y=seq(-2.2,2.2,length.out=7),  ye=0))
XMAX<-fcx[3]+4.6
XR<-c(-1.5,XMAX); YR<-c(min(ST$y1)-4.4,6.6)
# operation arrows between slabs
arr<-data.frame(x=head(ST$x2,-1)+DEPTH*DX+0.55, xe=tail(ST$x1,-1)-0.45, y=0, ye=0)
set.seed(7)
NB<-26L; BCOL<-c(A="#3D9A50",C="#2C6FB5",G="#E8A33D",T="#C6413B")
oh<-expand.grid(j=seq_len(NB),row=1:4)%>%
  mutate(base=names(BCOL)[row],
         cw=(ST$x2[1]-ST$x1[1]-0.7)/NB, rh=(ST$y2[1]-ST$y1[1]-0.7)/4,
         x=ST$x1[1]+0.35+(j-0.5)*cw, y=ST$y2[1]-0.35-(row-0.5)*rh)
hot<-sample(1:4,NB,replace=TRUE)
oh<-oh%>%mutate(on=row==hot[j])
p<-ggplot()+
  geom_rect(data=back,aes(xmin=x1,xmax=x2,ymin=y1,ymax=y2,fill=I(fill)),colour=INK,linewidth=LWB)+
  geom_segment(data=edges,aes(x,y,xend=xe,yend=ye),colour=GTX,linewidth=.10,alpha=.30)+
  geom_circle(data=nodes,aes(x0=x,y0=y,r=r,fill=I(fill)),colour=INK,linewidth=LWB)+
  geom_rect(data=ST,aes(xmin=x1,xmax=x2,ymin=y1,ymax=y2,fill=I(fill)),colour=INK,linewidth=LW)+
  geom_tile(data=oh,aes(x,y,width=cw*.82,height=rh*.72,
                        fill=I(ifelse(on,BCOL[base],"#FFFFFF")),alpha=I(ifelse(on,1,.22))))+
  geom_segment(data=arr,aes(x,y,xend=xe,yend=ye),colour=GTX,linewidth=.32,
               arrow=arrow(length=unit(3.4,"pt"),type="closed"))+
  # name, then parameters, then dimensions, all hanging under the block they describe
  geom_text(data=ST,aes(xm,LB,label=lab),size=2.65,colour=INK,fontface="bold",vjust=1)+
  geom_text(data=ST%>%filter(sub!=""),aes(xm,LB-1.2,label=sub),size=2.3,colour=GTX,vjust=1)+
  geom_text(data=ST,aes(xm,LB-2.4,label=sprintf("%s x %s",ch,format(len,big.mark=",",trim=TRUE))),
            size=2.5,colour=INK,vjust=1)+
  # the head
  geom_segment(aes(x=ST$x2[7]+DEPTH*DX+0.55,xend=fcx[1]-1.25,y=0,yend=0),colour=GTX,linewidth=.32,
               arrow=arrow(length=unit(3.4,"pt"),type="closed"))+
  geom_text(aes(x=ST$x2[7]+DEPTH*DX+0.6,y=LB,label="flatten"),size=2.65,colour=INK,fontface="bold",vjust=1,hjust=0)+
  geom_text(aes(x=ST$x2[7]+DEPTH*DX+0.6,y=LB-1.2,label="2,400"),size=2.5,colour=INK,vjust=1,hjust=0)+
  geom_text(aes(x=fcx[1],y=LB,label="dense 64"),size=2.65,colour=INK,fontface="bold",vjust=1)+
  geom_text(aes(x=fcx[1],y=LB-1.2,label="ReLU, Dropout 0.4"),size=2.3,colour=GTX,vjust=1)+
  geom_text(aes(x=fcx[3],y=LB,label="logit"),size=2.65,colour=INK,fontface="bold",vjust=1)+
  # the two things a reader needs that are not dimensions

  annotate("text",x=-1.0,y=6.0,hjust=0,size=2.55,colour=GTX,
           label=sprintf("one output bin sees %d nt of sequence",RF))+
  coord_fixed(xlim=XR,ylim=YR,expand=FALSE,clip="off")+
  theme_void(base_size=BASE)+
  theme(plot.background=element_rect(fill=PAPER,colour=NA),plot.margin=margin(6,8,6,8))
W<-9.4; H<-W*diff(YR)/diff(XR)+0.10                 # +margins
ggsave("figure_architecture.pdf",p,width=W,height=H,device=cairo_pdf)
ggsave("figure_architecture.png",p,width=W,height=H,dpi=400,bg=PAPER)
cat(sprintf("[size] data aspect %.2f, saved %.2f x %.2f in\n",diff(XR)/diff(YR),W,H))
cat(sprintf("[arch] receptive field %d nt, stride %d nt, %d bins\n",RF,STRIDE,L0/STRIDE))
cat("[out] figure_architecture.pdf / .png\n")
```

### `make_figure_rescue.R`

Panel a of Figure 3.8. 238 lines, sha256 `74cfb93b679f`.

```r
# make_figure_rescue.R -- the rescue as a flow, not a stack of bar charts.
# Reads rescue_rate_genes.tsv and rescue_ranks.tsv only. No computation.
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(tidyr);library(ggforce)})
INK<-"#16191C"; GTX<-"#5A6068"; PAPER<-"#FFFFFF"
NEU<-"#B6BDC4"; FAIL<-"#2A6DB0"; GOOD<-"#14395E"; MISS<-"#D9A441"; SPL<-"#E8A33D"
BASE<-8
G<-read.delim("rescue_rate_genes.tsv",stringsAsFactors=FALSE)
RK<-read.delim("rescue_ranks.tsv",stringsAsFactors=FALSE)
B<-G%>%filter(candidate_set=="full_shortlist")
N_ALL<-15560; N_RES<-8977; N_CON<-nrow(B)
N_REC<-sum(B$cnn); N_MISS<-N_CON-N_REC
N_OK0<-N_RES; N_OK1<-N_RES+N_REC
stopifnot(N_CON==6583)
# attribution, needed by both the caption and panel c
sp1<-RK$rank_splice==1; cn1<-RK$rank_cnn==1; cb1<-RK$rank_combo==1
nA<-nrow(RK); fCON<-N_CON/N_ALL; fRES<-N_RES/N_ALL
TOT_CB<-fRES+mean(cb1)*fCON
TOT_OR<-fRES+mean(sp1|cn1)*fCON

## ---------------- a: what each model reads ----------------
# The baseline's features are 10 core polyadenylation PWMs from BaselineModelPWMs, PAS in three
# variants, four downstream elements, UGUA, poly-U and NUDT, each scanned in 30 nt micro-windows at
# stride 10 across a 500 nt macro window, 48 bins per motif. The extended models read 2,500 nt,
# [-2000, +500) in transcript orientation.
#
# The four core elements sit within 250 nt of each other, which is 10% of the drawn span, so their
# labels cannot be placed on the main axis. The 500 nt baseline window is therefore magnified below,
# which also makes the point of the panel: the baseline reads a small neighbourhood in detail and
# the extended models read the whole span. Element positions are schematic; spans and motifs are not.
mag<-function(x) -2000+(x+250)/500*2500          # [-250,+250] stretched onto the full axis
yTOP<-0; yMAG<--1.55
WIDE<-tribble(~x,~lab,~col,
  -1750,"5' splice site",SPL, -1180,"5' splice site",SPL,
   -760,"3' splice site",FAIL, -430,"5' splice site",SPL)
# spaced so the labels clear: the magnifier is 5 plot units per nt and a label needs about 500
CORE<-tribble(~x,~lab,
  -170,"UGUA",  -58,"AAUAAA",  56,"U-rich",  170,"GU-rich")
pA<-ggplot()+
  annotate("rect",xmin=-2000,xmax=500,ymin=yTOP-.30,ymax=yTOP+.30,fill=FAIL,alpha=.13)+
  annotate("segment",x=-2000,xend=500,y=yTOP,yend=yTOP,colour="#B9C0C7",linewidth=.9)+
  annotate("rect",xmin=-250,xmax=250,ymin=yTOP-.30,ymax=yTOP+.30,fill=NEU,alpha=.75)+
  geom_point(data=WIDE,aes(x,yTOP,fill=I(col)),shape=21,colour=PAPER,size=2.7,stroke=.5)+
  geom_text(data=WIDE%>%distinct(lab,.keep_all=TRUE),aes(x,yTOP+.44,label=lab),
            size=2.2,colour=GTX)+
  annotate("segment",x=0,xend=0,y=yTOP+.30,yend=yTOP+.80,colour=INK,linewidth=.45)+
  annotate("point",x=0,y=yTOP+.30,shape=25,size=1.6,fill=INK,colour=INK)+
  annotate("text",x=0,y=yTOP+.96,label="cleavage site",size=2.35,colour=INK)+
  annotate("text",x=-2000,y=yTOP+1.16,hjust=0,size=2.7,colour=INK,fontface="bold",
           label="extended models")+
  annotate("text",x=-2000,y=yTOP+.80,hjust=0,size=2.35,colour=GTX,label="2,500 nt")+
  # magnifier
  annotate("segment",x=-250,xend=mag(-250),y=yTOP-.32,yend=yMAG+.34,colour="#C6CBD0",
           linewidth=.3,linetype=2)+
  annotate("segment",x=250,xend=mag(250),y=yTOP-.32,yend=yMAG+.34,colour="#C6CBD0",
           linewidth=.3,linetype=2)+
  annotate("rect",xmin=mag(-250),xmax=mag(250),ymin=yMAG-.30,ymax=yMAG+.30,fill=NEU,alpha=.55)+
  annotate("segment",x=mag(-250),xend=mag(250),y=yMAG,yend=yMAG,colour="#B9C0C7",linewidth=.9)+
  annotate("segment",x=mag(0),xend=mag(0),y=yMAG-.30,yend=yMAG+.30,colour=INK,linewidth=.4)+
  geom_point(data=CORE,aes(mag(x),yMAG),fill=GOOD,shape=21,colour=PAPER,size=2.7,stroke=.5)+
  geom_text(data=CORE,aes(mag(x),yMAG-.50,label=lab),size=2.2,colour=GTX)+
  annotate("text",x=-2000,y=yMAG+1.12,hjust=0,size=2.7,colour=INK,fontface="bold",
           label="baseline")+
  annotate("text",x=-2000,y=yMAG+.78,hjust=0,size=2.35,colour=GTX,label="500 nt")+
  # span ends sit outside each bar, so nothing crowds the gap between the two rows
  annotate("text",x=-2030,y=yTOP,hjust=1,label="-2,000",size=2.0,colour=GTX)+
  annotate("text",x=530,y=yTOP,hjust=0,label="+500",size=2.0,colour=GTX)+
  annotate("text",x=mag(-250)-30,y=yMAG,hjust=1,label="-250",size=2.0,colour=GTX)+
  annotate("text",x=mag(250)+30,y=yMAG,hjust=0,label="+250",size=2.0,colour=GTX)+
  annotate("text",x=0,y=yMAG-1.00,size=2.1,colour=GTX,
           label="nt relative to the cleavage site")+
  scale_x_continuous(limits=c(-2300,780))+
  coord_cartesian(ylim=c(-2.60,1.86),expand=FALSE,clip="off")+
  labs(title="a   Sequence context read by each model")+
  theme_void(base_size=BASE)+
  theme(text=element_text(colour=INK),
        plot.title=element_text(face="bold",size=BASE+0.5,colour=INK,margin=margin(b=6)),
        plot.background=element_rect(fill=PAPER,colour=NA),plot.margin=margin(6,8,10,8))

## ---------------- a: the flow, with attribution ----------------
# Panel c is folded in here. That forces one universe on the whole figure, because the attribution
# and the flow cannot share an axis otherwise: the CNN's 3,800 rescues are counted against the FULL
# candidate list of all 6,583 contested genes, while the per-model attribution exists only for the
# 4,847 whose candidates both models score, and those two candidate sets are not the same, so their
# counts are not additive. The universe here is therefore the 8,977 genes the baseline resolves plus
# those 4,847, and every number in the flow is exact within it. The CNN's figure over all 6,583,
# 0.577 to 0.821, is in the caption.
N_UNI<-N_RES+nA
N_BOTH<-sum(sp1&cn1); N_CNNO<-sum(!sp1&cn1); N_SPLO<-sum(sp1&!cn1); N_NEI<-sum(!sp1&!cn1)
N_ANY<-N_BOTH+N_CNNO+N_SPLO
N_OKU<-N_RES+N_ANY
P_BASE<-N_RES/N_UNI; P_ORAC<-N_OKU/N_UNI; P_COMB<-(N_RES+sum(cb1))/N_UNI
stopifnot(N_BOTH+N_CNNO+N_SPLO+N_NEI==nA)
GAP<-620
nodes<-tribble(~id,~x,~lo,~fill,~lab,~n,~side,
  "all",  0, 0,                                    NEU,  "all genes",         N_UNI,  "l",
  "res",  1, 0,                                    NEU,  "baseline resolves", N_RES,  "r",
  "con",  1, N_RES+GAP,                            FAIL, "baseline fails",    nA,     "t",
  "keep", 2, 0,                                    NEU,  "already correct",   N_RES,  "n",
  "both", 2, N_RES+GAP,                            GOOD, "both models",       N_BOTH, "n",
  "cnno", 2, N_RES+GAP+N_BOTH+220,                 FAIL, "sequence CNN only", N_CNNO, "n",
  "splo", 2, N_RES+GAP+N_BOTH+N_CNNO+440,          SPL,  "splice only",       N_SPLO, "n",
  "nei",  2, N_RES+GAP+N_BOTH+N_CNNO+N_SPLO+660,   MISS, "neither",           N_NEI,  "n",
  "ok",   3, 0,                                    GOOD, "correct",           N_OKU,  "r",
  "bad",  3, N_OKU+GAP,                            MISS, "incorrect",         N_NEI,  "r")%>%
  mutate(hi=lo+n)
nd<-function(i) nodes[nodes$id==i,]
rib<-function(a,b,ao,bo,n,fill,alpha){
  A<-nd(a); B_<-nd(b)
  data.frame(x=c(A$x+.055,B_$x-.055,B_$x-.055,A$x+.055),
             y=c(A$lo+ao,B_$lo+bo,B_$lo+bo+n,A$lo+ao+n),
             fill=fill,alpha=alpha,g=paste0(a,b,ao))
}
ribs<-rbind(
  rib("all","res",0,0,N_RES,NEU,.42),
  rib("all","con",N_RES,0,nA,FAIL,.36),
  rib("res","keep",0,0,N_RES,NEU,.38),
  rib("con","both",0,0,N_BOTH,GOOD,.50),
  rib("con","cnno",N_BOTH,0,N_CNNO,FAIL,.46),
  rib("con","splo",N_BOTH+N_CNNO,0,N_SPLO,SPL,.60),
  rib("con","nei",N_BOTH+N_CNNO+N_SPLO,0,N_NEI,MISS,.50),
  rib("keep","ok",0,0,N_RES,NEU,.38),
  rib("both","ok",0,N_RES,N_BOTH,GOOD,.55),
  rib("cnno","ok",0,N_RES+N_BOTH,N_CNNO,FAIL,.50),
  rib("splo","ok",0,N_RES+N_BOTH+N_CNNO,N_SPLO,SPL,.62),
  rib("nei","bad",0,0,N_NEI,MISS,.50))
lab<-nodes%>%mutate(ym=(lo+hi)/2,
  txt=sprintf("%s\n%s   %.1f%%",lab,format(n,big.mark=","),100*n/N_UNI))
mid<-lab%>%filter(side=="n",id!="keep")%>%
  mutate(tx=sprintf("%s   %s   %.1f%%",lab,format(n,big.mark=","),100*n/N_UNI))
pB2<-ggplot()+
  geom_diagonal_wide(data=ribs,aes(x,y,group=g,fill=I(fill),alpha=I(alpha)),
                     strength=.72,colour=NA)+
  geom_rect(data=nodes,aes(xmin=x-.028,xmax=x+.028,ymin=lo,ymax=hi,fill=I(fill)),colour=NA)+
  geom_text(data=lab%>%filter(side=="l"),aes(x-.055,ym,label=txt),hjust=1,size=2.5,
            colour=INK,lineheight=1.06)+
  geom_text(data=lab%>%filter(side=="r"),aes(x+.055,ym,label=txt),hjust=0,size=2.5,
            colour=INK,lineheight=1.06)+
  geom_text(data=lab%>%filter(side=="t"),aes(x,hi+560,label=txt),hjust=.5,vjust=0,size=2.5,
            colour=INK,lineheight=1.06)+
  geom_text(data=mid,aes(x-.075,ym,label=tx),hjust=1,size=2.3,colour=INK)+
  annotate("text",x=-.34,y=N_UNI+1750,
           label=sprintf("P@1   %.3f   \u2192   %.3f",P_BASE,P_ORAC),
           size=3.5,colour=INK,fontface="bold",hjust=0)+
  coord_cartesian(xlim=c(-.46,3.40),ylim=c(-1200,N_UNI+3100),expand=FALSE,clip="off")+
  labs(title="b   Recovery of baseline-unresolved 3' ends")+
  theme_void(base_size=BASE)+
  theme(plot.title=element_text(face="bold",size=BASE+0.5,colour=INK,margin=margin(b=6)),
        plot.background=element_rect(fill=PAPER,colour=NA),plot.margin=margin(6,8,4,8))

## ---------------- b: the mechanism, as curves ----------------
MK<-c(baseline="rank_baseline",splice="rank_splice",`sequence CNN`="rank_cnn",
      combined="rank_combo")
KS<-1:12
CU<-do.call(rbind,lapply(names(MK),function(m)
  data.frame(model=m,k=KS,f=sapply(KS,function(k) mean(RK[[MK[m]]]<=k)))))%>%
  mutate(model=factor(model,levels=names(MK)))
CO<-c(baseline=NEU,splice=SPL,`sequence CNN`=FAIL,combined=GOOD)
# combined and the CNN sit 1.3 points apart at rank 1 and converge near 98%, so a label at each
# end collides both times. One label per curve on the right, carrying its rank-1 value, and nudged
# apart by hand.
r1<-setNames(CU$f[CU$k==1],as.character(CU$model[CU$k==1]))
# combined and the CNN end within 0.001 of each other, so labels anchored to the curve ends cannot
# be separated by nudging. They go in a fixed, evenly spaced column at the right instead, in curve
# order and colour-matched, which is its own legend.
ends<-CU%>%filter(k==max(KS))%>%
  mutate(txt=sprintf("%s  %.0f%% at rank 1",model,100*r1[as.character(model)]),
         ylab=c(combined=.965,`sequence CNN`=.885,splice=.805,baseline=.725)[as.character(model)])
pB<-ggplot(CU,aes(k,f,colour=model))+
  geom_line(linewidth=.65)+
  geom_point(data=CU%>%filter(k==1),size=2.1)+
  geom_segment(data=ends,aes(x=max(KS),xend=max(KS)+.9,y=f,yend=ylab),
               linewidth=.25,linetype=3,show.legend=FALSE)+
  geom_text(data=ends,aes(x=max(KS)+1.1,y=ylab,label=txt),hjust=0,size=2.5,show.legend=FALSE)+
  scale_colour_manual(values=CO)+
  scale_x_continuous(breaks=c(1,3,6,9,12),limits=c(1,22))+
  scale_y_continuous(limits=c(0,1),breaks=seq(0,1,.25),
                     labels=scales::percent_format(1),expand=c(0,0))+
  labs(title="c   Rank of the annotated 3' end among candidate sites",
       x="rank threshold",y="genes at or above threshold")+
  theme_classic(base_size=BASE)+
  theme(text=element_text(colour=INK),legend.position="none",
        plot.title=element_text(face="bold",size=BASE+0.5,margin=margin(b=6)),
        axis.line=element_line(colour=INK,linewidth=.3),
        axis.ticks=element_line(colour=INK,linewidth=.3),
        axis.title=element_text(colour=GTX,size=BASE-1),
        plot.background=element_rect(fill=PAPER,colour=NA),plot.margin=margin(6,8,6,8))
CAP<-paste0(
 "(a) The baseline scores ten core polyadenylation motifs, the polyadenylation signal in three ",
 "variants, four downstream elements, UGUA, poly-U and NUDT, each scanned in 30 nt windows at ",
 "stride 10 across a 500 nt span. The extended models read 2,500 nt, [-2000, +500) in transcript ",
 "orientation, and add splice context and learned sequence. The magnified row is that 500 nt span; ",
 "element positions within it are schematic, while the spans and the motif list are not.\n",
 "(b) Genes are contested when the baseline ranks at least one competitor at or above the ",
 "annotated 3' end, which it cannot avoid: build_shortlist.py retains a competitor only if the ",
 "random forest, logistic regression and baseline CNN all score it at or above that site, so the ",
 "baseline is never first on a contested gene. The universe is the ",format(N_RES,big.mark=","),
 " genes it resolves plus the ",format(nA,big.mark=",")," contested genes whose candidates both ",
 "extended models score, ",format(N_UNI,big.mark=",")," in total; within it every ribbon closes. ",
 "Both models are leave-one-chromosome-out. Categories give which model places the annotated site ",
 "first. The ",sprintf("%.3f",P_ORAC)," assumes the better model is chosen per gene and is an ",
 "upper bound; a single combined ranking, the sum of the two scores z-scored within each gene, ",
 "gives ",sprintf("%.3f",P_COMB),".\n",

 "Scope. Across all ",format(N_CON,big.mark=",")," contested genes, scored against their full ",
 "candidate lists, the sequence CNN alone recovers ",sprintf("%.3f",N_REC/N_CON)," and lifts ",
 "precision at 1 over all ",format(N_ALL,big.mark=",")," genes from ",sprintf("%.3f",N_OK0/N_ALL),
 " to ",sprintf("%.3f",N_OK1/N_ALL),". The ",format(N_CON-nA,big.mark=",")," contested genes outside ",
 "(b) are neither harder nor unscored, the CNN ranking the annotated site first in 66.8% of them ",
 "against 68.1% elsewhere; they are excluded because every one of their competitors is an ",
 "annotated polyadenylation site or a reference terminal exon, leaving no contest once annotated ",
 "competitors are removed.")
wrapcap<-function(x,w=120) paste(unlist(lapply(strsplit(x,"\n")[[1]],
                   function(p) paste(strwrap(p,width=w),collapse="\n"))),collapse="\n")
library(patchwork)
p<-(pA/pB2)+plot_layout(heights=c(.78,1.5))+
  plot_annotation(caption=wrapcap(CAP),
   theme=theme(plot.caption=element_text(size=BASE-2.2,colour=GTX,hjust=0,lineheight=1.28,
                                         margin=margin(t=8)),
               plot.background=element_rect(fill=PAPER,colour=NA)))
ggsave("figure_rescue.pdf",p,width=7.2,height=6.6,device=cairo_pdf)
ggsave("figure_rescue.png",p,width=7.2,height=6.6,dpi=400,bg=PAPER)
cat(sprintf("[a] %s genes -> %s resolved + %s contested; CNN rescues %s, misses %s\n",
  format(N_ALL,big.mark=","),format(N_RES,big.mark=","),format(N_CON,big.mark=","),
  format(N_REC,big.mark=","),format(N_MISS,big.mark=",")))
cat(sprintf("[a] correct %s -> %s ; P@1 %.4f -> %.4f\n",format(N_OK0,big.mark=","),
  format(N_OK1,big.mark=","),N_OK0/N_ALL,N_OK1/N_ALL))
cat(sprintf("[b] rank<=1 baseline %.4f  splice %.4f  CNN %.4f  combined %.4f  (n=%s)\n",
  mean(RK$rank_baseline<=1),mean(RK$rank_splice<=1),mean(RK$rank_cnn<=1),
  mean(RK$rank_combo<=1),format(nrow(RK),big.mark=",")))
cat(sprintf("[b] rank<=3 baseline %.4f  splice %.4f  CNN %.4f  combined %.4f\n",
  mean(RK$rank_baseline<=3),mean(RK$rank_splice<=3),mean(RK$rank_cnn<=3),
  mean(RK$rank_combo<=3)))
cat(sprintf("[c] both %s (%.1f%%)  CNN only %s (%.1f%%)  splice only %s (%.1f%%)  neither %s (%.1f%%)\n",
  format(sum(sp1&cn1),big.mark=","),100*mean(sp1&cn1),format(sum(!sp1&cn1),big.mark=","),
  100*mean(!sp1&cn1),format(sum(sp1&!cn1),big.mark=","),100*mean(sp1&!cn1),
  format(sum(!sp1&!cn1),big.mark=","),100*mean(!sp1&!cn1)))
cat(sprintf("[c] genome-wide total: combined %.4f, either-model upper bound %.4f\n",TOT_CB,TOT_OR))
cat("[out] figure_rescue.pdf / .png\n")
```

### `make_figure_loco.R`

Panels harvested for Figure 3.9. 236 lines, sha256 `63f042be03c5`.

```r
#!/usr/bin/env Rscript
# make_figure_loco.R -- sequence CNN against the 26-feature splice model, everything
# leave-one-chromosome-out over the whole analysis set.
#
#   A  ROC and precision-recall curves, pooled over all analysis peaks
#   B  within-gene recall@k, the primary comparison
#   C  effect of dinucleotide shuffling on the within-gene win-rate
#   D  genome-wide P@1, baseline against both models, at three scopes
#
# PANEL C ALREADY USED THE 26-FEATURE MODEL. loco_shuffle.tsv is written by loco_figure_data.py,
# which re-derives u1_dens4 and the MaxEnt acceptor on dinucleotide-shuffled sequence for the
# LOCKED 26-feature model under the same folds. The 51-feature shuffle bars are in the OTHER
# figure, figure_cnn_vs_splice, which is the single-split comparison.
#
# Two models only. No GC baseline, no acceptor-alone arm, no single-split scores anywhere.
# The figure computes nothing: loco_figure_data.py wrote every input and this script re-asserts
# the numbers that carry the argument before writing the file.
#
# No prose under any panel subheading, by request: panel letter, title, axis labels. Everything
# that would have been a subtitle is in the caption.
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(tidyr);library(patchwork);library(hexbin);library(scales)})
e<-function(...) cat(...,"\n",file=stderr())
INK<-"#1A1A1A"; BASE<-7
CNNC<-"#08519C"; SPLC<-"#E8A33D"; GREY<-"#8A8F98"; GTX<-"#5A6068"

CV<-read_tsv("figLC_curves.tsv",show_col_types=FALSE)
RC<-read_tsv("figLC_recall.tsv",show_col_types=FALSE)
SH<-read_tsv("loco_shuffle.tsv",show_col_types=FALSE)
WG<-read_tsv("cnn_wholegenome_p1.tsv",show_col_types=FALSE)
WL<-read_tsv("wg_by_length.tsv",show_col_types=FALSE)
PG<-read_tsv("wg_per_gene.tsv",show_col_types=FALSE)
SM<-read_tsv("figLC_summary.tsv",show_col_types=FALSE)
NG<-SM$n_genes[1]; NP<-SM$n_peaks[1]; BR<-SM$base_rate[1]; NGA<-NG
CNNL<-"Sequence CNN"; SPLL<-"Splice model (26 features)"
ANC<-RC$recall[RC$model==CNNL&RC$k==1]; ANS<-RC$recall[RC$model==SPLL&RC$k==1]
col<-function(m) ifelse(m==CNNL,CNNC,SPLC)
stopifnot(nrow(SM)==2, setequal(SM$model,c(CNNL,SPLL)))
# panel C must be the 26-feature model, not the 51-feature one
stopifnot(setequal(SH$model,c(CNNL,SPLL)), all(grepl("26 features",SH$model[SH$model!=CNNL])))

th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-1),axis.line=element_line(color=INK,linewidth=.3),
  axis.title=element_text(size=BASE),axis.ticks=element_line(color=INK,linewidth=.3),
  plot.title=element_text(size=BASE+1,face="bold",hjust=0,margin=margin(b=4)),
  legend.position="none")

g<-function(m,f) SM[[f]][SM$model==m]
au<-function(m,c_) CV$auc[CV$model==m&CV$curve==c_][1]

## ================= A: ROC and PR =================
ROC<-CV%>%filter(curve=="ROC"); PR<-CV%>%filter(curve=="PR")
pA1<-ggplot(ROC,aes(x,y,color=I(col(model))))+
  geom_abline(slope=1,intercept=0,linetype=2,color=GREY,linewidth=.35)+
  geom_line(linewidth=.55)+
  annotate("text",x=.97,y=.20,hjust=1,size=2.15,color=CNNC,
           label=sprintf("Sequence CNN   AUROC %.3f",au(CNNL,"ROC")))+
  annotate("text",x=.97,y=.12,hjust=1,size=2.15,color=SPLC,
           label=sprintf("Splice model, 26 features   AUROC %.3f",au(SPLL,"ROC")))+
  annotate("text",x=.97,y=.04,hjust=1,size=2.05,color=GREY,label="chance")+
  coord_equal(xlim=c(0,1),ylim=c(0,1))+
  scale_x_continuous(breaks=seq(0,1,.25))+scale_y_continuous(breaks=seq(0,1,.25))+
  labs(title="A   Pooled discrimination",x="False positive rate",y="True positive rate")+th
pA2<-ggplot(PR,aes(x,y,color=I(col(model))))+
  geom_hline(yintercept=BR,linetype=2,color=GREY,linewidth=.35)+
  geom_line(linewidth=.55)+
  annotate("text",x=.97,y=.93,hjust=1,size=2.15,color=CNNC,
           label=sprintf("AUPRC %.3f",au(CNNL,"PR")))+
  annotate("text",x=.97,y=.85,hjust=1,size=2.15,color=SPLC,
           label=sprintf("AUPRC %.3f",au(SPLL,"PR")))+
  annotate("text",x=.45,y=BR+.045,hjust=0,size=2.05,color=GREY,
           label=sprintf("base rate %.3f",BR))+
  coord_equal(xlim=c(0,1),ylim=c(0,1))+
  scale_x_continuous(breaks=seq(0,1,.25))+scale_y_continuous(breaks=seq(0,1,.25))+
  labs(title=" ",x="Recall",y="Precision")+th

## ================= B: within-gene recall@k =================
r1c<-RC$recall[RC$model==CNNL&RC$k==1]; r1s<-RC$recall[RC$model==SPLL&RC$k==1]
stopifnot(abs(r1c-g(CNNL,"p_at_1"))<1e-6, abs(r1s-g(SPLL,"p_at_1"))<1e-6)
CHD<-RC%>%filter(model==CNNL)%>%select(k,chance)
rc<-function(k) approx(RC$k[RC$model==CNNL],RC$recall[RC$model==CNNL],k)$y
rs<-function(k) approx(RC$k[RC$model==SPLL],RC$recall[RC$model==SPLL],k)$y
ch<-function(k) approx(CHD$k,CHD$chance,k)$y
KM<-max(RC$k)
pB<-ggplot(RC,aes(k,recall,color=I(col(model))))+
  geom_ribbon(aes(ymin=ci_lo,ymax=ci_hi,fill=I(col(model))),alpha=.14,color=NA)+
  geom_line(data=CHD,aes(k,chance),inherit.aes=FALSE,linetype=2,color=GREY,linewidth=.4)+
  geom_line(linewidth=.55)+geom_point(size=.85)+
  annotate("point",x=1,y=r1c,size=2.0,shape=21,fill="white",color=CNNC,stroke=.55)+
  annotate("point",x=1,y=r1s,size=2.0,shape=21,fill="white",color=SPLC,stroke=.55)+
  annotate("text",x=1.2,y=r1c,label=sprintf("recall@1 = %.3f",r1c),size=2.1,color=CNNC,
           hjust=0,vjust=0.0)+
  annotate("text",x=1.2,y=r1s,label=sprintf("recall@1 = %.3f",r1s),size=2.1,color=SPLC,
           hjust=0,vjust=1.0)+
  # labelled where the curves are apart; near k = KMAX all three converge
  annotate("text",x=KM*0.6,y=rc(KM*0.6)+.038,label="Sequence CNN",size=2.15,color=CNNC)+
  annotate("text",x=KM*0.63,y=rs(KM*0.63)-.055,label="Splice model, 26 features",size=2.15,
           color=SPLC)+
  annotate("text",x=KM*0.85,y=ch(KM*0.85)-.048,label="chance",size=2.05,color=GREY)+
  scale_x_continuous(breaks=1:KM)+scale_y_continuous(limits=c(.25,1.04),breaks=seq(.3,1,.1))+
  labs(title="B   Within-gene ranking",x="Rank threshold k",
       y="Fraction of genes with the real end in the top k")+th

## ================= C: dinucleotide shuffling =================
SH2<-SH%>%select(model,real,real_lo,real_hi,shuffled,shuffled_lo,shuffled_hi,drop,pct_of_lift_lost)
LONG<-bind_rows(
  SH2%>%transmute(model,state="Real sequence",v=real,lo=real_lo,hi=real_hi),
  SH2%>%transmute(model,state="Dinucleotide shuffled",v=shuffled,lo=shuffled_lo,hi=shuffled_hi))%>%
  mutate(model=factor(model,levels=c(CNNL,SPLL)),
         state=factor(state,levels=c("Real sequence","Dinucleotide shuffled")))
ANN<-SH2%>%mutate(model=factor(model,levels=c(CNNL,SPLL)))%>%
  transmute(model,top=real_hi,lab=sprintf("-%.3f  (%.0f%% of the lift)",drop,pct_of_lift_lost))
pC<-ggplot(LONG,aes(model,v,fill=I(ifelse(state=="Real sequence",col(as.character(model)),"#FFFFFF")),
                    color=I(col(as.character(model)))))+
  geom_col(position=position_dodge(width=.66),width=.56,linewidth=.4)+
  geom_errorbar(aes(ymin=lo,ymax=hi),position=position_dodge(width=.66),width=.14,
                linewidth=.3,color=INK)+
  geom_hline(yintercept=.5,linetype=2,color=GREY,linewidth=.35)+
  geom_text(data=ANN,aes(model,top+.052,label=lab),inherit.aes=FALSE,size=2.05,color=INK)+
  annotate("text",x=0.52,y=.503,label="chance",size=2.0,color=GREY,hjust=0,vjust=-0.4)+
  annotate("text",x=1.5,y=.935,label="filled = real sequence,  open = shuffled",size=2.0,
           color=GTX,hjust=0.5)+
  scale_x_discrete(labels=c("Sequence CNN","Splice model,\n26 features"))+
  # coord_cartesian, NOT scale_y_continuous(limits=): a column starts at 0, so a scale limit
  # DROPS the bar rather than zooming to it.
  coord_cartesian(ylim=c(.45,.96))+scale_y_continuous(breaks=seq(.5,.9,.1))+
  labs(title="C   Effect of dinucleotide shuffling",
       x=NULL,y="Within-gene win-rate")+th

## ============ D: genome-wide P@1 over gene length x candidate count ============
# All 15,560 genes, no subset. Difficulty is two-dimensional here -- a gene is hard because it
# is long AND because that length buys candidates -- so the panel is a map over both rather than
# three aggregate numbers or a single marginal curve. Fill is the fraction of genes in each hex
# whose used 3' end is ranked first, on one scale across the three models, so the comparison is
# how far each map stays blue into the top-right corner.
BASEW<-WG$baseline[WG$scope=="genome-wide"][1]
LV3<-c("Baseline",SPLL,CNNL)
AGG<-setNames(WG$p_at_1[WG$scope=="genome-wide"],WG$model[WG$scope=="genome-wide"])
PGL<-PG%>%filter(!is.na(gene_length),gene_length>0)%>%
  select(gene_length,n_candidates,Baseline=baseline,splice=splice,cnn=cnn)%>%
  pivot_longer(c(Baseline,splice,cnn),names_to="m",values_to="hit")%>%
  mutate(model=factor(ifelse(m=="cnn",CNNL,ifelse(m=="splice",SPLL,"Baseline")),levels=LV3),
         kb=gene_length/1000)
NHEX<-PGL%>%filter(model=="Baseline")%>%nrow()
# a hex needs at least MINN genes before its mean is drawn; sparse hexes are left blank
MINN<-12
mean_or_na<-function(z) if (length(z) >= MINN) mean(z) else NA_real_
pD<-ggplot(PGL,aes(kb,n_candidates,z=hit))+
  stat_summary_hex(bins=20,fun=mean_or_na,na.rm=TRUE)+
  facet_wrap(~model,nrow=1)+
  scale_x_log10(breaks=c(1,10,100,1000),labels=c("1","10","100","1,000"))+
  scale_y_log10(breaks=c(1,3,10,30,100,300),labels=c("1","3","10","30","100","300"))+
  scale_fill_gradientn(colours=c("#FBE9D2","#FDF7F0","#DCE6F2","#9FC0E0","#4E86BF",CNNC),
                       limits=c(0,1),breaks=seq(0,1,.25),na.value="transparent",
                       name="used 3' end\nranked first")+
  geom_text(data=data.frame(model=factor(LV3,levels=LV3),
                            lab=sprintf("genome-wide %.3f",AGG[LV3])),
            aes(x=0.62,y=330,label=lab),inherit.aes=FALSE,hjust=0,size=2.15,color=INK)+
  labs(title="D   Genome-wide identification of the used 3' end",
       x="Gene length (kb)",y="Candidate 3' ends in the gene")+
  th+theme(legend.position="right",legend.key.width=unit(6,"pt"),
           legend.key.height=unit(15,"pt"),legend.title=element_text(size=BASE-2.2),
           legend.text=element_text(size=BASE-2.2),
           strip.background=element_blank(),
           strip.text=element_text(size=BASE-0.5,color=INK,face="bold",hjust=0,
                                   margin=margin(b=2)),
           panel.spacing.x=unit(7,"pt"))

## ================= compose =================
CAP<-paste0(
 "Leave-one-chromosome-out over the analysis set: ",format(NG,big.mark=",")," genes and ",
 format(NP,big.mark=",")," peaks (one real 3' end per gene, ",
 format(NP-NG,big.mark=",")," intragenic competitors), 23 folds. Every peak is scored by a fit ",
 "that never saw its chromosome.\n",
 "Both models use the identical fold definition, and inside each fold a 25% gene holdout of ",
 "that fold's training genes: the splice model retunes its ridge penalty on it, the CNN early-",
 "stops on it. The two differ in the model and in nothing else.\n",
 "Panel A is pooled over peaks and is shown for reference; the within-gene comparison in panel ",
 "B is the primary metric, because a competitor is a different site in the same gene and pooled ",
 "metrics reward separating genes rather than sites.\n",
 "Panel B rank counts ties against the model, so recall@1 is the strict P@1; chance is the mean ",
 "over genes of min(k, candidates)/candidates, which is not a straight line because genes differ ",
 "in size.\n",
 "Panel C applies each fitted model, unchanged and not refitted, to sequence shuffled with the ",
 "Altschul-Erikson algorithm across the whole window. That holds mononucleotide and dinucleotide ",
 "content exactly, so the global GC and CpG of each window are unchanged, and destroys both local ",
 "order and where along the window any composition sits.\n",
 "The fraction of each model's lift over 0.5 that SURVIVES is therefore what global composition ",
 "alone can do: ", sprintf("%.0f%%",100-SH$pct_of_lift_lost[SH$model==CNNL]), " for the CNN and ",
 sprintf("%.0f%%",100-SH$pct_of_lift_lost[SH$model==SPLL]), " for the splice model. The fraction ",
 "LOST is the part that depends on order and position: ",
 sprintf("%.0f%%",SH$pct_of_lift_lost[SH$model==CNNL]), " and ",
 sprintf("%.0f%%",SH$pct_of_lift_lost[SH$model==SPLL]), ".\n",
 "Panel D covers all ",format(NHEX,big.mark=",")," genes with a GENCODE span, 85 of the 15,560 ",
 "having none. A gene is uncontested when it carries one candidate, so every ",
 "model ranks it first and ",
 "all ",format(WG$n_genes[1]-WG$n_genes[WG$scope=="contested"][1],big.mark=",")," of them enter ",
 "the denominator as correct for the baseline as well, which is why the genome-wide baseline is ",
 sprintf("%.3f",BASEW),".\nLength is what buys candidates, so that baseline is the bottom row ",
 "of each map, and how far each model stays blue above it is what sequence adds.\n",
 "Every gene appears once in each map and a hex is drawn where at least ",MINN,
 " genes fall in it.\nAll ",format(WG$n_genes[WG$scope=="contested"][1],big.mark=","),
 " contested genes are ranked over their full candidate sets, ",format(70723,big.mark=","),
 " candidates in total, scored out of fold by the same 23-fold scheme. No baseline probability ",
 "enters either model as an input.\n",
 "Bands and bars are 95% bootstrap intervals over genes.")

FIG<-((pA1|pA2|pB)+plot_layout(widths=c(0.78,0.78,1.44)))/
     ((pC|pD)+plot_layout(widths=c(1,2.05)))+
  plot_layout(heights=c(1,0.85))+
  plot_annotation(
    title=paste0("A sequence CNN outperforms splice features at within-gene 3' end ",
                 "discrimination"),
    caption=CAP,
    theme=theme(plot.title=element_text(size=BASE+2,face="bold",color=INK,hjust=0,
                                        margin=margin(b=6)),
                plot.caption=element_text(size=BASE-2.2,color=GTX,hjust=0,lineheight=1.3,
                                          margin=margin(t=8))))
ggsave("figure_loco_cnn_vs_splice.pdf",FIG,width=10.6,height=7.4,device=grDevices::cairo_pdf)
ggsave("figure_loco_cnn_vs_splice.png",FIG,width=10.6,height=7.4,dpi=320)

# structural checks: two models only, and no panel carries a subtitle
stopifnot(is.null(pA1$labels$subtitle),is.null(pA2$labels$subtitle),
          is.null(pB$labels$subtitle),is.null(pC$labels$subtitle))
stopifnot(setequal(unique(CV$model),c(CNNL,SPLL)),setequal(unique(RC$model),c(CNNL,SPLL)),
          setequal(unique(SH$model),c(CNNL,SPLL)))
stopifnot(!any(grepl("GC|acceptor|Acceptor",c(pA1$labels$title,pA2$labels$title,pB$labels$title,
                                              pC$labels$title))))
e("[fig] LOCO within-gene: CNN",sprintf("%.4f",g(CNNL,"within_gene")),
  " splice",sprintf("%.4f",g(SPLL,"within_gene")))
e("[fig] pooled AUROC/AUPRC: CNN",sprintf("%.4f / %.4f",au(CNNL,"ROC"),au(CNNL,"PR")),
  " splice",sprintf("%.4f / %.4f",au(SPLL,"ROC"),au(SPLL,"PR")))
e("[fig] recall@1: CNN",sprintf("%.4f",r1c)," splice",sprintf("%.4f",r1s))
e("[fig] shuffle drops:",paste(sprintf("%s -%.3f (%.0f%%)",SH$model,SH$drop,
                                       SH$pct_of_lift_lost),collapse="  "))
e("[check] two models only; no panel subtitles; no GC or acceptor panel")
e("[out] figure_loco_cnn_vs_splice.pdf figure_loco_cnn_vs_splice.png")
```

### `make_figure_cnn_interpretability.R`

Panels harvested for Figure 3.11. 238 lines, sha256 `cba096e59236`.

```r
#!/usr/bin/env Rscript
# make_figure_cnn_interpretability.R -- what the reported CNN reads, in two panels.
#
#   A  every first-layer filter, its logo, and where along the window it moves the ranking
#   B  what happens when the sequence the model wants, and the one it does not, are implanted
#
# The methods-defence panels (multi-scale occlusion, rank-stratified signal) are supplementary,
# in make_figure_cnn_interp_supp.R.
#
# ROW ORDER. A discrete ggplot axis draws factor level 1 at the BOTTOM; ggseqlogo with ncol=1
# draws its list from the TOP. Handing the same vector to both silently pairs every logo with
# the wrong filter, which is what happened in an earlier figure. Here the heatmap takes
# rev(ORD) and the logo list takes ORD, and the pairing is then CHECKED rather than trusted:
# the per-column argmax consensus is recomputed from each logo matrix as drawn and asserted
# against the consensus recorded for the filter that row claims to be.
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(readr);library(tidyr)
                               library(patchwork);library(ggseqlogo);library(scales)})
e<-function(...) cat(...,"\n",file=stderr())
INK<-"#1A1A1A"; BASE<-7; BLUE<-"#08519C"; ORNG<-"#E8A33D"; GREY<-"#8A8F98"; GTX<-"#5A6068"
FAINT<-"#C6CACF"

PM<-read_tsv("cnnmain_posmap.tsv",show_col_types=FALSE)
LG<-read_tsv("cnnmain_logos.tsv",show_col_types=FALSE)
GA<-read_tsv("gia_dose_matched_all.tsv",show_col_types=FALSE)   # ALL 14,198 held-out windows
GE<-read_tsv("gia_dose_evolved.tsv",show_col_types=FALSE)  # the evolved unit, for one reference number
MPW<-read_tsv("me3_pwm.tsv",show_col_types=FALSE)          # 23-mers near the score3 optimum
NPW<-read_tsv("me3_pwm_noc.tsv",show_col_types=FALSE)      # the same ensemble, no C allowed
UPW<-read_tsv("u1_pwm.tsv",show_col_types=FALSE)           # 9-mers near the U1 duplex optimum
MAPW<-read_tsv("canon_acc_pwm.tsv",show_col_types=FALSE)      # 23-mers near the MODEL's optimum
MAMETA<-read_tsv("canon_acc_pwm_meta.tsv",show_col_types=FALSE)  # how that ensemble was sampled
NT<-read_tsv("cnnmain_notes.tsv",col_types=cols(.default=col_character()))
nv<-function(k) NT$value[NT$key==k][1]
PPT<-nv("ppt_filter"); NF<-as.integer(nv("n_filters"))
NTE<-as.integer(nv("n_test")); NGE<-as.integer(nv("n_genes")); RF<-as.integer(nv("A:receptive_field"))

th<-theme_classic(base_size=BASE)+theme(text=element_text(color=INK),
  axis.text=element_text(color=INK,size=BASE-1),axis.line=element_line(color=INK,linewidth=.3),
  axis.title=element_text(size=BASE),axis.ticks=element_line(color=INK,linewidth=.3),
  plot.title=element_text(size=BASE+1,face="bold",hjust=0,margin=margin(b=4)),
  legend.position="none")

## ================= A =================
ORD<-PM%>%distinct(filter,row_order)%>%arrange(row_order)%>%pull(filter)
stopifnot(length(ORD)==NF, !any(duplicated(ORD)))
CONS<-PM%>%distinct(filter,consensus)%>%tibble::deframe()
mk<-function(f){
  m<-LG%>%filter(filter==f)%>%select(position,base,prob)%>%
     pivot_wider(names_from=position,values_from=prob)%>%as.data.frame()
  rn<-m$base; m<-as.matrix(m[,-1]); rownames(m)<-rn; m[c("A","C","G","T"),,drop=FALSE]}
mats<-lapply(ORD,mk)
# the check: consensus re-derived from the matrix about to be drawn, against the table
derived<-sapply(mats,function(m) paste(rownames(m)[apply(m,2,which.max)],collapse=""))
stopifnot(identical(unname(derived),unname(CONS[ORD])))
# The reference-match column is dropped from the figure. best_match is the nearest entry in a small
# hand-assembled reference list and r is a PWM correlation against it, which is a guess dressed as a
# measurement: a T-rich filter matches an acceptor reference because acceptors are T-rich, not
# because the filter detects acceptors. The identity, information content and site count stay,
# since those are measured. The matches and their caveats are in the thesis section.
LABF<-PM%>%distinct(filter,row_order,ic_per_col,n_sites)%>%arrange(row_order)%>%
  mutate(txt=sprintf("%s   %.2f b/col   %s sites",sub("filter","filter ",filter),
                     ic_per_col,format(n_sites,big.mark=",")))
names(mats)<-LABF$txt
pLog<-ggseqlogo(mats,method="bits",ncol=1)+
  theme_classic(base_size=BASE)+
  theme(text=element_text(color=INK),legend.position="none",strip.background=element_blank(),
        strip.text=element_text(size=BASE-2.4,color=GTX,hjust=0,margin=margin(b=0,t=1.5)),
        axis.text=element_blank(),axis.ticks=element_blank(),axis.line=element_blank(),
        axis.title=element_blank(),plot.margin=margin(l=2,r=1,t=18,b=2))

PM2<-PM%>%mutate(filter=factor(filter,levels=rev(ORD)))
stopifnot(identical(rev(levels(PM2$filter)),ORD))
LIM<-max(abs(PM2$contrib))
pHM<-ggplot(PM2,aes(centre,filter,fill=contrib))+geom_tile()+
  scale_fill_gradient2(low=ORNG,mid="white",high=BLUE,midpoint=0,limits=c(-LIM,LIM),
                       name="contribution")+
  scale_x_continuous(breaks=seq(-2000,500,500),expand=c(0,0))+
  scale_y_discrete(expand=c(0,0))+
  coord_cartesian(xlim=c(-2065,500),clip="off")+
  labs(title="A   First-layer filters and their positional contribution to within-gene ranking",
       x="Position relative to the cleavage site (nt)",y=NULL)+
  th+theme(axis.text.y=element_blank(),axis.ticks.y=element_blank(),axis.line.y=element_blank(),
           legend.position="right",legend.key.width=unit(6,"pt"),
           legend.key.height=unit(20,"pt"),legend.title=element_text(size=BASE-2),
           legend.text=element_text(size=BASE-2),plot.margin=margin(l=1,r=2,t=2,b=2))
pA<-(pLog|pHM)+plot_layout(widths=c(1.05,2.0))

## ================= B =================
# Panel B answers one question in one unit: implant a splice site into a COMPETITOR window and does
# the model start calling it a real 3' end.
#
# WHY PROBABILITY AND NOT THE LOGIT. The logit is unbounded, so the acceptor reaches +6.8 while the
# donor reaches -0.9 and no single axis serves both; dose_saturation.tsv also showed the logit still
# climbing at 24 copies after the prediction it encodes had stopped moving. Probability is bounded,
# puts both on one axis, and stops rewarding a construct for leaving the model's range.
#
# WHY COMPETITOR WINDOWS. Real 3' ends already sit at a median probability of 0.95 and cannot rise,
# so an activating construct has nowhere to go there. Competitors sit at 0.625 and can move both
# ways. It is also the more natural question of the two.
#
# WHY THESE UNITS. acc_best and acc_noC, drawn here before, are optima of MaxEntScan offered to the
# model, and it does not want them: both sit near zero however they are controlled. acc_model comes
# from searching FOR model score under a score3 floor (acceptor_model_optimum.py) and acc_agdep from
# requiring the AG to matter (acceptor_agdep_robust.py). Both are legitimate acceptors, score3 10.04
# and 10.38.
#
# WHY THE SHUFFLE WORKS HERE. For the T-rich unit the shuffle scored HIGHER than the intact
# sequence, so subtracting it destroyed the signal; the effect there was composition. For these two
# the shuffle sits flat on the baseline, which is what a control is supposed to do, and it is drawn
# rather than subtracted so the reader can see it is flat.
PB<-read_tsv("panelB_prob_ci.tsv",show_col_types=FALSE)%>%filter(background=="real_competitor")
PBASE<-PB$base_median_p[1]; PNBG<-PB$n_bg[1]
# THE UNITS. acc_canon is first because it answers the objection to acc_model, which is that its
# tract is 61% pyrimidine with 28% G and so does not look like a 3' splice site. acc_canon was found
# by the same search with the tract forced to at least 80% pyrimidine and NO G at all, and the model
# still wants it. acc_s3max is the same search with score3 forced above 16, near the MaxEnt optimum,
# and it does nothing: the better the acceptor by MaxEnt the less this model likes it.
UL<-c(acc_canon="3' splice site, tract 83% pyrimidine, no G, score3 10.28",
      donor="5' splice site, strongest U1 duplex, score5 10.86")
UC<-c(acc_canon=ORNG,donor=BLUE)
PART<-c(acc_canon="3' splice site",donor="5' splice site")
PB<-PB%>%filter(name%in%names(UL))%>%
   mutate(part=factor(PART[name],levels=c("3' splice site","5' splice site")),
          lab=UL[name],col=UC[name])
# TWO CONTROLS, BOTH DRAWN, because neither is flat everywhere and choosing one would hide that.
# The 20 mononucleotide shuffles hold length and composition exactly and vary only the order. The 9
# derangements substitute every base and hold nothing. For the donor the derangement is much the
# flatter of the two, staying within 0.015 of baseline across all eight doses, while the shuffle
# drifts down to 0.564. For the acceptors it is the other way round. The grey band is the full
# range they span, each widened by its own SD across draws, so it is the envelope of everything a
# control does rather than one line chosen after the fact.
NSHUF<-20; NDER<-9
CTL<-bind_rows(
  PB%>%transmute(part,n=n_copies,lo=shuffle_p-shuffle_sd/sqrt(NSHUF),
                 hi=shuffle_p+shuffle_sd/sqrt(NSHUF)),
  PB%>%transmute(part,n=n_copies,lo=derange_p-derange_sd/sqrt(NDER),
                 hi=derange_p+derange_sd/sqrt(NDER)))%>%
  group_by(part,n)%>%summarise(lo=min(lo),hi=max(hi),.groups="drop")
ENDS<-PB%>%group_by(name)%>%slice_max(n_copies,n=1)%>%ungroup()
YB<-range(c(PB$motif_lo,PB$motif_hi,CTL$lo,CTL$hi,PBASE)); YB<-YB+c(-1,1)*.09*diff(YB)
SEQL<-c(acc_canon="TTTAACTTTCTTTTTTACAGACG",donor="CAGGTGAGT")
SHORT<-c(acc_canon="83% pyrimidine, no G, score3 10.28",
         donor="strongest U1 duplex, score5 10.86")
TAGB<-PB%>%filter(n_copies==1)%>%
  left_join(ENDS%>%select(name,nend=n_copies,pend=motif_p),by="name")%>%
  mutate(y=motif_p,
         vj=c(acc_canon=1.25,donor=1.25)[name],
         txt=sprintf("%s\n%s\n%.3f [%.3f, %.3f] at 1 copy to %.3f at %d copies",
                     SEQL[name],SHORT[name],motif_p,motif_lo,motif_hi,pend,nend))
pB<-ggplot(PB,aes(n_copies,motif_p,group=name))+
  geom_ribbon(data=CTL,aes(n,ymin=lo,ymax=hi),fill=GREY,alpha=.30,inherit.aes=FALSE)+
  geom_hline(yintercept=PBASE,linetype=2,color=INK,linewidth=.35)+
  geom_text(data=PB%>%group_by(part)%>%summarise(x=max(n_copies),.groups="drop"),
            aes(x,PBASE,label=sprintf("untouched competitor, %.2f",PBASE)),
            hjust=1,vjust=-0.6,size=2.2,color=INK,inherit.aes=FALSE)+
  geom_errorbar(aes(ymin=motif_lo,ymax=motif_hi,color=I(col)),width=.10,linewidth=.45)+
  geom_line(aes(color=I(col)),linewidth=.62)+geom_point(aes(color=I(col)),size=1.15)+
  geom_text(data=TAGB,aes(1.06,y,label=txt,color=I(col),vjust=vj),hjust=0,size=1.95,
            lineheight=1.08,inherit.aes=FALSE)+
  facet_wrap(~part,nrow=1,scales="free_x")+
  scale_x_continuous(breaks=c(1,2,3,4,6,8))+
  scale_y_continuous(limits=YB,breaks=seq(.3,1,.1))+
  labs(title="B   Implanted into a competitor, does the model call it a real 3' end",
       x="Tandem copies implanted",
       y="Median predicted probability")+
  th+theme(strip.background=element_blank(),
           strip.text.x=element_text(size=BASE-1,color=GTX,hjust=0,margin=margin(b=2)),
           panel.spacing.x=unit(11,"pt"),legend.position="none")

# The logo beside each part is a real position weight matrix and both come from ONE recipe: sample
# sequences whose own measure lies within a threshold of that measure's optimum, weight each by how
# close it is. The measures differ, which is the comparison. For the donor it is the U1 duplex over
# all 262,144 9-mers, exhaustive. For the acceptor it is THE MODEL, sampled by Metropolis-Hastings
# over 23-mers with the AG pinned and score3 held at or above 10, so every sequence in the ensemble
# is still a splice site. Neither AG nor GT was pinned in the donor matrix and both appear anyway.
lmat<-function(d){m<-t(as.matrix(d[,c("A","C","G","T")])); rownames(m)<-c("A","C","G","T"); m}
MAM<-lmat(MAPW%>%arrange(offset)); U9<-lmat(UPW%>%arrange(offset))
stopifnot(ncol(MAM)==23,ncol(U9)==9)
cons<-function(m) paste(rownames(m)[apply(m,2,which.max)],collapse="")
CM<-cons(MAM); CU<-cons(U9)
stopifnot(substr(CM,19,20)=="AG",substr(CU,4,5)=="GT")
# each strip states the ensemble and the constraint it was sampled under, so the logo cannot be
# mistaken for a consensus of one sequence or for a database motif
# identity only; the sampling recipe and constraints are in the thesis section
LGN<-c(sprintf("3' splice site ensemble, n = %s",format(MAMETA$n_sequences[1],big.mark=",")),
       "5' splice site ensemble, n = 115")
LMATS<-setNames(list(MAM,U9),LGN)
stopifnot(identical(levels(PB$part),c("3' splice site","5' splice site")))
pBlogo<-ggseqlogo(LMATS,method="bits",ncol=1)+
  scale_y_continuous(limits=c(0,2.05),breaks=c(0,1,2),expand=c(0,0))+
  theme_classic(base_size=BASE)+
  theme(text=element_text(color=INK),legend.position="none",strip.background=element_blank(),
        strip.text=element_text(size=BASE-2.8,color=GTX,hjust=0,lineheight=1.05,
                                margin=margin(b=1,t=1)),
        axis.text.x=element_blank(),axis.ticks.x=element_blank(),axis.line.x=element_blank(),
        axis.text.y=element_text(size=BASE-2.4,color=GTX),axis.title.x=element_blank(),
        axis.title.y=element_text(size=BASE-2,color=GTX),
        plot.margin=margin(l=2,r=6,t=26,b=40))+
  labs(y="bits")
pB<-(pBlogo|pB)+plot_layout(widths=c(1,3.5))

gp<-function(nm,k,col) PB[[col]][PB$name==nm&PB$n_copies==k]
# the one number this panel borrows from the evolved unit, for scale
EVV<-GE%>%filter(arm=="motif",site=="evolved_23",n_copies==1,background=="real_canonical")%>%
     distinct(best_delta)%>%pull(best_delta)
# caption numbers read off the frame that is drawn, so they cannot drift from the lines
NA_<-max(PB$n_copies[PB$name=="acc_canon"]); ND_<-max(PB$n_copies[PB$name=="donor"])
# how far the drawn controls stray from the untouched baseline, and which control is flatter where
shufmax<-max(abs(PB$shuffle_p-PBASE)); dermax<-max(abs(PB$derange_p-PBASE))
dnr<-PB%>%filter(name=="donor")
donshuf<-max(abs(dnr$shuffle_p-PBASE)); donder<-max(abs(dnr$derange_p-PBASE))
# the T-rich acceptor, not drawn, whose shuffle overtakes the intact sequence
tsh<-read_tsv("panelB_prob_ci.tsv",show_col_types=FALSE)%>%
     filter(background=="real_competitor",name=="acc_noC")
tshmax<-max(tsh$shuffle_p-tsh$base_median_p)

## ================= caption =================
# There is no caption. Every method, number and caveat this block used to carry is in
# THESIS_cnn_splice_implantation.txt, which is the document the figure is read alongside.
CAP<-NULL

FIG<-pA/pB+plot_layout(heights=c(4.3,1.35))+
  plot_annotation(theme=theme(plot.margin=margin(4,6,4,6)))
ggsave("figure_cnn_interpretability.pdf",FIG,width=9.8,height=13.6,device=grDevices::cairo_pdf,
       limitsize=FALSE)
ggsave("figure_cnn_interpretability.png",FIG,width=9.8,height=13.6,dpi=220,limitsize=FALSE)
stopifnot(is.null(pHM$labels$subtitle),is.null(pB$labels$subtitle))
e("[A] logo/tile order ASSERTED: consensus re-derived from every drawn matrix matches the table")
e("[A] rows ordered by summed |contribution|; top:",paste(sub("filter","",ORD[1:5]),collapse=","),
  "| PPT filter",PPT,"at row",PM$row_order[PM$filter==PPT][1])
e("[B] competitor background, median predicted probability, baseline",sprintf("%.4f",PBASE))
for (st in names(UL))
  e("   ",sprintf("%-46s %s", UL[st],
    paste(sprintf("n%d %.3f (shuf %.3f)",PB$n_copies[PB$name==st],PB$motif_p[PB$name==st],
                  PB$shuffle_p[PB$name==st]),collapse="  ")))
e("[B] shuffle controls stay within",sprintf("%.4f",shufmax),
  "of baseline; the T-rich acceptor's reaches",sprintf("%+.4f",tshmax),"for contrast")
e("[B] logo consensus: acceptor",CM," donor",CU)
e("[out] figure_cnn_interpretability.pdf figure_cnn_interpretability.png")
```

### `nature_figs/make_nature_figs.R`

Re-renders all 19 coded figures in Nature style; the images in this document. 198 lines, sha256 `00d52ff0be0e`.

```r
#!/usr/bin/env Rscript
# make_nature_figs.R -- re-render every coded thesis figure (Ch2 2.1, 2.3, 2.4, 2.5, 2.9; Ch3 3.1-3.14)
# in a Nature-style presentation. Content is untouched by construction: each original figure script
# runs unchanged in a sandbox, the plot object it hands to ggsave is captured, and only presentation
# is changed before it is written here:
#   - panel titles reduced to a bold lowercase letter (a, b, c); descriptive titles dropped
#   - subtitles, captions and figure-level title/subtitle/caption removed
#   - 7 pt sans text (6 pt tick labels), black, 0.25 mm axis lines
#   - 183 mm (double-column) width, height scaled with the original aspect, capped at 247 mm
# The original scripts, their inputs and their outputs are not modified: ggsave and write.table are
# no-ops inside the sandbox. Run from pub/:
#   Rscript nature_figs/make_nature_figs.R [name ...]      (no names = all)
suppressPackageStartupMessages({library(ggplot2); library(patchwork)})
PUB <- normalizePath(".")
OUT <- file.path(PUB, "nature_figs")
FAM <- "Helvetica"
W_MM <- 183; H_MAX <- 247

# original output name -> thesis figure number
JOBS <- list(
  list(script = "sfig1.R", args = "constitutive_cpa.tsv", wd = PUB, map = c(supp_fig1 = "fig2_1")),
  list(script = "scripts/make_figure1.R", args = ".", wd = PUB, map = c(figure1 = "fig2_3")),
  list(script = "make_analysis_set_funnel.R", wd = PUB, map = c(figure_analysis_set_funnel = "fig2_4")),
  list(script = "scripts/make_figure2.R", args = ".", wd = PUB, map = c(figure2 = "fig2_5")),
  list(script = "make_figure_overlap_shared.R", wd = PUB, map = c(figure_overlap_shared = "fig2_9")),
  list(script = "thesis_ch3/R/fig3_01_hypothesis.R", wd = file.path(PUB, "thesis_ch3"), map = c(fig3_01_hypothesis = "fig3_1")),
  list(script = "thesis_ch3/R/fig3_02_genesets.R", wd = file.path(PUB, "thesis_ch3"), map = c(fig3_02_genesets = "fig3_2")),
  list(script = "thesis_ch3/R/fig3_regroup.R", wd = file.path(PUB, "thesis_ch3"),
       map = c(fig3_03_splice_environment = "fig3_3", fig3_04_donor_scores = "fig3_4",
               fig3_05_splice_model = "fig3_6", fig3_06_splice_model_performance = "fig3_7",
               fig3_07_architecture = "fig3_8", fig3_08_cnn_vs_splice = "fig3_9",
               fig3_12_interpretation = "fig3_11")),
  list(script = "thesis_ch3/R/fig3_hypothesis_dose.R", wd = file.path(PUB, "thesis_ch3"), map = c(fig3_hypothesis_dose = "fig3_5")),
  list(script = "thesis_ch3/R/fig3_09_robustness.R", wd = file.path(PUB, "thesis_ch3"), map = c(fig3_09_robustness = "fig3_10")),
  list(script = "thesis_ch3/R/fig3_13_mechanism.R", wd = file.path(PUB, "thesis_ch3"), map = c(fig3_13_mechanism = "fig3_12")),
  list(script = "thesis_ch3/R/fig3_implant_panel.R", wd = file.path(PUB, "thesis_ch3"), map = c(fig3_implant_panel = "fig3_13")),
  list(script = "thesis_ch3/R/fig3_14_recovery.R", wd = file.path(PUB, "thesis_ch3"), map = c(fig3_14_recovery = "fig3_14"))
)

# ------------------------------------------------------------------ capture
capture <- function(job) {
  got <- list()
  env <- new.env(parent = globalenv())
  env$ggsave <- function(filename, plot = last_plot(), width = NA, height = NA, units = "in", ...) {
    key <- tools::file_path_sans_ext(basename(filename))
    mm <- switch(units, `in` = 25.4, cm = 10, mm = 1, px = NA)
    got[[key]] <<- list(plot = plot, w = width * mm, h = height * mm)
    invisible(NULL)
  }
  env$write.table <- function(...) invisible(NULL)
  env$write.csv <- function(...) invisible(NULL)
  env$commandArgs <- function(trailingOnly = FALSE) if (trailingOnly) job$args %||% character() else c("R", job$args)
  old <- setwd(job$wd %||% PUB); on.exit(setwd(old))
  suppressMessages(suppressWarnings(sys.source(file.path(PUB, job$script), envir = env)))
  got
}

# ------------------------------------------------------------------ restyle
blank <- function(e) is.null(e) || inherits(e, "element_blank")
letter_of <- function(t) {
  if (is.null(t) || !is.character(t) || !nzchar(t)) return(NULL)
  m <- regmatches(t, regexec("^\\s*([A-Za-z])(\\s{2,}|\\s*$|[.)]\\s)", t))[[1]]
  if (length(m)) tolower(m[2]) else NULL
}

style_plot <- function(p, s) {
  full <- tryCatch(ggplot2:::plot_theme(p), error = function(e) theme_classic())
  lab <- letter_of(p$labels$title)
  p <- p + labs(title = lab, subtitle = NULL, caption = NULL)
  axes  <- !blank(full$axis.line) || !blank(full$axis.line.x) || !blank(full$axis.line.y)
  ticks <- !blank(full$axis.ticks)
  # data panels drawn with theme_minimal get real axes; heatmaps and schematics do not
  tiles <- any(vapply(p$layers, function(l) inherits(l$geom, c("GeomTile", "GeomRaster")), TRUE))
  data_axes <- !blank(full$axis.text) && !blank(full$axis.text.x) && !blank(full$axis.text.y)
  if (data_axes && !tiles) { axes <- TRUE; ticks <- TRUE }
  txt <- function(size, ...) element_text(family = FAM, size = size, colour = "black", ...)
  th <- theme(text = element_text(family = FAM, size = 7, colour = "black"),
              strip.background = element_blank(), strip.clip = "off",
              panel.grid = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
              panel.grid.major.x = element_blank(), panel.grid.major.y = element_blank(),
              panel.grid.minor.x = element_blank(), panel.grid.minor.y = element_blank(),
              panel.border = element_blank(),
              plot.title = element_text(family = FAM, size = 8, face = "bold", hjust = 0,
                                        margin = margin(b = 2)),
              plot.title.position = "plot",
              plot.subtitle = element_blank(), plot.caption = element_blank(),
              plot.tag = element_text(family = FAM, size = 8, face = "bold"),
              legend.key.size = unit(3, "mm"), legend.key.spacing = unit(1, "mm"),
              legend.margin = margin(0, 0, 0, 0), legend.box.spacing = unit(1.5, "mm"),
              plot.margin = margin(3, 4, 3, 3))
  # restyle an element only where the original draws it, so nothing hidden comes back;
  # theme addition merges, so angle and justification set by the original survive
  keep <- function(els, size, ...) for (el in els) if (!blank(full[[el]]))
    th <<- th + do.call(theme, setNames(list(txt(size, inherit.blank = isTRUE(full[[el]]$inherit.blank), ...)), el))
  keep(c("axis.title", "axis.title.x", "axis.title.y"), 7)
  keep(c("strip.text", "strip.text.x", "strip.text.y"), 7, face = "plain")
  keep(c("axis.text", "axis.text.x", "axis.text.y"), 6)
  keep("legend.title", 6.5); keep("legend.text", 6, margin = margin(l = 1.2, r = 2.5))
  if (axes) th <- th + theme(axis.line = element_line(linewidth = 0.25, colour = "black"))
  if (ticks) th <- th + theme(axis.ticks = element_line(linewidth = 0.25, colour = "black"),
                              axis.ticks.length = unit(1, "mm"))
  p <- p + th
  # geoms: text clamped to 5-7 pt; strokes and points thinned with the canvas, never below hairline
  g <- 0.6 * s
  for (l in p$layers) {
    ap <- l$aes_params
    if (inherits(l$geom, c("GeomText", "GeomLabel"))) {
      sz <- ap$size %||% 3.88
      l$aes_params$size <- min(max(sz * s, 5 / .pt), 7 / .pt)
      l$aes_params$family <- FAM
    } else {
      if (!is.null(ap$linewidth)) l$aes_params$linewidth <- max(ap$linewidth * g, 0.25)
      if (!is.null(ap$size) && is.numeric(ap$size)) l$aes_params$size <- max(ap$size * g, 0.5)
    }
  }
  p
}

# every ggplot leaf of a (nested) patchwork, in reading order; the patchwork itself is its last plot
map_leaves <- function(p, f) {
  if (inherits(p, "patchwork")) p$patches$plots <- lapply(p$patches$plots, map_leaves, f = f)
  if (inherits(p, "ggplot")) p <- f(p)
  p
}

restyle <- function(p, s) {
  if (inherits(p, "patchwork")) {
    p$patches$plots <- lapply(p$patches$plots, restyle, s = s)
    a <- p$patches$annotation
    a$title <- NULL; a$subtitle <- NULL; a$caption <- NULL
    if (identical(a$tag_levels, "A") || identical(a$tag_levels, list("A"))) a$tag_levels <- "a"
    p$patches$annotation <- a
  }
  if (inherits(p, "ggplot")) p <- style_plot(p, s)
  p
}

# a figure whose panels carried descriptive titles but no letters gets a, b, c in reading order
auto_letter <- function(p) {
  n <- 0; has <- FALSE
  map_leaves(p, function(q) { n <<- n + 1; if (!is.null(q$labels$title)) has <<- TRUE; q })
  if (has || n < 2) return(p)
  i <- 0
  map_leaves(p, function(q) { i <<- i + 1; q + labs(title = letters[i]) })
}

# ------------------------------------------------------------------ per-figure presentation fixes
lab_of <- function(l) paste(c(l$aes_params$label,
  if (is.data.frame(l$data) && "label" %in% names(l$data)) as.character(l$data$label),
  if (!is.null(l$mapping$label)) rlang::quo_text(l$mapping$label)), collapse = " | ")
drop_text <- function(q, re) {
  q$layers <- Filter(function(l) !(inherits(l$geom, c("GeomText", "GeomLabel")) && grepl(re, lab_of(l))), q$layers); q }
FIX <- list(
  # explanatory footnote under the funnel
  fig2_1 = function(q) drop_text(q, "retained"),
  # facet strips carry n in parentheses (it belongs in the legend); fold labels run past the panel
  fig2_9 = function(q) {
    if (inherits(q$facet, "FacetWrap") && "src" %in% names(q$facet$params$facets))
      q$facet$params$labeller <- as_labeller(function(x) vapply(sub("\\s*\\([^)]*sites?\\)$", "", x),
                                             function(y) paste(strwrap(y, 14), collapse = "\n"), ""))
    if (any(vapply(q$layers, function(l) grepl("fold", lab_of(l)), TRUE))) q$coordinates$clip <- "off"
    q },
  # the two score names are bottom strips that run into each other
  fig3_4 = function(q) {
    if (inherits(q$facet, "FacetWrap") && "score" %in% names(q$facet$params$facets))
      q$facet$params$labeller <- label_wrap_gen(32)
    q },
  fig3_12 = function(q) {
    if (is.character(q$labels$x) && grepl("annotated site", q$labels$x))
      q <- q + theme(axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1))
    q }
)

write_fig <- function(obj, name) {
  w <- W_MM
  s <- min(1, W_MM / obj$w)
  h <- min(obj$h * W_MM / obj$w, H_MAX)
  p <- obj$plot
  if (!is.null(FIX[[name]])) p <- map_leaves(p, FIX[[name]])
  fig <- auto_letter(restyle(p, s)) & theme(plot.subtitle = element_blank(), plot.caption = element_blank())
  ggsave(file.path(OUT, paste0(name, ".pdf")), fig, width = w, height = h, units = "mm",
         device = cairo_pdf, limitsize = FALSE)
  ggsave(file.path(OUT, paste0(name, ".png")), fig, width = w, height = h, units = "mm",
         dpi = 450, bg = "white", limitsize = FALSE)
  cat(sprintf("[out] %-8s %3.0f x %3.0f mm  (was %3.0f x %3.0f mm, stroke scale %.2f)\n",
              name, w, h, obj$w, obj$h, s))
}

want <- commandArgs(trailingOnly = TRUE)
for (job in JOBS) {
  if (length(want) && !any(job$map %in% want)) next
  got <- capture(job)
  for (k in names(job$map)) {
    if (length(want) && !(job$map[[k]] %in% want)) next
    if (is.null(got[[k]])) { cat("[miss]", job$script, k, "\n"); next }
    write_fig(got[[k]], job$map[[k]])
  }
}
```
<!-- END AUTO:appendix:code -->

## Cross-cutting notes

### Two splice-model LOCO fits exist. Know which one a number came from.
- `recall_at_k.py` and `loco_figure_data.py` retune lambda per fold and give **P@1 0.465** (Figures 3.7 and 3.9).
- `splice26_loco.py` uses a fixed lambda = 0.1 and gives **0.463** (Figure 3.14 and the combined ranking).
- **OPEN:** Methods 3.4.5 implies both retune. Section 3.2.5 mixes the two: it quotes 46.5% (first fit) alongside a 21.8-point difference and the 1,515 / 457 split (second fit). With the first fit the difference is 21.6 points.

### Two network LOCO runs exist
`cnn_loco_scores.npz` feeds Figures 3.9 and 3.10c; `cnn_wg_scores.npz` feeds Figure 3.14. Both give analysis-set P@1 0.681. No LOCO checkpoints were kept, so archive the score files.

### Things already settled; do not redo
- **Decision-rule 2x2** (`rule_2x2_final.py`): only 1.08-fold over base rate. `acc >= 8.0` admits 91% of peaks, so the rule fires on 74.7% of everything. Binarising destroys the gradient.
- **Joint score** (`joint_score_final.py`): `z(acc) - z(donor)` costs SpliceAI 0.057 AUROC, because its donor is positively predictive marginally. A linear difference cannot express an acceptor-conditional sign flip.
- **Calibrated (composition-corrected) splice features** lowered performance; thresholds are placed on the Figure 3.4c curve instead.
- **Repeat and exon-context analyses** were cut from the chapter by decision, not by failure. Scripts and data are kept in `T/py` and `T/data`.
- **All U1-inhibition material** was removed from Chapter 3: the competitor peak after U1 AMO is under 1% of the real-end peak. It survives as one cautious "preliminary analysis" sentence in Chapter 4 section 4.2.4. The downloaded GEO data remain in `PUB/external/geo/`.
- **XLIP site-level occupancy** is composition-confounded (the hnRNPC positive control fails, rho = -0.16 with U content). Do not use.

### Three statements that sound alike and are not
This distinction caused three rounds of wrong figure titles in the related SpliceAI analysis, and the same care applies here. A site with a strong upstream acceptor and a strong upstream donor falls **to** the base rate; it does not cross it. Level, ranking, and regression coefficient are three different claims and can all hold at once. The settled wording is "competes it back toward baseline", never "becomes" or "reverses to" a competitor.

### The confound to state openly
Competitors are all intronic while real ends all terminate an exon: the upstream 500 nt is 89% exonic for real ends against 1.5% for competitors, and annotated exonic fraction alone gives P@1 0.945 (a circular reference, not a predictor). An examiner will raise this. It is stated in both the Chapter 3 Discussion and Chapter 4. The sequence-level result that survives it is Figure 3.5, which held against matched intronic decoys, and the context-matched comparison (both sites >= 90% exonic and <= 10% repeat) where the network still wins 0.625 of pairs.

### Known inconsistent counts to settle before submission
- Constitutive sites: 15,700 genes (Fig 2.1 funnel) against 15,794 / 15,785 sites in the BED against 15,568 / 15,559 genes scanned against 15,560 in Chapter 3.
- Competitors per gene: median 3, range 1-897 (Ch3 and the corrected Ch2 text) against median 4, mean 10 in the original Ch2 draft.
- Figure 2.5 d/e use 5,898 + 38,186 sites, not the analysis set.
- Figure 2.9 uses all 64,140 competitors, not the analysis set.
- PolyA_DB v3 against v4.
- Figure 2.3a hard-coded metrics against `metadata.json`.
- Singh et al. 2018 is cited in the Ch2 draft but missing from the reference list; Feng 2024/2025 and Soles 2026 in `T/chapter3/ch3_refs.md` are placeholders with incomplete authors and titles.

### Scripts run by hand, with no sbatch and no log
`stats_ch3.py`, `maxent_calibration.py`, `u1_calibration.py`, `u1_null_analysis.py`, `acceptor_position_scan.py`, `hypothesis_grid.py`, `hypothesis_dose.py`, `pas_controlled.py`, `splice_model_26.py`, `splice_model_expand.py`, `hiconf_sets.py`, `overlap_shared_recompute.py`. Also unrecorded: the `make_figure1_data.py` invocation, the pandoc command that built the revised chapter docx files, and the `figure3_data.py` invocation.

### Stale documentation
`T/README.md` (figure map and numbering), the header of `PUB/make_figure_loco.R` (panel d), `PUB/THESIS_figure_splicing_environment_legend.txt` (describes an older 4-panel figure), and `PUB/rerank_corr_peaks_README.SUPERSEDED.md`.

### Chapter 3 legend and text fixes the audit found

None of these is a numeric error in a result. Every headline number in Chapter 3 that could
be traced to a plotted table reproduced exactly. These are wording, sign and count fixes to
the legends and the running text, listed in the order worth doing them.

**Substance**
1. Fig 3.11 legend defines the filter contribution as the "change in logit on silencing"; the
   plotted quantity is the **drop**. Methods 3.4.8 is already right, so copy its wording.
2. Section 3.2.7's "mutating the AG to AC cost only 0.034" has the sign backwards: the mutant
   is 0.034 higher. The fix strengthens the argument.
3. Fig 3.11 legend calls the grey band a "range"; it is two control means with their standard
   errors.

**Counts**
4. The 50 nt acceptor replacement is 1,003 windows, not 1,010 (legend and section 3.2.8).
5. Fig 3.9c covers 15,475 genes, not 15,560.
6. Section 3.2.9's "214 proteins" should be 212, and "by at most 0.15" should be 0.16 (DAZ4).

**Presentation**
7. "the motifs that lowered it most were those of the hnRNP A/B family" contradicts the figure,
   where DAZ4 is the most negative and is labelled in grey. Rewrite around UAG content.
8. MSI1 and DAZAP1 are coloured as hnRNP A/B by a regex in `fig3_implant_panel.R`.
9. Acceptor and donor colours are flipped between Fig 3.8a and Figs 3.11/3.12.
10. Fig 3.11 panel a's logos use a 4,000-window subsample while its heatmap uses all 14,198.
11. Fig 3.11's cross-reference to Section 3.4.8 is dangling, and the two logos come from
    different recipes (one exhaustive, one sampled against the network).
12. Smaller items: Fig 3.8a says ten elements and shows four; Fig 3.9b's open circles at k = 1
    are unexplained; Fig 3.9c's colourbar says "used" where the title says "annotated";
    Fig 3.14a's "Combined" appears on one facet only; NUDT21's consensus is written as UGUA in
    the text and AAUGUAA in the figure; the text and figures use different protein names
    (U1A/SNRPA, CFIm25/NUDT21, and so on); and one object is called the used, annotated, real
    and true 3′ end across seven figures.

### Nothing is in git
`git ls-files reanal/` returns nothing. This should be fixed before publication: commit the scripts, the small tables and this README, and gitignore `spliceai_bw/`, `PUB/models/`, `PUB/scans/`, `fig1_scoredist.tsv` and `PUB/external/geo/`.

## Archive manifest: what to put in the zip

The committee asked for the overlap datasets plus a README. A reasonable archive is under about 30 MB if the following rule is followed.

**Include**
- This README and `reporter_constructs.tsv` / `.fa`.
- `PUB/hiconf_out/sites/{atlas_3seq,oh_u1amo,pcf11_dep,singh_ipa,lee_ipa}.tsv` - the processed site files actually used for the overlaps.
- `PUB/hiconf.yaml`, `datasets.yaml`, `PUB/external/manifest.tsv`, `verify_report.tsv`.
- The downloaded source files: `PUB/external/geo/GSE111310/*.txt.gz`, `GSE111793/*.txt.gz`, `GSE103252/*.bw` (2 files), `GSE123105/3pmrnaseq/GSM3536480_*` and `GSM3536484_*` (4 bigWigs).
- `PUB/external/chain/hg19ToHg38.over.chain.gz`.
- The overlap scripts and outputs: `hiconf_sets.py`, `overlap_model_scores.py`, `overlap_rf_dense.py`, `overlap_shared_recompute.py`, `make_figure_overlap_shared.R`, `make_figure_overlap.R`, `task_overlap_accounting.py`, `scripts/03_simple_overlap.py`, plus `shared_distances.tsv`, `shared_containment.tsv`, `shared_hexamer.tsv`, `overlap_shared_genes.tsv`, `overlap_scores_summary.tsv`, `overlap_rf_dense.tsv`.
- `PUB/cpa_peaks_final.parquet` (1.9 MB) and its README, plus `analysis_set_funnel.tsv`. These define the competitor and constitutive sites, so include them even though they are "ours".
- Fetch scripts `scripts/00_download_and_liftover.sh`, `00b_verify_and_repair.sh`, `fetch_gsm.sh`.

**Exclude** (too large, and all re-derivable or re-downloadable)
- `PUB/models/` (each RF joblib is about 15.7 GB), `PUB/scans/`, `fig1_scoredist.tsv` (41 MB).
- `hg38.fa` (3.3 GB), `spliceai_bw/` (3.4 GB).
- `GSE123105_RAW.tar` (4.8 GB), `GSE135140` (2.3 GB), GSE250342 / GSE262383 / GSE275154 (about 214 MB together, and unused in any figure).
- `PUB/external/bin/liftOver` (24.8 MB binary; name the version instead).

**Mention in the README but note as not held on the cluster:** plasmid maps, backbone sequence, Sanger traces, raw gel images, primer order sheets, and the insert-selection script.

## One-line rebuild summary

```bash
# Chapter 2
cd $PUB
Rscript sfig1.R constitutive_cpa.tsv                 # Fig 2.1
Rscript scripts/make_figure1.R .                     # Fig 2.3  (fix panel a first)
Rscript make_analysis_set_funnel.R                   # Fig 2.4
Rscript scripts/make_figure2.R                       # Fig 2.5
Rscript make_figure_overlap_shared.R                 # Fig 2.9
# Figs 2.2, 2.6, 2.7, 2.8 are drawings, gels and traces: no code.

# Chapter 3  (R = the full /cvmfs path; run from $T)
cd $T
$R R/fig3_01_hypothesis.R        # 3.1
$R R/fig3_02_genesets.R          # 3.2
$R R/fig3_regroup.R              # 3.3, 3.4, 3.6, 3.7, 3.8, 3.9, 3.11
$R R/fig3_hypothesis_dose.R      # 3.5
$R R/fig3_09_robustness.R        # 3.10
$R R/fig3_13_mechanism.R         # 3.12
$R R/fig3_implant_panel.R        # 3.13
$R R/fig3_14_recovery.R          # 3.14

# Nature-style renders of all 19 (the images in this document); runs the scripts above
cd $PUB
$R nature_figs/make_nature_figs.R               # all, into nature_figs/
$R nature_figs/make_nature_figs.R fig2_3 fig3_9 # or just some
```

Figures rebuild in seconds from the cached tables. Only the GPU analyses (`context_sweep`, `splice_scramble`, `patching`, `implant_position`, `mechanism`, `implant_cisbp`, `interpret_deep`, `concept_erasure`, and any LOCO retraining) need a job submission; use an sbatch on a MIG slice, because the login node is capped at 6 CPUs and torch deadlocks on fork.
