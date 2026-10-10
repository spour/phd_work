# Figure 1. Sequence models of cleavage and polyadenylation site usage: held-out performance, between-model agreement, and behaviour at annotated sites.

Three models were trained to discriminate constitutive cleavage and polyadenylation
(CPA) windows from background genomic windows and then applied, at single-nucleotide
resolution, to whole gene bodies. **(A)** Held-out test performance. AUROC, AUPRC and
F1 for the random forest (RF), L1-penalised logistic regression (LR) and convolutional
neural network (CNN) on the 104,031 windows of the production test split (2,935
positive, 101,096 negative; chromosomes 18-22 and X, never seen in training or
validation). AUROC and AUPRC are threshold-free; F1 is evaluated at each model's
own operating point, chosen by maximising validation F1 over a grid of 91 thresholds
(RF 0.29, LR 0.93, CNN 0.37). Bars are single values from one locked evaluation, not
means over replicates, so no error bars are shown. **(B)** Per-window score agreement.
Pearson correlation between each pair of models, accumulated in a single streaming
pass over every scored window of all 15,568 genes. Pairwise r is 0.38-0.55, so the
uniformly high AUROC in (A) does not imply that the models agree window by window.
**(C)** Top-K canonical recovery. Fraction of 15,559 genes whose annotated
constitutive cleavage site lies within 250 nt of at least one of the model's K
highest-scoring windows, for K = 1-10. Scores are taken from the five-fold cross-fit
scan, so no gene is ever scored by a model that saw its chromosome during training
(the 9 chrY genes have no cross-fit model and are excluded). "Combined" takes the
best rank across the three models per gene. K = 1 is argmax recovery.
**(D)** Score traces along a single gene. Every window score for ERP27
(chr12:14,913,288-14,938,537, minus strand; 24,750 windows), plotted against window
offset from the gene's 5' end in the direction of transcription; the annotated 3' end
lies at offset 24,252. Facets have independent y axes. No threshold line is drawn:
peaks are called per gene from that gene's own 3'-end score (see below), so a single
horizontal cut would not correspond to the calling rule. **(E)** Score by site class
across the same 15,568 genes. *Canonical* is the window centred on the annotated
constitutive site (n = 15,568 per model); *Atlas* is any PolyASite 2.0 cluster
falling within 250 nt of a scored window, excluding clusters within 250 nt of the
canonical site itself, scored at its nearest window (n = 328,051); *Background* is
six non-canonical windows drawn uniformly at random per gene (seed 0; n = 93,408).
Violins are width-scaled, so shapes are comparable within a class but widths carry no
information across classes. **(F)** Peak-set agreement between models. Per-gene
Jaccard index of the window sets each pair of models calls as peaks, averaged over
15,568 genes. Each gene contributes its own cut, so the threshold varies by gene and
by model; genes with no annotated canonical site have no definable cut and are
excluded, and genes where a pair calls nothing at all contribute no value rather than
a Jaccard of zero. **(G)** Model score against site expression, for 367,488
gene-overlapping PolyASite 2.0 clusters with non-zero expression (a cluster
overlapping two scanned genes is counted once per gene). Clusters are grouped by
their atlas annotation as Terminal (TE), Intragenic (IN or EX) or Other (all
remaining classes). Within each model and group, sites are split into 12 equal-count
bins of log10 average expression; each point is the mean model score in a bin and the
ribbon is a 95% confidence interval on that mean (mean +/- 1.96 SE). Rho is the
Spearman correlation between score and expression computed on unbinned sites within
the group. A site's score is the maximum model score over all windows within 250 nt
of it.

**Methods common to all panels.** Positives are the 15,794 constitutive CPA sites
selected from PolyA_DB (hg38.PAS.main.tsv) as the site with the highest PSExRPM per
gene, minus 9 chrY sites excluded by the chromosome lists (15,785 used for
modelling); each is represented by a 500-nt window with the cleavage base at +250.
Negatives are genomic 500-nt windows drawn at a target ratio of 40 per positive
(realised 36.7), one in seven of which contains a canonical PAS hexamer ("decoys")
and six in seven of which do not, so PAS presence alone cannot separate the classes.
Sequence is taken strand-aware from hg38. RF and LR read a 720-column feature matrix:
15 k-mer and PWM families (PAS, UGUA, poly-U, CA/UA dinucleotide and DSE PWM
log-odds against a 0.3/0.2/0.2/0.3 ACGT background) counted in 48 overlapping 30-nt
bins tiled at 10-nt stride across the window. The CNN reads the one-hot 4x500
sequence directly (three convolutional blocks, 128/256/256 filters of width 8/8/4,
each with batch normalisation and ReLU, max-pooling to a fixed 4-position
representation, then a 128-unit dense layer; 661,633 parameters). Hyperparameters:
RF 30,000 trees, log-loss criterion, min_samples_leaf 4, balanced class weights,
with Platt (sigmoid) calibration fitted on the validation split; LR L1 penalty,
C = 0.1, saga solver, balanced class weights, uncalibrated; CNN AdamW (lr 1e-3,
weight decay 1e-4), batch size 256, cosine schedule with 5% linear warmup, binary
cross-entropy with 0.1 label smoothing, gradient clipping at 1.0, up to 40 epochs
with early stopping on validation AUPRC (patience 3). All three use seed 42. The
production split holds out whole chromosomes (train chr1, 2, 4-12, 14-17; validation
chr3, 13; test chr18-22, X; chrY excluded); the cross-fit models used in (C)
partition the same chromosomes into five folds by greedy bin-packing on positive
count, each fold's genes scored only by the model that held that fold out. Genome-wide
scans (panels B-G) apply the production models to every 500-nt window at 1-nt stride
across each annotated gene body, yielding one calibrated score per model per window
in 20,012 genes.

Peaks are called per gene, not against a global cut: a window is a peak for a model
if its score is at least the maximum score that model assigns within 250 nt of that
gene's own annotated 3' end. These per-gene cuts span the full score range (RF median
0.984, IQR 0.944-0.993, range 0.004-0.998) and do not produce degenerate peak sets
(median 0.9% of a gene's windows called). Every panel that requires a cut, and every
prod-scan panel (B, E, F, G), is therefore restricted to the 15,568 genes with an
annotated dominant CPA site in PolyA_DB. These genes are not a random subset of the
genome (genes lacking a dominant site skew toward low expression and dispersed
cleavage), so the figure characterises well-annotated genes specifically.
