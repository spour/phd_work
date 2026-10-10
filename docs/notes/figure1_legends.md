# Figure 1

Sequence models of cleavage and polyadenylation site usage: held-out performance, between-model agreement, and behaviour at annotated sites.

**(A)** Held-out test performance. AUROC, AUPRC and F1 for the random forest (RF), logistic regression (LR) and convolutional network (CNN) on 104,031 test windows windows held out from training. Bars are single values from the locked evaluation, not means, so no error bars are shown.

**(B)** Per-window score agreement. Pearson correlation between each pair of models across every scored window in 15,568 genes genes, accumulated in a single streaming pass. High AUROC in (A) does not imply the models agree window by window.

**(C)** Top-K canonical recovery. Fraction of 15,559 genes genes whose annotated constitutive cleavage site falls within 250 nt of one of the model's K highest-scoring windows, from cross-fit predictions so no gene is scored by a model that saw it in training. Combined takes the best rank across the three models. K = 1 is argmax recovery.

**(D)** Score traces along a single gene (ERP27, 24,750 windows windows). Each model's score at every window position. No threshold line is drawn: peaks are called per gene using the score at that gene's own 3' end, so a single horizontal cut would not correspond to the calling rule.

**(E)** Score by site class across 15,568 genes genes. Canonical is the annotated constitutive site; Atlas is a PolyASite site within 250 nt of a scored window, excluding the canonical site itself; Background is a random sample of non-canonical windows per gene. Violins are width-scaled, so widths are comparable within a class but not across classes.

**(F)** Peak-set agreement between models. Per-gene Jaccard index of the window sets called as peaks, averaged over 15,568 genes genes. Each gene's cut is the score at its own annotated 3' end, so the threshold varies by gene and model; genes with no annotated canonical site are excluded. Genes where both models call nothing contribute no value rather than a Jaccard of zero.

**(G)** Model score against site expression, for 367,488 sites PolyASite sites falling inside scanned genes. Sites are grouped as Terminal (TE), Intragenic (IN or EX) or Other; each point is the mean model score in an expression bin, with a 95% confidence interval on that mean, and rho is the Spearman correlation between score and expression across unbinned sites in that group. A site's score is the maximum over windows within 250 nt of it.

---

All panels use the same gene set: genes with an annotated dominant CPA site in PolyA_DB. Genes without one are excluded throughout, since no per-gene peak-calling cut can be defined for them. These are not a random subset — they skew toward low expression and dispersed cleavage — so the figure describes well-annotated genes.

Peaks are called per gene: a window is a peak for a model if its score is at least the score that model assigns at that gene's own annotated 3' end. Per-gene cut distributions are in `fig1_thresholds.tsv`; they span the full score range (RF median 0.98, range 0.004-0.998), and low-cut genes do not produce degenerate peak sets (median 0.9% of windows called).
