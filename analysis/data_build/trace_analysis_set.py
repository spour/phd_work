#!/usr/bin/env python3
# reconstruct in_analysis_set from the raw annotation caches and count what each filter
import sys, csv
import numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)

APA_MAX = 500

M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); strand = D.strand.to_numpy()
cen = D.center.to_numpy().astype(np.int64)
y = D.is_canonical.to_numpy().astype(bool)
N = len(D)
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])

At = np.load("atlas_peak_cache.npz")["dist"][ridx]
Pd = np.load("polyadb_peak_cache.npz")["dist"][ridx]
APA = (At <= APA_MAX) | (Pd <= APA_MAX)
TERMEX = np.load("termexon_gc_cache.npz")["te_window"][ridx]

gi = defaultdict(list)
for i in range(N): gi[gene[i]].append(i)
gi = {g: np.array(v) for g, v in gi.items()}
tpos = np.where(strand == "+", cen, -cen)
def ng(m): return len({gene[i] for i in np.where(m)[0]})

def gene_filter(mask):
    keep = np.zeros(N, bool)
    for g, ix in gi.items():
        sub = ix[mask[ix]]
        if (y[sub] == 1).sum() == 1 and (y[sub] == 0).sum() >= 1: keep[sub] = True
    return keep

COMP = ~y
SURV = COMP & ~APA & ~TERMEX
CORR = gene_filter(y | SURV)

UP = np.zeros(N, bool)
for g, ix in gi.items():
    sub = ix[CORR[ix]]
    if len(sub) == 0 or y[sub].sum() != 1: continue
    c = sub[np.argmax(y[sub])]
    UP[sub[(~y[sub]) & (tpos[sub] < tpos[c])]] = True
ROW = np.zeros(N, bool)
for g, ix in gi.items():
    sub = ix[CORR[ix]]
    if len(sub) == 0 or y[sub].sum() != 1: continue
    c = sub[np.argmax(y[sub])]
    up = sub[(~y[sub]) & UP[sub]]
    if len(up) >= 1: ROW[np.r_[c, up]] = True

# the reconstruction must equal what is on disk
assert (CORR == np.load("corr_rowmask.npz")["ROW"]).all(), "corrected set does not reproduce"
assert (ROW == np.load("upstream_rowmask.npz")["ROW"]).all(), "analysis set does not reproduce"
P = pd.read_parquet("cpa_peaks_final.parquet", engine="fastparquet",
                    columns=["gene", "chrom", "center", "strand", "in_analysis_set"])
kp = {(g, c, int(ce), st): b for g, c, ce, st, b in
      zip(P.gene, P.chrom, P.center, P.strand, P.in_analysis_set)}
shipped = np.array([kp[(gene[i], chrom[i], int(cen[i]), strand[i])] for i in range(N)], bool)
assert (ROW == shipped).all(), "reconstruction differs from the shipped in_analysis_set"
log("[check] reconstruction == corr_rowmask == upstream_rowmask == parquet in_analysis_set")

# the funnel
DNCUT = CORR & COMP & ~UP
rows = [
 ("all peaks", int(N), ng(np.ones(N, bool)), int(y.sum()), int(COMP.sum()), 0, ""),
 ("competitors: APA-annotated removed", int((y | (COMP & ~APA)).sum()),
  ng(y | (COMP & ~APA)), int(y.sum()), int((COMP & ~APA).sum()), int((COMP & APA).sum()),
  "PolyASite 2.0 or PolyA_DB v4 site within 500 nt, same strand"),
 ("competitors: reference terminal exon removed", int((y | SURV).sum()), ng(y | SURV),
  int(y.sum()), int(SURV.sum()), int((COMP & ~APA & TERMEX).sum()),
  "500 nt upstream window overlaps a GENCODE v49 last exon"),
 ("genes without a surviving competitor dropped", int(CORR.sum()), ng(CORR),
  int((CORR & y).sum()), int((CORR & COMP).sum()), int((y | SURV).sum()) - int(CORR.sum()),
  "gene must keep one canonical and >= 1 competitor"),
 ("competitors: at or downstream of the canonical removed",
  int((CORR & (y | UP)).sum()), ng(CORR & (y | UP)), int((CORR & y).sum()), int(UP.sum()),
  int(DNCUT.sum()), "transcript coordinate >= the gene's canonical"),
 ("analysis set", int(ROW.sum()), ng(ROW), int((ROW & y).sum()), int((ROW & COMP).sum()),
  int((CORR & (y | UP)).sum()) - int(ROW.sum()), "gene rule re-applied"),
]
with open("analysis_set_funnel.tsv", "w") as fh:
    fh.write("step\tn_peaks\tn_genes\tn_canonical\tn_competitor\tn_removed\tcriterion\n")
    for r in rows: fh.write("\t".join(str(x) for x in r) + "\n")

log("")
log(f"{'step':<52}{'peaks':>8}{'genes':>8}{'canon':>8}{'comp':>8}{'removed':>9}")
for nm, p_, g_, c_, k_, r_, _ in rows:
    log(f"{nm:<52}{p_:>8,}{g_:>8,}{c_:>8,}{k_:>8,}{r_:>9,}")
log("")
log(f"[overlap] APA and terminal-exon flags both set on {int((COMP & APA & TERMEX).sum()):,} "
    f"competitors; APA only {int((COMP & APA & ~TERMEX).sum()):,}, terminal exon only "
    f"{int((COMP & ~APA & TERMEX).sum()):,}, union {int((COMP & (APA | TERMEX)).sum()):,}")
log(f"[order]   the two exclusions are one conjunction in task_rerank_corr.py "
    f"(SURV = COMP & ~APA & ~TERMEX), so their order is not defined by the code; the "
    f"removed counts above attribute the overlap to APA, which is listed first")
log(f"[genes]   {ng(COMP) - ng(CORR):,} genes lose every competitor at the annotation stage "
    f"and {ng(CORR) - ng(ROW):,} more at the upstream stage; their canonical peaks leave with them "
    f"({int(y.sum()) - int((ROW & y).sum()):,} canonical peaks in total)")
log("[out] analysis_set_funnel.tsv")
