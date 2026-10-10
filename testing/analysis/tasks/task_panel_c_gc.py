#!/usr/bin/env python3
# two panel fixes: the polya_db comparison, and the discriminant-versus-gc scatter
import csv, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, HOLD_FRAC, RIDGE = 42, 0.20, 1.0
rng = np.random.default_rng(SEED)
NEAR = 500.0

M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
ROW = np.load("upstream_rowmask.npz")["ROW"]
Zc = np.load("corr_rowmask.npz"); APA = Zc["APA"]
PDB = np.load("polyadb_peak_cache.npz")
pdb_rpm = PDB["rpm"][ridx]; pdb_pse = PDB["pse"][ridx]; pdb_d = PDB["dist"][ridx]
GC = np.load("termexon_gc_cache.npz")["gc500"][ridx]
me3 = D["spl_upstream_max_me3"].to_numpy(float)
CS = np.load("control_contrast_scan.npz", allow_pickle=True); CSN = list(CS["names"])
gcc = CS["X"][:, CSN.index("gc_distal")] - CS["X"][:, CSN.index("gc_proximal")]
X2 = np.column_stack([me3, gcc]); N = len(D)
isc = y == 1; comp = y == 0

# panel c, common measurement rule
def med_ci(v):
    v = v[np.isfinite(v)]
    b = np.median(v[rng.integers(0, len(v), size=(500, len(v)))], axis=1)
    return float(np.median(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
GROUPS = [("canonical", ROW & isc), ("removed competitor (APA)", comp & APA),
          ("retained competitor", ROW & comp)]
print("\n===== PANEL C: where the values come from =====")
print(f"{'group':<26} {'n':>7} {'with a site <=500 nt':>21} {'median distance (nt)':>21}")
for nm, m in GROUPS:
    print(f"{nm:<26} {int(m.sum()):>7,} {float(np.mean(pdb_d[m] <= NEAR)):>21.4f} "
          f"{float(np.median(pdb_d[m])):>21,.0f}")
print("\n===== PANEL C: restricted to a matched site within 500 nt =====")
print(f"{'group':<26} {'n':>7} {'RPM median':>12} {'PSE median':>12}")
with open("up_panel_c_matched.tsv", "w") as fh:
    fh.write("group\tmeasure\tn\tmedian\tci_lo\tci_hi\tq25\tq75\n")
    for nm, m in GROUPS:
        mm = m & (pdb_d <= NEAR)
        if mm.sum() == 0:
            print(f"{nm:<26} {0:>7} {'-':>12} {'-':>12}   (no site within 500 nt by construction)")
            continue
        out = []
        for meas, v in [("RPM", pdb_rpm), ("PSE", pdb_pse)]:
            x = v[mm]; md, lo, hi = med_ci(x)
            q1, q3 = np.percentile(x[np.isfinite(x)], [25, 75])
            fh.write(f"{nm}\t{meas}\t{int(mm.sum())}\t{md:.5f}\t{lo:.5f}\t{hi:.5f}\t{q1:.5f}\t{q3:.5f}\n")
            out.append(md)
        print(f"{nm:<26} {int(mm.sum()):>7,} {out[0]:>12.4f} {out[1]:>12.4f}")
    nret = int((ROW & comp).sum())
    fh.write(f"#retained_n\t{nret}\t\t\t\t\t\t\n")
    fh.write(f"#retained_median_dist\t{float(np.median(pdb_d[ROW & comp])):.1f}\t\t\t\t\t\t\n")
    fh.write(f"#near\t{NEAR:.0f}\t\t\t\t\t\t\n")

# panel k, discriminant versus single-window gc
gi = defaultdict(list)
for i in range(N): gi[gene[i]].append(i)
gi = {g: np.array(v) for g, v in gi.items()}
allg = np.array(sorted(set(gene)))
hold_g = set(rng.choice(allg, size=int(round(HOLD_FRAC * len(allg))), replace=False))
is_hold = np.array([g in hold_g for g in gene])
CHROMS = sorted(set(chrom))
def fit_ridge(Xt, yt, lam=RIDGE):
    P = np.eye(Xt.shape[1]) * lam; P[0, 0] = 0.0; b = np.zeros(Xt.shape[1])
    for _ in range(80):
        eta = np.clip(Xt @ b, -30, 30); mu = 1 / (1 + np.exp(-eta)); w = np.maximum(mu * (1 - mu), 1e-9)
        z = eta + (yt - mu) / w; XtW = Xt.T * w
        try: nb = np.linalg.solve(XtW @ Xt + P, XtW @ z + P @ b)
        except np.linalg.LinAlgError: nb = np.linalg.lstsq(XtW @ Xt + P, XtW @ z + P @ b, rcond=None)[0]
        if np.max(np.abs(nb - b)) < 1e-9: b = nb; break
        b = nb
    return b
def prep(Xtr, Xte):
    mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0); sd[sd == 0] = 1.0
    mu = np.where(np.isfinite(mu), mu, 0.0)
    return (np.column_stack([np.ones(len(Xtr)), (np.where(np.isfinite(Xtr), Xtr, mu) - mu) / sd]),
            np.column_stack([np.ones(len(Xte)), (np.where(np.isfinite(Xte), Xte, mu) - mu) / sd]))
pred = np.full(N, np.nan)
for c in CHROMS:
    te = ROW & (chrom == c); tr = ROW & (chrom != c)
    if te.sum() == 0 or tr.sum() == 0: continue
    A, B = prep(X2[tr], X2[te]); pred[te] = B @ fit_ridge(A, y[tr])
def pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b); a = a[m] - a[m].mean(); b = b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan
rows = np.where(ROW)[0]; yv = y[rows]
r_all = pearson(pred[rows], GC[rows])
r_can = pearson(pred[rows][yv == 1], GC[rows][yv == 1])
r_com = pearson(pred[rows][yv == 0], GC[rows][yv == 0])
print("\n===== PANEL K: held-out discriminant versus single-window GC =====")
print(f"  r all {r_all:+.4f}   canonical {r_can:+.4f}   competitor {r_com:+.4f}   "
      f"GC explains {100*r_all**2:.1f}% of discriminant variance")
print(f"  r(gc_contrast, single-window GC) = {pearson(gcc[rows], GC[rows]):+.4f}")
print(f"  mean single-window GC: canonical {np.nanmean(GC[rows][yv==1]):.4f}, "
      f"competitor {np.nanmean(GC[rows][yv==0]):.4f}")
with open("gc_scatter.tsv", "w") as fh:
    fh.write("class\tgc\tdiscriminant\n")
    for i in rows:
        if np.isfinite(pred[i]) and np.isfinite(GC[i]):
            fh.write(f"{'canonical' if y[i]==1 else 'competitor'}\t{GC[i]:.5f}\t{pred[i]:.5f}\n")
    fh.write(f"#r_all\t{r_all:.5f}\t\n#r_canonical\t{r_can:.5f}\t\n#r_competitor\t{r_com:.5f}\t\n")
    fh.write(f"#r2_all\t{r_all**2:.6f}\t\n#r_gcc_gc\t{pearson(gcc[rows], GC[rows]):.5f}\t\n")
log("DONE_PANEL_C_GC")
