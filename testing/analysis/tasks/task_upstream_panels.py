#!/usr/bin/env python3
# panel data for the per-site figure, upstream-only candidate set
import csv, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, HOLD_FRAC, RIDGE, NBOOT = 42, 0.20, 1.0, 500
rng = np.random.default_rng(SEED)
NEW3 = ["spl_upstream_max_me3", "spl_pas_prox_u1", "spl_upstream_mean_u1"]

M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
rf = D.rf_prob.to_numpy(float); cen = D.center.to_numpy().astype(float); strand = D.strand.to_numpy()
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
ROW = np.load("upstream_rowmask.npz")["ROW"]
OLD = np.load("corr_rowmask.npz")["ROW"]; Zc = np.load("corr_rowmask.npz")
APA = Zc["APA"]; TERMEX = Zc["TERMEX"]
GC = np.load("termexon_gc_cache.npz")["gc500"][ridx]
PB = np.load("binned_features178.npz"); BN = [str(s) for s in PB["names"]]; BX = PB["X"][ridx].astype(float)
GCB = [BN.index(f"bin{b}_gc") for b in range(1, 26)]
PDB = np.load("polyadb_peak_cache.npz")
pdb_rpm = PDB["rpm"][ridx]; pdb_pse = PDB["pse"][ridx]; pdb_d = PDB["dist"][ridx]
X3 = np.column_stack([D[n].to_numpy(float) for n in NEW3])
tpos = np.where(strand == "+", cen, -cen)

gi = defaultdict(list)
for i in range(len(gene)): gi[gene[i]].append(i)
gi = {g: np.array(v) for g, v in gi.items()}
allg = np.array(sorted(set(gene)))
hold_g = set(rng.choice(allg, size=int(round(HOLD_FRAC * len(allg))), replace=False))
is_hold = np.array([g in hold_g for g in gene])
CHROMS = sorted(set(chrom))
ALL = np.ones(len(y), bool)
isc = y == 1; comp = y == 0

# b: pool composition, three filters
DOWN = OLD & comp & ~ROW
with open("up_panel_b.tsv", "w") as fh:
    fh.write("class\tcategory\tn\tfrac\n")
    tot = int(comp.sum())
    for lab, v in [("kept: upstream, unannotated, intronic", ROW & comp),
                   ("removed: downstream of the canonical", DOWN),
                   ("removed: reference terminal exon", comp & TERMEX & ~APA),
                   ("removed: APA-annotated", comp & APA)]:
        fh.write(f"competitor\t{lab}\t{int(v.sum())}\t{v.sum()/tot:.5f}\n")
    fh.write(f"#total\t{tot}\n#kept\t{int((ROW&comp).sum())}\n#genes\t{len({gene[i] for i in np.where(ROW&comp)[0]})}\n")

# c: usage of removed vs retained
def med_ci(v):
    v = v[np.isfinite(v)]
    b = np.median(v[rng.integers(0, len(v), size=(500, len(v)))], axis=1)
    return float(np.median(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
with open("up_panel_c.tsv", "w") as fh:
    fh.write("group\tmeasure\tn\tmedian\tci_lo\tci_hi\tq25\tq75\tfrac_site_within_500\n")
    for nm, m in [("canonical", ROW & isc), ("removed competitor (APA)", comp & APA),
                  ("retained competitor", ROW & comp)]:
        near = float(np.mean(pdb_d[m] <= 500))
        for meas, v in [("RPM", pdb_rpm), ("PSE", pdb_pse)]:
            x = v[m]; md, lo, hi = med_ci(x); q1, q3 = np.percentile(x[np.isfinite(x)], [25, 75])
            fh.write(f"{nm}\t{meas}\t{int(m.sum())}\t{md:.5f}\t{lo:.5f}\t{hi:.5f}\t{q1:.5f}\t{q3:.5f}\t{near:.5f}\n")

# model
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
def loco(X, test_mask=ALL, train_mask=ALL):
    pred = np.full(len(y), np.nan)
    for c in CHROMS:
        te = ROW & test_mask & (chrom == c); tr = ROW & train_mask & (chrom != c)
        if te.sum() == 0 or tr.sum() == 0: continue
        A, B = prep(X[tr], X[te]); pred[te] = B @ fit_ridge(A, y[tr])
    return pred
def auroc(yv, s):
    m = np.isfinite(s); yv = yv[m]; s = s[m]
    o = np.argsort(s, kind="mergesort"); sv = s[o]; n = len(s); rk = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    rr = np.empty(n); rr[o] = rk; npos = yv.sum(); nneg = n - npos
    return float((rr[yv == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def winrate(v, G):
    out = np.full(len(G), np.nan)
    for t, (g, ix) in enumerate(G):
        s = v[ix]; c = int(np.argmax(y[ix])); o = np.delete(np.arange(len(ix)), c)
        if not np.isfinite(s[c]): continue
        d = s[o][np.isfinite(s[o])]
        if len(d): out[t] = float(np.mean((s[c] > d) + 0.5 * (s[c] == d)))
    return out
pred3 = loco(X3); predG = loco(BX[:, GCB])
GEN = [(g, ix[ROW[ix]]) for g, ix in gi.items() if ROW[ix].sum() >= 2 and (y[ix[ROW[ix]]] == 1).sum() == 1]

# d: curves
rows_all = np.where(ROW)[0]
def curve(sv, tag, fh):
    o = rows_all[np.argsort(-sv[rows_all], kind="mergesort")]; yy = y[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy); P = yy.sum(); N = len(yy) - P
    for i in np.unique(np.linspace(0, len(yy) - 1, 400).astype(int)):
        fh.write(f"{tag}\t{fp[i]/N:.6f}\t{tp[i]/P:.6f}\t{tp[i]/(i+1):.6f}\n")
with open("up_curves.tsv", "w") as fh:
    fh.write("model\tfpr\ttpr\tprecision\n")
    curve(pred3, "three features", fh); curve(predG, "GC-binned", fh); curve(rf, "baseline model", fh)
    fh.write(f"#base_rate\t{float(y[rows_all].mean()):.5f}\t\t\n")
    fh.write(f"#auroc_three\t{auroc(y[rows_all], pred3[rows_all]):.5f}\t\t\n")
    fh.write(f"#auroc_gc\t{auroc(y[rows_all], predG[rows_all]):.5f}\t\t\n")
    fh.write(f"#auroc_rf\t{auroc(y[rows_all], rf[rows_all]):.5f}\t\t\n")

with open("up_bychrom.tsv", "w") as fh:
    fh.write("chrom\tn_genes\tn_rows\tauroc\tci_lo\tci_hi\n")
    for c in CHROMS:
        gl = [ix for g, ix in GEN if chrom[ix[0]] == c]
        if len(gl) < 20: continue
        rows_ = np.concatenate(gl)
        a = auroc(y[rows_], pred3[rows_])
        b = np.empty(200)
        for t in range(200):
            ii = np.concatenate([gl[j] for j in rng.integers(0, len(gl), len(gl))])
            b[t] = auroc(y[ii], pred3[ii])
        fh.write(f"{c}\t{len(gl)}\t{len(rows_)}\t{a:.5f}\t{np.percentile(b,2.5):.5f}\t{np.percentile(b,97.5):.5f}\n")

def pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b); a = a[m] - a[m].mean(); b = b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan
rows_ = np.concatenate([ix for _, ix in GEN])
yv = y[rows_]
with open("up_gc_points.tsv", "w") as fh:
    fh.write("gene\tclass\tgc\tdiscriminant\n")
    for i in rows_:
        fh.write(f"{gene[i]}\t{'canonical' if y[i]==1 else 'competitor'}\t{GC[i]:.5f}\t{pred3[i]:.5f}\n")
mfin = np.isfinite(pred3[rows_]) & np.isfinite(GC[rows_])
A1 = np.column_stack([np.ones(mfin.sum()), GC[rows_][mfin]])
bb = np.linalg.lstsq(A1, pred3[rows_][mfin], rcond=None)[0]
res = np.full(len(y), np.nan); res[rows_[mfin]] = pred3[rows_][mfin] - A1 @ bb
ENT = [("GC, single 500 nt window", GC), ("GC, 25 binned profile", predG),
       ("three features", pred3), ("three features, GC removed", res)]
with open("up_gc_stats.tsv", "w") as fh:
    fh.write("quantity\tauroc\tauroc_lo\tauroc_hi\twinrate\twin_lo\twin_hi\n")
    gl = [ix for _, ix in GEN]
    for nm, v in ENT:
        a = auroc(y[rows_], v[rows_]); w = winrate(v, GEN); wm = float(np.nanmean(w))
        ab = np.empty(200); wb = np.empty(200)
        for t in range(200):
            sel = rng.integers(0, len(gl), len(gl)); ii = np.concatenate([gl[j] for j in sel])
            ab[t] = auroc(y[ii], v[ii]); wb[t] = np.nanmean(w[sel])
        fh.write(f"{nm}\t{a:.5f}\t{np.percentile(ab,2.5):.5f}\t{np.percentile(ab,97.5):.5f}\t"
                 f"{wm:.5f}\t{np.percentile(wb,2.5):.5f}\t{np.percentile(wb,97.5):.5f}\n")
    for lab, sub in [("all peaks", np.ones(len(rows_), bool)), ("canonical", yv == 1), ("competitor", yv == 0)]:
        fh.write(f"#r_{lab}\t{pearson(pred3[rows_][sub], GC[rows_][sub]):.5f}\t\t\t\t\t\n")
    fh.write(f"#resid_r2\t{pearson(pred3[rows_][mfin], GC[rows_][mfin])**2:.6f}\t\t\t\t\t\n")
print("wrote up_panel_b/c, up_curves, up_bychrom, up_gc_points, up_gc_stats")
print(f"three features AUROC (all analysis rows) {auroc(y[rows_], pred3[rows_]):.4f}; "
      f"GC-binned {auroc(y[rows_], predG[rows_]):.4f}; baseline {auroc(y[rows_], rf[rows_]):.4f}; "
      f"base rate {float(y[rows_].mean()):.4f}")
log("DONE_UPSTREAM_PANELS")
