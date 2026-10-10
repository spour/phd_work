#!/usr/bin/env python3
# new candidate definition: nothing downstream of the gene's canonical pa
import csv, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, HOLD_FRAC, RIDGE, NBOOT = 42, 0.20, 1.0, 500
rng = np.random.default_rng(SEED)
K3 = ["spl_upstream_max_me3", "spl_pas_prox_u1", "spl_u1_suppression_up"]

M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
cen = D.center.to_numpy().astype(float); strand = D.strand.to_numpy()
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
OLD = np.load("corr_rowmask.npz")["ROW"]
PB = np.load("binned_features178.npz"); BN = [str(s) for s in PB["names"]]; BX = PB["X"][ridx].astype(float)
GCB = [BN.index(f"bin{b}_gc") for b in range(1, 26)]
SPL = [i for i, n in enumerate(BN) if n.startswith("bin") and not n.endswith("_gc")]
X3 = np.column_stack([D[n].to_numpy(float) for n in K3])
tpos = np.where(strand == "+", cen, -cen)

gi = defaultdict(list)
for i in range(len(gene)): gi[gene[i]].append(i)
gi = {g: np.array(v) for g, v in gi.items()}

UP = np.zeros(len(y), bool)
for g, ix in gi.items():
    sub = ix[OLD[ix]]
    if len(sub) == 0 or (y[sub] == 1).sum() != 1: continue
    c = sub[np.argmax(y[sub])]
    UP[sub[(y[sub] == 0) & (tpos[sub] < tpos[c])]] = True
ROW = np.zeros(len(y), bool)
for g, ix in gi.items():
    sub = ix[OLD[ix]]
    if len(sub) == 0 or (y[sub] == 1).sum() != 1: continue
    c = sub[np.argmax(y[sub])]
    up = sub[(y[sub] == 0) & UP[sub]]
    if len(up) >= 1: ROW[np.r_[c, up]] = True
def ng(m): return len({gene[i] for i in np.where(m)[0]})
oc = OLD & (y == 0); dn = oc & ~UP
print("\n===== NEW CANDIDATE DEFINITION: nothing downstream of the canonical =====")
print(f"{'step':<46} {'peaks':>8} {'genes':>7}")
rows = [("corrected-set competitors (in)", int(oc.sum()), ng(oc)),
        ("removed: downstream of the canonical", int(dn.sum()), ng(dn)),
        ("survive: upstream of the canonical", int((oc & UP).sum()), ng(oc & UP)),
        ("survive + gene keeps its canonical", int((ROW & (y == 0)).sum()), ng(ROW & (y == 0)))]
for nm, a, b in rows: print(f"{nm:<46} {a:>8} {b:>7}")
lost = ng(oc) - ng(ROW & (y == 0))
print(f"\ngenes before {ng(oc)}, genes after {ng(ROW & (y==0))}, genes losing every competitor {lost}")
print(f"analysis set: {int(ROW.sum())} peaks = {int((ROW&(y==1)).sum())} canonical + "
      f"{int((ROW&(y==0)).sum())} competitor in {ng(ROW & (y==0))} genes")
NSv = np.array([int(ROW[ix].sum()) for g, ix in gi.items() if ROW[ix].sum() >= 2], float)
print(f"candidates per gene: median {np.median(NSv):.0f}, mean {NSv.mean():.2f}")
hit = []
for g, ix in gi.items():
    sub = ix[ROW[ix]]
    if len(sub) < 2: continue
    c = sub[np.argmax(y[sub])]
    hit.append(float(tpos[c] == tpos[sub].max()))
print(f"most-downstream rule P@1 on the new set: {np.mean(hit):.4f} over {len(hit)} genes "
      f"(1.0000 by construction; within-gene ranking is retired)")
with open("upstream_counts.tsv", "w") as fh:
    fh.write("step\tn_peaks\tn_genes\n")
    for nm, a, b in rows: fh.write(f"{nm}\t{a}\t{b}\n")
    fh.write(f"#genes_lost\t{lost}\t\n#median_candidates\t{np.median(NSv):.1f}\t\n"
             f"#positional_rule_p1\t{np.mean(hit):.5f}\t\n")
np.savez_compressed("upstream_rowmask.npz", ROW=ROW, UP=UP, ridx=ridx)

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
def loco(X, test_mask, train_mask):
    pred = np.full(len(y), np.nan)
    for c in CHROMS:
        te = ROW & test_mask & (chrom == c); tr = ROW & train_mask & (chrom != c)
        if te.sum() == 0 or tr.sum() == 0: continue
        A, B = prep(X[tr], X[te]); pred[te] = B @ fit_ridge(A, y[tr])
    return pred
def auroc(yv, s):
    o = np.argsort(s, kind="mergesort"); sv = s[o]; n = len(s); rk = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    rr = np.empty(n); rr[o] = rk; npos = yv.sum(); nneg = n - npos
    return float((rr[yv == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def auprc(yv, s):
    o = np.argsort(-s, kind="mergesort"); yy = yv[o]; sv = s[o]
    tp = np.cumsum(yy); k = np.arange(1, len(yy) + 1)
    idx = np.where(np.r_[sv[1:] != sv[:-1], True])[0]
    prec = (tp[idx] / k[idx])[np.searchsorted(idx, np.arange(len(yy)))]
    return float((prec * yy).sum() / yy.sum())
HG = [(g, ix[ROW[ix]]) for g, ix in gi.items()
      if ROW[ix].sum() >= 2 and is_hold[ix[0]] and (y[ix[ROW[ix]]] == 1).sum() == 1]
BOOT = rng.integers(0, len(HG), size=(NBOOT, len(HG)))
ALL = np.ones(len(y), bool)
NEW3 = ["spl_upstream_max_me3", "spl_pas_prox_u1", "spl_upstream_mean_u1"]
XN = np.column_stack([D[n].to_numpy(float) for n in NEW3])
BLOCKS = {"old three features (3)": X3, "new three features (3)": XN, "GC-binned (25)": BX[:, GCB],
          "splice-binned (150)": BX[:, SPL],
          "union (178)": np.column_stack([X3, BX[:, SPL], BX[:, GCB]])}
print(f"\n===== PER-SITE CLASSIFICATION, holdout: {len(HG)} genes =====")
print(f"{'block':<22} {'AUROC':>7} {'95% CI':>17} {'AUPRC':>7} {'base':>6} {'win-rate':>9} {'95% CI':>17}")
with open("upstream_blocks.tsv", "w") as fh:
    fh.write("block\tn_feat\tn_genes\tn_rows\tauroc\tauroc_lo\tauroc_hi\tauprc\tbase_rate\t"
             "winrate\twin_lo\twin_hi\n")
    for nm, X in BLOCKS.items():
        pred = loco(X, is_hold, ~is_hold)
        wr = np.empty(len(HG))
        for t, (g, ix) in enumerate(HG):
            s = pred[ix]; c = int(np.argmax(y[ix])); o = np.delete(np.arange(len(ix)), c)
            wr[t] = float(np.mean((s[c] > s[o]) + 0.5 * (s[c] == s[o])))
        rows_ = np.concatenate([ix for _, ix in HG])
        gidx = [ix for _, ix in HG]
        ab = np.empty(NBOOT); wb = np.empty(NBOOT); pb = np.empty(NBOOT)
        for t in range(NBOOT):
            sel = BOOT[t]; ii = np.concatenate([gidx[j] for j in sel])
            ab[t] = auroc(y[ii], pred[ii]); pb[t] = auprc(y[ii], pred[ii]); wb[t] = wr[sel].mean()
        au = auroc(y[rows_], pred[rows_]); ap = auprc(y[rows_], pred[rows_]); br = float(y[rows_].mean())
        print(f"{nm:<22} {au:>7.4f} [{np.percentile(ab,2.5):.4f},{np.percentile(ab,97.5):.4f}] "
              f"{ap:>7.4f} {br:>6.4f} {wr.mean():>9.4f} [{np.percentile(wb,2.5):.4f},"
              f"{np.percentile(wb,97.5):.4f}]")
        fh.write(f"{nm}\t{X.shape[1]}\t{len(HG)}\t{len(rows_)}\t{au:.5f}\t{np.percentile(ab,2.5):.5f}\t"
                 f"{np.percentile(ab,97.5):.5f}\t{ap:.5f}\t{br:.5f}\t{wr.mean():.5f}\t"
                 f"{np.percentile(wb,2.5):.5f}\t{np.percentile(wb,97.5):.5f}\n")
    print(f"chance: AUROC 0.5000, AUPRC = base rate, win-rate 0.5000; "
          f"{NBOOT} gene-clustered bootstrap replicates")
log("DONE_UPSTREAM_BLOCKS")
