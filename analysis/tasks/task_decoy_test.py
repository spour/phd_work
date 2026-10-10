#!/usr/bin/env python3
# compartment control: retained competitor versus 5 matched intronic decoys
import csv, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, HOLD_FRAC, RIDGE, NBOOT = 42, 0.20, 1.0, 200
rng = np.random.default_rng(SEED)

M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
P = np.load("binned_features178.npz"); PN = [str(s) for s in P["names"]]; PX = P["X"][ridx].astype(float)
DEC = list(csv.DictReader(open("decoy_positions.tsv"), delimiter="\t"))
Q = np.load("decoy_features.npz"); QN = [str(s) for s in Q["names"]]; QX = Q["X"].astype(float)
assert PN == QN and len(QX) == len(DEC)
GCB = [i for i, n in enumerate(PN) if n.endswith("_gc")]
SPL = [i for i, n in enumerate(PN) if n.startswith("bin") and not n.endswith("_gc")]
K3 = [PN.index(n) for n in ["spl_upstream_max_me3", "spl_pas_prox_u1", "spl_u1_suppression_up"]]

crow = np.array([int(r["competitor_row"]) for r in DEC])
keep = np.unique(crow)
X = np.vstack([PX[keep], QX])
y = np.r_[np.ones(len(keep)), np.zeros(len(QX))]
gene = np.r_[D.gene.to_numpy()[keep], np.array([r["gene"] for r in DEC])]
chrom = np.r_[D.chrom.to_numpy()[keep], np.array([r["chrom"] for r in DEC])]
pos = {v: i for i, v in enumerate(keep)}
setid = np.r_[np.arange(len(keep)), np.array([pos[c] for c in crow])]
log(f"{len(keep)} competitors, {len(QX)} decoys, {X.shape[1]} features")

si = defaultdict(list)
for i in range(len(y)): si[setid[i]].append(i)
si = {k: np.array(v) for k, v in si.items()}
gi = defaultdict(list)
for k, ix in si.items(): gi[gene[ix[0]]].append(k)
allg = np.array(sorted(gi))
hold_g = set(rng.choice(allg, size=int(round(HOLD_FRAC * len(allg))), replace=False))
is_hold = np.array([gene[i] in hold_g for i in range(len(y))])
CHROMS = sorted(set(chrom))

def fit_ridge(Xt, yt, lam=RIDGE):
    Pm = np.eye(Xt.shape[1]) * lam; Pm[0, 0] = 0.0; b = np.zeros(Xt.shape[1])
    for _ in range(80):
        eta = np.clip(Xt @ b, -30, 30); mu = 1 / (1 + np.exp(-eta)); w = np.maximum(mu * (1 - mu), 1e-9)
        z = eta + (yt - mu) / w; XtW = Xt.T * w
        try: nb = np.linalg.solve(XtW @ Xt + Pm, XtW @ z + Pm @ b)
        except np.linalg.LinAlgError: nb = np.linalg.lstsq(XtW @ Xt + Pm, XtW @ z + Pm @ b, rcond=None)[0]
        if np.max(np.abs(nb - b)) < 1e-9: b = nb; break
        b = nb
    return b
def prep(Xtr, Xte):
    mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0); sd[sd == 0] = 1.0
    mu = np.where(np.isfinite(mu), mu, 0.0)
    return (np.column_stack([np.ones(len(Xtr)), (np.where(np.isfinite(Xtr), Xtr, mu) - mu) / sd]),
            np.column_stack([np.ones(len(Xte)), (np.where(np.isfinite(Xte), Xte, mu) - mu) / sd]))
def loco_hold(cols):
    Xs = X[:, cols]; pred = np.full(len(y), np.nan)
    for c in CHROMS:
        te = is_hold & (chrom == c); tr = ~is_hold & (chrom != c)
        if te.sum() == 0 or tr.sum() == 0: continue
        A, Bm = prep(Xs[tr], Xs[te]); pred[te] = Bm @ fit_ridge(A, y[tr])
    return pred
def per_set(pred):
    hits, rrs, ns, sl = [], [], [], []
    for k, ix in si.items():
        if not is_hold[ix[0]]: continue
        s = pred[ix]
        if not np.isfinite(s).all() or (y[ix] == 1).sum() != 1: continue
        n = len(ix); j = np.argsort(-(s + rng.random(n) * 1e-12), kind="mergesort")
        r = np.empty(n); r[j] = np.arange(1, n + 1)
        c = int(np.argmax(y[ix]))
        hits.append(1.0 if j[0] == c else 0.0); rrs.append(1.0 / r[c]); ns.append(n); sl.append((k, ix))
    return np.array(hits), np.array(rrs), np.array(ns, float), sl
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

BLOCKS = {"GC-binned (25)": GCB, "splice-binned (150)": SPL, "k3 splicing (3)": K3,
          "union (178)": K3 + SPL + GCB}
PRED = {nm: loco_hold(c) for nm, c in BLOCKS.items()}
log("fits done")
_, _, _, SL0 = per_set(PRED["k3 splicing (3)"])
sg = defaultdict(list)
for t, (k, ix) in enumerate(SL0): sg[gene[ix[0]]].append(t)
GK = list(sg); GIDX = [np.array(sg[g]) for g in GK]
BOOT = rng.integers(0, len(GK), size=(NBOOT, len(GK)))
print("\n===== COMPARTMENT CONTROL: competitor vs 5 matched intronic decoys =====")
print(f"holdout sets {len(SL0)} in {len(GK)} genes; chance P@1 {np.mean([1.0/len(ix) for _, ix in SL0]):.4f}, "
      f"chance MRR {np.mean([np.sum(1.0/np.arange(1,len(ix)+1))/len(ix) for _, ix in SL0]):.4f}, "
      f"base rate {np.mean([1.0/len(ix) for _, ix in SL0]):.4f}")
print(f"{'block':<22} {'P@1':>7} {'95% CI':>17} {'MRR':>7} {'AUROC':>7} {'95% CI':>17} {'AUPRC':>7}")
with open("decoy_test.tsv", "w") as fh:
    fh.write("block\tn_feat\tn_sets\tp1\tp1_lo\tp1_hi\tp1_chance\tmrr\tmrr_chance\tauroc\tauroc_lo\tauroc_hi\tauprc\tbase_rate\n")
    for nm in BLOCKS:
        h, rr, ns, sl = per_set(PRED[nm])
        keep_t = [t for t, (k, _) in enumerate(sl)]
        rows = np.concatenate([ix for _, ix in sl])
        hb = np.empty(NBOOT); ab = np.empty(NBOOT)
        for t in range(NBOOT):
            gs = np.concatenate([GIDX[j] for j in BOOT[t]])
            gs = gs[gs < len(h)]
            ii = np.concatenate([sl[j][1] for j in gs])
            hb[t] = h[gs].mean(); ab[t] = auroc(y[ii], PRED[nm][ii])
        ch = float(np.mean(1.0 / ns))
        mc = float(np.mean([np.sum(1.0 / np.arange(1, int(n) + 1)) / n for n in ns]))
        au = auroc(y[rows], PRED[nm][rows]); ap = auprc(y[rows], PRED[nm][rows])
        print(f"{nm:<22} {h.mean():>7.4f} [{np.percentile(hb,2.5):.4f},{np.percentile(hb,97.5):.4f}] "
              f"{rr.mean():>7.4f} {au:>7.4f} [{np.percentile(ab,2.5):.4f},{np.percentile(ab,97.5):.4f}] {ap:>7.4f}")
        fh.write(f"{nm}\t{len(BLOCKS[nm])}\t{len(h)}\t{h.mean():.5f}\t{np.percentile(hb,2.5):.5f}\t"
                 f"{np.percentile(hb,97.5):.5f}\t{ch:.5f}\t{rr.mean():.5f}\t{mc:.5f}\t{au:.5f}\t"
                 f"{np.percentile(ab,2.5):.5f}\t{np.percentile(ab,97.5):.5f}\t{ap:.5f}\t{float(y[rows].mean()):.5f}\n")
    print(f"\nchance: P@1 {ch:.4f}, MRR {mc:.4f}, AUROC 0.5000, AUPRC = base rate {float(y[rows].mean()):.4f}")
log("DONE_DECOY_TEST")
