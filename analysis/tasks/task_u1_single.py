#!/usr/bin/env python3
# can one interpretable u1 contrast replace the suppression pair?
import csv, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, HOLD_FRAC, RIDGE, NBOOT = 42, 0.20, 1.0, 500
rng = np.random.default_rng(SEED)

M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
ROW = np.load("upstream_rowmask.npz")["ROW"]
me3 = D["spl_upstream_max_me3"].to_numpy(float)
umean = D["spl_upstream_mean_u1"].to_numpy(float)
uprox = D["spl_pas_prox_u1"].to_numpy(float)
udist = (6 * umean - uprox) / 5.0
contrastB = udist - uprox
contrastA = umean - uprox
gi = defaultdict(list)
for i in range(len(gene)): gi[gene[i]].append(i)
gi = {g: np.array(v) for g, v in gi.items()}
allg = np.array(sorted(set(gene)))
hold_g = set(rng.choice(allg, size=int(round(HOLD_FRAC * len(allg))), replace=False))
is_hold = np.array([g in hold_g for g in gene])
CHROMS = sorted(set(chrom))
GEN = [(g, ix[ROW[ix]]) for g, ix in gi.items() if ROW[ix].sum() >= 2 and (y[ix[ROW[ix]]] == 1).sum() == 1]
HG = [(g, ix) for g, ix in GEN if is_hold[ix[0]]]
IDX = np.where(ROW)[0]

def auroc(yv, s):
    m = np.isfinite(s); yv = yv[m]; s = s[m]
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
def winrate(v, G):
    out = np.full(len(G), np.nan)
    for t, (g, ix) in enumerate(G):
        s = v[ix]; c = int(np.argmax(y[ix])); o = np.delete(np.arange(len(ix)), c)
        if not np.isfinite(s[c]): continue
        d = s[o][np.isfinite(s[o])]
        if len(d): out[t] = float(np.mean((s[c] > d) + 0.5 * (s[c] == d)))
    return out
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
def loco_hold(X):
    pred = np.full(len(y), np.nan)
    for c in CHROMS:
        te = ROW & is_hold & (chrom == c); tr = ROW & ~is_hold & (chrom != c)
        if te.sum() == 0 or tr.sum() == 0: continue
        A, B = prep(X[tr], X[te]); pred[te] = B @ fit_ridge(A, y[tr])
    return pred
def coefs(X, names):
    dev = np.where(ROW & ~is_hold)[0]
    A, _ = prep(X[dev], X[dev]); beta = fit_ridge(A, y[dev])[1:]
    devg = [ix for g, ix in GEN if not is_hold[ix[0]]]
    BB = np.empty((300, X.shape[1]))
    for t in range(300):
        ii = np.concatenate([devg[j] for j in rng.integers(0, len(devg), len(devg))])
        Ab, _ = prep(X[ii], X[ii]); BB[t] = fit_ridge(Ab, y[ii])[1:]
    out = []
    for a, nm in enumerate(names):
        lo, hi = np.percentile(BB[:, a], [2.5, 97.5])
        out.append((nm, float(beta[a]), float(lo), float(hi),
                    float(np.mean(np.sign(BB[:, a]) != np.sign(beta[a])))))
    return out

print("\n===== MARGINAL SUPPORT FOR THE CONTRAST =====")
print(f"r(contrast B, contrast A) = {np.corrcoef(contrastB[IDX], contrastA[IDX])[0,1]:.6f} "
      f"(proportional by construction, factor 5/6)")
print(f"{'feature':<38} {'AUROC':>7} {'win-rate':>9} {'95% CI':>17}")
gl = [ix for _, ix in GEN]
with open("u1_single.tsv", "w") as fh:
    fh.write("kind\tname\tauroc\twinrate\tlo\thi\tbeta\tbeta_lo\tbeta_hi\tflip\n")
    for nm, v in [("u1_contrast (distal - proximal)", contrastB),
                  ("u1_contrast (mean - proximal)", contrastA),
                  ("spl_upstream_mean_u1", umean), ("spl_pas_prox_u1", uprox),
                  ("spl_upstream_max_me3", me3)]:
        a = auroc(y[IDX], v[IDX]); w = winrate(v, GEN); wm = float(np.nanmean(w))
        b = np.array([np.nanmean(w[rng.integers(0, len(w), len(w))]) for _ in range(300)])
        print(f"{nm:<38} {a:>7.4f} {wm:>9.4f} [{np.percentile(b,2.5):.4f},{np.percentile(b,97.5):.4f}]")
        fh.write(f"marginal\t{nm}\t{a:.5f}\t{wm:.5f}\t{np.percentile(b,2.5):.5f}\t"
                 f"{np.percentile(b,97.5):.5f}\t\t\t\t\n")

    print("\n===== MODELS (holdout, %d genes; chance AUROC 0.5, win-rate 0.5) =====" % len(HG))
    print(f"{'model':<42} {'AUROC':>7} {'95% CI':>17} {'AUPRC':>7} {'win-rate':>9} {'95% CI':>17}")
    MODELS = [("three features", np.column_stack([me3, uprox, umean]),
               ["spl_upstream_max_me3", "spl_pas_prox_u1", "spl_upstream_mean_u1"]),
              ("two features: acceptor + u1_contrast", np.column_stack([me3, contrastB]),
               ["spl_upstream_max_me3", "u1_contrast"]),
              ("acceptor alone", me3[:, None], ["spl_upstream_max_me3"])]
    for nm, X, names in MODELS:
        pred = loco_hold(X)
        rows = np.concatenate([ix for _, ix in HG])
        a = auroc(y[rows], pred[rows]); ap = auprc(y[rows], pred[rows])
        w = winrate(pred, HG); wm = float(np.nanmean(w))
        gidx = [ix for _, ix in HG]
        ab = np.empty(NBOOT); wb = np.empty(NBOOT)
        for t in range(NBOOT):
            sel = rng.integers(0, len(gidx), len(gidx))
            ii = np.concatenate([gidx[j] for j in sel])
            ab[t] = auroc(y[ii], pred[ii]); wb[t] = np.nanmean(w[sel])
        print(f"{nm:<42} {a:>7.4f} [{np.percentile(ab,2.5):.4f},{np.percentile(ab,97.5):.4f}] "
              f"{ap:>7.4f} {wm:>9.4f} [{np.percentile(wb,2.5):.4f},{np.percentile(wb,97.5):.4f}]")
        fh.write(f"model\t{nm}\t{a:.5f}\t{wm:.5f}\t{np.percentile(ab,2.5):.5f}\t"
                 f"{np.percentile(ab,97.5):.5f}\t\t\t\t\n")
        for f, bt, lo, hi, fl in coefs(X, names):
            print(f"      {f:<32} {bt:>+8.4f} [{lo:>+8.4f},{hi:>+8.4f}]  flip {fl:.3f}")
            fh.write(f"coef\t{nm}|{f}\t\t\t\t\t{bt:.5f}\t{lo:.5f}\t{hi:.5f}\t{fl:.4f}\n")
log("DONE_U1_SINGLE")
