#!/usr/bin/env python3
# three questions about the binned blocks, on the corrected analysis set
import csv, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, HOLD_FRAC, RIDGE, NBOOT = 42, 0.20, 1.0, 200
rng = np.random.default_rng(SEED)
K3 = ["spl_upstream_max_me3", "spl_pas_prox_u1", "spl_u1_suppression_up"]
BINW, NBIN, WSTART = 100, 25, -2000

D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
ROW = np.load("corr_rowmask.npz")["ROW"]
B = np.load("binned_features.npz"); BN = [str(s) for s in B["names"]]; BX = B["X"][ridx].astype(float)
SPL = [i for i, n in enumerate(BN) if not n.endswith("_gc")]
GCB = [i for i, n in enumerate(BN) if n.endswith("_gc")]
SPLN = [BN[i] for i in SPL]; GCN = [BN[i] for i in GCB]
X3 = np.column_stack([D[n].to_numpy(float) for n in K3])
gi = defaultdict(list)
for i in range(len(gene)): gi[gene[i]].append(i)
gi = {g: np.array(v) for g, v in gi.items()}
allg = np.array(sorted(set(gene)))
hold_g = set(rng.choice(allg, size=int(round(HOLD_FRAC * len(allg))), replace=False))
is_hold = np.array([g in hold_g for g in gene])
CHROMS = sorted(set(chrom))
IDX = np.where(ROW)[0]
DEV = np.where(ROW & ~is_hold)[0]
def bincentre(nm):
    b = int(nm.split("_")[0][3:]); return WSTART + BINW * (b - 1) + BINW / 2

def imp(X, ref=None):
    ref = X if ref is None else ref
    mu = np.nanmean(ref, 0); mu = np.where(np.isfinite(mu), mu, 0.0)
    return np.where(np.isfinite(X), X, mu)
def zs(X):
    m = X.mean(0); s = X.std(0); s[s == 0] = 1.0
    return (X - m) / s

G = zs(imp(BX[np.ix_(IDX, GCB)])); S = zs(imp(BX[np.ix_(IDX, SPL)]))
def whiten(X, tol=1e-8):
    U, sv, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, sv > sv[0] * tol]
Qg, Qs = whiten(G), whiten(S)
cc = np.linalg.svd(Qg.T @ Qs, compute_uv=False)
def block_r2(A, Bm):
    Bm1 = np.column_stack([np.ones(len(Bm)), Bm])
    beta = np.linalg.lstsq(Bm1, A, rcond=None)[0]
    res = A - Bm1 @ beta
    return 1.0 - (res ** 2).sum(0) / ((A - A.mean(0)) ** 2).sum(0)
r2_g = block_r2(G, S); r2_s = block_r2(S, G)
print("\n===== 1. BLOCK COLLINEARITY (25 GC bins vs 150 splice bins, %d peaks) =====" % len(IDX))
print("canonical correlations, first 8: " + " ".join(f"{v:.3f}" for v in cc[:8]))
print(f"                       number above 0.9: {int((cc>0.9).sum())} of {len(cc)}; "
      f"above 0.5: {int((cc>0.5).sum())}; median {np.median(cc):.3f}")
print(f"GC bins explained by the splice block:  mean R2 {r2_g.mean():.3f}, "
      f"median {np.median(r2_g):.3f}, range [{r2_g.min():.3f},{r2_g.max():.3f}]")
print(f"splice bins explained by the GC block:  mean R2 {r2_s.mean():.3f}, "
      f"median {np.median(r2_s):.3f}, range [{r2_s.min():.3f},{r2_s.max():.3f}]")
def pear(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
STATS = ["me3_max", "me3_mean", "me5_max", "me5_mean", "u1_max", "u1_mean"]
print("\nper-bin correlation of GC with the co-located splice features:")
print(f"{'stat':<10} " + " ".join(f"{s:>7}" for s in ["min", "median", "max"]))
CO = {}
with open("block_collinearity.tsv", "w") as fh:
    fh.write("quantity\tindex\tvalue\n")
    for i, v in enumerate(cc): fh.write(f"canonical_correlation\t{i+1}\t{v:.5f}\n")
    for i, v in enumerate(r2_g): fh.write(f"r2_gcbin_on_splice\t{GCN[i]}\t{v:.5f}\n")
    for i, v in enumerate(r2_s): fh.write(f"r2_splicebin_on_gc\t{SPLN[i]}\t{v:.5f}\n")
    for st in STATS:
        vals = []
        for b in range(1, NBIN + 1):
            gj = GCN.index(f"bin{b}_gc"); sj = SPLN.index(f"bin{b}_{st}")
            m = np.isfinite(BX[IDX, GCB[gj]]) & np.isfinite(BX[IDX, SPL[sj]])
            r = pear(BX[IDX, GCB[gj]][m], BX[IDX, SPL[sj]][m])
            vals.append(r); fh.write(f"colocated_r\t{st}|bin{b}\t{r:.5f}\n")
        CO[st] = np.array(vals)
        print(f"{st:<10} " + " ".join(f"{v:>+7.3f}" for v in [CO[st].min(), np.median(CO[st]), CO[st].max()]))

# shared model machinery
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
        A, Bm = prep(X[tr], X[te]); pred[te] = Bm @ fit_ridge(A, y[tr])
    return pred
def per_gene(pred):
    hits, rrs, ns, gl = [], [], [], []
    for g, ix in gi.items():
        if not is_hold[ix[0]]: continue
        ix = ix[ROW[ix]]
        if len(ix) < 2 or (y[ix] == 1).sum() != 1: continue
        s = pred[ix]
        if not np.isfinite(s).all(): continue
        n = len(ix); j = np.argsort(-(s + rng.random(n) * 1e-12), kind="mergesort")
        srt = s[j]; rk = np.empty(n); i = 0
        while i < n:
            k = i
            while k + 1 < n and srt[k + 1] == srt[i]: k += 1
            rk[i:k + 1] = (i + k) / 2.0 + 1.0; i = k + 1
        r = np.empty(n); r[j] = rk
        c = int(np.argmax(y[ix]))
        hits.append(1.0 if j[0] == c else 0.0); rrs.append(1.0 / r[c]); ns.append(n); gl.append((g, ix))
    return np.array(hits), np.array(rrs), np.array(ns, float), gl
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

def dev_auroc(col):
    v = BX[DEV, col] if isinstance(col, (int, np.integer)) else col[DEV]
    m = np.isfinite(v)
    return auroc(y[DEV][m], v[m])
rank_gc = sorted(GCB, key=lambda j: -abs(dev_auroc(j) - 0.5))
rank_spl = sorted(SPL, key=lambda j: -abs(dev_auroc(j) - 0.5))
log("dev ranking done")

VAR = {
    "splice-binned (150)": BX[:, SPL],
    "GC-binned (25)": BX[:, GCB],
    "k3 splicing (3)": X3,
    "best-3 GC bins": BX[:, rank_gc[:3]],
    "best-25 splice bins": BX[:, rank_spl[:25]],
    "union: k3 + splice + GC (178)": np.column_stack([X3, BX[:, SPL], BX[:, GCB]]),
    "best-25 splice + 25 GC (50)": np.column_stack([BX[:, rank_spl[:25]], BX[:, GCB]]),
}
PRED = {nm: loco_hold(X) for nm, X in VAR.items()}
log("matched-count fits done")
_, _, _, GL0 = per_gene(PRED["k3 splicing (3)"])
gset = set(g for g, _ in GL0)
def ev(pred):
    h, rr, ns, gl = per_gene(pred)
    keep = [i for i, (g, _) in enumerate(gl) if g in gset]
    return h[keep], rr[keep], ns[keep], [gl[i] for i in keep]
h0, r0, n0, gl0 = ev(PRED["k3 splicing (3)"])
BOOT = rng.integers(0, len(gl0), size=(NBOOT, len(gl0)))
RES = {}
for nm in VAR:
    h, rr, ns, gl = ev(PRED[nm])
    rows = np.concatenate([ix for _, ix in gl]); gidx = [ix for _, ix in gl]
    hb = np.empty(NBOOT); ab = np.empty(NBOOT)
    for t in range(NBOOT):
        sel = BOOT[t]; ii = np.concatenate([gidx[j] for j in sel])
        hb[t] = h[sel].mean(); ab[t] = auroc(y[ii], PRED[nm][ii])
    RES[nm] = dict(nf=VAR[nm].shape[1], p1=h.mean(), hb=hb, mrr=rr.mean(),
                   auroc=auroc(y[rows], PRED[nm][rows]), ab=ab,
                   auprc=auprc(y[rows], PRED[nm][rows]),
                   chance=float(np.mean(1.0 / ns)))
print("\n===== 2. MATCHED FEATURE COUNT (holdout, %d genes, chance P@1 %.4f) =====" %
      (len(gl0), RES["k3 splicing (3)"]["chance"]))
print(f"{'variant':<30} {'feat':>5} {'P@1':>7} {'95% CI':>17} {'MRR':>7} {'AUROC':>7} {'AUPRC':>7}")
with open("block_matched.tsv", "w") as fh:
    fh.write("variant\tn_feat\tp1\tp1_lo\tp1_hi\tmrr\tauroc\tauroc_lo\tauroc_hi\tauprc\tchance\n")
    for nm, r in RES.items():
        print(f"{nm:<30} {r['nf']:>5} {r['p1']:>7.4f} [{np.percentile(r['hb'],2.5):.4f},"
              f"{np.percentile(r['hb'],97.5):.4f}] {r['mrr']:>7.4f} {r['auroc']:>7.4f} {r['auprc']:>7.4f}")
        fh.write(f"{nm}\t{r['nf']}\t{r['p1']:.5f}\t{np.percentile(r['hb'],2.5):.5f}\t"
                 f"{np.percentile(r['hb'],97.5):.5f}\t{r['mrr']:.5f}\t{r['auroc']:.5f}\t"
                 f"{np.percentile(r['ab'],2.5):.5f}\t{np.percentile(r['ab'],97.5):.5f}\t"
                 f"{r['auprc']:.5f}\t{r['chance']:.5f}\n")
    print("\npaired increments of one block over the other (same genes, same resamples):")
    for ref, add in [("GC-binned (25)", "union: k3 + splice + GC (178)"),
                     ("splice-binned (150)", "union: k3 + splice + GC (178)"),
                     ("best-25 splice bins", "best-25 splice + 25 GC (50)"),
                     ("GC-binned (25)", "best-25 splice + 25 GC (50)"),
                     ("k3 splicing (3)", "best-3 GC bins")]:
        a, b = RES[ref], RES[add]
        dp = b["p1"] - a["p1"]; dpb = b["hb"] - a["hb"]
        da = b["auroc"] - a["auroc"]; dab = b["ab"] - a["ab"]
        print(f"  {add:<32} over {ref:<22} dP@1 {dp:>+7.4f} [{np.percentile(dpb,2.5):+.4f},"
              f"{np.percentile(dpb,97.5):+.4f}]  dAUROC {da:>+7.4f} [{np.percentile(dab,2.5):+.4f},"
              f"{np.percentile(dab,97.5):+.4f}]")
        fh.write(f"#delta\t{add}\tover\t{ref}\t{dp:.5f}\t{np.percentile(dpb,2.5):.5f}\t"
                 f"{np.percentile(dpb,97.5):.5f}\t{da:.5f}\t{np.percentile(dab,2.5):.5f}\t"
                 f"{np.percentile(dab,97.5):.5f}\n")

print("\n===== 3. PER-BIN UNIVARIATE AUROC (all analysis rows) =====")
with open("block_perbin.tsv", "w") as fh:
    fh.write("feature\tscorer\tbin\tcentre_nt\tauroc\tmean_canonical\tmean_competitor\n")
    yv = y[IDX]
    for st in ["gc"] + STATS:
        vals = []
        for b in range(1, NBIN + 1):
            nm = f"bin{b}_gc" if st == "gc" else f"bin{b}_{st}"
            j = BN.index(nm); v = BX[IDX, j]; m = np.isfinite(v)
            a = auroc(yv[m], v[m])
            mc = float(v[m & (yv == 1)].mean()); mk = float(v[m & (yv == 0)].mean())
            vals.append((b, bincentre(nm), a, mc, mk))
            fh.write(f"{nm}\t{st}\t{b}\t{bincentre(nm):.0f}\t{a:.5f}\t{mc:.5f}\t{mk:.5f}\n")
        top = sorted(vals, key=lambda t: -abs(t[2] - 0.5))[:3]
        print(f"  {st:<9} top 3 bins by |AUROC-0.5|: " +
              ", ".join(f"{int(t[1]):+5d} nt AUROC {t[2]:.3f} (canonical {t[3]:.3f} vs competitor {t[4]:.3f})"
                        if st == "gc" else f"{int(t[1]):+5d} nt AUROC {t[2]:.3f}" for t in top))
    print("\n  cumulative holdout performance using only the best GC bins (ranked on dev):")
    for k in [1, 3, 5, 10, 25]:
        p = loco_hold(BX[:, rank_gc[:k]])
        h, rr, ns, gl = ev(p)
        rows = np.concatenate([ix for _, ix in gl])
        nmb = ", ".join(BN[j].replace("_gc", "") for j in rank_gc[:min(k, 3)])
        print(f"    best {k:>2} GC bins: P@1 {h.mean():.4f} MRR {rr.mean():.4f} AUROC {auroc(y[rows], p[rows]):.4f}"
              + (f"   ({nmb}{', ...' if k > 3 else ''})" if k <= 10 else ""))
        fh.write(f"#cum_gc\t{k}\t{h.mean():.5f}\t{rr.mean():.5f}\t{auroc(y[rows], p[rows]):.5f}\n")
log("DONE_BLOCK_ANALYSIS")
