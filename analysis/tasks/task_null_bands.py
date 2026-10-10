#!/usr/bin/env python3
# the pooled-auroc permutation null is not 0.5
import csv, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, HOLD_FRAC, RIDGE = 42, 0.20, 1.0
NPERM, NPERM_REFIT = 1000, 50
rng = np.random.default_rng(SEED)

M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
rf = D.rf_prob.to_numpy(float)
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
ROW = np.load("upstream_rowmask.npz")["ROW"]
GC = np.load("termexon_gc_cache.npz")["gc500"][ridx]
PB = np.load("binned_features178.npz"); BN = [str(s) for s in PB["names"]]; BX = PB["X"][ridx].astype(float)
GCB = [BN.index(f"bin{b}_gc") for b in range(1, 26)]
SPL = [i for i, n in enumerate(BN) if n.startswith("bin") and not n.endswith("_gc")]
me3 = D["spl_upstream_max_me3"].to_numpy(float)
uprox = D["spl_pas_prox_u1"].to_numpy(float)
umean = D["spl_upstream_mean_u1"].to_numpy(float)
u1c = (6 * umean - uprox) / 5.0 - uprox
X2 = np.column_stack([me3, u1c])
Z = np.load("anchor_sweep.npz", allow_pickle=True)
adist = Z["X"][:, list(Z["names"]).index("anchor_dist")]

gi = defaultdict(list)
for i in range(len(gene)): gi[gene[i]].append(i)
gi = {g: np.array(v) for g, v in gi.items()}
allg = np.array(sorted(set(gene)))
hold_g = set(rng.choice(allg, size=int(round(HOLD_FRAC * len(allg))), replace=False))
is_hold = np.array([g in hold_g for g in gene])
CHROMS = sorted(set(chrom))
GEN = [(g, ix[ROW[ix]]) for g, ix in gi.items() if ROW[ix].sum() >= 2 and (y[ix[ROW[ix]]] == 1).sum() == 1]
HG = [(g, ix) for g, ix in GEN if is_hold[ix[0]]]
ALLROWS = np.concatenate([ix for _, ix in GEN]); HROWS = np.concatenate([ix for _, ix in HG])

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
def loco(X, yv=None, test_mask=None, train_mask=None):
    yv = y if yv is None else yv
    tm = np.ones(len(y), bool) if test_mask is None else test_mask
    rm = np.ones(len(y), bool) if train_mask is None else train_mask
    pred = np.full(len(y), np.nan)
    for c in CHROMS:
        te = ROW & tm & (chrom == c); tr = ROW & rm & (chrom != c)
        if te.sum() == 0 or tr.sum() == 0: continue
        A, B = prep(X[tr], X[te]); pred[te] = B @ fit_ridge(A, yv[tr])
    return pred

def midrank(s):
    o = np.argsort(s, kind="mergesort"); sv = s[o]; n = len(s); rk = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    rr = np.empty(n); rr[o] = rk
    return rr
def auroc_from_rank(rr, yv):
    P = yv.sum(); N = len(yv)
    return float((rr[yv == 1].sum() - P * (P + 1) / 2) / (P * (N - P)))

def null_for(score, G):
    gl = []
    for g, ix in G:
        f = ix[np.isfinite(score[ix])]
        if len(f) >= 2: gl.append(f)
    rows = np.concatenate(gl)
    rr = midrank(score[rows])
    N = len(rows); Gn = len(gl)
    obs = auroc_from_rank(rr, y[rows])
    ng = np.array([len(ix) for ix in gl]); off = np.r_[0, np.cumsum(ng)[:-1]]
    pick = off[None, :] + (rng.random((NPERM, Gn)) * ng[None, :]).astype(int)
    sums = rr[pick].sum(1)
    perm = (sums - Gn * (Gn + 1) / 2) / (Gn * (N - Gn))
    mr = np.array([rr[off[k]:off[k] + ng[k]].mean() for k in range(Gn)])
    vr = np.array([rr[off[k]:off[k] + ng[k]].var() for k in range(Gn)])
    ex = (mr.mean() - (Gn + 1) / 2) / (N - Gn)
    sd = np.sqrt(vr.sum()) / (Gn * (N - Gn))
    r_size = float(np.corrcoef(ng, mr)[0, 1])
    return dict(auroc=obs, n=N, genes=Gn, perm_mean=float(perm.mean()),
                lo=float(np.percentile(perm, 2.5)), hi=float(np.percentile(perm, 97.5)),
                exact=float(ex), exact_sd=float(sd), excess=obs - float(ex), r_size=r_size)
def winrate(v, G, yv=None):
    yv = y if yv is None else yv
    out = np.full(len(G), np.nan)
    for t, (g, ix) in enumerate(G):
        s = v[ix]; c = int(np.argmax(yv[ix])); o = np.delete(np.arange(len(ix)), c)
        if not np.isfinite(s[c]): continue
        d = s[o][np.isfinite(s[o])]
        if len(d): out[t] = float(np.mean((s[c] > d) + 0.5 * (s[c] == d)))
    return out
def winrate_null(score, G):
    W = []; sizes = []
    for _, ix in G:
        f = ix[np.isfinite(score[ix])]
        if len(f) < 2: continue
        w = (midrank(score[f]) - 1.0) / (len(f) - 1.0)
        W.append(w); sizes.append(len(f))
    flat = np.concatenate(W); ng = np.array(sizes); off = np.r_[0, np.cumsum(ng)[:-1]]
    pick = off[None, :] + (rng.random((NPERM, len(ng))) * ng[None, :]).astype(int)
    out = flat[pick].mean(1)
    return float(out.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

pred2 = loco(X2)
pred2h = loco(X2, test_mask=is_hold, train_mask=~is_hold)
predG = loco(BX[:, GCB])
mfin = np.isfinite(pred2) & np.isfinite(GC) & ROW
A1 = np.column_stack([np.ones(int(mfin.sum())), GC[mfin]])
bb = np.linalg.lstsq(A1, pred2[mfin], rcond=None)[0]
res = np.full(len(y), np.nan); res[mfin] = pred2[mfin] - A1 @ bb
BLOCKS = {"two-feature model (2)": X2, "GC-binned (25)": BX[:, GCB],
          "splice-binned (150)": BX[:, SPL],
          "union (177)": np.column_stack([X2, BX[:, SPL], BX[:, GCB]])}
BPRED = {nm: loco(X, test_mask=is_hold, train_mask=~is_hold) for nm, X in BLOCKS.items()}

ITEMS = ([("holdout block", nm, BPRED[nm], HG) for nm in BLOCKS]
         + [("all rows", "two-feature model", pred2, GEN),
            ("all rows", "model, GC removed", res, GEN),
            ("all rows", "GC, 25 binned profile", predG, GEN),
            ("all rows", "GC, single 500 nt window", GC, GEN),
            ("all rows", "baseline model (rf_prob)", rf, GEN),
            ("feature", "spl_upstream_max_me3", me3, GEN),
            ("feature", "u1_contrast", u1c, GEN),
            ("feature", "spl_pas_prox_u1", uprox, GEN),
            ("feature", "spl_upstream_mean_u1", umean, GEN),
            ("feature", "anchor distance to pA", adist, GEN)])

print(f"\n===== POOLED-AUROC PERMUTATION NULL, {NPERM} permutations =====")
print(f"scheme: one positive per gene drawn uniformly, score held fixed; "
      f"{len(GEN):,} genes, {len(ALLROWS):,} sites (holdout {len(HG):,} genes, {len(HROWS):,} sites)")
print(f"\n{'set':<14} {'score':<26} {'AUROC':>7} {'null 95% band':>17} {'exact':>7} {'excess':>8} "
      f"{'r(n_g, mean rank)':>18}")
fh = open("null_bands.tsv", "w")
fh.write("set\tscore\tauroc\tnull_lo\tnull_hi\tnull_mean\tnull_exact\tnull_exact_sd\texcess\t"
         "r_size_meanrank\tn_sites\tn_genes\twinrate\twin_null\twin_null_lo\twin_null_hi\n")
for st, nm, sc, G in ITEMS:
    d = null_for(sc, G)
    w = float(np.nanmean(winrate(sc, G)))
    wn, wlo, whi = winrate_null(sc, G)
    print(f"{st:<14} {nm:<26} {d['auroc']:>7.4f} [{d['lo']:.4f},{d['hi']:.4f}] {d['exact']:>7.4f} "
          f"{d['excess']:>+8.4f} {d['r_size']:>+18.3f}")
    fh.write(f"{st}\t{nm}\t{d['auroc']:.5f}\t{d['lo']:.5f}\t{d['hi']:.5f}\t{d['perm_mean']:.5f}\t"
             f"{d['exact']:.5f}\t{d['exact_sd']:.5f}\t{d['excess']:.5f}\t{d['r_size']:.4f}\t"
             f"{d['n']}\t{d['genes']}\t{w:.5f}\t{wn:.5f}\t{wlo:.5f}\t{whi:.5f}\n")
fh.close()

print(f"\n===== WIN-RATE NULL, same scheme, {NPERM} permutations =====")
T = pd.read_csv("null_bands.tsv", sep="\t")
for _, r in T.iterrows():
    print(f"{r['score']:<26} win-rate {r['winrate']:.4f}  null {r['win_null']:.4f} "
          f"[{r['win_null_lo']:.4f},{r['win_null_hi']:.4f}]  excess {r['winrate']-r['win_null']:+.4f}")

print(f"\n===== REFIT NULL, two-feature model, {NPERM_REFIT} refits =====")
gl = [ix for _, ix in GEN]
ref = np.empty(NPERM_REFIT)
for t in range(NPERM_REFIT):
    yp = np.zeros(len(y))
    for ix in gl: yp[ix[rng.integers(0, len(ix))]] = 1.0
    p = loco(X2, yv=yp)
    rr = midrank(p[ALLROWS]); ref[t] = auroc_from_rank(rr, yp[ALLROWS])
d2 = null_for(pred2, GEN)
print(f"score-fixed null {d2['exact']:.4f} (band [{d2['lo']:.4f},{d2['hi']:.4f}]); "
      f"refit null {ref.mean():.4f} [{np.percentile(ref,2.5):.4f},{np.percentile(ref,97.5):.4f}]")
with open("null_bands.tsv", "a") as f2:
    f2.write(f"refit\ttwo-feature model\t{d2['auroc']:.5f}\t{np.percentile(ref,2.5):.5f}\t"
             f"{np.percentile(ref,97.5):.5f}\t{ref.mean():.5f}\t{ref.mean():.5f}\t\t"
             f"{d2['auroc']-ref.mean():.5f}\t\t{d2['n']}\t{d2['genes']}\t\t\t\t\n")

print(f"\n===== PER CHROMOSOME, two-feature model =====")
with open("null_bychrom.tsv", "w") as f3:
    f3.write("chrom\tn_genes\tauroc\tci_lo\tci_hi\tnull\tnull_lo\tnull_hi\texcess\n")
    print(f"{'chrom':<8} {'AUROC':>7} {'null':>7} {'excess':>8}")
    for c in CHROMS:
        G = [(g, ix) for g, ix in GEN if chrom[ix[0]] == c]
        if len(G) < 20: continue
        d = null_for(pred2, G)
        gl2 = [ix for _, ix in G]
        b = np.empty(200)
        for t in range(200):
            ii = np.concatenate([gl2[j] for j in rng.integers(0, len(gl2), len(gl2))])
            rr = midrank(pred2[ii]); b[t] = auroc_from_rank(rr, y[ii])
        print(f"{c:<8} {d['auroc']:>7.4f} {d['exact']:>7.4f} {d['excess']:>+8.4f}")
        f3.write(f"{c}\t{len(G)}\t{d['auroc']:.5f}\t{np.percentile(b,2.5):.5f}\t{np.percentile(b,97.5):.5f}\t"
                 f"{d['exact']:.5f}\t{d['lo']:.5f}\t{d['hi']:.5f}\t{d['excess']:.5f}\n")
log("DONE_NULL_BANDS")
