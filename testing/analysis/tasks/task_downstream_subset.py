#!/usr/bin/env python3
# the subset where the positional rule cannot work
import csv, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, RIDGE, NBOOT, NPERM = 42, 1.0, 2000, 2000
rng = np.random.default_rng(SEED)
K3 = ["spl_upstream_max_me3", "spl_pas_prox_u1", "spl_u1_suppression_up"]

M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
cen = D.center.to_numpy().astype(float); strand = D.strand.to_numpy()
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
ROW = np.load("corr_rowmask.npz")["ROW"]
PB = np.load("binned_features178.npz"); BN = [str(s) for s in PB["names"]]; BX = PB["X"][ridx].astype(float)
GCB = [BN.index(f"bin{b}_gc") for b in range(1, 26)]
SPL = [i for i, n in enumerate(BN) if n.startswith("bin") and not n.endswith("_gc")]
X3 = np.column_stack([D[n].to_numpy(float) for n in K3])
XFULL = np.column_stack([X3, BX[:, SPL], BX[:, GCB]])
tpos = np.where(strand == "+", cen, -cen)

gi = defaultdict(list)
for i in range(len(gene)): gi[gene[i]].append(i)
gi = {g: np.array(v) for g, v in gi.items()}
CHROMS = sorted(set(chrom))
GEN = []
for g, ix in gi.items():
    sub = ix[ROW[ix]]
    if len(sub) < 2 or (y[sub] == 1).sum() != 1: continue
    c = sub[np.argmax(y[sub])]
    dn = sub[(y[sub] == 0) & (tpos[sub] > tpos[c])]
    if len(dn): GEN.append((g, sub, c, dn))
NS = np.array([len(s) for _, s, _, _ in GEN], float)
NDN = np.array([len(d) for _, _, _, d in GEN], float)
CH = float(np.mean(1.0 / NS))
perm = np.array([np.mean([float(rng.integers(0, int(n)) == 0) for n in NS]) for _ in range(500)])
print(f"\n===== DOWNSTREAM-COMPETITOR SUBSET =====")
print(f"genes {len(GEN)} of 4923 ({len(GEN)/4923:.3f}); candidates {int(NS.sum())}; "
      f"median candidates/gene {np.median(NS):.0f}; median downstream competitors/gene {np.median(NDN):.0f}")
print(f"chance P@1 analytic {CH:.4f}; within-gene permutation {perm.mean():.4f} "
      f"[{np.percentile(perm,2.5):.4f},{np.percentile(perm,97.5):.4f}]")
print("most-downstream rule P@1 0.0000 by construction (the last candidate is a competitor here)")

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
def loco(X):
    pred = np.full(len(y), np.nan)
    for c in CHROMS:
        te = ROW & (chrom == c); tr = ROW & (chrom != c)
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
BOOT = rng.integers(0, len(GEN), size=(NBOOT, len(GEN)))
def ci(v):
    b = np.nanmean(v[BOOT], axis=1)
    return float(np.nanmean(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

MODELS = {"three features": X3, "GC-binned (25)": BX[:, GCB], "full model (178)": XFULL}
print(f"\n{'predictor':<20} {'P@1':>7} {'95% CI':>17} {'excess':>17} {'AUROC':>7} "
      f"{'win-rate':>9} {'95% CI':>17} {'pair win-rate':>14} {'95% CI':>17}")
with open("downstream_subset.tsv", "w") as fh:
    fh.write("predictor\tn_genes\tp1\tp1_lo\tp1_hi\tchance\texcess\texcess_lo\texcess_hi\tauroc\t"
             "winrate\twin_lo\twin_hi\tpair_winrate\tpair_lo\tpair_hi\tn_pairs\n")
    for nm, X in MODELS.items():
        pred = loco(X)
        hit = np.empty(len(GEN)); wr = np.empty(len(GEN)); pw = np.empty(len(GEN)); npair = 0
        for t, (g, sub, c, dn) in enumerate(GEN):
            s = pred[sub]
            mx = s.max(); top = np.where(s == mx)[0]
            ci_ = int(np.where(sub == c)[0][0])
            hit[t] = (1.0 / len(top)) if ci_ in top else 0.0
            o = np.delete(np.arange(len(sub)), ci_)
            wr[t] = float(np.mean((s[ci_] > s[o]) + 0.5 * (s[ci_] == s[o])))
            sd_ = pred[dn]
            pw[t] = float(np.mean((pred[c] > sd_) + 0.5 * (pred[c] == sd_))); npair += len(dn)
        rows = np.concatenate([sub for _, sub, _, _ in GEN])
        m, lo, hi = ci(hit); e, elo, ehi = ci(hit - 1.0 / NS)
        wm, wlo, whi = ci(wr); pm, plo, phi = ci(pw)
        au = auroc(y[rows], pred[rows])
        print(f"{nm:<20} {m:>7.4f} [{lo:.4f},{hi:.4f}] {e:>+8.4f}[{elo:+.3f},{ehi:+.3f}] {au:>7.4f} "
              f"{wm:>9.4f} [{wlo:.4f},{whi:.4f}] {pm:>14.4f} [{plo:.4f},{phi:.4f}]")
        fh.write(f"{nm}\t{len(GEN)}\t{m:.5f}\t{lo:.5f}\t{hi:.5f}\t{CH:.5f}\t{e:.5f}\t{elo:.5f}\t{ehi:.5f}\t"
                 f"{au:.5f}\t{wm:.5f}\t{wlo:.5f}\t{whi:.5f}\t{pm:.5f}\t{plo:.5f}\t{phi:.5f}\t{npair}\n")
    print(f"\nchance: P@1 {CH:.4f}, AUROC 0.5000, win-rate 0.5000, pair win-rate 0.5000")
    print(f"canonical-versus-downstream-competitor pairs: {npair}")
    fh.write(f"#chance\t{CH:.5f}\t#pairs\t{npair}\n")

pred3 = loco(X3)
pairs = []
for g, sub, c, dn in GEN:
    for d in dn: pairs.append((g, float(pred3[c] > pred3[d]) + 0.5 * float(pred3[c] == pred3[d])))
pg = defaultdict(list)
for t, (g, v) in enumerate(pairs): pg[g].append(t)
GKEY = list(pg); GI = [np.array(pg[g]) for g in GKEY]
pv = np.array([v for _, v in pairs])
bb = np.array([pv[np.concatenate([GI[j] for j in rng.integers(0, len(GI), len(GI))])].mean()
               for _ in range(1000)])
print(f"\nthree features, pair-level win-rate over {len(pv)} canonical-vs-downstream pairs: "
      f"{pv.mean():.4f} [{np.percentile(bb,2.5):.4f},{np.percentile(bb,97.5):.4f}] "
      f"(gene-clustered bootstrap; chance 0.5000)")
log("DONE_DOWNSTREAM_SUBSET")
