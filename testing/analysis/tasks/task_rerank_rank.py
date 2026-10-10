#!/usr/bin/env python3
# rank of the true 3' end within its gene, primary 3-feature reranker (loco, all genes)
import csv, numpy as np, pandas as pd
from collections import defaultdict
SEED, RIDGE = 42, 1.0
FINAL = ["spl_upstream_max_me3", "spl_pas_prox_u1", "spl_u1_suppression_up"]
rng = np.random.default_rng(SEED)

D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
X = np.column_stack([D[n].to_numpy(float) for n in FINAL])
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
UNANNOT = (np.load("atlas_peak_cache.npz")["dist"][ridx] > 500) & \
          (np.load("polyadb_peak_cache.npz")["dist"][ridx] > 500)
gi = defaultdict(list)
for i in range(len(gene)): gi[gene[i]].append(i)
gi = {g: np.array(v) for g, v in gi.items()}
ROW = np.zeros(len(y), bool)
for g, ix in gi.items():
    sub = ix[(y[ix] == 1) | ((y[ix] == 0) & UNANNOT[ix])]
    if (y[sub] == 1).sum() == 1 and (y[sub] == 0).sum() >= 1: ROW[sub] = True

def fit_ridge(Xt, yt, lam=RIDGE):
    P = np.eye(Xt.shape[1]) * lam; P[0, 0] = 0.0; b = np.zeros(Xt.shape[1])
    for _ in range(80):
        eta = np.clip(Xt @ b, -30, 30); mu = 1 / (1 + np.exp(-eta)); w = np.maximum(mu * (1 - mu), 1e-9)
        z = eta + (yt - mu) / w; XtW = Xt.T * w
        nb = np.linalg.solve(XtW @ Xt + P, XtW @ z + P @ b)
        if np.max(np.abs(nb - b)) < 1e-9: b = nb; break
        b = nb
    return b
pred = np.full(len(y), np.nan)
for c in sorted(set(chrom)):
    te = ROW & (chrom == c); tr = ROW & (chrom != c)
    if te.sum() == 0 or tr.sum() == 0: continue
    mu = np.nanmean(X[tr], 0); sd = np.nanstd(X[tr], 0); sd[sd == 0] = 1.0
    A = (np.where(np.isfinite(X[tr]), X[tr], mu) - mu) / sd
    B = (np.where(np.isfinite(X[te]), X[te], mu) - mu) / sd
    pred[te] = np.column_stack([np.ones(len(B)), B]) @ fit_ridge(np.column_stack([np.ones(len(A)), A]), y[tr])

def cat(r, n): return "last" if r == n else ("1" if r == 1 else "2" if r == 2 else "3" if r == 3 else "4+")
CATS = ["1", "2", "3", "4+", "last"]
obs = defaultdict(float); exp = defaultdict(float); ranks = []; ns = []
for g, ix in gi.items():
    ix = ix[ROW[ix]]
    if len(ix) < 2: continue
    s = pred[ix]
    if not np.isfinite(s).all(): continue
    n = len(ix); j = np.argsort(-(s + rng.random(n) * 1e-12), kind="mergesort")
    r = np.empty(n, int); r[j] = np.arange(1, n + 1)
    c = int(np.argmax(y[ix])); ranks.append(r[c]); ns.append(n)
    obs[cat(r[c], n)] += 1
    for rr in range(1, n + 1): exp[cat(rr, n)] += 1.0 / n
NG = len(ranks); ranks = np.array(ranks); ns = np.array(ns)
print(f"\n===== RANK OF THE TRUE 3' END, primary set, {NG} genes (LOCO) =====")
print(f"{'category':<8} {'genes':>7} {'observed':>9} {'chance':>9} {'ratio':>7}")
with open("rerank_pri_rank.tsv", "w") as fh:
    fh.write("category\tn_genes\tobserved\tchance\n")
    for c in CATS:
        o = obs[c] / NG; e = exp[c] / NG
        print(f"{c:<8} {int(obs[c]):>7} {o:>9.4f} {e:>9.4f} {o/e if e else float('nan'):>7.2f}")
        fh.write(f"{c}\t{int(obs[c])}\t{o:.5f}\t{e:.5f}\n")
    top3 = float(np.mean(ranks <= 3)); e3 = float(np.mean(np.minimum(3, ns) / ns))
    med = float(np.median(ranks)); pct = float(np.mean((ranks - 1) / np.maximum(ns - 1, 1)))
    fh.write(f"#top3\t{top3:.5f}\t{e3:.5f}\n#median_rank\t{med:.3f}\n#mean_pctile\t{pct:.5f}\n"
             f"#median_n\t{float(np.median(ns)):.1f}\n#n_genes\t{NG}\n")
print(f"\nrank <= 3: {top3:.4f} observed vs {e3:.4f} chance; median rank {med:.1f} "
      f"(median candidates {np.median(ns):.0f}); mean position {pct:.3f} of the way down the list")
mis = ranks > 1
print(f"misses only ({int(mis.sum())} genes): rank <= 3 {np.mean(ranks[mis]<=3):.4f}, "
      f"last {np.mean(ranks[mis]==ns[mis]):.4f}, median rank {np.median(ranks[mis]):.1f}")
