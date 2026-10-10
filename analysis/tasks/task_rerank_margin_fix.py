#!/usr/bin/env python3
# recompute the score-margin panel data for the final 4-feature reranker
import csv, numpy as np, pandas as pd
from collections import defaultdict
SEED, HOLD_FRAC, RIDGE = 42, 0.20, 1.0
FINAL_NAMES = ["scan_n3ss_dn5k", "spl_upstream_max_me3", "spl_pas_prox_u1", "spl_n_5ss_1k"]
rng = np.random.default_rng(SEED)

D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
S = np.load("splice_feats_cache.npz")
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
col = {"scan_n3ss_dn5k": S["C_n3ss_dn5k"][ridx]}
for n in FINAL_NAMES:
    if n not in col: col[n] = D[n].to_numpy(float)
X = np.column_stack([col[n] for n in FINAL_NAMES])

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
    te = chrom == c; tr = ~te
    mu = np.nanmean(X[tr], 0); sd = np.nanstd(X[tr], 0); sd[sd == 0] = 1.0
    A = (np.where(np.isfinite(X[tr]), X[tr], mu) - mu) / sd
    B = (np.where(np.isfinite(X[te]), X[te], mu) - mu) / sd
    pred[te] = np.column_stack([np.ones(len(B)), B]) @ fit_ridge(np.column_stack([np.ones(len(A)), A]), y[tr])

gi = defaultdict(list)
for i in range(len(gene)): gi[gene[i]].append(i)
rows = []
for g, ix in gi.items():
    ix = np.array(ix); s = pred[ix]
    if len(ix) < 2 or not np.isfinite(s).all(): continue
    c = int(np.argmax(y[ix])); top = int(np.argmax(s + rng.random(len(ix)) * 1e-12))
    rngspan = s.max() - s.min(); sdv = np.std(s)
    frac = 0.0 if rngspan <= 0 else float((s[top] - s[c]) / rngspan)
    rank = int((s > s[c]).sum()) + 1
    rows.append((g, 1 if top == c else 0, len(ix), float((s[top] - s[c]) / (sdv if sdv > 0 else 1.0)),
                 frac, rank / len(ix)))
with open("rerank_min_margin.tsv", "w") as fh:
    fh.write("gene\thit\tn_candidates\tmargin_sd\tmargin_frac\trank_pct\n")
    for r in rows: fh.write("%s\t%d\t%d\t%.5f\t%.5f\t%.5f\n" % r)
mf = np.array([r[4] for r in rows if r[1] == 0])
print(f"misses {len(mf)}: median range-normalised gap {np.median(mf):.3f}; "
      f"{100*np.mean(mf<=0.10):.1f}% within 10% of the top, {100*np.mean(mf<=0.25):.1f}% within 25%")
