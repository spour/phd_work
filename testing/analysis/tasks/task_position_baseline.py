#!/usr/bin/env python3
# blocking check: do trivial positional rules beat the three-feature model?
import csv, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, HOLD_FRAC, RIDGE, NBOOT = 42, 0.20, 1.0, 2000
rng = np.random.default_rng(SEED)
K3 = ["spl_upstream_max_me3", "spl_pas_prox_u1", "spl_u1_suppression_up"]

M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
cen = D.center.to_numpy().astype(float); strand = D.strand.to_numpy()
ROW = np.load("corr_rowmask.npz")["ROW"]
X3 = np.column_stack([D[n].to_numpy(float) for n in K3])
gi = defaultdict(list)
for i in range(len(gene)): gi[gene[i]].append(i)
gi = {g: np.array(v) for g, v in gi.items()}
allg = np.array(sorted(set(gene)))
hold_g = set(rng.choice(allg, size=int(round(HOLD_FRAC * len(allg))), replace=False))
is_hold = np.array([g in hold_g for g in gene])
CHROMS = sorted(set(chrom))
ALL = np.ones(len(y), bool)

tpos = np.where(strand == "+", cen, -cen)
rank3 = np.full(len(y), np.nan)
for g, ix in gi.items():
    sub = ix[ROW[ix]]
    if len(sub) < 2: continue
    o = np.argsort(-tpos[sub], kind="mergesort")
    r = np.empty(len(sub)); r[o] = np.arange(1, len(sub) + 1)
    rank3[sub] = r

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
GENES = [(g, ix[ROW[ix]]) for g, ix in gi.items() if ROW[ix].sum() >= 2 and (y[ix[ROW[ix]]] == 1).sum() == 1]
NS = np.array([len(ix) for _, ix in GENES], float)
CH = float(np.mean(1.0 / NS))
def p1(score):
    out = np.empty(len(GENES))
    for t, (g, ix) in enumerate(GENES):
        s = score[ix]
        if not np.isfinite(s).all(): out[t] = np.nan; continue
        mx = s.max(); top = np.where(s == mx)[0]
        c = int(np.argmax(y[ix]))
        out[t] = (1.0 / len(top)) if c in top else 0.0
    return out
BOOT = rng.integers(0, len(GENES), size=(NBOOT, len(GENES)))
def ci(v):
    b = np.nanmean(v[BOOT], axis=1)
    return float(np.nanmean(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
perm = np.array([np.mean([1.0 / n for n in NS] * 0 + [float(rng.integers(0, int(n)) == 0) for n in NS])
                 for _ in range(500)])

SCORES = {}
SCORES["most-downstream rule"] = np.where(np.isfinite(rank3), -rank3, np.nan)
SCORES["most-upstream rule"] = np.where(np.isfinite(rank3), rank3, np.nan)
SCORES["position model (rank from 3' end)"] = loco(rank3[:, None])
SCORES["three features"] = loco(X3)
SCORES["three features + position"] = loco(np.column_stack([X3, rank3]))
SCORES["chance (constant score)"] = np.zeros(len(y))

print(f"\n===== POSITIONAL BASELINES, corrected set: {len(GENES)} genes, "
      f"{int((ROW&(y==0)).sum())} competitors =====")
print(f"analytic chance {CH:.4f}; within-gene permutation {perm.mean():.4f} "
      f"[{np.percentile(perm,2.5):.4f},{np.percentile(perm,97.5):.4f}]; "
      f"median candidates {np.median(NS):.0f}")
print(f"{'predictor':<36} {'P@1':>7} {'95% CI':>17} {'excess':>8}")
RES = {}
with open("position_baseline.tsv", "w") as fh:
    fh.write("predictor\tp1\tci_lo\tci_hi\tchance\texcess\texcess_lo\texcess_hi\tn_genes\n")
    for nm in ["chance (constant score)", "most-upstream rule", "most-downstream rule",
               "position model (rank from 3' end)", "three features", "three features + position"]:
        v = p1(SCORES[nm]); RES[nm] = v
        m, lo, hi = ci(v); e, elo, ehi = ci(v - 1.0 / NS)
        print(f"{nm:<36} {m:>7.4f} [{lo:.4f},{hi:.4f}] {e:>+8.4f}")
        fh.write(f"{nm}\t{m:.5f}\t{lo:.5f}\t{hi:.5f}\t{CH:.5f}\t{e:.5f}\t{elo:.5f}\t{ehi:.5f}\t{len(GENES)}\n")
    print("\npaired increments (same genes, same 2,000 gene resamples):")
    for a, b in [("three features + position", "position model (rank from 3' end)"),
                 ("three features + position", "three features"),
                 ("three features", "most-downstream rule"),
                 ("position model (rank from 3' end)", "most-downstream rule")]:
        d = RES[a] - RES[b]; m, lo, hi = ci(d)
        print(f"  {a:<36} over {b:<36} {m:>+8.4f} [{lo:+.4f},{hi:+.4f}]")
        fh.write(f"#delta\t{a}\tover\t{b}\t{m:.5f}\t{lo:.5f}\t{hi:.5f}\n")
GEN0 = GENES
v0 = np.nanmean(p1(SCORES["three features"]))
c0 = np.nanmean(p1(SCORES["chance (constant score)"]))
GENES = [(g, ix[rng.permutation(len(ix))]) for g, ix in GEN0]
v1 = np.nanmean(p1(SCORES["three features"]))
c1 = np.nanmean(p1(SCORES["chance (constant score)"]))
GENES = GEN0
print(f"\nrow-order shuffle check (candidate order permuted inside every gene):")
print(f"  three features P@1 {v0:.6f} -> {v1:.6f};  constant score {c0:.6f} -> {c1:.6f} "
      f"(analytic chance {CH:.6f})")
log("DONE_POSITION_BASELINE")
