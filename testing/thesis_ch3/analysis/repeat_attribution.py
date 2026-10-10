#!/usr/bin/env python3
# how much of the network's within-gene performance is attributable to repeats?
import os, sys, collections, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
def log(*a): print(*a, file=sys.stderr, flush=True)
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); y = D["y"].astype(int); g = D["gene"]
C = np.load(os.path.join(PUB, "cnn_loco_scores.npz"), allow_pickle=True); assert (C["gene"] == g).all()
L = C["logit"].astype(float)
R = np.genfromtxt(os.path.join(ROOT, "data/repeat_sites.tsv"), delimiter="\t", names=True, dtype=None, encoding=None)
assert len(R) == len(y)
cov = np.column_stack([R[n] for n in R.dtype.names if n.startswith("cov_")]).astype(float)
grp = np.array(R["site_group"], dtype=object); ori = np.array(R["site_orientation"]).astype(int)
site_cat = np.array([f"{a}|{o}" if a != "none" else "none" for a, o in zip(grp, ori)], dtype=object)
cats = sorted(set(site_cat) - {"none"})
site_oh = np.column_stack([(site_cat == c).astype(float) for c in cats])
in_rep = grp != "none"
Xrep = np.column_stack([cov, cov.sum(1, keepdims=True), site_oh])
W = np.load(os.path.join(PUB, "cnn_wg_scores.npz"), allow_pickle=True); S = np.load(os.path.join(ROOT, "data/stress_context.npz"), allow_pickle=True)
k = lambda a, b, c, d: (str(a), str(b), int(c), str(d))
wrow = {k(*t): i for i, t in enumerate(zip(W["gene"], W["chrom"], W["center"], W["strand"]))}
m = np.array([wrow[k(*t)] for t in zip(D["gene"], D["chrom"], D["center"], D["strand"])])
exo = S["exo"][m].astype(float)
Xboth = np.column_stack([Xrep, exo])
gidx = collections.defaultdict(list)
for i, gg in enumerate(g): gidx[gg].append(i)
genes = list(gidx); gidx = [np.array(gidx[x]) for x in genes]
def centre(X):
    X = X.astype(float).copy()
    for ii in gidx: X[ii] -= X[ii].mean(0)
    return X
def p_at_1(s, keep=None):
    out = []
    for ii in gidx:
        k_ = ii if keep is None else ii[keep[ii] | (y[ii] == 1)]
        r = k_[y[k_] == 1]; c = k_[y[k_] == 0]
        if len(r) != 1 or len(c) == 0: continue
        mx = s[c].max(); sr = s[r[0]]
        out.append(1.0 if sr > mx else (1.0 / (1 + (s[c] == mx).sum()) if sr == mx else 0.0))
    return np.mean(out), len(out)
def chance(keep=None):
    out = []
    for ii in gidx:
        k_ = ii if keep is None else ii[keep[ii] | (y[ii] == 1)]
        if (y[k_] == 1).sum() == 1 and (y[k_] == 0).sum() > 0: out.append(1 / len(k_))
    return np.mean(out)
OUT = []
def out(*a): OUT.append(a); log("  ".join(str(x) for x in a))
for lab, keep in (("all competitors", None), ("non-repeat competitors only", ~in_rep), ("repeat-site competitors only", in_rep)):
    p, n = p_at_1(L, keep); out("A", lab, f"genes {n}", f"network P@1 {p:.3f}", f"chance {chance(keep):.3f}")
base, _ = p_at_1(L); ch0 = chance()
Lc = centre(L[:, None])[:, 0]
for lab, X in (("repeat features", Xrep), ("exonic fraction", exo[:, None]), ("repeat features + exonic fraction", Xboth)):
    Xc = centre(X); beta, *_ = np.linalg.lstsq(Xc, Lc, rcond=None); res = Lc - Xc @ beta
    r2 = 1 - (res ** 2).sum() / (Lc ** 2).sum(); p, n = p_at_1(res)
    out("B", f"network with {lab} regressed out", f"logit variance explained {r2:.3f}", f"P@1 {p:.3f}",
        f"share of gain over chance lost {(base - p) / (base - ch0):.2f}")
# c: repeat-only predictors, 5-fold by gene
rng = np.random.default_rng(7); fold = rng.integers(0, 5, len(genes)); fold_i = np.zeros(len(y), int)
for f_, ii in zip(fold, gidx): fold_i[ii] = f_
for lab, X in (("repeat features", Xrep), ("exonic fraction", exo[:, None]), ("repeat features + exonic fraction", Xboth)):
    Xc = centre(X); yc = centre(y[:, None])[:, 0]; s = np.zeros(len(y))
    for f_ in range(5):
        tr = fold_i != f_; beta, *_ = np.linalg.lstsq(Xc[tr], yc[tr], rcond=None); s[~tr] = Xc[~tr] @ beta
    p, n = p_at_1(s); out("C", f"score from {lab} alone (5-fold gene CV)", f"P@1 {p:.3f}", f"share of network gain over chance {(p - ch0) / (base - ch0):.2f}")
with open(os.path.join(ROOT, "data/repeat_attribution.tsv"), "w") as fh:
    for r in OUT: fh.write("\t".join(str(x) for x in r) + "\n")
log("[out] data/repeat_attribution.tsv")
