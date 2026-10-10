#!/usr/bin/env python3
# primary reranker analysis: unannotated competitors only, 5 kb count features excluded
import os, sys, csv, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)

SEED, HOLD_FRAC, RIDGE, NBOOT, MAXK = 42, 0.20, 1.0, 400, 12
NPERM = 200
PHASE = os.environ.get("PHASE", "select")
rng = np.random.default_rng(SEED)

D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
rf = D.rf_prob.to_numpy(float)
S = np.load("splice_feats_cache.npz")
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
def sc(n): return S[n][ridx]
DROP = {"spl_dist_dn_5ss", "spl_dist_up_5ss"}
FEAT = {c: D[c].to_numpy(float) for c in D.columns if c.startswith("spl_") and c not in DROP}
FEAT["scan_acc_rank2k"] = sc("B_acc_rank2k"); FEAT["scan_acc_margin2k"] = sc("B_acc_margin2k")
FEAT["scan_in_intron"] = sc("C_in_intron")
FEAT["acc_minus_max5"] = FEAT["spl_upstream_max_me3"] - FEAT["spl_interv_me5_max"]
NAMES = sorted(FEAT); X = np.column_stack([FEAT[n] for n in NAMES])

At = np.load("atlas_peak_cache.npz")["dist"][ridx]
Pd = np.load("polyadb_peak_cache.npz")["dist"][ridx]
UNANNOT = (At > 500) & (Pd > 500)
gi = defaultdict(list)
for i in range(len(gene)): gi[gene[i]].append(i)
gi = {g: np.array(v) for g, v in gi.items()}
def gene_filter(mask):
    keep = np.zeros(len(y), bool)
    for g, ix in gi.items():
        sub = ix[mask[ix]]
        if (y[sub] == 1).sum() == 1 and (y[sub] == 0).sum() >= 1: keep[sub] = True
    return keep
ROW = gene_filter((y == 1) | ((y == 0) & UNANNOT))
allg = np.array(sorted(set(gene)))
hold_g = set(rng.choice(allg, size=int(round(HOLD_FRAC * len(allg))), replace=False))
is_hold = np.array([g in hold_g for g in gene])
CHROMS = sorted(set(chrom))
log(f"primary set: {int((ROW&(y==0)).sum())} competitor peaks, "
    f"{len({gene[i] for i in np.where(ROW&(y==0))[0]})} genes; pool {len(NAMES)} features")

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
    A = (np.where(np.isfinite(Xtr), Xtr, mu) - mu) / sd
    B = (np.where(np.isfinite(Xte), Xte, mu) - mu) / sd
    return np.column_stack([np.ones(len(A)), A]), np.column_stack([np.ones(len(B)), B])
def loco_pred(cols, train_mask, test_mask, Xm=None, yv=None):
    XX = X if Xm is None else Xm; yy = y if yv is None else yv
    pred = np.full(len(yy), np.nan); Xs = XX[:, cols]
    for c in CHROMS:
        te = ROW & test_mask & (chrom == c); tr = ROW & train_mask & (chrom != c)
        if te.sum() == 0 or tr.sum() == 0: continue
        A, B = prep(Xs[tr], Xs[te]); pred[te] = B @ fit_ridge(A, yy[tr])
    return pred
def per_gene(pred, mask, yv=None):
    yy = y if yv is None else yv
    hits, rrs, ns, gl = [], [], [], []
    for g, ix in gi.items():
        if not mask[ix[0]]: continue
        ix = ix[ROW[ix]]
        if len(ix) < 2 or (yy[ix] == 1).sum() != 1: continue
        s = pred[ix]
        if not np.isfinite(s).all(): continue
        n = len(ix); j = np.argsort(-(s + rng.random(n) * 1e-12), kind="mergesort")
        srt = s[j]; rk = np.empty(n); i = 0
        while i < n:
            k = i
            while k + 1 < n and srt[k + 1] == srt[i]: k += 1
            rk[i:k + 1] = (i + k) / 2.0 + 1.0; i = k + 1
        r = np.empty(n); r[j] = rk
        c = int(np.argmax(yy[ix]))
        hits.append(1.0 if j[0] == c else 0.0); rrs.append(1.0 / r[c]); ns.append(n); gl.append((g, ix))
    return np.array(hits), np.array(rrs), np.array(ns, float), gl
def bci(v):
    b = v[rng.integers(0, len(v), size=(NBOOT, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

CACHE = "rerank_primary_path_cache.tsv"
chosen, path, prev = [], [], 0.0
remaining = list(range(len(NAMES)))
print("\n===== FORWARD SELECTION, PRIMARY SET (unannotated competitors; pool %d) =====" % len(NAMES))
print(f"{'k':>2} {'added':<24} {'dev P@1':>8} {'dev 95% CI':>18} {'hold P@1':>9} {'hold 95% CI':>18} {'gain':>8}")
if os.path.exists(CACHE):
    for L in open(CACHE):
        p = L.rstrip("\n").split("\t")
        if p[0] == "k": continue
        path.append(dict(k=int(p[0]), added=p[1], dev=float(p[2]), dlo=float(p[3]), dhi=float(p[4]),
                         hold=float(p[5]), hlo=float(p[6]), hhi=float(p[7]), gain=float(p[8]), hchance=float(p[9])))
        j = NAMES.index(p[1]); chosen.append(j); remaining.remove(j)
        print(f"{p[0]:>2} {p[1]:<24} {float(p[2]):>8.4f} [{float(p[3]):.4f},{float(p[4]):.4f}] "
              f"{float(p[5]):>9.4f} [{float(p[6]):.4f},{float(p[7]):.4f}] {float(p[8]):>+8.4f}   (cached)")
    prev = path[-1]["dev"] if path else 0.0
for k in range(len(path) + 1, MAXK + 1):
    best = None
    for j in remaining:
        h, _, _, _ = per_gene(loco_pred(chosen + [j], ~is_hold, ~is_hold), ~is_hold)
        if best is None or h.mean() > best[0]: best = (h.mean(), j, h)
    v, j, h = best; chosen.append(j); remaining.remove(j)
    dm, dlo, dhi = bci(h)
    hh, _, hns, _ = per_gene(loco_pred(chosen, ~is_hold, is_hold), is_hold)
    hm, hlo, hhi = bci(hh)
    path.append(dict(k=k, added=NAMES[j], dev=dm, dlo=dlo, dhi=dhi, hold=hm, hlo=hlo, hhi=hhi,
                     gain=dm - prev, hchance=float(np.mean(1.0 / hns))))
    print(f"{k:>2} {NAMES[j]:<24} {dm:>8.4f} [{dlo:.4f},{dhi:.4f}] {hm:>9.4f} [{hlo:.4f},{hhi:.4f}] {dm-prev:>+8.4f}")
    prev = dm; log(f"  step {k}: {NAMES[j]}")
    with open(CACHE, "w") as fh:
        fh.write("k\tadded\tdev_p1\tdev_lo\tdev_hi\thold_p1\thold_lo\thold_hi\tgain\thold_chance\n")
        for p in path:
            fh.write(f"{p['k']}\t{p['added']}\t{p['dev']:.5f}\t{p['dlo']:.5f}\t{p['dhi']:.5f}\t"
                     f"{p['hold']:.5f}\t{p['hlo']:.5f}\t{p['hhi']:.5f}\t{p['gain']:+.5f}\t{p['hchance']:.5f}\n")

kp = max(path, key=lambda p: p["hold"])
cover = [p["k"] for p in path if p["hhi"] >= kp["hold"]]
kstar = min(cover); kA = kp["k"]
print(f"\nholdout plateau {kp['hold']:.4f} at k={kA} (CI [{kp['hlo']:.4f},{kp['hhi']:.4f}])")
print(f"k whose holdout CI covers the plateau value: {cover} -> rule gives k = {kstar}")
CHOSEN = chosen[:kA]
print(f"plateau-attaining set (k={kA}): " + ", ".join(NAMES[j] for j in CHOSEN))
print(f"rule-minimal set (k={kstar}): " + ", ".join(NAMES[j] for j in chosen[:kstar]))
with open("rerank_primary_path.tsv", "w") as fh:
    fh.write("k\tadded\tdev_p1\tdev_lo\tdev_hi\thold_p1\thold_lo\thold_hi\tgain\thold_chance\tchosen\n")
    for p in path:
        fh.write(f"{p['k']}\t{p['added']}\t{p['dev']:.5f}\t{p['dlo']:.5f}\t{p['dhi']:.5f}\t"
                 f"{p['hold']:.5f}\t{p['hlo']:.5f}\t{p['hhi']:.5f}\t{p['gain']:+.5f}\t{p['hchance']:.5f}\t"
                 f"{1 if p['k']==kA else 0}\n")
with open("rerank_primary_chosen.tsv", "w") as fh:
    fh.write("feature\n")
    for j in CHOSEN: fh.write(NAMES[j] + "\n")

dev = np.where(ROW & ~is_hold)[0]; Xs = X[:, CHOSEN]
A, _ = prep(Xs[dev], Xs[dev]); beta = fit_ridge(A, y[dev])[1:]
devg = [ix[ROW[ix]] for g, ix in gi.items() if not is_hold[ix[0]] and ROW[ix].sum() >= 2]
BB = np.empty((300, len(CHOSEN)))
for t in range(300):
    ii = np.concatenate([devg[j] for j in rng.integers(0, len(devg), len(devg))])
    Ab, _ = prep(Xs[ii], Xs[ii]); BB[t] = fit_ridge(Ab, y[ii])[1:]
print("\nstandardized coefficients of the chosen set (dev fit, bootstrap over genes):")
with open("rerank_primary_coefs.tsv", "w") as fh:
    fh.write("feature\tbeta_std\tci_lo\tci_hi\tsign_flip_frac\n")
    for a, j in enumerate(CHOSEN):
        lo, hi = np.percentile(BB[:, a], [2.5, 97.5]); fl = float(np.mean(np.sign(BB[:, a]) != np.sign(beta[a])))
        print(f"  {NAMES[j]:<24} {beta[a]:>+8.4f} [{lo:>+8.4f},{hi:>+8.4f}]  flip {fl:.3f}")
        fh.write(f"{NAMES[j]}\t{beta[a]:.5f}\t{lo:.5f}\t{hi:.5f}\t{fl:.4f}\n")
log("DONE_PRIMARY_SELECT")
