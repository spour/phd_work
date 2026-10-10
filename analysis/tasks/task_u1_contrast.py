#!/usr/bin/env python3
# is the u1 pair a real contrast or a suppression artifact, and does it match the metagene?
import csv, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, HOLD_FRAC, RIDGE, NBOOT = 42, 0.20, 1.0, 500
rng = np.random.default_rng(SEED)
NEW3 = ["spl_upstream_max_me3", "spl_pas_prox_u1", "spl_upstream_mean_u1"]

M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
Z = np.load("upstream_rowmask.npz"); ROW = Z["ROW"]
FEAT = {n: D[n].to_numpy(float) for n in NEW3}
FEAT["distal upstream U1 (bins 1-5)"] = (6 * FEAT["spl_upstream_mean_u1"] - FEAT["spl_pas_prox_u1"]) / 5.0
W = np.load("u1_window_cache.npz")
FEAT["U1 hit density at 5.25 (per kb)"] = W["dens_5.25"][ridx]
gi = defaultdict(list)
for i in range(len(gene)): gi[gene[i]].append(i)
gi = {g: np.array(v) for g, v in gi.items()}
allg = np.array(sorted(set(gene)))
hold_g = set(rng.choice(allg, size=int(round(HOLD_FRAC * len(allg))), replace=False))
is_hold = np.array([g in hold_g for g in gene])
CHROMS = sorted(set(chrom))
IDX = np.where(ROW)[0]
GEN = [(g, ix[ROW[ix]]) for g, ix in gi.items() if ROW[ix].sum() >= 2 and (y[ix[ROW[ix]]] == 1).sum() == 1]
HG = [(g, ix) for g, ix in GEN if is_hold[ix[0]]]

def auroc(yv, s):
    m = np.isfinite(s); yv = yv[m]; s = s[m]
    o = np.argsort(s, kind="mergesort"); sv = s[o]; n = len(s); rk = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    rr = np.empty(n); rr[o] = rk; npos = yv.sum(); nneg = n - npos
    return float((rr[yv == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def winrate(v, G):
    out = np.full(len(G), np.nan)
    for t, (g, ix) in enumerate(G):
        s = v[ix]; c = int(np.argmax(y[ix])); o = np.delete(np.arange(len(ix)), c)
        if not np.isfinite(s[c]): continue
        d = s[o][np.isfinite(s[o])]
        if len(d) == 0: continue
        out[t] = float(np.mean((s[c] > d) + 0.5 * (s[c] == d)))
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
def loco(X):
    pred = np.full(len(y), np.nan)
    for c in CHROMS:
        te = ROW & is_hold & (chrom == c); tr = ROW & ~is_hold & (chrom != c)
        if te.sum() == 0 or tr.sum() == 0: continue
        A, B = prep(X[tr], X[te]); pred[te] = B @ fit_ridge(A, y[tr])
    return pred
def pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b); a = a[m] - a[m].mean(); b = b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan

print("\n===== 1. THE U1 PAIR: MARGINAL VERSUS CONDITIONAL =====")
print(f"nested by definition: pas_prox_u1 is bin 6 of the six bins averaged in upstream_mean_u1")
print(f"correlation r(upstream_mean_u1, pas_prox_u1) = "
      f"{pearson(FEAT['spl_upstream_mean_u1'][IDX], FEAT['spl_pas_prox_u1'][IDX]):+.4f}")
print(f"correlation r(distal bins 1-5, pas_prox_u1) = "
      f"{pearson(FEAT['distal upstream U1 (bins 1-5)'][IDX], FEAT['spl_pas_prox_u1'][IDX]):+.4f}")
print(f"\n{'feature':<34} {'marg AUROC':>11} {'marg win-rate':>14} {'univ beta':>10} {'model beta':>11}")
MB = {"spl_upstream_max_me3": +0.4290, "spl_pas_prox_u1": -0.3236, "spl_upstream_mean_u1": +0.2730}
ROWS = []
with open("u1_contrast.tsv", "w") as fh:
    fh.write("feature\tmarg_auroc\tmarg_winrate\twin_lo\twin_hi\tuniv_beta\tmodel_beta\n")
    for nm in ["spl_upstream_mean_u1", "spl_pas_prox_u1", "distal upstream U1 (bins 1-5)",
               "U1 hit density at 5.25 (per kb)", "spl_upstream_max_me3"]:
        v = FEAT[nm]
        au = auroc(y[IDX], v[IDX])
        w = winrate(v, GEN); wm = float(np.nanmean(w))
        b = np.array([np.nanmean(w[rng.integers(0, len(w), len(w))]) for _ in range(200)])
        dev = np.where(ROW & ~is_hold)[0]
        A, _ = prep(v[dev, None], v[dev, None]); ub = float(fit_ridge(A, y[dev])[1])
        mb = MB.get(nm, np.nan)
        print(f"{nm:<34} {au:>11.4f} {wm:>14.4f} {ub:>+10.4f} " +
              (f"{mb:>+11.4f}" if np.isfinite(mb) else f"{'-':>11}"))
        fh.write(f"{nm}\t{au:.5f}\t{wm:.5f}\t{np.percentile(b,2.5):.5f}\t{np.percentile(b,97.5):.5f}\t"
                 f"{ub:.5f}\t{mb if np.isfinite(mb) else ''}\n")
    X3 = np.column_stack([FEAT[n] for n in NEW3])
    full = loco(X3); rowsH = np.concatenate([ix for _, ix in HG])
    aF = auroc(y[rowsH], full[rowsH])
    print(f"\nmodel AUROC (holdout, {len(HG)} genes): three features {aF:.4f}")
    for drop in range(3):
        cols = [i for i in range(3) if i != drop]
        p = loco(X3[:, cols]); a = auroc(y[rowsH], p[rowsH])
        w = np.nanmean(winrate(p, HG))
        print(f"  drop {NEW3[drop]:<24} -> AUROC {a:.4f} ({a-aF:+.4f}), win-rate {w:.4f}")
        fh.write(f"#drop\t{NEW3[drop]}\t{a:.5f}\t{a-aF:.5f}\t{w:.5f}\n")
    fh.write(f"#full\tthree features\t{aF:.5f}\t0\t{np.nanmean(winrate(full, HG)):.5f}\n")
    d525 = FEAT["U1 hit density at 5.25 (per kb)"]
    can = ROW & (y == 1); com = ROW & (y == 0)
    print(f"\nreconciliation with the earlier hit-density result, recomputed on THIS set:")
    print(f"  U1 hit density at 5.25: canonical {np.nanmean(d525[can]):.2f} vs competitor "
          f"{np.nanmean(d525[com]):.2f} per kb (earlier, all 70,723 peaks: 23.28 vs 26.12)")
    print(f"  mean upstream_mean_u1:  canonical {np.nanmean(FEAT['spl_upstream_mean_u1'][can]):.4f} vs "
          f"competitor {np.nanmean(FEAT['spl_upstream_mean_u1'][com]):.4f}")
    print(f"  mean pas_prox_u1:       canonical {np.nanmean(FEAT['spl_pas_prox_u1'][can]):.4f} vs "
          f"competitor {np.nanmean(FEAT['spl_pas_prox_u1'][com]):.4f}")
    print(f"  mean distal bins 1-5:   canonical {np.nanmean(FEAT['distal upstream U1 (bins 1-5)'][can]):.4f} vs "
          f"competitor {np.nanmean(FEAT['distal upstream U1 (bins 1-5)'][com]):.4f}")
    fh.write(f"#dens525\t{np.nanmean(d525[can]):.5f}\t{np.nanmean(d525[com]):.5f}\n")

MG = np.load("u1_metagene_cache.npz"); Mm = MG["M"][ridx]; cen = MG["centers"]
print("\n===== 2. U1 METAGENE, current canonical and competitor sets =====")
print(f"{len(cen)} bins of {cen[1]-cen[0]:.0f} nt from {cen[0]:.0f} to {cen[-1]:.0f} nt around the pA; "
      f"U1 hit density per kb at threshold 5.25")
gidx_c = [ix[y[ix] == 1] for _, ix in GEN]; gidx_k = [ix[y[ix] == 0] for _, ix in GEN]
def prof(gidx):
    v = np.array([np.nanmean(Mm[ix], 0) for ix in gidx if len(ix)])
    b = np.array([np.nanmean(v[rng.integers(0, len(v), len(v))], 0) for _ in range(300)])
    return np.nanmean(v, 0), np.percentile(b, 2.5, axis=0), np.percentile(b, 97.5, axis=0)
mc, lc, hc = prof(gidx_c); mk, lk, hk = prof(gidx_k)
with open("u1_metagene.tsv", "w") as fh:
    fh.write("position\tclass\tu1_density\tci_lo\tci_hi\n")
    for i, p in enumerate(cen):
        fh.write(f"{p:.1f}\tcanonical\t{mc[i]:.4f}\t{lc[i]:.4f}\t{hc[i]:.4f}\n")
        fh.write(f"{p:.1f}\tcompetitor\t{mk[i]:.4f}\t{lk[i]:.4f}\t{hk[i]:.4f}\n")
jc = int(np.argmin(mc)); jk = int(np.argmin(mk))
print(f"  canonical: minimum {mc[jc]:.1f} per kb at {cen[jc]:+.0f} nt; "
      f"level at -487 nt {mc[0]:.1f}; depth {mc[0]-mc[jc]:.1f} ({100*(1-mc[jc]/mc[0]):.0f}% below the -487 level)")
print(f"  competitor: minimum {mk[jk]:.1f} per kb at {cen[jk]:+.0f} nt; "
      f"level at -487 nt {mk[0]:.1f}; depth {mk[0]-mk[jk]:.1f} ({100*(1-mk[jk]/mk[0]):.0f}% below)")
print(f"  canonical - competitor at the pA bin ({cen[jc]:+.0f} nt): {mc[jc]-mk[jc]:+.2f} per kb")
print(f"  canonical - competitor averaged over -487..-262 nt: "
      f"{np.mean(mc[:10])-np.mean(mk[:10]):+.2f} per kb")
log("DONE_U1_CONTRAST")
