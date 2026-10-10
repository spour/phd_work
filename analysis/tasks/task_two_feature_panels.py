#!/usr/bin/env python3
# panel data for the final two-feature model
import csv, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, HOLD_FRAC, RIDGE, NBOOT = 42, 0.20, 1.0, 500
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
me3 = D["spl_upstream_max_me3"].to_numpy(float)
uprox = D["spl_pas_prox_u1"].to_numpy(float)
udist = (6 * D["spl_upstream_mean_u1"].to_numpy(float) - uprox) / 5.0
u1c = udist - uprox
NAMES = ["spl_upstream_max_me3", "u1_contrast"]
X2 = np.column_stack([me3, u1c])

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
def loco(X, test_mask=None, train_mask=None):
    tm = np.ones(len(y), bool) if test_mask is None else test_mask
    rm = np.ones(len(y), bool) if train_mask is None else train_mask
    pred = np.full(len(y), np.nan)
    for c in CHROMS:
        te = ROW & tm & (chrom == c); tr = ROW & rm & (chrom != c)
        if te.sum() == 0 or tr.sum() == 0: continue
        A, B = prep(X[tr], X[te]); pred[te] = B @ fit_ridge(A, y[tr])
    return pred
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
def pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b); a = a[m] - a[m].mean(); b = b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan

pred2 = loco(X2); pred2h = loco(X2, is_hold, ~is_hold); predG = loco(BX[:, GCB])
rows_all = np.where(ROW)[0]
rowsH = np.concatenate([ix for _, ix in HG])
yv = y[rows_all]

# coefficients
dev = np.where(ROW & ~is_hold)[0]
A, _ = prep(X2[dev], X2[dev]); beta = fit_ridge(A, y[dev])[1:]
devg = [ix for g, ix in GEN if not is_hold[ix[0]]]
BB = np.empty((300, 2))
for t in range(300):
    ii = np.concatenate([devg[j] for j in rng.integers(0, len(devg), len(devg))])
    Ab, _ = prep(X2[ii], X2[ii]); BB[t] = fit_ridge(Ab, y[ii])[1:]
with open("tf_coefs.tsv", "w") as fh:
    fh.write("feature\tbeta_std\tci_lo\tci_hi\tsign_flip_frac\n")
    print("\n===== TWO-FEATURE MODEL =====")
    for a, nm in enumerate(NAMES):
        lo, hi = np.percentile(BB[:, a], [2.5, 97.5]); fl = float(np.mean(np.sign(BB[:, a]) != np.sign(beta[a])))
        print(f"  {nm:<24} {beta[a]:>+8.4f} [{lo:>+8.4f},{hi:>+8.4f}]  flip {fl:.3f}")
        fh.write(f"{nm}\t{beta[a]:.5f}\t{lo:.5f}\t{hi:.5f}\t{fl:.4f}\n")

# curves
def curve(sv, tag, fh):
    o = rows_all[np.argsort(-sv[rows_all], kind="mergesort")]; yy = y[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy); P = yy.sum(); N = len(yy) - P
    for i in np.unique(np.linspace(0, len(yy) - 1, 400).astype(int)):
        fh.write(f"{tag}\t{fp[i]/N:.6f}\t{tp[i]/P:.6f}\t{tp[i]/(i+1):.6f}\n")
with open("tf_curves.tsv", "w") as fh:
    fh.write("model\tfpr\ttpr\tprecision\n")
    curve(pred2, "two-feature model", fh); curve(rf, "baseline model", fh)
    fh.write(f"#base_rate\t{float(yv.mean()):.5f}\t\t\n")
    fh.write(f"#auroc_model\t{auroc(yv, pred2[rows_all]):.5f}\t\t\n")
    fh.write(f"#auprc_model\t{auprc(yv, pred2[rows_all]):.5f}\t\t\n")
    fh.write(f"#auroc_rf\t{auroc(yv, rf[rows_all]):.5f}\t\t\n")
    fh.write(f"#auprc_rf\t{auprc(yv, rf[rows_all]):.5f}\t\t\n")

# per chromosome
with open("tf_bychrom.tsv", "w") as fh:
    fh.write("chrom\tn_genes\tauroc\tci_lo\tci_hi\n")
    for c in CHROMS:
        gl = [ix for g, ix in GEN if chrom[ix[0]] == c]
        if len(gl) < 20: continue
        rws = np.concatenate(gl); a = auroc(y[rws], pred2[rws])
        b = np.empty(200)
        for t in range(200):
            ii = np.concatenate([gl[j] for j in rng.integers(0, len(gl), len(gl))])
            b[t] = auroc(y[ii], pred2[ii])
        fh.write(f"{c}\t{len(gl)}\t{a:.5f}\t{np.percentile(b,2.5):.5f}\t{np.percentile(b,97.5):.5f}\n")

# blocks: the model, then the two control blocks
BL = np.load("binned_features178.npz")
SPL = [i for i, n in enumerate(BN) if n.startswith("bin") and not n.endswith("_gc")]
CTRL = {"two-feature model (2)": X2, "GC-binned (25)": BX[:, GCB],
        "splice-binned (150)": BX[:, SPL],
        "union (177)": np.column_stack([X2, BX[:, SPL], BX[:, GCB]])}
gidxH = [ix for _, ix in HG]
with open("tf_blocks.tsv", "w") as fh:
    fh.write("block\tn_feat\tauroc\tauroc_lo\tauroc_hi\tauprc\tbase_rate\twinrate\twin_lo\twin_hi\n")
    print(f"\nholdout {len(HG)} genes, base rate {float(y[rowsH].mean()):.4f}")
    for nm, X in CTRL.items():
        p = loco(X, is_hold, ~is_hold)
        a = auroc(y[rowsH], p[rowsH]); ap = auprc(y[rowsH], p[rowsH])
        w = winrate(p, HG); wm = float(np.nanmean(w))
        ab = np.empty(NBOOT); wb = np.empty(NBOOT)
        for t in range(NBOOT):
            sel = rng.integers(0, len(gidxH), len(gidxH))
            ii = np.concatenate([gidxH[j] for j in sel])
            ab[t] = auroc(y[ii], p[ii]); wb[t] = np.nanmean(w[sel])
        print(f"  {nm:<24} AUROC {a:.4f} [{np.percentile(ab,2.5):.4f},{np.percentile(ab,97.5):.4f}] "
              f"AUPRC {ap:.4f} win-rate {wm:.4f}")
        fh.write(f"{nm}\t{X.shape[1]}\t{a:.5f}\t{np.percentile(ab,2.5):.5f}\t{np.percentile(ab,97.5):.5f}\t"
                 f"{ap:.5f}\t{float(y[rowsH].mean()):.5f}\t{wm:.5f}\t{np.percentile(wb,2.5):.5f}\t"
                 f"{np.percentile(wb,97.5):.5f}\n")

print("\n===== GC CONTROLS, two-feature model =====")
print(f"  corr(u1_contrast, GC)           all {pearson(u1c[rows_all], GC[rows_all]):+.4f}  "
      f"canonical {pearson(u1c[rows_all][yv==1], GC[rows_all][yv==1]):+.4f}  "
      f"competitor {pearson(u1c[rows_all][yv==0], GC[rows_all][yv==0]):+.4f}")
print(f"  corr(spl_pas_prox_u1, GC)       all {pearson(uprox[rows_all], GC[rows_all]):+.4f} "
      f"(the term it replaces)")
print(f"  corr(spl_upstream_max_me3, GC)  all {pearson(me3[rows_all], GC[rows_all]):+.4f}")
mfin = np.isfinite(pred2[rows_all]) & np.isfinite(GC[rows_all])
A1 = np.column_stack([np.ones(mfin.sum()), GC[rows_all][mfin]])
bb = np.linalg.lstsq(A1, pred2[rows_all][mfin], rcond=None)[0]
res = np.full(len(y), np.nan); res[rows_all[mfin]] = pred2[rows_all][mfin] - A1 @ bb
r2 = pearson(pred2[rows_all][mfin], GC[rows_all][mfin]) ** 2
gl = [ix for _, ix in GEN]
with open("tf_gc_stats.tsv", "w") as fh:
    fh.write("quantity\tauroc\tauroc_lo\tauroc_hi\twinrate\n")
    for nm, v in [("two-feature model", pred2), ("model, GC removed", res),
                  ("GC, 25 binned profile", predG), ("GC, single 500 nt window", GC)]:
        a = auroc(y[rows_all], v[rows_all]); w = float(np.nanmean(winrate(v, GEN)))
        ab = np.empty(200)
        for t in range(200):
            ii = np.concatenate([gl[j] for j in rng.integers(0, len(gl), len(gl))])
            ab[t] = auroc(y[ii], v[ii])
        print(f"  {nm:<26} AUROC {a:.4f} [{np.percentile(ab,2.5):.4f},{np.percentile(ab,97.5):.4f}] "
              f"win-rate {w:.4f}")
        fh.write(f"{nm}\t{a:.5f}\t{np.percentile(ab,2.5):.5f}\t{np.percentile(ab,97.5):.5f}\t{w:.5f}\n")
    for lab, sub in [("all", np.ones(len(rows_all), bool)), ("canonical", yv == 1), ("competitor", yv == 0)]:
        fh.write(f"#r_disc_{lab}\t{pearson(pred2[rows_all][sub], GC[rows_all][sub]):.5f}\t\t\t\n")
    fh.write(f"#r_u1c_all\t{pearson(u1c[rows_all], GC[rows_all]):.5f}\t\t\t\n")
    fh.write(f"#r_uprox_all\t{pearson(uprox[rows_all], GC[rows_all]):.5f}\t\t\t\n")
    fh.write(f"#r_me3_all\t{pearson(me3[rows_all], GC[rows_all]):.5f}\t\t\t\n")
    fh.write(f"#resid_r2\t{r2:.6f}\t\t\t\n")
print(f"  discriminant vs GC: canonical {pearson(pred2[rows_all][yv==1], GC[rows_all][yv==1]):+.4f}, "
      f"competitor {pearson(pred2[rows_all][yv==0], GC[rows_all][yv==0]):+.4f}, "
      f"GC explains {100*r2:.1f}% of discriminant variance")
with open("tf_gc_points.tsv", "w") as fh:
    fh.write("gene\tclass\tgc\tdiscriminant\n")
    for i in rows_all:
        fh.write(f"{gene[i]}\t{'canonical' if y[i]==1 else 'competitor'}\t{GC[i]:.5f}\t{pred2[i]:.5f}\n")
log("DONE_TWO_FEATURE_PANELS")
