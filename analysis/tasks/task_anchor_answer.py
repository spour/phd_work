#!/usr/bin/env python3
# two questions, one protocol
import csv, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, HOLD_FRAC, RIDGE, NBOOT, NPERM = 42, 0.20, 1.0, 500, 500
rng = np.random.default_rng(SEED)
WS = [500, 1000, 2000, 5000]; SPLITS = [250, 437, 750, 1000]

M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
ROW = np.load("upstream_rowmask.npz")["ROW"]
Z = np.load("anchor_sweep.npz", allow_pickle=True)
S = Z["X"]; SN = list(Z["names"])
col = lambda nm: S[:, SN.index(nm)]

me3 = D["spl_upstream_max_me3"].to_numpy(float)
uprox = D["spl_pas_prox_u1"].to_numpy(float)
umean = D["spl_upstream_mean_u1"].to_numpy(float)
u1c = (6 * umean - uprox) / 5.0 - uprox
anchor_me3 = col("anchor_me3"); anchor_dist = col("anchor_dist")
pipe_dist = D["spl_dist_to_max_3ss"].to_numpy(float)

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

def auroc(yv, s):
    m = np.isfinite(s); yv = yv[m]; s = s[m]
    o = np.argsort(s, kind="mergesort"); sv = s[o]; n = len(s); rk = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    rr = np.empty(n); rr[o] = rk; npos = yv.sum(); nneg = n - npos
    return float((rr[yv == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def winrate(v, G, yy=None):
    yy = y if yy is None else yy
    out = np.full(len(G), np.nan)
    for t, (g, ix) in enumerate(G):
        s = v[ix]; c = int(np.argmax(yy[ix])); o = np.delete(np.arange(len(ix)), c)
        if not np.isfinite(s[c]): continue
        d = s[o][np.isfinite(s[o])]
        if len(d): out[t] = float(np.mean((s[c] > d) + 0.5 * (s[c] == d)))
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
def loco_hold(X):
    pred = np.full(len(y), np.nan)
    for c in CHROMS:
        te = ROW & is_hold & (chrom == c); tr = ROW & ~is_hold & (chrom != c)
        if te.sum() == 0 or tr.sum() == 0: continue
        A, B = prep(X[tr], X[te]); pred[te] = B @ fit_ridge(A, y[tr])
    return pred

HROWS = np.concatenate([ix for _, ix in HG])
HGIDX = [ix for _, ix in HG]
BOOTSEL = [rng.integers(0, len(HGIDX), len(HGIDX)) for _ in range(NBOOT)]
BOOTROWS = [np.concatenate([HGIDX[j] for j in sel]) for sel in BOOTSEL]

def model_metrics(X):
    pred = loco_hold(X)
    a = auroc(y[HROWS], pred[HROWS]); w = winrate(pred, HG)
    ab = np.array([auroc(y[ii], pred[ii]) for ii in BOOTROWS])
    wb = np.array([np.nanmean(w[sel]) for sel in BOOTSEL])
    return (a, float(np.percentile(ab, 2.5)), float(np.percentile(ab, 97.5)),
            float(np.nanmean(w)), float(np.percentile(wb, 2.5)), float(np.percentile(wb, 97.5)), pred, ab)

print("\n===== Q1  ANCHOR ACCEPTOR: STRENGTH vs DISTANCE =====")
print(f"analysis set {int(ROW.sum()):,} sites in {len(GEN):,} genes; holdout {len(HG):,} genes")
r_sd = np.corrcoef(anchor_me3[IDX][np.isfinite(anchor_me3[IDX]) & np.isfinite(anchor_dist[IDX])],
                   anchor_dist[IDX][np.isfinite(anchor_me3[IDX]) & np.isfinite(anchor_dist[IDX])])[0, 1]
mm = np.isfinite(anchor_dist[IDX]) & np.isfinite(pipe_dist[IDX])
print(f"r(strength, distance) = {r_sd:+.4f}; "
      f"r(anchor_dist, spl_dist_to_max_3ss) = {np.corrcoef(anchor_dist[IDX][mm], pipe_dist[IDX][mm])[0,1]:+.4f}; "
      f"same anchor in {float(np.mean(np.abs(anchor_dist[IDX][mm]-pipe_dist[IDX][mm])<=1))*100:.1f}% of sites")
for nm, v in [("anchor_dist", anchor_dist), ("spl_dist_to_max_3ss", pipe_dist)]:
    a = v[IDX]; yy = y[IDX]
    print(f"  median {nm}: canonical {np.nanmedian(a[yy==1]):.0f} nt, competitor {np.nanmedian(a[yy==0]):.0f} nt")

fh = open("anchor_distance.tsv", "w")
fh.write("kind\tname\tauroc\tauroc_lo\tauroc_hi\twinrate\twin_lo\twin_hi\tnull_win\tnull_win_lo\t"
         "null_win_hi\tnull_auroc_lo\tnull_auroc_hi\n")
print(f"\n{'marginal feature':<26} {'AUROC':>7} {'95% CI':>17} {'win-rate':>9} {'95% CI':>17}"
      f" {'perm null win-rate':>26}")
GALL = [ix for _, ix in GEN]
PERMY = []
for _ in range(NPERM):
    yp = np.zeros(len(y))
    for ix in GALL: yp[ix[rng.integers(0, len(ix))]] = 1.0
    PERMY.append(yp)
for nm, v in [("anchor acceptor score", anchor_me3), ("anchor distance to pA", anchor_dist),
              ("spl_dist_to_max_3ss", pipe_dist), ("u1_contrast", u1c)]:
    a = auroc(y[IDX], v[IDX]); w = winrate(v, GEN); wm = float(np.nanmean(w))
    ab = np.array([auroc(y[np.concatenate([GALL[j] for j in sel])], v[np.concatenate([GALL[j] for j in sel])])
                   for sel in [rng.integers(0, len(GALL), len(GALL)) for _ in range(200)]])
    wb = np.array([np.nanmean(w[rng.integers(0, len(w), len(w))]) for _ in range(300)])
    nw = np.array([np.nanmean(winrate(v, GEN, yy=yp)) for yp in PERMY[:200]])
    na = np.array([auroc(yp[IDX], v[IDX]) for yp in PERMY[:200]])
    print(f"{nm:<26} {a:>7.4f} [{np.percentile(ab,2.5):.4f},{np.percentile(ab,97.5):.4f}] "
          f"{wm:>9.4f} [{np.percentile(wb,2.5):.4f},{np.percentile(wb,97.5):.4f}]"
          f"   {nw.mean():.4f} [{np.percentile(nw,2.5):.4f},{np.percentile(nw,97.5):.4f}]")
    fh.write(f"marginal\t{nm}\t{a:.5f}\t{np.percentile(ab,2.5):.5f}\t{np.percentile(ab,97.5):.5f}\t"
             f"{wm:.5f}\t{np.percentile(wb,2.5):.5f}\t{np.percentile(wb,97.5):.5f}\t{nw.mean():.5f}\t"
             f"{np.percentile(nw,2.5):.5f}\t{np.percentile(nw,97.5):.5f}\t{np.percentile(na,2.5):.5f}\t"
             f"{np.percentile(na,97.5):.5f}\n")
print(f"  (perm null AUROC 95% band, all features: "
      f"[{np.percentile(na,2.5):.4f},{np.percentile(na,97.5):.4f}])")

print(f"\n{'model (holdout)':<44} {'AUROC':>7} {'95% CI':>17} {'win-rate':>9} {'95% CI':>17} {'dAUROC vs 2f':>22}")
MODS = [("strength alone", np.column_stack([anchor_me3])),
        ("distance alone", np.column_stack([anchor_dist])),
        ("strength + distance", np.column_stack([anchor_me3, anchor_dist])),
        ("two-feature model", np.column_stack([anchor_me3, u1c])),
        ("two-feature + distance", np.column_stack([anchor_me3, u1c, anchor_dist])),
        ("two-feature + spl_dist_to_max_3ss", np.column_stack([anchor_me3, u1c, pipe_dist]))]
base_ab = None
for nm, X in MODS:
    a, alo, ahi, w, wlo, whi, pred, ab = model_metrics(X)
    if nm == "two-feature model": base_ab = ab; base_a = a
    dtxt = ""
    if base_ab is not None and nm != "two-feature model":
        d = ab - base_ab
        dtxt = f"{a-base_a:+.4f} [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]"
    print(f"{nm:<44} {a:>7.4f} [{alo:.4f},{ahi:.4f}] {w:>9.4f} [{wlo:.4f},{whi:.4f}] {dtxt:>22}")
    fh.write(f"model\t{nm}\t{a:.5f}\t{alo:.5f}\t{ahi:.5f}\t{w:.5f}\t{wlo:.5f}\t{whi:.5f}\t\t\t\t\t\n")
fh.close()

print("\n===== Q2  WINDOW SWEEP, holdout AUROC of the two-feature model =====")
U1D = {s: col(f"u1_dist_{s}") for s in SPLITS}
U1P = {s: col(f"u1_prox_{s}") for s in SPLITS}
ME3 = {w: col(f"me3_W{w}") for w in WS}
print(f"rescanned s=437 contrast vs deployed u1_contrast: r = "
      f"{np.corrcoef((U1D[437]-U1P[437])[IDX], u1c[IDX])[0,1]:.6f}")
GRID = {}
with open("window_sweep.tsv", "w") as g:
    g.write("acc_W\tu1_split\tauroc\tauroc_lo\tauroc_hi\twinrate\twin_lo\twin_hi\n")
    print(f"\n{'acc W':>6} {'u1 split':>9} {'AUROC':>7} {'95% CI':>17} {'win-rate':>9} {'vs current':>10}")
    cur = None
    for w in WS:
        for s in SPLITS:
            X = np.column_stack([ME3[w], U1D[s] - U1P[s]])
            a, alo, ahi, wr, wlo, whi, _, ab = model_metrics(X)
            GRID[(w, s)] = (a, ab)
            if (w, s) == (2000, 437): cur = a
            g.write(f"{w}\t{s}\t{a:.5f}\t{alo:.5f}\t{ahi:.5f}\t{wr:.5f}\t{wlo:.5f}\t{whi:.5f}\n")
            print(f"{w:>6} {s:>9} {a:>7.4f} [{alo:.4f},{ahi:.4f}] {wr:>9.4f}"
                  f" {'' if cur is None else f'{a-cur:+.4f}'}")
best = max(GRID, key=lambda k: GRID[k][0])
d = GRID[best][1] - GRID[(2000, 437)][1]
print(f"\nbest cell W={best[0]}, split={best[1]}: AUROC {GRID[best][0]:.4f}; "
      f"current cell W=2000, split=437: {GRID[(2000,437)][0]:.4f}; "
      f"paired delta {GRID[best][0]-GRID[(2000,437)][0]:+.4f} "
      f"[{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]")
av = np.array([[GRID[(w, s)][0] for s in SPLITS] for w in WS])
print(f"grid range {av.min():.4f} to {av.max():.4f}, spread {av.max()-av.min():.4f}; "
      f"row (W) spread {np.ptp(av.mean(1)):.4f}, column (split) spread {np.ptp(av.mean(0)):.4f}")
log("DONE_ANCHOR_ANSWER")
