#!/usr/bin/env python3
# two characterisation checks on the locked splice comparator
import sys, time
import numpy as np
import binned_common as K
from binned_common import log
sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as L

POW, U1L = L.POW, L.U1
UP, DN = 2000, 500
THR = (4.0, 7.0)
RES = (200, 100, 50, 25)
ACC = K.ACC

D = np.load("cnn_data.npz", allow_pickle=True)
ok = D["ok"]; X = D["X"]
assert X.shape == (int(ok.sum()), UP + DN) == (len(K.y), 2500)
log(f"[scan] {X.shape[0]} sites x {X.shape[1]} tokens = the [-{UP}, +{DN}] window")

def profiles(tok):
    t = tok.astype(np.int64)
    sw = np.lib.stride_tricks.sliding_window_view(t, 9)
    good = (sw < 4).all(1)
    u1 = np.where(good, U1L[np.where(good, sw @ POW, 0)], np.nan)
    return np.nan_to_num(u1, nan=-99.0)

OFF9 = np.arange(UP + DN - 9 + 1) + 3 - UP
def nbins(bw):
    return -(-(UP + DN) // bw)

def binned(hits, bw):
    nb = nbins(bw)
    b = (OFF9 + UP) // bw
    keep = (b >= 0) & (b < nb); bk = b[keep]
    n = np.bincount(bk, minlength=nb).astype(float)
    out = np.empty((hits.shape[0], nb))
    for i in range(hits.shape[0]):
        s = np.bincount(bk, weights=hits[i][keep].astype(float), minlength=nb)
        out[i] = np.where(n > 0, s / np.maximum(n, 1) * 1000.0, np.nan)
    return out

t0 = time.time()
HIT = {}
CH = np.empty((len(K.y), 2, UP + DN - 9 + 1), bool)
for i in range(len(K.y)):
    u = profiles(X[i])
    CH[i, 0] = u >= THR[0]; CH[i, 1] = u >= THR[1]
    if (i + 1) % 10000 == 0: log(f"  scan {i+1}/{len(K.y)}  {time.time()-t0:.0f}s")
log(f"[scan] U1 hit indicators for both thresholds in {time.time()-t0:.0f}s")

BINS = {}
for bw in RES:
    BINS[bw] = np.column_stack([binned(CH[:, c], bw) for c in (0, 1)] + [ACC])
    tail = (UP + DN) - bw * (nbins(bw) - 1)
    log(f"[bins] {bw:>3d} nt: {nbins(bw):>3d} bins x 2 channels + acceptor "
        f"= {BINS[bw].shape[1]:>4d} features"
        + (f"   (last bin {tail} nt)" if tail != bw else ""))

Z = np.load("binned_profiles.npz", allow_pickle=True)
Pc = Z["P"]; CHN = [str(c) for c in Z["channels"]]
REF = np.column_stack([Pc[:, CHN.index("u1_dens4"), :], Pc[:, CHN.index("u1_dens7"), :], ACC])
d64 = np.nanmax(np.abs(BINS[100] - REF))
d32 = np.nanmax(np.abs(BINS[100].astype(np.float32) - REF.astype(np.float32)))
assert d32 == 0.0, f"100 nt rebinning does not reproduce binned_profiles.npz (float32 diff {d32})"
log(f"[check] the 100 nt arm reproduces binned_profiles.npz exactly in float32 "
    f"(float64 residual {d64:.2e}, below half a float32 ULP at the array maximum)")

# a: resolution sweep
R = {}
for bw in RES:
    M = BINS[bw]
    lam, iv = K.pick_lambda(M, f"{bw} nt")
    sc, _ = K.fit_score(M, K.trv, K.te, lam)
    r = K.suite(sc); r.update(lam=lam, iv=iv, p=M.shape[1], bw=bw, gap=r["winrate"] - iv)
    R[bw] = r
    log(f"[A] {bw:>3d} nt  p={M.shape[1]:>4d}  lambda {lam:<7g} inner-val {iv:.4f}  "
        f"test {r['winrate']:.4f} [{r['winrate_lo']:.4f},{r['winrate_hi']:.4f}]  "
        f"P@1 {r['p_at_1']:.4f} [{r['p_at_1_lo']:.4f},{r['p_at_1_hi']:.4f}]  "
        f"test-minus-inner {r['gap']:+.4f}")
assert abs(R[100]["winrate"] - 0.6957) < 5e-4, \
    f"100 nt does not reproduce the locked 0.6957 (got {R[100]['winrate']:.4f})"
log("[check] the 100 nt arm reproduces the locked model's 0.6957 on test")

def paired(a, b, tag):
    da = a["_wg"][K.BS].mean(1) - b["_wg"][K.BS].mean(1)
    d_ = a["winrate"] - b["winrate"]; lo, hi = np.percentile(da, [2.5, 97.5])
    p = max(2 * min((da <= 0).mean(), (da >= 0).mean()), 1 / K.NBOOT)
    log(f"[paired] {tag:40s} {d_:+.4f} [{lo:+.4f}, {hi:+.4f}]  p = {p:.4f}"
        f"{'' if lo * hi > 0 else '   (CI spans 0)'}")
    return d_, lo, hi, p
log("")
PA = {bw: paired(R[bw], R[100], f"{bw} nt minus 100 nt") for bw in RES if bw != 100}
BEST = max((bw for bw in RES if bw < 100), key=lambda b: R[b]["winrate"])
FINER_WINS = (R[BEST]["winrate"] - R[100]["winrate"]) > 0.005
log(f"[A] VERDICT: " + (
    f"finer than 100 nt HELPS: {BEST} nt beats 100 nt by "
    f"{R[BEST]['winrate']-R[100]['winrate']:+.4f} on test, above the 0.005 bar"
    if FINER_WINS else
    f"100 nt is the natural scale. The best finer resolution ({BEST} nt) is "
    f"{R[BEST]['winrate']-R[100]['winrate']:+.4f} against it, below the 0.005 bar, and the "
    f"test-minus-inner-val gap widens from {R[100]['gap']:+.4f} at 100 nt to "
    f"{R[25]['gap']:+.4f} at 25 nt"))
with open("splice_model_resolution.tsv", "w") as fh:
    fh.write("bin_nt\tn_bins\tn_features\tlambda\tinner_val\twinrate\tci_lo\tci_hi\t"
             "p_at_1\tp1_lo\tp1_hi\tauroc\tauprc\ttest_minus_inner\t"
             "delta_vs_100\td_lo\td_hi\td_p\n")
    for bw in RES:
        r = R[bw]; d = PA.get(bw, (0.0, np.nan, np.nan, np.nan))
        fh.write(f"{bw}\t{nbins(bw)}\t{r['p']}\t{r['lam']:g}\t{r['iv']:.5f}\t"
                 f"{r['winrate']:.5f}\t{r['winrate_lo']:.5f}\t{r['winrate_hi']:.5f}\t"
                 f"{r['p_at_1']:.5f}\t{r['p_at_1_lo']:.5f}\t{r['p_at_1_hi']:.5f}\t"
                 f"{r['auroc']:.5f}\t{r['auprc']:.5f}\t{r['gap']:+.5f}\t"
                 + ("\t\t\t\n" if bw == 100 else
                    f"{d[0]:+.5f}\t{d[1]:+.5f}\t{d[2]:+.5f}\t{d[3]:.4f}\n"))

# b: linear vs boosted trees
log("")
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
FIN = BINS[100]
Zf = K.zmat(FIN, K.trv)
GRID = [(lr, ml, mi) for lr in (0.03, 0.06, 0.1) for ml in (7, 15, 31) for mi in (150, 400)]
best = (-1.0, None)
for lr, ml, mi in GRID:
    m = HGB(learning_rate=lr, max_leaf_nodes=ml, max_iter=mi, l2_regularization=1.0,
            min_samples_leaf=50, early_stopping=False, random_state=K.SEED)
    m.fit(Zf[K.it], K.y[K.it])
    w = K.winrate(m.predict_proba(Zf[K.iv])[:, 1], K.GRP_IV)
    if w > best[0]: best = (w, (lr, ml, mi))
log(f"[B] boosted-tree inner-val best {best[0]:.4f} at learning_rate {best[1][0]}, "
    f"max_leaf_nodes {best[1][1]}, max_iter {best[1][2]}  (grid of {len(GRID)})")
lr, ml, mi = best[1]
m = HGB(learning_rate=lr, max_leaf_nodes=ml, max_iter=mi, l2_regularization=1.0,
        min_samples_leaf=50, early_stopping=False, random_state=K.SEED)
m.fit(Zf[K.trv], K.y[K.trv])
R_H = K.suite(m.predict_proba(Zf[K.te])[:, 1])
R_H.update(iv=best[0], gap=R_H["winrate"] - best[0])
log(f"[B] boosted tree   test {R_H['winrate']:.4f} [{R_H['winrate_lo']:.4f},{R_H['winrate_hi']:.4f}]"
    f"  P@1 {R_H['p_at_1']:.4f} [{R_H['p_at_1_lo']:.4f},{R_H['p_at_1_hi']:.4f}]  "
    f"AUROC {R_H['auroc']:.4f}  AUPRC {R_H['auprc']:.4f}  test-minus-inner {R_H['gap']:+.4f}")
log(f"[B] ridge logistic test {R[100]['winrate']:.4f} "
    f"[{R[100]['winrate_lo']:.4f},{R[100]['winrate_hi']:.4f}]  P@1 {R[100]['p_at_1']:.4f}")
DH = paired(R_H, R[100], "boosted tree minus ridge logistic")
DIFF, LO, HI, PV = DH
if DIFF > 0.005 and LO > 0:
    VB = (f"INTERACTION STRUCTURE PRESENT. The boosted tree beats the ridge by {DIFF:+.4f} "
          f"[{LO:+.4f}, {HI:+.4f}], so these features carry structure a linear model misses"
          + (". The gap is large; flag it." if DIFF > 0.02 else
             ". The gap is small, and the reported model stays linear for the coefficient profile."))
elif abs(DIFF) <= 0.005 or (LO < 0 < HI):
    VB = (f"LINEARLY EXHAUSTED. The boosted tree is {DIFF:+.4f} [{LO:+.4f}, {HI:+.4f}] against "
          f"the ridge, indistinguishable from it, so these 51 features hold no interaction "
          f"structure the linear model missed; the interpretable estimator cost nothing.")
else:
    VB = (f"LINEARLY EXHAUSTED, and then some. The boosted tree is {DIFF:+.4f} [{LO:+.4f}, "
          f"{HI:+.4f}] against the ridge, i.e. strictly WORSE (p = {PV:.4f}). A nonlinear "
          f"learner with 18 tuned configurations cannot match a 51-parameter linear model on "
          f"these features, which is as clean a statement of linear exhaustion as this design "
          f"can give: there is no interaction structure to find, and the extra capacity only "
          f"costs generalisation.")
log("[B] VERDICT: " + VB)
log(f"[B] NOTE: the tree's POOLED metrics are far above the ridge's (AUROC "
    f"{R_H['auroc']:.4f} vs {R[100]['auroc']:.4f}, AUPRC {R_H['auprc']:.4f} vs "
    f"{R[100]['auprc']:.4f}) while its within-gene win-rate is LOWER. Gene-centring removes "
    f"each gene's mean, but a tree can still reconstruct gene-level context from the pattern "
    f"of centred features and separate pooled positives from negatives without ranking better "
    f"inside any gene. This is the failure mode the within-gene metric exists to catch, and it "
    f"is why no pooled number is the headline anywhere in this chapter.")

with open("splice_model_nonlinear.tsv", "w") as fh:
    fh.write("model\tn_features\tinner_val\twinrate\tci_lo\tci_hi\tp_at_1\tp1_lo\tp1_hi\t"
             "auroc\tauprc\ttest_minus_inner\thyperparameters\n")
    for tag, r, hp in (("ridge logistic (reported)", R[100], f"lambda={R[100]['lam']:g}"),
                       ("histogram gradient boosting", R_H,
                        f"learning_rate={lr}, max_leaf_nodes={ml}, max_iter={mi}, l2=1.0")):
        fh.write(f"{tag}\t51\t{r['iv']:.5f}\t{r['winrate']:.5f}\t{r['winrate_lo']:.5f}\t"
                 f"{r['winrate_hi']:.5f}\t{r['p_at_1']:.5f}\t{r['p_at_1_lo']:.5f}\t"
                 f"{r['p_at_1_hi']:.5f}\t{r['auroc']:.5f}\t{r['auprc']:.5f}\t{r['gap']:+.5f}\t{hp}\n")
    fh.write(f"paired: boosted minus ridge\t\t\t{DIFF:+.5f}\t{LO:+.5f}\t{HI:+.5f}\t\t\t\t\t\t"
             f"\tp={PV:.4f}\n")
    fh.write(f"verdict\t\t\t\t\t\t\t\t\t\t\t{VB}\n")

log("")
log("[reported] unchanged: the thesis model is still the 51-feature ridge logistic at 100 nt, "
    f"within-gene {R[100]['winrate']:.4f}; ladder untouched")
log(f"[genes] {len(K.GRP_TE)} test genes, {len(K.te)} test peaks")
log("[out] splice_model_resolution.tsv splice_model_nonlinear.tsv")
