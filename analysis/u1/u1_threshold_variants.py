#!/usr/bin/env python3
# does the 51-feature splice model need both u1 thresholds?
import sys
import numpy as np
import binned_common as K
from binned_common import log
sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as L

UP, DN, BIN = 2000, 500, 100
NBIN = (UP + DN) // BIN
ACC = K.ACC; N = len(K.y)
X = np.load("cnn_data.npz", allow_pickle=True)["X"]
assert X.shape == (N, UP + DN)

o9 = np.arange(UP + DN - 9 + 1) + 3 - UP
b = (o9 + UP) // BIN; k = (b >= 0) & (b < NBIN); bk = b[k]
nb = np.bincount(bk, minlength=NBIN).astype(float)
D4 = np.empty((N, NBIN)); D7 = np.empty((N, NBIN)); MU = np.empty((N, NBIN))
for i in range(N):
    t = X[i].astype(np.int64)
    sw = np.lib.stride_tricks.sliding_window_view(t, 9); g = (sw < 4).all(1)
    u = np.where(g, L.U1[np.where(g, sw @ L.POW, 0)], np.nan)
    uu = np.nan_to_num(u, nan=-99.0)
    for arr, thr in ((D4, 4.0), (D7, 7.0)):
        s = np.bincount(bk, weights=(uu >= thr)[k].astype(float), minlength=NBIN)
        arr[i] = np.where(nb > 0, s / np.maximum(nb, 1) * 1000.0, np.nan)
    fin = np.isfinite(u)
    sm = np.bincount(bk, weights=np.where(fin, u, 0.0)[k], minlength=NBIN)
    cn = np.bincount(bk, weights=fin[k].astype(float), minlength=NBIN)
    MU[i] = np.where(cn > 0, sm / np.maximum(cn, 1), np.nan)
    if (i + 1) % 20000 == 0: log(f"  {i+1}/{N}")
Z = np.load("binned_profiles.npz", allow_pickle=True)
P = Z["P"]; CH = [str(c) for c in Z["channels"]]
for arr, nm in ((D4, "u1_dens4"), (D7, "u1_dens7")):
    d = np.nanmax(np.abs(arr.astype(np.float32) - P[:, CH.index(nm), :].astype(np.float32)))
    assert d == 0.0, f"{nm} does not reproduce binned_profiles.npz in float32 ({d})"
log("[check] both thresholded U1 channels reproduce binned_profiles.npz exactly in float32")
log(f"[feat] mean U1 -dG per bin: range [{np.nanmin(MU):.3f}, {np.nanmax(MU):.3f}], "
    f"overall mean {np.nanmean(MU):.3f}")

VAR = [("1  locked: >=4 and >=7 + acceptor", np.column_stack([D4, D7, ACC]), 51),
       ("2  >=4 only + acceptor",            np.column_stack([D4, ACC]), 26),
       ("3  >=7 only + acceptor",            np.column_stack([D7, ACC]), 26),
       ("4  mean U1 -dG per bin + acceptor", np.column_stack([MU, ACC]), 26)]
R = {}
log("")
for nm, M, p in VAR:
    assert M.shape[1] == p, (nm, M.shape[1], p)
    lam, iv = K.pick_lambda(M, nm)
    sc, _ = K.fit_score(M, K.trv, K.te, lam)
    r = K.suite(sc); r.update(lam=lam, iv=iv, p=p); R[nm] = r
    log(f"[var] {nm:36s} p={p:>3d}  within-gene {r['winrate']:.4f} "
        f"[{r['winrate_lo']:.4f},{r['winrate_hi']:.4f}]  P@1 {r['p_at_1']:.4f} "
        f"[{r['p_at_1_lo']:.4f},{r['p_at_1_hi']:.4f}]  inner-val {iv:.4f}")
LOCK = VAR[0][0]
assert abs(R[LOCK]["winrate"] - 0.6957) < 5e-4, f"anchor {R[LOCK]['winrate']:.4f} != 0.6957"
log("[check] the locked model reproduces 0.6957\n")

def paired(a, tag):
    da = R[a]["_wg"][K.BS].mean(1) - R[LOCK]["_wg"][K.BS].mean(1)
    d = R[a]["winrate"] - R[LOCK]["winrate"]; lo, hi = np.percentile(da, [2.5, 97.5])
    pv = max(2 * min((da <= 0).mean(), (da >= 0).mean()), 1 / K.NBOOT)
    cost = -d
    call = ("costs > 0.005" if (cost > 0.005 and hi < 0) else
            ("gains > 0.005" if (d > 0.005 and lo > 0) else "within noise"))
    log(f"[paired] {tag:38s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  p = {pv:.4f}   {call}")
    return d, lo, hi, pv, call
C = {nm: paired(nm, f"{nm.split(' ',1)[1].strip()} minus locked") for nm, _, _ in VAR[1:]}

V2, V3, V4 = VAR[1][0], VAR[2][0], VAR[3][0]
log("")
best_single = max((V2, V3, V4), key=lambda n: R[n]["winrate"])
if C[V4][4] in ("within noise", "gains > 0.005") and R[V4]["winrate"] >= R[V2]["winrate"]:
    verd = (f"CONTINUOUS IS CLEANEST. Mean U1 -dG per bin with no threshold at all reads "
            f"{R[V4]['winrate']:.4f}, {C[V4][0]:+.4f} [{C[V4][1]:+.4f}, {C[V4][2]:+.4f}] against the "
            f"locked model and at least as good as the best thresholded single track "
            f"({R[V2]['winrate']:.4f}). Half the features and no cut-off to justify.")
elif C[V2][4] == "within noise":
    verd = (f"ONE THRESHOLD SUFFICES. Keeping only >= 4 reads {R[V2]['winrate']:.4f} against the "
            f"locked {R[LOCK]['winrate']:.4f}, a difference of {C[V2][0]:+.4f} "
            f"[{C[V2][1]:+.4f}, {C[V2][2]:+.4f}] with the CI spanning zero, so the >= 7 track adds "
            f"nothing measurable and the model halves to 26 features. Note the inner split "
            f"disagreed ({R[LOCK]['iv']:.4f} for the locked model against {R[V2]['iv']:.4f}), so "
            f"selection on inner-val would have kept both tracks; on test they are "
            f"indistinguishable. The >= 7 track alone is much worse ({R[V3]['winrate']:.4f}, "
            f"{C[V3][0]:+.4f}), so the two cut-offs are not interchangeable: >= 4 is the one that "
            f"carries the signal. Removing the threshold entirely is NOT cleaner here, the raw "
            f"mean energy reads {R[V4]['winrate']:.4f}, below >= 4 by "
            f"{R[V2]['winrate']-R[V4]['winrate']:.4f}.")
else:
    verd = (f"BOTH THRESHOLDS ARE NEEDED. Dropping >= 7 costs {-C[V2][0]:+.4f} "
            f"[{C[V2][1]:+.4f}, {C[V2][2]:+.4f}] and dropping >= 4 costs {-C[V3][0]:+.4f}; the best "
            f"single-track variant is {best_single.split(' ',1)[1].strip()} at "
            f"{R[best_single]['winrate']:.4f}.")
log(f"[verdict] {verd}")
with open("u1_threshold_variants.tsv", "w") as fh:
    fh.write("variant\tn_features\tlambda\tinner_val\twinrate\tci_lo\tci_hi\tp_at_1\tp1_lo\t"
             "p1_hi\tauroc\tauprc\tdelta_vs_locked\td_lo\td_hi\tp\tcall\tn_genes\tn_peaks\n")
    for nm, _, p in VAR:
        r = R[nm]; c = C.get(nm)
        fh.write(f"{nm}\t{p}\t{r['lam']:g}\t{r['iv']:.5f}\t{r['winrate']:.5f}\t"
                 f"{r['winrate_lo']:.5f}\t{r['winrate_hi']:.5f}\t{r['p_at_1']:.5f}\t"
                 f"{r['p_at_1_lo']:.5f}\t{r['p_at_1_hi']:.5f}\t{r['auroc']:.5f}\t{r['auprc']:.5f}\t"
                 + ("\t\t\t\t\t" if c is None else
                    f"{c[0]:+.5f}\t{c[1]:+.5f}\t{c[2]:+.5f}\t{c[3]:.4f}\t{c[4]}\t")
                 + f"{len(K.GRP_TE)}\t{len(K.te)}\n")
log(f"[genes] {len(K.GRP_TE)} test genes, {len(K.te)} test peaks")
log("[out] u1_threshold_variants.tsv")
