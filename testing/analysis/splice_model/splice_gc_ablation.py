#!/usr/bin/env python3
# inside the 0.740 model, what do the splice features add over positional gc alone?
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

t = X.reshape(N, NBIN, BIN)
nok = (t < 4).sum(2).astype(float)
GCB = np.where(nok > 0, ((t == 1) | (t == 2)).sum(2) / np.maximum(nok, 1), np.nan)

o9 = np.arange(UP + DN - 9 + 1) + 3 - UP
b = (o9 + UP) // BIN; k = (b >= 0) & (b < NBIN); bk = b[k]
nb = np.bincount(bk, minlength=NBIN).astype(float)
U1B = np.empty((N, 2 * NBIN))
for i in range(N):
    ti = X[i].astype(np.int64)
    sw = np.lib.stride_tricks.sliding_window_view(ti, 9); g = (sw < 4).all(1)
    u = np.nan_to_num(np.where(g, L.U1[np.where(g, sw @ L.POW, 0)], np.nan), nan=-99.0)
    for c, thr in enumerate((4.0, 7.0)):
        s = np.bincount(bk, weights=(u >= thr)[k].astype(float), minlength=NBIN)
        U1B[i, c * NBIN:(c + 1) * NBIN] = np.where(nb > 0, s / np.maximum(nb, 1) * 1000.0, np.nan)
Z = np.load("binned_profiles.npz", allow_pickle=True)
P = Z["P"]; CHN = [str(c) for c in Z["channels"]]
ref = np.column_stack([P[:, CHN.index("u1_dens4"), :], P[:, CHN.index("u1_dens7"), :]])
assert np.nanmax(np.abs(U1B.astype(np.float32) - ref.astype(np.float32))) == 0.0
log(f"[u1] float64 rebinning matches binned_profiles.npz exactly in float32")
log(f"[gc] GC in {NBIN} bins of {BIN} nt over [-{UP}, +{DN}], mean {np.nanmean(GCB):.4f}")

ROWS = [("1  GC per bin", GCB, 25),
        ("2  GC per bin + acceptor", np.column_stack([GCB, ACC]), 26),
        ("3  GC per bin + U1 density", np.column_stack([GCB, U1B]), 75),
        ("4  GC per bin + acceptor + U1 density", np.column_stack([GCB, ACC, U1B]), 76)]
R = {}
for tag, M, p in ROWS:
    assert M.shape[1] == p, (tag, M.shape[1], p)
    lam, iv = K.pick_lambda(M, tag)
    sc, _ = K.fit_score(M, K.trv, K.te, lam)
    r = K.suite(sc); r.update(lam=lam, iv=iv, p=p); R[tag] = r
    log(f"[row] {tag:38s} p={p:>3d}  within-gene {r['winrate']:.4f} "
        f"[{r['winrate_lo']:.4f},{r['winrate_hi']:.4f}]  P@1 {r['p_at_1']:.4f} "
        f"[{r['p_at_1_lo']:.4f},{r['p_at_1_hi']:.4f}]  AUROC {r['auroc']:.4f}")
A4 = "4  GC per bin + acceptor + U1 density"
assert abs(R[A4]["winrate"] - 0.7400) < 5e-4, \
    f"row 4 does not reproduce 0.7400 (got {R[A4]['winrate']:.4f})"
log("[check] row 4 reproduces the 0.7400 anchor\n")

def paired(a, b, tag):
    da = R[a]["_wg"][K.BS].mean(1) - R[b]["_wg"][K.BS].mean(1)
    d = R[a]["winrate"] - R[b]["winrate"]
    lo, hi = np.percentile(da, [2.5, 97.5])
    pv = max(2 * min((da <= 0).mean(), (da >= 0).mean()), 1 / K.NBOOT)
    call = "clears +0.005" if (d > 0.005 and lo > 0) else \
           ("CI spans 0" if lo * hi <= 0 else "below +0.005")
    log(f"[paired] {tag:46s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  p = {pv:.4f}   {call}")
    return d, lo, hi, pv, call

R1, R2, R3 = "1  GC per bin", "2  GC per bin + acceptor", "3  GC per bin + U1 density"
C = {
 "acceptor over GC":            paired(R2, R1, "acceptor added to GC (2 - 1)"),
 "U1 over GC":                  paired(R3, R1, "U1 density added to GC (3 - 1)"),
 "both over GC":                paired(A4, R1, "acceptor + U1 added to GC (4 - 1)"),
 "U1 over GC+acceptor":         paired(A4, R2, "U1 density added to GC + acceptor (4 - 2)"),
 "acceptor over GC+U1":         paired(A4, R3, "acceptor added to GC + U1 (4 - 3)"),
}
dA, dU, dB = C["acceptor over GC"][0], C["U1 over GC"][0], C["both over GC"][0]
dU2, dA3 = C["U1 over GC+acceptor"], C["acceptor over GC+U1"]
log("")
log(f"[verdict] Splice over GC alone: the acceptor adds {dA:+.4f} and U1 density adds "
    f"{dU:+.4f}; together they add {dB:+.4f}.")
log("[verdict] Does the acceptor carry splice's contribution over GC? " + (
    f"YES. Adding the acceptor to GC gets {R[R2]['winrate']:.4f} of the full model's "
    f"{R[A4]['winrate']:.4f}, and U1 on top of GC + acceptor is worth {dU2[0]:+.4f} "
    f"[{dU2[1]:+.4f}, {dU2[2]:+.4f}], {dU2[4]}."
    if dU2[0] <= 0.005 or dU2[1] <= 0 else
    f"NO. U1 still adds {dU2[0]:+.4f} [{dU2[1]:+.4f}, {dU2[2]:+.4f}] on top of GC + acceptor."))
log("[verdict] Does U1 density add anything over positional GC? " + (
    f"NO, not on its own terms: {dU:+.4f} [{C['U1 over GC'][1]:+.4f}, "
    f"{C['U1 over GC'][2]:+.4f}], {C['U1 over GC'][4]}. The U1 signal is largely redundant "
    f"with positional composition, which is what the dinucleotide-shuffle result predicted."
    if not (dU > 0.005 and C["U1 over GC"][1] > 0) else
    f"YES: {dU:+.4f} [{C['U1 over GC'][1]:+.4f}, {C['U1 over GC'][2]:+.4f}], "
    f"{C['U1 over GC'][4]}. U1 is not reducible to positional composition."))
with open("splice_gc_ablation.tsv", "w") as fh:
    fh.write("section\trow\tn_features\tlambda\tinner_val\twinrate\tci_lo\tci_hi\tp_at_1\t"
             "p1_lo\tp1_hi\tauroc\tauprc\tdelta\td_lo\td_hi\tp\tcall\tn_genes\tn_peaks\n")
    for tag, _, p in ROWS:
        r = R[tag]
        fh.write(f"row\t{tag}\t{p}\t{r['lam']:g}\t{r['iv']:.5f}\t{r['winrate']:.5f}\t"
                 f"{r['winrate_lo']:.5f}\t{r['winrate_hi']:.5f}\t{r['p_at_1']:.5f}\t"
                 f"{r['p_at_1_lo']:.5f}\t{r['p_at_1_hi']:.5f}\t{r['auroc']:.5f}\t"
                 f"{r['auprc']:.5f}\t\t\t\t\t\t{len(K.GRP_TE)}\t{len(K.te)}\n")
    for nm, (d, lo, hi, pv, call) in C.items():
        fh.write(f"contrast\t{nm}\t\t\t\t\t\t\t\t\t\t\t\t{d:+.5f}\t{lo:+.5f}\t{hi:+.5f}\t"
                 f"{pv:.4f}\t{call}\t{len(K.GRP_TE)}\t{len(K.te)}\n")
log(f"\n[genes] {len(K.GRP_TE)} test genes, {len(K.te)} test peaks")
log("[out] splice_gc_ablation.tsv")
