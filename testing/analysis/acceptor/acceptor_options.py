#!/usr/bin/env python3
# two ways to draw the acceptor in panel a, and what each costs
import sys
import numpy as np
import binned_common as K
from binned_common import log

UP, DN, BIN = 2000, 500, 100
NBIN = (UP + DN) // BIN
ACC = K.ACC; N = len(K.y)
Z = np.load("binned_profiles.npz", allow_pickle=True)
P = Z["P"]; CH = [str(c) for c in Z["channels"]]; BINLO = Z["bin_lo"].astype(int)
D4 = P[:, CH.index("u1_dens4"), :].astype(float)
ME3 = P[:, CH.index("me3_max"), :].astype(float)
NUP = UP // BIN
log(f"[in] u1_dens4 and me3_max, {NBIN} bins of {BIN} nt over [-{UP}, +{DN}]")
log(f"[in] me3_max is finite in {100*np.isfinite(ME3).mean():.1f}% of bins "
    f"({100*np.isfinite(ME3[:, :NUP]).mean():.1f}% upstream)")

MODELS = [("26  U1 x25 + acceptor scalar", np.column_stack([D4, ACC]), 26),
          ("50  U1 x25 + acceptor per bin", np.column_stack([D4, ME3]), 50),
          ("51  U1 x25 + acceptor per bin + scalar", np.column_stack([D4, ME3, ACC]), 51)]
R = {}
for nm, M, p in MODELS:
    assert M.shape[1] == p
    lam, iv = K.pick_lambda(M, nm)
    sc, b = K.fit_score(M, K.trv, K.te, lam)
    r = K.suite(sc); r.update(lam=lam, iv=iv, p=p, beta=b); R[nm] = r
    log(f"[fit] {nm:40s} within-gene {r['winrate']:.4f} "
        f"[{r['winrate_lo']:.4f},{r['winrate_hi']:.4f}]  P@1 {r['p_at_1']:.4f} "
        f"[{r['p_at_1_lo']:.4f},{r['p_at_1_hi']:.4f}]  inner-val {iv:.4f}")
BASE = MODELS[0][0]
assert abs(R[BASE]["winrate"] - 0.6993) < 5e-4, "26-feature anchor not reproduced"
assert abs(R[BASE]["p_at_1"] - 0.4720) < 5e-4
log("[check] the 26-feature model reproduces 0.6993 / 0.4720\n")

def paired(a):
    da = R[a]["_wg"][K.BS].mean(1) - R[BASE]["_wg"][K.BS].mean(1)
    d = R[a]["winrate"] - R[BASE]["winrate"]; lo, hi = np.percentile(da, [2.5, 97.5])
    pv = max(2 * min((da <= 0).mean(), (da >= 0).mean()), 1 / K.NBOOT)
    call = ("gains > 0.005" if (d > 0.005 and lo > 0) else
            "costs > 0.005" if (d < -0.005 and hi < 0) else "NEUTRAL (within noise)")
    log(f"[paired] {a:40s} minus 26-feature  {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
        f"p = {pv:.4f}   {call}")
    return d, lo, hi, pv, call
C = {nm: paired(nm) for nm, _, _ in MODELS[1:]}
H = MODELS[1][0]
log("")
log("[verdict-model] " + (
    f"Option 2's grid is performance-neutral: {C[H][0]:+.4f} [{C[H][1]:+.4f}, {C[H][2]:+.4f}], "
    f"{C[H][4]}. The 25 acceptor bins buy nothing over the one scalar, so if the heatmap is "
    f"drawn it is a drawing device and the 26-feature model stays THE model."
    if C[H][4] == "NEUTRAL (within noise)" else
    f"Option 2's grid is NOT neutral: {C[H][0]:+.4f} [{C[H][1]:+.4f}, {C[H][2]:+.4f}], "
    f"{C[H][4]}. That changes the decision -- state it."))

# coefficients for both drawings
with open("acceptor_options_coefficients.tsv", "w") as fh:
    fh.write("model\tfeature\tchannel\tbin_lo\tbin_centre\tcoefficient\n")
    for nm, _, p in MODELS[:2]:
        b = R[nm]["beta"]
        fh.write(f"{nm}\t(intercept)\t\t\t\t{b[0]:.6f}\n")
        for j in range(NBIN):
            fh.write(f"{nm}\tu1_dens4[{BINLO[j]}]\tu1_dens4\t{BINLO[j]}\t{BINLO[j]+BIN//2}\t"
                     f"{b[1+j]:.6f}\n")
        if p == 26:
            fh.write(f"{nm}\tspl_upstream_max_me3\tacceptor_scalar\t\t\t{b[-1]:.6f}\n")
        else:
            for j in range(NBIN):
                fh.write(f"{nm}\tme3_max[{BINLO[j]}]\tme3_max\t{BINLO[j]}\t{BINLO[j]+BIN//2}\t"
                         f"{b[1+NBIN+j]:.6f}\n")

# where the winning acceptor sits: the scalar's positional origin
up = ME3[:, :NUP].copy()
ok = np.isfinite(up).any(1)
arg = np.full(N, -1, int)
arg[ok] = np.nanargmax(np.where(np.isfinite(up[ok]), up[ok], -np.inf), axis=1)
log(f"[pos] winning upstream acceptor bin resolved for {100*ok.mean():.1f}% of peaks")
te = K.te
rows = []
for lab, m in (("real", K.y[te] == 1), ("competitor", K.y[te] == 0)):
    a = arg[te][m & ok[te]]
    cnt = np.bincount(a, minlength=NUP).astype(float); cnt /= cnt.sum()
    med = BINLO[int(np.median(a))] + BIN // 2
    log(f"[pos] {lab:11s} n={int(m.sum())}  median winning-acceptor bin centre {med:+d} nt  "
        f"modal bin {BINLO[int(np.argmax(cnt))]:+d}")
    for j in range(NUP):
        rows.append((lab, BINLO[j], BINLO[j] + BIN // 2, cnt[j], int(m.sum())))
with open("acceptor_position.tsv", "w") as fh:
    fh.write("class\tbin_lo\tbin_centre\tfraction\tn_peaks\n")
    for r in rows: fh.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.6f}\t{r[4]}\n")

with open("acceptor_options_metrics.tsv", "w") as fh:
    fh.write("model\tn_features\tlambda\tinner_val\twinrate\tci_lo\tci_hi\tp_at_1\tp1_lo\tp1_hi\t"
             "auroc\tauprc\tdelta_vs_26\td_lo\td_hi\tp\tcall\tn_genes\tn_peaks\n")
    for nm, _, p in MODELS:
        r = R[nm]; c = C.get(nm)
        fh.write(f"{nm}\t{p}\t{r['lam']:g}\t{r['iv']:.5f}\t{r['winrate']:.5f}\t"
                 f"{r['winrate_lo']:.5f}\t{r['winrate_hi']:.5f}\t{r['p_at_1']:.5f}\t"
                 f"{r['p_at_1_lo']:.5f}\t{r['p_at_1_hi']:.5f}\t{r['auroc']:.5f}\t{r['auprc']:.5f}\t"
                 + ("\t\t\t\t\t" if c is None else
                    f"{c[0]:+.5f}\t{c[1]:+.5f}\t{c[2]:+.5f}\t{c[3]:.4f}\t{c[4]}\t")
                 + f"{len(K.GRP_TE)}\t{len(K.te)}\n")
log("[out] acceptor_options_metrics.tsv acceptor_options_coefficients.tsv acceptor_position.tsv")
