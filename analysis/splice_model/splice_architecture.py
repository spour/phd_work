#!/usr/bin/env python3
# do explicit terminal-exon-architecture features raise the locked 51-feature splice model?
import sys
import numpy as np
import binned_common as K
from binned_common import log
sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as L

UP, DN, BIN = 2000, 500, 100
NBIN = (UP + DN) // BIN
ND5 = 8.0
ACC = K.ACC; N = len(K.y)
X = np.load("cnn_data.npz", allow_pickle=True)["X"]
Z = np.load("binned_profiles.npz", allow_pickle=True)
P = Z["P"]; CHN = [str(c) for c in Z["channels"]]; BINLO = Z["bin_lo"].astype(int)
ME3B = P[:, CHN.index("me3_max"), :]
U1B = np.column_stack([P[:, CHN.index("u1_dens4"), :], P[:, CHN.index("u1_dens7"), :]])
FIN = np.column_stack([U1B, ACC]); assert FIN.shape[1] == 51

# per-position maxent 5'ss from the same lookup, verified against the cached channel
o9 = np.arange(UP + DN - 9 + 1) + 3 - UP
bidx = (o9 + UP) // BIN
HITDN = np.full(N, np.nan); HAS300 = np.zeros(N); IVANY = np.zeros(N); IVMAX = np.full(N, np.nan)
m5max = np.full((N, NBIN), -np.inf)
ACCBIN = np.full(N, -1, int)
FLOOR = -20.0
for i in range(N):
    t = X[i].astype(np.int64)
    sw = np.lib.stride_tricks.sliding_window_view(t, 9); good = (sw < 4).all(1)
    me5 = np.where(good, L.ME5[np.where(good, sw @ L.POW, 0)], -99.0)
    np.maximum.at(m5max[i], bidx, me5)
    hit = me5 >= ND5
    dn = o9[hit & (o9 > 0)]
    HITDN[i] = float(dn.min()) if dn.size else float(DN)
    HAS300[i] = float((dn <= 300).any()) if dn.size else 0.0
    up = ME3B[i, :UP // BIN]
    if np.isfinite(up).any():
        a = int(np.nanargmax(up)); ACCBIN[i] = a
        aoff = BINLO[a] + BIN
        m = hit & (o9 > aoff) & (o9 < 0)
        IVANY[i] = float(m.any())
        IVMAX[i] = float(me5[m].max()) if m.any() else FLOOR
    else:
        IVANY[i] = 0.0; IVMAX[i] = FLOOR
    if (i + 1) % 10000 == 0: log(f"  {i+1}/{N}")
ref = P[:, CHN.index("me5_max"), :]
got = np.where(np.isfinite(m5max) & (m5max > -np.inf), m5max, np.nan)
d = np.nanmax(np.abs(got.astype(np.float32) - ref.astype(np.float32)))
assert d == 0.0, f"recomputed me5_max does not match binned_profiles.npz in float32 ({d})"
log(f"[check] per-position MaxEnt 5'ss reproduces the cached me5_max channel exactly in float32")
log(f"[feat] downstream donor: median distance {np.median(HITDN):.0f} nt, "
    f"{100*np.mean(HITDN < DN):.1f}% have one inside +{DN}; {100*HAS300.mean():.1f}% within +300")
log(f"[feat] intervening donor: {100*IVANY.mean():.1f}% of peaks have a strong donor between "
    f"the acceptor bin and the site")
for nm, v in (("downstream present<=300", HAS300), ("intervening present", IVANY)):
    r = v[K.y == 1].mean(); c = v[K.y == 0].mean()
    log(f"[feat]   {nm:26s} real {r:.4f} vs competitor {c:.4f}  (difference {r-c:+.4f})")

DNF = np.column_stack([HITDN, HAS300])
IVF = np.column_stack([IVANY, IVMAX])
MODELS = [("locked (51)", FIN, 51),
          ("+ downstream donor (2)", np.column_stack([FIN, DNF]), 53),
          ("+ intervening donor (2)", np.column_stack([FIN, IVF]), 53),
          ("+ both (4)", np.column_stack([FIN, DNF, IVF]), 55)]
R = {}
log("")
for nm, M, p in MODELS:
    assert M.shape[1] == p
    lam, iv = K.pick_lambda(M, nm)
    sc, _ = K.fit_score(M, K.trv, K.te, lam)
    r = K.suite(sc); r.update(lam=lam, iv=iv, p=p, sc=sc); R[nm] = r
    log(f"[row] {nm:26s} p={p:>3d}  within-gene {r['winrate']:.4f} "
        f"[{r['winrate_lo']:.4f},{r['winrate_hi']:.4f}]  P@1 {r['p_at_1']:.4f} "
        f"[{r['p_at_1_lo']:.4f},{r['p_at_1_hi']:.4f}]")
assert abs(R["locked (51)"]["winrate"] - 0.6957) < 5e-4, "anchor not reproduced"
log("[check] the locked model reproduces 0.6957 before anything is added\n")

def paired(a, b, tag):
    da = R[a]["_wg"][K.BS].mean(1) - R[b]["_wg"][K.BS].mean(1)
    d_ = R[a]["winrate"] - R[b]["winrate"]; lo, hi = np.percentile(da, [2.5, 97.5])
    pv = max(2 * min((da <= 0).mean(), (da >= 0).mean()), 1 / K.NBOOT)
    call = "clears +0.005" if (d_ > 0.005 and lo > 0) else \
           ("CI spans 0" if lo * hi <= 0 else "below +0.005")
    log(f"[paired] {tag:36s} {d_:+.4f} [{lo:+.4f}, {hi:+.4f}]  p = {pv:.4f}   {call}")
    return d_, lo, hi, pv, call
C = {nm: paired(nm, "locked (51)", f"{nm} minus locked") for nm, _, _ in MODELS[1:]}
CLEARS = [nm for nm, v in C.items() if v[4] == "clears +0.005"]
BEST = max(C, key=lambda n: C[n][0])

# localization: does the best addition help where the acceptor is weak?
log("")
accg = np.array([ACC[K.te[r[0]]] for r, _ in K.GRP_TE])
med = np.median(accg)
WEAK = accg <= med; STRONG = ~WEAK
log(f"[loc] acceptor median {med:.3f}; weak {int(WEAK.sum())} genes, strong {int(STRONG.sum())}")
def p1v(nm):
    sc = R[nm]["sc"]
    return np.array([float(sc[r].max() > sc[c].max()) + 0.5 * float(sc[r].max() == sc[c].max())
                     for r, c in K.GRP_TE])
b0 = p1v("locked (51)"); b1 = p1v(BEST)
for lab, m in (("weak acceptor", WEAK), ("strong acceptor", STRONG)):
    d_ = b1[m].mean() - b0[m].mean()
    bs = np.random.default_rng(K.SEED).integers(0, int(m.sum()), size=(2000, int(m.sum())))
    dd = b1[m][bs].mean(1) - b0[m][bs].mean(1); lo, hi = np.percentile(dd, [2.5, 97.5])
    log(f"[loc] {BEST} minus locked, P@1 in {lab:16s}: {b0[m].mean():.4f} -> {b1[m].mean():.4f} "
        f"= {d_:+.4f} [{lo:+.4f}, {hi:+.4f}]")

with open("splice_architecture.tsv", "w") as fh:
    fh.write("section\tkey\tn_features\tlambda\tinner_val\twinrate\tci_lo\tci_hi\tp_at_1\t"
             "p1_lo\tp1_hi\tdelta\td_lo\td_hi\tp\tcall\n")
    for nm, _, p in MODELS:
        r = R[nm]
        fh.write(f"model\t{nm}\t{p}\t{r['lam']:g}\t{r['iv']:.5f}\t{r['winrate']:.5f}\t"
                 f"{r['winrate_lo']:.5f}\t{r['winrate_hi']:.5f}\t{r['p_at_1']:.5f}\t"
                 f"{r['p_at_1_lo']:.5f}\t{r['p_at_1_hi']:.5f}\t\t\t\t\t\n")
    for nm, v in C.items():
        fh.write(f"contrast\t{nm} minus locked\t\t\t\t\t\t\t\t\t\t"
                 f"{v[0]:+.5f}\t{v[1]:+.5f}\t{v[2]:+.5f}\t{v[3]:.4f}\t{v[4]}\n")
log("")
if CLEARS:
    log(f"[verdict] REAL ARCHITECTURE SIGNAL: {', '.join(CLEARS)} clears +0.005 with a CI clear "
        f"of zero. The acceptor scalar did not already carry it.")
else:
    log(f"[verdict] THE ACCEPTOR ALREADY CAPTURED IT. No architecture feature clears +0.005; the "
        f"best is {BEST} at {C[BEST][0]:+.4f} [{C[BEST][1]:+.4f}, {C[BEST][2]:+.4f}], {C[BEST][4]}. "
        f"Explicit terminal-exon geometry adds nothing the upstream acceptor score and positional "
        f"U1 density do not already encode, so 0.696 within gene stands as the splice ceiling and "
        f"the whole-genome 0.749 with it. No whole-genome rerun is warranted.")
log("[out] splice_architecture.tsv")
