#!/usr/bin/env python3
# per-site positional splice profiles, 100 nt bins over [-2000, +500]
import sys, os, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as L
from cnn_interp_lib import dinuc_shuffle

log = L.log
B, POW, ME5, U1 = L.B, L.POW, L.ME5, L.U1
UP, DN, BIN = 2000, 500, 100
NBIN = (UP + DN) // BIN
CH = ("u1_dens4", "u1_dens7", "me5_dens8", "me5_max", "me3_max")
NCH = len(CH)
THR4, THR7, ND5 = 4.0, 7.0, 8.0
COMP_LO, COMP_HI = -2000, -500
NT = np.array(list("ACGT") + ["N"])
SEED = 42

M3 = None
def _init():
    global M3
    from maxentpy.maxent import load_matrix3
    M3 = load_matrix3()

def _profile(tok):
    from maxentpy import maxent
    out = np.full((NCH, NBIN), np.nan)
    t = tok.astype(np.int64)
    sw = np.lib.stride_tricks.sliding_window_view(t, 9)
    good = (sw < 4).all(1)
    code = np.where(good, sw @ POW, 0)
    me5 = np.where(good, ME5[code], np.nan)
    u1  = np.where(good, U1[code], np.nan)
    off9 = np.arange(me5.shape[0]) + 3 - UP
    b9 = (off9 + UP) // BIN
    keep9 = (b9 >= 0) & (b9 < NBIN)
    b9k = b9[keep9]
    n9 = np.bincount(b9k, minlength=NBIN).astype(float)
    def dens(hit):
        s = np.bincount(b9k, weights=hit[keep9].astype(float), minlength=NBIN)
        return np.where(n9 > 0, s / np.maximum(n9, 1) * 1000.0, np.nan)
    out[0] = dens(np.nan_to_num(u1, nan=-99.0) >= THR4)
    out[1] = dens(np.nan_to_num(u1, nan=-99.0) >= THR7)
    out[2] = dens(np.nan_to_num(me5, nan=-99.0) >= ND5)
    m5 = np.full(NBIN, -np.inf)
    np.maximum.at(m5, b9k, np.nan_to_num(me5[keep9], nan=-99.0))
    out[3] = np.where(np.isfinite(m5) & (n9 > 0), m5, np.nan)

    seq = "".join(NT[np.where(t < 4, t, 4)])
    n23 = len(seq) - 23 + 1
    ag = np.flatnonzero((t[18:18 + n23] == 0) & (t[19:19 + n23] == 2))
    m3 = np.full(NBIN, -np.inf)
    comp = -np.inf
    for i in ag:
        i = int(i); k = seq[i:i + 23]
        if "N" in k: continue
        try: v = maxent.score3(k, matrix=M3)
        except Exception: continue
        o = i + 19 - UP
        b = (o + UP) // BIN
        if 0 <= b < NBIN and v > m3[b]: m3[b] = v
        if COMP_LO <= o < COMP_HI and v > comp: comp = v
    out[4] = np.where(np.isfinite(m3), m3, np.nan)
    return out, (comp if np.isfinite(comp) else np.nan)

SH = {}
def _job(i):
    tok = SH["X"][i]
    p, c = _profile(tok)
    rng = np.random.default_rng(SEED + i)
    s = dinuc_shuffle(tok.astype(np.int64), rng).astype(np.uint8)
    ps, cs = _profile(s)
    return p, c, ps, cs

if __name__ == "__main__":
    D = np.load("cnn_data.npz", allow_pickle=True)
    ok = D["ok"]; X = D["X"]
    assert X.shape == (int(ok.sum()), UP + DN), f"unexpected token array {X.shape}"
    assert int(D["up"]) == UP and int(D["dn"]) == DN, "window mismatch with cnn_data.npz"
    SH["X"] = X
    N = len(X)
    log(f"token array {X.shape}, window [-{UP},+{DN}], {NCH} channels x {NBIN} bins")

    t0 = time.time()
    with Pool(min(32, os.cpu_count() or 8), initializer=_init) as p:
        res = p.map(_job, range(N), chunksize=32)
    P  = np.stack([r[0] for r in res]).astype(np.float32)
    CU = np.array([r[1] for r in res], np.float32)
    PS = np.stack([r[2] for r in res]).astype(np.float32)
    CS = np.array([r[3] for r in res], np.float32)
    log(f"scanned {N} sites (+1 dinuc shuffle each) in {time.time()-t0:.0f}s")

    f = CU[np.isfinite(CU)]
    log(f"[comp_up_3ss FIXED] genuine MaxEnt-max over [{COMP_LO},{COMP_HI}): "
        f"n={len(f)} finite, min {f.min():.2f}, median {np.median(f):.2f}, max {f.max():.2f}")
    assert f.max() > 2.0, "fixed comp_up_3ss is not on the MaxEnt scale"
    for c in range(NCH):
        v = P[:, c, :][np.isfinite(P[:, c, :])]
        log(f"[channel] {CH[c]:10s} finite {v.size/(N*NBIN):.3f}  "
            f"mean {v.mean():+.3f}  sd {v.std():.3f}")
    np.savez_compressed("binned_profiles.npz", P=P, PS=PS, comp_up_fixed=CU,
                        comp_up_fixed_sh=CS, channels=np.array(CH),
                        bin_lo=np.arange(NBIN) * BIN - UP, up=UP, dn=DN, nbin=NBIN)
    log("[out] binned_profiles.npz")
