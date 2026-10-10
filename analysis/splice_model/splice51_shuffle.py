#!/usr/bin/env python3
# dinucleotide-shuffle benchmark of the locked 51-feature splice model
import sys, time
import numpy as np
import binned_common as K
from binned_common import log
sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as L
from cnn_interp_lib import dinuc_shuffle
from maxentpy import maxent
from maxentpy.maxent import load_matrix3

UP, DN, BIN = 2000, 500, 100
NBIN = (UP + DN) // BIN
ME3_LO, ME3_HI = -2000, -126
KSH = 3
NT = np.array(list("ACGT") + ["N"])
M3 = load_matrix3()

D = np.load("cnn_data.npz", allow_pickle=True)
X = D["X"][K.te]
NTE = len(K.te)
log(f"[set] {len(K.GRP_TE)} test genes, {NTE} test peaks; window [-{UP}, +{DN}]")

o9 = np.arange(UP + DN - 9 + 1) + 3 - UP
bidx = (o9 + UP) // BIN
nb9 = np.bincount(bidx, minlength=NBIN).astype(float)

def featurise(tok2d, tag):
    n = len(tok2d)
    out = np.empty((n, 2 * NBIN + 1))
    t0 = time.time()
    for i in range(n):
        t = tok2d[i].astype(np.int64)
        sw = np.lib.stride_tricks.sliding_window_view(t, 9); g = (sw < 4).all(1)
        u = np.nan_to_num(np.where(g, L.U1[np.where(g, sw @ L.POW, 0)], np.nan), nan=-99.0)
        for c, thr in enumerate((4.0, 7.0)):
            s = np.bincount(bidx, weights=(u >= thr).astype(float), minlength=NBIN)
            out[i, c * NBIN:(c + 1) * NBIN] = np.where(nb9 > 0, s / np.maximum(nb9, 1) * 1000.0,
                                                       np.nan)
        seq = "".join(NT[np.where(t < 4, t, 4)])
        lo = max(0, ME3_LO + UP - 19); hi = min(len(seq) - 23 + 1, ME3_HI + UP - 19)
        best = -99.0
        for j in range(lo, hi):
            if t[j + 18] != 0 or t[j + 19] != 2: continue
            k = seq[j:j + 23]
            if "N" in k: continue
            try: v = maxent.score3(k, matrix=M3)
            except Exception: continue
            if v > best: best = v
        out[i, -1] = best
        if (i + 1) % 5000 == 0: log(f"  {tag} {i+1}/{n}  {time.time()-t0:.0f}s")
    log(f"[feat] {tag}: {n} peaks in {time.time()-t0:.0f}s")
    return out

# the fitted model
FM = np.load("splice_model_final.npz", allow_pickle=True)
BETA, MU, SD = FM["beta"], FM["mu"], FM["sd"]
assert len(BETA) == 52
gte = K.gene[K.te]
def centre(Mx):
    out = Mx.astype(float).copy()
    o = np.argsort(gte, kind="stable"); gs = gte[o]
    cut = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1, len(gs)]
    for j in range(len(cut) - 1):
        ix = o[cut[j]:cut[j + 1]]
        mu = np.nanmean(out[ix], 0)
        out[ix] = out[ix] - np.where(np.isfinite(mu), mu, 0.0)
    return out
def score(Mx):
    Cg = centre(Mx)
    Zs = (np.where(np.isfinite(Cg), Cg, MU) - MU) / SD
    return np.column_stack([np.ones(len(Zs)), Zs]) @ BETA
def per_gene(sc):
    return np.array([((sc[r][:, None] > sc[c][None, :]).sum()
                      + 0.5 * (sc[r][:, None] == sc[c][None, :]).sum()) / (len(r) * len(c))
                     for r, c in K.GRP_TE])
def ci(v, nb=2000):
    bs = np.random.default_rng(K.SEED).integers(0, len(v), size=(nb, len(v)))
    m = v[bs].mean(1); return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

REAL = featurise(X, "real")
r = np.corrcoef(REAL[:, -1], K.ACC[K.te])[0, 1]
log(f"[check] re-derived acceptor vs the pipeline's spl_upstream_max_me3: r = {r:.5f}")
assert r > 0.99, "acceptor re-derivation does not track the pipeline column"
wr_real = per_gene(score(REAL))
log(f"[real] within-gene {wr_real.mean():.4f} (the locked model reads 0.6957 on its own "
    f"features; this arm re-derives them from the CNN token window)")

rng = np.random.default_rng(K.SEED)
t0 = time.time()
SH = np.stack([np.stack([dinuc_shuffle(s.astype(np.int64), rng) for s in X])
               for _ in range(KSH)])
log(f"[shuffle] {KSH} dinucleotide shuffles per site in {time.time()-t0:.0f}s -> {SH.shape}")
gcf = lambda a: float(((a == 1) | (a == 2)).mean())
def cpg(a):
    return float(np.mean([np.mean((s[:-1] == 1) & (s[1:] == 2)) for s in a]))
log(f"[check] GC  real {gcf(X):.6f}  shuffled {np.mean([gcf(s) for s in SH]):.6f}")
log(f"[check] CpG real {cpg(X):.6f}  shuffled {np.mean([cpg(s) for s in SH]):.6f}")

wrs = []
for j in range(KSH):
    wrs.append(per_gene(score(featurise(SH[j], f"shuffle {j+1}/{KSH}"))))
    log(f"[shuf] shuffle {j+1}: within-gene {wrs[-1].mean():.4f}")
wr_shuf = np.mean(wrs, 0)
lo_r, hi_r = ci(wr_real); lo_s, hi_s = ci(wr_shuf)
d = wr_real - wr_shuf; lo_d, hi_d = ci(d)
lift = wr_real.mean() - 0.5
pct = 100 * (wr_real.mean() - wr_shuf.mean()) / lift
log("")
log(f"[51] real {wr_real.mean():.4f} [{lo_r:.4f},{hi_r:.4f}]   "
    f"shuffled {wr_shuf.mean():.4f} [{lo_s:.4f},{hi_s:.4f}]   "
    f"drop {d.mean():+.4f} [{lo_d:+.4f},{hi_d:+.4f}]   {pct:.1f}% of the lift over 0.5 lost")

ROWS = [("splice_positional_51", wr_real.mean(), lo_r, hi_r, wr_shuf.mean(), lo_s, hi_s,
         d.mean(), lo_d, hi_d, pct, 51)]
import csv
for rr in csv.DictReader(open("shuffle_benchmark.tsv"), delimiter="\t"):
    if rr["model"] in ("cnn", "splice_selected"):
        ROWS.append((rr["model"], float(rr["real"]), float(rr["real_lo"]), float(rr["real_hi"]),
                     float(rr["shuffled"]), float(rr["shuffled_lo"]), float(rr["shuffled_hi"]),
                     float(rr["drop"]), float(rr["drop_lo"]), float(rr["drop_hi"]),
                     float(rr["pct_of_lift_lost"]), ""))
        log(f"[ctx] {rr['model']}: real {float(rr['real']):.4f} -> shuffled "
            f"{float(rr['shuffled']):.4f}, {float(rr['pct_of_lift_lost']):.1f}% of lift lost")
with open("splice51_shuffle.tsv", "w") as fh:
    fh.write("model\treal\treal_lo\treal_hi\tshuffled\tshuffled_lo\tshuffled_hi\t"
             "drop\tdrop_lo\tdrop_hi\tpct_of_lift_lost\tn_features\tk_shuffles\tn_genes\n")
    for rr in ROWS:
        fh.write("\t".join(f"{x:.5f}" if isinstance(x, float) else str(x) for x in rr)
                 + f"\t{KSH}\t{len(K.GRP_TE)}\n")
log("[out] splice51_shuffle.tsv")
