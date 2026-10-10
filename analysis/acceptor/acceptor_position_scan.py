#!/usr/bin/env python3
# where does the max-scoring upstream 3' splice site sit?
import sys, time
import numpy as np
from multiprocessing import Pool
import u1_5ss_landscape_data as L
import gc_metagene_data as GM

UP, DN, PAD = 2000, 500, 40
LO, HI = -2000, -126
THR = 6.5
NT = np.array(list("ACGT") + ["N"])
CODE = np.full(256, 4, np.int64)
for i, ch in enumerate("ACGT"): CODE[ord(ch)] = i
log = L.log
SITES = {}
M3 = None

def _init():
    global M3
    from maxentpy.maxent import load_matrix3
    M3 = load_matrix3()
    L.G_["G"].fh = open(L.FASTA, "rb")

def _one(i):
    from maxentpy import maxent
    ch = SITES["chrom"][i]; c = int(SITES["center"][i]); st = SITES["strand"][i]
    a, b = (c - UP - PAD, c + DN + PAD) if st == "+" else (c - DN - PAD + 1, c + UP + PAD + 1)
    s = L.G_["G"].fetch(ch, a, b, st)
    if not s or len(s) != UP + DN + 2 * PAD: return (np.nan, np.nan)
    t = CODE[np.frombuffer(s.encode("ascii"), np.uint8)]
    j0 = max(0, LO + UP + PAD - 18); j1 = min(len(t) - 23 + 1, HI + UP + PAD - 18 + 1)
    best, bo = -99.0, np.nan
    for j in range(j0, j1):
        if t[j + 18] != 0 or t[j + 19] != 2: continue
        k = s[j:j + 23]
        if "N" in k: continue
        try: v = maxent.score3(k, matrix=M3)
        except Exception: continue
        if v > best: best, bo = v, j + 18 - (UP + PAD)
    return (best if best > -99.0 else np.nan, bo)

if __name__ == "__main__":
    S = GM.build_sites(); n = len(S["cls"])
    SITES.update(S)
    log(f"[scope] {n} sites: " + "  ".join(f"{c} {int((S['cls']==c).sum())}"
                                           for c in ("real", "competitor", "decoy")))
    log(f"[window] [{LO}, {HI}] transcript-oriented; acceptor position is the AG of the "
        f"23-mer; threshold for a real acceptor MaxEnt score3 >= {THR}")
    t0 = time.time()
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 16, initializer=_init) as P:
        out = P.map(_one, range(n), chunksize=64)
    sc = np.array([o[0] for o in out]); po = np.array([o[1] for o in out])
    log(f"[scan] {n} sites in {time.time()-t0:.0f}s; a scored AG was found in "
        f"{100*np.isfinite(sc).mean():.2f}% of windows")
    with open("acceptor_position_scan.tsv", "w") as fh:
        fh.write("gene\tchrom\tcenter\tstrand\tclass\tmax_me3\tposition\n")
        for i in range(n):
            fh.write(f"{S['gene'][i]}\t{S['chrom'][i]}\t{int(S['center'][i])}\t"
                     f"{S['strand'][i]}\t{S['cls'][i]}\t"
                     f"{'' if not np.isfinite(sc[i]) else f'{sc[i]:.4f}'}\t"
                     f"{'' if not np.isfinite(po[i]) else int(po[i])}\n")

    def q(v): return np.percentile(v, [25, 50, 75])
    rows = []
    log("")
    for lab, m in (("all windows", np.isfinite(po)),
                   (f"max >= {THR}", np.isfinite(po) & (sc >= THR))):
        log(f"[{lab}]")
        for c in ("real", "competitor", "decoy"):
            k = m & (S["cls"] == c)
            v = po[k]
            if not v.size: continue
            lo, md, hi = q(v)
            log(f"[{lab:12s}] {c:11s} n={int(k.sum()):>6d} ({100*k.sum()/(S['cls']==c).sum():5.1f}% of class)"
                f"  median {md:+8.0f}  IQR {hi-lo:7.0f}  [{lo:+.0f}, {hi:+.0f}]  "
                f"within 500 nt {100*np.mean(v>=-500):5.1f}%")
            rows.append((lab, c, int(k.sum()), md, lo, hi, hi - lo,
                         float(np.mean(v >= -500)), float(np.median(sc[k]))))
        log("")
    log("[by score] IQR of the position within tertiles of the maximum score")
    fin = np.isfinite(po) & np.isfinite(sc)
    for c in ("real", "competitor", "decoy"):
        k = fin & (S["cls"] == c)
        cut = np.percentile(sc[k], [100 / 3, 200 / 3])
        parts = []
        for t_, (a_, b_) in enumerate((( -1e9, cut[0]), (cut[0], cut[1]), (cut[1], 1e9))):
            kk = k & (sc > a_) & (sc <= b_)
            lo, md, hi = q(po[kk])
            parts.append(f"T{t_+1} med {md:+.0f} IQR {hi-lo:.0f}")
            rows.append((f"score tertile {t_+1}", c, int(kk.sum()), md, lo, hi, hi - lo,
                         float(np.mean(po[kk] >= -500)), float(np.median(sc[kk]))))
        log(f"[by score] {c:11s} " + "   ".join(parts))
    with open("acceptor_position_stats.tsv", "w") as fh:
        fh.write("subset\tclass\tn\tmedian\tq25\tq75\tiqr\tfrac_within_500\tmedian_score\n")
        for r in rows:
            fh.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.1f}\t{r[4]:.1f}\t{r[5]:.1f}\t{r[6]:.1f}\t"
                     f"{r[7]:.5f}\t{r[8]:.4f}\n")
    log("[out] acceptor_position_scan.tsv acceptor_position_stats.tsv")

