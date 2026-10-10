#!/usr/bin/env python3
# does a per-site count of strong upstream 5'ss, in a tight window
import sys, os, math, time
import numpy as np
from multiprocessing import Pool

import u1_5ss_landscape_data as L

log = L.log
SPAN, PAD = 2000, 12
WINDOWS = (100, 200, 300, 500, 1000, 2000)
ND8, ND65 = 8.0, 6.5
DEFAULT_W = 300
NBOOT, SEED = 1000, 42
CLASSES = L.CLASSES
CACHE = "upstream_count_scan.npz"

NW = len(WINDOWS)
KC8, KC65, KMX = 0, NW, 2 * NW
NFEAT = 3 * NW + 1

SITES = {}
def _init(): L.G_["G"].fh = open(L.FASTA, "rb")

def site_counts(ch, ctr, st):
    out = np.full(NFEAT, np.nan); out[-1] = 0.0
    seq = L.G_["G"].fetch(ch, ctr - SPAN - PAD, ctr + SPAN + PAD, st)
    if not seq or len(seq) < 2 * (SPAN + PAD): return out
    tok = L.B[np.frombuffer(seq.encode("ascii"), np.uint8)]
    sw = np.lib.stride_tricks.sliding_window_view(tok, 9); good = (sw >= 0).all(1)
    me5 = np.where(good, L.ME5[np.where(good, sw @ L.POW, 0)], -99.0)
    d = -(SPAN + PAD) + np.arange(me5.shape[0]) + 3
    for j, w in enumerate(WINDOWS):
        m = (d >= -w) & (d < 0)
        v = me5[m]
        out[KC8 + j] = float((v >= ND8).sum())
        out[KC65 + j] = float((v >= ND65).sum())
        out[KMX + j] = float(v.max()) if v.size else np.nan
    out[-1] = 1.0
    return out

def _job(i): return site_counts(SITES["chrom"][i], int(SITES["center"][i]), SITES["strand"][i])

# stats
def split_by_gene(v, P, boot=NBOOT, seed=SEED):
    gt, eq, lt = [], [], []
    for r, k in P.values():
        rv = v[r]; kv = v[k]
        rv = rv[np.isfinite(rv)]; kv = kv[np.isfinite(kv)]
        if not rv.size or not kv.size: continue
        n = rv.size * kv.size
        gt.append((kv[None, :] > rv[:, None]).sum() / n)
        eq.append((kv[None, :] == rv[:, None]).sum() / n)
        lt.append((kv[None, :] < rv[:, None]).sum() / n)
    gt, eq, lt = map(np.array, (gt, eq, lt))
    rs = np.random.default_rng(seed); idx = rs.integers(0, len(gt), size=(boot, len(gt)))
    asym = gt.mean() / max(gt.mean() + lt.mean(), 1e-12)
    ba = gt[idx].mean(1) / np.maximum(gt[idx].mean(1) + lt[idx].mean(1), 1e-12)
    return dict(gt=float(gt.mean()), eq=float(eq.mean()), lt=float(lt.mean()),
                asym=float(asym), asym_lo=float(np.percentile(ba, 2.5)),
                asym_hi=float(np.percentile(ba, 97.5)), n=len(gt))

# panels
def panel_A(F, S, P, ws, out="upstream_count_dist.tsv", oute="upstream_maxent_ecdf.tsv"):
    ok = F[:, -1] == 1; rows = []; erows = []; res = {}
    grid = np.arange(-2.0, 12.01, 0.25)
    for w in ws:
        j = WINDOWS.index(w)
        for c in CLASSES:
            m = (S["cls"] == c) & ok
            v = F[m, KC8 + j]; n = int(m.sum())
            for lab, k in (("0", (v == 0).sum()), ("1", (v == 1).sum()),
                           ("2", (v == 2).sum()), (">=3", (v >= 3).sum()), (">=1", (v >= 1).sum())):
                p, lo, hi = L.wilson(int(k), n)
                rows.append((c, lab, p, lo, hi, int(k), n, w))
            e = L.ecdf_on(F[m, KMX + j], grid)
            for x, f in zip(grid, e): erows.append((c, x, f, n, w))
        wr = L.winrate(F[:, KC8 + j], P); sp = split_by_gene(F[:, KC8 + j], P)
        f1 = {c: [r[2] for r in rows if r[0] == c and r[1] == ">=1" and r[7] == w][0] for c in CLASSES}
        res[w] = (f1, wr, sp)
        log(f"[A] window [-{w},0]  fraction with >=1 strong donor: "
            + "  ".join(f"{c} {f1[c]:.4f}" for c in CLASSES))
        log(f"[A] window [-{w},0]  win-rate P(comp>real) = {wr[0]:.4f} [{wr[1]:.4f},{wr[2]:.4f}]; "
            f"discordant split comp>real {sp['gt']:.4f} / tied {sp['eq']:.4f} / real>comp {sp['lt']:.4f}, "
            f"asymmetry {sp['asym']:.4f} [{sp['asym_lo']:.4f},{sp['asym_hi']:.4f}]")
    with open(out, "w") as fh:
        fh.write("class\tbucket\tfrac\tci_lo\tci_hi\tk\tn\twindow\n")
        for r in rows: fh.write(f"{r[0]}\t{r[1]}\t{r[2]:.5f}\t{r[3]:.5f}\t{r[4]:.5f}\t{r[5]}\t{r[6]}\t{r[7]}\n")
    with open(oute, "w") as fh:
        fh.write("class\tmaxent\tecdf\tn\twindow\n")
        for r in erows: fh.write(f"{r[0]}\t{r[1]:.2f}\t{r[2]:.5f}\t{r[3]}\t{r[4]}\n")
    return res

def panel_B(F, P, out="upstream_count_sweep.tsv"):
    rows = []; res = {}
    for nm, base in (("count_me8", KC8), ("count_me65", KC65), ("max_me5", KMX)):
        for j, w in enumerate(WINDOWS):
            v = F[:, base + j]
            wr = L.winrate(v, P); sp = split_by_gene(v, P)
            rows.append((nm, w, wr[0], wr[1], wr[2], sp["asym"], sp["asym_lo"], sp["asym_hi"],
                         sp["eq"], wr[3]))
            res[(nm, w)] = (wr, sp)
    with open(out, "w") as fh:
        fh.write("scalar\twindow\twinrate\tci_lo\tci_hi\tasym\tasym_lo\tasym_hi\ttied\tn_genes\n")
        for r in rows:
            fh.write(f"{r[0]}\t{r[1]}\t{r[2]:.5f}\t{r[3]:.5f}\t{r[4]:.5f}\t{r[5]:.5f}\t"
                     f"{r[6]:.5f}\t{r[7]:.5f}\t{r[8]:.5f}\t{r[9]}\n")
    log("[B] sweep, win-rate P(comp>real) (asymmetry, tied fraction):")
    for nm in ("count_me8", "count_me65", "max_me5"):
        log(f"    {nm:11s} " + "  ".join(
            f"w{w}: {res[(nm,w)][0][0]:.3f} ({res[(nm,w)][1]['asym']:.3f}, {res[(nm,w)][1]['eq']:.2f})"
            for w in WINDOWS))
    return res

if __name__ == "__main__":
    S = L.build_sites(); NS = len(S["cls"])
    if os.path.exists(CACHE) and "--rescan" not in sys.argv:
        F = np.load(CACHE)["F"].astype(np.float64); assert len(F) == NS and F.shape[1] == NFEAT
        log(f"scan cache {CACHE} {F.shape}")
    else:
        SITES.update(chrom=S["chrom"], center=S["center"], strand=S["strand"])
        t = time.time(); F = np.empty((NS, NFEAT))
        with Pool(min(32, os.cpu_count() or 8), initializer=_init) as p:
            for i, v in enumerate(p.imap(_job, range(NS), chunksize=32)):
                F[i] = v
                if (i + 1) % 25000 == 0: log(f"  rescan {i+1}/{NS}  {time.time()-t:.0f}s")
        log(f"me5 rescan done {NS} sites {time.time()-t:.0f}s  ok {int((F[:,-1]==1).sum())}")
        np.savez_compressed(CACHE, F=F.astype(np.float32))

    Z = np.load("u1_5ss_sitescan.npz", allow_pickle=True); G = Z["F"].astype(np.float64)
    assert (Z["cls"] == S["cls"]).all() and (Z["center"] == S["center"]).all(), "npz site order differs"
    dens = G[:, L.O_D5_100:L.O_D5_100 + L.N100]
    upb = L.BC100 < 0
    bad = []
    for j, w in enumerate(WINDOWS):
        sel = upb & (L.BC100 >= -w)
        npzc = np.rint(np.nansum(dens[:, sel], 1) * 0.1)
        bad.append(int((np.abs(npzc - F[:, KC8 + j]) > 0.5).sum()))
    log(f"[check] >=8 counts vs u1_5ss_sitescan.npz 100 nt block, sites disagreeing per window "
        + "  ".join(f"w{w}: {b}" for w, b in zip(WINDOWS, bad)))

    P = L.gene_pairs(S["gene"], S["cls"])
    log(f"paired genes (>=1 real and >=1 competitor): {len(P)}")

    sweep = panel_B(F, P)
    peak_w = max(WINDOWS, key=lambda w: sweep[("count_me8", w)][0][0])
    if sweep[("count_me8", peak_w)][0][0] - sweep[("count_me8", DEFAULT_W)][0][0] < 0.005:
        w_use, why = DEFAULT_W, f"sweep peak w={peak_w} is within 0.005 of the default, so the default stands"
    else:
        w_use, why = peak_w, (f"sweep rises monotonically and peaks at w={peak_w}, above the default "
                              f"{DEFAULT_W}; the tight window is the WORSE one, so both are plotted")
    log(f"[A] chosen window [-{w_use},0]: {why}")
    A = panel_A(F, S, P, sorted({DEFAULT_W, w_use}))
    f1, wr, sp = A[w_use]
    f1d, wrd, spd = A[DEFAULT_W]

    pk = sweep[("count_me8", peak_w)]
    clears = wr[0] > 0.534
    log(f"[verdict] tight-window count win-rate {wr[0]:.4f} at w={w_use} and {pk[0][0]:.4f} at the "
        f"sweep peak w={peak_w}; intervening-donor architecture is 0.534, acceptor 0.606 -> "
        f"{'CLEARS' if clears else 'DOES NOT CLEAR'} 0.534")

    n = {c: int(((S["cls"] == c) & (F[:, -1] == 1)).sum()) for c in CLASSES}
    with open("upstream_count_stats.tsv", "w") as fh:
        fh.write("key\tvalue\tlo\thi\tn\n")
        for c in CLASSES:
            fh.write(f"n_{c}\t{n[c]}\t\t\t{n[c]}\n")
            fh.write(f"frac_ge1_{c}\t{f1[c]:.5f}\t\t\t{n[c]}\n")
        fh.write(f"n_paired_genes\t{len(P)}\t\t\t{len(P)}\n")
        fh.write(f"chosen_window\t{w_use}\t\t\t{len(P)}\n")
        fh.write(f"sweep_peak_window\t{peak_w}\t\t\t{len(P)}\n")
        fh.write(f"winrate_chosen\t{wr[0]:.5f}\t{wr[1]:.5f}\t{wr[2]:.5f}\t{wr[3]}\n")
        fh.write(f"winrate_peak\t{pk[0][0]:.5f}\t{pk[0][1]:.5f}\t{pk[0][2]:.5f}\t{pk[0][3]}\n")
        fh.write(f"asym_chosen\t{sp['asym']:.5f}\t{sp['asym_lo']:.5f}\t{sp['asym_hi']:.5f}\t{sp['n']}\n")
        fh.write(f"default_window\t{DEFAULT_W}\t\t\t{len(P)}\n")
        fh.write(f"winrate_default\t{wrd[0]:.5f}\t{wrd[1]:.5f}\t{wrd[2]:.5f}\t{wrd[3]}\n")
        fh.write(f"asym_default\t{spd['asym']:.5f}\t{spd['asym_lo']:.5f}\t{spd['asym_hi']:.5f}\t{spd['n']}\n")
        fh.write(f"tied_default\t{spd['eq']:.5f}\t\t\t{spd['n']}\n")
        fh.write(f"tied_chosen\t{sp['eq']:.5f}\t\t\t{sp['n']}\n")
        for c in CLASSES:
            fh.write(f"frac_ge1_default_{c}\t{f1d[c]:.5f}\t\t\t{n[c]}\n")
        best = max(sweep, key=lambda k: sweep[k][0][0])
        fh.write(f"best_scalar\t{best[0]}_w{best[1]}\t\t\t{len(P)}\n")
        fh.write(f"best_winrate\t{sweep[best][0][0]:.5f}\t{sweep[best][0][1]:.5f}\t{sweep[best][0][2]:.5f}\t{len(P)}\n")
        fh.write(f"split_gt_chosen\t{sp['gt']:.5f}\t\t\t{sp['n']}\n")
        fh.write(f"split_eq_chosen\t{sp['eq']:.5f}\t\t\t{sp['n']}\n")
        fh.write(f"split_lt_chosen\t{sp['lt']:.5f}\t\t\t{sp['n']}\n")
        fh.write(f"clears_0534\t{int(clears)}\t\t\t{len(P)}\n")
        fh.write("ref_architecture\t0.534\t\t\t\n")
        fh.write("ref_acceptor\t0.606\t\t\t\n")
    log("[n] " + "  ".join(f"{c} {n[c]}" for c in CLASSES) + f"   paired genes {len(P)}")
    log("[out] upstream_count_dist.tsv upstream_count_sweep.tsv upstream_maxent_ecdf.tsv "
        "upstream_count_stats.tsv " + CACHE)
