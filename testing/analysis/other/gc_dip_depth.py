#!/usr/bin/env python3
# statistics on one claim: the at-rich gc dip at the cleavage site is deeper at real 3'
import sys, os, time
import numpy as np
from math import erfc, sqrt, log10, pi
from multiprocessing import Pool
sys.path.insert(0, "scripts")
import gc_metagene_data as GM
import u1_5ss_landscape_data as L

log = L.log
UP, DN = GM.UP, GM.DN
CODE = GM.CODE
SEED, NBOOT = 42, 2000

BASELINES = (("[-2000,-1000)", -2000, -1000),
             ("[-1500,-750)",  -1500,  -750))
TROUGHS   = (("mean [-150,0)",   "mean", -150, 0, 0),
             ("mean [-100,0)",   "mean", -100, 0, 0),
             ("min50 [-250,0)",  "min",  -250, 0, 50))
MINSTEP = 10
NB, NT = len(BASELINES), len(TROUGHS)

SITES = {}
def _init(): L.G_["G"].fh = open(L.FASTA, "rb")

def _one(i):
    ch = SITES["chrom"][i]; c = int(SITES["center"][i]); st = SITES["strand"][i]
    a, b = (c - UP, c + DN) if st == "+" else (c - DN + 1, c + UP + 1)
    s = L.G_["G"].fetch(ch, a, b, st)
    out = np.full(NB + NT, np.nan)
    if not s or len(s) != UP + DN: return out
    t = CODE[np.frombuffer(s.encode("ascii"), np.uint8)]
    ok = t >= 0; gc = ((t == 1) | (t == 2)) & ok
    cok = np.r_[0.0, np.cumsum(ok.astype(np.float64))]
    cgc = np.r_[0.0, np.cumsum(gc.astype(np.float64))]
    def frac(lo, hi):
        n = cok[hi + UP] - cok[lo + UP]
        return (cgc[hi + UP] - cgc[lo + UP]) / n if n > 0 else np.nan
    for k, (_, lo, hi) in enumerate(BASELINES):
        out[k] = frac(lo, hi)
    for k, (_, mode, lo, hi, w) in enumerate(TROUGHS):
        if mode == "mean":
            out[NB + k] = frac(lo, hi)
        else:
            v = [frac(p, p + w) for p in range(lo, hi - w + 1, MINSTEP)]
            v = [x for x in v if np.isfinite(x)]
            out[NB + k] = min(v) if v else np.nan
    return out

# inline statistics (no scipy)
def _norm_p(z):
    z = abs(float(z))
    p = erfc(z / sqrt(2.0))
    lp = log10(2.0) - z*z/(2.0*np.log(10.0)) - log10(z*sqrt(2.0*pi)) if z > 5 else (
        log10(p) if p > 0 else float("-inf"))
    return p, lp

def wilcoxon(d):
    d = d[np.isfinite(d)]; d = d[d != 0]; n = len(d)
    if n < 6: return float("nan"), float("nan"), float("nan"), n
    a = np.abs(d); o = np.argsort(a, kind="stable"); a = a[o]; sg = np.sign(d[o])
    r = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and a[j+1] == a[i]: j += 1
        r[i:j+1] = (i + j + 2) / 2.0; i = j + 1
    W = float(r[sg > 0].sum()); mu = n*(n+1)/4.0
    _, cnt = np.unique(a, return_counts=True); tie = float((cnt**3 - cnt).sum())
    sd = sqrt(n*(n+1)*(2*n+1)/24.0 - tie/48.0)
    z = (W - mu - 0.5*np.sign(W - mu)) / sd
    p, lp = _norm_p(z)
    return z, p, lp, n

def _ranks(v):
    o = np.argsort(v, kind="stable"); s = v[o]; n = len(v)
    r = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j+1] == s[i]: j += 1
        r[i:j+1] = (i + j + 2) / 2.0; i = j + 1
    out = np.empty(n); out[o] = r
    _, cnt = np.unique(s, return_counts=True)
    return out, float((cnt**3 - cnt).sum())

def mannwhitney(x, y):
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    r, tie = _ranks(np.r_[x, y])
    R1 = float(r[:n1].sum())
    U = R1 - n1*(n1+1)/2.0
    auc = U / (n1*n2)
    mu = n1*n2/2.0
    N = n1 + n2
    sd = sqrt(n1*n2/12.0 * ((N + 1) - tie/(N*(N-1))))
    z = (U - mu - 0.5*np.sign(U - mu)) / sd
    p, lp = _norm_p(z)
    return dict(n1=n1, n2=n2, auroc=auc, rb=2*auc - 1, z=z, p=p, log10p=lp)

def cohen_d(x, y):
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    sp = sqrt(((n1-1)*x.var(ddof=1) + (n2-1)*y.var(ddof=1)) / (n1 + n2 - 2))
    return float((x.mean() - y.mean()) / sp)

def boot_ci(v, fn, rng, nboot=NBOOT):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    B = rng.integers(0, len(v), size=(nboot, len(v)))
    s = np.array([fn(v[b]) for b in B])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

def pfmt(p, lp):
    if not np.isfinite(p): return "nan"
    if p == 0.0:
        return f"< 1e-300 (log10p ~ {lp:.0f})" if np.isfinite(lp) else "< 1e-300"
    return f"{p:.3g}"

# main
if __name__ == "__main__":
    S = GM.build_sites(); SITES.update(S)
    N = len(S["cls"])

    G = L.G_["G"]; rc = str.maketrans("ACGTacgt", "TGCAtgca")
    bad = chk = badm = chkm = 0
    for i in np.random.default_rng(0).choice(N, 300, replace=False):
        c = int(S["center"][i]); st = S["strand"][i]
        b0 = G.fetch(S["chrom"][i], c, c + 1, "+")
        a, b = (c - UP, c + DN) if st == "+" else (c - DN + 1, c + UP + 1)
        seq = G.fetch(S["chrom"][i], a, b, st)
        if not b0 or not seq or len(seq) != UP + DN: continue
        chk += 1; chkm += (st == "-")
        if seq[UP].upper() != (b0 if st == "+" else b0.translate(rc)).upper():
            bad += 1; badm += (st == "-")
    assert bad == 0, f"orientation check failed on {bad}/{chk} ({badm} minus-strand)"
    log(f"[check] transcript offset 0 == base at center on {chk} sites "
        f"({chkm} minus-strand): OK, minus-strand +1 offset held")

    t0 = time.time()
    with Pool(min(32, os.cpu_count() or 8), initializer=_init) as p:
        M = np.array(p.map(_one, range(N), chunksize=64))
    log(f"scanned {N} sites in {time.time()-t0:.0f}s; "
        f"{int(np.isfinite(M).all(1).sum())} sites with all windows finite")

    cls = S["cls"]; gsel = S["gene"]

    # per-site depth table
    combos = [(bn, blo, bhi, tn, ti, bi)
              for bi, (bn, blo, bhi) in enumerate(BASELINES)
              for ti, (tn, _, _, _, _) in enumerate(TROUGHS)]
    PRIMARY = ("[-2000,-1000)", "mean [-150,0)")
    with open("gc_dip_depth.tsv", "w") as fh:
        cols = ["gene", "chrom", "center", "strand", "class"]
        cols += [f"baseline_{bn}" for bn, _, _ in BASELINES]
        cols += [f"trough_{tn}" for tn, _, _, _, _ in TROUGHS]
        cols += [f"depth_{bn}_{tn}" for bn, _, _, tn, _, _ in combos]
        fh.write("\t".join(cols) + "\n")
        for i in range(N):
            row = [gsel[i], S["chrom"][i], str(int(S["center"][i])), S["strand"][i], cls[i]]
            row += [f"{M[i,k]:.6f}" for k in range(NB + NT)]
            row += [f"{M[i,bi] - M[i,NB+ti]:.6f}" for _, _, _, _, ti, bi in combos]
            fh.write("\t".join(row) + "\n")
    log(f"[out] gc_dip_depth.tsv  ({N} sites x {len(combos)} depth definitions)")

    # within-gene pairing index
    by = {}
    for i in range(N):
        if cls[i] == "decoy": continue
        r, k = by.setdefault(gsel[i], ([], []))
        (r if cls[i] == "real" else k).append(i)
    genes = [g for g, (r, k) in by.items() if r and k]
    log(f"within-gene: {len(genes)} genes with a real end and >=1 competitor\n")

    rows = []
    for bn, _, _, tn, ti, bi in combos:
        D = M[:, bi] - M[:, NB + ti]
        prim = (bn, tn) == PRIMARY
        tag = f"{bn} - {tn}"

        wr, dl = [], []
        for g in genes:
            r, k = by[g]
            vr = np.nanmean(D[r]); kk = D[k]; kk = kk[np.isfinite(kk)]
            if not (np.isfinite(vr) and len(kk)): continue
            wr.append(((kk < vr).sum() + 0.5*(kk == vr).sum()) / len(kk))
            dl.append(vr - kk.mean())
        wr = np.array(wr); dl = np.array(dl)
        rng = np.random.default_rng(SEED)
        wlo, whi = boot_ci(wr, np.mean, rng)
        mlo, mhi = boot_ci(dl, np.median, rng)
        z, p, lp, nz = wilcoxon(dl)
        dz = float(dl.mean() / dl.std(ddof=1))

        cm = {}
        for c in ("real", "competitor", "decoy"):
            v = D[cls == c]; v = v[np.isfinite(v)]
            lo, hi = boot_ci(v, np.mean, rng, 1000)
            cm[c] = (float(v.mean()), lo, hi, len(v))
        Dr = D[cls == "real"]; Dc = D[cls == "competitor"]; Dd = D[cls == "decoy"]
        rd = mannwhitney(Dr, Dd); cd = mannwhitney(Dc, Dd); rc_ = mannwhitney(Dr, Dc)
        d_rd, d_cd, d_rc = cohen_d(Dr, Dd), cohen_d(Dc, Dd), cohen_d(Dr, Dc)

        rows.append(dict(
            baseline=bn, trough=tn, primary=int(prim), n_genes=len(dl),
            wg_winrate=wr.mean(), wg_ci_lo=wlo, wg_ci_hi=whi,
            wg_median_delta=float(np.median(dl)), wg_md_ci_lo=mlo, wg_md_ci_hi=mhi,
            wg_mean_delta=float(dl.mean()), wg_dz=dz,
            wg_wilcoxon_z=z, wg_wilcoxon_p=p, wg_wilcoxon_log10p=lp, wg_n_nonzero=nz,
            mean_depth_real=cm["real"][0], real_ci_lo=cm["real"][1], real_ci_hi=cm["real"][2],
            mean_depth_competitor=cm["competitor"][0], comp_ci_lo=cm["competitor"][1],
            comp_ci_hi=cm["competitor"][2],
            mean_depth_decoy=cm["decoy"][0], decoy_ci_lo=cm["decoy"][1],
            decoy_ci_hi=cm["decoy"][2],
            n_real=cm["real"][3], n_competitor=cm["competitor"][3], n_decoy=cm["decoy"][3],
            auroc_real_vs_decoy=rd["auroc"], rb_real_vs_decoy=rd["rb"],
            mw_z_real_vs_decoy=rd["z"], mw_log10p_real_vs_decoy=rd["log10p"],
            d_real_vs_decoy=d_rd,
            auroc_comp_vs_decoy=cd["auroc"], rb_comp_vs_decoy=cd["rb"],
            mw_z_comp_vs_decoy=cd["z"], mw_log10p_comp_vs_decoy=cd["log10p"],
            d_comp_vs_decoy=d_cd,
            auroc_real_vs_comp=rc_["auroc"], rb_real_vs_comp=rc_["rb"],
            mw_z_real_vs_comp=rc_["z"], mw_log10p_real_vs_comp=rc_["log10p"],
            d_real_vs_comp=d_rc))

        mark = "  <== PRIMARY" if prim else ""
        log(f"=== depth = {tag}{mark}")
        log(f"  [1 within-gene] n={len(dl)} genes  win-rate P(real depth > competitor depth) "
            f"{wr.mean():.4f} [{wlo:.4f},{whi:.4f}]")
        log(f"                  median paired delta {np.median(dl):+.5f} "
            f"[{mlo:+.5f},{mhi:+.5f}]  mean {dl.mean():+.5f}  dz {dz:+.3f}  "
            f"Wilcoxon z={z:+.2f} p={pfmt(p, lp)}")
        log(f"  [2 class mean depth] real {cm['real'][0]:.4f} "
            f"[{cm['real'][1]:.4f},{cm['real'][2]:.4f}]   "
            f"competitor {cm['competitor'][0]:.4f} "
            f"[{cm['competitor'][1]:.4f},{cm['competitor'][2]:.4f}]   "
            f"decoy {cm['decoy'][0]:.4f} [{cm['decoy'][1]:.4f},{cm['decoy'][2]:.4f}]")
        log(f"     real>decoy  AUROC {rd['auroc']:.4f} rank-biserial {rd['rb']:+.3f} "
            f"d {d_rd:+.3f}  MW z={rd['z']:+.1f} p={pfmt(rd['p'], rd['log10p'])}")
        log(f"     comp>decoy  AUROC {cd['auroc']:.4f} rank-biserial {cd['rb']:+.3f} "
            f"d {d_cd:+.3f}  MW z={cd['z']:+.1f} p={pfmt(cd['p'], cd['log10p'])}")
        log(f"  [3 effect size] pooled real vs competitor: Cohen's d {d_rc:+.3f}   "
            f"per-site AUROC {rc_['auroc']:.4f}   within-gene dz {dz:+.3f}")
        log("")

    hdr = list(rows[0].keys())
    with open("gc_dip_stats.tsv", "w") as fh:
        fh.write("\t".join(hdr) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[k]) if isinstance(r[k], (int, str))
                               else f"{r[k]:.6g}" for k in hdr) + "\n")

    log("=== raw window GC vs own-frame depth (real vs competitor)")
    aux = []
    for k, (tn, _, _, _, _) in enumerate(TROUGHS):
        V = M[:, NB + k]
        a = mannwhitney(Dc0 := V[cls == "competitor"], V[cls == "real"])["auroc"]
        w = []
        for g in genes:
            r, kk_ = by[g]
            vr = np.nanmean(V[r]); kk = V[kk_]; kk = kk[np.isfinite(kk)]
            if np.isfinite(vr) and len(kk):
                w.append(((kk > vr).sum() + 0.5*(kk == vr).sum()) / len(kk))
        w = float(np.mean(w))
        dep = [r for r in rows if r["trough"] == tn and r["baseline"] == BASELINES[0][0]][0]
        aux.append(dict(trough=tn, raw_auroc=a, raw_wg_winrate=w,
                        depth_auroc=dep["auroc_real_vs_comp"],
                        depth_wg_winrate=dep["wg_winrate"]))
        log(f"  {tn:18s} raw GC: AUROC {a:.4f}  within-gene win-rate {w:.4f}   ->   "
            f"depth: AUROC {dep['auroc_real_vs_comp']:.4f}  "
            f"within-gene win-rate {dep['wg_winrate']:.4f}")
    with open("gc_dip_rawvsdepth.tsv", "w") as fh:
        h = list(aux[0].keys()); fh.write("\t".join(h) + "\n")
        for r in aux:
            fh.write("\t".join(str(r[k]) if isinstance(r[k], str) else f"{r[k]:.6g}"
                                for k in h) + "\n")

    P = {}
    for ln in open("gc_metaprofile.tsv").read().splitlines()[1:]:
        o, c, g = ln.split("\t")[:3]; P.setdefault(c, []).append((int(o), float(g)))
    log("\n=== profile-level dip depth (class-mean curve, far-upstream level - deepest bin)")
    for c in ("real", "competitor", "decoy"):
        v = sorted(P[c]); base = np.mean([g for o, g in v if -2000 <= o < -1000])
        omin, gmin = min(v, key=lambda t: t[1])
        log(f"  {c:11s} baseline {base:.4f}  trough {gmin:.4f} at {omin:+d} nt  "
            f"depth {base-gmin:.4f}")

    W = np.array([r["wg_winrate"] for r in rows])
    Dd_ = np.array([r["d_real_vs_comp"] for r in rows])
    A = np.array([r["auroc_real_vs_comp"] for r in rows])
    Z = np.array([r["wg_dz"] for r in rows])
    log("=== robustness across 3 trough x 2 baseline definitions")
    log(f"  within-gene win-rate  {W.min():.4f} - {W.max():.4f}  (spread {W.max()-W.min():.4f})")
    log(f"  within-gene dz        {Z.min():+.3f} - {Z.max():+.3f}")
    log(f"  pooled Cohen's d      {Dd_.min():+.3f} - {Dd_.max():+.3f}")
    log(f"  per-site AUROC        {A.min():.4f} - {A.max():.4f}")
    log("[out] gc_dip_depth.tsv gc_dip_stats.tsv")
