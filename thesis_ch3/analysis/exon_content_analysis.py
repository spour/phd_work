#!/usr/bin/env python3
# what property of exonic sequence does the network read?
import os, sys, math, csv, collections, itertools, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
def log(*a): print(*a, file=sys.stderr, flush=True)
src = open(os.path.join(HERE, "stats_ch3.py")).read(); cut = src.index("# ---------------------------------------------------------------- data")
ns = {"__file__": os.path.join(HERE, "stats_ch3.py")}; exec(src[:cut], ns); wilcoxon = ns["wilcoxon"]
rsrc = open(os.path.join(HERE, "recalibrated.py")).read()
irls_src = rsrc[rsrc.index("def irls("):rsrc.index("def within(")]; exec(irls_src)
RNG = np.random.default_rng(1); ROWS = []
def out(sec, what, a, b="", c="", d=""):
    ROWS.append((sec, what, a, b, c, d)); log(f"[{sec}] {what:62s} {a}  {b}  {c}  {d}")
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); y = D["y"].astype(int); g = D["gene"]; split = D["split"]; X = D["X"]
C = np.load(os.path.join(PUB, "cnn_loco_scores.npz"), allow_pickle=True); assert (C["gene"] == g).all(); L = C["logit"].astype(float)
FT = np.genfromtxt(os.path.join(ROOT, "data/window_features.tsv"), delimiter="\t", names=True)
COLS = [c for c in FT.dtype.names if c != "idx"]; F = np.column_stack([FT[c] for c in COLS]).astype(float)
BN = np.load(os.path.join(ROOT, "data/window_feature_bins.npz"))["bins"]
EB = np.load(os.path.join(ROOT, "data/exon_bins.npz"))["exon_bin"]
gidx = collections.defaultdict(list)
for i, gg in enumerate(g): gidx[gg].append(i)
gidx = [np.array(v) for v in gidx.values()]
for j, c in enumerate(COLS):
    d = np.array([F[ii, j][y[ii] == 1][0] - F[ii, j][y[ii] == 0].mean() for ii in gidx]); z, p, n = wilcoxon(d)
    out("1", f"{c}, upstream 2 kb: real minus mean competitor", f"mean {d.mean():+.4g}", f"real higher in {np.mean(d > 0):.3f}", f"z = {z:.1f}, r = {abs(z)/math.sqrt(n):.3f}", f"P = {p:.2g}")
def centre(M):
    M = M.copy()
    for ii in gidx: M[ii] -= M[ii].mean(0)
    return M
def within(sc, idx):
    sub = collections.defaultdict(list)
    for i in idx: sub[g[i]].append(i)
    w, p1 = [], []
    for ii in sub.values():
        ii = np.array(ii); r = sc[ii][y[ii] == 1]; cc = sc[ii][y[ii] == 0]
        if len(r) == 0 or len(cc) == 0: continue
        w.append(((r[:, None] > cc[None, :]).sum() + 0.5 * (r[:, None] == cc[None, :]).sum()) / (len(r) * len(cc))); p1.append(float(r.max() > cc.max()))
    return np.array(w), np.array(p1)
tr = np.flatnonzero(split == "train"); va = np.flatnonzero(split == "val"); te = np.flatnonzero(split == "test")
ci = lambda v: np.percentile([v[RNG.integers(0, len(v), len(v))].mean() for _ in range(1000)], [2.5, 97.5])
col = lambda *n: [COLS.index(k) for k in n]
GROUPS = {"composition (GC, T, A, CpG o/e, ATTTA, TTTTT)": col("gc", "t_frac", "a_frac", "cpg_oe", "attta_per_kb", "polyT5_per_kb"),
          "repeat fraction": col("repeat_frac"), "period-3 coding signal": col("period3"),
          "all sequence-derived": col("gc", "t_frac", "a_frac", "cpg_oe", "attta_per_kb", "polyT5_per_kb", "repeat_frac", "period3"),
          "annotation: CDS and UTR fraction (reference)": col("cds_frac", "utr_frac")}
wn, pn = within(L, te); out("2", "sequence network (LOCO), same test genes", f"win-rate {wn.mean():.3f}", f"P@1 {pn.mean():.3f}")
for name, js in GROUPS.items():
    M = centre(F[:, js]); mu = M[tr].mean(0); sd = M[tr].std(0); sd[sd == 0] = 1; Mz = (M - mu) / sd
    best = None
    for lam in (0.01, 1, 100):
        b = irls(Mz[tr], y[tr], lam); wv, _ = within(np.column_stack([np.ones(len(Mz)), Mz]) @ b, va)
        if best is None or wv.mean() > best[0]: best = (wv.mean(), lam)
    b = irls(Mz[np.r_[tr, va]], y[np.r_[tr, va]], best[1]); wt, pt = within(np.column_stack([np.ones(len(Mz)), Mz]) @ b, te)
    lw, hw = ci(wt); lp, hp = ci(pt)
    out("2", f"surrogate: {name}", f"win-rate {wt.mean():.3f} [{lw:.3f}, {hw:.3f}]", f"P@1 {pt.mean():.3f} [{lp:.3f}, {hp:.3f}]", f"coef {np.round(b[1:], 3).tolist()}")
Lc = centre(L[:, None])[:, 0]
for name, js in GROUPS.items():
    M = centre(F[:, js]); Mb = np.column_stack([np.ones(len(M)), M]); beta, *_ = np.linalg.lstsq(Mb, Lc, rcond=None)
    r2 = 1 - ((Lc - Mb @ beta) ** 2).sum() / ((Lc - Lc.mean()) ** 2).sum()
    out("3", f"gene-centred network logit explained by {name}", f"R^2 {r2:.3f}")
P = np.load(os.path.join(ROOT, "data/patching.npz"), allow_pickle=True); comp, real, gain, pg = P["comp"], P["real"], P["gain"], P["gene"]
ug = np.unique(pg); pgi = [np.flatnonzero(pg == k) for k in ug]
def stratum_mean(mask2d):
    v = np.array([gain[s][mask2d[s]].sum() for s in pgi]); n = np.array([mask2d[s].sum() for s in pgi])
    m = v.sum() / n.sum(); bs = []
    for _ in range(1000):
        k = RNG.integers(0, len(pgi), len(pgi)); bs.append(v[k].sum() / max(1, n[k].sum()))
    return m, np.percentile(bs, [2.5, 97.5]), int(n.sum())
UPB = np.zeros(25, bool); UPB[:18] = True
Rr, Rc = BN[real], BN[comp]; Er, Ec = EB[real], EB[comp]
cintr_norep = (Ec <= 0.1) & (Rc[:, :, 1] <= 0.1) & UPB
cintr_rep = (Ec <= 0.1) & (Rc[:, :, 1] >= 0.9) & UPB
KIND = {"3' UTR": (Rr[:, :, 3] >= 0.9) & (Rr[:, :, 1] <= 0.1), "CDS": (Rr[:, :, 2] >= 0.9) & (Rr[:, :, 1] <= 0.1),
        "repeat-free intron": (Er <= 0.1) & (Rr[:, :, 1] <= 0.1), "intronic repeat": (Er <= 0.1) & (Rr[:, :, 1] >= 0.9)}
for cname, cm in (("repeat-free intronic", cintr_norep), ("repeat", cintr_rep)):
    for kname, km in KIND.items():
        m, (lo, hi), n = stratum_mean(cm & km)
        out("4", f"patching gain: real bin {kname} into competitor bin {cname}", f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]", f"cells {n}")
KM = collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(PUB, "kmer_scan.tsv")), delimiter="\t"):
    if r["background"] == "real_competitor": KM[r["kmer"]].append(float(r["delta_logit"]))
kmers = ["".join(t) for t in itertools.product("ACGT", repeat=6)]; idx6 = {k: i for i, k in enumerate(kmers)}
eff = np.array([np.mean(KM[k]) for k in kmers])
cnt = {n: np.ones(4096) for n in ("utr", "intron", "repeat")}
POW = 4 ** np.arange(5, -1, -1)
for i in range(len(y)):
    w = X[i].astype(np.int64)
    for j in range(20):
        seg = w[100 * j:100 * (j + 1)]
        if (seg > 3).any(): continue
        codes = np.lib.stride_tricks.sliding_window_view(seg, 6) @ POW
        if BN[i, j, 3] >= 0.9 and BN[i, j, 1] <= 0.1: cnt["utr"] += np.bincount(codes, minlength=4096)
        elif EB[i, j] <= 0.1 and BN[i, j, 1] <= 0.1: cnt["intron"] += np.bincount(codes, minlength=4096)
        elif EB[i, j] <= 0.1 and BN[i, j, 1] >= 0.9: cnt["repeat"] += np.bincount(codes, minlength=4096)
fr = {k: v / v.sum() for k, v in cnt.items()}
e_utr = np.log2(fr["utr"] / fr["intron"]); e_rep = np.log2(fr["repeat"] / fr["intron"])
gc6 = np.array([(k.count("G") + k.count("C")) / 6 for k in kmers]); t6 = np.array([k.count("T") / 6 for k in kmers])
def rk(v): o = np.argsort(v, kind="mergesort"); r = np.empty(len(v)); r[o] = np.arange(len(v)); return r
def spear(a, b): return np.corrcoef(rk(a), rk(b))[0, 1]
def partial(a, b, zs):
    Zm = np.column_stack([np.ones(len(a))] + [rk(z) for z in zs]); ra = rk(a) - Zm @ np.linalg.lstsq(Zm, rk(a), rcond=None)[0]
    rb = rk(b) - Zm @ np.linalg.lstsq(Zm, rk(b), rcond=None)[0]; return np.corrcoef(ra, rb)[0, 1]
out("5", "6-mer network effect vs log2 enrichment 3' UTR / repeat-free intron", f"Spearman {spear(eff, e_utr):+.3f}", f"partial (GC, T) {partial(eff, e_utr, [gc6, t6]):+.3f}", f"6-mers {len(eff)}")
out("5", "6-mer network effect vs log2 enrichment repeat / repeat-free intron", f"Spearman {spear(eff, e_rep):+.3f}", f"partial (GC, T) {partial(eff, e_rep, [gc6, t6]):+.3f}")
out("5", "6-mer network effect vs GC and T content", f"Spearman GC {spear(eff, gc6):+.3f}", f"T {spear(eff, t6):+.3f}")
o = np.argsort(eff); out("5", "most repressive 6-mers", ", ".join(kmers[i] for i in o[:8])); out("5", "most activating 6-mers", ", ".join(kmers[i] for i in o[-8:][::-1]))
o = np.argsort(e_rep); out("5", "6-mers most enriched in repeats over repeat-free intron", ", ".join(kmers[i] for i in o[-8:][::-1]))
with open(os.path.join(ROOT, "data/exon_content.tsv"), "w") as fh:
    fh.write("section\tcomparison\tresult\tresult2\tresult3\tresult4\n")
    for r in ROWS: fh.write("\t".join(str(x) for x in r) + "\n")
log("[out] data/exon_content.tsv")
