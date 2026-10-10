#!/usr/bin/env python3
# with the poly(a) signal held fixed, what else separates a real 3' end
import os, sys, math, collections, numpy as np, pysam
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
def log(*a): print(*a, file=sys.stderr, flush=True)
src = open(os.path.join(HERE, "stats_ch3.py")).read(); cut = src.index("# ---------------------------------------------------------------- data")
ns = {"__file__": os.path.join(HERE, "stats_ch3.py")}; exec(src[:cut], ns); wilcoxon = ns["wilcoxon"]
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); y = D["y"].astype(int); g = D["gene"]
ch, cen, st = D["chrom"], D["center"].astype(np.int64), D["strand"]; N = len(y)
C = np.load(os.path.join(PUB, "cnn_loco_scores.npz"), allow_pickle=True); assert (C["gene"] == g).all(); logit = C["logit"].astype(float)
key = lambda a, b, c, d: (str(a), str(b), int(c), str(d))
dkeys = [key(*t) for t in zip(g, ch, cen, st)]
W = np.load(os.path.join(PUB, "cnn_wg_scores.npz"), allow_pickle=True); S = np.load(os.path.join(ROOT, "data/stress_context.npz"), allow_pickle=True)
wrow = {key(*t): i for i, t in enumerate(zip(W["gene"], W["chrom"], W["center"], W["strand"]))}
exo = S["exo"][[wrow[k] for k in dkeys]].astype(float)
R = np.genfromtxt(os.path.join(ROOT, "data/repeat_sites.tsv"), delimiter="\t", names=True, dtype=None, encoding=None)
rep = np.column_stack([R[n] for n in R.dtype.names if n.startswith("cov_")]).astype(float).sum(1)
def table(path, cols):
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t"); ix = {h: j for j, h in enumerate(hdr)}; out = {}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            out[key(f[ix["gene"]], f[ix["chrom"]], float(f[ix["center"]]), f[ix["strand"]])] = [float(f[ix[c]]) if f[ix[c]] not in ("", "nan", "NA") else np.nan for c in cols]
    return out
U = table(os.path.join(PUB, "fig3_ucube_meta.tsv"), ["acc", "u1_dw_up"]); SA = table(os.path.join(PUB, "sa_peak_scalars_final.tsv"), ["sa_acc_max", "sa_don_sum"])
spl = np.array([U.get(k, [np.nan] * 2) + SA.get(k, [np.nan] * 2) for k in dkeys])
log(f"[join] splice features found for {np.mean(~np.isnan(spl[:, 0])):.3f} of candidates")
fa = pysam.FastaFile(os.path.join(PUB, "../train/data/hg38.fa")); tr = str.maketrans("ACGTN", "TGCAN")
UP, DN = 500, 40
feat = collections.defaultdict(lambda: np.full(N, np.nan)); pas_start = np.full(N, -999); pas_hex = np.array([""] * N, dtype=object)
for i in range(N):
    p = int(cen[i]); c_ = str(ch[i])
    s = fa.fetch(c_, p - UP, p + DN + 1).upper() if st[i] == "+" else fa.fetch(c_, p - DN, p + UP + 1).upper().translate(tr)[::-1]
    s = s.replace("T", "U"); z = UP
    for k in range(z - 35, z - 9):
        h = s[k:k + 6]
        if h in ("AAUAAA", "AUUAAA"):
            if pas_hex[i] != "AAUAAA": pas_hex[i] = h; pas_start[i] = k - z
    up = s[:z]; feat["GC, upstream 500 nt"][i] = (up.count("G") + up.count("C")) / max(1, len(up))
    feat["UGUA, -100 to -20"][i] = sum(s[k:k + 4] == "UGUA" for k in range(z - 100, z - 23))
    dn = s[z + 1:z + 41]; feat["U fraction, +1 to +40"][i] = dn.count("U") / 40
    feat["GU count, +1 to +40"][i] = sum(dn[k:k + 2] == "GU" for k in range(39))
    feat["U fraction, -60 to -36"][i] = s[z - 60:z - 35].count("U") / 25
    if i % 10000 == 0: log(f"  {i:,}")
feat["PAS start (nt before site)"] = np.where(pas_hex != "", -pas_start, np.nan).astype(float)
F = {"annotated exon, upstream 500 nt": exo, "repeat, upstream 2 kb": rep, "GC, upstream 500 nt": feat["GC, upstream 500 nt"],
     "MaxEnt strongest upstream 3' splice site": spl[:, 0], "U1 sites upstream": spl[:, 1],
     "SpliceAI acceptor max, upstream 2 kb": spl[:, 2], "SpliceAI donor sum, upstream 2 kb": spl[:, 3],
     "PAS start (nt before site)": feat["PAS start (nt before site)"], "UGUA, -100 to -20": feat["UGUA, -100 to -20"],
     "U fraction, -60 to -36": feat["U fraction, -60 to -36"], "U fraction, +1 to +40": feat["U fraction, +1 to +40"],
     "GU count, +1 to +40": feat["GU count, +1 to +40"], "network logit": logit}
GROUP = {"annotated exon, upstream 500 nt": "context", "repeat, upstream 2 kb": "context", "GC, upstream 500 nt": "context",
         "network logit": "network", "PAS start (nt before site)": "cleavage", "UGUA, -100 to -20": "cleavage",
         "U fraction, -60 to -36": "cleavage", "U fraction, +1 to +40": "cleavage", "GU count, +1 to +40": "cleavage"}
gidx = collections.defaultdict(list)
for i, gg in enumerate(g): gidx[gg].append(i)
gidx = [np.array(v) for v in gidx.values()]
OUT = []
for level in ("all candidates", "canonical PAS", "strict AAUAAA, same position"):
    pairs = []
    for ii in gidx:
        r = ii[y[ii] == 1]
        if len(r) != 1: continue
        r = r[0]; c = ii[y[ii] == 0]
        if level == "canonical PAS":
            if pas_hex[r] == "": continue
            c = c[pas_hex[c] != ""]
        elif level == "strict AAUAAA, same position":
            if pas_hex[r] != "AAUAAA": continue
            c = c[(pas_hex[c] == "AAUAAA") & (np.abs(pas_start[c] - pas_start[r]) <= 3)]
        if len(c): pairs.append((r, c))
    p1 = np.mean([float(logit[r] > logit[c].max()) for r, c in pairs]); chance = np.mean([1 / (len(c) + 1) for r, c in pairs])
    ncomp = sum(len(c) for _, c in pairs)
    OUT.append((level, "genes", len(pairs), "competitors", ncomp, "network P@1", f"{p1:.3f}", "chance", f"{chance:.3f}"))
    log(f"\n== {level}: {len(pairs)} genes, {ncomp} competitors; network P@1 {p1:.3f} (chance {chance:.3f})")
    for name, v in F.items():
        d = np.array([v[r] - np.nanmean(v[c]) for r, c in pairs if not np.isnan(v[r]) and not np.all(np.isnan(v[c]))])
        if len(d) < 20: continue
        z, p, n = wilcoxon(d); rr = abs(z) / math.sqrt(max(n, 1))
        OUT.append((level, GROUP.get(name, "splicing"), name, len(d), f"{np.median(d):+.4g}", f"{np.mean(d > 0):.3f}", f"{z:.2f}", f"{rr:.3f}", f"{p:.2g}"))
        log(f"  {GROUP.get(name,'splicing'):9s} {name:42s} n {len(d):5d}  median diff {np.median(d):+9.4g}  real higher in {np.mean(d>0):.2f}  r {rr:.2f}  P {p:.2g}")
with open(os.path.join(ROOT, "data/pas_controlled.tsv"), "w") as fh:
    fh.write("level\tgroup\tfeature\tgenes\tmedian_diff\tfrac_real_higher\tz\tr\tP\n")
    for r in OUT: fh.write("\t".join(str(x) for x in r) + "\n")
log("[out] data/pas_controlled.tsv")
