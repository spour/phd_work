#!/usr/bin/env python3
# interpretable properties of each window, to ask what exonic content the network reads
import re, sys, os, numpy as np, pysam
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
GTF = os.path.join(PUB, "../train/data/annotation/gencode.v49.annotation.gtf"); FA = os.path.join(PUB, "../train/data/hg38.fa")
def log(*a): print(*a, file=sys.stderr, flush=True)
feat = {"CDS": defaultdict(list), "UTR": defaultdict(list)}
with open(GTF) as fh:
    for line in fh:
        if line.startswith("#"): continue
        f = line.split("\t", 8)
        if f[2] not in feat or 'transcript_type "protein_coding"' not in f[8]: continue
        feat[f[2]][(f[0], f[6])].append((int(f[3]) - 1, int(f[4])))
def merged(iv):
    iv = sorted(iv); out = []
    for a, b in iv:
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return np.array(out, np.int64)
M = {k: {kk: merged(v) for kk, v in d.items()} for k, d in feat.items()}; log("[gtf] CDS/UTR intervals built")
def cover(Mi, a, b):
    if Mi is None or len(Mi) == 0: return 0
    i = np.searchsorted(Mi[:, 1], a, side="right"); tot = 0
    while i < len(Mi) and Mi[i, 0] < b:
        tot += max(0, min(b, Mi[i, 1]) - max(a, Mi[i, 0])); i += 1
    return tot
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True)
ch, cen, st, X = D["chrom"], D["center"], D["strand"], D["X"]
fa = pysam.FastaFile(FA)
def p3(w):
    tot = 0.0
    for b in range(4):
        u = (w == b).astype(float); u -= u.mean()
        P = np.abs(np.fft.rfft(u)) ** 2
        if P[1:].mean() == 0: continue
        k = int(round(len(u) / 3)); tot += P[k] / P[1:].mean()
    return tot
def elems(s, pat): return len(re.findall(f"(?={pat})", s))
N = len(cen); up = np.zeros((N, 10), np.float32); BIN = np.zeros((N, 25, 4), np.float32)
NT = np.array(list("ACGTN"))
for i in range(N):
    w = X[i]; s = "".join(NT[np.minimum(w, 4)]); u = s[:2000]; wu = w[:2000]
    c = int(cen[i]); a, b = (c - 2000, c + 500) if st[i] == "+" else (c - 499, c + 2001)
    g = fa.fetch(ch[i], a, b)
    if st[i] == "-": g = g[::-1]
    low = np.frombuffer(g.encode(), np.uint8) >= 97
    gc = np.isin(wu, (1, 2)).mean(); cpg = u.count("CG") / max(1, (u.count("C") * u.count("G")) / len(u))
    ka, kb = (c - 2000, c) if st[i] == "+" else (c + 1, c + 2001)
    k = (ch[i], st[i])
    up[i] = [gc, (wu == 3).mean(), (wu == 0).mean(), cpg, low[:2000].mean(),
             cover(M["CDS"].get(k), ka, kb) / 2000, cover(M["UTR"].get(k), ka, kb) / 2000,
             p3(wu), elems(u, "ATTTA") / 2.0, elems(u, "TTTTT") / 2.0]
    for j in range(25):
        lo, hi = -2000 + 100 * j, -1900 + 100 * j
        ba, bb = (c + lo, c + hi) if st[i] == "+" else (c - hi + 1, c - lo + 1)
        seg = w[100 * j:100 * (j + 1)]
        BIN[i, j] = [np.isin(seg, (1, 2)).mean(), low[100 * j:100 * (j + 1)].mean(),
                     cover(M["CDS"].get(k), ba, bb) / 100, cover(M["UTR"].get(k), ba, bb) / 100]
    if i % 10000 == 0: log(f"  {i:,}")
cols = ["gc", "t_frac", "a_frac", "cpg_oe", "repeat_frac", "cds_frac", "utr_frac", "period3", "attta_per_kb", "polyT5_per_kb"]
with open(os.path.join(ROOT, "data/window_features.tsv"), "w") as fh:
    fh.write("idx\t" + "\t".join(cols) + "\n")
    for i in range(N): fh.write(f"{i}\t" + "\t".join(f"{v:.5g}" for v in up[i]) + "\n")
np.savez_compressed(os.path.join(ROOT, "data/window_feature_bins.npz"), bins=BIN, cols=np.array(["gc", "repeat", "cds", "utr"]))
y = D["y"]
for j, cname in enumerate(cols):
    log(f"  {cname:14s} real {up[y==1, j].mean():.4f}   competitor {up[y==0, j].mean():.4f}")
log("[out] data/window_features.tsv data/window_feature_bins.npz")
