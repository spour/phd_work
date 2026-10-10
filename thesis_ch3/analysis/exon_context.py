#!/usr/bin/env python3
# how much of each candidate's upstream window is annotated exon?
import re, sys, os, numpy as np
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
GTF = os.path.join(PUB, "../train/data/annotation/gencode.v49.annotation.gtf")
UP = 2000
TID = re.compile(r'transcript_id "([^"]+)"'); TT = re.compile(r'transcript_type "([^"]+)"')
tx = defaultdict(list); txinfo = {}
with open(GTF) as fh:
    for line in fh:
        if line.startswith("#"): continue
        f = line.split("\t", 8)
        if f[2] != "exon": continue
        if 'transcript_type "protein_coding"' not in f[8]: continue
        t = TID.search(f[8]).group(1)
        tx[t].append((int(f[3]) - 1, int(f[4])))
        txinfo[t] = (f[0], f[6])
print(f"[gtf] {len(tx):,} protein-coding transcripts", file=sys.stderr)
allx = defaultdict(list); lastx = defaultdict(list)
for t, ex in tx.items():
    ch, st = txinfo[t]; ex.sort()
    for e in ex: allx[(ch, st)].append(e)
    lastx[(ch, st)].append(ex[-1] if st == "+" else ex[0])
def merged(iv):
    iv = sorted(iv); out = []
    for a, b in iv:
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return np.array(out, dtype=np.int64)
MA = {k: merged(v) for k, v in allx.items()}; ML = {k: merged(v) for k, v in lastx.items()}
def cover(M, a, b):
    if M is None or len(M) == 0: return 0
    i = np.searchsorted(M[:, 1], a, side="right"); tot = 0
    while i < len(M) and M[i, 0] < b:
        tot += max(0, min(b, M[i, 1]) - max(a, M[i, 0])); i += 1
    return tot
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True)
ch = D["chrom"]; c = D["center"]; st = D["strand"]; y = D["y"]; g = D["gene"]
out = open(os.path.join(ROOT, "data/exon_context.tsv"), "w")
out.write("idx\tgene\ty\texon_frac_up2000\tlastexon_frac_up2000\texon_frac_up500\n")
for i in range(len(c)):
    k = (ch[i], st[i])
    a, b = (c[i] - UP, c[i]) if st[i] == "+" else (c[i] + 1, c[i] + 1 + UP)
    a5, b5 = (c[i] - 500, c[i]) if st[i] == "+" else (c[i] + 1, c[i] + 501)
    out.write(f"{i}\t{g[i]}\t{y[i]}\t{cover(MA.get(k), a, b)/UP:.4f}\t{cover(ML.get(k), a, b)/UP:.4f}\t{cover(MA.get(k), a5, b5)/500:.4f}\n")
out.close()
BIN = np.zeros((len(c), 25), np.float32)
for i in range(len(c)):
    k = (ch[i], st[i]); M = MA.get(k)
    for j in range(25):
        lo, hi = -2000 + 100 * j, -2000 + 100 * (j + 1)
        a, b = (c[i] + lo, c[i] + hi) if st[i] == "+" else (c[i] - hi + 1, c[i] - lo + 1)
        BIN[i, j] = cover(M, a, b) / 100.0
np.savez_compressed(os.path.join(ROOT, "data/exon_bins.npz"), exon_bin=BIN)
E = np.loadtxt(os.path.join(ROOT, "data/exon_context.tsv"), skiprows=1, usecols=(2, 3, 4, 5))
for lab, m in (("real", E[:, 0] == 1), ("competitor", E[:, 0] == 0)):
    print(f"[{lab}] n={m.sum():,}  exonic fraction of upstream 2 kb: median {np.median(E[m,1]):.3f} mean {E[m,1].mean():.3f}; "
          f"last-exon fraction mean {E[m,2].mean():.3f}; upstream 500 nt exonic mean {E[m,3].mean():.3f}", file=sys.stderr)
