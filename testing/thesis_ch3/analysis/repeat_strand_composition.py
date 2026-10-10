#!/usr/bin/env python3
# what does each repeat contribute to the gene's strand in each orientation?
import os, collections, numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); PUB = os.path.join(ROOT, "..")
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); X = D["X"]; y = D["y"]
ch, cen, st = D["chrom"], D["center"].astype(np.int64), D["strand"]
comp = np.flatnonzero(y == 0)
fam = lambda cls, f: "L1" if f == "L1" else "LTR" if cls.startswith("LTR") else "Alu" if f == "Alu" else "other repeat"
win = collections.defaultdict(list)
for i in comp:
    c = int(cen[i]); a, b = (c - 2000, c) if st[i] == "+" else (c + 1, c + 2001); win[str(ch[i])].append((a, b, i))
for k in win: win[k].sort()
W0 = {k: np.array([w[0] for w in v]) for k, v in win.items()}
lab = {i: np.full(2000, "none", dtype=object) for i in comp}
with open(os.path.join(PUB, "rmsk.hg38.txt")) as fh:
    for line in fh:
        f = line.rstrip("\n").split("\t")
        if f[5] not in win: continue
        s_, e_, strand = int(f[6]), int(f[7]), f[9]; F = fam(f[11], f[12]); V = win[f[5]]
        j = np.searchsorted(W0[f[5]], s_ - 2001)
        while j < len(V) and V[j][0] < e_:
            a, b, i = V[j]; lo, hi = max(a, s_), min(b, e_)
            if hi > lo:
                o = "sense" if strand == str(st[i]) else "antisense"
                if st[i] == "+": wa, wb = lo - (int(cen[i]) - 2000), hi - (int(cen[i]) - 2000)
                else: wa, wb = (int(cen[i]) + 2000) - hi + 1, (int(cen[i]) + 2000) - lo + 1
                lab[i][max(0, wa):min(2000, wb)] = f"{F}, {o}"
            j += 1
NT = "ACGT"; acc = collections.defaultdict(lambda: np.zeros(4)); pas = collections.Counter(); tt = collections.Counter(); L = collections.Counter()
for i in comp:
    w = X[i][:2000]; s = "".join(NT[b] if b < 4 else "N" for b in w); l = lab[i]
    for k in range(2000):
        if w[k] < 4: acc[l[k]][w[k]] += 1
    for k in range(1994):
        g = l[k]
        if g != l[k + 5]: continue
        L[g] += 1; h = s[k:k + 6]
        if h in ("AATAAA", "ATTAAA"): pas[g] += 1
        if h.startswith("TTTTT"): tt[g] += 1
out = open(os.path.join(ROOT, "data/repeat_strand_composition.tsv"), "w")
out.write("sequence\tbases\tA\tC\tG\tT\tPAS_per_kb\tTTTTT_per_kb\n")
for g in sorted(acc, key=lambda x: (x != "none", x)):
    v = acc[g]; n = v.sum(); fr = v / n
    out.write(f"{g}\t{int(n)}\t" + "\t".join(f"{x:.3f}" for x in fr) + f"\t{1000*pas[g]/max(1,L[g]):.2f}\t{1000*tt[g]/max(1,L[g]):.2f}\n")
    print(f"{g:22s} bases {int(n):9,d}  A {fr[0]:.3f} C {fr[1]:.3f} G {fr[2]:.3f} T {fr[3]:.3f}   AATAAA/ATTAAA per kb {1000*pas[g]/max(1,L[g]):.2f}   TTTTT per kb {1000*tt[g]/max(1,L[g]):.2f}")
out.close()
