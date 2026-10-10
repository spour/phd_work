#!/usr/bin/env python3
# is the position of competitor sites within repeat elements different from the position
import os, collections, numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); PUB = os.path.join(ROOT, "..")
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); y = D["y"]; ch, cen, st = D["chrom"], D["center"].astype(np.int64), D["strand"]
comp = np.flatnonzero(y == 0); rng = np.random.default_rng(0)
win = collections.defaultdict(list)
for i in comp:
    c = int(cen[i]); a, b = (c - 2000, c) if st[i] == "+" else (c + 1, c + 2001); win[str(ch[i])].append((a, b, str(st[i]), c))
for k in win: win[k].sort()
W0 = {k: np.array([w[0] for w in v]) for k, v in win.items()}
fam = lambda cls, f: "L1" if f == "L1" else "LTR" if cls.startswith("LTR") else "Alu" if f == "Alu" else None
rel = collections.defaultdict(list); site_rel = collections.defaultdict(list)
with open(os.path.join(PUB, "rmsk.hg38.txt")) as fh:
    for line in fh:
        f = line.rstrip("\n").split("\t"); F = fam(f[11], f[12])
        if F is None or f[5] not in win: continue
        s_, e_, strand = int(f[6]), int(f[7]), f[9]
        cs, ce = (int(f[13]), int(f[14])) if strand == "+" else (int(f[15]), int(f[14]))
        L = ce + (-int(f[15]) if strand == "+" else -int(f[13]))
        V = win[f[5]]; j = np.searchsorted(W0[f[5]], s_ - 2001); 
        while j < len(V) and V[j][0] < e_:
            a, b, gs, c = V[j]; lo, hi = max(a, s_), min(b, e_)
            if hi > lo:
                p = int(rng.integers(lo, hi)); frac = (p - s_) / max(1, e_ - s_)
                pos = cs + frac * (ce - cs) if strand == "+" else ce - frac * (ce - cs)
                o = "sense" if strand == gs else "antisense"; rel[(F, o)].append(pos / max(1, L))
            j += 1
out = open(os.path.join(ROOT, "data/repeat_position_background.tsv"), "w")
out.write("family\torientation\tn\tdeciles_5prime_to_3prime\n")
for (F, o), v in sorted(rel.items()):
    h, _ = np.histogram(v, bins=np.linspace(0, 1, 11)); fr = h / h.sum()
    out.write(f"{F}\t{o}\t{len(v)}\t" + " ".join(f"{x:.3f}" for x in fr) + "\n")
    print(f"background {F:4s} {o:9s} n {len(v):6d}  deciles " + " ".join(f"{x:.2f}" for x in fr))
out.close()
