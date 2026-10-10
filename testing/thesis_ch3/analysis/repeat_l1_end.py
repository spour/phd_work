#!/usr/bin/env python3
# position of competitor sites within l1 and alu elements in nucleotides from the element's
import os, collections, numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); PUB = os.path.join(ROOT, "..")
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); y = D["y"]
ch, cen, st = D["chrom"], D["center"].astype(np.int64), D["strand"]
sites = collections.defaultdict(list); win = collections.defaultdict(list)
for i in np.flatnonzero(y == 0):
    c = int(cen[i]); s = str(st[i]); sites[str(ch[i])].append((c, s))
    win[str(ch[i])].append(((c - 2000, c) if s == "+" else (c + 1, c + 2001)) + (s,))
for k in sites: sites[k].sort()
for k in win: win[k].sort()
SP = {k: np.array([p for p, _ in v]) for k, v in sites.items()}; W0 = {k: np.array([w[0] for w in v]) for k, v in win.items()}
H = collections.defaultdict(list); B = collections.defaultdict(list); LEN = collections.defaultdict(list); NAME = collections.defaultdict(collections.Counter)
rng = np.random.default_rng(0)
with open(os.path.join(PUB, "rmsk.hg38.txt")) as fh:
    for line in fh:
        f = line.rstrip("\n").split("\t")
        F = "L1" if f[12] == "L1" else "Alu" if f[12] == "Alu" else None
        if F is None or f[5] not in SP: continue
        c_ = f[5]; s_, e_, strand = int(f[6]), int(f[7]), f[9]
        cs, ce = (int(f[13]), int(f[14])) if strand == "+" else (int(f[15]), int(f[14]))
        L = max(1, ce + (-int(f[15]) if strand == "+" else -int(f[13])))
        def from_end(p):
            frac = (p - s_) / max(1, e_ - s_)
            return L - (cs + frac * (ce - cs) if strand == "+" else ce - frac * (ce - cs))
        a, b = np.searchsorted(SP[c_], s_), np.searchsorted(SP[c_], e_)
        for k in range(a, b):
            p, gs = sites[c_][k]; o = "sense" if gs == strand else "antisense"; d = from_end(p)
            H[(F, o)].append(d); LEN[(F, o)].append(L); NAME[(F, o, int(d // 100))][f[10]] += 1
        V = win[c_]; j = np.searchsorted(W0[c_], s_ - 2001)
        while j < len(V) and V[j][0] < e_:
            wa, wb, gs = V[j]; lo, hi = max(wa, s_), min(wb, e_)
            if hi > lo: B[(F, "sense" if gs == strand else "antisense")].append(from_end(int(rng.integers(lo, hi))))
            j += 1
out = open(os.path.join(ROOT, "data/repeat_l1_end.tsv"), "w")
out.write("family\torientation\tbin_start_nt_from_3prime_end\tsites\tsite_frac\tbackground_frac\tratio\ttop_subfamilies\n")
for key, step, top in ((("L1", "antisense"), 100, 3000), (("L1", "sense"), 100, 3000), (("Alu", "antisense"), 20, 320), (("Alu", "sense"), 20, 320)):
    h = np.array(H[key]); b = np.array(B[key]); edges = np.arange(0, top + step, step)
    hs, _ = np.histogram(h, edges); bs, _ = np.histogram(b, edges)
    print(f"\n{key}  n {len(h)}  consensus length median {np.median(LEN[key]):.0f}  sites within {top} nt of 3' end {np.mean(h < top):.2f} (background {np.mean(b < top):.2f})")
    for e, x, y_ in zip(edges[:-1], hs, bs):
        fs, fb = x / len(h), y_ / len(b); nm = ", ".join(f"{a} {n}" for a, n in NAME[key + (int(e // 100),)].most_common(3)) if key[0] == "L1" else ""
        out.write(f"{key[0]}\t{key[1]}\t{e}\t{x}\t{fs:.4f}\t{fb:.4f}\t{fs/max(fb,1e-9):.2f}\t{nm}\n")
        print(f"  {e:5d}  sites {fs:6.3f}  bg {fb:6.3f}  ratio {fs/max(fb,1e-9):5.2f}  {nm}")
out.close()
