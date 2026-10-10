#!/usr/bin/env python3
# relative consensus position of competitor sites within l1, ltr and alu elements
import os, csv, collections, numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); PUB = os.path.join(ROOT, "..")
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); y = D["y"]; ch, cen, st = D["chrom"], D["center"].astype(np.int64), D["strand"]
fam = lambda cls, f: "L1" if f == "L1" else "LTR" if cls.startswith("LTR") else "Alu" if f == "Alu" else None
sites = collections.defaultdict(list)
for i in np.flatnonzero(y == 0): sites[str(ch[i])].append((int(cen[i]), str(st[i])))
S = {k: np.array(sorted(v)) for k, v in sites.items()}; SS = {k: [s for _, s in sorted(v)] for k, v in sites.items()}
rel = collections.defaultdict(list)
with open(os.path.join(PUB, "rmsk.hg38.txt")) as fh:
    for line in fh:
        f = line.rstrip("\n").split("\t"); F = fam(f[11], f[12])
        if F is None or f[5] not in S: continue
        s_, e_, strand = int(f[6]), int(f[7]), f[9]; P = S[f[5]][:, 0] if S[f[5]].ndim == 2 else None
        pos_arr = np.array([p for p, _ in sorted(sites[f[5]])]); a = np.searchsorted(pos_arr, s_); b = np.searchsorted(pos_arr, e_)
        if a == b: continue
        cs, ce = (int(f[13]), int(f[14])) if strand == "+" else (int(f[15]), int(f[14]))
        L = ce + (-int(f[15]) if strand == "+" else -int(f[13]))
        for k in range(a, b):
            p = pos_arr[k]; gs = SS[f[5]][k]; frac = (p - s_) / max(1, e_ - s_)
            pos = cs + frac * (ce - cs) if strand == "+" else ce - frac * (ce - cs)
            rel[(F, "sense" if strand == gs else "antisense")].append(pos / max(1, L))
B = {tuple(r[:2]): np.array([float(x) for x in r[3].split()]) for r in csv.reader(open(os.path.join(ROOT, "data/repeat_position_background.tsv")), delimiter="\t") if r[0] != "family"}
out = open(os.path.join(ROOT, "data/repeat_site_positions.tsv"), "w"); out.write("family\torientation\tn\tsite_deciles\tbackground_deciles\tchi2_df9\n")
for k in sorted(rel):
    h, _ = np.histogram(rel[k], bins=np.linspace(0, 1, 11)); n = h.sum(); e = B[k] * n
    chi = float(((h - e) ** 2 / e).sum())
    out.write(f"{k[0]}\t{k[1]}\t{n}\t" + " ".join(f"{x/n:.3f}" for x in h) + "\t" + " ".join(f"{x:.3f}" for x in B[k]) + f"\t{chi:.1f}\n")
    print(f"{k[0]:4s} {k[1]:9s} n {n:5d}  sites " + " ".join(f"{x/n:.2f}" for x in h) + "\n" + " " * 21 + "backg " + " ".join(f"{x:.2f}" for x in B[k]) + f"   chi2 {chi:.0f}")
out.close()
