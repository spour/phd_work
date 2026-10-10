#!/usr/bin/env python3
# frequency of real 3' ends by the maxentscan scores of the upstream splice sites
import os, sys, collections, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); y = D["y"].astype(int); gene = D["gene"]
key = lambda a, b, c, d: (str(a), str(b), int(float(c)), str(d)); M = {}
with open(os.path.join(PUB, "fig3_ucube_meta.tsv")) as fh:
    h = fh.readline().rstrip("\n").split("\t"); ix = {k: j for j, k in enumerate(h)}
    for l in fh:
        f = l.rstrip("\n").split("\t"); g = lambda c: float(f[ix[c]]) if f[ix[c]] not in ("", "nan", "NA") else np.nan
        M[key(f[0], f[ix["chrom"]], f[ix["center"]], f[ix["strand"]])] = (g("acc"), g("interv_donor"))
K = [key(*t) for t in zip(gene, D["chrom"], D["center"], D["strand"])]
acc = np.array([M[k][0] for k in K]); don = np.array([M[k][1] for k in K]); ok = np.isfinite(acc) & np.isfinite(don)
genes = sorted(set(gene)); gix = {g_: k for k, g_ in enumerate(genes)}; kg = np.array([gix[g_] for g_ in gene]); NG = len(genes)
n = np.bincount(kg, minlength=NG); e = 1.0 / n[kg]; RNG = np.random.default_rng(0)
def oe(m):
    o_ = np.bincount(kg[m], weights=y[m].astype(float), minlength=NG); e_ = np.bincount(kg[m], weights=e[m], minlength=NG)
    b = []
    for _ in range(1000):
        k = RNG.integers(0, NG, NG); b.append(o_[k].sum() / max(e_[k].sum(), 1e-9))
    return o_.sum() / e_.sum(), *np.percentile(b, [2.5, 97.5]), int(m.sum()), int(y[m].sum())
AB = [(-99, 6)] + [(a, a + 1) for a in range(6, 13)] + [(13, 99)]; DB = [(-99, 4), (4, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 99)]
lab = lambda a, b: f"<{b}" if a == -99 else (f">={a}" if b == 99 else f"{a}-{b}"); mid = lambda a, b: b - 0.5 if a == -99 else (a + 0.5 if b == 99 else (a + b) / 2)
with open(os.path.join(ROOT, "data/hypothesis_curve.tsv"), "w") as fh:
    fh.write("score\tbin\tmid\tcandidates\treal\tobserved_over_expected\tlo\thi\n")
    for name, v, B in (("3' splice site", acc, AB), ("5' splice site", don, DB)):
        for a, b in B:
            r = oe(ok & (v >= a) & (v < b)); fh.write(f"{name}\t{lab(a,b)}\t{mid(a,b)}\t{r[3]}\t{r[4]}\t{r[0]:.3f}\t{r[1]:.3f}\t{r[2]:.3f}\n")
            print(f"{name:16s} {lab(a,b):6s} n {r[3]:6d}  O/E {r[0]:.2f} [{r[1]:.2f},{r[2]:.2f}]")
with open(os.path.join(ROOT, "data/hypothesis_grid.tsv"), "w") as fh:
    fh.write("acc_bin\tdon_bin\tacc_mid\tdon_mid\tcandidates\treal\tobserved_over_expected\tlo\thi\n")
    for a, b in AB:
        for c, d in DB:
            m = ok & (acc >= a) & (acc < b) & (don >= c) & (don < d)
            if m.sum() < 200: continue
            r = oe(m); fh.write(f"{lab(a,b)}\t{lab(c,d)}\t{mid(a,b)}\t{mid(c,d)}\t{r[3]}\t{r[4]}\t{r[0]:.3f}\t{r[1]:.3f}\t{r[2]:.3f}\n")
print("[out] data/hypothesis_curve.tsv data/hypothesis_grid.tsv")
