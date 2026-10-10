#!/usr/bin/env python3
# summary tables for figure 3.10 (exonic context), with 95% gene-bootstrap intervals
import os, csv, collections, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
RNG = np.random.default_rng(1)
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); y = D["y"].astype(int); g = D["gene"]
C = np.load(os.path.join(PUB, "cnn_loco_scores.npz"), allow_pickle=True); assert (C["gene"] == g).all()
B = np.load(os.path.join(ROOT, "data/exon_bins.npz"))["exon_bin"]
E = np.loadtxt(os.path.join(ROOT, "data/exon_context.tsv"), skiprows=1, usecols=(3, 4, 5)); ex2, ex5 = E[:, 0], E[:, 2]
gi = collections.defaultdict(list)
for i, gg in enumerate(g): gi[gg].append(i)
gi = [np.array(v) for v in gi.values()]
def ci(v, B_=2000):
    v = np.asarray(v, float); b = [v[RNG.integers(0, len(v), len(v))].mean() for _ in range(B_)]; return np.percentile(b, [2.5, 97.5])
# a: exonic fraction per bin by class (per-site means
with open(os.path.join(ROOT, "data/fig3_10_bins.tsv"), "w") as fh:
    fh.write("bin_start\tclass\texonic\n")
    for b in range(25):
        for lab, m in (("real 3′ end", y == 1), ("competitor", y == 0)): fh.write(f"{-2000+100*b}\t{lab}\t{B[m, b].mean():.4f}\n")
# b: p@1 on the analysis set
RR = list(csv.DictReader(open(os.path.join(PUB, "rescue_ranks.tsv")), delimiter="\t"))
chance = np.array([1 / int(r["n_candidates"]) for r in RR])
def hits(v): return np.array([float(v[ii][y[ii] == 1][0] > v[ii][y[ii] == 0].max()) for ii in gi])
rows = [("chance", chance), ("splice model", np.array([int(r["rank_splice"]) == 1 for r in RR], float)),
        ("sequence network", hits(C["logit"])), ("annotated exon, upstream 500 nt", hits(ex5))]
with open(os.path.join(ROOT, "data/fig3_10_p1.tsv"), "w") as fh:
    fh.write("method\tp1\tlo\thi\n")
    for lab, v in rows: lo, hi = ci(v); fh.write(f"{lab}\t{v.mean():.4f}\t{lo:.4f}\t{hi:.4f}\n")
# c: network p@1 by the real end's exonic fraction of the upstream 2 kb
edges = [0, 0.25, 0.5, 0.75, 1.0001]; labs = ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1"]
fr = np.array([ex2[ii][y[ii] == 1][0] for ii in gi]); h = hits(C["logit"])
with open(os.path.join(ROOT, "data/fig3_10_byfrac.tsv"), "w") as fh:
    fh.write("bin\tn\tp1\tlo\thi\n")
    for a, b, l in zip(edges[:-1], edges[1:], labs):
        m = (fr >= a) & (fr < b); lo, hi = ci(h[m]); fh.write(f"{l}\t{m.sum()}\t{h[m].mean():.4f}\t{lo:.4f}\t{hi:.4f}\n")
# d: patching gain per bin, donor bin exonic (>=0.9) vs intronic (<=0.1), competitor bin
P = np.load(os.path.join(ROOT, "data/patching.npz"), allow_pickle=True); gain = P["gain"]; Er = B[P["real"]]; Ec = B[P["comp"]]; pg = P["gene"]
ug = np.unique(pg); pgi = {k: np.flatnonzero(pg == k) for k in ug}
with open(os.path.join(ROOT, "data/fig3_10_patch.tsv"), "w") as fh:
    fh.write("bin_start\tkind\tn\tgain\tlo\thi\n")
    for b in range(25):
        for kind, m in (("exonic", (Er[:, b] >= 0.9) & (Ec[:, b] <= 0.1)), ("intronic", (Er[:, b] <= 0.1) & (Ec[:, b] <= 0.1))):
            if m.sum() < 200: continue
            gl = [pgi[k] for k in ug]; v = []
            for _ in range(500):
                s = np.concatenate([gl[j] for j in RNG.integers(0, len(gl), len(gl))]); mm = m[s]
                if mm.any(): v.append(gain[s[mm], b].mean())
            lo, hi = np.percentile(v, [2.5, 97.5])
            fh.write(f"{-2000+100*b}\t{kind}\t{m.sum()}\t{gain[m, b].mean():.4f}\t{lo:.4f}\t{hi:.4f}\n")
print("[out] data/fig3_10_{bins,p1,byfrac,patch}.tsv")
with open(os.path.join(ROOT, "data/fig3_10_patch_diff.tsv"), "w") as fh:
    fh.write("bin_start\tn_exonic\tn_intronic\tdiff\tlo\thi\n")
    gl = [pgi[k] for k in ug]
    for b in range(25):
        ex = (Er[:, b] >= 0.9) & (Ec[:, b] <= 0.1); it = (Er[:, b] <= 0.1) & (Ec[:, b] <= 0.1)
        if ex.sum() < 200 or it.sum() < 200: continue
        d = gain[ex, b].mean() - gain[it, b].mean(); v = []
        for _ in range(1000):
            s = np.concatenate([gl[j] for j in RNG.integers(0, len(gl), len(gl))])
            e2, i2 = ex[s], it[s]
            if e2.any() and i2.any(): v.append(gain[s[e2], b].mean() - gain[s[i2], b].mean())
        lo, hi = np.percentile(v, [2.5, 97.5])
        fh.write(f"{-2000+100*b}\t{ex.sum()}\t{it.sum()}\t{d:.4f}\t{lo:.4f}\t{hi:.4f}\n")
print("[out] data/fig3_10_patch_diff.tsv")
