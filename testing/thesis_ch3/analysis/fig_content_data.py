#!/usr/bin/env python3
# tables for the figure on what the network reads in the upstream sequence
import os, csv, collections, itertools, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); y = D["y"].astype(int)
RS = list(csv.DictReader(open(os.path.join(ROOT, "data/repeat_sites.tsv")), delimiter="\t"))
fams = [k[4:] for k in RS[0] if k.startswith("cov_")]
with open(os.path.join(ROOT, "data/figc_repeat_cov.tsv"), "w") as fh:
    fh.write("family\tclass\tcoverage\n")
    for f in fams:
        v = np.array([float(r["cov_" + f]) for r in RS])
        for lab, m in (("real 3′ end", y == 1), ("competitor", y == 0)): fh.write(f"{f.replace('_',' ')}\t{lab}\t{v[m].mean():.5f}\n")
grp = np.array([r["site_group"] for r in RS]); ori = np.array([int(r["site_orientation"]) for r in RS])
with open(os.path.join(ROOT, "data/figc_repeat_site.tsv"), "w") as fh:
    fh.write("family\tclass\torientation\tfraction\n")
    for f in [x.replace("_", " ") for x in fams]:
        for lab, m in (("real 3′ end", y == 1), ("competitor", y == 0)):
            for o, on in ((1, "sense"), (-1, "antisense")): fh.write(f"{f}\t{lab}\t{on}\t{np.mean((grp[m] == f) & (ori[m] == o)):.5f}\n")
with open(os.path.join(ROOT, "data/figc_surrogate.tsv"), "w") as fh:
    fh.write("model\tp1\tlo\thi\n")
    for r in csv.reader(open(os.path.join(ROOT, "data/exon_content.tsv")), delimiter="\t"):
        if r[0] != "2": continue
        name = r[1].replace("surrogate: ", "").replace("sequence network (LOCO), same test genes", "sequence network")
        p = r[3]
        if "[" in p: v = p.split("P@1 ")[1]; m = float(v.split(" ")[0]); lo, hi = [float(x) for x in v.split("[")[1].rstrip("]").split(", ")]
        else: m = float(p.split("P@1 ")[1]); lo = hi = m
        fh.write(f"{name}\t{m:.4f}\t{lo:.4f}\t{hi:.4f}\n")
    ch = D["split"] == "test"; gi = collections.defaultdict(list)
    for i in np.flatnonzero(ch): gi[D["gene"][i]].append(i)
    fh.write(f"chance\t{np.mean([1/len(v) for v in gi.values()]):.4f}\tNA\tNA\n")
with open(os.path.join(ROOT, "data/figc_patch_kind.tsv"), "w") as fh:
    fh.write("transplanted\treplaced\tgain\tlo\thi\n")
    for r in csv.reader(open(os.path.join(ROOT, "data/exon_content.tsv")), delimiter="\t"):
        if r[0] != "4": continue
        t = r[1].split("real bin ")[1].split(" into")[0]; rep = r[1].split("competitor bin ")[1]
        m = float(r[2].split(" ")[0]); lo, hi = [float(x) for x in r[2].split("[")[1].rstrip("]").split(", ")]
        fh.write(f"{t}\t{rep}\t{m:.4f}\t{lo:.4f}\t{hi:.4f}\n")
X = D["X"]; BN = np.load(os.path.join(ROOT, "data/window_feature_bins.npz"))["bins"]; EB = np.load(os.path.join(ROOT, "data/exon_bins.npz"))["exon_bin"]
KM = collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(PUB, "kmer_scan.tsv")), delimiter="\t"):
    if r["background"] == "real_competitor": KM[r["kmer"]].append(float(r["delta_logit"]))
kmers = ["".join(t) for t in itertools.product("ACGT", repeat=6)]; eff = np.array([np.mean(KM[k]) for k in kmers])
cnt = {n: np.ones(4096) for n in ("utr", "intron")}; POW = 4 ** np.arange(5, -1, -1)
for i in range(len(y)):
    w = X[i].astype(np.int64)
    for j in range(20):
        seg = w[100 * j:100 * (j + 1)]
        if (seg > 3).any(): continue
        codes = np.lib.stride_tricks.sliding_window_view(seg, 6) @ POW
        if BN[i, j, 3] >= 0.9 and BN[i, j, 1] <= 0.1: cnt["utr"] += np.bincount(codes, minlength=4096)
        elif EB[i, j] <= 0.1 and BN[i, j, 1] <= 0.1: cnt["intron"] += np.bincount(codes, minlength=4096)
e = np.log2((cnt["utr"] / cnt["utr"].sum()) / (cnt["intron"] / cnt["intron"].sum()))
with open(os.path.join(ROOT, "data/figc_kmer.tsv"), "w") as fh:
    fh.write("kmer\teffect\tutr_enrichment\tgc\n")
    for k, a, b in zip(kmers, eff, e): fh.write(f"{k}\t{a:.5f}\t{b:.5f}\t{(k.count('G')+k.count('C'))/6:.3f}\n")
print("[out] figc tables")
