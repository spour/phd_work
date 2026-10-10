#!/usr/bin/env python3
# 3' end sequencing signal around unannotated competitors and random intronic positions
import os, sys, glob, numpy as np, pyBigWig
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, ".."); G = os.path.join(PUB, "external/geo")
Z = np.load(os.path.join(ROOT, "data/usage_sites.npz"), allow_pickle=True); cls = Z["cls"]; ana = Z["analysis"]; ch = Z["chrom"].astype(str); cen = Z["center"]; st = Z["strand"].astype(str)
gene = Z["gene"]; cand = cls != "intronic control"; far = np.zeros(len(cls), bool); import collections
byg = collections.defaultdict(list)
for i in np.flatnonzero(cand): byg[gene[i]].append(cen[i])
byg = {k: np.array(v) for k, v in byg.items()}
for i in np.flatnonzero(~cand): far[i] = np.abs(byg[gene[i]] - cen[i]).min() >= 1000
anag = set(gene[(cls == "competitor") & ana])
real = (cls == "real 3' end") & np.isin(gene, list(anag))
sel = np.flatnonzero(real | ((cls == "competitor") & ana) | far)
grp = np.where(cls[sel] == "real 3' end", "real 3' ends", np.where(cls[sel] == "competitor", "unannotated competitors", "random intronic positions"))
print({g: int((grp == g).sum()) for g in set(grp)}, flush=True)
F = lambda pat: sorted(glob.glob(os.path.join(G, pat)))
EXP = [("HeLa (Feng 2024)", F("GSE262383/*HeLa-ctr-*"), F("GSE262383/*HeLa-U1-*")), ("HeLa (Feng 2025)", F("GSE250342/*_Control-[ab]_1*"), F("GSE250342/*_U1-[ab]_1*")),
       ("HEK293 (Feng 2025)", F("GSE250342/*_293-ctr-*"), F("GSE250342/*_293-U1-*")), ("SW480 (Feng 2025)", F("GSE250342/*SW480-NC-*"), F("GSE250342/*SW480-U1-*"))]
W, B = 500, 10
def work(f):
    bw = pyBigWig.open(f); sz = bw.chroms(); tot = bw.header()["sumData"]; out = {g: np.zeros(2 * W // B) for g in set(grp)}; n = {g: 0 for g in set(grp)}
    for k, i in enumerate(sel):
        c = ch[i][3:]; p = int(cen[i])
        if c not in sz or p - W < 0 or p + W + 1 > sz[c]: continue
        v = np.nan_to_num(np.asarray(bw.values(c, p - W, p + W), np.float32)) if st[i] == "+" else np.nan_to_num(np.asarray(bw.values(c, p - W + 1, p + W + 1), np.float32))[::-1]
        out[grp[k]] += v.reshape(-1, B).mean(1); n[grp[k]] += 1
    return f, {g: out[g] / n[g] / tot * 1e9 for g in out}
if __name__ == "__main__":
    files = sorted({f for e in EXP for f in e[1] + e[2]})
    with Pool(6) as pool: R = dict(pool.map(work, files))
    with open(os.path.join(ROOT, "data/usage_metaprofile.tsv"), "w") as fh:
        fh.write("experiment\tcondition\tclass\tbin_start\tsignal\n"); N_ = {g: int((grp == g).sum()) for g in set(grp)}; open(os.path.join(ROOT, "data/usage_metaprofile_n.tsv"), "w").write("\n".join(f"{k}\t{v}" for k, v in N_.items()) + "\n")
        for lab, fc, ft in EXP:
            for cond, fs in (("control", fc), ("U1 AMO", ft)):
                for g in ("real 3' ends", "unannotated competitors", "random intronic positions"):
                    m = np.mean([R[f][g] for f in fs], 0)
                    for b, v in zip(range(-W, W, B), m): fh.write(f"{lab}\t{cond}\t{g}\t{b}\t{v:.6g}\n")
    print("[out] data/usage_metaprofile.tsv")
