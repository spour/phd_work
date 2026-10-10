#!/usr/bin/env python3
# nearest cisbp-rna motif of each first-layer filter
import os, csv, collections, numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); PUB = os.path.join(ROOT, "..")
F = collections.defaultdict(lambda: np.zeros((19, 4)))
for r in csv.DictReader(open(os.path.join(PUB, "cnnmain_logos.tsv")), delimiter="\t"): F[r["filter"]][int(r["position"]) - 1, "ACGT".index(r["base"])] = float(r["prob"])
meta = {}
for r in csv.DictReader(open(os.path.join(PUB, "cnnmain_posmap.tsv")), delimiter="\t"):
    m = meta.setdefault(r["filter"], {"contrib": [], "centre": [], "abs_total": float(r["abs_total"]), "ic": float(r["ic_per_col"]), "n": int(r["n_sites"])})
    m["contrib"].append(float(r["contrib"])); m["centre"].append(float(r["centre"]))
MOT = []
with open(os.path.join(PUB, "../cisbp_rna.meme")) as fh:
    name = None; rows = []
    for line in fh:
        if line.startswith("MOTIF"):
            if name and rows: MOT.append((name, np.array(rows)))
            name = line.split()[1]; rows = []
        elif name and line.strip() and line.strip()[0].isdigit():
            v = [float(x) for x in line.split()]
            if len(v) == 4: rows.append(v)
    if name and rows: MOT.append((name, np.array(rows)))
def trim(P):
    P = P / P.sum(1, keepdims=True); ic = 2 + (P * np.log2(np.clip(P, 1e-9, 1))).sum(1); k = np.flatnonzero(ic >= 0.25)
    return P[k[0]:k[-1] + 1] if len(k) else P[:0]
MOT = [(n, trim(P)) for n, P in MOT]; MOT = [(n, P) for n, P in MOT if 5 <= len(P) <= 19]
def best(Fm, P):
    b = -2; at = 0
    for o in range(0, 19 - len(P) + 1):
        a = (Fm[o:o + len(P)] - 0.25).ravel(); c = (P - 0.25).ravel(); r = np.corrcoef(a, c)[0, 1]
        if r > b: b, at = r, o
    return b, at
rng = np.random.default_rng(0)
with open(os.path.join(ROOT, "data/filter_annotation.tsv"), "w") as fh:
    fh.write("filter\tfilter_consensus\tinformation_per_column\tsites\ttotal_contribution\tsigned_contribution\tpeak_band_centre\tbest_protein\tmotif\tmotif_consensus\tr\tP_permutation\tsecond_protein\tsecond_r\n")
    for f in sorted(F, key=lambda k: -meta[k]["abs_total"]):
        Fm = F[f] / F[f].sum(1, keepdims=True); sc = {}
        for n, P in MOT:
            r, at = best(Fm, P); prot = n.split("_")[-1]
            if prot not in sc or r > sc[prot][0]: sc[prot] = (r, n, P, at)
        top = sorted(sc.items(), key=lambda kv: -kv[1][0])[:2]; (p1, (r1, n1, P1, at1)), (p2, (r2, _, _, _)) = top
        null = [best(Fm, P1[rng.permutation(len(P1))][:, rng.permutation(4)])[0] for _ in range(1000)]; pv = (np.sum(np.array(null) >= r1) + 1) / 1001
        c = np.array(meta[f]["contrib"]); k = int(np.argmax(np.abs(c)))
        fh.write(f"{f}\t{''.join('ACGU'[j] for j in Fm.argmax(1))}\t{meta[f]['ic']:.2f}\t{meta[f]['n']}\t{meta[f]['abs_total']:.3f}\t{c.sum():+.3f}\t{meta[f]['centre'][k]:.0f}\t{p1}\t{n1}\t{''.join('ACGU'[j] for j in P1.argmax(1))}\t{r1:.2f}\t{pv:.3g}\t{p2}\t{r2:.2f}\n")
        print(f"{f} {''.join('ACGU'[j] for j in Fm.argmax(1))} total {meta[f]['abs_total']:.2f} signed {c.sum():+.2f} peak {meta[f]['centre'][k]:.0f}  {p1:10s} {''.join('ACGU'[j] for j in P1.argmax(1)):14s} r {r1:.2f} P {pv:.3g}   ({p2} {r2:.2f})")
