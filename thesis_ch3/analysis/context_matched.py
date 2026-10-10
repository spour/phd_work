#!/usr/bin/env python3
# network discrimination between a real 3' end and competitors from the same gene whose
import os, collections, numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); PUB = os.path.join(ROOT, "..")
W = np.load(os.path.join(PUB, "cnn_wg_scores.npz"), allow_pickle=True); Z = np.load(os.path.join(ROOT, "data/stress_context.npz"), allow_pickle=True)
L, exo, rep, grp, y, gene = W["logit"].astype(float), Z["exo"], Z["rep"], Z["grp"], W["y"], W["gene"]
byg = collections.defaultdict(list)
for i, gg in enumerate(gene): byg[gg].append(i)
RNG = np.random.default_rng(1); rows = []
for label, gsel in (("all competitor groups", ("intronic", "terminal exon", "annotated site")), ("terminal exon", ("terminal exon",)), ("annotated site", ("annotated site",))):
    per = {}
    for gg, ii in byg.items():
        ii = np.array(ii); r = ii[y[ii] == 1]
        if len(r) != 1 or exo[r[0]] < 0.9 or rep[r[0]] > 0.1: continue
        cc = ii[np.isin(grp[ii], gsel) & (exo[ii] >= 0.9) & (rep[ii] <= 0.1)]
        if len(cc) == 0: continue
        per[gg] = [((S[r[0]] > S[cc]).mean() + 0.5 * (S[r[0]] == S[cc]).mean()) for S in (L, exo, 1 - rep)] + [len(cc)]
    if not per: continue
    A = np.array(list(per.values()))
    for j, name in enumerate(("sequence network", "annotated exon, upstream 500 nt", "repeat-free fraction")):
        v = A[:, j]; b = [v[RNG.integers(0, len(v), len(v))].mean() for _ in range(2000)]
        rows.append((label, name, len(v), int(A[:, 3].sum()), f"{v.mean():.3f}", f"[{np.percentile(b,2.5):.3f}, {np.percentile(b,97.5):.3f}]"))
        print(f"{label:22s} {name:34s} genes {len(v):4d} pairs {int(A[:,3].sum()):5d}  win-rate {v.mean():.3f} [{np.percentile(b,2.5):.3f}, {np.percentile(b,97.5):.3f}]")
with open(os.path.join(ROOT, "data/context_matched.tsv"), "w") as fh:
    fh.write("competitors\tscore\tgenes\tpairs\twin_rate\tci\n")
    for r in rows: fh.write("\t".join(str(x) for x in r) + "\n")
