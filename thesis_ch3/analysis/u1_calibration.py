#!/usr/bin/env python3
# where the u1 thresholds sit on the calibration curve used for maxentscan
import os, re, collections, numpy as np, pysam
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
U = np.load(os.path.join(PUB, "../train/splice_train/data/u1_lookup.npy"), allow_pickle=True).item()
fa = pysam.FastaFile(os.path.join(PUB, "../train/data/hg38.fa")); comp = str.maketrans("ACGTN", "TGCAN")
TID = re.compile(r'transcript_id "([^"]+)"'); tx = collections.defaultdict(list); info = {}
with open(os.path.join(PUB, "../train/data/annotation/gencode.v49.annotation.gtf")) as fh:
    for line in fh:
        if line.startswith("#"): continue
        f = line.split("\t", 8)
        if f[2] != "exon" or 'transcript_type "protein_coding"' not in f[8]: continue
        t = TID.search(f[8]).group(1); tx[t].append((int(f[3]) - 1, int(f[4]))); info[t] = (f[0], f[6])
don = set()
for t, ex in tx.items():
    if len(ex) < 2: continue
    c, s = info[t]; ex.sort()
    for e in (ex[:-1] if s == "+" else ex[1:]): don.add((c, s, e[1] if s == "+" else e[0]))
don = sorted(don); pick = np.random.default_rng(0).choice(len(don), 20000, replace=False)
sc = []; gt = []
for k in pick:
    c, s, p = don[k]
    q = fa.fetch(c, p - 3, p + 6).upper() if s == "+" else fa.fetch(c, p - 6, p + 3).upper().translate(comp)[::-1]
    if q in U: sc.append(-U[q]); gt.append(q[3:5] == "GT")
sc = np.array(sc); print(f"annotated donors with GT at intron +1,+2: {np.mean(gt):.3f}")
S = {r.split("\t")[0]: r.split("\t")[1] for r in open(os.path.join(PUB, "splice_score_stats.tsv")).read().splitlines()[1:]}
with open(os.path.join(ROOT, "data/u1_calibration.tsv"), "w") as fh:
    fh.write("threshold\tsensitivity_annotated_donors\tbackground_frac_positions\tbackground_per_kb\n")
    for t, key in ((4, "exceed_u1_4"), (7, "exceed_u1_7")):
        bg = float(S[key]); fh.write(f"{t}\t{np.mean(sc >= t):.4f}\t{bg:.5f}\t{bg*1000:.1f}\n")
        print(f"U1 -dG >= {t}: sensitivity at {len(sc):,} annotated donors {np.mean(sc >= t):.3f}; background {bg:.4f} of positions, {bg*1000:.1f} per kb")
