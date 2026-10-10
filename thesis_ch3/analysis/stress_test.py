#!/usr/bin/env python3
# does the network separate a real 3' end from competitors that share its exonic context?
import os, re, sys, collections, numpy as np, pysam
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
sys.path.insert(0, "/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/arrow/17.0.0/lib/python3.11/site-packages")
import pyarrow.parquet as pq
def log(*a): print(*a, file=sys.stderr, flush=True)
W = np.load(os.path.join(PUB, "cnn_wg_scores.npz"), allow_pickle=True)
T = pq.read_table(os.path.join(PUB, "cpa_peaks_final.parquet")).to_pydict()
key = lambda g, c, x, s: (str(g), str(c), int(x), str(s))
row = {key(*t): i for i, t in enumerate(zip(T["gene"], T["chrom"], T["center"], T["strand"]))}
ix = np.array([row[key(*t)] for t in zip(W["gene"], W["chrom"], W["center"], W["strand"])])
apa = np.array(T["flag_apa_annotated"])[ix]; te = np.array(T["flag_reference_terminal_exon"])[ix]
dn = np.array(T["flag_downstream_of_canonical"])[ix]; can = np.array(T["is_canonical"])[ix]
assert (can == (W["y"] == 1)).all()
comp = ~can
log(f"competitors {comp.sum():,}: annotated-site {int((comp & apa).sum()):,} (funnel removed 18,119); "
    f"terminal-exon not annotated {int((comp & te & ~apa).sum()):,} (funnel removed 1,076)")
grp = np.full(len(comp), "", dtype=object)
grp[comp & ~apa & ~te & ~dn] = "intronic"; grp[comp & te & ~apa & ~dn] = "terminal exon"; grp[comp & apa & ~dn] = "annotated site"
for gname in ("intronic", "terminal exon", "annotated site"): log(f"  upstream {gname}: {(grp == gname).sum():,}")
# context scores for every peak
GTF = os.path.join(PUB, "../train/data/annotation/gencode.v49.annotation.gtf")
ex = collections.defaultdict(list)
with open(GTF) as fh:
    for line in fh:
        if line.startswith("#"): continue
        f = line.split("\t", 8)
        if f[2] == "exon" and 'transcript_type "protein_coding"' in f[8]: ex[(f[0], f[6])].append((int(f[3]) - 1, int(f[4])))
def merged(iv):
    iv = sorted(iv); out = []
    for a, b in iv:
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return np.array(out, np.int64)
MA = {k: merged(v) for k, v in ex.items()}
def cover(M, a, b):
    if M is None: return 0
    i = np.searchsorted(M[:, 1], a, side="right"); tot = 0
    while i < len(M) and M[i, 0] < b: tot += max(0, min(b, M[i, 1]) - max(a, M[i, 0])); i += 1
    return tot
fa = pysam.FastaFile(os.path.join(PUB, "../train/data/hg38.fa"))
N = len(comp); exo = np.zeros(N); rep = np.zeros(N)
for i in range(N):
    c, s, ch = int(W["center"][i]), str(W["strand"][i]), str(W["chrom"][i])
    a5, b5 = (c - 500, c) if s == "+" else (c + 1, c + 501); a2, b2 = (c - 2000, c) if s == "+" else (c + 1, c + 2001)
    exo[i] = cover(MA.get((ch, s)), a5, b5) / 500
    q = fa.fetch(ch, a2, b2); rep[i] = sum(1 for x in q if x.islower()) / max(1, len(q))
log("[context] exonic and repeat fractions computed")
# per-group ranking
gene = W["gene"]; L = W["logit"].astype(float)
byg = collections.defaultdict(list)
for i, gg in enumerate(gene): byg[gg].append(i)
RNG = np.random.default_rng(1)
def ci(v): return np.percentile([v[RNG.integers(0, len(v), len(v))].mean() for _ in range(1000)], [2.5, 97.5])
rows = []
for gname in ("intronic", "terminal exon", "annotated site"):
    for sname, S in (("sequence network", L), ("annotated exon, upstream 500 nt", exo), ("repeat-free fraction, upstream 2 kb", 1 - rep)):
        w, p, ch_ = [], [], []
        for ii in byg.values():
            ii = np.array(ii); r = ii[can[ii]]; c = ii[grp[ii] == gname]
            if len(r) != 1 or len(c) == 0: continue
            sr = S[r[0]]; sc = S[c]
            w.append(((sr > sc).sum() + 0.5 * (sr == sc).sum()) / len(sc)); p.append(float(sr > sc.max())); ch_.append(1 / (len(c) + 1))
        w, p = np.array(w), np.array(p); lw, hw = ci(w); lp, hp = ci(p)
        rows.append((gname, sname, len(w), f"{w.mean():.3f}", f"[{lw:.3f}, {hw:.3f}]", f"{p.mean():.3f}", f"[{lp:.3f}, {hp:.3f}]", f"{np.mean(ch_):.3f}"))
        log(f"  vs {gname:15s} {sname:36s} genes {len(w):5d}  win-rate {w.mean():.3f} [{lw:.3f},{hw:.3f}]  P@1 {p.mean():.3f} [{lp:.3f},{hp:.3f}]  chance {np.mean(ch_):.3f}")
with open(os.path.join(ROOT, "data/stress_test.tsv"), "w") as fh:
    fh.write("competitors\tscore\tgenes\twin_rate\twin_rate_ci\tp_at_1\tp_at_1_ci\tchance_p_at_1\n")
    for r in rows: fh.write("\t".join(str(x) for x in r) + "\n")
np.savez_compressed(os.path.join(ROOT, "data/stress_context.npz"), exo=exo, rep=rep, grp=grp.astype(str), ix=ix)
log("[out] data/stress_test.tsv")
