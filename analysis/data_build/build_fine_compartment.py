#!/usr/bin/env python3
# fine-grained gencode compartment for each peak (annotation only, no genome/maxent scan)
import csv, re, sys
import numpy as np
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)

GTF = "../train/data/annotation/gencode.v49.annotation.gtf"
LAB = "cpa_peaks_labeled.tsv"
WIN = 500
CLASSES = ["first exon", "internal exon", "last exon-3'UTR",
           "first intron", "internal intron", "last intron", "intergenic-other"]
PRI = np.array([4, 5, 6, 1, 2, 3, 0], float) * 1e-3

# master
R = list(csv.DictReader(open(LAB), delimiter="\t"))
chrom = [r["chrom"] for r in R]; cen = [int(r["center"]) for r in R]; strd = [r["strand"] for r in R]
iscanon = np.array([r["is_canonical"] == "True" for r in R])
contested = np.array([r["contested"] == "True" for r in R])
gene = np.array([r["gene"] for r in R]); NP = len(R)

# gtf: per-transcript exons -> tagged positional intervals
TID = re.compile(r'transcript_id "([^"]+)"')
tx = defaultdict(lambda: {"chrom": None, "strand": None, "exon": []})
n = 0
with open(GTF) as fh:
    for line in fh:
        if line[0] == "#": continue
        f = line.split("\t")
        if f[2] != "exon": continue
        m = TID.search(f[8])
        if not m: continue
        t = tx[m.group(1)]; t["chrom"] = f[0]; t["strand"] = f[6]
        t["exon"].append((int(f[3]) - 1, int(f[4])))
        n += 1
        if n % 4_000_000 == 0: log(f"  gtf exon {n:,}")

cls_by = defaultdict(lambda: [[] for _ in range(6)])
tx_by = defaultdict(list)
for t in tx.values():
    ex = t["exon"]
    if not ex: continue
    st = t["strand"]; key = (t["chrom"], st)
    ex = sorted(ex)
    order = ex if st == "+" else ex[::-1]
    N = len(order)
    tx_by[key].append((ex[0][0], ex[-1][1]))
    for i, (a, b) in enumerate(order):
        if N == 1: c = 2
        elif i == 0: c = 0
        elif i == N - 1: c = 2
        else: c = 1
        cls_by[key][c].append((a, b))
    for i in range(N - 1):
        e1 = order[i]; e2 = order[i + 1]
        a, b = (e1[1], e2[0]) if st == "+" else (e2[1], e1[0])
        if b <= a: continue
        if N == 2: c = 5
        elif i == 0: c = 3
        elif i == N - 2: c = 5
        else: c = 4
        cls_by[key][c].append((a, b))

def merge(ivs):
    if not ivs: return (np.empty(0, np.int64), np.empty(0, np.int64))
    ivs = sorted(ivs); cs = []; ce = []
    for a, b in ivs:
        if cs and a <= ce[-1]: ce[-1] = max(ce[-1], b)
        else: cs.append(a); ce.append(b)
    return np.array(cs, np.int64), np.array(ce, np.int64)

CLS = {k: [merge(v[c]) for c in range(6)] for k, v in cls_by.items()}
TXS = {k: merge(v) for k, v in tx_by.items()}
empt = (np.empty(0, np.int64), np.empty(0, np.int64))
def ov(a, b, cse):
    cs, ce = cse
    if cs.size == 0: return 0.0
    lo = np.searchsorted(ce, a, "right"); hi = np.searchsorted(cs, b, "left")
    if hi <= lo: return 0.0
    s = np.maximum(cs[lo:hi], a); e = np.minimum(ce[lo:hi], b)
    return float(np.maximum(e - s, 0).sum())

fine = np.full(NP, 6, np.int8)
for i in range(NP):
    a, b = (cen[i] - WIN, cen[i]) if strd[i] == "+" else (cen[i], cen[i] + WIN)
    key = (chrom[i], strd[i]); cl = CLS.get(key)
    vals = np.zeros(7)
    if cl is not None:
        for c in range(6): vals[c] = ov(a, b, cl[c])
    txv = ov(a, b, TXS.get(key, empt))
    vals[6] = max(WIN - txv, 0.0)
    fine[i] = int(np.argmax(vals + PRI))
np.save("fine_compartment.npy", fine)
log("saved fine_compartment.npy")

# per-group fractions with gene-clustered bootstrap
_, gid = np.unique(gene, return_inverse=True)
RNG = np.random.default_rng(11)
def cboot(ind, ids, B=1000):
    v = np.asarray(ind, float)
    o = np.argsort(ids, kind="mergesort"); ids_s = ids[o]; v_s = v[o]
    cut = np.r_[0, np.flatnonzero(ids_s[1:] != ids_s[:-1]) + 1, len(ids_s)]
    ng = len(cut) - 1; sg = np.empty(ng); cg = np.empty(ng)
    for j in range(ng):
        seg = slice(cut[j], cut[j + 1]); sg[j] = v_s[seg].sum(); cg[j] = seg.stop - seg.start
    obs = sg.sum() / cg.sum()
    idx = RNG.integers(0, ng, size=(B, ng)); dr = sg[idx].sum(1) / cg[idx].sum(1)
    lo, hi = np.percentile(dr, [2.5, 97.5]); return obs, lo, hi, dr

grp = {"canonical": iscanon & contested, "competitor": ~iscanon}
with open("fig3env_panelB_fine.tsv", "w") as fh:
    fh.write("group\tclass\tfrac\tlo\thi\n")
    for G, m in grp.items():
        cc = fine[m]; ids = gid[m]
        for c in range(7):
            o, lo, hi, _ = cboot((cc == c).astype(float), ids)
            fh.write(f"{G}\t{CLASSES[c]}\t{o:.5f}\t{lo:.5f}\t{hi:.5f}\n")
def intron_draws(m):
    return cboot(np.isin(fine[m], [3, 4, 5]).astype(float), gid[m])
oc, _, _, drc = intron_draws(grp["competitor"]); oa, _, _, dra = intron_draws(grp["canonical"])
d = drc - dra
with open("fig3env_panelB_fine_cmp.tsv", "w") as fh:
    fh.write("comparison\tmetric\teffect\tlo\thi\n")
    fh.write(f"competitor - canonical\tfrac_any_intron_diff\t{oc-oa:.5f}\t{np.percentile(d,2.5):.5f}\t{np.percentile(d,97.5):.5f}\n")

# console: validate against coarse compartment
print("\nfine compartment fractions (%):")
for G, m in grp.items():
    cc = fine[m]; frs = [f"{CLASSES[c][:12]:>13} {100*(cc==c).mean():5.1f}" for c in range(7)]
    print(f"  {G:<11} n={int(m.sum())}")
    for s in frs: print("   ", s)
print(f"\ncompetitor - canonical any-intron diff: {oc-oa:+.4f} [{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]")
print(f"any-intron: competitor {oc:.4f}  canonical {oa:.4f}")
try:
    coarse = np.load("env_annot_all.npz")["comp"]
    for G, m in grp.items():
        term_fine = np.isin(fine[m], [0, 1, 2]).mean()
        log(f"  [check] {G}: fine any-exon {term_fine:.3f} vs coarse TERM+INTERNAL {(np.isin(coarse[m],[0,1])).mean():.3f}; "
            f"fine any-intron {np.isin(fine[m],[3,4,5]).mean():.3f} vs coarse INTRON {(coarse[m]==2).mean():.3f}")
except Exception as e:
    log("coarse check skipped:", e)
log("DONE_FINE_COMPARTMENT")
