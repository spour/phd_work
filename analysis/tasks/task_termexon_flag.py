#!/usr/bin/env python3
# gencode terminal-exon / 3'utr flag and 500 nt window gc for every contested peak
import csv, re, sys
import numpy as np
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)

GTF = "../train/data/annotation/gencode.v49.annotation.gtf"
FASTA = "../train/data/hg38.fa"
WIN = 500

class Faidx:
    _T = str.maketrans("ACGTNacgtn", "TGCANTGCAN")
    def __init__(s, fa):
        s.fh = open(fa, "rb"); s.idx = {}
        for L in open(fa + ".fai"):
            p = L.split(); s.idx[p[0]] = (int(p[1]), int(p[2]), int(p[3]), int(p[4]))
    def fetch(s, c, a, b):
        c = c if c in s.idx else c.lstrip("chr")
        if c not in s.idx: return None
        Ln, off, lb, lw = s.idx[c]
        if a < 0 or b > Ln: return None
        bp = lambda p: off + (p // lb) * lw + (p % lb)
        s.fh.seek(bp(a)); raw = s.fh.read(bp(b) - bp(a))
        return raw.replace(b"\n", b"").replace(b"\r", b"")[:b - a].decode("ascii").upper()

R = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t")); NP = len(R)
chrom = [r["chrom"] for r in R]; cen = [int(r["center"]) for r in R]; strd = [r["strand"] for r in R]

# gencode last exons
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
log(f"gtf exons {n:,}, transcripts {len(tx):,}")

term = defaultdict(list)
for t in tx.values():
    ex = t["exon"]
    if not ex: continue
    ex = sorted(ex)
    a, b = ex[-1] if t["strand"] == "+" else ex[0]
    term[(t["chrom"], t["strand"])].append((a, b))

def merge(ivs):
    if not ivs: return (np.empty(0, np.int64), np.empty(0, np.int64))
    ivs = sorted(ivs); cs = []; ce = []
    for a, b in ivs:
        if cs and a <= ce[-1]: ce[-1] = max(ce[-1], b)
        else: cs.append(a); ce.append(b)
    return np.array(cs, np.int64), np.array(ce, np.int64)
TERM = {k: merge(v) for k, v in term.items()}
log(f"merged terminal-exon intervals on {len(TERM)} (chrom,strand) keys")

empt = (np.empty(0, np.int64), np.empty(0, np.int64))
def ov(a, b, cse):
    cs, ce = cse
    if cs.size == 0: return 0.0
    lo = np.searchsorted(ce, a, "right"); hi = np.searchsorted(cs, b, "left")
    if hi <= lo: return 0.0
    s = np.maximum(cs[lo:hi], a); e = np.minimum(ce[lo:hi], b)
    return float(np.maximum(e - s, 0).sum())

G = Faidx(FASTA)
te_center = np.zeros(NP, bool); te_window = np.zeros(NP, bool); gc = np.full(NP, np.nan)
for i in range(NP):
    a, b = (cen[i] - WIN, cen[i]) if strd[i] == "+" else (cen[i], cen[i] + WIN)
    key = (chrom[i], strd[i]); iv = TERM.get(key, empt)
    te_window[i] = ov(a, b, iv) > 0
    te_center[i] = ov(cen[i], cen[i] + 1, iv) > 0
    q = G.fetch(chrom[i], a, b)
    if q:
        nb = q.count("A") + q.count("C") + q.count("G") + q.count("T")
        if nb > 0: gc[i] = (q.count("G") + q.count("C")) / nb
    if (i + 1) % 20000 == 0: log(f"  peak {i+1:,}/{NP:,}")

Lb = list(csv.DictReader(open("cpa_peaks_labeled.tsv"), delimiter="\t"))
fc = np.load("fine_compartment.npy")
fmap = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): int(fc[j]) for j, r in enumerate(Lb)}
te_argmax = np.array([fmap[(r["chrom"], int(r["center"]), r["strand"], r["gene"])] == 2 for r in R])

np.savez_compressed("termexon_gc_cache.npz", te_center=te_center, te_window=te_window,
                    te_argmax=te_argmax, gc500=gc)
isc = np.array([int(float(r["is_canonical"])) for r in R]).astype(bool)
print(f"rows {NP}, canonical {int(isc.sum())}, competitor {int((~isc).sum())}")
for nm, v in [("te_center", te_center), ("te_window", te_window), ("te_argmax", te_argmax)]:
    print(f"  {nm:<10} all {v.mean():.4f}  canonical {v[isc].mean():.4f}  competitor {v[~isc].mean():.4f}")
print(f"  GC500 missing {int(np.isnan(gc).sum())}, mean canonical {np.nanmean(gc[isc]):.4f}, "
      f"competitor {np.nanmean(gc[~isc]):.4f}")
log("DONE_TERMEXON_FLAG")
