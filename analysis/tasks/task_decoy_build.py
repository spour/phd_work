#!/usr/bin/env python3
# matched intronic decoys for every retained competitor: compartment fixed by design
import csv, re, sys, bisect, numpy as np
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)

GTF = "../train/data/annotation/gencode.v49.annotation.gtf"
ATL = "../train/data/atlas.clusters.2.0.GRCh38.96.tsv"
STEP, NDEC, EXCL_ATLAS, EXCL_PEAK, SEED = 50, 5, 1000, 1000, 42
rng = np.random.default_rng(SEED)

R = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
Z = np.load("corr_rowmask.npz"); ROW = Z["ROW"]
import pandas as pd
D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
D = D[D.gene.isin(set(r["gene"] for r in R))].reset_index(drop=True)
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(R)}
y = D.is_canonical.to_numpy().astype(bool)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); cen = D.center.to_numpy().astype(int)
strd = D.strand.to_numpy()
COMP = ROW & ~y
log(f"retained competitors {int(COMP.sum())} in {len({gene[i] for i in np.where(COMP)[0]})} genes")

# gencode: introns, terminal exons, splice sites, gene spans
TID = re.compile(r'transcript_id "([^"]+)"'); GNM = re.compile(r'gene_name "([^"]+)"')
tx = defaultdict(lambda: {"chrom": None, "strand": None, "gene": None, "exon": []})
n = 0
with open(GTF) as fh:
    for line in fh:
        if line[0] == "#": continue
        f = line.split("\t")
        if f[2] != "exon": continue
        m = TID.search(f[8])
        if not m: continue
        t = tx[m.group(1)]; t["chrom"] = f[0]; t["strand"] = f[6]
        g = GNM.search(f[8])
        if g: t["gene"] = g.group(1)
        t["exon"].append((int(f[3]) - 1, int(f[4])))
        n += 1
        if n % 4_000_000 == 0: log(f"  gtf exon {n:,}")
log(f"gtf exons {n:,}, transcripts {len(tx):,}")

intr = defaultdict(list); term = defaultdict(list)
acc = defaultdict(list); don = defaultdict(list); span = defaultdict(list)
for t in tx.values():
    ex = sorted(t["exon"])
    if not ex: continue
    key = (t["chrom"], t["strand"]); st = t["strand"]
    a, b = ex[-1] if st == "+" else ex[0]
    term[key].append((a, b))
    if t["gene"]: span[(t["gene"], t["chrom"], st)].append((ex[0][0], ex[-1][1]))
    for i in range(len(ex) - 1):
        lo, hi = ex[i][1], ex[i + 1][0]
        if hi <= lo: continue
        intr[key].append((lo, hi))
        if st == "+": don[key].append(lo); acc[key].append(hi)
        else: don[key].append(hi); acc[key].append(lo)
def merge(ivs):
    if not ivs: return []
    ivs = sorted(ivs); out = [list(ivs[0])]
    for a, b in ivs[1:]:
        if a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return [(a, b) for a, b in out]
INTR = {k: merge(v) for k, v in intr.items()}
TERM = {k: merge(v) for k, v in term.items()}
ACC = {k: np.array(sorted(set(v))) for k, v in acc.items()}
DON = {k: np.array(sorted(set(v))) for k, v in don.items()}
SPAN = {k: (min(a for a, _ in v), max(b for _, b in v)) for k, v in span.items()}
log(f"merged: introns on {len(INTR)} keys, terminal exons on {len(TERM)}")

ATLP = defaultdict(list)
with open(ATL) as fh:
    fh.readline()
    for line in fh:
        p = line.rstrip("\n").split("\t")
        ch = p[0] if p[0].startswith("chr") else "chr" + p[0]
        ATLP[(ch, p[5])].append(int(p[6]))
ATLP = {k: np.array(sorted(v)) for k, v in ATLP.items()}
log(f"atlas repSites on {len(ATLP)} keys")

def nearest(arr, x):
    if arr is None or len(arr) == 0: return np.full(len(x), 1e9)
    j = np.searchsorted(arr, x)
    lo = arr[np.clip(j - 1, 0, len(arr) - 1)]; hi = arr[np.clip(j, 0, len(arr) - 1)]
    return np.minimum(np.abs(x - lo), np.abs(hi - x)).astype(float)
def in_any(ivs, x):
    if not ivs: return np.zeros(len(x), bool)
    st = np.array([a for a, _ in ivs]); en = np.array([b for _, b in ivs])
    j = np.searchsorted(st, x, "right") - 1
    ok = j >= 0
    out = np.zeros(len(x), bool)
    out[ok] = x[ok] < en[j[ok]]
    return out

gidx = defaultdict(list)
for i in np.where(ROW)[0]: gidx[gene[i]].append(i)
rows = []
short = 0; nogene = 0; tot_need = 0
D3C, D5C, D3D, D5D = [], [], [], []
for g, ix in gidx.items():
    ix = np.array(ix); ci = ix[~y[ix]]
    if len(ci) == 0: continue
    ch = chrom[ci[0]]; st = strd[ci[0]]; key = (ch, st)
    sp = SPAN.get((g, ch, st))
    if sp is None: nogene += len(ci); continue
    iv = [(max(a, sp[0]), min(b, sp[1])) for a, b in INTR.get(key, []) if b > sp[0] and a < sp[1]]
    iv = [(a, b) for a, b in iv if b - a >= STEP]
    if not iv: nogene += len(ci); continue
    cand = np.concatenate([np.arange(a, b, STEP) for a, b in iv])
    cand = cand[~in_any(TERM.get(key, []), cand)]
    if len(cand): cand = cand[nearest(ATLP.get(key), cand) >= EXCL_ATLAS]
    if len(cand):
        pk = np.array(sorted(cen[ix]))
        cand = cand[nearest(pk, cand) >= EXCL_PEAK]
    if len(cand) == 0: nogene += len(ci); continue
    d3c = nearest(ACC.get(key), cand); d5c = nearest(DON.get(key), cand)
    U = np.column_stack([np.log10(1 + d3c), np.log10(1 + d5c)])
    d3p = nearest(ACC.get(key), cen[ci]); d5p = nearest(DON.get(key), cen[ci])
    V = np.column_stack([np.log10(1 + d3p), np.log10(1 + d5p)])
    used = np.zeros(len(cand), bool)
    order = rng.permutation(len(ci))
    tot_need += len(ci) * NDEC
    for t in order:
        d = np.abs(U - V[t]).sum(1)
        d[used] = np.inf
        take = np.argsort(d, kind="mergesort")[:NDEC]
        take = take[np.isfinite(d[take])]
        if len(take) < NDEC:
            extra = np.argsort(np.abs(U - V[t]).sum(1), kind="mergesort")[:NDEC - len(take)]
            take = np.concatenate([take, extra]); short += NDEC - len(take)
        used[take[:min(len(take), len(cand))]] = True
        D3C.append(d3p[t]); D5C.append(d5p[t])
        for k in take:
            rows.append((ch, int(cand[k]), st, g, int(ci[t])))
            D3D.append(d3c[k]); D5D.append(d5c[k])
with open("decoy_positions.tsv", "w") as fh:
    fh.write("chrom\tcenter\tstrand\tgene\tcompetitor_row\n")
    for r in rows: fh.write("\t".join(str(x) for x in r) + "\n")
D3C = np.array(D3C); D5C = np.array(D5C); D3D = np.array(D3D); D5D = np.array(D5D)
print(f"decoys {len(rows)} for {len(D3C)} competitors ({NDEC} each), "
      f"{nogene} competitors with no usable intronic candidate, {short} slots reused")
print(f"match quality, distance to nearest annotated 3'ss (nt): "
      f"competitor median {np.median(D3C):.0f} IQR [{np.percentile(D3C,25):.0f},{np.percentile(D3C,75):.0f}] | "
      f"decoy median {np.median(D3D):.0f} IQR [{np.percentile(D3D,25):.0f},{np.percentile(D3D,75):.0f}]")
print(f"match quality, distance to nearest annotated 5'ss (nt): "
      f"competitor median {np.median(D5C):.0f} IQR [{np.percentile(D5C,25):.0f},{np.percentile(D5C,75):.0f}] | "
      f"decoy median {np.median(D5D):.0f} IQR [{np.percentile(D5D,25):.0f},{np.percentile(D5D,75):.0f}]")
log("DONE_DECOY_BUILD")
