#!/usr/bin/env python3
# cheap (no genome scan) per-peak environment annotation for all 79
import csv, re, sys, bisect
import numpy as np
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)

GTF  = "../train/data/annotation/gencode.v49.annotation.gtf"
ATL  = "../train/data/atlas.clusters.2.0.GRCh38.96.tsv"
PADB = "../train/data/polyadb_v4/HumanPas/hg38.PAS.main.tsv"
LAB  = "cpa_peaks_labeled.tsv"
META = "fig3_ucube_meta.tsv"
WIN  = 500

# read master peaks
R = list(csv.DictReader(open(LAB), delimiter="\t"))
chrom = [r["chrom"] for r in R]; cen = [int(r["center"]) for r in R]; strd = [r["strand"] for r in R]
gname = [r["gene"] for r in R]; NP = len(R)
log(f"peaks {NP}")

# gtf: compartment intervals + gene bounds, one pass
TID = re.compile(r'transcript_id "([^"]+)"')
GNM = re.compile(r'gene_name "([^"]+)"')
tx = defaultdict(lambda: {"chrom": None, "strand": None, "exon": [], "utr": [], "cds": []})
gbounds = {}
n = 0
with open(GTF) as fh:
    for line in fh:
        if line[0] == "#": continue
        f = line.split("\t"); ft = f[2]
        if ft == "gene":
            m = GNM.search(f[8])
            if m:
                gn = m.group(1); a = int(f[3]) - 1; b = int(f[4])
                if gn in gbounds:
                    gb = gbounds[gn]; gb[2] = min(gb[2], a); gb[3] = max(gb[3], b)
                else:
                    gbounds[gn] = [f[0], f[6], a, b]
            continue
        if ft not in ("exon", "UTR", "CDS"): continue
        m = TID.search(f[8])
        if not m: continue
        t = tx[m.group(1)]; t["chrom"] = f[0]; t["strand"] = f[6]
        iv = (int(f[3]) - 1, int(f[4]))
        t["exon" if ft == "exon" else ("utr" if ft == "UTR" else "cds")].append(iv)
        n += 1
        if n % 4_000_000 == 0: log(f"  gtf {n:,}")

exon_by = defaultdict(list); tx_by = defaultdict(list); term_by = defaultdict(list)
for t in tx.values():
    if not t["exon"]: continue
    st = t["strand"]; key = (t["chrom"], st); ex = t["exon"]
    exon_by[key].extend(ex); tx_by[key].append((min(a for a, _ in ex), max(b for _, b in ex)))
    term_by[key].append(max(ex, key=lambda iv: iv[1]) if st == "+" else min(ex, key=lambda iv: iv[0]))
    if t["cds"] and t["utr"]:
        cmin = min(a for a, _ in t["cds"]); cmax = max(b for _, b in t["cds"])
        for a, b in t["utr"]:
            if (st == "+" and a >= cmax) or (st == "-" and b <= cmin): term_by[key].append((a, b))

def merge(ivs):
    if not ivs: return (np.empty(0, np.int64), np.empty(0, np.int64))
    ivs = sorted(ivs); cs = []; ce = []
    for a, b in ivs:
        if cs and a <= ce[-1]: ce[-1] = max(ce[-1], b)
        else: cs.append(a); ce.append(b)
    return np.array(cs, np.int64), np.array(ce, np.int64)
TERM = {k: merge(v) for k, v in term_by.items()}
EXON = {k: merge(v) for k, v in exon_by.items()}
TXS  = {k: merge(v) for k, v in tx_by.items()}
empt = (np.empty(0, np.int64), np.empty(0, np.int64))
def ov(a, b, cse):
    cs, ce = cse
    if cs.size == 0: return 0.0
    lo = np.searchsorted(ce, a, "right"); hi = np.searchsorted(cs, b, "left")
    if hi <= lo: return 0.0
    s = np.maximum(cs[lo:hi], a); e = np.minimum(ce[lo:hi], b); return float(np.maximum(e - s, 0).sum())

def compartment(c, ctr, st):
    a, b = (ctr - WIN, ctr) if st == "+" else (ctr, ctr + WIN); key = (c, st)
    term = ov(a, b, TERM.get(key, empt)); exon = ov(a, b, EXON.get(key, empt)); txv = ov(a, b, TXS.get(key, empt))
    internal = max(exon - term, 0.0); intron = max(txv - exon, 0.0); inter = max(WIN - txv, 0.0)
    vals = np.array([term, internal, intron, inter]); j = int(vals.argmax())
    return j if vals[j] >= 0.8 * WIN else 3

comp = np.full(NP, 3, np.int8)
fracpos = np.full(NP, np.nan); tss_dist = np.full(NP, np.nan); glen = np.full(NP, np.nan)
miss_gene = 0
for i in range(NP):
    comp[i] = compartment(chrom[i], cen[i], strd[i])
    gb = gbounds.get(gname[i])
    if gb is None: miss_gene += 1; continue
    _, gst, a0, b1 = gb; c = cen[i]
    if gst == "+": d = c - a0; L = b1 - a0
    else:          d = b1 - c; L = b1 - a0
    glen[i] = L; tss_dist[i] = d
    fracpos[i] = d / L if L > 0 else np.nan
log(f"compartment done; genes missing bounds: {miss_gene}")

# atlas / polyadb nearest-site helpers
def build_atlas():
    recs = defaultdict(list)
    with open(ATL) as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t"); ch = p[0]
            if not ch.startswith("chr"): ch = "chr" + ch
            st = p[5]; rep = int(p[6]); tpm = float(p[4]); frac = float(p[7]); nprot = int(p[8])
            nsamp = sum(1 for x in p[13:] if x not in ("0", "0.0", "", "0.000000"))
            recs[(ch, st)].append((rep, tpm, frac, nprot, nsamp))
    for k in recs: recs[k].sort()
    return recs

def fv(x):
    try: return float(x)
    except (ValueError, TypeError): return np.nan
def build_padb():
    recs = defaultdict(list)
    with open(PADB) as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        for p in rd:
            pid = p["PAS_ID"].split(":")
            if len(pid) != 3: continue
            ch, st, pos = pid[0], pid[1], int(pid[2])
            if not ch.startswith("chr"): ch = "chr" + ch
            recs[(ch, st)].append((pos, fv(p["PolyaStrength"]), fv(p["PolyaStrength_percentile"]),
                                   fv(p["AvgRPM.3READS"]), fv(p["PSE.3READS"])))
    for k in recs: recs[k].sort()
    return recs

def nearest(recs, chs, cns, sts, nfield):
    m = len(chs)
    dist = np.full(m, 1e9); vals = [np.full(m, np.nan) for _ in range(nfield)]
    for i in range(m):
        a = recs.get((chs[i], sts[i]))
        if not a: continue
        keys = [x[0] for x in a]; c = cns[i]
        j = bisect.bisect_left(keys, c); best = None; bd = 1e9
        for k in (j - 1, j):
            if 0 <= k < len(a):
                dd = abs(keys[k] - c)
                if dd < bd: bd = dd; best = a[k]
        dist[i] = bd
        if best is not None:
            for f in range(nfield): vals[f][i] = best[1 + f]
    return dist, vals

log("building atlas / polyadb indices ...")
ATLR = build_atlas(); PADR = build_padb()
log(f"atlas clusters {sum(len(v) for v in ATLR.values())}  padb sites {sum(len(v) for v in PADR.values())}")

a_dist, (a_tpm, a_frac, a_nprot, a_nsamp) = nearest(ATLR, chrom, cen, strd, 4)
p_dist, (p_str, p_pct, p_rpm, p_pse) = nearest(PADR, chrom, cen, strd, 4)

np.savez("env_annot_all.npz",
         comp=comp, fracpos=fracpos, tss_dist=tss_dist, gene_len=glen,
         atlas_dist=a_dist, atlas_tpm=a_tpm, atlas_frac=a_frac, atlas_nprot=a_nprot, atlas_nsamp=a_nsamp,
         padb_dist=p_dist, padb_strength=p_str, padb_pct=p_pct, padb_rpm=p_rpm, padb_pse=p_pse)
log(f"saved env_annot_all.npz  atlas matched@25nt {(a_dist<=25).sum()}  padb matched@25nt {(p_dist<=25).sum()}")

# decoys
M = list(csv.DictReader(open(META), delimiter="\t"))
mchrom = [r["chrom"] for r in M]; mstr = [r["strand"] for r in M]
D = np.load("decoy_set.npz"); dpos = D["pos"].astype(int); dcomp = D["comp"].astype(int)
dchrom = [mchrom[c] for c in dcomp]; dstr = [mstr[c] for c in dcomp]; dcen = list(dpos)
da_dist, (da_tpm, da_frac, da_nprot, da_nsamp) = nearest(ATLR, dchrom, dcen, dstr, 4)
dp_dist, (dp_str, dp_pct, dp_rpm, dp_pse) = nearest(PADR, dchrom, dcen, dstr, 4)
dcomp_comp = np.array([compartment(dchrom[j], dcen[j], dstr[j]) for j in range(len(dpos))], np.int8)
np.savez("env_annot_decoy.npz",
         comp=dcomp_comp, atlas_dist=da_dist, atlas_tpm=da_tpm, atlas_frac=da_frac,
         atlas_nprot=da_nprot, atlas_nsamp=da_nsamp,
         padb_dist=dp_dist, padb_strength=dp_str, padb_pct=dp_pct, padb_rpm=dp_rpm, padb_pse=dp_pse)
log(f"saved env_annot_decoy.npz  n={len(dpos)}  atlas matched@25nt {(da_dist<=25).sum()}")
log("DONE_ENV_ANNOT")
