#!/usr/bin/env python3
# companion to as_overlap_test.py: is the null a real null
import sys, math, re
import numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
GTF = "../train/data/annotation/gencode.v49.annotation.gtf"
def two_sided_p(z): return math.erfc(abs(z) / math.sqrt(2.0))
def plab(p): return "p < 1e-300" if not (p > 1e-300) else ("p = %.1e" % p if p < 1e-3 else "p = %.3f" % p)
def woolf(a, b, c, d):
    if min(a, b, c, d) == 0: a, b, c, d = a+.5, b+.5, c+.5, d+.5
    lor = math.log(a*d/(b*c)); se = math.sqrt(1/a+1/b+1/c+1/d)
    return math.exp(lor), math.exp(lor-1.96*se), math.exp(lor+1.96*se), two_sided_p(lor/se)
def mh_or(a, b, c, d):
    a, b, c, d = (np.asarray(x, float) for x in (a, b, c, d)); n = a+b+c+d
    ok = (n > 0) & ((a+b) > 0) & ((c+d) > 0) & ((a+c) > 0) & ((b+d) > 0)
    a, b, c, d, n = a[ok], b[ok], c[ok], d[ok], n[ok]
    R, S = a*d/n, b*c/n
    if a.size == 0 or R.sum() == 0 or S.sum() == 0: return (float("nan"),)*4+(int(a.size),)
    orr = R.sum()/S.sum(); P, Q = (a+d)/n, (b+c)/n
    se = math.sqrt((P*R).sum()/(2*R.sum()**2) + (P*S+Q*R).sum()/(2*R.sum()*S.sum()) + (Q*S).sum()/(2*S.sum()**2))
    E = (a+b)*(a+c)/n; V = np.where(n > 1, (a+b)*(c+d)*(a+c)*(b+d)/(n*n*np.maximum(n-1, 1)), 0.0)
    chi = (abs(a.sum()-E.sum())-0.5)**2/V.sum() if V.sum() > 0 else float("nan")
    return orr, math.exp(math.log(orr)-1.96*se), math.exp(math.log(orr)+1.96*se), \
           (two_sided_p(math.sqrt(chi)) if chi == chi else float("nan")), int(a.size)

Z = np.load("as_overlap_flags.npz", allow_pickle=True)
def get(nm, k): return Z[f"{nm}__{k}"]
GENES = set(np.unique(get("COMP", "gene")).tolist())

# gencode introns of the analysis genes, per gene
log("[gencode] reading ...")
TID = re.compile(r'transcript_id "([^"]+)"'); GNM = re.compile(r'gene_name "([^"]+)"')
tx = defaultdict(list); tmeta = {}
with open(GTF) as fh:
    for line in fh:
        if line[0] == "#": continue
        f = line.split("\t", 9)
        if f[2] != "exon": continue
        m = TID.search(f[8]); g = GNM.search(f[8])
        if not m or not g or g.group(1) not in GENES: continue
        tx[m.group(1)].append((int(f[3])-1, int(f[4]))); tmeta[m.group(1)] = (f[0], f[6], g.group(1))
gintr = defaultdict(list)
for t, ex in tx.items():
    c, s, g = tmeta[t]; ex.sort()
    for (a1, b1), (a2, b2) in zip(ex, ex[1:]):
        if a2 > b1: gintr[(g, c, s)].append((b1, a2))
def merge(iv):
    iv = sorted(iv); out = []
    for a, b in iv:
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out
gintr = {k: merge(v) for k, v in gintr.items()}
log(f"[gencode] intron space for {len({k[0] for k in gintr}):,} genes")

# draw rand: same per-gene count as the decoy arm, uniform in intron space
dg = get("DEC", "gene"); need = pd.Series(dg).value_counts().to_dict()
rng = np.random.default_rng(42)
rc, rs, rp, rg = [], [], [], []
for (g, c, s), iv in gintr.items():
    n = need.get(g, 0)
    if n == 0: continue
    L = np.array([b-a for a, b in iv], float); cum = np.cumsum(L)
    if cum[-1] < 1: continue
    u = rng.random(n)*cum[-1]; j = np.searchsorted(cum, u)
    off = u - np.r_[0.0, cum[:-1]][j]
    pos = np.array([iv[k][0] for k in j]) + off.astype(np.int64)
    rc += [c]*n; rs += [s]*n; rp += pos.tolist(); rg += [g]*n
    need[g] = 0
RAND = dict(chrom=np.array(rc), strand=np.array(rs), pos=np.array(rp, np.int64), gene=np.array(rg))
log(f"[rand] {RAND['pos'].size:,} uniform intronic positions in {len(set(rg)):,} genes")

E = pd.read_csv("as_events.tsv", sep="\t")
def region(t, v):
    sub = E[(E.etype == t) & (E.variable == v)]; d = defaultdict(list)
    for c, a, b, s in zip(sub.chrom, sub.start0, sub.end0, sub.strand): d[(c, s)].append((a, b))
    return {k: (np.array([x[0] for x in m]), np.array([x[1] for x in m])) for k, m in ((k, merge(v2)) for k, v2 in d.items())}
def inside(M, chrom, strand, pos):
    out = np.zeros(pos.size, bool)
    for k in np.unique(np.char.add(np.char.add(chrom, "\t"), strand)):
        c, s = k.split("\t"); m = M.get((c, s))
        if m is None: continue
        sel = (chrom == c) & (strand == s); st, en = m; p = pos[sel]
        j = np.searchsorted(st, p, "right")-1
        ok = (j >= 0) & (en[np.clip(j, 0, len(en)-1)] > p)
        o = out[sel]; o[ok] = True; out[sel] = o
    return out
TYPES = ["INT", "EXFLANK", "ALTA_W", "ALTD_W"]
for t in TYPES:
    M = region(t, 1); RAND["v_"+t] = inside(M, RAND["chrom"], RAND["strand"], RAND["pos"])
RAND["v_ANYAS"] = np.zeros(RAND["pos"].size, bool)
for t in TYPES: RAND["v_ANYAS"] |= RAND["v_"+t]

LAB = {"INT": "intron retention (INT)", "EXFLANK": "introns flanking a cassette exon",
       "ALTA_W": "alt 3'ss +/-300 nt", "ALTD_W": "alt 5'ss +/-300 nt", "ANYAS": "any variable AS region"}
cm = get("COMP", "pure_intron"); dm = get("DEC", "pure_intron")
rows = []
print("=" * 86)
print("THIRD ARM: unmatched uniform intronic background (RAND), same genes, same n as decoys")
print("=" * 86)
for t in TYPES + ["ANYAS"]:
    col = "v_"+t
    yc = get("COMP", col)[cm]; yd = get("DEC", col)[dm]; yr = RAND[col]
    gc = get("COMP", "gene")[cm]; gr = RAND["gene"]
    a, b = int(yc.sum()), int((~yc).sum()); c, d = int(yr.sum()), int((~yr).sum())
    o1, l1, h1, p1 = woolf(a, b, c, d)
    genes = np.unique(np.concatenate([gc, gr])); gi = {g: i for i, g in enumerate(genes)}
    ic = np.array([gi[g] for g in gc]); ir = np.array([gi[g] for g in gr])
    ac = np.bincount(ic, yc.astype(float), len(genes)); nc = np.bincount(ic, None, len(genes))
    ar = np.bincount(ir, yr.astype(float), len(genes)); nr = np.bincount(ir, None, len(genes))
    m1, ml, mh, mp, ns = mh_or(ac, nc-ac, ar, nr-ar)
    print(f"\n### {LAB[t]}")
    print(f"  competitor {yc.mean():7.3%}   matched decoy {yd.mean():7.3%}   unmatched RAND {yr.mean():7.3%}")
    print(f"  competitor vs RAND   pooled OR {o1:.3f} [{l1:.3f},{h1:.3f}] {plab(p1)}"
          f"   within-gene MH OR {m1:.3f} [{ml:.3f},{mh:.3f}] {plab(mp)}  ({ns:,} genes)")
    rows.append(dict(contrast="COMP vs RAND: "+LAB[t], flag=col, comp_rate=yc.mean(),
                     dec_rate=yd.mean(), rand_rate=yr.mean(), or_pooled=o1, ci_lo=l1, ci_hi=h1,
                     p_pooled=p1, or_mh=m1, mh_lo=ml, mh_hi=mh, p_mh=mp, n_genes=ns))
pd.DataFrame(rows).to_csv("as_overlap_background.tsv", sep="\t", index=False, float_format="%.6g")
log("[out] as_overlap_background.tsv")
