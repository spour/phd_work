#!/usr/bin/env python3
# do cryptic poly(a) sites (competitors) sit in alternatively spliced regions more often
import sys, math, csv
import numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)

GTF = "../train/data/annotation/gencode.v49.annotation.gtf"

def two_sided_p(z): return math.erfc(abs(z) / math.sqrt(2.0))
def plab(p):
    if p != p: return "p = NA"
    if not (p > 1e-300): return "p < 1e-300"
    return "p = %.1e" % p if p < 1e-3 else "p = %.3f" % p

def wilcoxon_signed_rank_p(d):
    d = d[d != 0]; n = d.size
    if n < 10: return float("nan")
    a = np.abs(d); order = np.argsort(a, kind="mergesort"); sa = a[order]
    newgrp = np.r_[True, sa[1:] != sa[:-1]]; grp = np.cumsum(newgrp)
    starts = np.r_[np.nonzero(newgrp)[0], n]
    avg = (starts[:-1] + starts[1:] + 1) / 2.0
    ranks = np.empty(n); ranks[order] = avg[grp - 1]
    W = ranks[d > 0].sum(); mu = n * (n + 1) / 4.0
    tie = np.bincount(grp)[1:].astype(float)
    var = (n * (n + 1) * (2 * n + 1) - (tie ** 3 - tie).sum() / 2.0) / 24.0
    return two_sided_p((W - mu) / math.sqrt(var)) if var > 0 else float("nan")

def woolf(a, b, c, d):
    if min(a, b, c, d) == 0: a, b, c, d = a + .5, b + .5, c + .5, d + .5
    lor = math.log(a * d / (b * c)); se = math.sqrt(1/a + 1/b + 1/c + 1/d)
    return math.exp(lor), math.exp(lor - 1.96*se), math.exp(lor + 1.96*se), two_sided_p(lor/se)

def mh_or(a, b, c, d):
    a, b, c, d = (np.asarray(x, float) for x in (a, b, c, d))
    n = a + b + c + d
    ok = (n > 0) & ((a + b) > 0) & ((c + d) > 0) & ((a + c) > 0) & ((b + d) > 0)
    a, b, c, d, n = a[ok], b[ok], c[ok], d[ok], n[ok]
    if a.size == 0: return (float("nan"),)*4 + (0,)
    R = a*d/n; S = b*c/n
    if R.sum() == 0 or S.sum() == 0: return (float("nan"),)*4 + (int(a.size),)
    orr = R.sum()/S.sum()
    P, Q = (a+d)/n, (b+c)/n
    var = ((P*R).sum()/(2*R.sum()**2) + ((P*S + Q*R).sum())/(2*R.sum()*S.sum())
           + (Q*S).sum()/(2*S.sum()**2))
    se = math.sqrt(var)
    E = (a+b)*(a+c)/n
    V = (a+b)*(c+d)*(a+c)*(b+d)/(n*n*(n-1))
    V = np.where(n > 1, V, 0.0)
    chi = (abs(a.sum() - E.sum()) - 0.5)**2 / V.sum() if V.sum() > 0 else float("nan")
    p = two_sided_p(math.sqrt(chi)) if chi == chi else float("nan")
    return orr, math.exp(math.log(orr) - 1.96*se), math.exp(math.log(orr) + 1.96*se), p, int(a.size)

# sites
P = pd.read_parquet("cpa_peaks_final.parquet")
A = P[P.in_analysis_set]
COMP = A[~A.is_canonical][["gene","chrom","center","strand"]].copy()
REAL = A[A.is_canonical][["gene","chrom","center","strand"]].copy()
log(f"[sites] analysis set: {len(COMP):,} competitors, {len(REAL):,} real 3' ends, {A.gene.nunique():,} genes")

R = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = pd.read_parquet("shortlist_feat_all.parquet")
D = D[D.gene.isin(set(r["gene"] for r in R))].reset_index(drop=True)
dec = pd.read_csv("decoy_positions.tsv", sep="\t")
par = D.iloc[dec.competitor_row.to_numpy()][["gene","chrom","center","strand"]].to_numpy()
assert (par[:,0] == dec.gene.to_numpy()).all()
key = set(map(tuple, COMP[["gene","chrom","center","strand"]].to_numpy()))
keep = np.array([tuple(r) in key for r in par])
DEC = dec[keep].copy()
DEC["parent"] = [f"{r[0]}|{r[1]}|{r[2]}|{r[3]}" for r in par[keep]]
npar = DEC.parent.nunique()
log(f"[sites] decoys of analysis-set competitors: {len(DEC):,} for {npar:,} competitors "
    f"({len(COMP)-npar:,} competitors have no decoy: no usable intronic candidate in the gene)")

# gencode compartment
log("[gencode] reading exons ...")
import re
TID = re.compile(r'transcript_id "([^"]+)"')
tx = defaultdict(list); tmeta = {}
with open(GTF) as fh:
    for line in fh:
        if line[0] == "#": continue
        f = line.split("\t", 9)
        if f[2] != "exon": continue
        m = TID.search(f[8])
        if not m: continue
        t = m.group(1); tx[t].append((int(f[3]) - 1, int(f[4]))); tmeta[t] = (f[0], f[6])
ex_iv = defaultdict(list); in_iv = defaultdict(list)
for t, ex in tx.items():
    c, s = tmeta[t]; ex.sort()
    for a, b in ex: ex_iv[(c, s)].append((a, b))
    for (a1, b1), (a2, b2) in zip(ex, ex[1:]):
        if a2 > b1: in_iv[(c, s)].append((b1, a2))
log(f"[gencode] {len(tx):,} transcripts")

def merge(iv):
    iv = sorted(iv); out = []
    for a, b in iv:
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])

def build(d):
    return {k: merge(v) for k, v in d.items()}
EXM, INM = build(ex_iv), build(in_iv)

def inside(M, chrom, strand, pos):
    out = np.zeros(pos.size, bool)
    for k in np.unique(np.array([f"{c}\t{s}" for c, s in zip(chrom, strand)])):
        c, s = k.split("\t")
        m = M.get((c, s))
        sel = (chrom == c) & (strand == s)
        if m is None: continue
        st, en = m; p = pos[sel]
        j = np.searchsorted(st, p, "right") - 1
        ok = (j >= 0) & (en[np.clip(j, 0, len(en)-1)] > p)
        o = out[sel]; o[ok] = True; out[sel] = o
    return out

def sitearr(df, poscol="center"):
    return (df.chrom.to_numpy().astype(str), df.strand.to_numpy().astype(str),
            df[poscol].to_numpy().astype(np.int64), df.gene.to_numpy().astype(str))

SETS = {}
for nm, df, pc in (("COMP", COMP, "center"), ("DEC", DEC, "center"), ("REAL", REAL, "center")):
    ch, sd, po, gn = sitearr(df, pc)
    SETS[nm] = dict(chrom=ch, strand=sd, pos=po, gene=gn)
    SETS[nm]["intron"] = inside(INM, ch, sd, po)
    SETS[nm]["exon"] = inside(EXM, ch, sd, po)
for nm in SETS:
    s = SETS[nm]; n = s["pos"].size
    pure = s["intron"] & ~s["exon"]
    log(f"[compartment] {nm:5s} n={n:7,}  in an annotated intron {s['intron'].mean():6.1%}  "
        f"in an annotated exon {s['exon'].mean():6.1%}  intron-only {pure.mean():6.1%}")
    s["pure_intron"] = pure

# as regions
E = pd.read_csv("as_events.tsv", sep="\t")
def region(etype, variable):
    sub = E[(E.etype == etype) & (E.variable == variable)]
    d = defaultdict(list)
    for c, a, b, s in zip(sub.chrom, sub.start0, sub.end0, sub.strand): d[(c, s)].append((a, b))
    return build(d)

TYPES = ["INT", "EXFLANK", "EX", "ALTA_W", "ALTD_W", "ALTA", "ALTD"]
REG = {t: region(t, 1) for t in TYPES}
REG_ANY = {t: region(t, 0) for t in TYPES}
for nm, s in SETS.items():
    for t in TYPES:
        s["v_" + t] = inside(REG[t], s["chrom"], s["strand"], s["pos"])
        s["n_" + t] = inside(REG_ANY[t], s["chrom"], s["strand"], s["pos"])
    s["v_ANYAS"] = np.zeros(s["pos"].size, bool)
    for t in ("INT", "EXFLANK", "ALTA_W", "ALTD_W"):
        s["v_ANYAS"] |= s["v_" + t]
np.savez_compressed("as_overlap_flags.npz",
                    **{f"{nm}__{k}": v for nm, s in SETS.items() for k, v in s.items()})

# reporting
OUT = []
def emit(**kw): OUT.append(kw)

def contrast(tag, col, mask_name, note=""):
    C, Dc = SETS["COMP"], SETS["DEC"]
    mc = C[mask_name] if mask_name else np.ones(C["pos"].size, bool)
    md = Dc[mask_name] if mask_name else np.ones(Dc["pos"].size, bool)
    yc, yd = C[col][mc], Dc[col][md]
    a, b = int(yc.sum()), int((~yc).sum())
    c, d = int(yd.sum()), int((~yd).sum())
    orr, lo, hi, p = woolf(a, b, c, d)
    gc, gd = C["gene"][mc], Dc["gene"][md]
    genes = np.unique(np.concatenate([gc, gd]))
    gi = {g: i for i, g in enumerate(genes)}
    ic = np.array([gi[g] for g in gc]); idd = np.array([gi[g] for g in gd])
    ac = np.bincount(ic, yc.astype(float), len(genes)); nc = np.bincount(ic, None, len(genes))
    ad = np.bincount(idd, yd.astype(float), len(genes)); nd = np.bincount(idd, None, len(genes))
    rng = np.random.default_rng(42); bs = []
    for _ in range(2000):
        k = rng.integers(0, len(genes), len(genes))
        A1, N1, A2, N2 = ac[k].sum(), nc[k].sum(), ad[k].sum(), nd[k].sum()
        B1, B2 = N1 - A1, N2 - A2
        if min(A1, B1, A2, B2) <= 0: continue
        bs.append(math.log(A1 * B2 / (B1 * A2)))
    blo, bhi = (math.exp(np.percentile(bs, 2.5)), math.exp(np.percentile(bs, 97.5))) if len(bs) > 100 else (float("nan"),)*2
    m_or, m_lo, m_hi, m_p, nstrat = mh_or(ac, nc - ac, ad, nd - ad)
    both = (nc > 0) & (nd > 0)
    diff = (ac[both]/nc[both]) - (ad[both]/nd[both])
    w_p = wilcoxon_signed_rank_p(diff)
    up, dn = int((diff > 0).sum()), int((diff < 0).sum())
    print(f"\n### {tag}{'  ['+note+']' if note else ''}")
    print(f"  competitors {a:,}/{a+b:,} = {a/(a+b):.3%}   decoys {c:,}/{c+d:,} = {c/(c+d):.3%}")
    print(f"  pooled OR {orr:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  {plab(p)}"
          f"   gene-cluster bootstrap CI [{blo:.3f}, {bhi:.3f}]")
    print(f"  within-gene (Mantel-Haenszel, {nstrat:,} genes) OR {m_or:.3f} "
          f"95% CI [{m_lo:.3f}, {m_hi:.3f}]  {plab(m_p)}")
    print(f"  paired per-gene rates: {up:,} genes competitor-higher, {dn:,} decoy-higher, "
          f"median diff {np.median(diff):+.4f}, Wilcoxon {plab(w_p)}")
    emit(contrast=tag, mask=mask_name or "all", flag=col,
         comp_hit=a, comp_n=a+b, comp_rate=a/(a+b), dec_hit=c, dec_n=c+d, dec_rate=c/(c+d),
         or_pooled=orr, ci_lo=lo, ci_hi=hi, p_pooled=p, boot_lo=blo, boot_hi=bhi,
         or_mh=m_or, mh_lo=m_lo, mh_hi=m_hi, p_mh=m_p, n_genes=nstrat,
         genes_up=up, genes_down=dn, median_gene_diff=float(np.median(diff)), p_wilcoxon=w_p)

print("=" * 78)
print("STEP 2  INTRON RETENTION: is the intron a cryptic pA site sits in a variable one?")
print("=" * 78)
contrast("IR / all intronic sites", "v_INT", "pure_intron",
         "site inside a variable intron-retention event")
for nm in ("COMP", "DEC", "REAL"):
    s = SETS[nm]; s["in_any_INT"] = (s["v_INT"] | s["n_INT"]) & s["pure_intron"]
    log(f"[step2] {nm}: {int(s['in_any_INT'].sum()):,} sites lie in a VastDB-annotated intron "
        f"({int(s['v_INT'].sum()):,} of them variable)")
contrast("IR | intron is in VastDB", "v_INT", "in_any_INT",
         "denominator = introns VastDB scored at all")

print()
print("=" * 78)
print("STEP 1  OVERLAP WITH VARIABLE AS REGIONS, BY EVENT TYPE (intronic sites only)")
print("=" * 78)
LAB = {"INT": "intron retention (INT)", "EXFLANK": "introns flanking a cassette exon (EX)",
       "ALTA_W": "alt 3'ss segment +/-300 nt (ALTA)", "ALTD_W": "alt 5'ss segment +/-300 nt (ALTD)",
       "ALTA": "alt 3'ss segment, exact", "ALTD": "alt 5'ss segment, exact",
       "EX": "the cassette exon itself", "ANYAS": "any variable AS region (pooled)"}
for t in ("INT", "EXFLANK", "ALTA_W", "ALTD_W", "ALTA", "ALTD", "EX", "ANYAS"):
    contrast(LAB[t], "v_" + t, "pure_intron")

print()
print("=" * 78)
print("REFERENCE DIRECTION  real 3' ends (canonical, terminal-exon) vs the same regions")
print("=" * 78)
Cc, Rr, Dd = SETS["COMP"], SETS["REAL"], SETS["DEC"]
rows = []
for t in ("INT", "EXFLANK", "ALTA_W", "ALTD_W", "ANYAS"):
    col = "v_" + t
    rc = Cc[col][Cc["pure_intron"]].mean()
    rd = Dd[col][Dd["pure_intron"]].mean()
    rr = Rr[col].mean()
    a, b = int(Rr[col].sum()), int((~Rr[col]).sum())
    c, d = int(Dd[col][Dd["pure_intron"]].sum()), int((~Dd[col][Dd["pure_intron"]]).sum())
    orr, lo, hi, p = woolf(a, b, c, d)
    print(f"  {LAB[t]:42s} real {rr:6.2%}  competitor {rc:6.2%}  decoy {rd:6.2%}   "
          f"real-vs-decoy OR {orr:.3f} [{lo:.3f},{hi:.3f}] {plab(p)}")
    rows.append(dict(contrast="REAL vs DECOY: " + LAB[t], mask="real: all; decoy: pure_intron",
                     flag=col, comp_hit=a, comp_n=a+b, comp_rate=rr, dec_hit=c, dec_n=c+d,
                     dec_rate=rd, or_pooled=orr, ci_lo=lo, ci_hi=hi, p_pooled=p))
OUT.extend(rows)
pd.DataFrame(OUT).to_csv("as_overlap_stats.tsv", sep="\t", index=False, float_format="%.6g")
log("[out] as_overlap_stats.tsv, as_overlap_flags.npz")
