#!/usr/bin/env python3
# what distinguishes the model false positives that a u1 amo experiment actually used
import os, sys
import numpy as np
import pandas as pd
from collections import defaultdict

def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, NBOOT, NPERM = 42, 2000, 10000
rng = np.random.default_rng(SEED)
PC_GENES = "../train/data/pc_genes.bed"
EXONS = "gencode_canonical_exons.tsv"

# the data
T = pd.read_parquet("cpa_peaks_final.parquet", engine="fastparquet")
A = T[T.in_analysis_set].reset_index(drop=True)
P = pd.read_csv("ovl_ana_per_peak.tsv", sep="\t")
P["is_canonical"] = P.is_canonical.astype(bool)
key = ["chrom", "pos", "strand", "gene"]
A = A.rename(columns={"center": "pos"})
D = A.merge(P[key + ["hit_oh_u1amo_50", "hit_oh_camo_50", "hit_atlas3seq_50",
                     "dist_oh_u1amo", "dist_oh_camo"]], on=key, how="left", validate="1:1")
assert len(D) == len(A) and D.hit_oh_u1amo_50.notna().all(), "join lost rows"
for c in ["hit_oh_u1amo_50", "hit_oh_camo_50", "hit_atlas3seq_50"]:
    D[c] = D[c].astype(bool)

CAN = D.is_canonical.to_numpy()
FP = ~CAN
U1 = FP & D.hit_oh_u1amo_50.to_numpy()
CT = FP & D.hit_oh_camo_50.to_numpy()
NH = FP & ~D.hit_oh_u1amo_50.to_numpy()
print(f"analysis set {len(D):,}: {int(CAN.sum()):,} canonical, {int(FP.sum()):,} false positives")
print(f"  U1 AMO hit      {int(U1.sum()):>6,}    not hit {int(NH.sum()):>7,}")
print(f"  control AMO hit {int(CT.sum()):>6,}")
print(f"  hit by both     {int((U1 & CT).sum()):>6,}")

# stats
def midrank(s):
    o = np.argsort(s, kind="mergesort"); sv = s[o]; n = len(s); rk = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    rr = np.empty(n); rr[o] = rk
    return rr

def cl_A(x, y):
    s = np.concatenate([x, y]); rr = midrank(s)
    n1, n2 = len(x), len(y)
    return float((rr[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))

def mannwhitney(x, y):
    n1, n2 = len(x), len(y); N = n1 + n2
    s = np.concatenate([x, y]); rr = midrank(s)
    U = rr[:n1].sum() - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    _, cnt = np.unique(s, return_counts=True)
    tie = float((cnt ** 3 - cnt).sum())
    var = n1 * n2 / 12 * ((N + 1) - tie / (N * (N - 1)))
    if var <= 0: return float(U), float("nan"), float("nan")
    z = (U - mu) / np.sqrt(var)
    p = 2 * 0.5 * erfc_(abs(z) / np.sqrt(2))
    return float(U), float(z), float(min(1.0, p))

def erfc_(x):
    x = np.asarray(x, float)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
         + 0.254829592) * t * np.exp(-x * x)
    return float(y)

GENE = D.gene.to_numpy()
def boot_delta(x_mask, y_mask, v):
    gl = defaultdict(list)
    idx = np.where(x_mask | y_mask)[0]
    for i in idx: gl[GENE[i]].append(i)
    gg = [np.array(v_) for v_ in gl.values()]
    out = np.empty(NBOOT); out[:] = np.nan
    for b in range(NBOOT):
        sel = rng.integers(0, len(gg), len(gg))
        ii = np.concatenate([gg[j] for j in sel])
        xm = x_mask[ii]; ym = y_mask[ii]
        if xm.sum() < 2 or ym.sum() < 2: continue
        out[b] = 2 * cl_A(v[ii][xm], v[ii][ym]) - 1
    out = out[np.isfinite(out)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

def q(v): return np.percentile(v, [25, 50, 75])

def compare(name, v, a_mask, b_mask, alab, blab, boot=True):
    m = np.isfinite(v)
    x = v[a_mask & m]; y = v[b_mask & m]
    d = 2 * cl_A(x, y) - 1
    U, z, p = mannwhitney(x, y)
    lo, hi = boot_delta(a_mask & m, b_mask & m, v) if boot else (np.nan, np.nan)
    qa, qb = q(x), q(y)
    print(f"  {name:<22}{alab:<16}n={len(x):>6,}  median {qa[1]:>8.3f} [{qa[0]:.3f}, {qa[2]:.3f}]")
    print(f"  {'':<22}{blab:<16}n={len(y):>6,}  median {qb[1]:>8.3f} [{qb[0]:.3f}, {qb[2]:.3f}]")
    print(f"  {'':<22}Cliff delta {d:+.4f} [{lo:+.4f}, {hi:+.4f}]   "
          f"Mann-Whitney z {z:+.2f}, p {p:.3g}")
    return dict(quantity=name, group_a=alab, group_b=blab, n_a=len(x), n_b=len(y),
                median_a=qa[1], q1_a=qa[0], q3_a=qa[2], median_b=qb[1], q1_b=qb[0], q3_b=qb[2],
                cliff_delta=d, delta_lo=lo, delta_hi=hi, mw_z=z, mw_p=p)

rows = []
QTY = [("discriminant", D.discriminant.to_numpy(float)),
       ("spl_upstream_max_me3", D.spl_upstream_max_me3.to_numpy(float)),
       ("gc_contrast", D.gc_contrast.to_numpy(float))]
for nm, v in QTY:
    print(f"\n===== {nm} =====")
    rows.append(compare(nm, v, U1, NH, "U1 AMO hit", "not hit"))
    rows.append(compare(nm, v, CT, FP & ~D.hit_oh_camo_50.to_numpy(), "control AMO hit", "not hit"))
    rows.append(compare(nm, v, CAN, U1, "canonical", "U1 AMO hit"))
    rows.append(compare(nm, v, CAN, NH, "canonical", "not hit"))
pd.DataFrame(rows).to_csv("u1_hit_stats.tsv", sep="\t", index=False)

# distributions for the figure
with open("u1_hit_dist.tsv", "w") as fh:
    fh.write("quantity\tgroup\tcentre\tdensity\tn\n")
    for nm, v in QTY:
        m = np.isfinite(v)
        lo, hi = np.percentile(v[m], [0.5, 99.5])
        edges = np.linspace(lo, hi, 61); ctr = .5 * (edges[1:] + edges[:-1]); w = edges[1] - edges[0]
        for glab, gm in [("canonical", CAN), ("U1 AMO hit", U1),
                         ("control AMO hit", CT), ("not hit", NH)]:
            c, _ = np.histogram(v[gm & m], bins=edges)
            for xx, dd in zip(ctr, c / max(c.sum(), 1) / w):
                fh.write(f"{nm}\t{glab}\t{xx:.6f}\t{dd:.6f}\t{int((gm & m).sum())}\n")

# within gene
print("\n===== WITHIN GENE, discriminant =====")
disc = D.discriminant.to_numpy(float)
gi = defaultdict(list)
for i in range(len(D)):
    if FP[i] and np.isfinite(disc[i]): gi[GENE[i]].append(i)
prs = []
for g, ix in gi.items():
    ix = np.array(ix); h = U1[ix]
    if h.sum() == 0 or (~h).sum() == 0: continue
    prs.append((g, float(disc[ix][h].mean() - disc[ix][~h].mean()), int(h.sum()), len(ix)))
PR = pd.DataFrame(prs, columns=["gene", "diff", "n_hit", "n_fp"])
d = PR["diff"].to_numpy()
print(f"  genes with at least one hit and one non-hit false positive: {len(PR):,}")
print(f"  they hold {int(PR.n_hit.sum()):,} of the {int(U1.sum()):,} hits and "
      f"{int(PR.n_fp.sum() - PR.n_hit.sum()):,} non-hit false positives")
print(f"  mean paired difference {d.mean():+.4f}   median {np.median(d):+.4f}   "
      f"fraction above zero {np.mean(d > 0):.4f}")
GX = [np.array(ix) for g, ix in gi.items()
      if U1[np.array(ix)].sum() > 0 and (~U1[np.array(ix)]).sum() > 0]
obs = float(np.mean([disc[ix][U1[ix]].mean() - disc[ix][~U1[ix]].mean() for ix in GX]))
null = np.empty(NPERM)
for b in range(NPERM):
    acc = 0.0
    for ix in GX:
        k = int(U1[ix].sum()); n = len(ix)
        pick = rng.permutation(n)[:k]
        s = disc[ix]; mh = s[pick].mean()
        tot = s.sum()
        acc += mh - (tot - s[pick].sum()) / (n - k)
    null[b] = acc / len(GX)
ge = int((null >= obs).sum())
pperm = (1 + ge) / (1 + NPERM)
print(f"  within-gene permutation null {null.mean():+.4f} "
      f"[{np.percentile(null,2.5):+.4f}, {np.percentile(null,97.5):+.4f}]   "
      f"observed {obs:+.4f}   p {'< %.0e' % (1/NPERM) if ge == 0 else '= %.3g' % pperm} "
      f"({NPERM:,} shuffles)")
PR.to_csv("u1_hit_paired.tsv", sep="\t", index=False)
with open("u1_hit_paired.tsv", "a") as fh:
    fh.write(f"#n_genes\t{len(PR)}\t\t\n#mean_diff\t{d.mean():.6f}\t\t\n")
    fh.write(f"#median_diff\t{np.median(d):.6f}\t\t\n#frac_above\t{np.mean(d>0):.6f}\t\t\n")
    fh.write(f"#null_mean\t{null.mean():.6f}\t\t\n#p_perm\t{pperm:.6g}\t\t\n#nperm\t{NPERM}\t\t\n")

print("\n===== GEOMETRY =====")
GB = pd.read_csv(PC_GENES, sep="\t", header=None,
                 names=["chrom", "start", "end", "gene", "score", "strand"])
GB = GB.drop_duplicates("gene").set_index("gene")
tss = np.full(len(D), np.nan)
have = D.gene.isin(GB.index).to_numpy()
gsel = GB.loc[D.gene[have]]
tss_pos = np.where(gsel.strand.to_numpy() == "+", gsel.start.to_numpy(), gsel.end.to_numpy())
pos = D.pos.to_numpy()
tss[have] = np.abs(pos[have] - tss_pos)

canpos = D[D.is_canonical].set_index("gene").pos
cp = D.gene.map(canpos).to_numpy(float)
sgn = np.where(D.strand.to_numpy() == "+", 1.0, -1.0)
dcan = (pos - cp) * sgn

if os.path.exists(EXONS):
    EX = pd.read_csv(EXONS, sep="\t", header=None,
                     names=["chrom", "start", "end", "strand", "gene", "exon"])
    EX = EX[EX.gene.isin(set(D.gene))]
    intr = defaultdict(list)
    for g, sub in EX.groupby("gene", sort=False):
        sub = sub.sort_values("start")
        st = sub.start.to_numpy(); en = sub.end.to_numpy(); sd = sub.strand.iloc[0]
        if len(st) < 2: continue
        iv = [(en[k], st[k + 1] - 1) for k in range(len(st) - 1) if st[k + 1] - 1 > en[k]]
        if sd == "-": iv = iv[::-1]
        intr[g] = iv
    ilen = np.full(len(D), np.nan); irank = np.full(len(D), np.nan)
    gn = D.gene.to_numpy()
    for i in range(len(D)):
        iv = intr.get(gn[i])
        if not iv: continue
        p = pos[i]
        for r, (a, b) in enumerate(iv, 1):
            if a <= p <= b:
                ilen[i] = b - a + 1; irank[i] = r; break
    print(f"  introns built for {len(intr):,} genes from the Ensembl canonical transcript")
else:
    ilen = np.full(len(D), np.nan); irank = np.full(len(D), np.nan)
    print(f"  {EXONS} missing, host intron skipped")

grows = []
for nm, v in [("distance to TSS (nt)", tss), ("distance to canonical (nt, signed)", dcan),
              ("host intron length (nt)", ilen), ("host intron rank from TSS", irank)]:
    print(f"\n  --- {nm} ---")
    grows.append(compare(nm, v, U1, NH, "U1 AMO hit", "not hit"))
    grows.append(compare(nm, v, CT, FP & ~D.hit_oh_camo_50.to_numpy(), "control AMO hit", "not hit"))
ok = np.isfinite(irank)
for glab, gm in [("U1 AMO hit", U1), ("control AMO hit", CT), ("not hit", NH)]:
    m = gm & ok
    f12 = float(np.mean(irank[m] <= 2)) if m.sum() else np.nan
    print(f"  in the first two introns: {glab:<16}{f12:.4f}  ({int((irank[m] <= 2).sum()):,} "
          f"of {int(m.sum()):,} with a host intron)")
    grows.append(dict(quantity="fraction in the first two introns", group_a=glab, group_b="",
                      n_a=int(m.sum()), median_a=f12))
pd.DataFrame(grows).to_csv("u1_hit_geometry.tsv", sep="\t", index=False)
print("\nwrote u1_hit_stats.tsv, u1_hit_dist.tsv, u1_hit_paired.tsv, u1_hit_geometry.tsv")
log("DONE_U1_HIT_CHARACTER")
