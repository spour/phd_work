#!/usr/bin/env python3
# accounting for the external overlap, and the peak table for the corrected rerun
import sys
import numpy as np
import pandas as pd

def log(*a): print(*a, file=sys.stderr, flush=True)

PC_GENES = "../train/data/pc_genes.bed"
BW = {"oh_camo":  "external/geo/GSE103252/GSE103252_cAMO_15nmol_odT_selection.polyA_sites.bw",
      "oh_u1amo": "external/geo/GSE103252/GSE103252_U1AMO_15nmol_odT_selection.polyA_sites.bw"}
TOL = 50
DEDUP = 25

# the set
T = pd.read_parquet("cpa_peaks_final.parquet", engine="fastparquet")
A = T[T.in_analysis_set]
n_all, n_can, n_com, n_gene = len(A), int(A.is_canonical.sum()), int((~A.is_canonical).sum()), A.gene.nunique()
print("===== THE SET =====")
print(f"in_analysis_set  {n_all:>7,} peaks   expected 48,394   {'OK' if n_all == 48394 else 'MISMATCH'}")
print(f"  canonical      {n_can:>7,}         expected  4,847   {'OK' if n_can == 4847 else 'MISMATCH'}")
print(f"  competitor     {n_com:>7,}         expected 43,547   {'OK' if n_com == 43547 else 'MISMATCH'}")
print(f"  genes          {n_gene:>7,}         expected  4,847   {'OK' if n_gene == 4847 else 'MISMATCH'}")
bad = [n_all != 48394, n_can != 4847, n_com != 43547, n_gene != 4847]
if any(bad):
    sys.exit("STOP: the parquet does not match the stated counts. Nothing was written.")

A[["gene", "chrom", "center", "strand", "is_canonical"]].to_csv(
    "ovl_peaks_analysis.tsv", sep="\t", index=False)
print(f"\nwrote ovl_peaks_analysis.tsv ({n_all:,} rows)")

# competitor accounting, sequential
comp = ~T.is_canonical
def gcount(m): return T.loc[m, "gene"].nunique()
steps = []
cur = comp.copy()
g0 = gcount(cur)
steps.append(dict(step="model false positives, all", peaks_removed=0, genes_emptied=0,
                  peaks_remaining=int(cur.sum()), genes_remaining=g0))
FILT = [("APA-annotated: a PolyASite or PolyA_DB site within 500 nt", T.flag_apa_annotated),
        ("in a reference terminal exon", T.flag_reference_terminal_exon),
        ("at or downstream of the gene's canonical site", T.flag_downstream_of_canonical)]
for lab, f in FILT:
    drop = cur & f
    gb = gcount(cur)
    cur = cur & ~f
    ga = gcount(cur)
    steps.append(dict(step=lab, peaks_removed=int(drop.sum()),
                      genes_touched=T.loc[drop, "gene"].nunique(),
                      genes_emptied=gb - ga,
                      peaks_remaining=int(cur.sum()), genes_remaining=ga))
S = pd.DataFrame(steps)
S.to_csv("ovl_accounting_competitor.tsv", sep="\t", index=False)
print("\n===== COMPETITOR ACCOUNTING (model false positives) =====")
print(f"{'step':<56}{'removed':>9}{'genes hit':>11}{'genes emptied':>15}{'remaining':>11}{'genes':>8}")
print("-" * 110)
for r in steps:
    print(f"{r['step']:<56}{r['peaks_removed']:>9,}{r.get('genes_touched', 0):>11,}"
          f"{r['genes_emptied']:>15,}{r['peaks_remaining']:>11,}{r['genes_remaining']:>8,}")
print(f"\nfinal {int(cur.sum()):,} competitors in {gcount(cur):,} genes"
      f"   {'OK' if int(cur.sum()) == 43547 and gcount(cur) == 4847 else 'MISMATCH'}")
assert int(cur.sum()) == 43547 and gcount(cur) == 4847
assert bool((cur == (comp & T.in_analysis_set)).all())

# external sets
def read_lifted(path):
    rows = []
    with open(path, newline="\n") as fh:
        hdr = fh.readline().replace("\r", "").rstrip("\n").split("\t")
        ic, ip, ist = hdr.index("hg38_chrom"), hdr.index("hg38_pos"), hdr.index("strand")
        for line in fh:
            p = line.replace("\r", "").rstrip("\n").split("\t")
            if len(p) <= max(ic, ip, ist) or p[ist] not in "+-" or not p[ip]: continue
            c = p[ic]
            rows.append((c if c.startswith("chr") else "chr" + c, int(p[ip]), p[ist]))
    return pd.DataFrame(rows, columns=["chrom", "pos", "strand"])

def read_plain(path):
    d = pd.read_csv(path, sep="\t", low_memory=False)
    c = d["hg38_chrom"].astype(str)
    return pd.DataFrame({"chrom": np.where(c.str.startswith("chr"), c, "chr" + c),
                         "pos": d["hg38_pos"].astype(int), "strand": d["strand"].astype(str)})

def raw_rows(path):
    with open(path, newline="\n") as fh:
        return sum(1 for _ in fh) - 1

def bw_intervals(path, merge_within=25):
    import pyBigWig
    bw = pyBigWig.open(path); n = 0; blocks = 0
    for c in bw.chroms():
        try: iv = bw.intervals(c)
        except RuntimeError: continue
        if not iv: continue
        n += len(iv)
        end = iv[0][1]; blocks += 1
        for s_, e_, _ in iv[1:]:
            if s_ - end <= merge_within: end = max(end, e_)
            else: blocks += 1; end = e_
    bw.close()
    return n, blocks

im = read_lifted("external/lifted/immune_hg38.tsv")
cl = read_lifted("external/lifted/cll_hg38.tsv")
u = pd.concat([im, cl], ignore_index=True).sort_values(["chrom", "strand", "pos"])
out, last = [], None
for (c, s), sub in u.groupby(["chrom", "strand"], sort=False):
    last = None
    for p in sub["pos"].to_numpy():
        if last is None or p - last > DEDUP:
            out.append((c, p, s)); last = p
atlas = pd.DataFrame(out, columns=["chrom", "pos", "strand"])
atlas.to_csv("ovl_atlas3seq.tsv", sep="\t", index=False)
oh = {k: read_plain(f"external/peaks/oh2017_{v}_hg38.tsv")
      for k, v in [("oh_camo", "cAMO"), ("oh_u1amo", "U1AMO")]}

# contested-gene membership
GB = pd.read_csv(PC_GENES, sep="\t", header=None,
                 names=["chrom", "start", "end", "gene", "score", "strand"])
SHORT = set(T.gene.unique()); ANA = set(A.gene.unique())
def in_genes(D, keep):
    G = GB[GB.gene.isin(keep)]
    hit = np.zeros(len(D), bool)
    for (c, s), sub in G.groupby(["chrom", "strand"], sort=False):
        m = (D.chrom.to_numpy() == c) & (D.strand.to_numpy() == s)
        if not m.any(): continue
        st = np.sort(sub.start.to_numpy()); en = sub.end.to_numpy()[np.argsort(sub.start.to_numpy())]
        mx = np.maximum.accumulate(en)
        p = D.pos.to_numpy()[m]
        j = np.searchsorted(st, p, side="right") - 1
        ok = (j >= 0) & (mx[np.clip(j, 0, len(mx) - 1)] > p)
        idx = np.where(m)[0]
        hit[idx[ok]] = True
    return hit

def uniq(D): return D.drop_duplicates(["chrom", "pos"])

rows = []
RAWDESC = {
    "atlas3seq": ("GSE111793 + GSE111310 IPA tables", None),
    "oh_camo": ("GSE103252 cAMO polyA_sites.bw", BW["oh_camo"]),
    "oh_u1amo": ("GSE103252 U1AMO polyA_sites.bw", BW["oh_u1amo"]),
}
print("\n===== EXTERNAL DATASET ACCOUNTING =====")
hdr = (f"{'dataset':<12}{'raw':>10}{'blocks':>9}{'topN cut':>10}{'lift lost':>11}"
       f"{'dup merged':>12}{'strand dup':>12}{'final':>9}{'in gene':>9}{'in ana':>8}")
print(hdr); print("-" * len(hdr))

raw_atlas = raw_rows("external/lifted/immune_hg38.tsv") + raw_rows("external/lifted/cll_hg38.tsv")
lift_lost = 0
merged = (len(im) + len(cl)) - len(atlas)
g_short = int(in_genes(atlas, SHORT).sum()); g_ana = int(in_genes(atlas, ANA).sum())
print(f"{'atlas3seq':<12}{raw_atlas:>10,}{'-':>9}{'-':>10}{lift_lost:>11,}{merged:>12,}{'-':>12}"
      f"{len(atlas):>9,}{g_short:>9,}{g_ana:>8,}")
rows.append(dict(dataset="atlas3seq", source=RAWDESC["atlas3seq"][0], raw_sites=raw_atlas,
                 merged_blocks=raw_atlas, topn_cut=0, lost_liftover=lift_lost, pct_lost_liftover=0.0,
                 lost_duplicate_merge=merged, lost_strand_duplication=0,
                 final_sites=len(atlas), in_contested_gene=g_short, in_analysis_gene=g_ana))

for k, (src, bwp) in [("oh_camo", RAWDESC["oh_camo"]), ("oh_u1amo", RAWDESC["oh_u1amo"])]:
    nm = "cAMO" if k == "oh_camo" else "U1AMO"
    raw, blocks = bw_intervals(bwp)
    n_hg19 = raw_rows(f"external/peaks/oh2017_{nm}_hg19.tsv")
    n_pos19 = n_hg19 // 2
    D = oh[k]; U = uniq(D)
    lost = n_hg19 - len(D)
    g_short = int(in_genes(U, SHORT).sum()); g_ana = int(in_genes(U, ANA).sum())
    print(f"{k:<12}{raw:>10,}{blocks:>9,}{blocks - n_pos19:>10,}{lost // 2:>11,}{'-':>12}"
          f"{len(D) - len(U):>12,}{len(U):>9,}{g_short:>9,}{g_ana:>8,}")
    rows.append(dict(dataset=k, source=src, raw_sites=raw, merged_blocks=blocks,
                     topn_cut=blocks - n_pos19,
                     lost_liftover=lost // 2, pct_lost_liftover=100 * (lost // 2) / n_pos19,
                     lost_duplicate_merge=0, lost_strand_duplication=len(D) - len(U),
                     final_sites=len(U), in_contested_gene=g_short, in_analysis_gene=g_ana))
    oh[k] = D
E = pd.DataFrame(rows)
E.to_csv("ovl_accounting_external.tsv", sep="\t", index=False)
print("\nraw = intervals in the downloaded bigWig, or rows in the two downloaded IPA tables.")
print("blocks = after merging intervals whose gap is at most 25 nt, which is the peak call.")
print("topN cut = blocks discarded by the existing top-40,000 cut, which lands inside a tie")
print("           block at the signal floor.")
print("strand dup = the same position emitted on both strands; the bigWig is unstranded.")
print("in gene / in ana = final sites inside the body of any contested gene (6,583) and of")
print("           any analysis gene (4,847), same strand, from pc_genes.bed.")

def wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan")
    ph = k / n; d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    hw = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - hw), min(1.0, c + hw)

def nearest(q, t):
    out = np.full(len(q), np.inf)
    idx = {}
    for (c, s), sub in t.groupby(["chrom", "strand"], sort=False):
        idx[(c, s)] = np.sort(sub["pos"].to_numpy())
    for (c, s), sub in q.groupby(["chrom", "strand"], sort=False):
        arr = idx.get((c, s))
        if arr is None or arr.size == 0: continue
        p = sub["pos"].to_numpy()
        i = np.searchsorted(arr, p)
        lo = np.where(i > 0, np.abs(p - arr[np.clip(i - 1, 0, arr.size - 1)]), np.inf)
        hi = np.where(i < arr.size, np.abs(arr[np.clip(i, 0, arr.size - 1)] - p), np.inf)
        out[sub.index.to_numpy()] = np.minimum(lo, hi)
    return out

print("\n===== DATASET-SIDE FRACTION, UNIQUE POSITIONS =====")
print("A site counts once. For the unstranded Oh sets it counts as a hit if a competitor on")
print("either strand is within the tolerance.\n")
SETS = {"atlas3seq": atlas, "oh_camo": oh["oh_camo"], "oh_u1amo": oh["oh_u1amo"]}
prow = []
for setname, mask in [("unfiltered", comp), ("analysis", comp & T.in_analysis_set)]:
    C = T.loc[mask, ["chrom", "center", "strand"]].rename(columns={"center": "pos"}).reset_index(drop=True)
    for dsn, D in SETS.items():
        d = nearest(D.reset_index(drop=True), C)
        H = D.reset_index(drop=True).assign(hit=d <= TOL)
        g = H.groupby(["chrom", "pos"])["hit"].any()
        k, n = int(g.sum()), len(g)
        lo, hi = wilson(k, n)
        print(f"  {setname:<11}{dsn:<11}{k:>7,} of {n:>7,}   {100*k/n:>6.2f}%   "
              f"[{100*lo:.2f}, {100*hi:.2f}]")
        prow.append(dict(set=setname, dataset=dsn, tolerance=TOL, n_sites=n, n_hit=k,
                         pct_of_dataset=100 * k / n, ci_lo=100 * lo, ci_hi=100 * hi,
                         n_competitors=int(mask.sum())))
pd.DataFrame(prow).to_csv("ovl_panel_a_sites.tsv", sep="\t", index=False)
print("\nwrote ovl_accounting_competitor.tsv, ovl_accounting_external.tsv, "
      "ovl_peaks_analysis.tsv, ovl_panel_a_sites.tsv")
log("DONE_OVERLAP_ACCOUNTING")
