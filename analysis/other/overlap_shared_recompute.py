#!/usr/bin/env python3
# the overlap panels, recomputed on the genes every dataset covers
import sys, csv
import numpy as np

def log(*a): print(*a, file=sys.stderr, flush=True)
FA = "/scratch/spour98/jobs/hg38.fa"
TOL, UP, SHIFT = 50, 60, 5000

S = {}
for r in csv.DictReader(open("overlap_scores_sites.tsv"), delimiter="\t"):
    if r["in_shared"] != "1": continue
    S.setdefault(r["set"], []).append((r["chrom"], int(r["pos"]), r["strand"], r["gene"]))
SETS = sorted(S)
for k in SETS: log(f"[set] {k:<12} {len(S[k]):>6,} sites in shared genes")

IDX = {}; IDXC = {}
for k in SETS:
    dg, dc = {}, {}
    for c, p, _, g in S[k]: dg.setdefault(g, []).append(p); dc.setdefault(c, []).append(p)
    IDX[k] = {g: np.array(sorted(v)) for g, v in dg.items()}
    IDXC[k] = {c: np.array(sorted(v)) for c, v in dc.items()}

def nearest(a, b, by_gene=True):
    out = []
    for c, p, _, g in S[a]:
        q = (IDX[b].get(g) if by_gene else IDXC[b].get(c))
        if q is None or not len(q): continue
        i = np.searchsorted(q, p)
        cand = [q[j] for j in (i - 1, i) if 0 <= j < len(q)]
        out.append(min(abs(p - x) for x in cand))
    return np.array(out, float)

with open("shared_distances.tsv", "w") as fh, open("shared_containment.tsv", "w") as fc:
    fh.write("set_a\tset_b\tdist\n")
    fc.write("set_a\tset_b\tcontainment\tn_a\tn_hit\tcontainment_chrom_only\n")
    DELTA = []
    log(f"\n{'containment |A within 50 nt of B| / |A|':<46}")
    log("  " + " ".join(f"{b[:9]:>10}" for b in SETS))
    for a in SETS:
        row = []
        for b in SETS:
            if a == b: row.append(np.nan); continue
            d = nearest(a, b, True); dc = nearest(a, b, False)
            for x in d: fh.write(f"{a}\t{b}\t{x:.0f}\n")
            hit = int((d <= TOL).sum()); cont = hit / max(len(d), 1)
            hc = int((dc <= TOL).sum()) / max(len(dc), 1)
            fc.write(f"{a}\t{b}\t{cont:.6f}\t{len(d)}\t{hit}\t{hc:.6f}\n"); row.append(cont)
            DELTA.append((cont, hc, float(np.median(d)), float(np.median(dc))))
        log(f"  {a:<12} " + " ".join("     ." if np.isnan(v) else f"{v:>10.3f}" for v in row))
D2 = np.array(DELTA)
log(f"\n[matching] within gene vs chromosome only, over {len(D2)} ordered pairs")
log(f"  containment     mean {D2[:,0].mean():.4f} within gene, {D2[:,1].mean():.4f} by chromosome; "
    f"chromosome-only is higher in {(D2[:,1] > D2[:,0] + 1e-9).sum()} pairs, "
    f"largest inflation {np.max(D2[:,1]-D2[:,0]):+.4f}")
log(f"  median distance {np.median(D2[:,2]):>10,.0f} nt within gene, "
    f"{np.median(D2[:,3]):,.0f} nt by chromosome")
log("[out] shared_distances.tsv shared_containment.tsv")

# hexamer
FAI = {}
for line in open(FA + ".fai"):
    n, ln, off, lb, wb = line.split()[:5]
    FAI[n] = (int(ln), int(off), int(lb), int(wb))
fh = open(FA, "rb")
def fetch(c, s, e):
    if c not in FAI: return None
    ln, off, lb, wb = FAI[c]
    s = max(0, s); e = min(ln, e)
    if e <= s: return None
    fh.seek(off + s // lb * wb + s % lb)
    raw = fh.read((e - s) + (e - s) // lb + 2).decode()
    return raw.replace("\n", "")[:e - s].upper()
def rc(s): return s.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]
def has_pas(c, p, strand, shift=0):
    q = p + shift
    seq = fetch(c, q - UP, q + 1) if strand != "-" else fetch(c, q, q + UP + 1)
    if not seq: return None
    if strand == "-": seq = rc(seq)
    return int("AATAAA" in seq or "ATTAAA" in seq)

log(f"\n{'set':<12} {'n':>6} {'observed':>9} {'+5 kb':>8} {'fold':>7}")
with open("shared_hexamer.tsv", "w") as fx:
    fx.write("set\tn\tpct_obs\tpct_rand\tfold\n")
    for k in SETS:
        o = [has_pas(c, p, s) for c, p, s, _g in S[k]]
        r = [has_pas(c, p, s, SHIFT) for c, p, s, _g in S[k]]
        o = [x for x in o if x is not None]; r = [x for x in r if x is not None]
        po, pr = 100 * np.mean(o), 100 * np.mean(r)
        fx.write(f"{k}\t{len(o)}\t{po:.4f}\t{pr:.4f}\t{po/max(pr,1e-9):.4f}\n")
        log(f"{k:<12} {len(o):>6,} {po:>8.1f}% {pr:>7.1f}% {po/max(pr,1e-9):>6.1f}x")
log("[out] shared_hexamer.tsv")
