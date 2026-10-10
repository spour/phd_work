#!/usr/bin/env python3
# gc composition around the pa: canonical vs competitor, heatmap rows + mean profile
import csv, sys, numpy as np
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)

FASTA = "../train/data/hg38.fa"
UP, DN, WIN, STEP, NSAMP, NBOOT = 2000, 500, 50, 10, 500, 1000
rng = np.random.default_rng(42)

class Faidx:
    _T = str.maketrans("ACGTNacgtn", "TGCANTGCAN")
    def __init__(s, fa):
        s.fh = open(fa, "rb"); s.idx = {}
        for L in open(fa + ".fai"):
            p = L.split(); s.idx[p[0]] = (int(p[1]), int(p[2]), int(p[3]), int(p[4]))
    def fetch(s, c, a, b, st="+"):
        c = c if c in s.idx else c.lstrip("chr")
        if c not in s.idx: return None
        Ln, off, lb, lw = s.idx[c]
        if a < 0 or b > Ln: return None
        bp = lambda p: off + (p // lb) * lw + (p % lb)
        s.fh.seek(bp(a)); raw = s.fh.read(bp(b) - bp(a))
        q = raw.replace(b"\n", b"").replace(b"\r", b"")[:b - a].decode("ascii").upper()
        return q.translate(s._T)[::-1] if st == "-" else q

R = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t")); NP = len(R)
gene = np.array([r["gene"] for r in R]); isc = np.array([int(float(r["is_canonical"])) for r in R])
chrom = [r["chrom"] for r in R]; cen = [int(r["center"]) for r in R]; strd = [r["strand"] for r in R]
dpa = np.array([float(r["d_anchor_to_pa"]) if r["d_anchor_to_pa"] not in ("", "nan") else np.nan for r in R])

gmap = defaultdict(list)
for i in range(NP): gmap[gene[i]].append(i)
pairs = []
for g, ix in gmap.items():
    ix = np.array(ix)
    c = ix[(isc[ix] == 1) & np.isfinite(dpa[ix])]; k = ix[(isc[ix] == 0) & np.isfinite(dpa[ix])]
    if len(c) == 1 and len(k) >= 1: pairs.append((g, int(c[0]), k))
pairs.sort(key=lambda t: t[0])
sel = rng.choice(len(pairs), size=NSAMP, replace=False)
PAIR = [(pairs[j][0], pairs[j][1], int(rng.choice(pairs[j][2]))) for j in sorted(sel)]
log(f"gene-paired sample: {len(PAIR)} genes (of {len(pairs)} eligible), one canonical + one competitor each")

G = Faidx(FASTA)
L = UP + DN + 1
starts = np.arange(0, L - WIN + 1, STEP)
POS = starts + WIN // 2 - UP
def gcrow(i):
    a = cen[i] - UP if strd[i] == "+" else cen[i] - DN
    b = cen[i] + DN + 1 if strd[i] == "+" else cen[i] + UP + 1
    s = G.fetch(chrom[i], a, b, strd[i])
    if s is None or len(s) != L: return None
    v = np.frombuffer(s.encode("ascii"), np.uint8)
    isgc = ((v == 71) | (v == 67)).astype(np.float64)
    isn = (v == 78)
    cs = np.r_[0.0, np.cumsum(isgc)]; cn = np.r_[0.0, np.cumsum(isn.astype(float))]
    tot = WIN - (cn[starts + WIN] - cn[starts])
    gc = (cs[starts + WIN] - cs[starts]) / np.where(tot > 0, tot, np.nan)
    return gc

CAN, COMP, KEEPG, DC, DK = [], [], [], [], []
for g, ci, ki in PAIR:
    a = gcrow(ci); b = gcrow(ki)
    if a is None or b is None: continue
    CAN.append(a); COMP.append(b); KEEPG.append(g); DC.append(dpa[ci]); DK.append(dpa[ki])
A = np.array(CAN); B = np.array(COMP); DC = np.array(DC); DK = np.array(DK); NG = len(A)
log(f"fetched {NG} pairs, {A.shape[1]} columns ({POS[0]} .. {POS[-1]})")

# per-position stats, bootstrap over genes (the pair is the resampling unit)
mA = np.nanmean(A, 0); mB = np.nanmean(B, 0); d = mA - mB
bi = rng.integers(0, NG, size=(NBOOT, NG))
bA = np.array([np.nanmean(A[bi[t]], 0) for t in range(NBOOT)])
bB = np.array([np.nanmean(B[bi[t]], 0) for t in range(NBOOT)])
qA = np.percentile(bA, [2.5, 97.5], axis=0); qB = np.percentile(bB, [2.5, 97.5], axis=0)
qD = np.percentile(bA - bB, [2.5, 97.5], axis=0)

def auroc_col(a, b):
    n1, n2 = len(a), len(b); allv = np.r_[a, b]
    o = np.argsort(allv, kind="mergesort"); sv = allv[o]; N = n1 + n2; rk = np.empty(N); i = 0
    while i < N:
        j = i
        while j + 1 < N and sv[j + 1] == sv[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    rr = np.empty(N); rr[o] = rk
    return (rr[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n2)
AU = np.array([auroc_col(A[:, j], B[:, j]) for j in range(A.shape[1])])

with open("gc_profile.tsv", "w") as fh:
    fh.write("position\tmean_canonical\tcan_lo\tcan_hi\tmean_competitor\tcomp_lo\tcomp_hi\t"
             "diff\tdiff_lo\tdiff_hi\tauroc\n")
    for j, p in enumerate(POS):
        fh.write(f"{p}\t{mA[j]:.6f}\t{qA[0,j]:.6f}\t{qA[1,j]:.6f}\t{mB[j]:.6f}\t{qB[0,j]:.6f}\t{qB[1,j]:.6f}\t"
                 f"{d[j]:.6f}\t{qD[0,j]:.6f}\t{qD[1,j]:.6f}\t{AU[j]:.5f}\n")

# raster rows, both blocks sorted by d_anchor_to_pa
oA = np.argsort(DC, kind="mergesort"); oB = np.argsort(DK, kind="mergesort")
with open("gc_heatmap_matrix.tsv", "w") as fh:
    fh.write("group\trow\tdpa\tposition\tgc\n")
    for grp, M, o, dv in (("canonical", A, oA, DC), ("competitor", B, oB, DK)):
        for r, i in enumerate(o):
            for j, p in enumerate(POS):
                if np.isfinite(M[i, j]): fh.write(f"{grp}\t{r+1}\t{dv[i]:.0f}\t{p}\t{M[i,j]:.4f}\n")

ov = float(np.nanmean(np.r_[A, B]))
jmax = int(np.nanargmax(np.abs(d)))
m50 = (POS >= -50) & (POS <= 0)
jau = int(np.nanargmax(np.abs(AU - 0.5)))
print(f"rows: {NG} canonical + {NG} competitor, gene-paired, seed 42; columns {len(POS)} ({POS[0]}..{POS[-1]})")
print(f"overall mean GC {ov:.4f}")
print(f"max |canonical - competitor| GC = {abs(d[jmax]):.5f} at position {POS[jmax]:+d} "
      f"(canonical {mA[jmax]:.4f} vs competitor {mB[jmax]:.4f}, diff CI [{qD[0,jmax]:+.5f},{qD[1,jmax]:+.5f}], "
      f"AUROC {AU[jmax]:.3f})")
print(f"-50..0 band: mean diff {d[m50].mean():+.5f}, mean AUROC {AU[m50].mean():.3f}, "
      f"best AUROC in band {AU[m50].max():.3f}")
print(f"best per-position AUROC overall {AU[jau]:.3f} at {POS[jau]:+d}")
log("wrote gc_profile.tsv / gc_heatmap_matrix.tsv  DONE_GC")
