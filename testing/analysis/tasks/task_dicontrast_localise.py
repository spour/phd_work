#!/usr/bin/env python3
# d_ca is real (competitor direction on [-500,0] average)
import sys, csv
import numpy as np
def log(*a): print(*a, file=sys.stderr, flush=True)
FASTA = "../train/data/hg38.fa"; POS = "fig3_new_positional_features.tsv"
LO, HI, BW = -500, 100, 10
NB = (HI - LO) // BW
PAD = 3
R = list(csv.DictReader(open(POS), delimiter="\t")); NP = len(R)
isc = np.array([int(float(r["is_canonical"])) for r in R])
chrom = [r["chrom"] for r in R]; center = [int(r["center"]) for r in R]; strand = [r["strand"] for r in R]
Bm = np.full(256, -1, np.int64)
for ch, v in zip(b"ACGT", (0, 1, 2, 3)): Bm[ch] = v
class Faidx:
    _T = str.maketrans("ACGTNacgtn", "TGCANTGCAN")
    def __init__(s, fa):
        s.fh = open(fa, "rb"); s.idx = {}
        for L in open(fa + ".fai"):
            p = L.split(); s.idx[p[0]] = (int(p[1]), int(p[2]), int(p[3]), int(p[4]))
    def fetch(s, c, a, b, st="+"):
        c = c if c in s.idx else c.lstrip("chr")
        if c not in s.idx: return None
        Ln, off, lb, lw = s.idx[c]; a = max(0, a); b = min(b, Ln)
        if b <= a: return None
        bp = lambda p: off + (p // lb) * lw + (p % lb)
        s.fh.seek(bp(a)); raw = s.fh.read(bp(b) - bp(a))
        q = raw.replace(b"\n", b"").replace(b"\r", b"")[:b - a].decode("ascii").upper()
        return q.translate(s._T)[::-1] if st == "-" else q
G = Faidx(FASTA)
CA = {0: np.zeros(NB), 1: np.zeros(NB)}; AC = {0: np.zeros(NB), 1: np.zeros(NB)}; TOT = {0: np.zeros(NB), 1: np.zeros(NB)}
for i in range(NP):
    ce, st, ch = center[i], strand[i], chrom[i]
    if st == "+": a, b = ce + LO - PAD, ce + HI + PAD
    else:         a, b = ce - HI - PAD, ce - LO + PAD
    seq = G.fetch(ch, a, b, st)
    if not seq: continue
    o = -LO + PAD
    tok = Bm[np.frombuffer(seq.encode("ascii"), np.uint8)]
    t0 = tok[:-1]; t1 = tok[1:]
    pos = np.arange(len(t0)) - o
    m = (pos >= LO) & (pos < HI) & (t0 >= 0) & (t1 >= 0)
    binv = ((pos[m] - LO) // BW).astype(int)
    ca = (t0[m] == 1) & (t1[m] == 0)
    ac = (t0[m] == 0) & (t1[m] == 1)
    cl = isc[i]
    TOT[cl] += np.bincount(binv, minlength=NB)
    CA[cl] += np.bincount(binv[ca], minlength=NB)
    AC[cl] += np.bincount(binv[ac], minlength=NB)
    if (i + 1) % 10000 == 0: log(f"  {i+1}/{NP}")

centers = LO + BW * np.arange(NB) + BW // 2
fCA = {c: CA[c] / TOT[c] for c in (0, 1)}; fAC = {c: AC[c] / TOT[c] for c in (0, 1)}
dcan = fCA[1] - fAC[1]; dcomp = fCA[0] - fAC[0]; diff = dcan - dcomp

print("\n==== LOCALISATION of d_CA = freq(CA) - freq(AC), per 10-nt bin, [-500,+100] ====")
print("d = CA-AC per class; diff = canonical(d) - competitor(d). Spike at bin [-10,0] = mechanism; flat = composition.")
print(f"{'bin(nt)':>10s} {'canon d':>9s} {'comp d':>9s} {'canon-comp':>11s}")
for b in range(NB):
    lo_b = LO + BW * b; mark = "  <== cleavage-site bin [-10,0]" if lo_b == -10 else ""
    print(f"[{lo_b:>4d},{lo_b+BW:>4d}) {dcan[b]:>+9.4f} {dcomp[b]:>+9.4f} {diff[b]:>+11.4f}{mark}")

cleav = np.where(centers == -5)[0][0]
absdiff = np.abs(diff)
frac_at_cleav = absdiff[cleav] / absdiff.sum()
rank = int(np.sum(absdiff > absdiff[cleav])) + 1
print(f"\ncleavage-site bin [-10,0): canon-comp d = {diff[cleav]:+.4f} ; |diff| rank {rank}/{NB} "
      f"({frac_at_cleav*100:.1f}% of total |diff|; uniform would be {100/NB:.1f}%).")
mA = np.mean(np.abs(diff[(centers < -50)]))
print(f"mean |canon-comp d| upstream of -50 nt = {mA:.4f} vs cleavage-site bin |{diff[cleav]:.4f}|")
print("verdict:", "SPIKE at cleavage site (mechanism)" if (rank <= 3 and absdiff[cleav] > 3*mA)
      else "DIFFUSE across the window (composition, not a cleavage-site mechanism)")
