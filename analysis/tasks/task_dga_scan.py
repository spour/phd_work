#!/usr/bin/env python3
# d_ga follow-up, fasta-dependent items
import sys, csv
import numpy as np
from collections import defaultdict
sys.path.insert(0, "scripts")
from extract_splicing_features import init_maxentscan
def log(*a): print(*a, file=sys.stderr, flush=True)
FASTA = "../train/data/hg38.fa"; POS = "fig3_new_positional_features.tsv"
LO, HI, BW = -500, 100, 10; NB = (HI - LO) // BW; PAD = 25; STRONG3 = 6.5
R = list(csv.DictReader(open(POS), delimiter="\t")); NP = len(R)
gene = np.array([r["gene"] for r in R]); isc = np.array([int(float(r["is_canonical"])) for r in R])
chrom = [r["chrom"] for r in R]; center = [int(r["center"]) for r in R]; strand = [r["strand"] for r in R]
F1 = np.load("hypothesis_test_features.npy")
dGA_real = F1[:, 24 + 8] - F1[:, 24 + 2]; acc = F1[:, 66]
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
G = Faidx(FASTA); _, score3, m5, m3 = init_maxentscan()
from maxentpy import maxent

# real-peak pass: ga/ag bins [-500,+100] + me3 hit density [-500,0]
GA = {0: np.zeros(NB), 1: np.zeros(NB)}; AG = {0: np.zeros(NB), 1: np.zeros(NB)}; TOT = {0: np.zeros(NB), 1: np.zeros(NB)}
me3d = np.full(NP, np.nan)
for i in range(NP):
    ce, st, ch = center[i], strand[i], chrom[i]
    if st == "+": a, b = ce + LO - PAD, ce + HI + PAD
    else:         a, b = ce - HI - PAD, ce - LO + PAD
    seq = G.fetch(ch, a, b, st)
    if not seq: continue
    o = -LO + PAD
    tok = Bm[np.frombuffer(seq.encode("ascii"), np.uint8)]
    t0 = tok[:-1]; t1 = tok[1:]; pos = np.arange(len(t0)) - o
    m = (pos >= LO) & (pos < HI) & (t0 >= 0) & (t1 >= 0)
    binv = ((pos[m] - LO) // BW).astype(int); cl = isc[i]
    TOT[cl] += np.bincount(binv, minlength=NB)
    GA[cl] += np.bincount(binv[(t0[m] == 2) & (t1[m] == 0)], minlength=NB)
    AG[cl] += np.bincount(binv[(t0[m] == 0) & (t1[m] == 2)], minlength=NB)
    hits = 0
    slo = max(o - 500, 0); shi = min(o, len(seq) - 23)
    for s in range(slo - 18, shi - 18 + 1):
        if s < 0 or s + 23 > len(seq): continue
        if tok[s + 18] == 0 and tok[s + 19] == 2:
            agpos = (s + 18) - o
            if not (LO <= agpos < 0): continue
            km = seq[s:s + 23]
            if "N" in km: continue
            try:
                if maxent.score3(km, matrix=m3) >= STRONG3: hits += 1
            except Exception: pass
    me3d[i] = hits / 0.5
    if (i + 1) % 10000 == 0: log(f"  real {i+1}/{NP}")
np.save("dga_me3_density.npy", me3d)

fGA = {c: GA[c] / TOT[c] for c in (0, 1)}; fAG = {c: AG[c] / TOT[c] for c in (0, 1)}
dcan = fGA[1] - fAG[1]; dcomp = fGA[0] - fAG[0]; diff = dcan - dcomp
centers = LO + BW * np.arange(NB) + BW // 2

# decoy pass: d_ga at matched intronic decoys
D = np.load("decoy_set.npz"); dpos = D["pos"]; dgene = D["gene"]; dcompN = D["comp"]
Z = np.load("u1_window_cache.npz"); gi = Z["gene_idx"]; isc_z = Z["isc"]
o2 = np.argsort(gi, kind="stable"); gs = gi[o2]; cut = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1, len(gs)]
gcanon = []; gmemb = []
for j in range(len(cut) - 1):
    idx = o2[cut[j]:cut[j + 1]]; c = idx[isc_z[idx] == 1]; k = idx[isc_z[idx] == 0]
    if len(c) == 1 and len(k) >= 1: gcanon.append(int(c[0])); gmemb.append(idx)
NG = len(gcanon)
def dga_at(pos, ch, st):
    a, b = (pos - 500, pos) if st == "+" else (pos, pos + 500)
    seq = G.fetch(ch, a, b, st)
    if not seq or len(seq) < 3: return np.nan
    tok = Bm[np.frombuffer(seq.encode("ascii"), np.uint8)]; t0 = tok[:-1]; t1 = tok[1:]
    ok = (t0 >= 0) & (t1 >= 0); n = ok.sum()
    if n < 2: return np.nan
    nGA = np.sum((t0[ok] == 2) & (t1[ok] == 0)); nAG = np.sum((t0[ok] == 0) & (t1[ok] == 2))
    return (nGA - nAG) / n
d_dga = np.full(len(dpos), np.nan)
for k in range(len(dpos)):
    g = dgene[k]; ch = chrom[gcanon[g]]; st = strand[gcanon[g]]
    d_dga[k] = dga_at(int(dpos[k]), ch, st)
    if (k + 1) % 30000 == 0: log(f"  decoy {k+1}/{len(dpos)}")
by_dec = [[] for _ in range(NG)]; by_cmp = [[] for _ in range(NG)]
for k in range(len(dpos)):
    g = dgene[k]; by_dec[g].append(d_dga[k]); by_cmp[g].append(dcompN[k])
rng = np.random.default_rng(42)
half = np.full(NG, np.nan)
for g in range(NG):
    dv = np.array(by_dec[g], float); cm = np.array(by_cmp[g]); ok = np.isfinite(dv)
    if ok.sum() == 0: continue
    cv = dGA_real[cm[ok]]; dvv = dv[ok]; okk = np.isfinite(cv)
    if okk.sum() == 0: continue
    dd = cv[okk] - dvv[okk]; w = (dd > 0).sum(); t = (dd == 0).sum(); half[g] = (w + 0.5 * t) / okk.sum()
hv = half[np.isfinite(half)]
bs = hv[rng.integers(0, len(hv), size=(4000, len(hv)))].mean(1)
dec_obs, dec_lo, dec_hi = hv.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

# report
print("\n==== ITEM 1: LOCALISE d_GA = freq(GA)-freq(AG), 10-nt bins [-500,+100], canonical vs competitor ====")
absd = np.abs(diff); top5 = np.argsort(-absd)[:5]
print("top 5 bins by |canon-comp diff|:")
print(f"{'bin(nt)':>12s} {'canon d':>9s} {'comp d':>9s} {'canon-comp':>11s}")
for b in sorted(top5, key=lambda x: -absd[x]):
    lo_b = LO + BW * b
    print(f"[{lo_b:>4d},{lo_b+BW:>4d}) {dcan[b]:>+9.4f} {dcomp[b]:>+9.4f} {diff[b]:>+11.4f}")
in50 = (centers >= -50) & (centers < 0); frac50 = absd[in50].sum() / absd.sum()
print(f"fraction of total |canon-comp diff| in the [-50,0] region: {frac50*100:.1f}% "
      f"(that region is {in50.sum()}/{NB} bins = {in50.sum()/NB*100:.1f}% of the window)")
print("  -> diffuse if ~proportional to bin count; concentrated if >> that.")

print("\n==== ITEM 3: DECOY (competitor vs matched intronic decoy on d_GA) ====")
print(f"win-rate P(competitor d_GA > matched decoy d_GA), ties=0.5, gene-averaged over {len(hv)} genes:")
print(f"  obs {dec_obs:.4f}  excess {dec_obs-0.5:+.4f}  boot 95% CI [{dec_lo:.4f},{dec_hi:.4f}]  (null 0.5)")
print("  ~0.5 -> competitors indistinguishable from intronic decoys -> d_GA is compartment, not a CPA-site property.")

print("\n==== ITEM 4: is d_GA acceptor-like sequence content? ====")
ok = np.isfinite(dGA_real) & np.isfinite(acc); r_acc = np.corrcoef(dGA_real[ok], acc[ok])[0, 1]
ok2 = np.isfinite(dGA_real) & np.isfinite(me3d); r_me3 = np.corrcoef(dGA_real[ok2], me3d[ok2])[0, 1]
print(f"corr(d_GA, anchor acceptor MaxEnt) = {r_acc:+.3f} ; corr(d_GA, MaxEnt 3'ss hit density [-500,0]) = {r_me3:+.3f}")
