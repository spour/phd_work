#!/usr/bin/env python3
# task b: threshold-robustness of the u1 suppression signal
import sys, csv, math
import numpy as np

FASTA = "../train/data/hg38.fa"
U1NPY = "../train/splice_train/data/u1_lookup.npy"
META  = "fig3_ucube_meta.tsv"
WIN   = 500
LADDER = [4.0, 5.25, 6.5, 7.0, 8.0]
PRIMARY = 5.25
SEED = 42
def log(*a): print(*a, file=sys.stderr, flush=True)
rng = np.random.default_rng(SEED)

# u1 score array + per-9mer feature vectors for the analytic dinuc expectation
B = np.full(256, -1, np.int64)
for ch, v in zip(b"ACGT", (0, 1, 2, 3)): B[ch] = v
POW = (4 ** np.arange(8, -1, -1)).astype(np.int64)
arr = np.load(U1NPY, allow_pickle=True)
d = arr.item() if arr.dtype == object else dict(arr)
kbytes = np.frombuffer("".join(d.keys()).encode("ascii"), np.uint8).reshape(-1, 9)
ktok = B[kbytes]
kidx = ktok @ POW
SCORE = np.empty(4 ** 9, np.float64); SCORE[kidx] = -np.fromiter(d.values(), np.float64, len(d))
log(f"[u1] {len(d):,} 9-mers ; score mean {SCORE.mean():.3f}")

order = np.argsort(kidx)
tok_by_code = ktok[order]
init_ind = np.zeros((4 ** 9, 4)); init_ind[np.arange(4 ** 9), tok_by_code[:, 0]] = 1.0
steps = tok_by_code[:, :-1] * 4 + tok_by_code[:, 1:]
dinuc_cnt = np.zeros((4 ** 9, 16))
for i in range(16):
    dinuc_cnt[:, i] = (steps == i).sum(1)
FEAT = np.hstack([init_ind, dinuc_cnt])

def score_window(tok):
    if tok.shape[0] < 9: return np.empty(0)
    sw = np.lib.stride_tricks.sliding_window_view(tok, 9)
    good = (sw >= 0).all(1)
    out = np.full(sw.shape[0], np.nan)
    out[good] = SCORE[sw[good] @ POW]
    return out

# meta + genome
class Faidx:
    _T = str.maketrans("ACGTNacgtn", "TGCANTGCAN")
    def __init__(self, fa):
        self.fh = open(fa, "rb"); self.idx = {}
        for line in open(fa + ".fai"):
            p = line.split(); self.idx[p[0]] = (int(p[1]), int(p[2]), int(p[3]), int(p[4]))
    def fetch(self, chrom, start, end, strand="+"):
        c = chrom if chrom in self.idx else chrom.lstrip("chr")
        if c not in self.idx: return None
        L, off, lb, lw = self.idx[c]; start = max(0, start); end = min(end, L)
        if end <= start: return None
        bp = lambda p: off + (p // lb) * lw + (p % lb)
        self.fh.seek(bp(start)); raw = self.fh.read(bp(end) - bp(start))
        s = raw.replace(b"\n", b"").replace(b"\r", b"")[:end - start].decode("ascii").upper()
        return s.translate(self._T)[::-1] if strand == "-" else s
G = Faidx(FASTA)

gene, isc, chrom, center, strand = [], [], [], [], []
with open(META) as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        gene.append(row["gene"]); isc.append(int(row["is_canonical"]))
        chrom.append(row["chrom"]); center.append(int(row["center"])); strand.append(row["strand"])
gene = np.array(gene); isc = np.array(isc); NP = len(isc)
log(f"[meta] {NP:,} peaks ; {int((isc==1).sum()):,} canonical / {int((isc==0).sum()):,} competitor")

# rescore the pa-500..pa window per peak
counts = {t: np.zeros(NP) for t in LADDER}
npos = np.zeros(NP)
Wlog = np.zeros((NP, 20))
kb = WIN / 1000.0
for i in range(NP):
    if strand[i] == "-":
        seq = G.fetch(chrom[i], center[i], center[i] + WIN, "-")
    else:
        seq = G.fetch(chrom[i], center[i] - WIN, center[i], "+")
    tok = B[np.frombuffer(seq.encode("ascii"), np.uint8)] if seq else np.empty(0, np.int64)
    sc = score_window(tok)
    valid = np.isfinite(sc); npos[i] = valid.sum()
    v = sc[valid]
    for t in LADDER: counts[t][i] = (v >= t).sum()
    tg = tok[tok >= 0]
    if tg.size >= 10:
        bc = np.bincount(tg, minlength=4).astype(float)
        st = np.bincount(tg[:-1] * 4 + tg[1:], minlength=16).astype(float).reshape(4, 4)
        p0 = (bc + 0.5) / (bc.sum() + 2.0)
        T = (st + 0.25) / (st.sum(1, keepdims=True) + 1.0)
        Wlog[i] = np.concatenate([np.log(p0), np.log(T).ravel()])
    else:
        Wlog[i] = np.nan
    if (i + 1) % 20000 == 0: log(f"  scored {i+1:,}/{NP:,}")
log(f"[scan] mean valid positions/window {npos.mean():.1f}")

density = {t: counts[t] / kb for t in LADDER}

# analytic dinucleotide expected count at the primary threshold -> u1_excess
Ht = np.where(SCORE >= PRIMARY)[0]
Fh = FEAT[Ht]
log(f"[excess] |H_t| at {PRIMARY} = {len(Ht):,} 9-mers ; computing analytic expectation")
q = np.zeros(NP)
CH = 2000
for s in range(0, NP, CH):
    Wl = Wlog[s:s + CH]
    ok = np.isfinite(Wl).all(1)
    lp = Fh @ Wl[ok].T
    q_chunk = np.exp(lp).sum(0)
    qq = np.full(Wl.shape[0], np.nan); qq[ok] = q_chunk
    q[s:s + CH] = qq
exp_count = npos * q
excess_density = (counts[PRIMARY] - exp_count) / kb

gorder = np.argsort(gene, kind="stable")
g_s = gene[gorder]
cut = np.r_[0, np.flatnonzero(g_s[1:] != g_s[:-1]) + 1, len(g_s)]
groups = []
for j in range(len(cut) - 1):
    idx = gorder[cut[j]:cut[j + 1]]
    c = idx[isc[idx] == 1]; k = idx[isc[idx] == 0]
    if len(c) == 1 and len(k) >= 1: groups.append((int(c[0]), k))
log(f"[within] {len(groups):,} genes with 1 canonical + >=1 competitor")

def wilcoxon_p(dd):
    dd = dd[dd != 0]; n = dd.size
    if n < 10: return float("nan")
    a = np.abs(dd); o = np.argsort(a, kind="mergesort"); sa = a[o]
    ng = np.r_[True, sa[1:] != sa[:-1]]; grp = np.cumsum(ng)
    r = np.empty(n); st = np.r_[np.nonzero(ng)[0], n]
    av = (st[:-1] + st[1:] + 1) / 2.0; r[o] = av[grp - 1]
    W = r[dd > 0].sum(); mu = n * (n + 1) / 4.0
    tie = np.bincount(grp)[1:].astype(float)
    var = (n * (n + 1) * (2 * n + 1) - (tie ** 3 - tie).sum() / 2.0) / 24.0
    return math.erfc(abs((W - mu) / math.sqrt(var)) / math.sqrt(2)) if var > 0 else float("nan")

def auroc(pos, neg):
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0: return float("nan")
    a = np.concatenate([pos, neg]); o = np.argsort(a, kind="mergesort"); sa = a[o]
    ng = np.r_[True, sa[1:] != sa[:-1]]; grp = np.cumsum(ng)
    ranks = np.empty(len(a)); st = np.r_[np.nonzero(ng)[0], len(a)]
    av = (st[:-1] + st[1:] + 1) / 2.0; ranks[o] = av[grp - 1]
    R1 = ranks[:n1].sum()
    return (R1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)

def within_stats(val):
    wr, pdiff = [], []
    for c, ks in groups:
        vc = val[c]; vk = val[ks]
        vk = vk[np.isfinite(vk)]
        if not np.isfinite(vc) or vk.size == 0: continue
        win = np.mean((vk > vc) * 1.0 + (vk == vc) * 0.5)
        wr.append(win); pdiff.append(vk.mean() - vc)
    wr = np.array(wr); pdiff = np.array(pdiff)
    bs = pdiff[rng.integers(0, len(pdiff), size=(2000, len(pdiff)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pooled = 1.0 - auroc(val[isc == 1], val[isc == 0])
    return dict(genes=len(wr), winrate=wr.mean(), pdiff=pdiff.mean(),
                ci_lo=lo, ci_hi=hi, wilcox=wilcoxon_p(pdiff), pooled_auroc=1.0 - pooled)

rows = []
for t in LADDER:
    r = within_stats(density[t]); r["feature"] = f"U1 density @ {t:g}"; rows.append(r)
r = within_stats(excess_density); r["feature"] = "u1_excess @ 5.25 (supp.)"; rows.append(r)

print("\n================  TASK B: U1 within-gene robustness across the threshold ladder  ================")
print("feature = PAS-proximal U1 density (pA-500 -> pA); within-gene canonical vs competitor")
print(f"{len(groups):,} genes (1 canonical + >=1 competitor)\n")
hdr = f"{'feature':<26} {'genes':>6} {'win-rate':>9} {'paired_diff':>12} {'95% CI':>20} {'Wilcoxon':>10} {'pooled_AUROC':>12}"
print(hdr); print("-" * len(hdr))
def pf(p): return "<1e-300" if p < 1e-300 else f"{p:.1e}"
for r in rows:
    ci = f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]"
    print(f"{r['feature']:<26} {r['genes']:>6} {r['winrate']:>9.4f} {r['pdiff']:>+12.4f} {ci:>20} "
          f"{pf(r['wilcox']):>10} {r['pooled_auroc']:>12.4f}")
print("\nwin-rate > 0.5 = competitors carry more U1 (suppression direction).")
print("pooled AUROC < 0.5 expected: U1 predicts the competitor, so as a canonical predictor it is below chance.")

with open("u1_threshold_robustness.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["feature", "genes", "winrate", "paired_diff", "ci_lo", "ci_hi", "wilcoxon_p", "pooled_auroc"])
    for r in rows:
        w.writerow([r["feature"], r["genes"], f"{r['winrate']:.6f}", f"{r['pdiff']:.6f}",
                    f"{r['ci_lo']:.6f}", f"{r['ci_hi']:.6f}", f"{r['wilcox']:.3e}", f"{r['pooled_auroc']:.6f}"])
print("\n[out] u1_threshold_robustness.tsv")
