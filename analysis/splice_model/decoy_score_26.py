#!/usr/bin/env python3
# where the 26-feature splice model places the intronic decoys
import sys, math, time
import numpy as np
import binned_common as K
from binned_common import log

FASTA = "../train/data/hg38.fa"
U1NPY = "../train/splice_train/data/u1_lookup.npy"
UP, DN, BIN = 2000, 500, 100
NBIN = (UP + DN) // BIN; SEQLEN = UP + DN
ACC_NSTART = 1875
rng = np.random.default_rng(K.SEED)

# sequence fetch (cnn_data.py convention)
class Faidx:
    _T = str.maketrans("ACGTNacgtn", "TGCANTGCAN")
    def __init__(s, fa):
        s.fh = open(fa, "rb"); s.idx = {}
        for L_ in open(fa + ".fai"):
            p = L_.split(); s.idx[p[0]] = (int(p[1]), int(p[2]), int(p[3]), int(p[4]))
    def fetch(s, c, a, b, st="+"):
        c = c if c in s.idx else c.lstrip("chr")
        if c not in s.idx: return None
        Ln, off, lb, lw = s.idx[c]
        if a < 0 or b > Ln: return None
        bp = lambda p: off + (p // lb) * lw + (p % lb)
        s.fh.seek(bp(a)); raw = s.fh.read(bp(b) - bp(a))
        q = raw.replace(b"\n", b"").replace(b"\r", b"")[:b - a].decode("ascii").upper()
        return q.translate(s._T)[::-1] if st == "-" else q

CODE = np.full(256, 255, np.uint8)
for ch_, v_ in zip(b"ACGT", (0, 1, 2, 3)): CODE[ch_] = v_
G = Faidx(FASTA)
def fetch_tok(ch, c, st):
    a, b = (c - UP, c + DN) if st == "+" else (c - DN + 1, c + UP + 1)
    s = G.fetch(ch, a, b, st)
    if not s or len(s) != SEQLEN: return None
    return CODE[np.frombuffer(s.encode("ascii"), np.uint8)]

# u1 density (splice_model_26.py dens4)
POW = (4 ** np.arange(8, -1, -1)).astype(np.int64)
_a = np.load(U1NPY, allow_pickle=True); _d = _a.item() if _a.dtype == object else dict(_a)
U1 = np.full(4 ** 9, np.nan)
_ks = CODE[np.frombuffer("".join(_d.keys()).encode("ascii"), np.uint8)].reshape(-1, 9).astype(np.int64)
U1[_ks @ POW] = -np.fromiter(_d.values(), np.float64, len(_d))
o9 = np.arange(UP + DN - 9 + 1) + 3 - UP
bb = (o9 + UP) // BIN; kk = (bb >= 0) & (bb < NBIN); bk = bb[kk]
nb9 = np.bincount(bk, minlength=NBIN).astype(float)
def dens4(t):
    t = t.astype(np.int64)
    sw = np.lib.stride_tricks.sliding_window_view(t, 9); g = (sw < 4).all(1)
    u = np.nan_to_num(np.where(g, U1[np.where(g, sw @ POW, 0)], np.nan), nan=-99.0)
    s = np.bincount(bk, weights=(u >= 4.0)[kk].astype(float), minlength=NBIN)
    return np.where(nb9 > 0, s / np.maximum(nb9, 1) * 1000.0, np.nan)

# maxent score3, vectorised (task_binned_features.py)
from maxentpy import maxent
M3 = maxent.load_matrix3()
MT = [np.array([M3[i][j] for j in range(len(M3[i]))], float) for i in range(9)]
CONST3 = np.log2(maxent.cons1_3["A"] * maxent.cons2_3["G"] /
                 (maxent.bgd_3["A"] * maxent.bgd_3["G"]))
LT = [np.log2(np.maximum(m, 1e-300)) for m in MT]
def khash(c, k, n):
    w = np.lib.stride_tricks.sliding_window_view(c, k)[:n]
    bad = (w < 0).any(1)
    h = w @ (4 ** np.arange(k - 1, -1, -1))
    h[bad] = -1
    return h
def me3_scores(c, n):
    out = np.full(n, np.nan)
    ok = (c[18:18 + n] == 0) & (c[19:19 + n] == 2)
    h7 = [khash(c[o:], 7, n) for o in (0, 7, 4, 11)]
    h4a = khash(c[14:], 4, n); h3b = khash(c[20:], 3, n)
    h3 = khash(c[4:], 3, n); h4b = khash(c[7:], 4, n); h3c = khash(c[11:], 3, n)
    parts = h7 + [h4a, h3b, h3, h4b, h3c]
    ok &= np.all(np.array([p >= 0 for p in parts]), 0)
    if not ok.any(): return out
    i = np.where(ok)[0]
    h14 = h4a[i] * 64 + h3b[i]
    v = (LT[0][h7[0][i]] + LT[1][h7[1][i]] + LT[2][h14] + LT[3][h7[2][i]] + LT[4][h7[3][i]]
         - LT[5][h3[i]] - LT[6][h4b[i]] - LT[7][h3c[i]] - LT[8][h4a[i]])
    out[i] = CONST3 + v
    return out
def acc_max(t):
    c = t.astype(np.int64); c[c == 255] = -1
    return np.nanmax(me3_scores(c, ACC_NSTART))

# peak side, and the three checks
Dn = np.load("cnn_data.npz", allow_pickle=True); ok = Dn["ok"]
X = Dn["X"][ok]; center = Dn["center"][ok]; strand = Dn["strand"][ok]
gene = K.gene; y = K.y; chrom = K.C0["chrom"][K.ok]; N = len(y)
Z = np.load("binned_profiles.npz", allow_pickle=True)
P = Z["P"]; CH = [str(c) for c in Z["channels"]]
D4P = P[:, CH.index("u1_dens4"), :].astype(float)
XP = np.column_stack([D4P, K.ACC]); assert XP.shape[1] == 26

chk = rng.choice(N, 300, replace=False)
bad = sum(1 for i in chk if not np.array_equal(fetch_tok(chrom[i], int(center[i]), strand[i]), X[i]))
assert bad == 0, f"fetch check failed on {bad}/300 peaks"
log("[check] fetch reproduces cnn_data.npz X on 300 peaks: OK")

chk = rng.choice(N, 3000, replace=False)
d4err = max(np.nanmax(np.abs(dens4(X[i]).astype(np.float32) - D4P[i].astype(np.float32))) for i in chk)
assert d4err == 0.0, f"dens4 mismatch {d4err}"
log("[check] dens4 reproduces binned_profiles.npz u1_dens4 on 3,000 peaks: OK")
aerr = np.array([abs(acc_max(X[i]) - K.ACC[i]) for i in chk])
log(f"[check] acceptor vs spl_upstream_max_me3 on 3,000 peaks: max |diff| {aerr.max():.2e}, "
    f"{(aerr < 1e-3).mean():.4f} within 1e-3")
assert np.quantile(aerr, 0.999) < 1e-3, "acceptor definition does not reproduce the pipeline"

S = np.load("u1_5ss_sitescan.npz", allow_pickle=True)
dm = S["cls"] == "decoy"
dgene = S["gene"][dm]; dcen = S["center"][dm].astype(np.int64)
real_ix = {gene[i]: i for i in np.flatnonzero(y == 1)}
assert all(g in real_ix for g in dgene), "decoy gene missing from analysis set"
dchrom = np.array([chrom[real_ix[g]] for g in dgene])
dstrand = np.array([strand[real_ix[g]] for g in dgene])
t0 = time.time()
XD = np.full((len(dgene), 26), np.nan); dok = np.zeros(len(dgene), bool)
for j in range(len(dgene)):
    t = fetch_tok(dchrom[j], int(dcen[j]), dstrand[j])
    if t is None or (t == 255).all(): continue
    XD[j, :25] = dens4(t); XD[j, 25] = acc_max(t); dok[j] = np.isfinite(XD[j, 25])
    if (j + 1) % 10000 == 0: log(f"  decoys {j+1}/{len(dgene)}  {time.time()-t0:.0f}s")
log(f"[decoys] {len(dgene)} in sitescan, {dok.sum()} featurised, in {len(set(dgene))} genes")
XD = XD[dok]; dgene = dgene[dok]; dchrom = dchrom[dok]

GI = {g: np.flatnonzero(gene == g) for g in sorted(set(gene))}
Cg = K.centre_by_gene(XP)
gmu = {g: np.where(np.isfinite(m), m, 0.0) for g, ix in GI.items()
       for m in [np.nanmean(XP[ix], 0)]}
CgD = XD - np.array([gmu[g] for g in dgene])

def groups(ii):
    by = {}
    for k_, g in enumerate(gene[ii]):
        r, c = by.setdefault(g, ([], []))
        (r if y[ii][k_] == 1 else c).append(k_)
    return [(np.array(r), np.array(c)) for r, c in by.values() if r and c]
def fe(fi, ei, lam, dei=None):
    m = np.nanmean(Cg[fi], 0); m = np.where(np.isfinite(m), m, 0.0)
    s = np.nanstd(Cg[fi], 0); s[~np.isfinite(s) | (s == 0)] = 1.0
    Zm = (np.where(np.isfinite(Cg), Cg, m) - m) / s
    b = K.fit_ridge(np.column_stack([np.ones(len(fi)), Zm[fi]]), y[fi], lam)
    out = np.column_stack([np.ones(len(ei)), Zm[ei]]) @ b
    if dei is None: return out
    Zd = (np.where(np.isfinite(CgD[dei]), CgD[dei], m) - m) / s
    return out, np.column_stack([np.ones(len(dei)), Zd]) @ b

sc = np.full(N, np.nan); scd = np.full(len(dgene), np.nan)
for ch in sorted(set(chrom)):
    te = np.flatnonzero(chrom == ch); tr = np.flatnonzero(chrom != ch)
    trg = np.array(sorted(set(gene[tr]))); rs = np.random.default_rng(K.SEED)
    sh = trg.copy(); rs.shuffle(sh); ivg = set(sh[:int(round(0.25 * len(sh)))])
    iv = tr[np.array([g in ivg for g in gene[tr]])]
    it = tr[np.array([g not in ivg for g in gene[tr]])]
    gi = groups(iv); best = (-1.0, 1.0)
    for lam in K.LAMBDAS:
        w = K.winrate(fe(it, iv, lam), gi)
        if w > best[0]: best = (w, lam)
    de = np.flatnonzero(dchrom == ch)
    sc[te], scd[de] = fe(tr, te, best[1], de)
assert np.isfinite(sc).all() and np.isfinite(scd).all()

gmean = {g: sc[ix].mean() for g, ix in GI.items()}
cen = sc - np.array([gmean[g] for g in gene])
cend = scd - np.array([gmean[g] for g in dgene])
import csv
ref = np.array([float(r["centred_score"]) for r in csv.DictReader(open("score_separation_sites.tsv"), delimiter="\t")])
serr = np.abs(cen - ref).max()
assert serr < 1e-4, f"LOCO does not reproduce panel D ({serr})"
log(f"[check] LOCO peak scores reproduce score_separation_sites.tsv (max |diff| {serr:.1e}): OK")

def wilcoxon(v):
    nz = v[v != 0]; n = len(nz); a = np.abs(nz)
    o = np.argsort(a, kind="stable"); sa = a[o]
    _, inv, cnt = np.unique(sa, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); mid = ((cs - cnt) + cs + 1) / 2.0
    rank = np.empty(n); rank[o] = mid[inv]
    V = float(rank[nz > 0].sum()); mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - float(np.sum(cnt ** 3 - cnt)) / 48.0
    z = (V - mu + (0.5 if V < mu else -0.5)) / math.sqrt(var)
    pv = math.erfc(abs(z) / math.sqrt(2))
    return z, ("< 2.2e-16" if pv < 2.2e-16 else f"{pv:.3g}"), n

DI = {}
for j, g in enumerate(dgene): DI.setdefault(g, []).append(j)
GD = sorted(DI)
wr_rc, wr_rd, wr_cd, d_rc, d_cd, d_rd, frac_d_above_real = [], [], [], [], [], [], []
for g in GD:
    ix = GI[g]; r = sc[ix][y[ix] == 1][0]; c = sc[ix][y[ix] == 0]; dd = scd[DI[g]]
    wr_rc.append(float((r > c).mean() + 0.5 * (r == c).mean()))
    wr_rd.append(float((r > dd).mean() + 0.5 * (r == dd).mean()))
    wr_cd.append(float((c[:, None] > dd[None, :]).mean() + 0.5 * (c[:, None] == dd[None, :]).mean()))
    d_rc.append(r - c.mean()); d_cd.append(c.mean() - dd.mean()); d_rd.append(r - dd.mean())
    frac_d_above_real.append((dd > r).mean())
wr_rc, wr_rd, wr_cd = map(np.array, (wr_rc, wr_rd, wr_cd))
d_rc, d_cd, d_rd = map(np.array, (d_rc, d_cd, d_rd))
bs = np.random.default_rng(K.SEED).integers(0, len(GD), size=(2000, len(GD)))
ci = lambda v: np.percentile(v[bs].mean(1), [2.5, 97.5])

R = {}
R["n_decoys"] = len(dgene); R["n_genes_with_decoys"] = len(GD)
for nm, v in (("real", cen[y == 1]), ("competitor", cen[y == 0]), ("decoy", cend)):
    R[f"median_centred_{nm}"] = f"{np.median(v):+.4f}"
    R[f"mean_centred_{nm}"] = f"{np.mean(v):+.4f}"
for nm, v in (("real_vs_competitor", wr_rc), ("real_vs_decoy", wr_rd), ("competitor_vs_decoy", wr_cd)):
    lo, hi = ci(v); R[f"winrate_{nm}"] = f"{v.mean():.4f} [{lo:.4f}, {hi:.4f}]"
for nm, v in (("real_minus_competitor", d_rc), ("competitor_minus_decoy", d_cd), ("real_minus_decoy", d_rd)):
    z, p, n = wilcoxon(v)
    R[f"delta_{nm}"] = f"median {np.median(v):+.4f} mean {v.mean():+.4f} positive {np.mean(v>0):.3f} z {z:+.2f} p {p} n {n}"
R["gap_share_competitor_minus_decoy"] = f"{d_cd.mean() / d_rd.mean():.3f}"
R["frac_decoys_above_own_real_end_gene_avg"] = f"{np.mean(frac_d_above_real):.4f}"
cenX = XP - np.array([gmu[g] for g in gene])
up5 = slice(15, 20)
for nm, A_, B_ in (("u1dens4_upstream500", cenX[:, up5].mean(1), CgD[:, up5].mean(1)),
                   ("acceptor_max", cenX[:, 25], CgD[:, 25])):
    R[f"feat_{nm}_real"] = f"{np.mean(A_[y == 1]):+.3f}"
    R[f"feat_{nm}_competitor"] = f"{np.mean(A_[y == 0]):+.3f}"
    R[f"feat_{nm}_decoy"] = f"{np.mean(B_):+.3f}"

with open("decoy_score_26.tsv", "w") as fh:
    fh.write("key\tvalue\n")
    for k_, v_ in R.items(): fh.write(f"{k_}\t{v_}\n")
for k_, v_ in R.items(): print(f"{k_:42s} {v_}")
log("[out] decoy_score_26.tsv")
