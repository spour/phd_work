#!/usr/bin/env python3
# decoy_score_26.py with the u1 track replaced by strong 5'ss density
import sys, math, time, csv
import numpy as np
import binned_common as K
from binned_common import log

FASTA = "../train/data/hg38.fa"
U1NPY = "../train/splice_train/data/u1_lookup.npy"
ME5NPY = "../train/splice_train/data/me5_lookup.npy"
UP, DN, BIN = 2000, 500, 100
NBIN = (UP + DN) // BIN; SEQLEN = UP + DN
ACC_NSTART = 1875
rng = np.random.default_rng(K.SEED)

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

POW = (4 ** np.arange(8, -1, -1)).astype(np.int64)
def dense(path, neg):
    a = np.load(path, allow_pickle=True); d = a.item() if a.dtype == object else dict(a)
    A = np.full(4 ** 9, np.nan)
    ks = CODE[np.frombuffer("".join(d.keys()).encode("ascii"), np.uint8)].reshape(-1, 9).astype(np.int64)
    A[ks @ POW] = (-1.0 if neg else 1.0) * np.fromiter(d.values(), np.float64, len(d))
    return A
U1 = dense(U1NPY, True); ME5 = dense(ME5NPY, False)
o9 = np.arange(UP + DN - 9 + 1) + 3 - UP
bb = (o9 + UP) // BIN; kk = (bb >= 0) & (bb < NBIN); bk = bb[kk]
nb9 = np.bincount(bk, minlength=NBIN).astype(float)
def dens(t, tab, thr):
    t = t.astype(np.int64)
    sw = np.lib.stride_tricks.sliding_window_view(t, 9); g = (sw < 4).all(1)
    u = np.nan_to_num(np.where(g, tab[np.where(g, sw @ POW, 0)], np.nan), nan=-99.0)
    s = np.bincount(bk, weights=(u >= thr)[kk].astype(float), minlength=NBIN)
    return np.where(nb9 > 0, s / np.maximum(nb9, 1) * 1000.0, np.nan)

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

# peaks
Dn = np.load("cnn_data.npz", allow_pickle=True); ok = Dn["ok"]
X = Dn["X"][ok]; center = Dn["center"][ok]; strand = Dn["strand"][ok]
gene = K.gene; y = K.y; chrom = K.C0["chrom"][K.ok]; N = len(y)
Z = np.load("binned_profiles.npz", allow_pickle=True)
P = Z["P"]; CH = [str(c) for c in Z["channels"]]
U1P = P[:, CH.index("u1_dens4"), :].astype(float)
M5P = P[:, CH.index("me5_dens8"), :].astype(float)

chk = rng.choice(N, 300, replace=False)
bad = sum(1 for i in chk if not np.array_equal(fetch_tok(chrom[i], int(center[i]), strand[i]), X[i]))
assert bad == 0, f"fetch check failed on {bad}/300"
chk = rng.choice(N, 3000, replace=False)
e4 = max(np.nanmax(np.abs(dens(X[i], U1, 4.0).astype(np.float32) - U1P[i].astype(np.float32))) for i in chk)
e8 = max(np.nanmax(np.abs(dens(X[i], ME5, 8.0).astype(np.float32) - M5P[i].astype(np.float32))) for i in chk)
assert e4 == 0.0 and e8 == 0.0, f"density mismatch u1 {e4} me5 {e8}"
aerr = np.array([abs(acc_max(X[i]) - K.ACC[i]) for i in chk[:1000]])
assert np.quantile(aerr, 0.999) < 1e-3
log("[check] fetch (300), u1_dens4 and me5_dens8 (3,000, exact), acceptor (1,000): OK")

# decoys
S = np.load("u1_5ss_sitescan.npz", allow_pickle=True)
dm = S["cls"] == "decoy"
dgene = S["gene"][dm]; dcen = S["center"][dm].astype(np.int64)
real_ix = {gene[i]: i for i in np.flatnonzero(y == 1)}
dchrom = np.array([chrom[real_ix[g]] for g in dgene])
dstrand = np.array([strand[real_ix[g]] for g in dgene])
t0 = time.time()
DU1 = np.full((len(dgene), NBIN), np.nan); DM5 = np.full((len(dgene), NBIN), np.nan)
DACC = np.full(len(dgene), np.nan)
for j in range(len(dgene)):
    t = fetch_tok(dchrom[j], int(dcen[j]), dstrand[j])
    if t is None or (t == 255).all(): continue
    DU1[j] = dens(t, U1, 4.0); DM5[j] = dens(t, ME5, 8.0); DACC[j] = acc_max(t)
    if (j + 1) % 10000 == 0: log(f"  decoys {j+1}/{len(dgene)}  {time.time()-t0:.0f}s")
dok = np.isfinite(DACC)
np.savez_compressed("decoy_feats.npz", gene=dgene, chrom=dchrom, strand=dstrand, center=dcen,
                    u1_dens4=DU1, me5_dens8=DM5, acc=DACC, ok=dok)
DU1, DM5, DACC = DU1[dok], DM5[dok], DACC[dok]
dgene, dchrom = dgene[dok], dchrom[dok]
log(f"[decoys] {dok.sum()} featurised in {len(set(dgene))} genes -> decoy_feats.npz")

up5 = slice(15, 20)
log(f"[check] pooled me5>=8 per kb over [-500,0): real {np.nanmean(M5P[y==1][:, up5]):.2f} "
    f"competitor {np.nanmean(M5P[y==0][:, up5]):.2f} decoy {np.nanmean(DM5[:, up5]):.2f} "
    f"(figure: 0.64 / 0.81 / 0.87)")

# loco
GI = {g: np.flatnonzero(gene == g) for g in sorted(set(gene))}
def groups(ii):
    by = {}
    for k_, g in enumerate(gene[ii]):
        r, c = by.setdefault(g, ([], []))
        (r if y[ii][k_] == 1 else c).append(k_)
    return [(np.array(r), np.array(c)) for r, c in by.values() if r and c]

def gene_means(M):
    return {g: np.where(np.isfinite(m), m, 0.0) for g, ix in GI.items() for m in [np.nanmean(M[ix], 0)]}

def loco(XP, XD):
    Cg = K.centre_by_gene(XP)
    gm = gene_means(XP); CgD = XD - np.array([gm[g] for g in dgene])
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
    return sc, scd

def wilcoxon(v):
    nz = v[v != 0]; n = len(nz); a = np.abs(nz)
    if n == 0: return 0.0, "nan", 0
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
bs = np.random.default_rng(K.SEED).integers(0, len(GD), size=(2000, len(GD)))

def contrast(tag, vp, vd, R):
    cen = vp - np.array([vp[GI[g]].mean() for g in gene])
    cend = vd - np.array([vp[GI[g]].mean() for g in dgene])
    for nm, v in (("real", cen[y == 1]), ("competitor", cen[y == 0]), ("decoy", cend)):
        R[f"{tag}.centred_mean_{nm}"] = f"{np.mean(v):+.4f}"
    wr = {k: [] for k in ("rc", "rd", "cd")}; dl = {k: [] for k in ("rc", "cd", "rd")}
    for g in GD:
        ix = GI[g]; r = vp[ix][y[ix] == 1][0]; c = vp[ix][y[ix] == 0]; d = vd[DI[g]]
        wr["rc"].append(float((r > c).mean() + 0.5 * (r == c).mean()))
        wr["rd"].append(float((r > d).mean() + 0.5 * (r == d).mean()))
        wr["cd"].append(float((c[:, None] > d[None, :]).mean() + 0.5 * (c[:, None] == d[None, :]).mean()))
        dl["rc"].append(r - c.mean()); dl["cd"].append(c.mean() - d.mean()); dl["rd"].append(r - d.mean())
    for k, nm in (("rc", "real>competitor"), ("rd", "real>decoy"), ("cd", "competitor>decoy")):
        v = np.array(wr[k]); lo, hi = np.percentile(v[bs].mean(1), [2.5, 97.5])
        R[f"{tag}.winrate_{nm}"] = f"{v.mean():.4f} [{lo:.4f}, {hi:.4f}]"
    for k, nm in (("rc", "real-competitor"), ("cd", "competitor-decoy"), ("rd", "real-decoy")):
        v = np.array(dl[k]); z, p, n = wilcoxon(v)
        R[f"{tag}.delta_{nm}"] = f"mean {v.mean():+.4f} median {np.median(v):+.4f} z {z:+.2f} p {p} n {n}"
    drd = np.mean(dl["rd"])
    if drd != 0: R[f"{tag}.competitor_share_of_real_decoy_gap"] = f"{np.mean(dl['cd']) / drd:.3f}"

R = {"n_decoys": len(dgene), "n_genes_with_decoys": len(GD)}

contrast("raw_me5_up500", np.nanmean(M5P[:, up5], 1), np.nanmean(DM5[:, up5], 1), R)
contrast("raw_me5_window", np.nanmean(M5P, 1), np.nanmean(DM5, 1), R)

for tag, XP, XD in (("model_U1+acc", np.column_stack([U1P, K.ACC]), np.column_stack([DU1, DACC])),
                    ("model_ME5+acc", np.column_stack([M5P, K.ACC]), np.column_stack([DM5, DACC])),
                    ("model_ME5only", M5P, DM5)):
    t1 = time.time(); sc, scd = loco(XP, XD)
    if tag == "model_U1+acc":
        ref = np.array([float(r["centred_score"]) for r in
                        csv.DictReader(open("score_separation_sites.tsv"), delimiter="\t")])
        cen = sc - np.array([sc[GI[g]].mean() for g in gene])
        err = np.abs(cen - ref).max(); assert err < 1e-4, f"panel D not reproduced ({err})"
        log(f"[check] U1 model reproduces panel D (max |diff| {err:.1e})")
    contrast(tag, sc, scd, R)
    log(f"[loco] {tag} done {time.time()-t1:.0f}s")

with open("decoy_score_me5.tsv", "w") as fh:
    fh.write("key\tvalue\n")
    for k_, v_ in R.items(): fh.write(f"{k_}\t{v_}\n")
for k_, v_ in R.items(): print(f"{k_:52s} {v_}")
log("[out] decoy_score_me5.tsv decoy_feats.npz")
