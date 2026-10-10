#!/usr/bin/env python3
# one genome-wide identification number per model, computed one way
import sys, csv, math, time
import numpy as np
import binned_common as K
from binned_common import log

UP, DN, BIN = 2000, 500, 100
NBIN = (UP + DN) // BIN
U1NPY = "../train/splice_train/data/u1_lookup.npy"

# universe
LAB = list(csv.DictReader(open("cpa_peaks_labeled.tsv"), delimiter="\t"))
tb = lambda x: x in ("True", "true", "1")
gcomp = {}
for r in LAB: gcomp[r["gene"]] = gcomp.get(r["gene"], 0) + (not tb(r["is_canonical"]))
NALL = len(gcomp); NUNC = sum(1 for v in gcomp.values() if v == 0)
assert NALL == 15560 and NUNC == 8977, (NALL, NUNC)

M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
wg_gene = np.array([r["gene"] for r in M]); wg_chrom = np.array([r["chrom"] for r in M])
wg_cen = np.array([int(r["center"]) for r in M]); wg_strand = np.array([r["strand"] for r in M])
wg_y = np.array([int(float(r["is_canonical"])) for r in M]); NPK = len(M)
WKEY = list(zip(wg_gene, wg_chrom, wg_cen.tolist(), wg_strand))
CONT = np.array(sorted(set(wg_gene)))
GIW = {g: np.flatnonzero(wg_gene == g) for g in CONT}
PAIRW = {g: (ix[wg_y[ix] == 1], ix[wg_y[ix] == 0]) for g, ix in GIW.items()}
assert all(len(r) == 1 and len(c) >= 1 for r, c in PAIRW.values())
assert len(CONT) == NALL - NUNC
log(f"[universe] {NALL} genes, {NUNC} uncontested, {len(CONT)} contested, {NPK} contested candidates")

def hits(sc):
    return np.array([float((sc[PAIRW[g][1]] >= sc[PAIRW[g][0]][0]).sum() == 0) for g in CONT])

# cnn (already analysis-trained, genome-wide scored)
S = np.load("cnn_wg_scores.npz", allow_pickle=True)
assert list(zip(S["gene"], S["chrom"], S["center"].tolist(), S["strand"])) == WKEY
SC_CNN = S["logit"].astype(float)
ref = {r["model"] + "|" + r["scope"]: float(r["p_at_1"])
       for r in csv.DictReader(open("cnn_wholegenome_p1.tsv"), delimiter="\t")}
assert abs(hits(SC_CNN).mean() - ref["Sequence CNN|contested"]) < 1e-5
log("[check] CNN contested P@1 reproduces cnn_wholegenome_p1.tsv")

# splice features for all contested candidates
POW = (4 ** np.arange(8, -1, -1)).astype(np.int64)
CODE = np.full(256, 4, np.uint8)
for i_, ch_ in enumerate(b"ACGT"): CODE[ch_] = i_
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

TOK = np.load("cnn_wg_tokens.npy", mmap_mode="r"); assert TOK.shape == (NPK, UP + DN)
t0 = time.time(); D4W = np.empty((NPK, NBIN))
for i in range(NPK):
    D4W[i] = dens4(np.asarray(TOK[i]))
    if (i + 1) % 20000 == 0: log(f"  u1_dens4 {i+1}/{NPK}  {time.time()-t0:.0f}s")

import pyarrow.parquet as pq
F = pq.read_table("shortlist_feat_all.parquet",
                  columns=["gene", "chrom", "center", "strand", "spl_upstream_max_me3"]).to_pandas()
akey = dict(zip(zip(F.gene, F.chrom, F.center.astype(int).tolist(), F.strand), F.spl_upstream_max_me3))
ACCW = np.array([akey[k] for k in WKEY], float)

Dn = np.load("cnn_data.npz", allow_pickle=True); ok = Dn["ok"]
AKEY = list(zip(K.gene, Dn["chrom"][ok], Dn["center"][ok].tolist(), Dn["strand"][ok]))
wpos = {k: i for i, k in enumerate(WKEY)}
A2W = np.array([wpos[k] for k in AKEY])
Z = np.load("binned_profiles.npz", allow_pickle=True)
U1P = Z["P"][:, [str(c) for c in Z["channels"]].index("u1_dens4"), :].astype(float)
assert np.nanmax(np.abs(D4W[A2W].astype(np.float32) - U1P.astype(np.float32))) == 0.0
assert np.nanmax(np.abs(ACCW[A2W] - K.ACC)) < 1e-6
assert (wg_y[A2W] == K.y).all()
log("[check] token-cache u1_dens4 and parquet acceptor reproduce the analysis-set features exactly")

XA = np.column_stack([U1P, K.ACC]); XW = np.column_stack([D4W, ACCW])

# splice refit on analysis set, scored genome-wide
gene = K.gene; y = K.y; chrom = K.C0["chrom"][K.ok]; N = len(y)
CgA = K.centre_by_gene(XA)
CgW = XW.copy()
for g, ix in GIW.items():
    m_ = np.nanmean(XW[ix], 0); CgW[ix] = XW[ix] - np.where(np.isfinite(m_), m_, 0.0)
def groups(ii):
    by = {}
    for k_, g in enumerate(gene[ii]):
        r, c = by.setdefault(g, ([], []))
        (r if y[ii][k_] == 1 else c).append(k_)
    return [(np.array(r), np.array(c)) for r, c in by.values() if r and c]
def fit(fi, lam):
    m_ = np.nanmean(CgA[fi], 0); m_ = np.where(np.isfinite(m_), m_, 0.0)
    s_ = np.nanstd(CgA[fi], 0); s_[~np.isfinite(s_) | (s_ == 0)] = 1.0
    Zm = (np.where(np.isfinite(CgA), CgA, m_) - m_) / s_
    b = K.fit_ridge(np.column_stack([np.ones(len(fi)), Zm[fi]]), y[fi], lam)
    return b, m_, s_, Zm
def apply(b, m_, s_, C):
    return np.column_stack([np.ones(len(C)), (np.where(np.isfinite(C), C, m_) - m_) / s_]) @ b

SC_A = np.full(N, np.nan); SC_SPL = np.full(NPK, np.nan); LAMS = {}
for ch in sorted(set(chrom)):
    te = np.flatnonzero(chrom == ch); tr = np.flatnonzero(chrom != ch)
    trg = np.array(sorted(set(gene[tr]))); rs = np.random.default_rng(K.SEED)
    sh = trg.copy(); rs.shuffle(sh); ivg = set(sh[:int(round(0.25 * len(sh)))])
    iv = tr[np.array([g in ivg for g in gene[tr]])]
    it = tr[np.array([g not in ivg for g in gene[tr]])]
    gi = groups(iv); best = (-1.0, 1.0)
    for lam in K.LAMBDAS:
        b, m_, s_, Zm = fit(it, lam)
        w = K.winrate(np.column_stack([np.ones(len(iv)), Zm[iv]]) @ b, gi)
        if w > best[0]: best = (w, lam)
    b, m_, s_, Zm = fit(tr, best[1]); LAMS[ch] = best[1]
    SC_A[te] = np.column_stack([np.ones(len(te)), Zm[te]]) @ b
    hold = np.flatnonzero(wg_chrom == ch)
    SC_SPL[hold] = apply(b, m_, s_, CgW[hold])
assert set(np.unique(wg_chrom)) == set(chrom) and np.isfinite(SC_SPL).all()
refc = np.array([float(r["centred_score"]) for r in
                 csv.DictReader(open("score_separation_sites.tsv"), delimiter="\t")])
GIA = {g: np.flatnonzero(gene == g) for g in set(gene)}
cenA = SC_A - np.array([SC_A[GIA[g]].mean() for g in gene])
assert np.abs(cenA - refc).max() < 1e-4
log("[check] splice refit reproduces panel D analysis-set scores")

def zin(v):
    out = np.empty_like(v)
    for g, ix in GIW.items():
        s_ = v[ix].std(); out[ix] = (v[ix] - v[ix].mean()) / (s_ if s_ > 1e-9 else 1.0)
    return out
SC_Z = zin(SC_CNN) + zin(SC_SPL)

bs = np.random.default_rng(K.SEED).integers(0, len(CONT), size=(2000, len(CONT)))
R = []; H = {}
H["Baseline"] = np.zeros(len(CONT))
for nm, sc in (("Splice model (26 features)", SC_SPL), ("Sequence CNN", SC_CNN),
               ("Combined (within-gene z-sum)", SC_Z)):
    H[nm] = hits(sc)
for nm, h in H.items():
    c_lo, c_hi = np.percentile(h[bs].mean(1), [2.5, 97.5])
    wg = (NUNC + h.sum()) / NALL
    w_lo, w_hi = np.percentile((NUNC + h[bs].sum(1)) / NALL, [2.5, 97.5])
    R.append((nm, wg, w_lo, w_hi, h.mean(), c_lo, c_hi, int(h.sum())))

hs, hc = H["Splice model (26 features)"], H["Sequence CNN"]
both = int(((hs == 1) & (hc == 1)).sum()); cnn_only = int(((hs == 0) & (hc == 1)).sum())
spl_only = int(((hs == 1) & (hc == 0)).sum()); neither = int(((hs == 0) & (hc == 0)).sum())
nd = cnn_only + spl_only
zmc = (abs(cnn_only - spl_only) - 1) / math.sqrt(nd) if nd else 0.0
p_mc = math.erfc(zmc / math.sqrt(2))
dlt = hc - hs; d_lo, d_hi = np.percentile(dlt[bs].mean(1), [2.5, 97.5])

ncand = np.array([len(GIW[g]) for g in CONT])
strata = [(2, 2), (3, 5), (6, 10), (11, 30), (31, 10**9)]

with open("genomewide_p1.tsv", "w") as fh:
    fh.write("model\tp_at_1_all_genes\tci_lo\tci_hi\tp_at_1_contested\tci_lo\tci_hi\t"
             "contested_genes_correct\tn_all\tn_contested\n")
    for nm, wg, wl, wh, c, cl, chh, k_ in R:
        fh.write(f"{nm}\t{wg:.4f}\t{wl:.4f}\t{wh:.4f}\t{c:.4f}\t{cl:.4f}\t{chh:.4f}\t{k_}\t{NALL}\t{len(CONT)}\n")
    fh.write(f"\n# paired CNN vs splice on contested genes: both {both}, CNN only {cnn_only}, "
             f"splice only {spl_only}, neither {neither}; McNemar z {zmc:.2f} p {p_mc:.3g}; "
             f"CNN - splice {dlt.mean():+.4f} [{d_lo:+.4f}, {d_hi:+.4f}]\n")
    for lo_, hi_ in strata:
        m_ = (ncand >= lo_) & (ncand <= hi_)
        lab = f"{lo_}-{hi_}" if hi_ < 10**9 else f">={lo_}"
        fh.write(f"# candidates {lab}: n {m_.sum()}, chance {np.mean(1.0/ncand[m_]):.3f}, "
                 + ", ".join(f"{nm} {H[nm][m_].mean():.3f}" for nm in H) + "\n")
with open("genomewide_p1_genes.tsv", "w") as fh:
    fh.write("gene\tn_candidates\t" + "\t".join(H) + "\n")
    for j, g in enumerate(CONT):
        fh.write(f"{g}\t{ncand[j]}\t" + "\t".join(str(int(H[nm][j])) for nm in H) + "\n")

print(f"{'model':32s} {'all 15,560 genes':>26s} {'contested 6,583 genes':>28s}")
for nm, wg, wl, wh, c, cl, chh, k_ in R:
    print(f"{nm:32s} {wg:.4f} [{wl:.4f},{wh:.4f}]    {c:.4f} [{cl:.4f},{chh:.4f}] ({k_})")
print(f"paired: both {both}, CNN only {cnn_only}, splice only {spl_only}, neither {neither}; "
      f"McNemar z {zmc:.2f} p {p_mc:.3g}; CNN - splice {dlt.mean():+.4f} [{d_lo:+.4f},{d_hi:+.4f}]")
print(f"chance on contested genes (random pick): {np.mean(1.0/ncand):.4f}")
for lo_, hi_ in strata:
    m_ = (ncand >= lo_) & (ncand <= hi_)
    lab = f"{lo_}-{hi_}" if hi_ < 10**9 else f">={lo_}"
    print(f"  candidates {lab:>6s}: n {m_.sum():5d}  chance {np.mean(1.0/ncand[m_]):.3f}  "
          + "  ".join(f"{nm.split(' (')[0]} {H[nm][m_].mean():.3f}" for nm in H if nm != 'Baseline'))
log("[out] genomewide_p1.tsv genomewide_p1_genes.tsv")
