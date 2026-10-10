#!/usr/bin/env python3
# lock the 26-feature splice model and produce everything the figure needs
import sys, csv, time
import numpy as np
import binned_common as K
from binned_common import log
sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as L

UP, DN, BIN = 2000, 500, 100
NBIN = (UP + DN) // BIN
PAD = 40
ACC = K.ACC; N = len(K.y)
CODE = np.full(256, 4, np.int64)
for i, ch in enumerate("ACGT"): CODE[ord(ch)] = i

o9 = np.arange(UP + DN - 9 + 1) + 3 - UP
bb = (o9 + UP) // BIN; kk = (bb >= 0) & (bb < NBIN); bk = bb[kk]
nb9 = np.bincount(bk, minlength=NBIN).astype(float)

def dens4(tok2d):
    out = np.empty((len(tok2d), NBIN))
    for i in range(len(tok2d)):
        t = tok2d[i].astype(np.int64)
        sw = np.lib.stride_tricks.sliding_window_view(t, 9); g = (sw < 4).all(1)
        u = np.nan_to_num(np.where(g, L.U1[np.where(g, sw @ L.POW, 0)], np.nan), nan=-99.0)
        s = np.bincount(bk, weights=(u >= 4.0)[kk].astype(float), minlength=NBIN)
        out[i] = np.where(nb9 > 0, s / np.maximum(nb9, 1) * 1000.0, np.nan)
    return out

# analysis set: lock and score
X = np.load("cnn_data.npz", allow_pickle=True)["X"]
D4 = dens4(X)
Z = np.load("binned_profiles.npz", allow_pickle=True)
P = Z["P"]; CH = [str(c) for c in Z["channels"]]; BINLO = Z["bin_lo"].astype(int)
assert np.nanmax(np.abs(D4.astype(np.float32)
                        - P[:, CH.index("u1_dens4"), :].astype(np.float32))) == 0.0
log("[check] u1_dens4 reproduces binned_profiles.npz exactly in float32")

M26 = np.column_stack([D4, ACC]); assert M26.shape[1] == 26
NAMES = [f"u1_dens4[{BINLO[b]},{BINLO[b]+BIN})" for b in range(NBIN)] + ["spl_upstream_max_me3"]
LAM, IV = K.pick_lambda(M26, "26-feature model")
sc, _ = K.fit_score(M26, K.trv, K.te, LAM)
R = K.suite(sc)
log(f"[lock] within-gene {R['winrate']:.4f} [{R['winrate_lo']:.4f},{R['winrate_hi']:.4f}]  "
    f"P@1 {R['p_at_1']:.4f} [{R['p_at_1_lo']:.4f},{R['p_at_1_hi']:.4f}]  "
    f"AUROC {R['auroc']:.4f}  AUPRC {R['auprc']:.4f}  lambda {LAM:g}  inner-val {IV:.4f}")
assert abs(R["winrate"] - 0.6993) < 5e-4, f"win-rate {R['winrate']:.4f} != 0.6993"
assert abs(R["p_at_1"] - 0.4720) < 5e-4, f"P@1 {R['p_at_1']:.4f} != 0.4720"
log("[check] reproduces the committed 0.6993 / 0.4720")

Cg = K.centre_by_gene(M26)
mu = np.nanmean(Cg[K.trv], 0); mu = np.where(np.isfinite(mu), mu, 0.0)
sd = np.nanstd(Cg[K.trv], 0); sd[~np.isfinite(sd) | (sd == 0)] = 1.0
Zs = (np.where(np.isfinite(Cg), Cg, mu) - mu) / sd
BETA = K.fit_ridge(np.column_stack([np.ones(len(K.trv)), Zs[K.trv]]), K.y[K.trv], LAM)
assert abs(K.winrate(np.column_stack([np.ones(len(K.te)), Zs[K.te]]) @ BETA, K.GRP_TE)
           - R["winrate"]) < 1e-12
np.savez_compressed("splice_model_26.npz", beta=BETA, mu=mu, sd=sd,
                    names=np.array(NAMES, dtype=object), lam=LAM, bin_lo=BINLO,
                    up=UP, dn=DN, inner_val=IV,
                    note=np.array("logit = b0 + sum_j b_j (gene-centred x_j - mu_j)/sd_j"))
with open("splice_model_26_coefficients.tsv", "w") as fh:
    fh.write("feature\tchannel\tbin_lo\tbin_centre\tcoefficient\n")
    fh.write(f"(intercept)\t\t\t\t{BETA[0]:.6f}\n")
    for j, nm in enumerate(NAMES):
        if nm == "spl_upstream_max_me3":
            fh.write(f"{nm}\tacceptor\t\t\t{BETA[j+1]:.6f}\n")
        else:
            fh.write(f"{nm}\tu1_dens4\t{BINLO[j]}\t{BINLO[j]+BIN//2}\t{BETA[j+1]:.6f}\n")
u1b = BETA[1:1 + NBIN]; accb = BETA[-1]
jmin = int(np.argmin(u1b))
log(f"[coef] U1 minimum {u1b[jmin]:+.4f} at the [{BINLO[jmin]}, {BINLO[jmin]+BIN}) bin "
    f"(centre {BINLO[jmin]+BIN//2:+d} nt); U1 range [{u1b.min():+.4f}, {u1b.max():+.4f}]")
log(f"[coef] acceptor weight {accb:+.4f}; mean U1 weight over [-600, +100] "
    f"{u1b[(BINLO >= -600) & (BINLO < 100)].mean():+.4f}")

# whole genome, loco over contested genes
LAB = list(csv.DictReader(open("cpa_peaks_labeled.tsv"), delimiter="\t"))
tb = lambda x: x in ("True", "true", "1")
gcomp = {}
for r in LAB:
    gcomp[r["gene"]] = gcomp.get(r["gene"], 0) + (not tb(r["is_canonical"]))
NALL = len(gcomp); NUNC = sum(1 for v in gcomp.values() if v == 0)
M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
gene = np.array([r["gene"] for r in M]); chrom = np.array([r["chrom"] for r in M])
cen = np.array([int(r["center"]) for r in M]); strand = np.array([r["strand"] for r in M])
yy = np.array([int(float(r["is_canonical"])) for r in M]); NP_ = len(M)
from fastparquet import ParquetFile
F = ParquetFile("shortlist_feat_all.parquet").to_pandas()
key = {(g, c, int(ct), s): i for i, (g, c, ct, s) in
       enumerate(zip(F.gene, F.chrom, F.center, F.strand))}
rw = np.array([key[(gene[i], chrom[i], int(cen[i]), strand[i])] for i in range(NP_)])
ACC2 = F["spl_upstream_max_me3"].to_numpy(float)[rw]
G = L.G_["G"]; t0 = time.time()
TOK = np.empty((NP_, UP + DN), np.int64)
for i in range(NP_):
    c = int(cen[i]); st = strand[i]
    a, b = (c - UP - PAD, c + DN + PAD) if st == "+" else (c - DN - PAD + 1, c + UP + PAD + 1)
    s = G.fetch(chrom[i], a, b, st)
    TOK[i] = CODE[np.frombuffer(s.encode("ascii"), np.uint8)][PAD:PAD + UP + DN]
    if (i + 1) % 20000 == 0: log(f"  scan {i+1}/{NP_}  {time.time()-t0:.0f}s")
WG = np.column_stack([dens4(TOK), ACC2])
log(f"[scan] {NP_} contested-gene peaks in {time.time()-t0:.0f}s")
CONT = np.array(sorted(set(gene)))
GI = {g: np.flatnonzero(gene == g) for g in CONT}
PAIR = {g: (ix[yy[ix] == 1], ix[yy[ix] == 0]) for g, ix in GI.items()}
def groups(ii):
    by = {}
    for k_, g in enumerate(gene[ii]):
        r, c = by.setdefault(g, ([], []))
        (r if yy[ii][k_] == 1 else c).append(k_)
    return [(np.array(r), np.array(c)) for r, c in by.values() if r and c]
def centre(Mx):
    out = Mx.astype(float).copy()
    o = np.argsort(gene, kind="stable"); gs = gene[o]
    cut = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1, len(gs)]
    for j in range(len(cut) - 1):
        ix = o[cut[j]:cut[j + 1]]
        m_ = np.nanmean(out[ix], 0); out[ix] = out[ix] - np.where(np.isfinite(m_), m_, 0.0)
    return out
def fe(Cg, fi, ei, lam):
    m_ = np.nanmean(Cg[fi], 0); m_ = np.where(np.isfinite(m_), m_, 0.0)
    s_ = np.nanstd(Cg[fi], 0); s_[~np.isfinite(s_) | (s_ == 0)] = 1.0
    Zm = (np.where(np.isfinite(Cg), Cg, m_) - m_) / s_
    bt = K.fit_ridge(np.column_stack([np.ones(len(fi)), Zm[fi]]), yy[fi].astype(float), lam)
    return np.column_stack([np.ones(len(ei)), Zm[ei]]) @ bt
Cg = centre(WG); scw = np.full(NP_, np.nan)
for ch in sorted(set(chrom)):
    te = np.flatnonzero(chrom == ch); tr = np.flatnonzero(chrom != ch)
    trg = np.array(sorted(set(gene[tr]))); rs = np.random.default_rng(K.SEED)
    sh = trg.copy(); rs.shuffle(sh); ivg = set(sh[:int(round(0.25 * len(sh)))])
    iv = tr[np.array([g in ivg for g in gene[tr]])]
    it = tr[np.array([g not in ivg for g in gene[tr]])]
    gi = groups(iv); best = (-1.0, 1.0)
    for lam in K.LAMBDAS:
        w = K.winrate(fe(Cg, it, iv, lam), gi)
        if w > best[0]: best = (w, lam)
    scw[te] = fe(Cg, tr, te, best[1])
assert np.isfinite(scw).all()
hitc = np.array([float(scw[PAIR[g][0]].max() > scw[PAIR[g][1]].max()) for g in CONT])
hall = np.r_[hitc, np.ones(NUNC)]
def ci(h, nb=2000):
    bs = np.random.default_rng(K.SEED).integers(0, len(h), size=(nb, len(h)))
    v = h[bs].mean(1); return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
BASEWG = NUNC / NALL
lo_a, hi_a = ci(hall); lo_c, hi_c = ci(hitc)
log(f"[wg] contested {len(CONT)} genes: P@1 {hitc.mean():.4f} [{lo_c:.4f},{hi_c:.4f}]")
log(f"[wg] all {NALL} genes: baseline {BASEWG:.4f} -> model {hall.mean():.4f} "
    f"[{lo_a:.4f},{hi_a:.4f}]  lift {hall.mean()-BASEWG:+.4f}")

p1c = float(np.mean([1.0 / (1 + len(c)) for _, c in K.GRP_TE]))
with open("splice_model_26_metrics.tsv", "w") as fh:
    fh.write("metric\tscope\tvalue\tci_lo\tci_hi\treference\tn\n")
    fh.write(f"within_gene_winrate\ttest genes\t{R['winrate']:.5f}\t{R['winrate_lo']:.5f}\t"
             f"{R['winrate_hi']:.5f}\t0.50000\t{len(K.GRP_TE)}\n")
    fh.write(f"p_at_1\ttest genes\t{R['p_at_1']:.5f}\t{R['p_at_1_lo']:.5f}\t"
             f"{R['p_at_1_hi']:.5f}\t{p1c:.5f}\t{len(K.GRP_TE)}\n")
    fh.write(f"p_at_1_whole_genome\tall genes\t{hall.mean():.5f}\t{lo_a:.5f}\t{hi_a:.5f}\t"
             f"{BASEWG:.5f}\t{NALL}\n")
    fh.write(f"p_at_1_contested\tcontested genes\t{hitc.mean():.5f}\t{lo_c:.5f}\t{hi_c:.5f}\t"
             f"0.00000\t{len(CONT)}\n")
    fh.write(f"n_test_peaks\ttest genes\t{len(K.te)}\t\t\t\t{len(K.te)}\n")
    fh.write(f"n_uncontested\tall genes\t{NUNC}\t\t\t\t{NALL}\n")
log("[out] splice_model_26.npz splice_model_26_coefficients.tsv splice_model_26_metrics.tsv")
