#!/usr/bin/env python3
# genome-wide p@1 resolved by gene length, for the baseline
import sys, csv, time
import numpy as np
import binned_common as K
from binned_common import log
sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as L

UP, DN, BIN, NBIN = 2000, 500, 100, 25
GTF = "../train/data/annotation/gencode.v49.annotation.gtf"
SEED, NB = 42, 2000

# genes and candidates
LAB = list(csv.DictReader(open("cpa_peaks_labeled.tsv"), delimiter="\t"))
tb = lambda x: x in ("True", "true", "1")
ncand, ncomp = {}, {}
for r in LAB:
    g = r["gene"]
    ncand[g] = ncand.get(g, 0) + 1
    ncomp[g] = ncomp.get(g, 0) + (not tb(r["is_canonical"]))
ALLG = np.array(sorted(ncand)); NALL = len(ALLG)
UNC = np.array([ncomp[g] == 0 for g in ALLG])
log(f"[genes] {NALL} genes, {int(UNC.sum())} uncontested, {int((~UNC).sum())} contested; "
    f"baseline P@1 = {UNC.mean():.5f}")
assert NALL == 15560 and int(UNC.sum()) == 8977

# contested candidates
M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
gene = np.array([r["gene"] for r in M]); chrom = np.array([r["chrom"] for r in M])
cen = np.array([int(r["center"]) for r in M]); strand = np.array([r["strand"] for r in M])
yy = np.array([int(float(r["is_canonical"])) for r in M]); NPK = len(M)
CONT = np.array(sorted(set(gene)))
log(f"[wg] {NPK} contested candidates in {len(CONT)} genes")

TOK = np.load("cnn_wg_tokens.npy")
assert TOK.shape == (NPK, UP + DN)
o9 = np.arange(UP + DN - 9 + 1) + 3 - UP
bb = (o9 + UP) // BIN; kk = (bb >= 0) & (bb < NBIN); bk = bb[kk]
nb9 = np.bincount(bk, minlength=NBIN).astype(float)
t0 = time.time(); D4 = np.empty((NPK, NBIN))
for i in range(NPK):
    t = np.where(TOK[i] == 255, 4, TOK[i]).astype(np.int64)
    sw = np.lib.stride_tricks.sliding_window_view(t, 9); g = (sw < 4).all(1)
    u = np.nan_to_num(np.where(g, L.U1[np.where(g, sw @ L.POW, 0)], np.nan), nan=-99.0)
    s = np.bincount(bk, weights=(u >= 4.0)[kk].astype(float), minlength=NBIN)
    D4[i] = np.where(nb9 > 0, s / np.maximum(nb9, 1) * 1000.0, np.nan)
log(f"[feat] u1_dens4 for {NPK} candidates in {time.time()-t0:.0f}s")
from fastparquet import ParquetFile
F = ParquetFile("shortlist_feat_all.parquet").to_pandas()
key = {(g, c, int(ct), s): i for i, (g, c, ct, s) in
       enumerate(zip(F.gene, F.chrom, F.center, F.strand))}
rw = np.array([key[(gene[i], chrom[i], int(cen[i]), strand[i])] for i in range(NPK)])
WG = np.column_stack([D4, F["spl_upstream_max_me3"].to_numpy(float)[rw]])
assert WG.shape[1] == 26

# splice loco, per gene
def groups(ii):
    by = {}
    for k_, g in enumerate(gene[ii]):
        r, c = by.setdefault(g, ([], []))
        (r if yy[ii][k_] == 1 else c).append(k_)
    return [(np.array(r), np.array(c)) for r, c in by.values() if r and c]
def centre_by(Mx, g_):
    out = Mx.astype(float).copy()
    o = np.argsort(g_, kind="stable"); gs = g_[o]
    cut = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1, len(gs)]
    for j in range(len(cut) - 1):
        ix = o[cut[j]:cut[j + 1]]
        mu = np.nanmean(out[ix], 0)
        out[ix] = out[ix] - np.where(np.isfinite(mu), mu, 0.0)
    return out
Cg = centre_by(WG, gene)
SPL = np.full(NPK, np.nan)
for ch in sorted(set(chrom)):
    te = np.flatnonzero(chrom == ch); tr = np.flatnonzero(chrom != ch)
    trg = np.array(sorted(set(gene[tr]))); rs = np.random.default_rng(K.SEED)
    sh = trg.copy(); rs.shuffle(sh); ivg = set(sh[:int(round(0.25 * len(sh)))])
    iv = tr[np.array([g in ivg for g in gene[tr]])]
    it = tr[np.array([g not in ivg for g in gene[tr]])]
    gi = groups(iv); best = (-1.0, 1.0)
    for lam in K.LAMBDAS:
        m = np.nanmean(Cg[it], 0); m = np.where(np.isfinite(m), m, 0.0)
        s = np.nanstd(Cg[it], 0); s[~np.isfinite(s) | (s == 0)] = 1.0
        Z = (np.where(np.isfinite(Cg), Cg, m) - m) / s
        b = K.fit_ridge(np.column_stack([np.ones(len(it)), Z[it]]), yy[it].astype(float), lam)
        w = K.winrate(np.column_stack([np.ones(len(iv)), Z[iv]]) @ b, gi)
        if w > best[0]: best = (w, lam)
    m = np.nanmean(Cg[tr], 0); m = np.where(np.isfinite(m), m, 0.0)
    s = np.nanstd(Cg[tr], 0); s[~np.isfinite(s) | (s == 0)] = 1.0
    Z = (np.where(np.isfinite(Cg), Cg, m) - m) / s
    b = K.fit_ridge(np.column_stack([np.ones(len(tr)), Z[tr]]), yy[tr].astype(float), best[1])
    SPL[te] = np.column_stack([np.ones(len(te)), Z[te]]) @ b
assert np.isfinite(SPL).all()
CNN = np.load("cnn_wg_scores.npz", allow_pickle=True)
assert (CNN["gene"] == gene).all()
CNNS = CNN["logit"].astype(float)

GI = {g: np.flatnonzero(gene == g) for g in CONT}
PAIR = {g: (ix[yy[ix] == 1], ix[yy[ix] == 0]) for g, ix in GI.items()}
hit_s = {g: float((SPL[c] >= SPL[r[0]]).sum() == 0) for g, (r, c) in PAIR.items()}
hit_c = {g: float((CNNS[c] >= CNNS[r[0]]).sum() == 0) for g, (r, c) in PAIR.items()}
sc_, cc_ = np.mean(list(hit_s.values())), np.mean(list(hit_c.values()))
log(f"[contested] splice P@1 {sc_:.5f} (committed 0.40270)   CNN P@1 {cc_:.5f} (committed 0.57724)")
assert abs(sc_ - 0.40270) < 3e-3, f"splice contested P@1 {sc_:.5f} does not reproduce 0.40270"
assert abs(cc_ - 0.57724) < 1e-5

# gene spans
GLEN = {}
t0 = time.time()
with open(GTF) as fh:
    for line in fh:
        if line[0] == "#": continue
        f = line.split("\t", 9)
        if f[2] != "gene": continue
        a = f[8]; i = a.find('gene_name "')
        if i < 0: continue
        nm = a[i + 11:a.index('"', i + 11)]
        Ln = int(f[4]) - int(f[3])
        if Ln > GLEN.get(nm, 0): GLEN[nm] = Ln
log(f"[gtf] spans for {len(GLEN):,} gene names in {time.time()-t0:.0f}s")

BASE = UNC.astype(float)
SPH = np.array([1.0 if UNC[i] else hit_s.get(ALLG[i], np.nan) for i in range(NALL)])
CNH = np.array([1.0 if UNC[i] else hit_c.get(ALLG[i], np.nan) for i in range(NALL)])
LEN = np.array([GLEN.get(g, np.nan) for g in ALLG], float)
assert np.isfinite(SPH).all() and np.isfinite(CNH).all()
log(f"[wg] genome-wide P@1  baseline {BASE.mean():.5f}  splice {SPH.mean():.5f}  "
    f"CNN {CNH.mean():.5f}")
assert abs(SPH.mean() - 0.74730) < 3e-3 and abs(CNH.mean() - 0.82114) < 1e-5
nolen = int(np.isnan(LEN).sum())
log(f"[gtf] {nolen} of {NALL} genes have no GENCODE span and are dropped from the length panel "
    f"({100*nolen/NALL:.1f}%)")

with open("wg_per_gene.tsv", "w") as fh:
    fh.write("gene\tn_candidates\tuncontested\tgene_length\tbaseline\tsplice\tcnn\n")
    for i in range(NALL):
        fh.write(f"{ALLG[i]}\t{ncand[ALLG[i]]}\t{int(UNC[i])}\t"
                 f"{'' if not np.isfinite(LEN[i]) else int(LEN[i])}\t"
                 f"{BASE[i]:.0f}\t{SPH[i]:.0f}\t{CNH[i]:.0f}\n")
log("[out] wg_per_gene.tsv")

ok = np.isfinite(LEN)
EDG = np.array([0, 5, 10, 20, 40, 80, 160, 320, 640, 1e9]) * 1000.0
rs = np.random.default_rng(SEED)
with open("wg_by_length.tsv", "w") as fh:
    fh.write("bin\tlo_kb\thi_kb\tlabel\tn_genes\tmedian_candidates\tmodel\tp_at_1\tci_lo\tci_hi\n")
    log(f"\n{'gene length':>14} {'n':>6} {'cand':>5}  {'baseline':>9} {'splice':>9} {'CNN':>9}")
    for j in range(len(EDG) - 1):
        m = ok & (LEN >= EDG[j]) & (LEN < EDG[j + 1])
        if m.sum() < 30: continue
        lab = (f"{EDG[j]/1000:.0f}-{EDG[j+1]/1000:.0f}" if EDG[j + 1] < 1e8
               else f">{EDG[j]/1000:.0f}")
        mc = np.median([ncand[g] for g in ALLG[m]])
        vals = []
        for nm, v in (("Baseline", BASE[m]), ("Splice model (26 features)", SPH[m]),
                      ("Sequence CNN", CNH[m])):
            bs = rs.integers(0, m.sum(), size=(NB, m.sum()))
            lo, hi = np.percentile(v[bs].mean(1), [2.5, 97.5])
            fh.write(f"{j}\t{EDG[j]/1000:.0f}\t{EDG[j+1]/1000:.0f}\t{lab} kb\t{int(m.sum())}\t"
                     f"{mc:.0f}\t{nm}\t{v.mean():.5f}\t{lo:.5f}\t{hi:.5f}\n")
            vals.append(v.mean())
        log(f"{lab+' kb':>14} {int(m.sum()):>6} {mc:>5.0f}  "
            + "  ".join(f"{x:>9.3f}" for x in vals))
log("[out] wg_by_length.tsv")
