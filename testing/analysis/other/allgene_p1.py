#!/usr/bin/env python3
# p@1 over all 4,847 analysis-set genes, out of fold, and the rescue rate over the baseline
import sys
import numpy as np
import binned_common as K
from binned_common import log
sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as L

UP, DN, BIN = 2000, 500, 100
NBIN = (UP + DN) // BIN
ACC = K.ACC; N = len(K.y)
GENE = K.gene; CHROM = K.C0["chrom"][K.ok]
X = np.load("cnn_data.npz", allow_pickle=True)["X"]
rng = np.random.default_rng(K.SEED)

t = X.reshape(N, NBIN, BIN); nok = (t < 4).sum(2).astype(float)
GCB = np.where(nok > 0, ((t == 1) | (t == 2)).sum(2) / np.maximum(nok, 1), np.nan)
o9 = np.arange(UP + DN - 9 + 1) + 3 - UP
b = (o9 + UP) // BIN; k = (b >= 0) & (b < NBIN); bk = b[k]
nb = np.bincount(bk, minlength=NBIN).astype(float)
U1B = np.empty((N, 2 * NBIN))
for i in range(N):
    ti = X[i].astype(np.int64)
    sw = np.lib.stride_tricks.sliding_window_view(ti, 9); g = (sw < 4).all(1)
    u = np.nan_to_num(np.where(g, L.U1[np.where(g, sw @ L.POW, 0)], np.nan), nan=-99.0)
    for c, thr in enumerate((4.0, 7.0)):
        s = np.bincount(bk, weights=(u >= thr)[k].astype(float), minlength=NBIN)
        U1B[i, c * NBIN:(c + 1) * NBIN] = np.where(nb > 0, s / np.maximum(nb, 1) * 1000.0, np.nan)
Zc = np.load("binned_profiles.npz", allow_pickle=True)
_P = Zc["P"]; _C = [str(c) for c in Zc["channels"]]
assert np.nanmax(np.abs(U1B.astype(np.float32) -
    np.column_stack([_P[:, _C.index("u1_dens4"), :],
                     _P[:, _C.index("u1_dens7"), :]]).astype(np.float32))) == 0.0
MODELS = [("Acceptor 3'ss alone", ACC[:, None], 1),
          ("Positional splice (51)", np.column_stack([U1B, ACC]), 51),
          ("Splice + GC per bin (76)", np.column_stack([GCB, ACC, U1B]), 76)]

# baseline: verify the construction rather than trust it
from fastparquet import ParquetFile
F = ParquetFile("shortlist_feat_all.parquet").to_pandas()
CE = K.C0["center"][K.ok].astype(int); ST = K.C0["strand"][K.ok]
key = {(g, c, int(ct), s): i for i, (g, c, ct, s) in
       enumerate(zip(F.gene, F.chrom, F.center, F.strand))}
brow = np.array([key[(GENE[i], CHROM[i], int(CE[i]), ST[i])] for i in range(N)])
GENES = np.array(sorted(set(GENE)))
GIDX = {g: np.flatnonzero(GENE == g) for g in GENES}
PAIR = {g: (ix[K.y[ix] == 1], ix[K.y[ix] == 0]) for g, ix in GIDX.items()}
PAIR = {g: v for g, v in PAIR.items() if len(v[0]) and len(v[1])}
GL = np.array(sorted(PAIR))
log(f"[set] {len(GL)} analysis-set genes with a real end and >=1 competitor, "
    f"{N} peaks, {len(set(CHROM))} chromosomes")
for col in ("rf_prob", "lr_prob", "cnn_prob"):
    v = F[col].to_numpy(float)[brow]
    top = np.array([float(v[PAIR[g][0]].max() > v[PAIR[g][1]].max()) for g in GL])
    log(f"[baseline] {col:9s} strict top-1 P@1 over all genes = {top.mean():.4f} "
        f"({int(top.sum())} of {len(GL)} genes)")
    assert top.sum() == 0, f"{col} is not 0 by construction"
log("[baseline] all three baselines rank a competitor at or above the true end in EVERY gene,")
log("[baseline] which is the definition of the set. Baseline P@1 = 0, so rescue rate == P@1")
log("[baseline] and the break rate has an empty denominator (no gene the baseline got right).\n")

def top1(sc, genes, off):
    hit = np.empty(len(genes)); tie = 0
    for j, g in enumerate(genes):
        r, c = PAIR[g]
        m = sc[off[r]].max(); n = sc[off[c]].max()
        hit[j] = float(m > n); tie += int(m == n)
    return hit, tie

def ci(hit, nboot=2000):
    bs = np.random.default_rng(K.SEED).integers(0, len(hit), size=(nboot, len(hit)))
    v = hit[bs].mean(1)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

# loco over all genes
CHS = sorted(set(CHROM))
ROWS = []
for name, M, p in MODELS:
    sc = np.full(N, np.nan)
    for ch in CHS:
        te = np.flatnonzero(CHROM == ch); tr = np.flatnonzero(CHROM != ch)
        trg = np.array(sorted(set(GENE[tr]))); rs = np.random.default_rng(K.SEED)
        sh = trg.copy(); rs.shuffle(sh)
        ivg = set(sh[:int(round(0.25 * len(sh)))])
        iv = tr[np.array([g in ivg for g in GENE[tr]])]
        it = tr[np.array([g not in ivg for g in GENE[tr]])]
        gi = K.make_groups(iv)
        best = (-1.0, 1.0)
        for lam in K.LAMBDAS:
            s, _ = K.fit_score(M, it, iv, lam)
            w = K.winrate(s, gi)
            if w > best[0]: best = (w, lam)
        s, _ = K.fit_score(M, tr, te, best[1])
        sc[te] = s
    assert np.isfinite(sc).all(), "LOCO left peaks unscored"
    off = np.arange(N)
    hit, tie = top1(sc, GL, off)
    lo, hi = ci(hit)
    ROWS.append(("all genes (LOCO)", name, p, len(GL), hit.mean(), lo, hi, tie))
    log(f"[all]  {name:26s} p={p:>3d}  P@1 {hit.mean():.4f} [{lo:.4f},{hi:.4f}] "
        f"over {len(GL)} genes, {tie} ties")

# the 1,623 held-out genes, where the cnn can be compared like for like
TEG = np.array(sorted(set(GENE[K.te])))
offte = np.full(N, -1); offte[K.te] = np.arange(len(K.te))
log("")
for name, M, p in MODELS:
    s, _ = K.fit_score(M, K.trv, K.te, 0.1)
    hit, tie = top1(s, TEG, offte)
    lo, hi = ci(hit)
    ROWS.append(("held-out 1,623 genes", name, p, len(TEG), hit.mean(), lo, hi, tie))
    log(f"[test] {name:26s} p={p:>3d}  P@1 {hit.mean():.4f} [{lo:.4f},{hi:.4f}] "
        f"over {len(TEG)} genes, {tie} ties")
CS = np.load("cnn_model_scores.npz", allow_pickle=True)
assert (CS["gene"][K.ok] == GENE).all() and (CS["y"][K.ok] == K.y).all()
hit, tie = top1(CS["logit"][K.ok][K.te].astype(float), TEG, offte)
lo, hi = ci(hit)
ROWS.append(("held-out 1,623 genes", "Sequence CNN", "-", len(TEG), hit.mean(), lo, hi, tie))
log(f"[test] {'Sequence CNN':26s} p=  -   P@1 {hit.mean():.4f} [{lo:.4f},{hi:.4f}] "
    f"over {len(TEG)} genes, {tie} ties")

with open("allgene_p1.tsv", "w") as fh:
    fh.write("denominator\tmodel\tn_features\tn_genes\tp_at_1_strict\tci_lo\tci_hi\tties\t"
             "baseline_p_at_1\trescue_rate\tbreak_rate\toof_scheme\n")
    for d, nm, p, ng, v, lo, hi, tie in ROWS:
        sch = "leave-one-chromosome-out, lambda retuned per fold" if d.startswith("all") \
              else ("saved train/test fit" if nm == "Sequence CNN" else
                    "train+val fit, held-out test genes")
        fh.write(f"{d}\t{nm}\t{p}\t{ng}\t{v:.5f}\t{lo:.5f}\t{hi:.5f}\t{tie}\t0.00000\t"
                 f"{v:.5f}\tundefined (no gene the baseline ranked first)\t{sch}\n")
log("")
log("[rescue] baseline P@1 = 0 on every analysis-set gene, so for each model the rescue rate")
log("[rescue] over the baseline is numerically its own P@1, and the break rate is 0 out of 0.")
log("[out] allgene_p1.tsv")
