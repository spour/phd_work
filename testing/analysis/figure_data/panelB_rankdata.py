#!/usr/bin/env python3
# per-gene rank of the true 3' end under the 26-feature model
import sys, csv, time
import numpy as np
import binned_common as K
from binned_common import log
sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as L

UP, DN, BIN, PAD = 2000, 500, 100, 40
NBIN = (UP + DN) // BIN
CODE = np.full(256, 4, np.int64)
for i, ch in enumerate("ACGT"): CODE[ord(ch)] = i
o9 = np.arange(UP + DN - 9 + 1) + 3 - UP
bb = (o9 + UP) // BIN; kk = (bb >= 0) & (bb < NBIN); bk = bb[kk]
nb9 = np.bincount(bk, minlength=NBIN).astype(float)

LAB = list(csv.DictReader(open("cpa_peaks_labeled.tsv"), delimiter="\t"))
tb = lambda x: x in ("True", "true", "1")
gcomp = {}
for r in LAB: gcomp[r["gene"]] = gcomp.get(r["gene"], 0) + (not tb(r["is_canonical"]))
UNCG = sorted(g for g, v in gcomp.items() if v == 0)
NALL = len(gcomp)
log(f"[set] {NALL} genes: {len(UNCG)} uncontested (1 candidate, rank 1 for any scorer) + "
    f"{NALL-len(UNCG)} contested")

M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
gene = np.array([r["gene"] for r in M]); chrom = np.array([r["chrom"] for r in M])
cen = np.array([int(r["center"]) for r in M]); strand = np.array([r["strand"] for r in M])
y = np.array([int(float(r["is_canonical"])) for r in M]); NP_ = len(M)
from fastparquet import ParquetFile
F = ParquetFile("shortlist_feat_all.parquet").to_pandas()
key = {(g, c, int(ct), s): i for i, (g, c, ct, s) in
       enumerate(zip(F.gene, F.chrom, F.center, F.strand))}
rw = np.array([key[(gene[i], chrom[i], int(cen[i]), strand[i])] for i in range(NP_)])
ACC = F["spl_upstream_max_me3"].to_numpy(float)[rw]

G = L.G_["G"]; t0 = time.time()
D4 = np.empty((NP_, NBIN))
for i in range(NP_):
    c = int(cen[i]); st = strand[i]
    a, b = (c - UP - PAD, c + DN + PAD) if st == "+" else (c - DN - PAD + 1, c + UP + PAD + 1)
    s = G.fetch(chrom[i], a, b, st)
    t = CODE[np.frombuffer(s.encode("ascii"), np.uint8)][PAD:PAD + UP + DN].astype(np.int64)
    sw = np.lib.stride_tricks.sliding_window_view(t, 9); g = (sw < 4).all(1)
    u = np.nan_to_num(np.where(g, L.U1[np.where(g, sw @ L.POW, 0)], np.nan), nan=-99.0)
    ss = np.bincount(bk, weights=(u >= 4.0)[kk].astype(float), minlength=NBIN)
    D4[i] = np.where(nb9 > 0, ss / np.maximum(nb9, 1) * 1000.0, np.nan)
    if (i + 1) % 20000 == 0: log(f"  scan {i+1}/{NP_}  {time.time()-t0:.0f}s")
log(f"[scan] {NP_} contested-gene peaks in {time.time()-t0:.0f}s")
X = np.column_stack([D4, ACC]); assert X.shape[1] == 26

def groups(ii):
    by = {}
    for k_, g in enumerate(gene[ii]):
        r, c = by.setdefault(g, ([], []))
        (r if y[ii][k_] == 1 else c).append(k_)
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
    bt = K.fit_ridge(np.column_stack([np.ones(len(fi)), Zm[fi]]), y[fi].astype(float), lam)
    return np.column_stack([np.ones(len(ei)), Zm[ei]]) @ bt
Cg = centre(X); sc = np.full(NP_, np.nan)
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
    sc[te] = fe(Cg, tr, te, best[1])
assert np.isfinite(sc).all()
log("[loco] leave-one-chromosome-out complete, lambda retuned per fold")

CONT = np.array(sorted(set(gene)))
GI = {g: np.flatnonzero(gene == g) for g in CONT}
rows = []
for g in CONT:
    ix = GI[g]; r = ix[y[ix] == 1][0]; c = ix[y[ix] == 0]
    rank = 1 + int((sc[c] >= sc[r]).sum())
    n = len(c) + 1
    rows.append((g, chrom[r], n, rank, (n - rank) / (n - 1) if n > 1 else 1.0, ACC[r], 1))
for g in UNCG:
    rows.append((g, "", 1, 1, 1.0, np.nan, 0))
log(f"[rank] ranks computed for {len(CONT)} contested genes; {len(UNCG)} uncontested added "
    f"at rank 1 with a single candidate")
with open("panelB_gene_ranks.tsv", "w") as fh:
    fh.write("gene\tchrom\tn_candidates\trank\tpercentile\tacceptor\tcontested\n")
    for r in rows:
        fh.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]:.6f}\t"
                 f"{'' if not np.isfinite(r[5]) else f'{r[5]:.5f}'}\t{r[6]}\n")

rk = np.array([r[3] for r in rows]); nc = np.array([r[2] for r in rows])
ct = np.array([r[6] for r in rows], bool)
def dist(mask, cap=6):
    v = rk[mask]; out = [float((v == k).mean()) for k in range(1, cap)]
    out.append(float((v >= cap).mean())); return out
def chance_dist(mask, cap=6):
    n = nc[mask].astype(float); out = []
    for k in range(1, cap): out.append(float(np.mean(np.where(n >= k, 1.0 / n, 0.0))))
    out.append(float(np.mean(np.where(n >= cap, (n - cap + 1) / n, 0.0)))); return out
def cum(mask, K_=10):
    v = rk[mask]; return [float((v <= k).mean()) for k in range(1, K_ + 1)]
def chance_cum(mask, K_=10):
    n = nc[mask].astype(float)
    return [float(np.mean(np.minimum(k, n) / n)) for k in range(1, K_ + 1)]
LBL = ["1", "2", "3", "4", "5", "6+"]
with open("panelB_rank_summary.tsv", "w") as fh:
    fh.write("scope\tkind\tk\tvalue\tn_genes\n")
    for nm, m in (("contested", ct), ("all genes", np.ones(len(rk), bool))):
        d = dist(m); cd = chance_dist(m)
        log(f"[dist] {nm:10s} n={int(m.sum()):>5d}  " +
            "  ".join(f"r{L}={v:.3f}" for L, v in zip(LBL, d)))
        log(f"[dist] {nm:10s} chance      " + "  ".join(f"r{L}={v:.3f}"
                                                        for L, v in zip(LBL, cd)))
        for L_, v in zip(LBL, d): fh.write(f"{nm}\tdist\t{L_}\t{v:.6f}\t{int(m.sum())}\n")
        for L_, v in zip(LBL, cd): fh.write(f"{nm}\tdist_chance\t{L_}\t{v:.6f}\t{int(m.sum())}\n")
        for k, v in enumerate(cum(m), 1): fh.write(f"{nm}\tcum\t{k}\t{v:.6f}\t{int(m.sum())}\n")
        for k, v in enumerate(chance_cum(m), 1):
            fh.write(f"{nm}\tcum_chance\t{k}\t{v:.6f}\t{int(m.sum())}\n")
    MT = {r["metric"]: r for r in csv.DictReader(open("splice_model_26_metrics.tsv"),
                                                 delimiter="\t")}
    for k in ("within_gene_winrate", "p_at_1", "p_at_1_whole_genome"):
        r = MT[k]
        fh.write(f"headline\t{k}\tvalue\t{r['value']}\t{r['n']}\n")
        fh.write(f"headline\t{k}\treference\t{r['reference']}\t{r['n']}\n")
        fh.write(f"headline\t{k}\tci_lo\t{r['ci_lo']}\t{r['n']}\n")
        fh.write(f"headline\t{k}\tci_hi\t{r['ci_hi']}\t{r['n']}\n")
log(f"[check] contested rank-1 fraction {dist(ct)[0]:.4f} against the LOCO P@1 0.4027")
log(f"[check] all-genes rank-1 fraction {dist(np.ones(len(rk),bool))[0]:.4f} against 0.7473")
log("[out] panelB_gene_ranks.tsv panelB_rank_summary.tsv")
