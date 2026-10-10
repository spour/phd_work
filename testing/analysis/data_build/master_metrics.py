#!/usr/bin/env python3
# one table, every model, pooled and within-gene side by side, all on the same 1
import sys, csv, math
import numpy as np
import binned_common as K
from binned_common import log

UP, DN, BIN = 2000, 500, 100
NBIN = (UP + DN) // BIN
ACC = K.ACC
G = K.C0["gene"][K.ok]; CHR = K.C0["chrom"][K.ok]
CEN = K.C0["center"][K.ok].astype(np.int64); STR = K.C0["strand"][K.ok]
N = len(K.y)
NPOS = int(K.y[K.te].sum()); NTE = len(K.te); BASE = NPOS / NTE
log(f"[split] {len(K.GRP_TE)} test genes, {NTE} test peaks, {NPOS} real ends, "
    f"base rate {BASE:.5f}")

Z = np.load("binned_profiles.npz", allow_pickle=True)
P = Z["P"]; CHN = [str(c) for c in Z["channels"]]
FULL = np.column_stack([P.reshape(len(P), -1), ACC])

X = np.load("cnn_data.npz", allow_pickle=True)["X"]

import sys as _sys
_sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as L
_o9 = np.arange(UP + DN - 9 + 1) + 3 - UP
_b = (_o9 + UP) // BIN; _k = (_b >= 0) & (_b < NBIN); _bk = _b[_k]
_nb = np.bincount(_bk, minlength=NBIN).astype(float)
U1B = np.empty((N, 2 * NBIN))
for i in range(N):
    _t = X[i].astype(np.int64)
    _sw = np.lib.stride_tricks.sliding_window_view(_t, 9); _g = (_sw < 4).all(1)
    _u = np.nan_to_num(np.where(_g, L.U1[np.where(_g, _sw @ L.POW, 0)], np.nan), nan=-99.0)
    for _c, _thr in enumerate((4.0, 7.0)):
        _s = np.bincount(_bk, weights=(_u >= _thr)[_k].astype(float), minlength=NBIN)
        U1B[i, _c * NBIN:(_c + 1) * NBIN] = np.where(_nb > 0, _s / np.maximum(_nb, 1) * 1000.0,
                                                     np.nan)
_ref = np.column_stack([P[:, CHN.index("u1_dens4"), :], P[:, CHN.index("u1_dens7"), :]])
assert np.nanmax(np.abs(U1B.astype(np.float32) - _ref.astype(np.float32))) == 0.0, \
    "float64 rebinning does not match binned_profiles.npz in float32"
log(f"[u1] rebuilt in float64; matches binned_profiles.npz exactly in float32 "
    f"(float64 residual {np.nanmax(np.abs(U1B - _ref)):.2e})")
FIN = np.column_stack([U1B, ACC])
t = X.reshape(N, NBIN, BIN)
GCB = np.where((t < 4).sum(2) > 0, ((t == 1) | (t == 2)).sum(2) / np.maximum((t < 4).sum(2), 1), np.nan)
w = X[:, UP - 500:UP]
GCW = np.where((w < 4).sum(1) > 0,
               ((w == 1) | (w == 2)).sum(1) / np.maximum((w < 4).sum(1), 1), np.nan)
log(f"[gc] window GC over [-500, 0): mean {np.nanmean(GCW):.4f}")

DEPTH_COL = "depth_[-2000,-1000)_mean [-150,0)"
dp = {}
with open("gc_dip_depth.tsv") as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        if r["class"] != "decoy":
            dp[(r["gene"], r["chrom"], int(r["center"]), r["strand"])] = float(r[DEPTH_COL])
DIP = np.array([dp.get((G[i], CHR[i], int(CEN[i]), STR[i]), np.nan) for i in range(N)])
assert np.isfinite(DIP).all(), "dip-depth join incomplete"

ROWS = []
def emit(name, sc, nfeat, prov, expect=None, note="", ladder=True):
    r = K.suite(sc)
    if expect is not None:
        assert abs(r["winrate"] - expect) < 5e-4, \
            f"{name}: within-gene {r['winrate']:.4f} does not reproduce {expect:.4f}"
    ROWS.append(dict(name=name, p=nfeat, prov=prov, note=note, ladder=ladder, **r))
    log(f"[{prov:8s}] {name:34s} p={str(nfeat):>4s}  within-gene {r['winrate']:.4f} "
        f"[{r['winrate_lo']:.4f},{r['winrate_hi']:.4f}]  P@1 {r['p_at_1']:.4f}  "
        f"pooled AUROC {r['auroc']:.4f}  AUPRC {r['auprc']:.4f}"
        + (f"  (reproduces {expect:.4f})" if expect is not None else "  (no prior number)"))
    return r

def fit(M, name, nfeat, expect=None, note="", ladder=True, lam=None):
    if lam is None: lam, _ = K.pick_lambda(M, name)
    sc, _ = K.fit_score(M, K.trv, K.te, lam)
    return emit(name, sc, nfeat, "REFIT", expect, note, ladder)

# chance: win-rate 0.5 by definition, p@1 from the group sizes, auprc the base rate
p1c = float(np.mean([1.0 / (1 + len(c)) for _, c in K.GRP_TE]))
ROWS.append(dict(name="Chance", p=0, prov="ANALYTIC", note="win-rate 0.5 and pooled AUROC 0.5 by "
                 "definition; P@1 is the mean over test genes of 1/(1+competitors); pooled AUPRC "
                 "is the base rate", ladder=True,
                 winrate=0.5, winrate_lo=np.nan, winrate_hi=np.nan, p_at_1=p1c,
                 p_at_1_lo=np.nan, p_at_1_hi=np.nan, auroc=0.5, auroc_lo=np.nan, auroc_hi=np.nan,
                 auprc=BASE, auprc_lo=np.nan, auprc_hi=np.nan))
log(f"[ANALYTIC] {'Chance':34s} p=   0  within-gene 0.5000  P@1 {p1c:.4f}  "
    f"pooled AUROC 0.5000  AUPRC {BASE:.4f}")

# single features, fitted so the sign is learnt, not assumed
fit(ACC[:, None], "Acceptor 3'ss alone", 1, expect=0.6198)
fit(GCW[:, None], "Window GC alone [-500, 0)", 1, ladder=False,
    note="POOLED-vs-WITHIN-GENE ILLUSTRATION, not a ladder step. It scores higher within gene "
         "than pooled, in the expected direction, but the gap is small (+0.018); the boosted-tree "
         "row is the striking case, not this one")
fit(DIP[:, None], "GC dip depth alone", 1,
    note=f"{DEPTH_COL}, in each site's own frame")

# scalar splice model
from fastparquet import ParquetFile
_F = ParquetFile("shortlist_feat_all.parquet").to_pandas()
_SPL = [c for c in _F.columns if c.startswith("spl_")]
_key = {(g, c, int(ct), s_): i for i, (g, c, ct, s_) in
        enumerate(zip(_F.gene, _F.chrom, _F.center, _F.strand))}
_row = np.array([_key.get((G[i], CHR[i], int(CEN[i]), STR[i]), -1) for i in range(N)])
_X = np.full((N, len(_SPL)), np.nan); _h = _row >= 0
_X[_h] = _F.loc[_row[_h], _SPL].to_numpy(float)
S3 = [_SPL.index(c) for c in ("spl_pas_prox_u1", "spl_upstream_max_me3", "spl_comp_up_3ss")]
fit(_X[:, S3], "Scalar splice model", 3, expect=0.66557, lam=1.0)

# the locked model, re-scored from its saved parameters
M = np.load("splice_model_final.npz", allow_pickle=True)
beta, mu, sd, lam51 = M["beta"], M["mu"], M["sd"], float(M["lam"])
assert [str(x) for x in M["names"]][-1] == "spl_upstream_max_me3" and len(beta) == 52
Cg = K.centre_by_gene(FIN)
Zs = (np.where(np.isfinite(Cg), Cg, mu) - mu) / sd
sc51 = np.column_stack([np.ones(len(K.te)), Zs[K.te]]) @ beta
emit("Positional splice (locked)", sc51, 51, "RESCORED", expect=0.6957,
     note="scored from splice_model_final.npz (beta, mu, sd); not refitted")

fit(FULL, "Full binned splice", 126, expect=0.7101)
fit(np.column_stack([FIN, DIP]), "Positional splice + dip depth", 52, expect=0.7135)
fit(np.column_stack([FIN, GCB]), "Positional splice + GC per bin", 76, expect=0.7400,
    note="splice biology PLUS positional composition, a different axis from the rows above")

from sklearn.ensemble import HistGradientBoostingClassifier as HGB
Zf = K.zmat(FIN, K.trv)
hgb = HGB(learning_rate=0.06, max_leaf_nodes=31, max_iter=150, l2_regularization=1.0,
          min_samples_leaf=50, early_stopping=False, random_state=K.SEED)
hgb.fit(Zf[K.trv], K.y[K.trv])
emit("Boosted tree on the 51 splice features", hgb.predict_proba(Zf[K.te])[:, 1], 51, "REFIT",
     expect=0.6834, ladder=False,
     note="POOLED-vs-WITHIN-GENE ILLUSTRATION, not a ladder step: strong pooled, below the "
          "ridge within gene, because it reconstructs gene context that gene-centring removes. "
          "Also the only row sensitive to input storage precision: 0.6834 on float64 inputs, "
          "0.6797 if the same channels are read back from their float32 cache, while the ridge "
          "gives 0.6957 either way")

CS = np.load("cnn_model_scores.npz", allow_pickle=True)
assert (CS["gene"][K.ok] == K.gene).all() and (CS["y"][K.ok] == K.y).all()
emit("Sequence CNN", CS["logit"][K.ok][K.te].astype(float), "-", "SAVED", expect=0.82110,
     note="per-peak scores read from cnn_model_scores.npz; not refitted")

ORDER = ["Chance", "Acceptor 3'ss alone", "Window GC alone [-500, 0)", "GC dip depth alone",
         "Scalar splice model", "Positional splice (locked)", "Full binned splice",
         "Positional splice + dip depth", "Positional splice + GC per bin",
         "Boosted tree on the 51 splice features", "Sequence CNN"]
BY = {r["name"]: r for r in ROWS}
assert set(ORDER) == set(BY), set(ORDER) ^ set(BY)
def f(v, n=4): return "" if v is None or not np.isfinite(v) else f"{v:.{n}f}"
with open("master_model_metrics.tsv", "w") as fh:
    fh.write("model\tn_features\twithin_gene_winrate\twinrate_ci_lo\twinrate_ci_hi\tp_at_1\t"
             "pooled_auroc\tpooled_auprc\tladder_step\trefit_flag\tn_genes\tn_peaks\tnotes\n")
    for nm in ORDER:
        r = BY[nm]
        fh.write(f"{nm}\t{r['p']}\t{f(r['winrate'])}\t{f(r['winrate_lo'])}\t{f(r['winrate_hi'])}\t"
                 f"{f(r['p_at_1'])}\t{f(r['auroc'])}\t{f(r['auprc'])}\t"
                 f"{'yes' if r['ladder'] else 'no'}\t{r['prov']}\t{len(K.GRP_TE)}\t{NTE}\t"
                 f"{r['note']}\n")

CAP = (f"All models scored on the same {len(K.GRP_TE):,} held-out test genes "
       f"({NTE:,} peaks, {NPOS:,} real 3' ends, base rate {BASE:.3f}) over the window "
       f"[-2000, +500]. Every metric is computed by the same code (binned_common.suite), so "
       f"the pooled and within-gene columns are comparable across rows. **The within-gene "
       f"win-rate is the headline**; pooled AUROC and AUPRC are shown for completeness. Two "
       f"rows are marked as illustrations rather than ladder steps. Window GC scores higher "
       f"within gene ({BY['Window GC alone [-500, 0)']['winrate']:.3f}) than pooled "
       f"({BY['Window GC alone [-500, 0)']['auroc']:.3f}), though the gap is modest; the "
       f"boosted tree is the striking case, strong pooled "
       f"({BY['Boosted tree on the 51 splice features']['auroc']:.3f}) but below the ridge "
       f"within gene ({BY['Boosted tree on the 51 splice features']['winrate']:.3f}). "
       f"{BY['Positional splice (locked)']['winrate']:.3f} is the splice-only ceiling; "
       f"{BY['Positional splice + GC per bin']['winrate']:.3f} is splice plus positional "
       f"composition, which is a different axis.")
with open("master_model_metrics.md", "w") as fh:
    fh.write("# Model comparison: within-gene and pooled\n\n")
    fh.write("| Model | Features | Within-gene win-rate [95% CI] | P@1 | Pooled AUROC | "
             "Pooled AUPRC | Source |\n|---|---:|---|---:|---:|---:|---|\n")
    for nm in ORDER:
        r = BY[nm]
        ci = "" if not np.isfinite(r["winrate_lo"]) else \
             f" [{r['winrate_lo']:.3f}, {r['winrate_hi']:.3f}]"
        star = "" if r["ladder"] else " *"
        fh.write(f"| {nm}{star} | {r['p']} | **{r['winrate']:.3f}**{ci} | {r['p_at_1']:.3f} | "
                 f"{r['auroc']:.3f} | {r['auprc']:.3f} | {r['prov'].lower()} |\n")
    fh.write(f"\n\\* not a ladder step; see caption.\n\n{CAP}\n")

log("")
gw, gb = BY["Window GC alone [-500, 0)"], BY["Boosted tree on the 51 splice features"]
log(f"[contrast] window GC   pooled AUROC {gw['auroc']:.4f}, within-gene {gw['winrate']:.4f}: "
    f"the within-gene number is the larger, but only by {gw['winrate']-gw['auroc']:+.4f}. The "
    f"expected near-chance pooled value (~0.56) does NOT reproduce with this definition of "
    f"window GC ([-500, 0) transcript-oriented, fitted 1-feature logistic); the direction of "
    f"the contrast holds, its size does not")
log(f"[contrast] boosted tree pooled AUROC {gb['auroc']:.4f} (strong) but within-gene "
    f"{gb['winrate']:.4f}, BELOW the ridge's "
    f"{BY['Positional splice (locked)']['winrate']:.4f}")
log(f"[provenance] " + "  ".join(
    f"{p}={sum(1 for r in ROWS if r['prov'] == p)}"
    for p in ("SAVED", "RESCORED", "REFIT", "ANALYTIC")))
log(f"[check] every model with a prior reported within-gene number reproduced it to 5e-4")
log(f"[out] master_model_metrics.tsv master_model_metrics.md")
