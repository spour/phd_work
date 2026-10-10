#!/usr/bin/env python3
# per-site test scores for the four models in figure 1b
import sys
import numpy as np
from fastparquet import ParquetFile, write as pq_write
import pandas as pd

def log(*a): print(*a, file=sys.stderr, flush=True)

D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
split = D["split"][ok]; y = D["y"][ok].astype(float); gene = D["gene"][ok]
gc1 = D["window_gc"][ok]
C = np.load("analysis_columns.npz", allow_pickle=True)
chrom = C["chrom"][ok]; center = C["center"][ok]; strand = C["strand"][ok]
acc = C["spl_upstream_max_me3"][ok]
assert (C["is_canonical"][ok] == (y == 1)).all(), "row alignment"

S = np.load("cnn_model_scores.npz", allow_pickle=True)
assert len(S["logit"]) == len(y) and (S["y"] == y).all() and (S["split"] == split).all(), \
    "cnn_model_scores.npz is not aligned with cnn_data.npz"
cnn = S["logit"].astype(float)
log(f"[loaded]   sequence CNN logits <- cnn_model_scores.npz ({len(cnn)} peaks; "
    f"NOT cnn_interp*_test_scored.parquet, which are the old two-branch model)")

# splice model: apply the already fitted coefficients
F = ParquetFile("shortlist_feat_all.parquet").to_pandas()
FIT = np.load("splice_model_fit.npz", allow_pickle=True)
SEL = [str(x) for x in FIT["sel_names"]]; BETA = FIT["beta"]; MU = FIT["mu"]; SD = FIT["sd"]
key = {(g, c, int(ct), s): i for i, (g, c, ct, s) in
       enumerate(zip(F.gene, F.chrom, F.center, F.strand))}
row = np.array([key.get((g, c, int(ct), s), -1) for g, c, ct, s in
                zip(gene, chrom, center, strand)])
assert (row >= 0).all(), "some analysis peaks are missing from shortlist_feat_all"
X = F.loc[row, SEL].to_numpy(float)

def centre_by_gene(M, g):
    out = M.copy(); order = np.argsort(g, kind="stable"); gs = g[order]
    cut = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1, len(gs)]
    for j in range(len(cut) - 1):
        idx = order[cut[j]:cut[j+1]]
        mu = np.nanmean(out[idx], 0)
        out[idx] = out[idx] - np.where(np.isfinite(mu), mu, 0.0)
    return out
Cg = centre_by_gene(X, gene)
Z = (np.where(np.isfinite(Cg), Cg, MU) - MU) / SD
spl = np.column_stack([np.ones(len(Z)), Z]) @ BETA
log(f"[re-scored] splice features (no GC) <- splice_model_fit.npz coefficients "
    f"{np.round(BETA,4)} on {SEL} (not refitted, not reselected)")
log(f"[re-scored] acceptor 3'ss alone  <- spl_upstream_max_me3, raw")
log(f"[re-scored] window GC alone      <- -window_gc, the sign convention cnn_eval.py used")
gcs = -gc1

te = np.flatnonzero(split == "test")
by = {}
for k, g in enumerate(gene[te]):
    r, c = by.setdefault(g, ([], []))
    (r if y[te][k] == 1 else c).append(k)
GRP = [(np.array(r), np.array(c)) for r, c in by.values() if r and c]
def auroc(pos, neg):
    a = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); rk = ((cs - cnt) + cs + 1) / 2.0
    return float((rk[inv][:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))
def auprc(pos, neg):
    s = np.concatenate([pos, neg]); yy = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    o = np.argsort(-s, kind="stable"); yy = yy[o]; s = s[o]
    end = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s)-1]
    tp = np.cumsum(yy)[end]; prec = tp/(end+1); rec = tp/len(pos)
    return float(np.sum(prec * np.diff(np.r_[0.0, rec])))
def winrate(s):
    return float(np.mean([((s[r][:, None] > s[c][None, :]).sum()
                           + 0.5*(s[r][:, None] == s[c][None, :]).sum())/(len(r)*len(c))
                          for r, c in GRP]))

def ref(path, score, metric):
    for ln in open(path):
        f = ln.rstrip("\n").split("\t")
        if len(f) > 2 and f[0] == score and f[1] == metric: return float(f[2])
    raise KeyError(f"{score}/{metric} not in {path}")

CHECK = [("sequence CNN", cnn, ("cnn_metrics.tsv", "full")),
         ("splice features (no GC)", spl, ("splice_feature_metrics.tsv", "splice_selected")),
         ("acceptor 3'ss", acc, ("cnn_metrics.tsv", "acceptor_alone")),
         ("window GC", gcs, ("cnn_metrics.tsv", "window_gc_alone"))]
for nm, sc, (path, key_) in CHECK:
    s = np.asarray(sc, float)[te]
    got = dict(winrate=winrate(s), auroc=auroc(s[y[te] == 1], s[y[te] == 0]),
               auprc=auprc(s[y[te] == 1], s[y[te] == 0]))
    for m, v in got.items():
        want = ref(path, key_, m)
        assert abs(v - want) < 2e-3, f"{nm} {m}: re-scored {v:.5f} vs published {want:.5f}"
    log(f"[verify]   {nm:24s} winrate {got['winrate']:.4f}  AUROC {got['auroc']:.4f}  "
        f"AUPRC {got['auprc']:.4f}  -- matches {path}:{key_}")

# roc and pr curve points, so the r layer stays plotting-only
def curve(sc, yy, npt=700):
    o = np.argsort(-sc, kind="stable"); yo = yy[o]; so = sc[o]
    end = np.r_[np.flatnonzero(so[1:] != so[:-1]), len(so) - 1]
    tp = np.cumsum(yo)[end]; fp = (end + 1) - tp
    P = yy.sum(); N = len(yy) - P
    tpr = tp / P; fpr = fp / N; prec = tp / (end + 1); rec = tpr
    if len(end) > npt:
        k = np.unique(np.r_[0, np.linspace(0, len(end) - 1, npt).astype(int), len(end) - 1])
    else:
        k = np.arange(len(end))
    return fpr[k], tpr[k], rec[k], prec[k]

MODELS = [("sequence CNN", cnn), ("splice features (no GC)", spl),
          ("acceptor 3'ss", np.asarray(acc, float)), ("window GC", gcs)]
with open("curves_roc_pr.tsv", "w") as fh:
    fh.write("model\tfpr\ttpr\trecall\tprecision\tauroc\tauprc\tn_test\tbase_rate\n")
    for nm, sc in MODELS:
        s_ = np.asarray(sc, float)[te]; yy = (y[te] == 1).astype(float)
        a = auroc(s_[yy == 1], s_[yy == 0]); q = auprc(s_[yy == 1], s_[yy == 0])
        f_, t_, r_, p_ = curve(s_, yy)
        for i2 in range(len(f_)):
            fh.write(f"{nm}\t{f_[i2]:.6f}\t{t_[i2]:.6f}\t{r_[i2]:.6f}\t{p_[i2]:.6f}\t"
                     f"{a:.5f}\t{q:.5f}\t{len(te)}\t{yy.mean():.5f}\n")
        log(f"[curve]    {nm:24s} {len(f_)} points  AUROC {a:.4f}  AUPRC {q:.4f}")
log("[out] curves_roc_pr.tsv")

out = pd.DataFrame(dict(gene=gene[te], chrom=chrom[te], center=center[te],
                        is_canonical=(y[te] == 1),
                        score_cnn=cnn[te], score_splice=spl[te],
                        score_acceptor=np.asarray(acc, float)[te], score_gc=gcs[te]))
pq_write("scores_for_curves.parquet", out, compression="SNAPPY")
log(f"[out] scores_for_curves.parquet  {len(out)} test peaks, "
    f"{int(out.is_canonical.sum())} real, base rate {out.is_canonical.mean():.5f}")
