#!/usr/bin/env python3
# the hand-built splice-biology comparator for the cnn figures
import sys
import numpy as np
from fastparquet import ParquetFile

def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, NBOOT = 42, 1000
ADD_MIN, CEIL_GAP, INNER_VAL_FRAC = 0.005, 0.02, 0.25
CONTENT_COLS = ["spl_term_ppt", "spl_u2af65_ppt"]

D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
split = D["split"][ok]; y = D["y"][ok].astype(float); gene = D["gene"][ok]
C0 = np.load("analysis_columns.npz", allow_pickle=True)
assert (C0["is_canonical"][ok] == (y == 1)).all(), "row alignment"
A_gene = C0["gene"][ok]; A_chrom = C0["chrom"][ok]
A_center = C0["center"][ok]; A_strand = C0["strand"][ok]
A_gcc = C0["gc_contrast"][ok]; A_acc = C0["spl_upstream_max_me3"][ok]

F = ParquetFile("shortlist_feat_all.parquet").to_pandas()
SPL = [c for c in F.columns if c.startswith("spl_")]
log(f"shortlist_feat_all: {len(F)} rows, {len(SPL)} spl_* columns (no GC columns present)")
key = {(g, c, int(ct), s): i for i, (g, c, ct, s) in
       enumerate(zip(F.gene, F.chrom, F.center, F.strand))}
row = np.array([key.get((g, c, int(ct), s), -1) for g, c, ct, s in
                zip(A_gene, A_chrom, A_center, A_strand)])
hit = row >= 0
log(f"joined {int(hit.sum())}/{len(A_gene)} analysis peaks on (gene, chrom, center, strand)")
X = np.full((len(A_gene), len(SPL)), np.nan)
X[hit] = F.loc[row[hit], SPL].to_numpy(float)
X2 = np.column_stack([A_gcc, A_acc])

def centre_by_gene(M, g):
    out = M.copy()
    order = np.argsort(g, kind="stable"); gs = g[order]
    cut = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1, len(gs)]
    for j in range(len(cut) - 1):
        idx = order[cut[j]:cut[j + 1]]
        mu = np.nanmean(out[idx], 0)
        out[idx] = out[idx] - np.where(np.isfinite(mu), mu, 0.0)
    return out

def fit_ridge(Xt, yt, lam=1.0, iters=100):
    b = np.zeros(Xt.shape[1]); P = np.eye(Xt.shape[1]) * lam; P[0, 0] = 0
    for _ in range(iters):
        eta = np.clip(Xt @ b, -30, 30); mu = 1 / (1 + np.exp(-eta))
        w = np.maximum(mu * (1 - mu), 1e-9); z = eta + (yt - mu) / w
        XtW = Xt.T * w
        nb = np.linalg.solve(XtW @ Xt + P, XtW @ z + P @ b)
        if np.max(np.abs(nb - b)) < 1e-9: b = nb; break
        b = nb
    return b

def auroc(pos, neg):
    a = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); rk = ((cs - cnt) + cs + 1) / 2.0
    return float((rk[inv][:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
def auprc(pos, neg):
    s = np.concatenate([pos, neg]); yy = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    o = np.argsort(-s, kind="stable"); yy = yy[o]; s = s[o]
    end = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    tp = np.cumsum(yy)[end]; prec = tp / (end + 1); rec = tp / len(pos)
    return float(np.sum(prec * np.diff(np.r_[0.0, rec])))

def make_groups(ii):
    by = {}
    for k, g in enumerate(gene[ii]):
        r, c = by.setdefault(g, ([], []))
        (r if y[ii][k] == 1 else c).append(k)
    return [(np.array(r), np.array(c)) for r, c in by.values() if r and c]

def winrate(sc, groups):
    return float(np.mean([((sc[r][:, None] > sc[c][None, :]).sum()
                           + 0.5 * (sc[r][:, None] == sc[c][None, :]).sum()) / (len(r) * len(c))
                          for r, c in groups]))

te = np.flatnonzero(split == "test"); trv = np.flatnonzero(split != "test")
tr = np.flatnonzero(split == "train")
GRP_TE = make_groups(te)

# fit / score on an arbitrary column subset
def zmat(M, fit_idx):
    Cg = centre_by_gene(M, gene)
    mu = np.nanmean(Cg[fit_idx], 0); mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.nanstd(Cg[fit_idx], 0); sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return (np.where(np.isfinite(Cg), Cg, mu) - mu) / sd

def scaler(M, fit_idx):
    Cg = centre_by_gene(M, gene)
    mu = np.nanmean(Cg[fit_idx], 0); mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.nanstd(Cg[fit_idx], 0); sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return mu, sd

def fit_score(M, fit_idx, eval_idx):
    Z = zmat(M, fit_idx)
    b = fit_ridge(np.column_stack([np.ones(len(fit_idx)), Z[fit_idx]]), y[fit_idx])
    return np.column_stack([np.ones(len(eval_idx)), Z[eval_idx]]) @ b, b

# forward selection on train genes only
rs = np.random.default_rng(SEED)
tg = np.array(sorted(set(gene[tr])))
rs.shuffle(tg)
nval = int(round(INNER_VAL_FRAC * len(tg)))
inner_val_genes = set(tg[:nval])
iv = tr[np.array([g in inner_val_genes for g in gene[tr]])]
it = tr[np.array([g not in inner_val_genes for g in gene[tr]])]
GRP_IV = make_groups(iv)
log(f"inner selection split (TRAIN genes only): fit {len(it)} peaks / "
    f"{len(set(gene[it]))} genes, score {len(iv)} peaks / {len(GRP_IV)} genes")

def inner_wr(cols):
    sc, _ = fit_score(X[:, cols], it, iv)
    return winrate(sc, GRP_IV)

ceil_iv = inner_wr(list(range(len(SPL))))
log(f"inner-val ceiling, full {len(SPL)}-feature pool: {ceil_iv:.4f} "
    f"(selection stops within {CEIL_GAP} of this, or when a feature adds < {ADD_MIN})")

sel, cur, trace = [], 0.5, []
while len(sel) < len(SPL):
    best = (-1.0, None)
    for j in range(len(SPL)):
        if j in sel: continue
        w = inner_wr(sel + [j])
        if w > best[0]: best = (w, j)
    gain = best[0] - cur
    if gain < ADD_MIN:
        log(f"[select] stop: best remaining feature {SPL[best[1]]} adds only {gain:+.4f} "
            f"(< {ADD_MIN})")
        break
    sel.append(best[1]); cur = best[0]
    trace.append((SPL[best[1]], cur, gain))
    log(f"[select] +{SPL[best[1]]:26s} inner-val win-rate {cur:.4f} ({gain:+.4f})")
    if cur >= ceil_iv - CEIL_GAP:
        log(f"[select] stop: within {CEIL_GAP} of the ceiling {ceil_iv:.4f}")
        break
log(f"[select] surviving features ({len(sel)}): " + ", ".join(SPL[j] for j in sel))

sel_cap, cur_cap = list(sel), cur
while cur_cap < ceil_iv - CEIL_GAP and len(sel_cap) < len(SPL):
    best = (-1.0, None)
    for j in range(len(SPL)):
        if j in sel_cap: continue
        w = inner_wr(sel_cap + [j])
        if w > best[0]: best = (w, j)
    if best[1] is None or best[0] <= cur_cap: break
    sel_cap.append(best[1]); cur_cap = best[0]
    log(f"[select-capped] +{SPL[best[1]]:26s} inner-val win-rate {cur_cap:.4f}")
log(f"[select-capped] {len(sel_cap)} features reach inner-val {cur_cap:.4f} "
    f"vs ceiling {ceil_iv:.4f} (gap {ceil_iv-cur_cap:+.4f})")
content_in = [SPL[j] for j in sel if SPL[j] in CONTENT_COLS]
log(f"[select] of these, pyrimidine-CONTENT (invariant under dinucleotide shuffling): "
    + (", ".join(content_in) if content_in else "none"))

# final evaluation on the held-out test genes
def suite(sc):
    wg = np.array([((sc[r][:, None] > sc[c][None, :]).sum()
                    + 0.5 * (sc[r][:, None] == sc[c][None, :]).sum()) / (len(r) * len(c))
                   for r, c in GRP_TE])
    p1 = np.array([float(sc[r].max() > sc[c].max()) + 0.5 * float(sc[r].max() == sc[c].max())
                   for r, c in GRP_TE])
    pos = sc[y[te] == 1]; neg = sc[y[te] == 0]
    out = dict(winrate=wg.mean(), p_at_1=p1.mean(), auroc=auroc(pos, neg), auprc=auprc(pos, neg))
    r2 = np.random.default_rng(SEED); B = r2.integers(0, len(GRP_TE), size=(NBOOT, len(GRP_TE)))
    ba = np.empty(NBOOT); bq = np.empty(NBOOT)
    for t in range(NBOOT):
        pi = np.concatenate([GRP_TE[j][0] for j in B[t]]); ni = np.concatenate([GRP_TE[j][1] for j in B[t]])
        ba[t] = auroc(sc[pi], sc[ni]); bq[t] = auprc(sc[pi], sc[ni])
    for nm, v in (("winrate", wg[B].mean(1)), ("p_at_1", p1[B].mean(1)), ("auroc", ba), ("auprc", bq)):
        out[nm + "_lo"], out[nm + "_hi"] = np.percentile(v, [2.5, 97.5])
    return out

def run(M, name):
    sc, b = fit_score(M, trv, te)
    r = suite(sc); r["_beta"] = b
    log(f"[{name}] within-gene {r['winrate']:.4f} [{r['winrate_lo']:.4f},{r['winrate_hi']:.4f}]  "
        f"P@1 {r['p_at_1']:.4f} [{r['p_at_1_lo']:.4f},{r['p_at_1_hi']:.4f}]  "
        f"AUROC {r['auroc']:.4f} [{r['auroc_lo']:.4f},{r['auroc_hi']:.4f}]  "
        f"AUPRC {r['auprc']:.4f} [{r['auprc_lo']:.4f},{r['auprc_hi']:.4f}]")
    return r

log(f"\ntrain+val {len(trv)} peaks, test {len(te)} peaks in {len(GRP_TE)} genes, "
    f"base rate {y[te].mean():.4f}")
strict = [j for j in range(len(SPL)) if SPL[j] not in CONTENT_COLS]
res = {
    "splice_selected": run(X[:, sel], f"splice selected ({len(sel)}: {', '.join(SPL[j] for j in sel)})"),
    "splice_selected_capped": run(X[:, sel_cap], f"splice selected, ceiling-capped rule ({len(sel_cap)})"),
    "splice_ceiling":  run(X, f"splice ceiling (all {len(SPL)} spl_*)"),
    "splice_strict_no_content": run(X[:, strict], f"splice strict, PPT content dropped ({len(strict)})"),
    "acceptor_alone":  run(X[:, [SPL.index("spl_upstream_max_me3")]], "acceptor alone (spl_upstream_max_me3)"),
    "two_feature":     run(X2, "two-feature (gc_contrast + acceptor), stderr reference only"),
}
_mu, _sd = scaler(X[:, sel], trv)
np.savez_compressed("splice_model_fit.npz",
                    sel=np.array(sel), sel_names=np.array([SPL[j] for j in sel]),
                    beta=res["splice_selected"]["_beta"], spl_cols=np.array(SPL),
                    strict=np.array(strict), mu=_mu, sd=_sd,
                    sel_cap=np.array(sel_cap),
                    sel_cap_names=np.array([SPL[j] for j in sel_cap]))
log(f"[fit] saved scaler mu={np.round(_mu,4)} sd={np.round(_sd,4)} and beta="
    f"{np.round(res['splice_selected']['_beta'],4)}")
with open("splice_feature_metrics.tsv", "w") as fh:
    fh.write("score\tmetric\tvalue\tci_lo\tci_hi\tn_genes\tn_peaks\tn_features\n")
    nf = {"splice_selected": len(sel), "splice_selected_capped": len(sel_cap), "splice_ceiling": len(SPL),
          "splice_strict_no_content": len(strict), "acceptor_alone": 1, "two_feature": 2}
    for nm, r in res.items():
        for m in ("winrate", "p_at_1", "auroc", "auprc"):
            fh.write(f"{nm}\t{m}\t{r[m]:.5f}\t{r[m+'_lo']:.5f}\t{r[m+'_hi']:.5f}\t"
                     f"{len(GRP_TE)}\t{len(te)}\t{nf[nm]}\n")
    for i, (n, w, g) in enumerate(trace):
        fh.write(f"selection_step\t{n}\t{w:.5f}\t{g:.5f}\t\t{len(GRP_IV)}\t{len(iv)}\t{i+1}\n")
    fh.write(f"selection\tinner_val_ceiling\t{ceil_iv:.5f}\t\t\t{len(GRP_IV)}\t{len(iv)}\t{len(SPL)}\n")
    fh.write(f"selection\tselected_columns\t{'|'.join(SPL[j] for j in sel)}\t\t\t\t\t{len(sel)}\n")
    fh.write(f"selection\tselected_columns_capped\t{'|'.join(SPL[j] for j in sel_cap)}\t\t\t\t\t{len(sel_cap)}\n")
    fh.write(f"selection\tcontent_columns_in_selection\t{'|'.join(content_in) if content_in else 'none'}"
             f"\t\t\t\t\t{len(content_in)}\n")
    fh.write(f"splice_selected\tn_joined\t{int(hit.sum())}\t\t\t{len(GRP_TE)}\t{len(te)}\t{len(sel)}\n")
log("[out] splice_feature_metrics.tsv splice_model_fit.npz")
