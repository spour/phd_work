#!/usr/bin/env python3
# corrected-set reranker: performance, the gc confound check, and the shared peak table
import csv, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, HOLD_FRAC, RIDGE, NBOOT, NPERM = 42, 0.20, 1.0, 400, 2000
rng = np.random.default_rng(SEED)

FINAL = [L.strip() for L in open("rerank_corr_chosen.tsv").readlines()[1:] if L.strip()]
log("model features:", FINAL)

D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
S = np.load("splice_feats_cache.npz")
def col(n):
    if n in D.columns: return D[n].to_numpy(float)
    if n == "acc_minus_max5": return D["spl_upstream_max_me3"].to_numpy(float) - D["spl_interv_me5_max"].to_numpy(float)
    alias = {"scan_acc_rank2k": "B_acc_rank2k", "scan_acc_margin2k": "B_acc_margin2k", "scan_in_intron": "C_in_intron"}
    return S[alias[n]][ridx]
X = np.column_stack([col(n) for n in FINAL])
Z = np.load("corr_rowmask.npz"); ROW = Z["ROW"]; APA = Z["APA"]; TERMEX = Z["TERMEX"]
assert np.array_equal(Z["ridx"], ridx)
TE = np.load("termexon_gc_cache.npz"); GC = TE["gc500"][ridx]

gi = defaultdict(list)
for i in range(len(gene)): gi[gene[i]].append(i)
gi = {g: np.array(v) for g, v in gi.items()}
allg = np.array(sorted(set(gene)))
hold_g = set(rng.choice(allg, size=int(round(HOLD_FRAC * len(allg))), replace=False))
is_hold = np.array([g in hold_g for g in gene])
CHROMS = sorted(set(chrom))

def fit_ridge(Xt, yt, lam=RIDGE):
    P = np.eye(Xt.shape[1]) * lam; P[0, 0] = 0.0; b = np.zeros(Xt.shape[1])
    for _ in range(80):
        eta = np.clip(Xt @ b, -30, 30); mu = 1 / (1 + np.exp(-eta)); w = np.maximum(mu * (1 - mu), 1e-9)
        z = eta + (yt - mu) / w; XtW = Xt.T * w
        try: nb = np.linalg.solve(XtW @ Xt + P, XtW @ z + P @ b)
        except np.linalg.LinAlgError: nb = np.linalg.lstsq(XtW @ Xt + P, XtW @ z + P @ b, rcond=None)[0]
        if np.max(np.abs(nb - b)) < 1e-9: b = nb; break
        b = nb
    return b
def prep(Xtr, Xte):
    mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0); sd[sd == 0] = 1.0
    A = (np.where(np.isfinite(Xtr), Xtr, mu) - mu) / sd
    B = (np.where(np.isfinite(Xte), Xte, mu) - mu) / sd
    return np.column_stack([np.ones(len(A)), A]), np.column_stack([np.ones(len(B)), B])
def loco_pred(train_mask, test_mask):
    pred = np.full(len(y), np.nan)
    for c in CHROMS:
        te = ROW & test_mask & (chrom == c); tr = ROW & train_mask & (chrom != c)
        if te.sum() == 0 or tr.sum() == 0: continue
        A, B = prep(X[tr], X[te]); pred[te] = B @ fit_ridge(A, y[tr])
    return pred
def per_gene(pred, mask):
    hits, rrs, ns, gl = [], [], [], []
    for g, ix in gi.items():
        if not mask[ix[0]]: continue
        ix = ix[ROW[ix]]
        if len(ix) < 2 or (y[ix] == 1).sum() != 1: continue
        s = pred[ix]
        if not np.isfinite(s).all(): continue
        n = len(ix); j = np.argsort(-(s + rng.random(n) * 1e-12), kind="mergesort")
        srt = s[j]; rk = np.empty(n); i = 0
        while i < n:
            k = i
            while k + 1 < n and srt[k + 1] == srt[i]: k += 1
            rk[i:k + 1] = (i + k) / 2.0 + 1.0; i = k + 1
        r = np.empty(n); r[j] = rk
        c = int(np.argmax(y[ix]))
        hits.append(1.0 if j[0] == c else 0.0); rrs.append(1.0 / r[c]); ns.append(n); gl.append((g, ix))
    return np.array(hits), np.array(rrs), np.array(ns, float), gl
def auroc(yv, s):
    o = np.argsort(s, kind="mergesort"); sv = s[o]; n = len(s); rk = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    rr = np.empty(n); rr[o] = rk; npos = yv.sum(); nneg = n - npos
    return float((rr[yv == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def auprc(yv, s):
    o = np.argsort(-s, kind="mergesort"); yy = yv[o]; sv = s[o]
    tp = np.cumsum(yy); k = np.arange(1, len(yy) + 1)
    idx = np.where(np.r_[sv[1:] != sv[:-1], True])[0]
    prec = (tp[idx] / k[idx])[np.searchsorted(idx, np.arange(len(yy)))]
    return float((prec * yy).sum() / yy.sum())
def bci(v):
    b = v[rng.integers(0, len(v), size=(NBOOT, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

# permutation check of the analytic chance rates
def flat(vals):
    sz = np.array([len(v) for v in vals]); st = np.r_[0, np.cumsum(sz)[:-1]]
    return np.concatenate(vals), st, sz
def perm_chance(vals, nperm=NPERM):
    fl, st, sz = flat(vals)
    draws = st[None, :] + (rng.random((nperm, len(sz))) * sz[None, :]).astype(int)
    m = fl[draws].mean(1)
    return float(m.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

ALL = np.ones(len(y), bool)
pred_hold = loco_pred(~is_hold, is_hold)
pred_all = loco_pred(ALL, ALL)

def report(tag, pred, mask):
    hits, rrs, ns, gl = per_gene(pred, mask)
    rows = np.concatenate([ix for _, ix in gl]); yv = y[rows]; sv = pred[rows]
    Hn = np.array([np.sum(1.0 / np.arange(1, int(n) + 1)) for n in ns])
    p1, p1lo, p1hi = bci(hits); mr, mrlo, mrhi = bci(rrs)
    e1, e1lo, e1hi = bci(hits - 1.0 / ns); em, emlo, emhi = bci(rrs - Hn / ns)
    p1v = [np.r_[1.0, np.zeros(int(n) - 1)] for n in ns]
    mrv = [1.0 / np.arange(1, int(n) + 1) for n in ns]
    pc, pclo, pchi = perm_chance(p1v); mc, mclo, mchi = perm_chance(mrv)
    gidx = [ix for _, ix in gl]; ab = np.empty(200); pb = np.empty(200)
    for t in range(200):
        ii = np.concatenate([gidx[j] for j in rng.integers(0, len(gidx), len(gidx))])
        ab[t] = auroc(y[ii], pred[ii]); pb[t] = auprc(y[ii], pred[ii])
    r = dict(model=tag, n_genes=len(gl), n_rows=len(rows), p1=p1, p1lo=p1lo, p1hi=p1hi,
             p1_chance=float(np.mean(1.0 / ns)), p1_chance_perm=pc, p1_chance_perm_lo=pclo, p1_chance_perm_hi=pchi,
             p1_exc=e1, p1_exc_lo=e1lo, p1_exc_hi=e1hi,
             mrr=mr, mrr_lo=mrlo, mrr_hi=mrhi, mrr_chance=float(np.mean(Hn / ns)),
             mrr_chance_perm=mc, mrr_chance_perm_lo=mclo, mrr_chance_perm_hi=mchi,
             mrr_exc=em, mrr_exc_lo=emlo, mrr_exc_hi=emhi,
             auroc=auroc(yv, sv), auroc_lo=float(np.percentile(ab, 2.5)), auroc_hi=float(np.percentile(ab, 97.5)),
             auprc=auprc(yv, sv), auprc_lo=float(np.percentile(pb, 2.5)), auprc_hi=float(np.percentile(pb, 97.5)),
             base_rate=float(yv.mean()))
    print(f"{tag:<18} genes {r['n_genes']:>5} rows {r['n_rows']:>6}")
    print(f"    P@1   {p1:.4f} [{p1lo:.4f},{p1hi:.4f}]  chance {r['p1_chance']:.4f} "
          f"(perm {pc:.4f} [{pclo:.4f},{pchi:.4f}])  excess {e1:+.4f} [{e1lo:+.4f},{e1hi:+.4f}]")
    print(f"    MRR   {mr:.4f} [{mrlo:.4f},{mrhi:.4f}]  chance {r['mrr_chance']:.4f} "
          f"(perm {mc:.4f} [{mclo:.4f},{mchi:.4f}])  excess {em:+.4f} [{emlo:+.4f},{emhi:+.4f}]")
    print(f"    AUROC {r['auroc']:.4f} [{r['auroc_lo']:.4f},{r['auroc_hi']:.4f}]  chance 0.5000")
    print(f"    AUPRC {r['auprc']:.4f} [{r['auprc_lo']:.4f},{r['auprc_hi']:.4f}]  base rate {r['base_rate']:.4f}")
    return r

print("\n===== CORRECTED-SET PERFORMANCE (%d features: %s) =====" % (len(FINAL), ", ".join(FINAL)))
RES = [report("corr_hold", pred_hold, is_hold),
       report("corr_loco_all", pred_all, ALL)]
ks = [k for k in RES[0] if k != "model"]
with open("rerank_corr_metrics.tsv", "w") as fh:
    fh.write("model\tn_feat\t" + "\t".join(ks) + "\n")
    for r in RES:
        fh.write(f"{r['model']}\t{len(FINAL)}\t" + "\t".join(
            f"{r[k]:.5f}" if isinstance(r[k], float) else str(r[k]) for k in ks) + "\n")

# gc confound
def pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b); a = a[m]; b = b[m]
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan
def rankv(v):
    o = np.argsort(v, kind="mergesort"); sv = v[o]; n = len(v); rk = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    r = np.empty(n); r[o] = rk; return r
def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return pearson(rankv(a[m]), rankv(b[m]))
def boot_r(a, b, gl, fn):
    gidx = [ix for _, ix in gl]; out = np.empty(200)
    for t in range(200):
        ii = np.concatenate([gidx[j] for j in rng.integers(0, len(gidx), len(gidx))])
        out[t] = fn(a[ii], b[ii])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

hits, rrs, ns, gl = per_gene(pred_all, ALL)
rows = np.concatenate([ix for _, ix in gl])
yv = y[rows]; dv = pred_all[rows]; gv = GC[rows]
print("\n===== GC CONFOUND CHECK =====")
print(f"GC = G+C fraction of the 500 nt window ending at the pA; {len(rows)} peaks, "
      f"{len(gl)} genes; mean GC canonical {gv[yv==1].mean():.4f}, competitor {gv[yv==0].mean():.4f}")
GCROWS = []
def gcrow(label, afull, bfull, sub=None):
    idx = rows if sub is None else rows[sub]
    p = pearson(afull[idx], bfull[idx]); s = spearman(afull[idx], bfull[idx])
    plo, phi = boot_r(afull, bfull, gl, pearson) if sub is None else (np.nan, np.nan)
    print(f"  {label:<44} n {len(idx):>6}  Pearson {p:>+7.4f}  Spearman {s:>+7.4f}" +
          ("" if sub is not None else f"  [95% CI {plo:+.4f},{phi:+.4f}]"))
    GCROWS.append((label, len(idx), p, s, plo, phi))
gcrow("discriminant vs GC, all peaks", pred_all, GC)
gcrow("discriminant vs GC, canonical only", pred_all, GC, yv == 1)
gcrow("discriminant vs GC, competitor only", pred_all, GC, yv == 0)
for a, nm in enumerate(FINAL):
    gcrow(f"{nm} vs GC, all peaks", X[:, a], GC)
    gcrow(f"{nm} vs GC, canonical only", X[:, a], GC, yv == 1)
    gcrow(f"{nm} vs GC, competitor only", X[:, a], GC, yv == 0)

mfin = np.isfinite(dv) & np.isfinite(gv)
A = np.column_stack([np.ones(mfin.sum()), gv[mfin]])
bb = np.linalg.lstsq(A, dv[mfin], rcond=None)[0]
res_all = np.full(len(y), np.nan); res_all[rows[mfin]] = dv[mfin] - A @ bb
r2 = pearson(dv[mfin], gv[mfin]) ** 2
h2, _, n2, gl2 = per_gene(res_all, ALL)
rows2 = np.concatenate([ix for _, ix in gl2])
print(f"\n  residualised discriminant (disc ~ a + b*GC, R2 = {r2:.5f}):")
print(f"    P@1 {h2.mean():.4f} vs {hits.mean():.4f} raw   chance {np.mean(1.0/n2):.4f}")
print(f"    AUROC {auroc(y[rows2], res_all[rows2]):.4f} vs {auroc(yv, dv):.4f} raw")

gc_all = np.where(np.isfinite(GC), GC, np.nan)
h3, _, n3, gl3 = per_gene(gc_all, ALL)
win = []
for g, ix in gl:
    c = int(np.argmax(y[ix])); o = np.delete(np.arange(len(ix)), c)
    win.append(np.mean(GC[ix[c]] > GC[ix[o]]) if len(o) else np.nan)
win = np.array(win, float)
print(f"\n  GC alone: within-gene win-rate {np.nanmean(win):.4f} (0.5 = chance), "
      f"P@1 {h3.mean():.4f} vs chance {np.mean(1.0/n3):.4f}, AUROC {auroc(yv, gv):.4f}")
with open("rerank_corr_gc.tsv", "w") as fh:
    fh.write("quantity\tn\tpearson\tspearman\tci_lo\tci_hi\n")
    for lab, n_, p, s, lo, hi in GCROWS:
        fh.write(f"{lab}\t{n_}\t{p:.5f}\t{s:.5f}\t{lo:.5f}\t{hi:.5f}\n")
    fh.write(f"#resid_r2\t{r2:.6f}\n#resid_p1\t{h2.mean():.5f}\t#raw_p1\t{hits.mean():.5f}\n")
    fh.write(f"#resid_auroc\t{auroc(y[rows2], res_all[rows2]):.5f}\t#raw_auroc\t{auroc(yv, dv):.5f}\n")
    fh.write(f"#gc_winrate\t{np.nanmean(win):.5f}\n#gc_p1\t{h3.mean():.5f}\t#gc_p1_chance\t{np.mean(1.0/n3):.5f}\n")
    fh.write(f"#gc_auroc\t{auroc(yv, gv):.5f}\n")
    fh.write(f"#mean_gc_canonical\t{gv[yv==1].mean():.5f}\n#mean_gc_competitor\t{gv[yv==0].mean():.5f}\n")
with open("rerank_corr_gc_points.tsv", "w") as fh:
    fh.write("gene\tclass\tgc\tdiscriminant\n")
    for i in rows:
        fh.write(f"{gene[i]}\t{'canonical' if y[i]==1 else 'competitor'}\t{GC[i]:.5f}\t{pred_all[i]:.5f}\n")

# shared peak table
out = pd.DataFrame(dict(
    gene=gene, chrom=chrom, center=D.center.to_numpy(), strand=D.strand.to_numpy(),
    is_canonical=(y == 1),
    apa_annotated=APA, in_reference_terminal_exon=TERMEX,
    in_analysis_set=ROW, gc500=GC, discriminant_loco=pred_all))
for nm in FINAL: out[nm] = col(nm)
out.to_parquet("rerank_corr_peaks.parquet", engine="fastparquet", compression="snappy")
print(f"\nwrote rerank_corr_peaks.parquet: {len(out)} peaks, {int(out.in_analysis_set.sum())} in the analysis set")
log("DONE_CORR_GC")
