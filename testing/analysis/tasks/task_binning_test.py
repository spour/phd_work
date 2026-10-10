#!/usr/bin/env python3
# does binning help?
import csv, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, HOLD_FRAC, RIDGE, NBOOT, NPERM = 42, 0.20, 1.0, 400, 2000
rng = np.random.default_rng(SEED)
K3 = ["spl_upstream_max_me3", "spl_pas_prox_u1", "spl_u1_suppression_up"]

D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
Z = np.load("corr_rowmask.npz"); ROW = Z["ROW"]
B = np.load("binned_features.npz"); BN = list(B["names"]); BX = B["X"][ridx].astype(float)
SPL = [i for i, n in enumerate(BN) if not n.endswith("_gc")]
GCB = [i for i, n in enumerate(BN) if n.endswith("_gc")]
X3 = np.column_stack([D[n].to_numpy(float) for n in K3])
VAR = {"k3": X3,
       "k3 + splice-binned": np.column_stack([X3, BX[:, SPL]]),
       "gc-binned alone": BX[:, GCB],
       "k3 + gc-binned": np.column_stack([X3, BX[:, GCB]]),
       "k3 + splice + gc": np.column_stack([X3, BX[:, SPL], BX[:, GCB]])}

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
    mu = np.where(np.isfinite(mu), mu, 0.0)
    A = (np.where(np.isfinite(Xtr), Xtr, mu) - mu) / sd
    Bm = (np.where(np.isfinite(Xte), Xte, mu) - mu) / sd
    return np.column_stack([np.ones(len(A)), A]), np.column_stack([np.ones(len(Bm)), Bm])
def loco_pred(X, train_mask, test_mask):
    pred = np.full(len(y), np.nan)
    for c in CHROMS:
        te = ROW & test_mask & (chrom == c); tr = ROW & train_mask & (chrom != c)
        if te.sum() == 0 or tr.sum() == 0: continue
        A, Bm = prep(X[tr], X[te]); pred[te] = Bm @ fit_ridge(A, y[tr])
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
def flat(vals):
    sz = np.array([len(v) for v in vals]); st = np.r_[0, np.cumsum(sz)[:-1]]
    return np.concatenate(vals), st, sz
def perm_chance(vals, nperm=NPERM):
    fl, st, sz = flat(vals)
    draws = st[None, :] + (rng.random((nperm, len(sz))) * sz[None, :]).astype(int)
    m = fl[draws].mean(1)
    return float(m.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

# one held-out prediction per variant, on the same holdout genes
PRED = {}
for nm, X in VAR.items():
    PRED[nm] = loco_pred(X, ~is_hold, is_hold)
    log(f"fitted {nm} ({X.shape[1]} features)")
_, _, _, GL = per_gene(PRED["k3"], is_hold)
GENES = [g for g, _ in GL]
gset = set(GENES)
def evaluate(pred):
    hits, rrs, ns, gl = per_gene(pred, is_hold)
    keep = [i for i, (g, _) in enumerate(gl) if g in gset]
    hits, rrs, ns = hits[keep], rrs[keep], ns[keep]
    gl = [gl[i] for i in keep]
    rows = np.concatenate([ix for _, ix in gl])
    return hits, rrs, ns, gl, rows
BASE = evaluate(PRED["k3"])
BOOT = rng.integers(0, len(BASE[3]), size=(200, len(BASE[3])))

RES = []
for nm in VAR:
    hits, rrs, ns, gl, rows = evaluate(PRED[nm])
    Hn = np.array([np.sum(1.0 / np.arange(1, int(n) + 1)) for n in ns])
    pc, pclo, pchi = perm_chance([np.r_[1.0, np.zeros(int(n) - 1)] for n in ns])
    mc, mclo, mchi = perm_chance([1.0 / np.arange(1, int(n) + 1) for n in ns])
    gidx = [ix for _, ix in gl]
    ab = np.empty(200); pb = np.empty(200); hb = np.empty(200); rb = np.empty(200)
    for t in range(200):
        sel = BOOT[t]
        ii = np.concatenate([gidx[j] for j in sel])
        ab[t] = auroc(y[ii], PRED[nm][ii]); pb[t] = auprc(y[ii], PRED[nm][ii])
        hb[t] = hits[sel].mean(); rb[t] = rrs[sel].mean()
    RES.append(dict(model=nm, n_feat=VAR[nm].shape[1], n_genes=len(gl), n_rows=len(rows),
                    p1=hits.mean(), p1_lo=np.percentile(hb, 2.5), p1_hi=np.percentile(hb, 97.5),
                    p1_chance=float(np.mean(1.0 / ns)), p1_chance_perm=pc,
                    mrr=rrs.mean(), mrr_lo=np.percentile(rb, 2.5), mrr_hi=np.percentile(rb, 97.5),
                    mrr_chance=float(np.mean(Hn / ns)), mrr_chance_perm=mc,
                    auroc=auroc(y[rows], PRED[nm][rows]),
                    auroc_lo=np.percentile(ab, 2.5), auroc_hi=np.percentile(ab, 97.5),
                    auprc=auprc(y[rows], PRED[nm][rows]),
                    auprc_lo=np.percentile(pb, 2.5), auprc_hi=np.percentile(pb, 97.5),
                    base_rate=float(y[rows].mean())))
    RES[-1].update(_hb=hb, _rb=rb, _ab=ab, _pb=pb, _hits=hits, _rrs=rrs, _rows=rows)

print(f"\n===== BINNING TEST, corrected set, holdout genes {RES[0]['n_genes']} =====")
print(f"{'variant':<22} {'feat':>5} {'P@1':>7} {'95% CI':>17} {'chance':>7} {'MRR':>7} {'chance':>7} "
      f"{'AUROC':>7} {'AUPRC':>7} {'base':>6}")
for r in RES:
    print(f"{r['model']:<22} {r['n_feat']:>5} {r['p1']:>7.4f} [{r['p1_lo']:.4f},{r['p1_hi']:.4f}] "
          f"{r['p1_chance']:>7.4f} {r['mrr']:>7.4f} {r['mrr_chance']:>7.4f} {r['auroc']:>7.4f} "
          f"{r['auprc']:>7.4f} {r['base_rate']:>6.4f}")
print("\nanalytic vs permutation chance (P@1 / MRR): " +
      ", ".join(f"{r['p1_chance']:.4f} vs {r['p1_chance_perm']:.4f}" for r in RES[:1]) +
      f" / {RES[0]['mrr_chance']:.4f} vs {RES[0]['mrr_chance_perm']:.4f}")

byname = {r["model"]: r for r in RES}
CONTRASTS = [("k3", ["k3 + splice-binned", "gc-binned alone", "k3 + gc-binned", "k3 + splice + gc"]),
             ("k3 + gc-binned", ["k3 + splice + gc"]),
             ("gc-binned alone", ["k3 + gc-binned"])]
with open("binning_test_delta.tsv", "w") as fh:
    fh.write("reference\tvariant\tmetric\tdelta\tci_lo\tci_hi\n")
    for ref, others in CONTRASTS:
        base = byname[ref]
        print(f"\npaired increments over '{ref}' (same genes, same {len(BOOT)} gene resamples):")
        print(f"{'variant':<22} {'dP@1':>8} {'95% CI':>19} {'dMRR':>8} {'dAUROC':>8} {'95% CI':>19} {'dAUPRC':>8}")
        for nm in others:
            r = byname[nm]
            dp = r["_hits"].mean() - base["_hits"].mean(); dpb = r["_hb"] - base["_hb"]
            dm = r["_rrs"].mean() - base["_rrs"].mean(); dmb = r["_rb"] - base["_rb"]
            da = r["auroc"] - base["auroc"]; dab = r["_ab"] - base["_ab"]
            dq = r["auprc"] - base["auprc"]; dqb = r["_pb"] - base["_pb"]
            print(f"{nm:<22} {dp:>+8.4f} [{np.percentile(dpb,2.5):+.4f},{np.percentile(dpb,97.5):+.4f}] "
                  f"{dm:>+8.4f} {da:>+8.4f} [{np.percentile(dab,2.5):+.4f},{np.percentile(dab,97.5):+.4f}] {dq:>+8.4f}")
            for mn, d, b in [("P@1", dp, dpb), ("MRR", dm, dmb), ("AUROC", da, dab), ("AUPRC", dq, dqb)]:
                fh.write(f"{ref}\t{nm}\t{mn}\t{d:.5f}\t{np.percentile(b,2.5):.5f}\t{np.percentile(b,97.5):.5f}\n")
ks = ["n_feat", "n_genes", "n_rows", "p1", "p1_lo", "p1_hi", "p1_chance", "p1_chance_perm",
      "mrr", "mrr_lo", "mrr_hi", "mrr_chance", "mrr_chance_perm",
      "auroc", "auroc_lo", "auroc_hi", "auprc", "auprc_lo", "auprc_hi", "base_rate"]
with open("binning_test.tsv", "w") as fh:
    fh.write("model\t" + "\t".join(ks) + "\n")
    for r in RES:
        fh.write(r["model"] + "\t" + "\t".join(
            f"{r[k]:.5f}" if isinstance(r[k], (float, np.floating)) else str(r[k]) for k in ks) + "\n")
log("DONE_BINNING_TEST")
