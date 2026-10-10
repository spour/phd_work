#!/usr/bin/env python3
# within-gene evaluation of the cnn (--scores/--out parameterised
import sys, json, argparse
import numpy as np

def log(*a): print(*a, file=sys.stderr, flush=True)
_P = argparse.ArgumentParser()
_P.add_argument("--scores", default="cnn_model_scores.npz")
_P.add_argument("--out", default="cnn_metrics.tsv")
_A = _P.parse_args()
SEED, NBOOT = 42, 1000
REF = {"acceptor 3'ss (within-gene)": 0.606, "two-feature (LOCO)": 0.632,
       "two-feature (gene holdout)": 0.645, "existing sequence CNN (POOLED)": 0.740}

D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
S = np.load(_A.scores, allow_pickle=True)
log(f"scores {_A.scores}" + (f"  variant {S['tag']} seed {S['seed']}" if "tag" in S.files else ""))
y = S["y"]; gene = S["gene"]; split = S["split"]; logit = S["logit"].astype(np.float64)
gc1 = D["window_gc"][ok]; gc2 = D["gc_contrast"][ok]
C = np.load("analysis_columns.npz", allow_pickle=True)
acc = C["spl_upstream_max_me3"][ok]
assert len(acc) == len(y) and (C["is_canonical"][ok] == (y == 1)).all(), "row alignment"
te = np.flatnonzero(split == "test"); trv = np.flatnonzero(split != "test")
log(f"test {len(te)} peaks, {len(set(gene[te]))} genes, {int(y[te].sum())} real")

# natural cubic spline basis (esl 5.2.1)
def ns(x, df=6, knots=None):
    x = np.asarray(x, float)
    if knots is None:
        knots = np.nanquantile(x[np.isfinite(x)], np.linspace(0, 1, df))
    k = np.unique(knots); K = len(k)
    d = lambda j: (np.maximum(x - k[j], 0) ** 3 - np.maximum(x - k[-1], 0) ** 3) / (k[-1] - k[j])
    B = [x] + [d(j) - d(K - 2) for j in range(K - 2)]
    return np.column_stack(B), k

def fit_ridge(X, z, lam=1e-3):
    X = np.column_stack([np.ones(len(X)), X])
    mu = X[:, 1:].mean(0); sd = X[:, 1:].std(0); sd[sd == 0] = 1
    Xs = np.column_stack([np.ones(len(X)), (X[:, 1:] - mu) / sd])
    P = np.eye(Xs.shape[1]) * lam; P[0, 0] = 0
    b = np.linalg.solve(Xs.T @ Xs + P, Xs.T @ z)
    return b, mu, sd
def apply_ridge(X, b, mu, sd):
    return np.column_stack([np.ones(len(X)), (X - mu) / sd]) @ b

def basis(cols, knots=None):
    Bs, ks = [], []
    for i, c in enumerate(cols):
        B, k = ns(c, knots=None if knots is None else knots[i]); Bs.append(B); ks.append(k)
    return np.column_stack(Bs), ks

def remove(vars_tr, vars_te, name):
    Btr, ks = basis(vars_tr)
    Bte, _ = basis(vars_te, knots=ks)
    good = np.isfinite(Btr).all(1) & np.isfinite(logit[trv])
    b, mu, sd = fit_ridge(Btr[good], logit[trv][good])
    hat = apply_ridge(np.where(np.isfinite(Bte), Bte, 0.0), b, mu, sd)
    r2 = 1 - np.var(logit[te] - hat) / np.var(logit[te])
    log(f"[removal] {name}: spline fit on {int(good.sum())} train+val rows, "
        f"test R^2 of the logit = {r2:.4f}")
    return hat, r2

# metrics
def gene_groups(ii):
    by = {}
    for k, g in enumerate(gene[ii]):
        r, c = by.setdefault(g, ([], []))
        (r if y[ii][k] == 1 else c).append(k)
    return [(np.array(r), np.array(c)) for r, c in by.values() if r and c]

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

def suite(sc, groups, ii, boot=NBOOT):
    wg = np.array([((sc[r][:, None] > sc[c][None, :]).sum()
                    + 0.5 * (sc[r][:, None] == sc[c][None, :]).sum()) / (len(r) * len(c))
                   for r, c in groups])
    p1 = np.array([float(sc[r].max() > sc[c].max()) + 0.5 * float(sc[r].max() == sc[c].max())
                   for r, c in groups])
    pos = sc[y[ii] == 1]; neg = sc[y[ii] == 0]
    out = dict(winrate=float(wg.mean()), p_at_1=float(p1.mean()),
               auroc=auroc(pos, neg), auprc=auprc(pos, neg),
               base=float((y[ii] == 1).mean()), n_genes=len(groups))
    rs = np.random.default_rng(SEED); B = rs.integers(0, len(groups), size=(boot, len(groups)))
    bw = wg[B].mean(1); bp = p1[B].mean(1)
    ba = np.empty(boot); bq = np.empty(boot)
    for t in range(boot):
        pick = B[t]
        pi = np.concatenate([groups[j][0] for j in pick]); ni = np.concatenate([groups[j][1] for j in pick])
        ba[t] = auroc(sc[pi], sc[ni]); bq[t] = auprc(sc[pi], sc[ni])
    for nm, v in (("winrate", bw), ("p_at_1", bp), ("auroc", ba), ("auprc", bq)):
        out[nm + "_lo"], out[nm + "_hi"] = np.percentile(v, [2.5, 97.5])
    return out

groups = gene_groups(te)
gc_hat, r2_gc = remove([gc1[trv], gc2[trv]], [gc1[te], gc2[te]], "GC basis {window_gc, gc_contrast}")
ga_hat, r2_ga = remove([gc1[trv], gc2[trv], acc[trv]], [gc1[te], gc2[te], acc[te]],
                       "GC + acceptor {window_gc, gc_contrast, spl_upstream_max_me3}")
VAR = {"full": logit[te], "gc_only": gc_hat, "residual": logit[te] - gc_hat,
       "gc_acceptor_only": ga_hat, "residual_gc_acceptor": logit[te] - ga_hat,
       "acceptor_alone": acc[te], "window_gc_alone": -gc1[te]}
rows = []
for nm, sc in VAR.items():
    r = suite(np.asarray(sc, float), groups, te)
    rows.append((nm, r))
    log(f"[{nm:22s}] within-gene {r['winrate']:.4f} [{r['winrate_lo']:.4f},{r['winrate_hi']:.4f}]  "
        f"P@1 {r['p_at_1']:.4f} [{r['p_at_1_lo']:.4f},{r['p_at_1_hi']:.4f}]  "
        f"AUROC {r['auroc']:.4f}  AUPRC {r['auprc']:.4f} (base {r['base']:.4f})")
with open(_A.out, "w") as fh:
    fh.write("score\tmetric\tvalue\tci_lo\tci_hi\tn_genes\tn_peaks\tbase_rate\n")
    for nm, r in rows:
        for m in ("winrate", "p_at_1", "auroc", "auprc"):
            fh.write(f"{nm}\t{m}\t{r[m]:.5f}\t{r[m+'_lo']:.5f}\t{r[m+'_hi']:.5f}\t"
                     f"{r['n_genes']}\t{len(te)}\t{r['base']:.5f}\n")
    for k, v in REF.items(): fh.write(f"reference\t{k}\t{v}\t\t\t\t\t\n")
    fh.write(f"removal\tR2_gc_on_logit\t{r2_gc:.5f}\t\t\t\t\t\n")
    fh.write(f"removal\tR2_gc_acceptor_on_logit\t{r2_ga:.5f}\t\t\t\t\t\n")
f = dict(rows)
log(f"\n[headline] within-gene win-rate: FULL {f['full']['winrate']:.4f} -> "
    f"RESIDUAL after GC {f['residual']['winrate']:.4f} "
    f"({100*(f['residual']['winrate']-0.5)/(f['full']['winrate']-0.5):.0f}% of the lift survives), "
    f"GC-ONLY {f['gc_only']['winrate']:.4f}; after GC+acceptor {f['residual_gc_acceptor']['winrate']:.4f}")
log(f"[headline] pooled AUROC FULL {f['full']['auroc']:.4f} (existing sequence CNN 0.740); "
    f"acceptor alone here {f['acceptor_alone']['winrate']:.4f} within-gene (reference 0.606)")
log(f"[out] {_A.out}")
