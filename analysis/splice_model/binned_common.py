#!/usr/bin/env python3
# the split, gene-centring, ridge irls
import sys
import numpy as np

def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, NBOOT = 42, 1000
INNER_VAL_FRAC = 0.25
LAMBDAS = np.array([1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 3e3, 1e4])

D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
split = D["split"][ok]; y = D["y"][ok].astype(float); gene = D["gene"][ok]
C0 = np.load("analysis_columns.npz", allow_pickle=True)
assert (C0["is_canonical"][ok] == (y == 1)).all(), "row alignment"
ACC = C0["spl_upstream_max_me3"][ok]

def centre_by_gene(M, g=None):
    g = gene if g is None else g
    out = M.astype(float).copy()
    order = np.argsort(g, kind="stable"); gs = g[order]
    cut = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1, len(gs)]
    for j in range(len(cut) - 1):
        idx = order[cut[j]:cut[j + 1]]
        mu = np.nanmean(out[idx], 0)
        out[idx] = out[idx] - np.where(np.isfinite(mu), mu, 0.0)
    return out

def fit_ridge(Xt, yt, lam=1.0, iters=100):
    b = np.zeros(Xt.shape[1]); Pm = np.eye(Xt.shape[1]) * lam; Pm[0, 0] = 0
    for _ in range(iters):
        eta = np.clip(Xt @ b, -30, 30); mu = 1 / (1 + np.exp(-eta))
        w = np.maximum(mu * (1 - mu), 1e-9); z = eta + (yt - mu) / w
        XtW = Xt.T * w
        nb = np.linalg.solve(XtW @ Xt + Pm, XtW @ z + Pm @ b)
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

_rs = np.random.default_rng(SEED)
_tg = np.array(sorted(set(gene[tr]))); _rs.shuffle(_tg)
_ivg = set(_tg[:int(round(INNER_VAL_FRAC * len(_tg)))])
iv = tr[np.array([g in _ivg for g in gene[tr]])]
it = tr[np.array([g not in _ivg for g in gene[tr]])]
GRP_IV = make_groups(iv)
BS = np.random.default_rng(SEED).integers(0, len(GRP_TE), size=(NBOOT, len(GRP_TE)))

def zmat(M, fit_idx):
    Cg = centre_by_gene(M)
    mu = np.nanmean(Cg[fit_idx], 0); mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.nanstd(Cg[fit_idx], 0); sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return (np.where(np.isfinite(Cg), Cg, mu) - mu) / sd

def fit_score(M, fit_idx, eval_idx, lam=1.0):
    Zm = zmat(M, fit_idx)
    b = fit_ridge(np.column_stack([np.ones(len(fit_idx)), Zm[fit_idx]]), y[fit_idx], lam)
    return np.column_stack([np.ones(len(eval_idx)), Zm[eval_idx]]) @ b, b

def inner_wr(M, lam=1.0):
    sc, _ = fit_score(M, it, iv, lam)
    return winrate(sc, GRP_IV)

def pick_lambda(M, tag=""):
    best = (-1.0, None)
    for lam in LAMBDAS:
        w = inner_wr(M, lam)
        if w > best[0]: best = (w, lam)
    if tag: log(f"[lambda] {tag}: inner-val {best[0]:.4f} at lambda {best[1]:g}")
    return best[1], best[0]

def suite(sc):
    wg = np.array([((sc[r][:, None] > sc[c][None, :]).sum()
                    + 0.5 * (sc[r][:, None] == sc[c][None, :]).sum()) / (len(r) * len(c))
                   for r, c in GRP_TE])
    p1 = np.array([float(sc[r].max() > sc[c].max()) + 0.5 * float(sc[r].max() == sc[c].max())
                   for r, c in GRP_TE])
    pos = sc[y[te] == 1]; neg = sc[y[te] == 0]
    out = dict(winrate=wg.mean(), p_at_1=p1.mean(), auroc=auroc(pos, neg), auprc=auprc(pos, neg),
               _wg=wg, _p1=p1)
    ba = np.empty(NBOOT); bq = np.empty(NBOOT)
    for t in range(NBOOT):
        pi = np.concatenate([GRP_TE[j][0] for j in BS[t]])
        ni = np.concatenate([GRP_TE[j][1] for j in BS[t]])
        ba[t] = auroc(sc[pi], sc[ni]); bq[t] = auprc(sc[pi], sc[ni])
    for nm, v in (("winrate", wg[BS].mean(1)), ("p_at_1", p1[BS].mean(1)),
                  ("auroc", ba), ("auprc", bq)):
        out[nm + "_lo"], out[nm + "_hi"] = np.percentile(v, [2.5, 97.5])
    return out
