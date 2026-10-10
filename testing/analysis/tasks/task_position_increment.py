#!/usr/bin/env python3
# item 1: does sequence add anything on top of position?
import sys, csv
import numpy as np
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
POS = "fig3_new_positional_features.tsv"; NFOLD = 5; RIDGE = 1.0

R = list(csv.DictReader(open(POS), delimiter="\t")); NP = len(R)
gene = np.array([r["gene"] for r in R]); isc = np.array([int(float(r["is_canonical"])) for r in R])
center = np.array([int(r["center"]) for r in R]); strand = np.array([r["strand"] for r in R])
F1 = np.load("hypothesis_test_features.npy"); ESC = np.load("hypothesis_test_escape.npy")
have = np.isfinite(F1[:, 0:60]).all(1); keep = np.where(have)[0]
gene_k = gene[keep]; y = isc[keep].astype(float); ISC = isc[keep]
COMP = F1[keep, 0:60]; SPL = F1[keep, 66:71]; ESCk = ESC[keep]
cen = center[keep]; strd = strand[keep]; NK = len(keep); BASE = float(y.mean())

genes = sorted(set(gene_k)); rs = np.random.default_rng(7)
perm = rs.permutation(len(genes)); fold_of = {g: perm[i] % NFOLD for i, g in enumerate(genes)}
fold = np.array([fold_of[g] for g in gene_k])
gp = defaultdict(list)
for i in range(NK): gp[gene_k[i]].append(i)
GEN = [np.array(v) for v in gp.values()]; NG = len(GEN)
GENP = [idx for idx in GEN if len(idx) >= 2 and (ISC[idx] == 1).any()]; NGP = len(GENP)
CHANCE_P1 = float(np.mean([1.0 / len(idx) for idx in GENP]))
def hn(n): return float(np.sum(1.0 / np.arange(1, n + 1)))
CHANCE_MRR = float(np.mean([hn(len(idx)) / len(idx) for idx in GEN]))

# positional features (transcript coordinate tc: larger = more downstream)
tc = np.where(strd == "+", cen, -cen).astype(float)
POS_rank = np.full(NK, np.nan); POS_dist = np.full(NK, np.nan)
for idx in GEN:
    t = tc[idx]; order = np.argsort(-t, kind="mergesort")
    r = np.empty(len(idx)); r[order] = np.arange(1, len(idx) + 1)
    POS_rank[idx] = r
    POS_dist[idx] = t.max() - t
POSr = POS_rank.reshape(-1, 1); POSd = POS_dist.reshape(-1, 1)
def hs(*m): return np.column_stack(m)

def fit_ridge(X, yv, lam):
    P = np.eye(X.shape[1]) * lam; P[0, 0] = 0.0; beta = np.zeros(X.shape[1])
    for _ in range(100):
        eta = np.clip(X @ beta, -30, 30); mu = 1 / (1 + np.exp(-eta))
        w = np.maximum(mu * (1 - mu), 1e-9); z = eta + (yv - mu) / w; XtW = X.T * w
        try: new = np.linalg.solve(XtW @ X + P, XtW @ z + P @ beta)
        except np.linalg.LinAlgError: new = np.linalg.lstsq(XtW @ X + P, XtW @ z + P @ beta, rcond=None)[0]
        if np.max(np.abs(new - beta)) < 1e-9: beta = new; break
        beta = new
    return beta
def cv_scores(X):
    pred = np.full(NK, np.nan)
    for f in range(NFOLD):
        tr = fold != f; te = fold == f
        Xtr = X[tr].astype(float).copy(); Xte = X[te].astype(float).copy()
        mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0); sd[sd == 0] = 1.0
        Xtr = np.where(np.isfinite(Xtr), Xtr, mu); Xte = np.where(np.isfinite(Xte), Xte, mu)
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        Xtr = np.column_stack([np.ones(Xtr.shape[0]), Xtr]); Xte = np.column_stack([np.ones(Xte.shape[0]), Xte])
        beta = fit_ridge(Xtr, y[tr], RIDGE); pred[te] = Xte @ beta
    return pred

def auroc(yv, s):
    order = np.argsort(s, kind="mergesort"); sv = s[order]; n = len(s); rk = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    rr = np.empty(n); rr[order] = rk; npos = yv.sum(); nneg = n - npos
    return (rr[yv == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
def auprc(yv, s):
    order = np.argsort(-s, kind="mergesort"); ys = yv[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); P = ys.sum()
    prec = tp / (tp + fp); rec = tp / P; ap = 0.0; prev = 0.0
    for k in range(len(ys)):
        if ys[k] == 1: ap += prec[k] * (rec[k] - prev); prev = rec[k]
    return ap
def mrr(s):
    tot = 0.0
    for idx in GEN:
        sc = s[idx]; yv = y[idx]; order = np.argsort(-sc, kind="mergesort")
        ranks = np.empty(len(sc)); ranks[order] = np.arange(1, len(sc) + 1); pos = ranks[yv == 1]
        if len(pos): tot += 1.0 / pos.min()
    return tot / NG
def p1_credit(s):
    out = np.empty(NGP)
    for k, idx in enumerate(GENP):
        sc = s[idx]; mx = np.nanmax(sc); am = sc == mx
        out[k] = float((ISC[idx][am] == 1).sum()) / am.sum()
    return out

NBOOT = 500; rsb = np.random.default_rng(11); BOOT = [rsb.integers(0, NG, NG) for _ in range(NBOOT)]
rsbp = np.random.default_rng(11); BOOTP = [rsbp.integers(0, NGP, NGP) for _ in range(NBOOT)]
def pooled(sample, s, metric):
    peaks = np.concatenate([GEN[g] for g in sample])
    return auroc(y[peaks], s[peaks]) if metric == "auroc" else auprc(y[peaks], s[peaks])
def mrr_boot(sample, s):
    tot = 0.0
    for g in sample:
        idx = GEN[g]; sc = s[idx]; yv = y[idx]; order = np.argsort(-sc, kind="mergesort")
        ranks = np.empty(len(sc)); ranks[order] = np.arange(1, len(sc) + 1); pos = ranks[yv == 1]
        if len(pos): tot += 1.0 / pos.min()
    return tot / len(sample)
def metrics(s):
    ar, ap, mr = auroc(y, s), auprc(y, s), mrr(s)
    arv = np.array([pooled(bs, s, "auroc") for bs in BOOT]); apv = np.array([pooled(bs, s, "auprc") for bs in BOOT])
    mrv = np.array([mrr_boot(bs, s) for bs in BOOT]); cred = p1_credit(s); p1 = cred.mean()
    p1v = np.array([cred[bs].mean() for bs in BOOTP])
    return dict(auroc=ar, auroc_lo=np.percentile(arv, 2.5), auroc_hi=np.percentile(arv, 97.5),
                auprc=ap, auprc_lo=np.percentile(apv, 2.5), auprc_hi=np.percentile(apv, 97.5),
                mrr=mr, mrr_lo=np.percentile(mrv, 2.5), mrr_hi=np.percentile(mrv, 97.5),
                p_at1=p1, p_at1_lo=np.percentile(p1v, 2.5), p_at1_hi=np.percentile(p1v, 97.5), cred=cred, s=s)
def paired_inc(a, b, metric):
    sa, sb = a["s"], b["s"]
    if metric == "auroc": d = auroc(y, sa) - auroc(y, sb); v = np.array([pooled(bs, sa, "auroc") - pooled(bs, sb, "auroc") for bs in BOOT])
    elif metric == "auprc": d = auprc(y, sa) - auprc(y, sb); v = np.array([pooled(bs, sa, "auprc") - pooled(bs, sb, "auprc") for bs in BOOT])
    elif metric == "mrr": d = mrr(sa) - mrr(sb); v = np.array([mrr_boot(bs, sa) - mrr_boot(bs, sb) for bs in BOOT])
    else: ca, cb = a["cred"], b["cred"]; d = ca.mean() - cb.mean(); v = np.array([ca[bs].mean() - cb[bs].mean() for bs in BOOTP])
    return d, np.percentile(v, 2.5), np.percentile(v, 97.5)

log("fitting positional models ...")
M = {}
M["POS(rank3p)"]                    = metrics(cv_scores(POSr))
M["POS(rank3p)+COMP"]               = metrics(cv_scores(hs(POSr, COMP)))
M["POS(rank3p)+SPLICE"]             = metrics(cv_scores(hs(POSr, SPL)))
M["POS(rank3p)+COMP+SPLICE+escape"] = metrics(cv_scores(hs(POSr, COMP, SPL, ESCk)))
M["POS(dist3p)"]                    = metrics(cv_scores(POSd))
M["POS(dist3p)+COMP+SPLICE+escape"] = metrics(cv_scores(hs(POSd, COMP, SPL, ESCk)))

# append rows to canonical table (idempotent)
rows = list(csv.reader(open("hypothesis_test_results.tsv"), delimiter="\t")); HDR = rows[0]
body = [r for r in rows[1:] if r[0] not in M]
for blk, m in M.items():
    body.append([blk,
        f"{m['auroc']:.5f}", f"{m['auroc_lo']:.5f}", f"{m['auroc_hi']:.5f}",
        f"{m['auprc']:.5f}", f"{m['auprc_lo']:.5f}", f"{m['auprc_hi']:.5f}",
        f"{m['mrr']:.5f}", f"{m['mrr_lo']:.5f}", f"{m['mrr_hi']:.5f}",
        "0.5", f"{BASE:.5f}", f"{CHANCE_MRR:.5f}",
        f"{m['p_at1']:.5f}", f"{m['p_at1_lo']:.5f}", f"{m['p_at1_hi']:.5f}", f"{CHANCE_P1:.5f}"])
with open("hypothesis_test_results.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(HDR)
    for r in body: w.writerow(r)

# report
def cell(m, k): return f"{m[k]:.4f} [{m[k+'_lo']:.4f},{m[k+'_hi']:.4f}]"
print("\n==== POSITION + SEQUENCE (5-fold gene-split CV) ====")
print(f"{'model':32s} {'AUROC':>22s} {'AUPRC':>8s} {'MRR':>8s} {'P@1':>22s}")
for blk in M:
    m = M[blk]; print(f"{blk:32s} {cell(m,'auroc'):>22s} {m['auprc']:8.4f} {m['mrr']:8.4f} {cell(m,'p_at1'):>22s}")
print("\nincrement over POS(rank3p) alone (paired bootstrap over genes):")
base = M["POS(rank3p)"]
for blk in ["POS(rank3p)+COMP", "POS(rank3p)+SPLICE", "POS(rank3p)+COMP+SPLICE+escape"]:
    parts = []
    for mm in ("auroc", "auprc", "mrr", "p_at1"):
        d, lo, hi = paired_inc(M[blk], base, mm); parts.append(f"{mm} {d:+.4f}[{lo:+.4f},{hi:+.4f}]")
    print(f"  {blk:32s} " + "  ".join(parts))
d, lo, hi = paired_inc(M["POS(dist3p)+COMP+SPLICE+escape"], M["POS(dist3p)"], "p_at1")
da, la, ha = paired_inc(M["POS(dist3p)+COMP+SPLICE+escape"], M["POS(dist3p)"], "auroc")
print(f"  [dist encoding] full over POS(dist3p): auroc {da:+.4f}[{la:+.4f},{ha:+.4f}]  p_at1 {d:+.4f}[{lo:+.4f},{hi:+.4f}]")
print("[out] appended POS rows to hypothesis_test_results.tsv")
