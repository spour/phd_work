#!/usr/bin/env python3
# reissue hypothesis_test_results.tsv as the canonical model set + trivial baselines
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
COMP = F1[keep, 0:60]; SPL = F1[keep, 66:71]; SPL_arch = F1[keep, 66:70]; SPL_u1 = F1[keep, 70:71]
ESCk = ESC[keep]; cen = center[keep]; strd = strand[keep]
NK = len(keep); BASE = float(y.mean()); log(f"kept {NK}  base rate {BASE:.5f}")
def hs(*m): return np.column_stack(m)

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
    if npos == 0 or nneg == 0: return np.nan
    return (rr[yv == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
def auprc(yv, s):
    order = np.argsort(-s, kind="mergesort"); ys = yv[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); P = ys.sum()
    if P == 0: return np.nan
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
        sc = s[idx]; mx = np.nanmax(sc)
        if not np.isfinite(mx): out[k] = 0.0; continue
        am = sc == mx; out[k] = float((ISC[idx][am] == 1).sum()) / am.sum()
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
                p_at1=p1, p_at1_lo=np.percentile(p1v, 2.5), p_at1_hi=np.percentile(p1v, 97.5))

# trivial positional baseline score: within-gene downstream rank (comparable across genes)
dn_score = np.full(NK, np.nan)
for idx in GEN:
    c = cen[idx]; st = strd[idx][0]
    key = c if st == "+" else -c
    order = np.argsort(key, kind="mergesort"); rr = np.empty(len(idx)); rr[order] = np.arange(len(idx))
    dn_score[idx] = rr / max(len(idx) - 1, 1)
m_dn = metrics(dn_score)
m_up = metrics(-dn_score)
rch = np.random.default_rng(5); m_ch = metrics(rch.random(NK))
m_ch.update(auroc=0.5, auroc_lo=0.5, auroc_hi=0.5, auprc=BASE, auprc_lo=BASE, auprc_hi=BASE,
            mrr=CHANCE_MRR, mrr_lo=CHANCE_MRR, mrr_hi=CHANCE_MRR,
            p_at1=CHANCE_P1, p_at1_lo=CHANCE_P1, p_at1_hi=CHANCE_P1)

# fitted models
log("fitting canonical model set ...")
M = {}
M["COMP"]              = metrics(cv_scores(COMP))
M["SPLICE"]            = metrics(cv_scores(SPL))
M["SPLICE_arch"]      = metrics(cv_scores(SPL_arch))
M["SPLICE_u1"]        = metrics(cv_scores(SPL_u1))
M["COMP+SPLICE"]       = metrics(cv_scores(hs(COMP, SPL)))
M["COMP+SPLICE_arch"] = metrics(cv_scores(hs(COMP, SPL_arch)))
M["COMP+SPLICE+escape"] = metrics(cv_scores(hs(COMP, SPL, ESCk)))

# write table
ORDER = [("chance", m_ch), ("most-upstream", m_up), ("most-downstream", m_dn),
         ("COMP", M["COMP"]), ("SPLICE", M["SPLICE"]), ("SPLICE_arch", M["SPLICE_arch"]),
         ("SPLICE_u1", M["SPLICE_u1"]), ("COMP+SPLICE", M["COMP+SPLICE"]),
         ("COMP+SPLICE_arch", M["COMP+SPLICE_arch"]), ("COMP+SPLICE+escape", M["COMP+SPLICE+escape"])]
HDR = ["block", "auroc", "auroc_lo", "auroc_hi", "auprc", "auprc_lo", "auprc_hi",
       "mrr", "mrr_lo", "mrr_hi", "auroc_baseline", "auprc_baseline", "mrr_chance",
       "p_at1", "p_at1_lo", "p_at1_hi", "p_at1_chance"]
with open("hypothesis_test_results.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(HDR)
    for blk, m in ORDER:
        w.writerow([blk,
            f"{m['auroc']:.5f}", f"{m['auroc_lo']:.5f}", f"{m['auroc_hi']:.5f}",
            f"{m['auprc']:.5f}", f"{m['auprc_lo']:.5f}", f"{m['auprc_hi']:.5f}",
            f"{m['mrr']:.5f}", f"{m['mrr_lo']:.5f}", f"{m['mrr_hi']:.5f}",
            "0.5", f"{BASE:.5f}", f"{CHANCE_MRR:.5f}",
            f"{m['p_at1']:.5f}", f"{m['p_at1_lo']:.5f}", f"{m['p_at1_hi']:.5f}", f"{CHANCE_P1:.5f}"])

# u1 verdict
du_auroc = M["COMP+SPLICE"]["auroc"] - M["COMP+SPLICE_arch"]["auroc"]
du_p1 = M["COMP+SPLICE"]["p_at1"] - M["COMP+SPLICE_arch"]["p_at1"]
su_auroc = M["SPLICE"]["auroc"] - M["SPLICE_arch"]["auroc"]

def row(blk, m): return (f"{blk:20s} {m['auroc']:.4f} [{m['auroc_lo']:.4f},{m['auroc_hi']:.4f}]  "
    f"{m['auprc']:.4f}  {m['mrr']:.4f}  {m['p_at1']:.4f} [{m['p_at1_lo']:.4f},{m['p_at1_hi']:.4f}]")
print("\n==== REISSUED hypothesis_test_results.tsv (PAS & ARCH dropped permanently) ====")
print(f"{'block':20s} {'AUROC [95% CI]':>22s}  {'AUPRC':>6s}  {'MRR':>6s}  {'P@1 [95% CI]':>22s}")
for blk, m in ORDER: print(row(blk, m))
print(f"\nU1 contribution:  COMP+SPLICE - COMP+SPLICE_arch  = AUROC {du_auroc:+.4f}, P@1 {du_p1:+.4f}")
print(f"                  SPLICE     - SPLICE_arch        = AUROC {su_auroc:+.4f}")
print(f"base rate {BASE:.4f} ; chance MRR {CHANCE_MRR:.4f} ; chance P@1 {CHANCE_P1:.4f}")
print("[out] hypothesis_test_results.tsv  (old -> hypothesis_test_results_pre_canonical_20260730.tsv)")
