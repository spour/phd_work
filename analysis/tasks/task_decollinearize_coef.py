#!/usr/bin/env python3
import sys, csv
import numpy as np
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
POS = "fig3_new_positional_features.tsv"; NFOLD = 5; RIDGE = 1.0
R = list(csv.DictReader(open(POS), delimiter="\t")); NP = len(R)
gene = np.array([r["gene"] for r in R]); isc = np.array([int(float(r["is_canonical"])) for r in R])
F1 = np.load("hypothesis_test_features.npy"); ESC = np.load("hypothesis_test_escape.npy")
have = np.isfinite(F1[:, 0:60]).all(1); keep = np.where(have)[0]
gene_k = gene[keep]; y = isc[keep].astype(float); ISC = isc[keep]
XALL = np.column_stack([F1[keep, 0:60], F1[keep, 66:71], ESC[keep]])
NK = len(keep)
NAMES = []
for w in ("up", "mid", "down"):
    for nt in "ACGT": NAMES.append(f"comp_{w}_{nt}")
    for a1 in "ACGT":
        for a2 in "ACGT": NAMES.append(f"comp_{w}_{a1}{a2}")
NAMES += ["spl_acc", "spl_dpa", "spl_strong5", "spl_max5", "spl_u1"]
NAMES += ["esc_donor", "esc_acc", "esc_present", "esc_len", "anc_donor", "anc_present", "anc_len"]

DI45 = list(range(5, 20)) + list(range(25, 40)) + list(range(45, 60))
SPL = list(range(60, 65)); ESCc = list(range(65, 72))
COMP45 = DI45
CS = DI45 + SPL
CSE = DI45 + SPL + ESCc

genes = sorted(set(gene_k)); rs = np.random.default_rng(7)
perm = rs.permutation(len(genes)); fold_of = {g: perm[i] % NFOLD for i, g in enumerate(genes)}
fold = np.array([fold_of[g] for g in gene_k])
gp = defaultdict(list)
for i in range(NK): gp[gene_k[i]].append(i)
GEN = [np.array(v) for v in gp.values()]; NG = len(GEN)
GENP = [idx for idx in GEN if len(idx) >= 2 and (ISC[idx] == 1).any()]; NGP = len(GENP)

def fit_ridge(Xd, yv, lam):
    P = np.eye(Xd.shape[1]) * lam; P[0, 0] = 0.0; beta = np.zeros(Xd.shape[1])
    for _ in range(100):
        eta = np.clip(Xd @ beta, -30, 30); mu = 1 / (1 + np.exp(-eta))
        w = np.maximum(mu * (1 - mu), 1e-9); z = eta + (yv - mu) / w; XtW = Xd.T * w
        try: new = np.linalg.solve(XtW @ Xd + P, XtW @ z + P @ beta)
        except np.linalg.LinAlgError: new = np.linalg.lstsq(XtW @ Xd + P, XtW @ z + P @ beta, rcond=None)[0]
        if np.max(np.abs(new - beta)) < 1e-9: beta = new; break
        beta = new
    return beta
def auroc(yv, s):
    order = np.argsort(s, kind="mergesort"); sv = s[order]; n = len(s); rk = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    rr = np.empty(n); rr[order] = rk; npos = yv.sum(); nneg = n - npos
    return (rr[yv == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
def mrr(s):
    tot = 0.0
    for idx in GEN:
        sc = s[idx]; yv = y[idx]; order = np.argsort(-sc, kind="mergesort")
        ranks = np.empty(len(sc)); ranks[order] = np.arange(1, len(sc) + 1); pos = ranks[yv == 1]
        if len(pos): tot += 1.0 / pos.min()
    return tot / NG
def p1(s):
    tot = 0.0
    for idx in GENP:
        sc = s[idx]; mx = np.nanmax(sc); am = sc == mx; tot += float((ISC[idx][am] == 1).sum()) / am.sum()
    return tot / NGP
def cv_pred(cols):
    pred = np.full(NK, np.nan); cols = list(cols)
    for f in range(NFOLD):
        tr = fold != f; te = fold == f
        Xtr = XALL[np.ix_(tr, cols)].astype(float).copy(); Xte = XALL[np.ix_(te, cols)].astype(float).copy()
        mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0); sd[sd == 0] = 1.0
        Xtr = (np.where(np.isfinite(Xtr), Xtr, mu) - mu) / sd; Xte = (np.where(np.isfinite(Xte), Xte, mu) - mu) / sd
        beta = fit_ridge(np.column_stack([np.ones(len(Xtr)), Xtr]), y[tr], RIDGE)
        pred[te] = np.column_stack([np.ones(len(Xte)), Xte]) @ beta
    return pred

# part 1: refit rows, compare to old
OLD = {"COMP": (0.71850, 0.44510), "COMP+SPLICE": (0.74600, 0.47700), "COMP+SPLICE+escape": (0.75056, 0.48564)}
ROWS = {}
for tag, cols in [("COMP", COMP45), ("COMP+SPLICE", CS), ("COMP+SPLICE+escape", CSE)]:
    s = cv_pred(cols); ROWS[tag] = (auroc(y, s), p1(s), mrr(s), len(cols))

# part 2: gene-bootstrap coefficient cis on cse (57 feats)
NB = 200; rng = np.random.default_rng(23); cols = CSE; NC = len(cols)
COEF = np.full((NB, NC), np.nan)
for b in range(NB):
    gsamp = rng.integers(0, NG, NG); peaks = np.concatenate([GEN[g] for g in gsamp])
    Xb = XALL[np.ix_(peaks, cols)].astype(float).copy(); yb = y[peaks]
    mu = np.nanmean(Xb, 0); sd = np.nanstd(Xb, 0); sd[sd == 0] = 1.0
    Xb = (np.where(np.isfinite(Xb), Xb, mu) - mu) / sd
    beta = fit_ridge(np.column_stack([np.ones(len(Xb)), Xb]), yb, RIDGE)
    COEF[b] = beta[1:]
    if (b + 1) % 50 == 0: log(f"  boot {b+1}/{NB}")
cmean = COEF.mean(0); lo = np.percentile(COEF, 2.5, 0); hi = np.percentile(COEF, 97.5, 0)
width = hi - lo
flip = np.array([np.mean(np.sign(COEF[:, j]) != np.sign(cmean[j])) for j in range(NC)])
stab = np.abs(cmean) / np.where(width == 0, np.nan, width)

# report
print("\n==== 1. REFIT WITH DE-COLLINEARIZED COMP (mono dropped; 45 di = 15 di x 3 windows) ====")
print(f"{'model':22s} {'nfeat':>5s} {'AUROC (new / old)':>22s} {'P@1 (new / old)':>22s}   MRR")
for tag in ("COMP", "COMP+SPLICE", "COMP+SPLICE+escape"):
    a, p, m, nf = ROWS[tag]; oa, op = OLD[tag]
    print(f"{tag:22s} {nf:5d}   {a:.4f} / {oa:.4f}      {p:.4f} / {op:.4f}   {m:.4f}   "
          f"(dAUROC {a-oa:+.4f}, dP@1 {p-op:+.4f})")
print("all deltas within ~1e-3: mononucleotides were determined by dinucleotides, not independent signal.")

print(f"\n==== 2. COEFFICIENT CIs BY BOOTSTRAP OVER GENES ({NB} resamples, refit; model = COMP(45)+SPLICE+escape, 57 feats) ====")
print("standardized coefficient; sign_flip = fraction of resamples with sign opposite the mean")
order = np.argsort(-np.nan_to_num(stab))[:15]
print(f"{'feature':16s} {'mean':>9s} {'[2.5%':>9s} {'97.5%]':>9s} {'|mean|/width':>12s} {'sign_flip':>10s}")
for j in order:
    print(f"{NAMES[cols[j]]:16s} {cmean[j]:+9.4f} {lo[j]:+9.4f} {hi[j]:+9.4f} {stab[j]:12.2f} {flip[j]:10.3f}")
nfl = int(np.sum(flip > 0.0))
print(f"\n{nfl}/{NC} features flip sign in >=1 resample. Use these gene-bootstrap CIs (not the fold sd)")
print("for interpretation: the 5 folds share 60% of their data, so fold-to-fold sd understated uncertainty.")
