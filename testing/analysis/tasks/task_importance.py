#!/usr/bin/env python3
# feature importance for comp+splice+escape (72 feats), standardized, lambda=1.0
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
X = np.column_stack([F1[keep, 0:60], F1[keep, 66:71], ESC[keep]])
NK = len(keep); NF = X.shape[1]

NAMES = []
for w in ("up", "mid", "down"):
    for nt in "ACGT": NAMES.append(f"comp_{w}_{nt}")
    for a1 in "ACGT":
        for a2 in "ACGT": NAMES.append(f"comp_{w}_{a1}{a2}")
NAMES += ["spl_acc", "spl_dpa", "spl_strong5", "spl_max5", "spl_u1"]
NAMES += ["esc_donor", "esc_acc", "esc_present", "esc_len", "anc_donor", "anc_present", "anc_len"]
assert len(NAMES) == NF == 72
COMP_COLS = list(range(60))

genes = sorted(set(gene_k)); rs = np.random.default_rng(7)
perm = rs.permutation(len(genes)); fold_of = {g: perm[i] % NFOLD for i, g in enumerate(genes)}
fold = np.array([fold_of[g] for g in gene_k])
gp = defaultdict(list)
for i in range(NK): gp[gene_k[i]].append(i)
GEN = [np.array(v) for v in gp.values()]; NG = len(GEN)
GENP = [idx for idx in GEN if len(idx) >= 2 and (ISC[idx] == 1).any()]; NGP = len(GENP)
def hn(n): return float(np.sum(1.0 / np.arange(1, n + 1)))
CHANCE_MRR = float(np.mean([hn(len(i)) / len(i) for i in GEN]))
CHANCE_P1 = float(np.mean([1.0 / len(i) for i in GENP]))

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
def p1_credit(s):
    out = np.empty(NGP)
    for k, idx in enumerate(GENP):
        sc = s[idx]; mx = np.nanmax(sc); am = sc == mx; out[k] = float((ISC[idx][am] == 1).sum()) / am.sum()
    return out
def p1(s): return p1_credit(s).mean()

# full-model fold fits (store per fold: te, mu, sd, beta)
FOLD = []; pred_full = np.full(NK, np.nan)
for f in range(NFOLD):
    tr = fold != f; te = np.where(fold == f)[0]
    Xtr = X[tr].astype(float).copy(); mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0); sd[sd == 0] = 1.0
    Xtr = (np.where(np.isfinite(Xtr), Xtr, mu) - mu) / sd
    beta = fit_ridge(np.column_stack([np.ones(len(Xtr)), Xtr]), y[tr], RIDGE)
    FOLD.append((te, mu, sd, beta))
    Xte = X[te].astype(float).copy(); Xte = (np.where(np.isfinite(Xte), Xte, mu) - mu) / sd
    pred_full[te] = np.column_stack([np.ones(len(Xte)), Xte]) @ beta
BASE_P1 = p1(pred_full); BASE_AUROC = auroc(y, pred_full)
log(f"full model  P@1 {BASE_P1:.4f}  AUROC {BASE_AUROC:.4f}")

# item 1: permutation importance
def perm_eval(cols, seed):
    rng = np.random.default_rng(seed); pred = np.full(NK, np.nan)
    for (te, mu, sd, beta) in FOLD:
        Xte = X[te].astype(float).copy()
        pi = rng.permutation(len(te))
        Xte[:, cols] = Xte[np.ix_(pi, cols)]
        Xs = (np.where(np.isfinite(Xte), Xte, mu) - mu) / sd
        pred[te] = np.column_stack([np.ones(len(Xte)), Xs]) @ beta
    return p1(pred), auroc(y, pred)
def imp(cols, tag, reps=10):
    dp = np.empty(reps); da = np.empty(reps)
    for r in range(reps):
        pp, aa = perm_eval(cols, 1000 + r)
        dp[r] = BASE_P1 - pp; da[r] = BASE_AUROC - aa
    return dp.mean(), dp.std(), da.mean(), da.std()

GROUPS = {
    "up_mono": [0, 1, 2, 3], "mid_mono": [20, 21, 22, 23], "down_mono": [40, 41, 42, 43],
    "up_di": list(range(4, 20)), "mid_di": list(range(24, 40)), "down_di": list(range(44, 60)),
    "SPLICE:acc": [60], "SPLICE:dpa": [61], "SPLICE:strong5": [62], "SPLICE:max5": [63], "SPLICE:u1": [64],
    "escape(all)": list(range(65, 72)),
    "esc_donor": [65], "esc_acc": [66], "esc_present": [67], "esc_len": [68],
    "anc_donor": [69], "anc_present": [70], "anc_len": [71],
}
log("grouped permutation ...")
GRES = {g: imp(cols, g) for g, cols in GROUPS.items()}
log("ungrouped COMP permutation ...")
URES = {NAMES[c]: imp([c], NAMES[c]) for c in COMP_COLS}

# item 2: coefficient stability
B = np.array([fb[3][1:] for fb in FOLD])
cmean = B.mean(0); csd = B.std(0)
signflip = np.array([int(np.sum(np.sign(B[:, j]) != np.sign(cmean[j]))) for j in range(NF)])
stab = cmean / np.where(csd == 0, np.nan, csd)

# item 3: leave-one-block-out (refit)
NBOOT = 500; rsb = np.random.default_rng(11); BOOT = [rsb.integers(0, NG, NG) for _ in range(NBOOT)]
rsbp = np.random.default_rng(11); BOOTP = [rsbp.integers(0, NGP, NGP) for _ in range(NBOOT)]
def pooled_auroc(sample, s):
    peaks = np.concatenate([GEN[g] for g in sample]); return auroc(y[peaks], s[peaks])
def mrr_boot(sample, s):
    tot = 0.0
    for g in sample:
        idx = GEN[g]; sc = s[idx]; yv = y[idx]; order = np.argsort(-sc, kind="mergesort")
        ranks = np.empty(len(sc)); ranks[order] = np.arange(1, len(sc) + 1); pos = ranks[yv == 1]
        if len(pos): tot += 1.0 / pos.min()
    return tot / len(sample)
def cv_pred(cols):
    pred = np.full(NK, np.nan)
    for f in range(NFOLD):
        tr = fold != f; te = fold == f
        Xtr = X[np.ix_(tr, cols)].astype(float).copy(); Xte = X[np.ix_(te, cols)].astype(float).copy()
        mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0); sd[sd == 0] = 1.0
        Xtr = (np.where(np.isfinite(Xtr), Xtr, mu) - mu) / sd; Xte = (np.where(np.isfinite(Xte), Xte, mu) - mu) / sd
        beta = fit_ridge(np.column_stack([np.ones(len(Xtr)), Xtr]), y[tr], RIDGE)
        pred[te] = np.column_stack([np.ones(len(Xte)), Xte]) @ beta
    return pred
def paired_drop(sf, sr):
    da = auroc(y, sf) - auroc(y, sr); va = np.array([pooled_auroc(bs, sf) - pooled_auroc(bs, sr) for bs in BOOT])
    dm = mrr(sf) - mrr(sr); vm = np.array([mrr_boot(bs, sf) - mrr_boot(bs, sr) for bs in BOOT])
    cf = p1_credit(sf); cr = p1_credit(sr); dp = cf.mean() - cr.mean(); vp = np.array([cf[bs].mean() - cr[bs].mean() for bs in BOOTP])
    return (da, np.percentile(va, 2.5), np.percentile(va, 97.5),
            dp, np.percentile(vp, 2.5), np.percentile(vp, 97.5),
            dm, np.percentile(vm, 2.5), np.percentile(vm, 97.5))
BLK = {
    "COMP_mono": [0, 1, 2, 3, 20, 21, 22, 23, 40, 41, 42, 43],
    "COMP_di": list(range(4, 20)) + list(range(24, 40)) + list(range(44, 60)),
    "COMP_up": list(range(0, 20)), "COMP_mid": list(range(20, 40)), "COMP_down": list(range(40, 60)),
    "SPLICE": list(range(60, 65)), "escape": list(range(65, 72)),
}
log("leave-one-block-out refits ...")
LOBO = {}
for b, cols in BLK.items():
    keepc = [c for c in range(NF) if c not in set(cols)]
    LOBO[b] = paired_drop(pred_full, cv_pred(keepc))

print("\n(model: COMP+SPLICE+escape, 72 feats, standardized, ridge lambda=1.0, 5-fold gene-split CV)")
print(f"full model baseline:  P@1 {BASE_P1:.4f}   AUROC {BASE_AUROC:.4f}\n")
print("==== 1. GROUPED PERMUTATION IMPORTANCE (no refit; 10 repeats; drop = baseline - permuted) ====")
print(f"{'group':16s} {'dP@1 (mean+-sd)':>20s} {'dAUROC (mean+-sd)':>22s}")
order = sorted(GRES, key=lambda g: -GRES[g][0])
for g in order:
    dp, dps, da, das = GRES[g]
    print(f"{g:16s}   {dp:+.4f} +- {dps:.4f}      {da:+.4f} +- {das:.4f}")
print("\n-- ungrouped single-feature COMP permutation, TOP 10 by dP@1 --")
uo = sorted(URES, key=lambda n: -URES[n][0])[:10]
print(f"{'feature':16s} {'dP@1 (mean+-sd)':>20s} {'dAUROC (mean+-sd)':>22s}")
for n in uo:
    dp, dps, da, das = URES[n]; print(f"{n:16s}   {dp:+.4f} +- {dps:.4f}      {da:+.4f} +- {das:.4f}")
g_top_win = order[0]
best_mono_group = max(["up_mono","mid_mono","down_mono","up_di","mid_di","down_di"], key=lambda g: GRES[g][0])
print(f"\nGROUPED vs UNGROUPED: top grouped COMP effect = '{best_mono_group}' dP@1 {GRES[best_mono_group][0]:+.4f}; "
      f"top ungrouped single COMP feature dP@1 {URES[uo[0]][0]:+.4f}.")

print("\n==== 2. COEFFICIENT STABILITY (standardized coef across 5 folds; sort by |mean|/sd) ====")
idx = np.argsort(-np.abs(np.nan_to_num(stab)))
print(f"{'feature':16s} {'mean':>9s} {'sd':>8s} {'|mean|/sd':>10s} {'sign_flips':>10s}")
for j in idx[:20]:
    flag = "  UNSTABLE" if signflip[j] >= 1 else ""
    print(f"{NAMES[j]:16s} {cmean[j]:+9.4f} {csd[j]:8.4f} {np.abs(stab[j]):10.2f} {signflip[j]:10d}{flag}")
nunstable = int(np.sum(signflip >= 1))
print(f"... {nunstable}/{NF} features flip sign in >=1 fold (flagged unstable, not interpretable).")

print("\n==== 3. LEAVE-ONE-BLOCK-OUT (refit; drop = full - reduced; paired bootstrap over genes) ====")
print(f"{'block dropped':12s} {'dAUROC [95% CI]':>26s} {'dP@1 [95% CI]':>26s} {'dMRR [95% CI]':>26s}")
for b in sorted(LOBO, key=lambda b: -LOBO[b][3]):
    da, dal, dah, dp, dpl, dph, dm, dml, dmh = LOBO[b]
    print(f"{b:12s} {da:+.4f} [{dal:+.4f},{dah:+.4f}]  {dp:+.4f} [{dpl:+.4f},{dph:+.4f}]  {dm:+.4f} [{dml:+.4f},{dmh:+.4f}]")
print("\nNOTE: the per-window localiser (separate mono+di model per 100-nt window; [-100,0]=0.6256 AUROC,")
print("far-upstream windows 0.55-0.57) is the SPATIAL version of item 3 and should be read alongside it:")
print("block-drop shows which feature TYPE matters; the localiser shows WHERE along the sequence it lives.")
