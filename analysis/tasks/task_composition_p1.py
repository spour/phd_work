#!/usr/bin/env python3
# add p@1 (top-1 accuracy) to every row of hypothesis_test_results.tsv
import sys, csv
import numpy as np
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
POS = "fig3_new_positional_features.tsv"; NFOLD = 5; RIDGE = 1.0

R = list(csv.DictReader(open(POS), delimiter="\t"))
NP = len(R)
gene = np.array([r["gene"] for r in R]); isc = np.array([int(float(r["is_canonical"])) for r in R])
y = isc.astype(float)
F1 = np.load("hypothesis_test_features.npy")
F2 = np.load("hypothesis_test_features2.npy")
F3 = np.load("hypothesis_test_features3.npy")
CNT = F3[:, :200].reshape(NP, 50, 4)

def merge_freq(nm):
    g = 50 // nm; m = CNT[:, :g*nm, :].reshape(NP, g, nm, 4).sum(2); return (m / (nm*50)).reshape(NP, g*4)
X_50 = merge_freq(1); X_25 = merge_freq(2); X_10 = merge_freq(5)
X_3 = np.column_stack([CNT[:,0:30,:].sum(1)/1500.0, CNT[:,30:40,:].sum(1)/500.0, CNT[:,40:50,:].sum(1)/500.0])
FINE = np.arange(4, 104)

def hs(*ms): return np.column_stack(ms)
BLOCKX = {
    "COMP": F1[:, 0:60], "PAS": F1[:, [60,61,62,63]], "ARCH": F1[:, [64,65]],
    "SPLICE": F1[:, 66:71], "COMP+PAS": F1[:, 0:64], "COMP+SPLICE": hs(F1[:, 0:60], F1[:, 66:71]),
    "COMP+PAS+ARCH": F1[:, 0:66], "ALL": F1[:, 0:71],
    "COMP_fine": F2[:, 4:104], "COMP_fine+SPLICE": hs(F2[:, 4:104], F1[:, 66:71]),
    "COMP_mono_3win": X_3, "COMP_mono_10bin": X_10, "COMP_mono_25bin": X_25, "COMP_mono_50bin": X_50,
    "PAS_pres+bestscore": F1[:, [60, 62]],
}

genes = sorted(set(gene)); rs = np.random.default_rng(7)
perm = rs.permutation(len(genes)); fold_of = {g: perm[k] % NFOLD for k, g in enumerate(genes)}
fold = np.array([fold_of[g] for g in gene])
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
    pred = np.full(NP, np.nan)
    for f in range(NFOLD):
        tr = fold != f; te = fold == f
        Xtr = X[tr].astype(float).copy(); Xte = X[te].astype(float).copy()
        mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0); sd[sd == 0] = 1.0
        Xtr = np.where(np.isfinite(Xtr), Xtr, mu); Xte = np.where(np.isfinite(Xte), Xte, mu)
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        Xtr = np.column_stack([np.ones(Xtr.shape[0]), Xtr]); Xte = np.column_stack([np.ones(Xte.shape[0]), Xte])
        beta = fit_ridge(Xtr, y[tr], RIDGE); pred[te] = Xte @ beta
    return pred

gp = defaultdict(list)
for i in range(NP): gp[gene[i]].append(i)
GEN = [np.array(v) for v in gp.values() if len(v) >= 2 and (isc[np.array(v)] == 1).any()]
NG = len(GEN)
CHANCE = float(np.mean([1.0 / len(idx) for idx in GEN]))
rsp = np.random.default_rng(3)
perm_ch = np.mean([np.mean([1.0 if isc[idx[rsp.integers(0, len(idx))]] == 1 else 0.0 for idx in GEN]) for _ in range(200)])

def p_at1_pergene(s):
    out = np.empty(NG)
    for k, idx in enumerate(GEN):
        sc = s[idx]; mx = np.nanmax(sc)
        if not np.isfinite(mx): out[k] = 0.0; continue
        atmax = sc == mx
        out[k] = float((isc[idx][atmax] == 1).sum()) / atmax.sum()
    return out
NBOOT = 500; rsb = np.random.default_rng(11); BS = [rsb.integers(0, NG, NG) for _ in range(NBOOT)]
def p1_ci(cred):
    v = np.array([cred[bs].mean() for bs in BS]); return cred.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5)

P1 = {}
for nm, X in BLOCKX.items():
    cred = p_at1_pergene(cv_scores(X)); P1[nm] = p1_ci(cred)
    log(f"  {nm:18s} P@1 {P1[nm][0]:.4f} [{P1[nm][1]:.4f},{P1[nm][2]:.4f}]")
log(f"chance mean(1/n) {CHANCE:.4f} ; permutation {perm_ch:.4f}")

rows = list(csv.reader(open("hypothesis_test_results.tsv"), delimiter="\t"))
hdr = rows[0] + ["p_at1", "p_at1_lo", "p_at1_hi", "p_at1_chance"]
out = [hdr]
for r in rows[1:]:
    blk = r[0]; p, lo, hi = P1[blk]
    out.append(r + [f"{p:.5f}", f"{lo:.5f}", f"{hi:.5f}", f"{CHANCE:.5f}"])
with open("hypothesis_test_results.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    for r in out: w.writerow(r)

LD = {r["rule"]: r for r in csv.DictReader(open("arch_ladder.tsv"), delimiter="\t")}
acc_rule = LD["c. acceptor>=8"]; term_rule = LD["d. terminal-like @8 (rule1)"]

# comparison table (p@1 only), fitted blocks + rules on one axis
comp = [
    ("chance (1/n)",         CHANCE, CHANCE, CHANCE, "analytic/permutation"),
    ("acceptor-only rule",   float(acc_rule["top1_acc"]), float(acc_rule["top1_lo"]), float(acc_rule["top1_hi"]), "rule, full-data threshold"),
    ("terminal-like@8 rule", float(term_rule["top1_acc"]), float(term_rule["top1_lo"]), float(term_rule["top1_hi"]), "rule, full-data threshold"),
    ("COMP",   *P1["COMP"],   "fitted, gene-split CV"),
    ("SPLICE", *P1["SPLICE"], "fitted, gene-split CV"),
    ("PAS",    *P1["PAS"],    "fitted, gene-split CV"),
    ("ALL",    *P1["ALL"],    "fitted, gene-split CV"),
]
with open("p1_comparison.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(["row", "p_at1", "p_at1_lo", "p_at1_hi", "kind"])
    for nm, p, lo, hi, k in comp: w.writerow([nm, f"{p:.5f}", f"{lo:.5f}", f"{hi:.5f}", k])

print("\n==== P@1 COMPARISON (top-1 accuracy; chance = mean(1/n) = 0.2695, permutation-verified) ====")
print(f"{'row':22s} {'P@1':>7s} {'95% CI':>18s}   provenance")
for nm, p, lo, hi, k in comp:
    ci = f"[{lo:.3f},{hi:.3f}]" if nm != "chance (1/n)" else ""
    print(f"{nm:22s} {p:7.4f} {ci:>18s}   {k}")
print("\nNOTE: the two rule rows use a threshold chosen on the FULL data (no held-out split), which")
print("favours them; the fitted blocks are held-out gene-split CV. So the rules' edge is optimistic.")
print("\n[out] hypothesis_test_results.tsv (+p_at1 cols), p1_comparison.tsv")
