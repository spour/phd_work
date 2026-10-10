#!/usr/bin/env python3
# the last two questions about the splice comparator
import sys, math, csv
import numpy as np
import binned_common as K
from binned_common import log

UP, DN, BIN = 2000, 500, 100
NBIN = (UP + DN) // BIN
ACC = K.ACC
G = K.C0["gene"][K.ok]; CHR = K.C0["chrom"][K.ok]
CEN = K.C0["center"][K.ok].astype(np.int64); STR = K.C0["strand"][K.ok]
N = len(K.y)

Z = np.load("binned_profiles.npz", allow_pickle=True)
P = Z["P"]; CHN = [str(c) for c in Z["channels"]]
U1B = np.column_stack([P[:, CHN.index("u1_dens4"), :], P[:, CHN.index("u1_dens7"), :]])
ME3B = P[:, CHN.index("me3_max"), :]
FINAL = np.column_stack([U1B, ACC])
assert FINAL.shape[1] == 51

# gc dip depth, joined on the peak key
DEPTH_COL = "depth_[-2000,-1000)_mean [-150,0)"
dip = {}
with open("gc_dip_depth.tsv") as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        if r["class"] == "decoy": continue
        dip[(r["gene"], r["chrom"], int(r["center"]), r["strand"])] = float(r[DEPTH_COL])
DIP = np.array([dip.get((G[i], CHR[i], int(CEN[i]), STR[i]), np.nan) for i in range(N)])
log(f"[dip] {DEPTH_COL}: joined {int(np.isfinite(DIP).sum())}/{N} analysis-set peaks")
assert np.isfinite(DIP).mean() > 0.99, "dip-depth join is incomplete"

# gc per 100 nt bin, from the same token array everything else uses
D = np.load("cnn_data.npz", allow_pickle=True); X = D["X"]
assert X.shape == (N, UP + DN)
t = X.reshape(N, NBIN, BIN)
isgc = ((t == 1) | (t == 2)).sum(2).astype(float)
nok = (t < 4).sum(2).astype(float)
GCB = np.where(nok > 0, isgc / np.maximum(nok, 1), np.nan)
log(f"[gc] GC fraction in {NBIN} bins of {BIN} nt, mean {np.nanmean(GCB):.4f}")

# part 1
REPORT = {}
def report(M, tag, lam=None):
    if lam is None: lam, iv = K.pick_lambda(M, tag)
    else: iv = np.nan
    sc, _ = K.fit_score(M, K.trv, K.te, lam)
    r = K.suite(sc); r.update(lam=lam, iv=iv, p=M.shape[1]); REPORT[tag] = r
    log(f"[1] {tag:38s} p={M.shape[1]:>4d}  within-gene {r['winrate']:.4f} "
        f"[{r['winrate_lo']:.4f},{r['winrate_hi']:.4f}]  P@1 {r['p_at_1']:.4f} "
        f"[{r['p_at_1_lo']:.4f},{r['p_at_1_hi']:.4f}]")
    return r

BASE = "locked model (51)"
r0 = report(FINAL, BASE)
assert abs(r0["winrate"] - 0.6957) < 5e-4, f"locked model not reproduced ({r0['winrate']:.4f})"
log("[check] the locked 51-feature model reproduces 0.6957 before anything is added\n")

ADDS = [("+ GC dip depth (1)", np.column_stack([FINAL, DIP])),
        ("+ GC per bin (25)", np.column_stack([FINAL, GCB])),
        ("+ acceptor per bin (25)", np.column_stack([FINAL, ME3B])),
        ("+ all three (51)", np.column_stack([FINAL, DIP, GCB, ME3B]))]
for tag, M in ADDS: report(M, tag)

def paired(a, b, tag):
    da = REPORT[a]["_wg"][K.BS].mean(1) - REPORT[b]["_wg"][K.BS].mean(1)
    d = REPORT[a]["winrate"] - REPORT[b]["winrate"]
    lo, hi = np.percentile(da, [2.5, 97.5])
    p = max(2 * min((da <= 0).mean(), (da >= 0).mean()), 1 / K.NBOOT)
    log(f"[paired] {tag:40s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  p = {p:.4f}"
        f"{'' if lo * hi > 0 else '   (CI spans 0)'}")
    return d, lo, hi, p
log("")
PR = {tag: paired(tag, BASE, f"{tag} minus locked") for tag, _ in ADDS}
CLEARS = [t for t, (d, lo, hi, p) in PR.items() if d > 0.005 and lo > 0]
BESTT = max(PR, key=lambda t: PR[t][0])
log(f"[1] VERDICT: " + (
    f"{', '.join(CLEARS)} clears +0.005 with a CI clear of zero; the ladder changes"
    if CLEARS else
    f"nothing clears +0.005. The best addition is {BESTT} at {PR[BESTT][0]:+.4f} "
    f"[{PR[BESTT][1]:+.4f}, {PR[BESTT][2]:+.4f}]; the ladder is unchanged and the locked "
    f"51-feature model stands"))
with open("splice_sweep_features.tsv", "w") as fh:
    fh.write("model\tn_features\tlambda\tinner_val\twinrate\tci_lo\tci_hi\tp_at_1\tp1_lo\tp1_hi\t"
             "auroc\tauprc\tdelta_vs_locked\td_lo\td_hi\td_p\tclears_0.005\n")
    for tag in [BASE] + [t for t, _ in ADDS]:
        r = REPORT[tag]; d = PR.get(tag)
        iv_s = "" if not np.isfinite(r["iv"]) else f"{r['iv']:.5f}"
        fh.write(f"{tag}\t{r['p']}\t{r['lam']:g}\t{iv_s}\t"
                 f"{r['winrate']:.5f}\t{r['winrate_lo']:.5f}\t{r['winrate_hi']:.5f}\t"
                 f"{r['p_at_1']:.5f}\t{r['p_at_1_lo']:.5f}\t{r['p_at_1_hi']:.5f}\t"
                 f"{r['auroc']:.5f}\t{r['auprc']:.5f}\t"
                 + ("\t\t\t\t\n" if d is None else
                    f"{d[0]:+.5f}\t{d[1]:+.5f}\t{d[2]:+.5f}\t{d[3]:.4f}\t"
                    f"{'yes' if (d[0] > 0.005 and d[1] > 0) else 'no'}\n"))

# part 2
log("")
CS = np.load("cnn_model_scores.npz", allow_pickle=True)
assert (CS["gene"][K.ok] == K.gene).all() and (CS["y"][K.ok] == K.y).all()
cnn_te = CS["logit"][K.ok][K.te].astype(float)
spl_te, _ = K.fit_score(FINAL, K.trv, K.te, REPORT[BASE]["lam"])

def top1(sc):
    hit = np.zeros(len(K.GRP_TE), bool); tie = 0
    for j, (r, c) in enumerate(K.GRP_TE):
        m = sc[r].max(); k = sc[c].max()
        hit[j] = m > k
        if m == k: tie += 1
    return hit, tie
HS, tS = top1(spl_te); HC, tC = top1(cnn_te)
log(f"[2] strict-top1 ties: splice {tS}, CNN {tC} of {len(K.GRP_TE)} genes (counted as misses)")
SETS = {"both right": HS & HC, "CNN only": (~HS) & HC,
        "splice only": HS & (~HC), "both wrong": (~HS) & (~HC)}
for k, v in SETS.items():
    log(f"[2] {k:12s} {int(v.sum()):>5d} genes  ({100*v.mean():5.1f}%)")
log(f"[2] splice strict P@1 {HS.mean():.4f}   CNN strict P@1 {HC.mean():.4f}")

GLEN = {}
with open("../train/data/annotation/gencode.v49.annotation.gtf") as fh:
    for line in fh:
        if line[0] == "#": continue
        f = line.split("\t", 9)
        if f[2] != "gene": continue
        a = f[8]; i = a.find('gene_name "')
        if i < 0: continue
        nm = a[i + 11:a.index('"', i + 11)]
        L = int(f[4]) - int(f[3])
        if L > GLEN.get(nm, 0): GLEN[nm] = L
log(f"[2] GENCODE gene spans for {len(GLEN):,} gene names")

tpos = np.where(STR == "+", CEN, -CEN).astype(float)
u1tot = np.nanmean(P[:, CHN.index("u1_dens4"), :], 1)
prop = {k: np.full(len(K.GRP_TE), np.nan) for k in
        ("acceptor of the real end", "competitors in the gene", "GC dip depth of the real end",
         "U1 density of the real end", "gene span (GENCODE, kb)",
         "distance to nearest competitor (kb)")}
gnames = []
for j, (r, c) in enumerate(K.GRP_TE):
    ri = K.te[r[0]]; ci = K.te[c]
    g = G[ri]; gnames.append(g)
    prop["acceptor of the real end"][j] = ACC[ri]
    prop["competitors in the gene"][j] = len(c)
    prop["GC dip depth of the real end"][j] = DIP[ri]
    prop["U1 density of the real end"][j] = u1tot[ri]
    prop["gene span (GENCODE, kb)"][j] = GLEN.get(g, np.nan) / 1000.0
    prop["distance to nearest competitor (kb)"][j] = np.min(np.abs(tpos[ci] - tpos[ri])) / 1000.0

def ranks(a):
    o = np.argsort(a, kind="mergesort"); s = a[o]; n = len(a); rk = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    out = np.empty(n); out[o] = rk; return out

def mannwhitney(x, y):
    a = np.concatenate([x, y]); rk = ranks(a); n1, n2 = len(x), len(y)
    U = rk[:n1].sum() - n1 * (n1 + 1) / 2.0
    auc = U / (n1 * n2)
    _, cnt = np.unique(a, return_counts=True)
    n = n1 + n2
    var = n1 * n2 / 12.0 * ((n + 1) - np.sum(cnt ** 3 - cnt) / (n * (n - 1.0)))
    if var <= 0: return auc, np.nan, np.nan
    z = (U - n1 * n2 / 2.0) / math.sqrt(var)
    return auc, z, math.erfc(abs(z) / math.sqrt(2))

A, B = SETS["CNN only"], SETS["both right"]
NP_ = len(prop); ALPHA = 0.05 / NP_
log(f"\n[2] CNN-only-right ({int(A.sum())} genes) against both-right ({int(B.sum())} genes)")
log(f"[2] AUC is P(CNN-only gene has the larger value); Bonferroni threshold {ALPHA:.4f} "
    f"for {NP_} properties")
rows = []
for k, v in prop.items():
    x = v[A][np.isfinite(v[A])]; y = v[B][np.isfinite(v[B])]
    auc, z, p = mannwhitney(x, y)
    sig = "***" if p < ALPHA else ("*" if p < 0.05 else "")
    log(f"[2]   {k:38s} AUC {auc:.3f}  median {np.median(x):8.3f} vs {np.median(y):8.3f}  "
        f"p = {p:.2g} {sig}")
    rows.append((k, len(x), len(y), auc, float(np.median(x)), float(np.median(y)), z, p, sig))
HITS = [r for r in rows if r[7] < ALPHA]
BIG = [r for r in HITS if abs(r[3] - 0.5) >= 0.10]
log("\n[2] VERDICT: " + (
    "the residual DOES localise: " + "; ".join(
        f"{r[0]} (AUC {r[3]:.3f})" for r in BIG)
    if BIG else
    ("the residual does not localise to any named property with a usable effect size. " +
     (f"{len(HITS)} of {NP_} properties pass Bonferroni, but every AUC is within 0.10 of 0.5 "
      f"(largest |AUC-0.5| = {max(abs(r[3]-0.5) for r in rows):.3f}), which on "
      f"{int(A.sum())} against {int(B.sum())} genes is a shift too small to name a feature. "
      if HITS else
      f"No property passes Bonferroni; the largest |AUC-0.5| is "
      f"{max(abs(r[3]-0.5) for r in rows):.3f}. ") +
     "The sequence advantage does not reduce to an interpretable property measured here.")))
with open("splice_sweep_partition.tsv", "w") as fh:
    fh.write("section\tkey\tn_cnn_only\tn_both_right\tauc\tmedian_cnn_only\tmedian_both_right\t"
             "z\tp\tsignificant\n")
    for k, v in SETS.items():
        fh.write(f"partition\t{k}\t{int(v.sum())}\t\t\t\t\t\t\t\n")
    fh.write(f"partition\tsplice strict P@1\t{HS.mean():.5f}\t\t\t\t\t\t\t\n")
    fh.write(f"partition\tCNN strict P@1\t{HC.mean():.5f}\t\t\t\t\t\t\t\n")
    fh.write(f"partition\tBonferroni alpha\t{ALPHA:.5f}\t\t\t\t\t\t\t\n")
    for k, n1, n2, auc, m1, m2, z, p, sig in rows:
        fh.write(f"property\t{k}\t{n1}\t{n2}\t{auc:.5f}\t{m1:.5f}\t{m2:.5f}\t"
                 f"{z:.4f}\t{p:.3g}\t{sig}\n")
log(f"\n[genes] {len(K.GRP_TE)} test genes, {len(K.te)} test peaks")
log("[out] splice_sweep_features.tsv splice_sweep_partition.tsv")
