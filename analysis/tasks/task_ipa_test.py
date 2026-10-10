#!/usr/bin/env python3
# ipa contrast, part 3: what distinguishes a used intronic poly(a) site from a matched
import csv, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, HOLD_FRAC, RIDGE, NBOOT, NPERM = 42, 0.20, 1.0, 120, 200
rng = np.random.default_rng(SEED)
K3 = ["spl_upstream_max_me3", "spl_pas_prox_u1", "spl_u1_suppression_up"]

M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
PB = np.load("binned_features178.npz"); BN = [str(s) for s in PB["names"]]; PBX = PB["X"][ridx].astype(float)
DB = np.load("ipa_decoy_binned.npz"); assert [str(s) for s in DB["names"]] == BN
DBX = DB["X"].astype(float)
KP = np.load("ipa_kin_pos.npz"); KN = [str(s) for s in KP["names"]]; KPX = KP["X"].astype(float)
KD = np.load("ipa_kin_dec.npz"); KDX = KD["X"].astype(float)
POS = list(csv.DictReader(open("ipa_positions.tsv"), delimiter="\t"))
DEC = list(csv.DictReader(open("ipa_decoy_positions.tsv"), delimiter="\t"))
prow = np.array([int(r["row"]) for r in POS])
usage = np.array([float(r["usage"]) if r["usage"] else np.nan for r in POS])
GCB = [BN.index(f"bin{b}_gc") for b in range(1, 26)]
K3I = [BN.index(n) for n in K3]

NAMES = KN + K3 + [BN[j] for j in GCB]
XP = np.column_stack([KPX, PBX[np.ix_(prow, K3I)], PBX[np.ix_(prow, GCB)]])
XD = np.column_stack([KDX, DBX[:, K3I], DBX[:, GCB]])
X = np.vstack([XP, XD])
y = np.r_[np.ones(len(XP)), np.zeros(len(XD))]
gene = np.r_[np.array([r["gene"] for r in POS]), np.array([r["gene"] for r in DEC])]
chrom = np.r_[np.array([r["chrom"] for r in POS]), np.array([r["chrom"] for r in DEC])]
pmap = {v: i for i, v in enumerate(prow)}
setid = np.r_[np.arange(len(POS)), np.array([pmap[int(r["competitor_row"])] for r in DEC])]
log(f"{len(XP)} positives, {len(XD)} decoys, {X.shape[1]} features")

si = defaultdict(list)
for i in range(len(y)): si[setid[i]].append(i)
si = {k: np.array(v) for k, v in si.items()}
SETS = [(k, ix) for k, ix in si.items() if (y[ix] == 1).sum() == 1 and len(ix) >= 3]
sgene = {k: gene[ix[0]] for k, ix in SETS}
allg = np.array(sorted(set(sgene.values())))
hold_g = set(rng.choice(allg, size=int(round(HOLD_FRAC * len(allg))), replace=False))
CHROMS = sorted(set(chrom))

# per-feature
def auroc(yv, s):
    m = np.isfinite(s); yv = yv[m]; s = s[m]
    o = np.argsort(s, kind="mergesort"); sv = s[o]; n = len(s); rk = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    rr = np.empty(n); rr[o] = rk; npos = yv.sum(); nneg = n - npos
    if npos == 0 or nneg == 0: return np.nan
    return float((rr[yv == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def winrates(v):
    out = np.full(len(SETS), np.nan)
    for t, (k, ix) in enumerate(SETS):
        s = v[ix]; c = int(np.argmax(y[ix])); o = np.delete(np.arange(len(ix)), c)
        if not np.isfinite(s[c]): continue
        d = s[o][np.isfinite(s[o])]
        if len(d) == 0: continue
        out[t] = float(np.mean((s[c] > d) + 0.5 * (s[c] == d)))
    return out
def perm_win(v, nperm=NPERM):
    vals = []
    for k, ix in SETS:
        s = v[ix][np.isfinite(v[ix])]
        if len(s) < 2: continue
        vals.append(s)
    out = np.empty(nperm)
    for t in range(nperm):
        w = []
        for s in vals:
            c = rng.integers(0, len(s)); o = np.delete(np.arange(len(s)), c)
            w.append(np.mean((s[c] > s[o]) + 0.5 * (s[c] == s[o])))
        out[t] = np.mean(w)
    return float(out.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
gk = defaultdict(list)
for t, (k, ix) in enumerate(SETS): gk[sgene[k]].append(t)
GKEYS = list(gk); GIX = [np.array(gk[g]) for g in GKEYS]
BOOTG = rng.integers(0, len(GKEYS), size=(NBOOT, len(GKEYS)))
print(f"\n===== PER FEATURE: used intronic PAS vs its 5 matched intronic decoys "
      f"({len(SETS)} sets, {len(GKEYS)} genes) =====")
print(f"{'feature':<22} {'AUROC':>7} {'win-rate':>9} {'95% CI':>17} {'perm null':>11} {'excess':>8}")
with open("ipa_features.tsv", "w") as fh:
    fh.write("feature\tblock\tauroc\twinrate\tci_lo\tci_hi\tperm\tperm_lo\tperm_hi\texcess\n")
    for j, nm in enumerate(NAMES):
        blk = "kinetic" if nm.startswith("kin_") else ("three features" if nm in K3 else "GC-binned")
        if blk == "GC-binned" and nm not in ("bin20_gc", "bin1_gc", "bin13_gc"): continue
        v = X[:, j]
        w = winrates(v); wm = float(np.nanmean(w))
        wb = np.array([np.nanmean(w[np.concatenate([GIX[q] for q in BOOTG[t]])]) for t in range(NBOOT)])
        pm, plo, phi = perm_win(v, 200)
        au = auroc(y, v)
        print(f"{nm:<22} {au:>7.4f} {wm:>9.4f} [{np.percentile(wb,2.5):.4f},"
              f"{np.percentile(wb,97.5):.4f}] {pm:>11.4f} {wm-pm:>+8.4f}")
        fh.write(f"{nm}\t{blk}\t{au:.5f}\t{wm:.5f}\t{np.percentile(wb,2.5):.5f}\t"
                 f"{np.percentile(wb,97.5):.5f}\t{pm:.5f}\t{plo:.5f}\t{phi:.5f}\t{wm-pm:.5f}\n")

# blocks
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
    return (np.column_stack([np.ones(len(Xtr)), (np.where(np.isfinite(Xtr), Xtr, mu) - mu) / sd]),
            np.column_stack([np.ones(len(Xte)), (np.where(np.isfinite(Xte), Xte, mu) - mu) / sd]))
is_hold = np.array([gene[i] in hold_g for i in range(len(y))])
def loco_hold(cols):
    Xs = X[:, cols]; pred = np.full(len(y), np.nan)
    for c in CHROMS:
        te = is_hold & (chrom == c); tr = ~is_hold & (chrom != c)
        if te.sum() == 0 or tr.sum() == 0: continue
        A, B = prep(Xs[tr], Xs[te]); pred[te] = B @ fit_ridge(A, y[tr])
    return pred
def per_set(pred):
    h, rr, ns, sl = [], [], [], []
    for k, ix in SETS:
        if not is_hold[ix[0]]: continue
        s = pred[ix]
        if not np.isfinite(s).all(): continue
        n = len(ix); j = np.argsort(-(s + rng.random(n) * 1e-12), kind="mergesort")
        r = np.empty(n); r[j] = np.arange(1, n + 1)
        c = int(np.argmax(y[ix]))
        h.append(1.0 if j[0] == c else 0.0); rr.append(1.0 / r[c]); ns.append(n); sl.append((k, ix))
    return np.array(h), np.array(rr), np.array(ns, float), sl
def auprc(yv, s):
    o = np.argsort(-s, kind="mergesort"); yy = yv[o]; sv = s[o]
    tp = np.cumsum(yy); k = np.arange(1, len(yy) + 1)
    idx = np.where(np.r_[sv[1:] != sv[:-1], True])[0]
    prec = (tp[idx] / k[idx])[np.searchsorted(idx, np.arange(len(yy)))]
    return float((prec * yy).sum() / yy.sum())
KIN = list(range(len(KN))); TH = [len(KN) + i for i in range(3)]
GCC = [len(KN) + 3 + i for i in range(25)]
BLOCKS = {"kinetic geometry (6)": KIN[:6], "kinetic U1 (2)": KIN[6:], "kinetic all (8)": KIN,
          "three features (3)": TH, "GC-binned (25)": GCC, "union (36)": KIN + TH + GCC}
print(f"\n===== BLOCKS (holdout sets, chance P@1 0.1667, chance MRR 0.4083, base rate 0.1667) =====")
print(f"{'block':<22} {'sets':>6} {'P@1':>7} {'95% CI':>17} {'MRR':>7} {'AUROC':>7} {'AUPRC':>7} {'win-rate':>9}")
with open("ipa_models.tsv", "w") as fh:
    fh.write("block\tn_feat\tn_sets\tp1\tp1_lo\tp1_hi\tp1_chance\tmrr\tmrr_chance\tauroc\tauroc_lo\tauroc_hi\tauprc\tbase_rate\twinrate\n")
    for nm, cols in BLOCKS.items():
        pred = loco_hold(cols)
        h, rr, ns, sl = per_set(pred)
        rows = np.concatenate([ix for _, ix in sl])
        gh_ = defaultdict(list)
        for t, (k, ix) in enumerate(sl): gh_[sgene[k]].append(t)
        GG = [np.array(v) for v in gh_.values()]
        hb = np.empty(200); ab = np.empty(200)
        for t in range(200):
            sel = np.concatenate([GG[j] for j in rng.integers(0, len(GG), len(GG))])
            ii = np.concatenate([sl[j][1] for j in sel])
            hb[t] = h[sel].mean(); ab[t] = auroc(y[ii], pred[ii])
        ch = float(np.mean(1.0 / ns)); mc = float(np.mean([np.sum(1.0/np.arange(1,int(n)+1))/n for n in ns]))
        au = auroc(y[rows], pred[rows]); ap = auprc(y[rows], pred[rows])
        wr = float(np.nanmean(winrates(np.where(is_hold, pred, np.nan))))
        print(f"{nm:<22} {len(h):>6} {h.mean():>7.4f} [{np.percentile(hb,2.5):.4f},"
              f"{np.percentile(hb,97.5):.4f}] {rr.mean():>7.4f} {au:>7.4f} {ap:>7.4f} {wr:>9.4f}")
        fh.write(f"{nm}\t{len(cols)}\t{len(h)}\t{h.mean():.5f}\t{np.percentile(hb,2.5):.5f}\t"
                 f"{np.percentile(hb,97.5):.5f}\t{ch:.5f}\t{rr.mean():.5f}\t{mc:.5f}\t{au:.5f}\t"
                 f"{np.percentile(ab,2.5):.5f}\t{np.percentile(ab,97.5):.5f}\t{ap:.5f}\t{float(y[rows].mean()):.5f}\t{wr:.5f}\n")

# dose-response
def rankv(v):
    o = np.argsort(v, kind="mergesort"); sv = v[o]; n = len(v); rk = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    r = np.empty(n); r[o] = rk; return r
def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan
def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return pearson(rankv(a[m]), rankv(b[m])) if m.sum() > 10 else np.nan
pchrom = np.array([r["chrom"] for r in POS]); pgene = np.array([r["gene"] for r in POS])
print("\n===== DOSE-RESPONSE: usage of the used intronic PAS =====")
with open("ipa_dose.tsv", "w") as fh:
    fh.write("subset\tfeature\tspearman\tn\n")
    for lab, msk in [("all positives with usage", np.isfinite(usage)),
                     ("usage >= 1%", np.isfinite(usage) & (usage >= 1))]:
        u = usage[msk]; Xu = XP[msk]
        print(f"\n  {lab}: n {int(msk.sum())}, median usage {np.median(u):.2f}%")
        for j, nm in enumerate(NAMES[:len(KN) + 3]):
            r = spearman(Xu[:, j], u)
            print(f"    {nm:<22} Spearman {r:>+7.4f}")
            fh.write(f"{lab}\t{nm}\t{r:.5f}\t{int(msk.sum())}\n")
        for tgt, tname in [(np.log10(u + 0.5), "log10(usage+0.5)"), (rankv(u) / len(u), "rank")]:
            pred = np.full(len(u), np.nan)
            cc = pchrom[msk]
            for c in sorted(set(cc)):
                te = cc == c; tr = ~te
                if te.sum() == 0 or tr.sum() < 50: continue
                A, B = prep(Xu[tr][:, KIN], Xu[te][:, KIN])
                beta = np.linalg.lstsq(A.T @ A + np.eye(A.shape[1]) * RIDGE, A.T @ tgt[tr], rcond=None)[0]
                pred[te] = B @ beta
            m = np.isfinite(pred)
            r2 = 1.0 - ((tgt[m] - pred[m]) ** 2).sum() / ((tgt[m] - tgt[m].mean()) ** 2).sum()
            print(f"    kinetic block, held-out R2 on {tname:<18} {r2:>+7.4f}  "
                  f"(Spearman prediction vs truth {spearman(pred[m], u[m]):+.4f})")
            fh.write(f"{lab}\t#R2_{tname}\t{r2:.5f}\t{int(m.sum())}\n")
log("DONE_IPA_TEST")
