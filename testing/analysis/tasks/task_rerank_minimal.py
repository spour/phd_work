#!/usr/bin/env python3
# minimal splicing reranker: selection, redundancy check, and every panel's data
import sys, csv, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)

SEED, HOLD_FRAC, RIDGE, NBOOT, MAXK, NPERM = 42, 0.20, 1.0, 400, 12, 200
rng = np.random.default_rng(SEED)

D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
rf = D.rf_prob.to_numpy(float)

S = np.load("splice_feats_cache.npz")
kmaster = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([kmaster[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
def sc(n): return S[n][ridx]
DROP = {"spl_dist_dn_5ss", "spl_dist_up_5ss"}
FEAT = {c: D[c].to_numpy(float) for c in D.columns if c.startswith("spl_") and c not in DROP}
FEAT["scan_n3ss_up5k"] = sc("C_n3ss_up5k"); FEAT["scan_n3ss_dn5k"] = sc("C_n3ss_dn5k")
FEAT["scan_n3ss_diff"] = FEAT["scan_n3ss_up5k"] - FEAT["scan_n3ss_dn5k"]
FEAT["scan_n5ss_up5k"] = sc("C_n5ss_up5k"); FEAT["scan_n5ss_dn5k"] = sc("C_n5ss_dn5k")
FEAT["scan_acc_rank2k"] = sc("B_acc_rank2k"); FEAT["scan_acc_margin2k"] = sc("B_acc_margin2k")
FEAT["scan_in_intron"] = sc("C_in_intron")
FEAT["acc_minus_max5"] = FEAT["spl_upstream_max_me3"] - FEAT["spl_interv_me5_max"]
NAMES = sorted(FEAT); X = np.column_stack([FEAT[n] for n in NAMES])
log(f"rows {X.shape[0]}  candidates {X.shape[1]} (spl_dist_up_5ss dropped)")

allg = np.array(sorted(set(gene)))
hold_g = set(rng.choice(allg, size=int(round(HOLD_FRAC * len(allg))), replace=False))
is_hold = np.array([g in hold_g for g in gene])
gi = defaultdict(list)
for i in range(len(gene)): gi[gene[i]].append(i)
gi = {g: np.array(v) for g, v in gi.items()}
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
    A = (np.where(np.isfinite(Xtr), Xtr, mu) - mu) / sd
    B = (np.where(np.isfinite(Xte), Xte, mu) - mu) / sd
    return np.column_stack([np.ones(len(A)), A]), np.column_stack([np.ones(len(B)), B])
def loco_pred(cols, train_mask, test_mask, yv=None, Xm=None):
    yy = y if yv is None else yv; XX = X if Xm is None else Xm
    pred = np.full(len(yy), np.nan); Xs = XX[:, cols]
    for c in CHROMS:
        te = test_mask & (chrom == c); tr = train_mask & (chrom != c)
        if te.sum() == 0 or tr.sum() == 0: continue
        A, B = prep(Xs[tr], Xs[te]); pred[te] = B @ fit_ridge(A, yy[tr])
    return pred
def per_gene(pred, mask, yv=None):
    yy = y if yv is None else yv
    hits, rrs, ns, gl = [], [], [], []
    for g, ix in gi.items():
        if not mask[ix[0]]: continue
        s = pred[ix]
        if not np.isfinite(s).all() or len(ix) < 2: continue
        n = len(ix); j = np.argsort(-(s + rng.random(n) * 1e-12), kind="mergesort")
        srt = s[j]; rk = np.empty(n); i = 0
        while i < n:
            k = i
            while k + 1 < n and srt[k + 1] == srt[i]: k += 1
            rk[i:k + 1] = (i + k) / 2.0 + 1.0; i = k + 1
        r = np.empty(n); r[j] = rk
        c = int(np.argmax(yy[ix]))
        hits.append(1.0 if j[0] == c else 0.0); rrs.append(1.0 / r[c]); ns.append(n); gl.append(g)
    return np.array(hits), np.array(rrs), np.array(ns, float), gl
def auroc(yv, s):
    o = np.argsort(s, kind="mergesort"); sv = s[o]; n = len(s); rk = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    rr = np.empty(n); rr[o] = rk; npos = yv.sum(); nneg = n - npos
    return (rr[yv == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
def auprc(yv, s):
    o = np.argsort(-s, kind="mergesort"); yy = yv[o]; sv = s[o]
    tp = np.cumsum(yy); k = np.arange(1, len(yy) + 1)
    idx = np.where(np.r_[sv[1:] != sv[:-1], True])[0]
    prec = (tp[idx] / k[idx])[np.searchsorted(idx, np.arange(len(yy)))]
    return float((prec * yy).sum() / yy.sum())
def bci(v, nb=NBOOT):
    b = v[rng.integers(0, len(v), size=(nb, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
def metrics(pred, mask):
    hits, rrs, ns, gl = per_gene(pred, mask)
    rows = np.concatenate([gi[g] for g in gl]); yv = y[rows]; sv = pred[rows]
    Hn = np.array([np.sum(1.0 / np.arange(1, int(n) + 1)) for n in ns])
    p1, p1lo, p1hi = bci(hits); mr, mrlo, mrhi = bci(rrs)
    e1, e1lo, e1hi = bci(hits - 1.0 / ns); em, emlo, emhi = bci(rrs - Hn / ns)
    gidx = [gi[g] for g in gl]; ab = np.empty(200); pb = np.empty(200)
    for t in range(200):
        ii = np.concatenate([gidx[j] for j in rng.integers(0, len(gidx), len(gidx))])
        ab[t] = auroc(y[ii], pred[ii]); pb[t] = auprc(y[ii], pred[ii])
    return dict(n_genes=len(gl), n_rows=len(rows), p1=p1, p1lo=p1lo, p1hi=p1hi,
                p1_chance=float(np.mean(1.0 / ns)), p1_exc=e1, p1_exc_lo=e1lo, p1_exc_hi=e1hi,
                mrr=mr, mrr_lo=mrlo, mrr_hi=mrhi, mrr_chance=float(np.mean(Hn / ns)),
                mrr_exc=em, mrr_exc_lo=emlo, mrr_exc_hi=emhi,
                auroc=auroc(yv, sv), auroc_lo=float(np.percentile(ab, 2.5)), auroc_hi=float(np.percentile(ab, 97.5)),
                auprc=auprc(yv, sv), auprc_lo=float(np.percentile(pb, 2.5)), auprc_hi=float(np.percentile(pb, 97.5)),
                base_rate=float(yv.mean()))

# forward selection, dev loco
print("\n===== FORWARD SELECTION (pool %d; dev = 80%% of genes, LOCO) =====" % len(NAMES))
print(f"{'k':>2} {'added':<24} {'dev P@1':>8} {'dev 95% CI':>18} {'hold P@1':>9} {'hold 95% CI':>18} {'gain':>8}")
chosen, path, prev = [], [], 0.0
remaining = list(range(len(NAMES)))
CACHE = "rerank_min_path_cache.tsv"
import os
if os.path.exists(CACHE):
    for L in open(CACHE):
        p = L.rstrip("\n").split("\t")
        if p[0] == "k": continue
        path.append(dict(k=int(p[0]), added=p[1], dev=float(p[2]), dlo=float(p[3]), dhi=float(p[4]),
                         hold=float(p[5]), hlo=float(p[6]), hhi=float(p[7]), gain=float(p[8]), hchance=float(p[9])))
        j = NAMES.index(p[1]); chosen.append(j); remaining.remove(j)
        print(f"{p[0]:>2} {p[1]:<24} {float(p[2]):>8.4f} [{float(p[3]):.4f},{float(p[4]):.4f}] "
              f"{float(p[5]):>9.4f} [{float(p[6]):.4f},{float(p[7]):.4f}] {float(p[8]):>+8.4f}   (cached)")
for k in range(len(path) + 1, MAXK + 1):
    best = None
    for j in remaining:
        h, _, _, _ = per_gene(loco_pred(chosen + [j], ~is_hold, ~is_hold), ~is_hold)
        if best is None or h.mean() > best[0]: best = (h.mean(), j, h)
    v, j, h = best; chosen.append(j); remaining.remove(j)
    dm, dlo, dhi = bci(h)
    hh, _, hns, _ = per_gene(loco_pred(chosen, ~is_hold, is_hold), is_hold)
    hm, hlo, hhi = bci(hh)
    path.append(dict(k=k, added=NAMES[j], dev=dm, dlo=dlo, dhi=dhi, hold=hm, hlo=hlo, hhi=hhi,
                     gain=dm - prev, hchance=float(np.mean(1.0 / hns))))
    print(f"{k:>2} {NAMES[j]:<24} {dm:>8.4f} [{dlo:.4f},{dhi:.4f}] {hm:>9.4f} [{hlo:.4f},{hhi:.4f}] {dm-prev:>+8.4f}")
    prev = dm; log(f"  step {k}: {NAMES[j]}")

with open(CACHE, "w") as fh:
    fh.write("k\tadded\tdev_p1\tdev_lo\tdev_hi\thold_p1\thold_lo\thold_hi\tgain\thold_chance\n")
    for p in path:
        fh.write(f"{p['k']}\t{p['added']}\t{p['dev']:.5f}\t{p['dlo']:.5f}\t{p['dhi']:.5f}\t"
                 f"{p['hold']:.5f}\t{p['hlo']:.5f}\t{p['hhi']:.5f}\t{p['gain']:+.5f}\t{p['hchance']:.5f}\n")

kp = max(path, key=lambda p: p["hold"])
cover = [p["k"] for p in path if p["hhi"] >= kp["hold"]]
kstar = min(cover)
kA = kp["k"]
print(f"\nholdout plateau {kp['hold']:.4f} at k={kA} (CI [{kp['hlo']:.4f},{kp['hhi']:.4f}])")
print(f"k whose holdout CI covers the plateau value: {cover}  -> rule gives k = {kstar}")
print("every holdout CI in the path overlaps every other; the plateau range is the honest statement, "
      "not any single k")
CHOSEN = chosen[:kA]
print(f"plateau-attaining set (k={kA}): " + ", ".join(NAMES[j] for j in CHOSEN))
print(f"rule-minimal set (k={kstar}): " + ", ".join(NAMES[j] for j in chosen[:kstar]))

# redundancy among the donor / u1 density terms
DENSALL = ["spl_u1_suppression_up", "spl_n_5ss_1k", "spl_pas_prox_u1", "spl_interv_me5_max"]
print("\n===== REDUNDANCY: donor / U1 density terms (pairwise Pearson r, all rows) =====")
RED = []
for a in range(len(DENSALL)):
    for b in range(a + 1, len(DENSALL)):
        u = FEAT[DENSALL[a]]; v = FEAT[DENSALL[b]]; m = np.isfinite(u) & np.isfinite(v)
        r = float(np.corrcoef(u[m], v[m])[0, 1]); RED.append((DENSALL[a], DENSALL[b], r))
        print(f"  r({DENSALL[a]:<22}, {DENSALL[b]:<22}) = {r:+.3f}")

def refit_beta(cols):
    dv = np.where(~is_hold)[0]; A, _ = prep(X[dv][:, cols], X[dv][:, cols])
    return fit_ridge(A, y[dv])[1:]
beta0 = refit_beta(CHOSEN)
inset = [(NAMES[j], a) for a, j in enumerate(CHOSEN) if NAMES[j] in DENSALL]
print(f"\ndensity terms inside the k={kA} set: " +
      (", ".join(f"{n} (beta {beta0[a]:+.3f})" for n, a in inset) if inset else "none"))
R_full = metrics(loco_pred(CHOSEN, ~is_hold, is_hold), is_hold)
print(f"  holdout P@1, all {len(CHOSEN)} features: {R_full['p1']:.4f} [{R_full['p1lo']:.4f},{R_full['p1hi']:.4f}]")
DROPTESTS = []
order = sorted(inset, key=lambda t: abs(beta0[t[1]]))
for lab, names in ([(f"drop {order[0][0]}", {order[0][0]}) ] if len(order) >= 1 else []) + \
                  ([(f"drop {order[1][0]}", {order[1][0]}) ] if len(order) >= 2 else []) + \
                  ([("drop both weakest", {order[0][0], order[1][0]})] if len(order) >= 2 else []):
    cols = [j for j in CHOSEN if NAMES[j] not in names]
    r = metrics(loco_pred(cols, ~is_hold, is_hold), is_hold)
    d = r["p1"] - R_full["p1"]
    DROPTESTS.append((lab, len(cols), r, d))
    print(f"  {lab:<34} k={len(cols)}  holdout P@1 {r['p1']:.4f} [{r['p1lo']:.4f},{r['p1hi']:.4f}]  delta {d:+.4f}")
PRUNE = bool(DROPTESTS) and (-DROPTESTS[-1][3] <= 0.005) and DROPTESTS[-1][2]["p1hi"] >= R_full["p1lo"]
FINAL = [j for j in CHOSEN if NAMES[j] not in {order[0][0], order[1][0]}] if PRUNE else CHOSEN
print(f"  decision: {'DROP the weakest two (costs <= 0.005)' if PRUNE else 'KEEP them (dropping costs more than 0.005)'}")
print("FINAL SET (%d): " % len(FINAL) + ", ".join(NAMES[j] for j in FINAL))
with open("rerank_min_redundancy.tsv", "w") as fh:
    fh.write("feat_a\tfeat_b\tpearson_r\n")
    for a, b, r in RED: fh.write(f"{a}\t{b}\t{r:.4f}\n")
    fh.write("#\n#test\tn_feat\thold_p1\tci_lo\tci_hi\tdelta\n")
    fh.write(f"#keep all\t{len(CHOSEN)}\t{R_full['p1']:.5f}\t{R_full['p1lo']:.5f}\t{R_full['p1hi']:.5f}\t0\n")
    for lab, nc, r, d in DROPTESTS:
        fh.write(f"#{lab}\t{nc}\t{r['p1']:.5f}\t{r['p1lo']:.5f}\t{r['p1hi']:.5f}\t{d:+.5f}\n")

with open("rerank_min_path.tsv", "w") as fh:
    fh.write("k\tadded\tdev_p1\tdev_lo\tdev_hi\thold_p1\thold_lo\thold_hi\tgain\thold_chance\tchosen\trule_min\tin_final\n")
    for p in path:
        fh.write(f"{p['k']}\t{p['added']}\t{p['dev']:.5f}\t{p['dlo']:.5f}\t{p['dhi']:.5f}\t"
                 f"{p['hold']:.5f}\t{p['hlo']:.5f}\t{p['hhi']:.5f}\t{p['gain']:+.5f}\t{p['hchance']:.5f}\t"
                 f"{1 if p['k']==len(FINAL) else 0}\t{1 if p['k']==kstar else 0}\t"
                 f"{1 if NAMES.index(p['added']) in FINAL else 0}\n")

# headline metrics
print("\n===== FINAL MODEL =====")
ALL = np.ones(len(y), bool)
RES = {"final_hold": metrics(loco_pred(FINAL, ~is_hold, is_hold), is_hold),
       "final_dev": metrics(loco_pred(FINAL, ~is_hold, ~is_hold), ~is_hold),
       "final_loco_all": metrics(loco_pred(FINAL, ALL, ALL), ALL)}
pred_all = loco_pred(FINAL, ALL, ALL)
RF = metrics(rf, ALL)
RES["rf_baseline_all"] = RF
for nm, r in RES.items():
    print(f"{nm:<16} genes {r['n_genes']:>5} P@1 {r['p1']:.4f}[{r['p1lo']:.4f},{r['p1hi']:.4f}] chance {r['p1_chance']:.4f} "
          f"exc {r['p1_exc']:+.4f}[{r['p1_exc_lo']:+.4f},{r['p1_exc_hi']:+.4f}] MRR {r['mrr']:.4f} chance {r['mrr_chance']:.4f} "
          f"AUROC {r['auroc']:.4f}[{r['auroc_lo']:.4f},{r['auroc_hi']:.4f}] AUPRC {r['auprc']:.4f} base {r['base_rate']:.4f}")
ks = ["n_genes","n_rows","p1","p1lo","p1hi","p1_chance","p1_exc","p1_exc_lo","p1_exc_hi","mrr","mrr_lo","mrr_hi",
      "mrr_chance","mrr_exc","mrr_exc_lo","mrr_exc_hi","auroc","auroc_lo","auroc_hi","auprc","auprc_lo","auprc_hi","base_rate"]
with open("rerank_min_metrics.tsv", "w") as fh:
    fh.write("model\tn_feat\t" + "\t".join(ks) + "\n")
    for nm, r in RES.items():
        fh.write(f"{nm}\t{len(FINAL)}\t" + "\t".join(f"{r[k]:.5f}" if isinstance(r[k], float) else str(r[k]) for k in ks) + "\n")

# coefficients
dev = np.where(~is_hold)[0]; Xs = X[:, FINAL]
A, _ = prep(Xs[dev], Xs[dev]); beta = fit_ridge(A, y[dev])[1:]
devg = [g for g in gi if not is_hold[gi[g][0]]]; gidx = [gi[g] for g in devg]
BB = np.empty((300, len(FINAL)))
for t in range(300):
    ii = np.concatenate([gidx[j] for j in rng.integers(0, len(gidx), len(gidx))])
    Ab, _ = prep(Xs[ii], Xs[ii]); BB[t] = fit_ridge(Ab, y[ii])[1:]
print("\n===== STANDARDIZED COEFFICIENTS (dev fit, bootstrap over genes) =====")
with open("rerank_min_coefs.tsv", "w") as fh:
    fh.write("feature\tbeta_std\tci_lo\tci_hi\tsign_flip_frac\n")
    for a, j in enumerate(FINAL):
        lo, hi = np.percentile(BB[:, a], [2.5, 97.5]); fl = float(np.mean(np.sign(BB[:, a]) != np.sign(beta[a])))
        print(f"  {NAMES[j]:<24} {beta[a]:>+8.4f} [{lo:>+8.4f},{hi:>+8.4f}] flip {fl:.3f}")
        fh.write(f"{NAMES[j]}\t{beta[a]:.5f}\t{lo:.5f}\t{hi:.5f}\t{fl:.4f}\n")

def curves(sv, tag, fh):
    o = np.argsort(-sv, kind="mergesort"); yy = y[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy); P = yy.sum(); N = len(yy) - P
    idx = np.unique(np.linspace(0, len(yy) - 1, 400).astype(int))
    for i in idx:
        fh.write(f"{tag}\t{fp[i]/N:.6f}\t{tp[i]/P:.6f}\t{tp[i]/(i+1):.6f}\n")
with open("rerank_min_curves.tsv", "w") as fh:
    fh.write("model\tfpr\ttpr\tprecision\n")
    curves(pred_all, "reranker", fh); curves(rf, "RF baseline", fh)

hits, rrs, ns, gl = per_gene(pred_all, ALL)
gch = np.array([chrom[gi[g][0]] for g in gl])
with open("rerank_min_bychrom.tsv", "w") as fh:
    fh.write("chrom\tn_genes\tp1\tci_lo\tci_hi\tchance\tlift\n")
    for c in CHROMS:
        m = gch == c
        if m.sum() < 20: continue
        v, lo, hi = bci(hits[m]); ch = float(np.mean(1.0 / ns[m]))
        fh.write(f"{c}\t{int(m.sum())}\t{v:.5f}\t{lo:.5f}\t{hi:.5f}\t{ch:.5f}\t{v-ch:.5f}\n")
ncomp = ns - 1
BUCK = [("1", ncomp == 1), ("2", ncomp == 2), ("3-4", (ncomp >= 3) & (ncomp <= 4)),
        ("5-8", (ncomp >= 5) & (ncomp <= 8)), ("9+", ncomp >= 9)]
with open("rerank_min_bycontest.tsv", "w") as fh:
    fh.write("bucket\tn_genes\tp1\tci_lo\tci_hi\tchance\tlift\n")
    print("\n===== CORRECT RATE BY CONTEST DIFFICULTY (LOCO, all contested genes) =====")
    for nm, m in BUCK:
        v, lo, hi = bci(hits[m]); ch = float(np.mean(1.0 / ns[m]))
        print(f"  {nm:>4} competitors  n {int(m.sum()):>5}  P@1 {v:.4f} [{lo:.4f},{hi:.4f}]  chance {ch:.4f}  lift {v-ch:+.4f}")
        fh.write(f"{nm}\t{int(m.sum())}\t{v:.5f}\t{lo:.5f}\t{hi:.5f}\t{ch:.5f}\t{v-ch:.5f}\n")

log("permutation null: %d LOCO refits" % NPERM)
obs = hits.mean(); nul = np.empty(NPERM)
for t in range(NPERM):
    yp = y.copy()
    for g, ix in gi.items():
        c = ix[np.argmax(y[ix])]; yp[ix] = 0.0; yp[ix[rng.integers(0, len(ix))]] = 1.0
    pp = loco_pred(FINAL, ALL, ALL, yv=yp)
    h, _, _, _ = per_gene(pp, ALL, yv=yp); nul[t] = h.mean()
pv = (np.sum(nul >= obs) + 1) / (NPERM + 1)
with open("rerank_min_permnull.tsv", "w") as fh:
    fh.write("perm\tp1\n")
    for t in range(NPERM): fh.write(f"{t+1}\t{nul[t]:.5f}\n")
    fh.write(f"#observed\t{obs:.5f}\n#p\t{pv:.5f}\n#chance\t{float(np.mean(1.0/ns)):.5f}\n")
print(f"\npermutation null (labels shuffled within gene, LOCO refit x{NPERM}): "
      f"mean {nul.mean():.4f} [{np.percentile(nul,2.5):.4f},{np.percentile(nul,97.5):.4f}] vs observed {obs:.4f}, p = {pv:.4f}")

ov = {"model correct, RF wrong": 0, "RF correct, model wrong": 0, "both correct": 0, "both wrong": 0}
marg = []
for k, g in enumerate(gl):
    ix = gi[g]; s = pred_all[ix]; b = rf[ix]; c = int(np.argmax(y[ix]))
    mp = int(np.argmax(s + rng.random(len(ix)) * 1e-12)); bp = int(np.argmax(b + rng.random(len(ix)) * 1e-12))
    mh, bh = mp == c, bp == c
    ov["both correct" if (mh and bh) else "model correct, RF wrong" if mh else
       "RF correct, model wrong" if bh else "both wrong"] += 1
    sd = np.std(s) if np.std(s) > 0 else 1.0
    marg.append((g, 1 if mh else 0, len(ix), float((s[mp] - s[c]) / sd)))
with open("rerank_min_override.tsv", "w") as fh:
    fh.write("outcome\tn_genes\tfrac\n")
    tot = sum(ov.values())
    print("\n===== OVERRIDE OUTCOME vs the RF pick =====")
    for kk, vv in ov.items():
        print(f"  {kk:<26} {vv:>5}  {vv/tot:.4f}")
        fh.write(f"{kk}\t{vv}\t{vv/tot:.5f}\n")
with open("rerank_min_margin.tsv", "w") as fh:
    fh.write("gene\thit\tn_candidates\tmargin_sd\n")
    for g, h, n, m in marg: fh.write(f"{g}\t{h}\t{n}\t{m:.5f}\n")
mm = np.array([m for _, h, _, m in marg if h == 0])
print(f"misses: median margin {np.median(mm):.3f} SD, {np.mean(mm<0.5)*100:.1f}% within 0.5 SD of the top pick")
log("DONE_RERANK_MIN")
