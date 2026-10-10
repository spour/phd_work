#!/usr/bin/env python3
# does reading the positional shape of the upstream splice profiles recover more
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

Z = np.load("binned_profiles.npz", allow_pickle=True)
P, PS = Z["P"], Z["PS"]
CU, CS = Z["comp_up_fixed"], Z["comp_up_fixed_sh"]
CH = [str(c) for c in Z["channels"]]; NBIN = int(Z["nbin"]); BINLO = Z["bin_lo"]
assert len(P) == len(y), f"profile rows {len(P)} != {len(y)}"
NCH = P.shape[1]
FLAT = [f"{c}@{int(b)}" for c in CH for b in BINLO]
log(f"profiles {P.shape} -> {NCH*NBIN} binned features, {len(CH)} channels x {NBIN} bins")

from fastparquet import ParquetFile
F = ParquetFile("shortlist_feat_all.parquet").to_pandas()
SPL = [c for c in F.columns if c.startswith("spl_")]
key = {(g, c, int(ct), s): i for i, (g, c, ct, s) in
       enumerate(zip(F.gene, F.chrom, F.center, F.strand))}
row = np.array([key.get((g, c, int(ct), s), -1) for g, c, ct, s in
                zip(C0["gene"][ok], C0["chrom"][ok], C0["center"][ok], C0["strand"][ok])])
hit = row >= 0
X26 = np.full((len(y), len(SPL)), np.nan); X26[hit] = F.loc[row[hit], SPL].to_numpy(float)
log(f"joined {int(hit.sum())}/{len(y)} peaks to shortlist_feat_all for the scalar baselines")
SEL3 = [SPL.index(c) for c in ("spl_pas_prox_u1", "spl_upstream_max_me3", "spl_comp_up_3ss")]
BUGGED = X26[:, SPL.index("spl_comp_up_3ss")]
b = BUGGED[np.isfinite(BUGGED)]; f = CU[np.isfinite(CU)]
log(f"[comp_up_3ss] shipped (PPT fraction) range {b.min():.3f}-{b.max():.3f}; "
    f"fixed (MaxEnt bits) range {f.min():.2f}-{f.max():.2f}; "
    f"corr {np.corrcoef(BUGGED[np.isfinite(BUGGED)&np.isfinite(CU)], CU[np.isfinite(BUGGED)&np.isfinite(CU)])[0,1]:+.3f}")

# machinery, identical to splice_feature_model.py
def centre_by_gene(M, g):
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

def zmat(M, fit_idx):
    Cg = centre_by_gene(M, gene)
    mu = np.nanmean(Cg[fit_idx], 0); mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.nanstd(Cg[fit_idx], 0); sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return (np.where(np.isfinite(Cg), Cg, mu) - mu) / sd

def fit_score(M, fit_idx, eval_idx, lam=1.0):
    Zm = zmat(M, fit_idx)
    b = fit_ridge(np.column_stack([np.ones(len(fit_idx)), Zm[fit_idx]]), y[fit_idx], lam)
    return np.column_stack([np.ones(len(eval_idx)), Zm[eval_idx]]) @ b, b

rs = np.random.default_rng(SEED)
tg = np.array(sorted(set(gene[tr]))); rs.shuffle(tg)
inner_val_genes = set(tg[:int(round(INNER_VAL_FRAC * len(tg)))])
iv = tr[np.array([g in inner_val_genes for g in gene[tr]])]
it = tr[np.array([g not in inner_val_genes for g in gene[tr]])]
GRP_IV = make_groups(iv)
log(f"inner split (TRAIN genes only): fit {len(it)} peaks / {len(set(gene[it]))} genes, "
    f"score {len(iv)} peaks / {len(GRP_IV)} genes")

def pick_lambda(M, tag):
    best = (-1.0, None)
    for lam in LAMBDAS:
        sc, _ = fit_score(M, it, iv, lam)
        w = winrate(sc, GRP_IV)
        if w > best[0]: best = (w, lam)
    log(f"[lambda] {tag}: inner-val win-rate {best[0]:.4f} at lambda {best[1]:g}")
    return best[1]

BS = np.random.default_rng(SEED).integers(0, len(GRP_TE), size=(NBOOT, len(GRP_TE)))

def suite(sc):
    wg = np.array([((sc[r][:, None] > sc[c][None, :]).sum()
                    + 0.5 * (sc[r][:, None] == sc[c][None, :]).sum()) / (len(r) * len(c))
                   for r, c in GRP_TE])
    p1 = np.array([float(sc[r].max() > sc[c].max()) + 0.5 * float(sc[r].max() == sc[c].max())
                   for r, c in GRP_TE])
    pos = sc[y[te] == 1]; neg = sc[y[te] == 0]
    out = dict(winrate=wg.mean(), p_at_1=p1.mean(), auroc=auroc(pos, neg), auprc=auprc(pos, neg))
    Bs = BS
    out["_wg"] = wg; out["_p1"] = p1
    ba = np.empty(NBOOT); bq = np.empty(NBOOT)
    for t in range(NBOOT):
        pi = np.concatenate([GRP_TE[j][0] for j in Bs[t]])
        ni = np.concatenate([GRP_TE[j][1] for j in Bs[t]])
        ba[t] = auroc(sc[pi], sc[ni]); bq[t] = auprc(sc[pi], sc[ni])
    for nm, v in (("winrate", wg[Bs].mean(1)),
                  ("p_at_1", p1[Bs].mean(1)), ("auroc", ba), ("auprc", bq)):
        out[nm + "_lo"], out[nm + "_hi"] = np.percentile(v, [2.5, 97.5])
    return out

RES = {}
def run(M, name, lam=None, tune=False):
    if tune: lam = pick_lambda(M, name)
    if lam is None: lam = 1.0
    sc, b = fit_score(M, trv, te, lam)
    r = suite(sc); r["_beta"] = b; r["_lam"] = lam; r["_nfeat"] = M.shape[1]
    RES[name] = r
    log(f"[{name:34s}] within-gene {r['winrate']:.4f} [{r['winrate_lo']:.4f},{r['winrate_hi']:.4f}]  "
        f"P@1 {r['p_at_1']:.4f} [{r['p_at_1_lo']:.4f},{r['p_at_1_hi']:.4f}]  "
        f"AUROC {r['auroc']:.4f}  AUPRC {r['auprc']:.4f}  "
        f"(p={M.shape[1]}, lambda={lam:g})")
    return r

FL  = P.reshape(len(P), -1)
FLS = PS.reshape(len(PS), -1)
IME3 = CH.index("me3_max"); UPB = 19
ACC_P  = np.nanmax(P[:,  IME3, :UPB], axis=1)
ACC_PS = np.nanmax(PS[:, IME3, :UPB], axis=1)
m = np.isfinite(ACC) & np.isfinite(ACC_P)
log(f"[derived acceptor] max me3_max over bins 0-{UPB-1} = [-2000,-100): "
    f"corr with pipeline spl_upstream_max_me3 {np.corrcoef(ACC[m], ACC_P[m])[0,1]:+.4f}, "
    f"mean {np.nanmean(ACC_P):.3f} real vs {np.nanmean(ACC_PS):.3f} shuffled")
log(f"\ntrain+val {len(trv)} peaks, test {len(te)} peaks in {len(GRP_TE)} genes, "
    f"base rate {y[te].mean():.4f}\n")

run(X26[:, SEL3],                          "1 scalar baseline (as-is)")
run(np.column_stack([X26[:, SEL3[0]], X26[:, SEL3[1]], CU]),
                                           "2a scalar, comp_up_3ss FIXED")
run(X26[:, SEL3[:2]],                      "2b scalar, comp_up_3ss DROPPED")
run(FL,                                    "3 binned profiles", tune=True)
run(np.column_stack([FL, ACC]),            "4 binned + acceptor scalar", tune=True)
run(np.column_stack([FL, ACC, CU]),        "4b binned + acceptor + comp_up fixed", tune=True)
run(ACC[:, None],                          "ref acceptor alone")
run(FLS, "5 binned profiles, dinuc-shuffled", lam=RES["3 binned profiles"]["_lam"])
run(np.column_stack([FL,  ACC_P]),  "4c binned + derived acceptor",
    lam=RES["4 binned + acceptor scalar"]["_lam"])
run(np.column_stack([FLS, ACC_PS]), "5c binned + acceptor, ALL shuffled",
    lam=RES["4 binned + acceptor scalar"]["_lam"])

def paired(a, b, tag):
    da = RES[a]["_wg"][BS].mean(1) - RES[b]["_wg"][BS].mean(1)
    d = RES[a]["winrate"] - RES[b]["winrate"]
    lo, hi = np.percentile(da, [2.5, 97.5])
    p = 2 * min((da <= 0).mean(), (da >= 0).mean())
    log(f"[paired] {tag:46s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  p = {max(p,1/NBOOT):.4f}"
        f"{'' if lo*hi > 0 else '   (CI spans 0)'}")
    return d, lo, hi, p

log("")
PAIRS = [
  ("4 binned + acceptor scalar", "1 scalar baseline (as-is)",    "shape vs scalar baseline"),
  ("4 binned + acceptor scalar", "2b scalar, comp_up_3ss DROPPED","shape vs scalar, bug removed"),
  ("4 binned + acceptor scalar", "ref acceptor alone",            "shape+acceptor vs acceptor alone"),
  ("3 binned profiles",          "ref acceptor alone",            "shape alone vs acceptor alone"),
  ("3 binned profiles",          "5 binned profiles, dinuc-shuffled", "shape: real minus dinuc-shuffled"),
  ("4c binned + derived acceptor","5c binned + acceptor, ALL shuffled","shape+acceptor: real minus ALL-shuffled"),
  ("2a scalar, comp_up_3ss FIXED","1 scalar baseline (as-is)",    "comp_up_3ss fixed minus as-is"),
  ("2b scalar, comp_up_3ss DROPPED","1 scalar baseline (as-is)",  "comp_up_3ss dropped minus as-is"),
]
PAIRRES = {t: paired(a, b, t) for a, b, t in PAIRS}

REF = {"acceptor (this split)": 0.61983, "scalar splice model": 0.66557,
       "splice ceiling, 26 scalars": 0.69686, "sequence CNN": 0.82110}
log("")
for k, v in REF.items(): log(f"[reference] {k:28s} within-gene {v:.4f}")
d_bug = RES["2a scalar, comp_up_3ss FIXED"]["winrate"] - RES["1 scalar baseline (as-is)"]["winrate"]
d_drop = RES["2b scalar, comp_up_3ss DROPPED"]["winrate"] - RES["1 scalar baseline (as-is)"]["winrate"]
d_shape = RES["4 binned + acceptor scalar"]["winrate"] - RES["1 scalar baseline (as-is)"]["winrate"]
d_shuf = (RES["4c binned + derived acceptor"]["winrate"]
          - RES["5c binned + acceptor, ALL shuffled"]["winrate"])
best = max(RES[k]["winrate"] for k in RES if k.startswith(("3", "4")))
log(f"\n[comp_up_3ss] fixing it moves the scalar model {d_bug:+.4f}; dropping it {d_drop:+.4f}")
log(f"[shape vs scalar] binned+acceptor minus scalar baseline {d_shape:+.4f}")
def pct_lost(a, b):
    return 100.0 * (RES[a]["winrate"] - RES[b]["winrate"]) / max(RES[a]["winrate"] - 0.5, 1e-9)
log(f"[dinuc shuffle] profiles alone lose {RES['3 binned profiles']['winrate']-RES['5 binned profiles, dinuc-shuffled']['winrate']:+.4f} "
    f"({pct_lost('3 binned profiles','5 binned profiles, dinuc-shuffled'):.1f}% of lift over 0.5)")
log(f"[dinuc shuffle] profiles + derived acceptor, both shuffled, lose {d_shuf:+.4f} "
    f"({pct_lost('4c binned + derived acceptor','5c binned + acceptor, ALL shuffled'):.1f}% of lift)")
log(f"[dinuc shuffle] reference, shuffle_benchmark.tsv k=3: CNN loses 68.6% of its lift, "
    f"the 3-feature scalar splice model 91.5%; this arm is k=1")
log(f"[ceiling] best binned model {best:.4f}: "
    f"{'CLEARS' if best>=0.78 else 'does not clear'} 0.78, "
    f"{'clears' if best>=0.72 else 'does not clear'} 0.72; CNN {REF['sequence CNN']:.4f}")

with open("binned_splice_metrics.tsv", "w") as fh:
    fh.write("model\tmetric\tvalue\tci_lo\tci_hi\tn_features\tlambda\tn_genes\tn_peaks\n")
    for nm, r in RES.items():
        for m in ("winrate", "p_at_1", "auroc", "auprc"):
            fh.write(f"{nm}\t{m}\t{r[m]:.5f}\t{r[m+'_lo']:.5f}\t{r[m+'_hi']:.5f}\t"
                     f"{r['_nfeat']}\t{r['_lam']:g}\t{len(GRP_TE)}\t{len(te)}\n")
    for k, v in REF.items():
        fh.write(f"reference: {k}\twinrate\t{v:.5f}\t\t\t\t\t{len(GRP_TE)}\t{len(te)}\n")
    for t, (d, lo, hi, p) in PAIRRES.items():
        fh.write(f"paired: {t}\tdelta_winrate\t{d:.5f}\t{lo:.5f}\t{hi:.5f}\t\t"
                 f"{max(p,1/NBOOT):.4f}\t{len(GRP_TE)}\t{len(te)}\n")

bb = RES["4 binned + acceptor scalar"]["_beta"]
with open("binned_coefficients.tsv", "w") as fh:
    fh.write("channel\tbin_lo\tbin_centre\tcoef\n")
    for j, nm in enumerate(FLAT):
        c, b0 = nm.split("@")
        fh.write(f"{c}\t{b0}\t{int(b0)+50}\t{bb[1+j]:.6f}\n")
    fh.write(f"spl_upstream_max_me3\t\t\t{bb[1+len(FLAT)]:.6f}\n")
log("[out] binned_splice_metrics.tsv binned_coefficients.tsv")
