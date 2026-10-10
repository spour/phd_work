#!/usr/bin/env python3
# lock the channel-selected binned splice model as the chapter's splice comparator, save it
import sys
import numpy as np
import binned_common as K
from binned_common import log

Z = np.load("binned_profiles.npz", allow_pickle=True)
P = Z["P"]; CH = [str(c) for c in Z["channels"]]
NBIN = int(Z["nbin"]); BINLO = Z["bin_lo"].astype(int)
UPW, DNW = int(Z["up"]), int(Z["dn"])
ACC = K.ACC
assert len(P) == len(K.y)

# step 0: window
assert (UPW, DNW, NBIN) == (2000, 500, 25), f"unexpected window {(UPW, DNW, NBIN)}"
assert BINLO[0] == -2000 and BINLO[-1] == 400 and np.all(np.diff(BINLO) == 100)
log(f"[step0] window [-{UPW}, +{DNW}] nt, {NBIN} bins of 100 nt, bin_lo {BINLO[0]}..{BINLO[-1]}")
log("[step0] binned_profiles.npz is built from cnn_data.npz['X'], the token array of that same")
log("[step0] window, so the model is ALREADY on [-2000, +500]. No refit for the window.")

FEATCH = ("u1_dens4", "u1_dens7")
IDX = [CH.index(c) for c in FEATCH]
FINAL = np.column_stack([P[:, i, :] for i in IDX] + [ACC])
NAMES = [f"{c}[{BINLO[b]},{BINLO[b]+100})" for c in FEATCH for b in range(NBIN)] + \
        ["spl_upstream_max_me3"]
assert FINAL.shape[1] == len(NAMES) == 51
log(f"[model] {len(FEATCH)} channels x {NBIN} bins + 1 acceptor scalar = {FINAL.shape[1]} features")
log(f"[model] channels kept: {', '.join(FEATCH)}")
log(f"[model] channels dropped: {', '.join(c for c in CH if c not in FEATCH)}  "
    f"(no marginal contribution); comp_up_3ss dropped (settled empty slot)")

def zparams(M, fit_idx):
    Cg = K.centre_by_gene(M)
    mu = np.nanmean(Cg[fit_idx], 0); mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.nanstd(Cg[fit_idx], 0); sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return (np.where(np.isfinite(Cg), Cg, mu) - mu) / sd, mu, sd

# step 1: fit, evaluate, save
REPORT = {}
def report(M, tag, lam=None):
    if lam is None: lam, _ = K.pick_lambda(M, tag)
    sc, b = K.fit_score(M, K.trv, K.te, lam)
    r = K.suite(sc); r["lam"] = lam; r["p"] = M.shape[1]; REPORT[tag] = r
    log(f"[test] {tag:34s} p={M.shape[1]:>4d}  within-gene {r['winrate']:.4f} "
        f"[{r['winrate_lo']:.4f},{r['winrate_hi']:.4f}]  P@1 {r['p_at_1']:.4f} "
        f"[{r['p_at_1_lo']:.4f},{r['p_at_1_hi']:.4f}]  AUROC {r['auroc']:.4f} "
        f"[{r['auroc_lo']:.4f},{r['auroc_hi']:.4f}]  AUPRC {r['auprc']:.4f} "
        f"[{r['auprc_lo']:.4f},{r['auprc_hi']:.4f}]")
    return r, b

LAM, IV = K.pick_lambda(FINAL, "FINAL 51")
R_FIN, _ = report(FINAL, "binned U1 + acceptor (51)", LAM)

Zf, MU, SD = zparams(FINAL, K.trv)
assert np.allclose(Zf, K.zmat(FINAL, K.trv), equal_nan=True), "standardisation mismatch"
BETA = K.fit_ridge(np.column_stack([np.ones(len(K.trv)), Zf[K.trv]]), K.y[K.trv], LAM)
sc_check = np.column_stack([np.ones(len(K.te)), Zf[K.te]]) @ BETA
assert abs(K.winrate(sc_check, K.GRP_TE) - R_FIN["winrate"]) < 1e-12, "saved model mismatch"
np.savez_compressed("splice_model_final.npz", beta=BETA, mu=MU, sd=SD,
                    names=np.array(NAMES, dtype=object), lam=LAM, channels=np.array(FEATCH),
                    bin_lo=BINLO, up=UPW, dn=DNW, inner_val=IV,
                    note=np.array("logit = beta0 + sum_j beta_j * (gene-centred x_j - mu_j)/sd_j"))
with open("splice_model_final_coefficients.tsv", "w") as fh:
    fh.write("feature\tchannel\tbin_lo\tbin_centre\tcoefficient\n")
    fh.write(f"(intercept)\t\t\t\t{BETA[0]:.6f}\n")
    for j, nm in enumerate(NAMES):
        if nm == "spl_upstream_max_me3":
            fh.write(f"{nm}\tacceptor\t\t\t{BETA[j+1]:.6f}\n")
        else:
            ch = nm.split("[")[0]; b = BINLO[j % NBIN]
            fh.write(f"{nm}\t{ch}\t{b}\t{b+50}\t{BETA[j+1]:.6f}\n")
log(f"[save] splice_model_final.npz  lambda {LAM:g}  inner-val {IV:.4f}  "
    f"|beta| max {np.abs(BETA[1:]).max():.4f}, acceptor beta {BETA[-1]:+.4f}")

# the ladder, all on this split and window
report(ACC[:, None], "acceptor alone (1)")
from fastparquet import ParquetFile
_F = ParquetFile("shortlist_feat_all.parquet").to_pandas()
_SPL = [c for c in _F.columns if c.startswith("spl_")]
_key = {(g, c, int(ct), s_): i for i, (g, c, ct, s_) in
        enumerate(zip(_F.gene, _F.chrom, _F.center, _F.strand))}
_row = np.array([_key.get((g, c, int(ct), s_), -1) for g, c, ct, s_ in
                 zip(K.C0["gene"][K.ok], K.C0["chrom"][K.ok],
                     K.C0["center"][K.ok], K.C0["strand"][K.ok])])
_h = _row >= 0
_X = np.full((len(K.y), len(_SPL)), np.nan); _X[_h] = _F.loc[_row[_h], _SPL].to_numpy(float)
_S3 = [_SPL.index(c) for c in ("spl_pas_prox_u1", "spl_upstream_max_me3", "spl_comp_up_3ss")]
r3, _ = report(_X[:, _S3], "scalar splice model (3)", 1.0)
assert abs(r3["winrate"] - 0.66557) < 5e-4, "scalar baseline does not reproduce 0.6656"
FULL = np.column_stack([P.reshape(len(P), -1), ACC])
rF, _ = report(FULL, "full binned + acceptor (126)")
assert abs(rF["winrate"] - 0.7101) < 5e-4, "full model does not reproduce 0.7101"
log("[check] scalar baseline reproduces 0.6656 and the full binned model reproduces 0.7101")

CS = np.load("cnn_model_scores.npz", allow_pickle=True)
assert (CS["gene"][K.ok] == K.gene).all() and (CS["y"][K.ok] == K.y).all(), "CNN row alignment"
cnn = CS["logit"][K.ok][K.te].astype(float)
R_CNN = K.suite(cnn); R_CNN["lam"] = np.nan; R_CNN["p"] = np.nan
REPORT["sequence CNN"] = R_CNN
log(f"[test] {'sequence CNN':34s} p=   -  within-gene {R_CNN['winrate']:.4f} "
    f"[{R_CNN['winrate_lo']:.4f},{R_CNN['winrate_hi']:.4f}]  P@1 {R_CNN['p_at_1']:.4f}  "
    f"AUROC {R_CNN['auroc']:.4f}  AUPRC {R_CNN['auprc']:.4f}")
assert abs(R_CNN["winrate"] - 0.82110) < 5e-4, "CNN does not reproduce 0.8211"

def paired(a, b, tag):
    da = REPORT[a]["_wg"][K.BS].mean(1) - REPORT[b]["_wg"][K.BS].mean(1)
    d = REPORT[a]["winrate"] - REPORT[b]["winrate"]
    lo, hi = np.percentile(da, [2.5, 97.5])
    p = max(2 * min((da <= 0).mean(), (da >= 0).mean()), 1 / K.NBOOT)
    log(f"[paired] {tag:46s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  p = {p:.4f}"
        f"{'' if lo * hi > 0 else '   (CI spans 0)'}")
    return d, lo, hi, p
log("")
FN = "binned U1 + acceptor (51)"
PRS = {
 "final minus acceptor": paired(FN, "acceptor alone (1)", "final (51) minus acceptor alone (1)"),
 "final minus scalar":   paired(FN, "scalar splice model (3)", "final (51) minus scalar model (3)"),
 "final minus full":     paired(FN, "full binned + acceptor (126)", "final (51) minus full binned (126)"),
 "final minus cnn":      paired(FN, "sequence CNN", "final (51) minus sequence CNN"),
}

# step 2: selection order, u1 against maxent 5'ss
log("")
ME5CH = ("me5_dens8", "me5_max")
BLK = {c: P[:, CH.index(c), :] for c in CH}
BASE_ME5 = np.column_stack([BLK[c] for c in ME5CH] + [ACC])
BASE_U1 = np.column_stack([BLK[c] for c in FEATCH] + [ACC])
BOTH = np.column_stack([BLK[c] for c in FEATCH] + [BLK[c] for c in ME5CH] + [ACC])
log("[step2] both arms hold the acceptor scalar fixed, so the comparison is donor-channel")
log("[step2] against donor-channel, not against nothing.")
ORD = {}
for tag, base, bname, aname in (
        ("U1 on MaxEnt", BASE_ME5, "MaxEnt 5'ss channels + acceptor", "U1 density channels"),
        ("MaxEnt on U1", BASE_U1, "U1 density channels + acceptor", "MaxEnt 5'ss channels")):
    lb, ivb = K.pick_lambda(base, f"base: {bname}")
    lj, ivj = K.pick_lambda(BOTH, "joint: both donor channel sets + acceptor") if tag == "U1 on MaxEnt" \
        else (ORD["U1 on MaxEnt"]["lam_joint"], ORD["U1 on MaxEnt"]["iv_joint"])
    rb, _ = report(base, f"base {bname}", lb)
    rj = REPORT.get("joint both + acceptor")
    if rj is None: rj, _ = report(BOTH, "joint both + acceptor", lj)
    dv, lo, hi, pv = paired("joint both + acceptor", f"base {bname}",
                            f"{tag}: adding {aname}")
    ORD[tag] = dict(base=bname, add=aname, iv_base=ivb, iv_joint=ivj, iv_gain=ivj - ivb,
                    lam_base=lb, lam_joint=lj, test_base=rb["winrate"], test_joint=rj["winrate"],
                    d=dv, lo=lo, hi=hi, p=pv)
    log(f"[step2] {tag:14s} inner-val {ivb:.4f} -> {ivj:.4f}  gain {ivj-ivb:+.4f} "
        f"({'PASSES' if ivj - ivb > 0.005 else 'fails'} the 0.005 rule)")
GA, GB = ORD["U1 on MaxEnt"]["iv_gain"], ORD["MaxEnt on U1"]["iv_gain"]
ASYM = (GA > 0.005) != (GB > 0.005)
log(f"[step2] VERDICT: {'ASYMMETRIC' if ASYM else 'SYMMETRIC'} -- "
    + (f"U1 adds {GA:+.4f} on a MaxEnt base while MaxEnt adds {GB:+.4f} on a U1 base, so U1 "
       f"carries donor signal MaxEnt misses and the reverse does not hold" if ASYM else
       f"U1 adds {GA:+.4f} and MaxEnt adds {GB:+.4f}; neither is clearly preferred by the rule"))
with open("splice_model_order.tsv", "w") as fh:
    fh.write("direction\tbase\tadded\tinner_val_base\tinner_val_joint\tinner_val_gain\t"
             "test_base\ttest_joint\ttest_delta\tci_lo\tci_hi\tp\tpasses_0.005\n")
    for t, o in ORD.items():
        fh.write(f"{t}\t{o['base']}\t{o['add']}\t{o['iv_base']:.5f}\t{o['iv_joint']:.5f}\t"
                 f"{o['iv_gain']:+.5f}\t{o['test_base']:.5f}\t{o['test_joint']:.5f}\t"
                 f"{o['d']:+.5f}\t{o['lo']:+.5f}\t{o['hi']:+.5f}\t{o['p']:.4f}\t"
                 f"{'yes' if o['iv_gain'] > 0.005 else 'no'}\n")

LADDER = ["acceptor alone (1)", "scalar splice model (3)", FN,
          "full binned + acceptor (126)", "sequence CNN"]
SHUF_FULL, SHUF_BASE = 0.59531, 0.71010
SURV = (SHUF_FULL - 0.5) / (SHUF_BASE - 0.5)
with open("splice_model_final_metrics.tsv", "w") as fh:
    fh.write("model\tladder\tmetric\tvalue\tci_lo\tci_hi\tn_features\tlambda\tn_genes\tn_peaks\n")
    def cell(v, fmt="g"):
        return "" if v is None or not np.isfinite(v) else format(v, fmt)
    for tag, r in REPORT.items():
        np_, lm = cell(r["p"], "d"), cell(r["lam"])
        for m in ("winrate", "p_at_1", "auroc", "auprc"):
            fh.write(f"{tag}\t{'yes' if tag in LADDER else 'no'}\t{m}\t{r[m]:.5f}\t"
                     f"{r[m+'_lo']:.5f}\t{r[m+'_hi']:.5f}\t{np_}\t{lm}\t"
                     f"{len(K.GRP_TE)}\t{len(K.te)}\n")
    for k, (d, lo, hi, pv) in PRS.items():
        fh.write(f"paired: {k}\tno\tdelta_winrate\t{d:+.5f}\t{lo:+.5f}\t{hi:+.5f}\t\t{pv:.4f}\t"
                 f"{len(K.GRP_TE)}\t{len(K.te)}\n")
    fh.write(f"dinuc-shuffle: fraction of lift surviving\tno\tfraction\t{SURV:.5f}\t\t\t\t\t"
             f"{len(K.GRP_TE)}\t{len(K.te)}\n")
log("")
log(f"[shuffle] full binned + acceptor {SHUF_BASE:.4f} -> all-shuffled {SHUF_FULL:.4f}; "
    f"{100*SURV:.1f}% of the lift over 0.5 survives dinucleotide shuffle")
log(f"[genes]   {len(K.GRP_TE)} test genes, {len(K.te)} test peaks")
log("[out] splice_model_final.npz splice_model_final_coefficients.tsv "
    "splice_model_final_metrics.tsv splice_model_order.tsv")
