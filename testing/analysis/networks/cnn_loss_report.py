#!/usr/bin/env python3
# collect the three-loss comparison and decide the verdict
import sys, glob, os
import numpy as np
import binned_common as K
from binned_common import log

SEEDS = [42, 43, 44]
LOSSES = ["bce", "pairwise", "listwise"]
THR = 0.02

def load(tag):
    f = f"{tag}_scores.npz"
    if not os.path.exists(f): return None
    Z = np.load(f, allow_pickle=True)
    return Z["logit"].astype(np.float64)[K.ok][K.te], Z

log(f"[set] {len(K.GRP_TE)} test genes, {len(K.te)} test sites, "
    f"{K.NBOOT} shared bootstrap resamples")

# anchor
ANC = np.load("cnn_model_scores.npz", allow_pickle=True)["logit"].astype(np.float64)[K.ok][K.te]
RA = K.suite(ANC)
log(f"[anchor] shipped cnn_model.pt: within-gene {RA['winrate']:.5f}  P@1 {RA['p_at_1']:.5f}  "
    f"AUROC {RA['auroc']:.5f}")
assert abs(RA["winrate"] - 0.82110) < 5e-5, \
    f"anchor is not the reported CNN: within-gene {RA['winrate']:.5f}, expected 0.82110"
assert abs(RA["auroc"] - 0.81849) < 5e-5, \
    f"anchor pooled AUROC {RA['auroc']:.5f}, expected 0.81849"

R = {("anchor", 0): RA}
for L in LOSSES:
    for s in SEEDS:
        got = load(f"cnn_loss_{L}_s{s}")
        if got is None: log(f"[warn] missing cnn_loss_{L}_s{s}_scores.npz"); continue
        sc, Z = got
        R[(L, s)] = K.suite(sc)
        R[(L, s)]["best_epoch"] = int(Z["best_epoch"]); R[(L, s)]["val_wg"] = float(Z["val_wg"])
missing = [(L, s) for L in LOSSES for s in SEEDS if (L, s) not in R]
if missing: log(f"[warn] {len(missing)} runs missing: {missing}")

# per-run table
with open("cnn_loss_metrics.tsv", "w") as fh:
    fh.write("loss\tseed\tbest_epoch\tval_winrate\twinrate\twinrate_lo\twinrate_hi\t"
             "p_at_1\tp_at_1_lo\tp_at_1_hi\tauroc\tauroc_lo\tauroc_hi\t"
             "auprc\tauprc_lo\tauprc_hi\tn_genes\tn_sites\n")
    for (L, s), r in R.items():
        fh.write(f"{L}\t{s}\t{r.get('best_epoch','')}\t{r.get('val_wg',float('nan')):.5f}\t"
                 + "\t".join(f"{r[m+sfx]:.5f}" for m in ("winrate", "p_at_1", "auroc", "auprc")
                             for sfx in ("", "_lo", "_hi"))
                 + f"\t{len(K.GRP_TE)}\t{len(K.te)}\n")
log("[out] cnn_loss_metrics.tsv")

log("\n[per run] test-set metrics, 1,623 held-out genes")
log(f"{'loss':>9} {'seed':>5} {'ep':>3} {'within-gene':>22} {'P@1':>22} {'AUROC':>8} {'AUPRC':>8}")
for (L, s), r in R.items():
    log(f"{L:>9} {s:>5} {str(r.get('best_epoch','-')):>3} "
        f"{r['winrate']:.4f} [{r['winrate_lo']:.4f},{r['winrate_hi']:.4f}] "
        f"{r['p_at_1']:.4f} [{r['p_at_1_lo']:.4f},{r['p_at_1_hi']:.4f}] "
        f"{r['auroc']:>8.4f} {r['auprc']:>8.4f}")

# seed spread per loss
log("\n[seed spread] the noise floor the 0.02 threshold has to clear")
SP = {}
for L in LOSSES:
    v = np.array([R[(L, s)]["winrate"] for s in SEEDS if (L, s) in R])
    q = np.array([R[(L, s)]["p_at_1"] for s in SEEDS if (L, s) in R])
    if not len(v): continue
    SP[L] = dict(wg_mean=v.mean(), wg_sd=v.std(ddof=1) if len(v) > 1 else np.nan,
                 wg_rng=v.max() - v.min(), p1_mean=q.mean(),
                 p1_rng=q.max() - q.min(), n=len(v))
    log(f"  {L:>9} n={len(v)}  within-gene mean {v.mean():.4f} "
        f"sd {SP[L]['wg_sd']:.4f} range {v.max()-v.min():.4f} "
        f"[{v.min():.4f}, {v.max():.4f}]   P@1 mean {q.mean():.4f} range {q.max()-q.min():.4f}")

# paired bootstrap vs bce
def paired(a, b, key):
    da = a["_wg"] if key == "winrate" else a["_p1"]
    db = b["_wg"] if key == "winrate" else b["_p1"]
    d = da[K.BS].mean(1) - db[K.BS].mean(1)
    return float(da.mean() - db.mean()), float(np.percentile(d, 2.5)), \
           float(np.percentile(d, 97.5)), float((d > 0).mean())

rows = []
log("\n[paired bootstrap] each arm minus BCE, same genes, same 1,000 resamples")
log(f"{'comparison':>34} {'metric':>10} {'delta':>9} {'95% CI':>20} {'P(A>B)':>8}")
for L in ("pairwise", "listwise"):
    for s in SEEDS:
        if (L, s) not in R or ("bce", s) not in R: continue
        for key in ("winrate", "p_at_1"):
            d, lo, hi, pr = paired(R[(L, s)], R[("bce", s)], key)
            rows.append((f"{L}_s{s}", f"bce_s{s}", key, d, lo, hi, pr))
            log(f"{L+' s'+str(s)+' vs bce s'+str(s):>34} {key:>10} {d:>+9.4f} "
                f"[{lo:>+7.4f},{hi:>+7.4f}] {pr:>8.3f}")
for L in LOSSES:
    for s in SEEDS:
        if (L, s) not in R: continue
        for key in ("winrate", "p_at_1"):
            d, lo, hi, pr = paired(R[(L, s)], RA, key)
            rows.append((f"{L}_s{s}", "anchor", key, d, lo, hi, pr))
with open("cnn_loss_paired.tsv", "w") as fh:
    fh.write("arm\treference\tmetric\tdelta\tci_lo\tci_hi\tp_arm_better\tn_boot\tn_genes\n")
    for a, b, k, d, lo, hi, pr in rows:
        fh.write(f"{a}\t{b}\t{k}\t{d:.5f}\t{lo:.5f}\t{hi:.5f}\t{pr:.4f}\t{K.NBOOT}\t"
                 f"{len(K.GRP_TE)}\n")
log("[out] cnn_loss_paired.tsv")

log("\n[VERDICT]")
if "bce" not in SP:
    log("  cannot decide: no BCE runs"); sys.exit(0)
base = SP["bce"]["wg_mean"]
noise = max(SP[L]["wg_rng"] for L in SP)
log(f"  BCE seed-mean within-gene {base:.4f}; largest within-loss seed range {noise:.4f}")
verdicts = []
for L in ("pairwise", "listwise"):
    if L not in SP: continue
    d = SP[L]["wg_mean"] - base
    ds = [(r[3], r[4], r[5]) for r in rows
          if r[0].startswith(L) and r[1].startswith("bce") and r[2] == "winrate"]
    excl = sum(1 for m, lo, hi in ds if lo > 0 or hi < 0)
    log(f"  {L:>9}: seed-mean within-gene {SP[L]['wg_mean']:.4f}, delta vs BCE {d:+.4f}; "
        f"paired CI excludes 0 in {excl}/{len(ds)} seeds; "
        f"per-seed deltas {', '.join(f'{m:+.4f}' for m, _, _ in ds)}")
    verdicts.append((L, d, excl, len(ds)))
big = [(L, d) for L, d, e, n in verdicts if d > THR and e == n and n > 0]
if big:
    for L, d in big:
        log(f"  IMPROVED: {L} beats BCE by {d:+.4f} > {THR} with every per-seed paired CI "
            f"excluding zero. This is a real gain -- {L} should become the reported model, "
            f"retrained under LOCO to confirm.")
elif all(abs(d) <= THR for _, d, _, _ in verdicts):
    log(f"  ROBUST: every loss lands within {THR} of BCE on within-gene win-rate "
        f"(deltas {', '.join(f'{L} {d:+.4f}' for L, d, _, _ in verdicts)}), and the spread is "
        f"comparable to the {noise:.4f} seed range inside a single loss. Within-gene performance "
        f"is a property of the architecture and the data, NOT of the pooled-BCE formulation; the "
        f"reported 0.821 is not an artifact of the loss.")
else:
    log(f"  MIXED: deltas {', '.join(f'{L} {d:+.4f}' for L, d, _, _ in verdicts)} against a "
        f"threshold of {THR} and a seed range of {noise:.4f}; see the per-seed paired CIs above.")
