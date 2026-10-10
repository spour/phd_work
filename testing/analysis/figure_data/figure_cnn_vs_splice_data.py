#!/usr/bin/env python3
# assemble the cnn-vs-splice comparison figure's four panels from outputs that already exist
import sys
import numpy as np
import binned_common as K
from binned_common import log

NB = 2000
te, GRP = K.te, K.GRP_TE
NG, NS = len(GRP), len(te)
log(f"[set] {NG} held-out test genes, {NS} peaks, base rate {K.y[te].mean():.5f}")

# scores
Z = np.load("binned_profiles.npz", allow_pickle=True)
CH = [str(c) for c in Z["channels"]]
D4 = Z["P"][:, CH.index("u1_dens4"), :].astype(np.float64)
M26 = np.column_stack([D4, K.ACC]); assert M26.shape[1] == 26
LAM, _ = K.pick_lambda(M26, "26-feature model")
SPL, _ = K.fit_score(M26, K.trv, te, LAM)
ACCS, _ = K.fit_score(K.ACC[:, None], K.trv, te, 1.0)
WGC = np.load("cnn_data.npz", allow_pickle=True)["window_gc"][K.ok]
GCS, _ = K.fit_score(WGC[:, None], K.trv, te, 1.0)
CNN = np.load("cnn_model_scores.npz", allow_pickle=True)["logit"].astype(np.float64)[K.ok][te]

R = {n: K.suite(s) for n, s in
     (("Sequence CNN", CNN), ("Splice model (26 features)", SPL),
      ("Acceptor 3'ss alone", ACCS), ("Window GC alone", GCS))}
assert abs(R["Sequence CNN"]["winrate"] - 0.82110) < 5e-5
assert abs(R["Sequence CNN"]["auroc"] - 0.81849) < 5e-5
assert abs(R["Sequence CNN"]["auprc"] - 0.44920) < 5e-5
assert abs(R["Splice model (26 features)"]["winrate"] - 0.69928) < 5e-4, \
    f"26-feature within-gene {R['Splice model (26 features)']['winrate']:.5f}, expected 0.69928"
assert abs(R["Splice model (26 features)"]["p_at_1"] - 0.47197) < 5e-4
log("[check] CNN reproduces 0.82110 / 0.81849 / 0.44920; splice-26 reproduces 0.69928 / 0.47197")

BASE = float(K.y[te].mean())
with open("figCS_pooled.tsv", "w") as fh:
    fh.write("model\tmetric\tvalue\tci_lo\tci_hi\tbase_rate\twithin_gene\tn_genes\tn_peaks\n")
    for n, r in R.items():
        for m in ("auroc", "auprc"):
            fh.write(f"{n}\t{m}\t{r[m]:.5f}\t{r[m+'_lo']:.5f}\t{r[m+'_hi']:.5f}\t"
                     f"{BASE:.5f}\t{r['winrate']:.5f}\t{NG}\t{NS}\n")
log("[out] figCS_pooled.tsv")
log("[A] pooled, FOR REFERENCE ONLY")
for n, r in R.items():
    log(f"  {n:>27}  AUROC {r['auroc']:.4f} [{r['auroc_lo']:.4f},{r['auroc_hi']:.4f}]  "
        f"AUPRC {r['auprc']:.4f} [{r['auprc_lo']:.4f},{r['auprc_hi']:.4f}]  "
        f"(within-gene {r['winrate']:.4f})")
log(f"  AUPRC base rate {BASE:.5f}")

# b: recall@k, matched genes
KMAX = 10
def recall_curve(sc):
    rk = np.array([1 + int((sc[c] >= sc[r[0]]).sum()) for r, c in GRP])
    n = np.array([len(r) + len(c) for r, c in GRP])
    hit = np.array([(rk <= k).astype(float) for k in range(1, KMAX + 1)])
    chc = np.array([np.minimum(k, n) / n for k in range(1, KMAX + 1)])
    return hit, chc, rk, n

HC, CH_, RKC, NN = recall_curve(CNN)
HS, CH2, RKS, _ = recall_curve(SPL)
assert np.allclose(CH_, CH2)
rs = np.random.default_rng(K.SEED).integers(0, NG, size=(NB, NG))
def ci(h): 
    b = h[:, rs].mean(2)
    return np.percentile(b, 2.5, axis=1), np.percentile(b, 97.5, axis=1)
loC, hiC = ci(HC); loS, hiS = ci(HS)
dl, dh = np.percentile((HC[:, rs].mean(2) - HS[:, rs].mean(2)), [2.5, 97.5], axis=1)

log(f"[B] recall@k on the SAME {NG} held-out genes, single chromosome holdout for both; "
    f"candidates/gene median {np.median(NN):.0f} mean {NN.mean():.2f}")
with open("figCS_recall.tsv", "w") as fh:
    fh.write("model\tk\trecall\tci_lo\tci_hi\tchance\tlift\tdelta_vs_splice\tdelta_lo\tdelta_hi\t"
             "scope\tn_genes\n")
    for lab, h, lo, hi in (("Sequence CNN", HC, loC, hiC),
                           ("Splice model (26 features)", HS, loS, hiS)):
        for i in range(KMAX):
            d = (HC[i].mean() - HS[i].mean()) if lab.startswith("Sequence") else float("nan")
            fh.write(f"{lab}\t{i+1}\t{h[i].mean():.5f}\t{lo[i]:.5f}\t{hi[i]:.5f}\t"
                     f"{CH_[i].mean():.5f}\t{h[i].mean()-CH_[i].mean():+.5f}\t"
                     f"{d:.5f}\t{dl[i]:.5f}\t{dh[i]:.5f}\t"
                     f"held-out test genes, chromosome holdout\t{NG}\n")
for k in (1, 3):
    i = k - 1
    log(f"  k={k}: CNN {HC[i].mean():.4f} [{loC[i]:.4f},{hiC[i]:.4f}]  "
        f"splice {HS[i].mean():.4f} [{loS[i]:.4f},{hiS[i]:.4f}]  "
        f"chance {CH_[i].mean():.4f}  delta {HC[i].mean()-HS[i].mean():+.4f} "
        f"[{dl[i]:+.4f},{dh[i]:+.4f}]")
log(f"  recall@1 is the strict P@1: CNN {HC[0].mean():.5f}, splice {HS[0].mean():.5f} "
    f"(tie-aware equivalents {R['Sequence CNN']['p_at_1']:.5f} and "
    f"{R['Splice model (26 features)']['p_at_1']:.5f}; the CNN has no ties)")
log("[out] figCS_recall.tsv")

# c: shuffle
S = np.genfromtxt("splice51_shuffle.tsv", names=True, delimiter="\t", dtype=None, encoding="utf-8")
LAB = {"cnn": "Sequence CNN", "splice_positional_51": "Splice model (51 features)"}
with open("figCS_shuffle.tsv", "w") as fh:
    fh.write("model\tstate\twinrate\tci_lo\tci_hi\tdrop\tdrop_lo\tdrop_hi\tpct_lost\t"
             "n_genes\tk_shuffles\n")
    for row in S:
        if str(row["model"]) not in LAB: continue
        m = LAB[str(row["model"])]
        for st, v, lo, hi in (("Real sequence", row["real"], row["real_lo"], row["real_hi"]),
                              ("Dinucleotide shuffled", row["shuffled"], row["shuffled_lo"],
                               row["shuffled_hi"])):
            fh.write(f"{m}\t{st}\t{v:.5f}\t{lo:.5f}\t{hi:.5f}\t{row['drop']:.5f}\t"
                     f"{row['drop_lo']:.5f}\t{row['drop_hi']:.5f}\t"
                     f"{row['pct_of_lift_lost']:.3f}\t{int(row['n_genes'])}\t"
                     f"{int(row['k_shuffles'])}\n")
        log(f"[C] {m}: {row['real']:.4f} -> {row['shuffled']:.4f}, drop {row['drop']:.4f} "
            f"[{row['drop_lo']:.4f},{row['drop_hi']:.4f}], "
            f"{row['pct_of_lift_lost']:.1f}% of the lift above 0.5 lost")
log("[C] the shuffle control exists for the 51-feature model (real 0.69452), NOT for the locked "
    "26-feature model (0.69928); the panel is labelled 51-feature accordingly")
log("[out] figCS_shuffle.tsv")

# d: partition, recomputed
Q = np.genfromtxt("splice_sweep_partition.tsv", names=True, delimiter="\t", dtype=None,
                  encoding="utf-8")
P51 = {str(r["key"]): float(r["n_cnn_only"]) for r in Q if str(r["section"]) == "partition"}
log(f"[D] for reference, the shipped CNN-vs-51-feature partition: "
    + ", ".join(f"{k} {int(P51[k])}" for k in
                ("both right", "CNN only", "splice only", "both wrong")))

okc, oks = RKC == 1, RKS == 1
CELL = [("Both rank the real end first", okc & oks), ("CNN only", okc & ~oks),
        ("Splice model only", ~okc & oks), ("Neither", ~okc & ~oks)]
assert sum(int(m.sum()) for _, m in CELL) == NG
log(f"[D] recomputed CNN-vs-26-feature partition of {NG} test genes: "
    + ", ".join(f"{k} {int(m.sum())} ({100*m.mean():.1f}%)" for k, m in CELL))

C0 = np.load("analysis_columns.npz", allow_pickle=True)
CEN = C0["center"][K.ok][te].astype(float); CHR = C0["chrom"][K.ok][te]
TRUE = np.array([r[0] for r, c in GRP])
U1M = D4[K.ok][te].mean(1)
def nearest_comp(gi):
    r, c = GRP[gi]
    return np.min(np.abs(CEN[c] - CEN[r[0]])) / 1000.0 if len(c) else np.nan
PROPS = [("Acceptor 3'ss of the real end", K.ACC[K.ok][te][TRUE]),
         ("Competitors in the gene", np.array([len(c) for r, c in GRP], float)),
         ("U1 duplex density of the real end", U1M[TRUE]),
         ("Distance to nearest competitor (kb)", np.array([nearest_comp(i) for i in range(NG)])),
         ("GC contrast of the real end", C0["gc_contrast"][K.ok][te][TRUE])]
ALPHA = 0.05 / len(PROPS)

def rank_auc(a, b):
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    v = np.concatenate([a, b]); _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); rk = ((cs - cnt) + cs + 1) / 2.0
    R1 = rk[inv][:n1].sum()
    U = R1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    N = n1 + n2
    tie = (cnt ** 3 - cnt).sum()
    sd = np.sqrt(n1 * n2 / 12.0 * ((N + 1) - tie / (N * (N - 1.0))))
    z = (U - mu) / sd if sd > 0 else 0.0
    az = abs(z)
    if az <= 5.0:
        x = az / np.sqrt(2.0); t = 1.0 / (1.0 + 0.3275911 * x)
        erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
                   + 0.254829592) * t * np.exp(-x * x)
        pv = max(1.0 - erf, 0.0)
    else:
        pv = 2.0 * np.exp(-0.5 * az * az) / (az * np.sqrt(2.0 * np.pi)) * (1.0 - 1.0 / (az * az))
    return U / (n1 * n2), float(z), float(pv), n1, n2

MA = dict(CELL)["CNN only"]; MB = dict(CELL)["Both rank the real end first"]
with open("figCS_partition.tsv", "w") as fh:
    fh.write("kind\tlabel\tvalue\tn\tn_ref\tz\tp\tsignificant\talpha\tmodel\tn_genes\n")
    for k, m in CELL:
        fh.write(f"partition\t{k}\t{m.mean():.5f}\t{int(m.sum())}\t\t\t\t\t\t"
                 f"26-feature\t{NG}\n")
    for nm, v in PROPS:
        auc, z, pv, n1, n2 = rank_auc(v[MA], v[MB])
        fh.write(f"property\t{nm}\t{auc:.5f}\t{n1}\t{n2}\t{z:.4f}\t{pv:.3e}\t"
                 f"{'yes' if pv < ALPHA else 'no'}\t{ALPHA:.5f}\t26-feature\t{NG}\n")
        log(f"  {nm:>36}  AUC {auc:.5f}  z {z:+.3f}  p {pv:.3g}  "
            f"medians {np.nanmedian(v[MA]):.3f} vs {np.nanmedian(v[MB]):.3f}"
            + ("  *" if pv < ALPHA else ""))
aA, _, _, _, _ = rank_auc(PROPS[0][1][MA], PROPS[0][1][MB])
log(f"[D] weak-acceptor separation: AUC {aA:.5f} separating the {int(MA.sum())} CNN-only genes "
    f"from the {int(MB.sum())} both-right genes on the acceptor of the real end. Below 0.5 means "
    f"the CNN's exclusive wins sit on WEAKER acceptors "
    f"({np.nanmedian(PROPS[0][1][MA]):.2f} against {np.nanmedian(PROPS[0][1][MB]):.2f}) -- "
    f"exactly where a splice-feature model has least to read")
log("[out] figCS_partition.tsv")
