#!/usr/bin/env python3
# what each block separates: canonical - competitor versus competitor - decoy
import csv, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, RIDGE, NBOOT = 42, 1.0, 300
rng = np.random.default_rng(SEED)

M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
P = np.load("binned_features178.npz"); PN = [str(s) for s in P["names"]]; PX = P["X"][ridx].astype(float)
Q = np.load("decoy_features.npz"); QX = Q["X"].astype(float)
DEC = list(csv.DictReader(open("decoy_positions.tsv"), delimiter="\t"))
dgene = np.array([r["gene"] for r in DEC]); dchrom = np.array([r["chrom"] for r in DEC])
ROW = np.load("corr_rowmask.npz")["ROW"]
y = D.is_canonical.to_numpy().astype(float); gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy()
GCB = [i for i, n in enumerate(PN) if n.endswith("_gc")]
SPL = [i for i, n in enumerate(PN) if n.startswith("bin") and not n.endswith("_gc")]
K3 = [PN.index(n) for n in ["spl_upstream_max_me3", "spl_pas_prox_u1", "spl_u1_suppression_up"]]
BLOCKS = {"three features (3)": K3, "splice-binned (150)": SPL, "GC-binned (25)": GCB}
DT = {r["block"]: r for r in csv.DictReader(open("decoy_test.tsv"), delimiter="\t")}
DTKEY = {"three features (3)": "k3 splicing (3)", "splice-binned (150)": "splice-binned (150)",
         "GC-binned (25)": "GC-binned (25)"}

def fit_ridge(Xt, yt, lam=RIDGE):
    Pm = np.eye(Xt.shape[1]) * lam; Pm[0, 0] = 0.0; b = np.zeros(Xt.shape[1])
    for _ in range(80):
        eta = np.clip(Xt @ b, -30, 30); mu = 1 / (1 + np.exp(-eta)); w = np.maximum(mu * (1 - mu), 1e-9)
        z = eta + (yt - mu) / w; XtW = Xt.T * w
        try: nb = np.linalg.solve(XtW @ Xt + Pm, XtW @ z + Pm @ b)
        except np.linalg.LinAlgError: nb = np.linalg.lstsq(XtW @ Xt + Pm, XtW @ z + Pm @ b, rcond=None)[0]
        if np.max(np.abs(nb - b)) < 1e-9: b = nb; break
        b = nb
    return b
CH = sorted(set(chrom))
gi = defaultdict(lambda: ([], []))
for i in np.where(ROW)[0]: gi[gene[i]][0].append(i)
for j in range(len(DEC)): gi[dgene[j]][1].append(j)
GK = [g for g in gi if len(gi[g][0]) and len(gi[g][1])]
PIX = {g: np.array(gi[g][0]) for g in GK}; DIX = {g: np.array(gi[g][1]) for g in GK}
BOOT = rng.integers(0, len(GK), size=(NBOOT, len(GK)))

print("\n===== WHAT EACH BLOCK SEPARATES (LOCO discriminant, fitted canonical vs competitor) =====")
print(f"{'block':<22} {'canonical':>10} {'competitor':>11} {'decoy':>9} "
      f"{'canon-comp':>12} {'95% CI':>17} {'comp-decoy':>12} {'95% CI':>17}")
with open("corr_panel_i.tsv", "w") as fh:
    fh.write("block\tn_feat\tcontrast\tgap\tci_lo\tci_hi\tmean_canonical\tmean_competitor\t"
             "mean_decoy\tdecoy_auroc\tdecoy_p1\tdecoy_p1_chance\n")
    for nm, cols in BLOCKS.items():
        Xp = PX[:, cols]; Xd = QX[:, cols]
        sp = np.full(len(y), np.nan); sd_ = np.full(len(Xd), np.nan)
        for c in CH:
            tr = ROW & (chrom != c)
            mu = np.nanmean(Xp[tr], 0); sdv = np.nanstd(Xp[tr], 0); sdv[sdv == 0] = 1.0
            mu = np.where(np.isfinite(mu), mu, 0.0)
            A = np.column_stack([np.ones(int(tr.sum())), (np.where(np.isfinite(Xp[tr]), Xp[tr], mu) - mu) / sdv])
            b = fit_ridge(A, y[tr])
            te = ROW & (chrom == c)
            if te.sum():
                sp[te] = np.column_stack([np.ones(int(te.sum())),
                                          (np.where(np.isfinite(Xp[te]), Xp[te], mu) - mu) / sdv]) @ b
            md = dchrom == c
            if md.sum():
                sd_[md] = np.column_stack([np.ones(int(md.sum())),
                                           (np.where(np.isfinite(Xd[md]), Xd[md], mu) - mu) / sdv]) @ b
        can = ROW & (y == 1); com = ROW & (y == 0)
        mc, mk, md_ = np.nanmean(sp[can]), np.nanmean(sp[com]), np.nanmean(sd_)
        g1 = np.empty(NBOOT); g2 = np.empty(NBOOT)
        for t in range(NBOOT):
            sel = [GK[j] for j in BOOT[t]]
            pi = np.concatenate([PIX[g] for g in sel]); di = np.concatenate([DIX[g] for g in sel])
            a = sp[pi]; yy = y[pi]
            g1[t] = np.nanmean(a[yy == 1]) - np.nanmean(a[yy == 0])
            g2[t] = np.nanmean(a[yy == 0]) - np.nanmean(sd_[di])
        dt = DT[DTKEY[nm]]
        print(f"{nm:<22} {mc:>10.3f} {mk:>11.3f} {md_:>9.3f} {mc-mk:>+12.3f} "
              f"[{np.percentile(g1,2.5):+.3f},{np.percentile(g1,97.5):+.3f}] {mk-md_:>+12.3f} "
              f"[{np.percentile(g2,2.5):+.3f},{np.percentile(g2,97.5):+.3f}]")
        for lab, gap, bb in [("canonical - competitor", mc - mk, g1),
                             ("competitor - intronic decoy", mk - md_, g2)]:
            fh.write(f"{nm}\t{len(cols)}\t{lab}\t{gap:.5f}\t{np.percentile(bb,2.5):.5f}\t"
                     f"{np.percentile(bb,97.5):.5f}\t{mc:.5f}\t{mk:.5f}\t{md_:.5f}\t"
                     f"{dt['auroc']}\t{dt['p1']}\t{dt['p1_chance']}\n")
log("DONE_DECOY_TRANSFER")
