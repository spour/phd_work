#!/usr/bin/env python3
# the loco comparison of the sequence cnn and the 26-feature splice model
import sys, time
import numpy as np
import binned_common as K
from binned_common import log
from cnn_interp_lib import dinuc_shuffle
sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as L
from maxentpy import maxent
from maxentpy.maxent import load_matrix3

SEED, KSH, KMAX, NB = 42, 3, 10, 2000
UP, DN, BIN, NBIN = 2000, 500, 100, 25
ME3_LO, ME3_HI = -2000, -126
M3 = load_matrix3(); NT = np.array(list("ACGT") + ["N"])
o9 = np.arange(UP + DN - 9 + 1) + 3 - UP
bidx = (o9 + UP) // BIN
nb9 = np.bincount(bidx, minlength=NBIN).astype(float)

y = K.y; gene = K.gene; chrom = K.C0["chrom"][K.ok].astype(str); N = len(y)
assert N == 48394 and int(y.sum()) == 4847
GI = {g: np.flatnonzero(gene == g) for g in sorted(set(gene))}
PAIR = {g: (ix[y[ix] == 1], ix[y[ix] == 0]) for g, ix in GI.items()}
PAIR = {g: v for g, v in PAIR.items() if len(v[0]) and len(v[1])}
GL = np.array(sorted(PAIR)); NG = len(GL)
assert NG == 4847
log(f"[scope] {NG} analysis genes, {N} peaks, {len(set(chrom))} chromosomes, "
    f"base rate {y.mean():.5f}")

# features
X = np.load("cnn_data.npz", allow_pickle=True)["X"]
Z = np.load("binned_profiles.npz", allow_pickle=True)
CH = [str(c) for c in Z["channels"]]
D4 = Z["P"][:, CH.index("u1_dens4"), :].astype(float)
XR = np.column_stack([D4, K.ACC]); assert XR.shape[1] == 26

def featurise(tok2d, tag):
    out = np.empty((len(tok2d), 26)); t0 = time.time()
    for i in range(len(tok2d)):
        t = tok2d[i].astype(np.int64)
        sw = np.lib.stride_tricks.sliding_window_view(t, 9); g = (sw < 4).all(1)
        u = np.nan_to_num(np.where(g, L.U1[np.where(g, sw @ L.POW, 0)], np.nan), nan=-99.0)
        s = np.bincount(bidx, weights=(u >= 4.0).astype(float), minlength=NBIN)
        out[i, :NBIN] = np.where(nb9 > 0, s / np.maximum(nb9, 1) * 1000.0, np.nan)
        seq = "".join(NT[np.where(t < 4, t, 4)])
        lo = max(0, ME3_LO + UP - 19); hi = min(len(seq) - 23 + 1, ME3_HI + UP - 19)
        best = -99.0
        for j in range(lo, hi):
            if t[j + 18] != 0 or t[j + 19] != 2: continue
            k = seq[j:j + 23]
            if "N" in k: continue
            try: v = maxent.score3(k, matrix=M3)
            except Exception: continue
            if v > best: best = v
        out[i, -1] = best
        if (i + 1) % 20000 == 0: log(f"  {tag} {i+1}/{len(tok2d)}  {time.time()-t0:.0f}s")
    log(f"[feat] {tag}: {len(tok2d)} peaks in {time.time()-t0:.0f}s")
    return out

RD = featurise(X, "real")
r_acc = np.corrcoef(RD[:, -1], K.ACC)[0, 1]
r_u1 = np.corrcoef(RD[:, 0], D4[:, 0])[0, 1]
log(f"[check] re-derived acceptor vs the pipeline column r = {r_acc:.5f}; "
    f"re-derived u1_dens4[bin 0] vs binned_profiles r = {r_u1:.5f}")
assert r_acc > 0.99 and r_u1 > 0.999

rng = np.random.default_rng(SEED)
SHT = [np.stack([dinuc_shuffle(s, rng) for s in X]) for _ in range(KSH)]
gcf = lambda a: float(((a == 1) | (a == 2)).mean())
cpgf = lambda a: float(np.mean((a[:, :-1] == 1) & (a[:, 1:] == 2)))
log(f"[check] GC  real {gcf(X):.6f}  shuffled {np.mean([gcf(s) for s in SHT]):.6f}")
log(f"[check] CpG real {cpgf(X):.6f}  shuffled {np.mean([cpgf(s) for s in SHT]):.6f}")
assert abs(gcf(X) - np.mean([gcf(s) for s in SHT])) < 1e-9
assert abs(cpgf(X) - np.mean([cpgf(s) for s in SHT])) < 1e-9
SHD = [featurise(s, f"shuffle {j+1}/{KSH}") for j, s in enumerate(SHT)]

# splice loco
CgR = K.centre_by_gene(XR)
CgS = [K.centre_by_gene(m) for m in SHD]

def groups(ii):
    by = {}
    for k_, g in enumerate(gene[ii]):
        r, c = by.setdefault(g, ([], []))
        (r if y[ii][k_] == 1 else c).append(k_)
    return [(np.array(r), np.array(c)) for r, c in by.values() if r and c]

def zsc(Cg, m, s):
    return (np.where(np.isfinite(Cg), Cg, m) - m) / s

SPL = np.full(N, np.nan); SPLS = np.full((KSH, N), np.nan); lams = []
for ch in sorted(set(chrom)):
    te = np.flatnonzero(chrom == ch); tr = np.flatnonzero(chrom != ch)
    trg = np.array(sorted(set(gene[tr]))); rs = np.random.default_rng(K.SEED)
    sh = trg.copy(); rs.shuffle(sh); ivg = set(sh[:int(round(0.25 * len(sh)))])
    iv = tr[np.array([g in ivg for g in gene[tr]])]
    it = tr[np.array([g not in ivg for g in gene[tr]])]
    gi = groups(iv); best = (-1.0, 1.0)
    for lam in K.LAMBDAS:
        m = np.nanmean(CgR[it], 0); m = np.where(np.isfinite(m), m, 0.0)
        s = np.nanstd(CgR[it], 0); s[~np.isfinite(s) | (s == 0)] = 1.0
        Zi = zsc(CgR, m, s)
        b = K.fit_ridge(np.column_stack([np.ones(len(it)), Zi[it]]), y[it], lam)
        w = K.winrate(np.column_stack([np.ones(len(iv)), Zi[iv]]) @ b, gi)
        if w > best[0]: best = (w, lam)
    m = np.nanmean(CgR[tr], 0); m = np.where(np.isfinite(m), m, 0.0)
    s = np.nanstd(CgR[tr], 0); s[~np.isfinite(s) | (s == 0)] = 1.0
    ZR = zsc(CgR, m, s)
    b = K.fit_ridge(np.column_stack([np.ones(len(tr)), ZR[tr]]), y[tr], best[1])
    SPL[te] = np.column_stack([np.ones(len(te)), ZR[te]]) @ b
    for j in range(KSH):
        Zs = zsc(CgS[j], m, s)
        SPLS[j, te] = np.column_stack([np.ones(len(te)), Zs[te]]) @ b
    lams.append(best[1])
assert np.isfinite(SPL).all() and np.isfinite(SPLS).all()
log(f"[loco] splice model: 23 folds, lambda {min(lams):g} to {max(lams):g}")

CN = np.load("cnn_loco_scores.npz", allow_pickle=True)
CNN = CN["logit"].astype(float); CNNS = CN["logit_shuffled"].astype(float)
assert (CN["gene"] == gene).all() and (CN["chrom"].astype(str) == chrom).all()
assert np.isfinite(CNN).all(), "cnn_loco_scores.npz is incomplete"
log(f"[loco] CNN scores loaded, {CNNS.shape[0]} shuffle replicates, row order verified")

# metrics
GRP = [PAIR[g] for g in GL]
def per_gene_wr(sc):
    return np.array([((sc[r][:, None] > sc[c][None, :]).sum()
                      + 0.5 * (sc[r][:, None] == sc[c][None, :]).sum()) / (len(r) * len(c))
                     for r, c in GRP])
BS = np.random.default_rng(SEED).integers(0, NG, size=(NB, NG))
ci = lambda v: (float(np.percentile(v[BS].mean(1), 2.5)), float(np.percentile(v[BS].mean(1), 97.5)))

MOD = {"Sequence CNN": (CNN, CNNS), "Splice model (26 features)": (SPL, SPLS)}
SUM = {}
for nm, (sc, shs) in MOD.items():
    wr = per_gene_wr(sc)
    rk = np.array([1 + int((sc[c] >= sc[r[0]]).sum()) for r, c in GRP])
    p1 = (rk == 1).astype(float)
    wrs = np.mean([per_gene_wr(shs[j]) for j in range(shs.shape[0])], 0)
    d = wr - wrs
    SUM[nm] = dict(wr=wr, p1=p1, rk=rk, wrs=wrs, d=d,
                   auroc=K.auroc(sc[y == 1], sc[y == 0]), auprc=K.auprc(sc[y == 1], sc[y == 0]))
    lo, hi = ci(wr); lo1, hi1 = ci(p1); los, his = ci(wrs); lod, hid = ci(d)
    pct = 100 * d.mean() / (wr.mean() - 0.5)
    SUM[nm].update(wr_lo=lo, wr_hi=hi, p1_lo=lo1, p1_hi=hi1, wrs_lo=los, wrs_hi=his,
                   d_lo=lod, d_hi=hid, pct=pct)
    log(f"[{nm}] LOCO within-gene {wr.mean():.5f} [{lo:.5f},{hi:.5f}]  strict P@1 {p1.mean():.5f} "
        f"[{lo1:.5f},{hi1:.5f}]  pooled AUROC {SUM[nm]['auroc']:.5f}  AUPRC {SUM[nm]['auprc']:.5f}")
    log(f"[{nm}] shuffled {wrs.mean():.5f} [{los:.5f},{his:.5f}]  drop {d.mean():+.5f} "
        f"[{lod:+.5f},{hid:+.5f}]  {pct:.1f}% of the lift over 0.5 lost")

dCW = SUM["Sequence CNN"]["wr"] - SUM["Splice model (26 features)"]["wr"]
log(f"[paired] CNN minus splice, within-gene {dCW.mean():+.5f} "
    f"[{ci(dCW)[0]:+.5f},{ci(dCW)[1]:+.5f}] on the same {NG} genes")

with open("loco_shuffle.tsv", "w") as fh:
    fh.write("model\treal\treal_lo\treal_hi\tshuffled\tshuffled_lo\tshuffled_hi\tdrop\tdrop_lo\t"
             "drop_hi\tpct_of_lift_lost\tk_shuffles\tscope\tn_genes\tn_peaks\n")
    for nm, s in SUM.items():
        fh.write(f"{nm}\t{s['wr'].mean():.5f}\t{s['wr_lo']:.5f}\t{s['wr_hi']:.5f}\t"
                 f"{s['wrs'].mean():.5f}\t{s['wrs_lo']:.5f}\t{s['wrs_hi']:.5f}\t"
                 f"{s['d'].mean():.5f}\t{s['d_lo']:.5f}\t{s['d_hi']:.5f}\t{s['pct']:.3f}\t"
                 f"{KSH}\tLOCO, analysis set\t{NG}\t{N}\n")
log("[out] loco_shuffle.tsv")

# a: roc and pr curves
def roc_pr(sc):
    o = np.argsort(-sc, kind="stable"); yy = y[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    P, Ng_ = yy.sum(), (1 - yy).sum()
    end = np.r_[np.flatnonzero(sc[o][1:] != sc[o][:-1]), len(yy) - 1]
    return (fp[end] / Ng_, tp[end] / P, tp[end] / (end + 1))
with open("figLC_curves.tsv", "w") as fh:
    fh.write("model\tcurve\tx\ty\tauc\tbase_rate\tn_peaks\n")
    for nm, (sc, _) in MOD.items():
        fpr, tpr, prec = roc_pr(sc)
        step = max(1, len(fpr) // 1500)
        for i in range(0, len(fpr), step):
            fh.write(f"{nm}\tROC\t{fpr[i]:.6f}\t{tpr[i]:.6f}\t{SUM[nm]['auroc']:.5f}\t"
                     f"{y.mean():.5f}\t{N}\n")
        for i in range(0, len(tpr), step):
            fh.write(f"{nm}\tPR\t{tpr[i]:.6f}\t{prec[i]:.6f}\t{SUM[nm]['auprc']:.5f}\t"
                     f"{y.mean():.5f}\t{N}\n")
log("[out] figLC_curves.tsv")

nn = np.array([len(r) + len(c) for r, c in GRP], float)
with open("figLC_recall.tsv", "w") as fh:
    fh.write("model\tk\trecall\tci_lo\tci_hi\tchance\tlift\tscope\tn_genes\n")
    for nm in MOD:
        rk = SUM[nm]["rk"]
        for k in range(1, KMAX + 1):
            h = (rk <= k).astype(float); c = (np.minimum(k, nn) / nn)
            lo, hi = ci(h)
            fh.write(f"{nm}\t{k}\t{h.mean():.5f}\t{lo:.5f}\t{hi:.5f}\t{c.mean():.5f}\t"
                     f"{h.mean()-c.mean():+.5f}\tLOCO, analysis set\t{NG}\n")
            if k in (1, 3):
                log(f"[recall] {nm}  k={k}  {h.mean():.5f} [{lo:.5f},{hi:.5f}]  "
                    f"chance {c.mean():.5f}")
log("[out] figLC_recall.tsv")

with open("figLC_summary.tsv", "w") as fh:
    fh.write("model\twithin_gene\twg_lo\twg_hi\tp_at_1\tp1_lo\tp1_hi\tauroc\tauprc\tbase_rate\t"
             "n_genes\tn_peaks\tscope\n")
    for nm, s in SUM.items():
        fh.write(f"{nm}\t{s['wr'].mean():.5f}\t{s['wr_lo']:.5f}\t{s['wr_hi']:.5f}\t"
                 f"{s['p1'].mean():.5f}\t{s['p1_lo']:.5f}\t{s['p1_hi']:.5f}\t{s['auroc']:.5f}\t"
                 f"{s['auprc']:.5f}\t{y.mean():.5f}\t{NG}\t{N}\tLOCO, analysis set\n")
log("[out] figLC_summary.tsv")
