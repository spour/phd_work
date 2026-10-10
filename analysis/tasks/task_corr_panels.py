#!/usr/bin/env python3
# panel data for the corrected-negative-set figure, on the winning feature set
import csv, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, HOLD_FRAC, RIDGE, NBOOT, NPERM = 42, 0.20, 1.0, 400, 2000
rng = np.random.default_rng(SEED)
K3 = ["spl_upstream_max_me3", "spl_pas_prox_u1", "spl_u1_suppression_up"]

D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
rf = D.rf_prob.to_numpy(float)
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
Z = np.load("corr_rowmask.npz"); ROW = Z["ROW"]; APA = Z["APA"]; TERMEX = Z["TERMEX"]
B = np.load("binned_features.npz"); BN = list(B["names"]); BX = B["X"][ridx].astype(float)
SPL = [i for i, n in enumerate(BN) if not n.endswith("_gc")]
GCB = [i for i, n in enumerate(BN) if n.endswith("_gc")]
X3 = np.column_stack([D[n].to_numpy(float) for n in K3])
XW = np.column_stack([X3, BX[:, SPL], BX[:, GCB]])
GC = np.load("termexon_gc_cache.npz")["gc500"][ridx]
PDB = np.load("polyadb_peak_cache.npz"); ATL = np.load("atlas_peak_cache.npz")
pdb_rpm = PDB["rpm"][ridx]; pdb_pse = PDB["pse"][ridx]; pdb_d = PDB["dist"][ridx]

gi = defaultdict(list)
for i in range(len(gene)): gi[gene[i]].append(i)
gi = {g: np.array(v) for g, v in gi.items()}
allg = np.array(sorted(set(gene)))
hold_g = set(rng.choice(allg, size=int(round(HOLD_FRAC * len(allg))), replace=False))
is_hold = np.array([g in hold_g for g in gene])
CHROMS = sorted(set(chrom))
ALL = np.ones(len(y), bool)

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
    A = (np.where(np.isfinite(Xtr), Xtr, mu) - mu) / sd
    Bm = (np.where(np.isfinite(Xte), Xte, mu) - mu) / sd
    return np.column_stack([np.ones(len(A)), A]), np.column_stack([np.ones(len(Bm)), Bm])
def loco_pred(X, train_mask, test_mask):
    pred = np.full(len(y), np.nan)
    for c in CHROMS:
        te = ROW & test_mask & (chrom == c); tr = ROW & train_mask & (chrom != c)
        if te.sum() == 0 or tr.sum() == 0: continue
        A, Bm = prep(X[tr], X[te]); pred[te] = Bm @ fit_ridge(A, y[tr])
    return pred
def per_gene(pred, mask):
    hits, rrs, ns, gl = [], [], [], []
    for g, ix in gi.items():
        if not mask[ix[0]]: continue
        ix = ix[ROW[ix]]
        if len(ix) < 2 or (y[ix] == 1).sum() != 1: continue
        s = pred[ix]
        if not np.isfinite(s).all(): continue
        n = len(ix); j = np.argsort(-(s + rng.random(n) * 1e-12), kind="mergesort")
        srt = s[j]; rk = np.empty(n); i = 0
        while i < n:
            k = i
            while k + 1 < n and srt[k + 1] == srt[i]: k += 1
            rk[i:k + 1] = (i + k) / 2.0 + 1.0; i = k + 1
        r = np.empty(n); r[j] = rk
        c = int(np.argmax(y[ix]))
        hits.append(1.0 if j[0] == c else 0.0); rrs.append(1.0 / r[c]); ns.append(n); gl.append((g, ix))
    return np.array(hits), np.array(rrs), np.array(ns, float), gl
def auroc(yv, s):
    o = np.argsort(s, kind="mergesort"); sv = s[o]; n = len(s); rk = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    rr = np.empty(n); rr[o] = rk; npos = yv.sum(); nneg = n - npos
    return float((rr[yv == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def auprc(yv, s):
    o = np.argsort(-s, kind="mergesort"); yy = yv[o]; sv = s[o]
    tp = np.cumsum(yy); k = np.arange(1, len(yy) + 1)
    idx = np.where(np.r_[sv[1:] != sv[:-1], True])[0]
    prec = (tp[idx] / k[idx])[np.searchsorted(idx, np.arange(len(yy)))]
    return float((prec * yy).sum() / yy.sum())
def bci(v):
    b = v[rng.integers(0, len(v), size=(NBOOT, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
def winrate(score):
    w = []
    for g, ix in gi.items():
        ix = ix[ROW[ix]]
        if len(ix) < 2 or (y[ix] == 1).sum() != 1: continue
        s = score[ix]
        if not np.isfinite(s).all(): continue
        c = int(np.argmax(y[ix])); o = np.delete(np.arange(len(ix)), c)
        w.append(float(np.mean(s[c] > s[o])))
    return np.array(w)

# predictions
pw_all = loco_pred(XW, ALL, ALL); pw_hold = loco_pred(XW, ~is_hold, is_hold)
p3_all = loco_pred(X3, ALL, ALL)
pg_all = loco_pred(BX[:, GCB], ALL, ALL)
log("predictions done")

# panel a: shortlist composition
isc = y == 1; comp = y == 0
with open("corr_panel_a.tsv", "w") as fh:
    fh.write("class\tcategory\tn\tfrac\n")
    for cls, m in [("canonical", isc), ("competitor", comp)]:
        tot = int(m.sum())
        for lab, v in [("terminal exon only", m & TERMEX & ~APA), ("APA-annotated only", m & APA & ~TERMEX),
                       ("both", m & APA & TERMEX), ("neither", m & ~APA & ~TERMEX)]:
            fh.write(f"{cls}\t{lab}\t{int(v.sum())}\t{v.sum()/tot:.5f}\n")
    fh.write(f"#removed_apa\t{int((comp & APA).sum())}\n#removed_term\t{int((comp & TERMEX).sum())}\n")
    fh.write(f"#removed_both\t{int((comp & APA & TERMEX).sum())}\n#retained\t{int((ROW & comp).sum())}\n")
    fh.write(f"#genes_dropped\t1660\n")

# panel b: usage of removed vs retained
GRP = [("canonical", isc), ("removed competitor (APA)", comp & APA),
       ("retained competitor", ROW & comp)]
def med_ci(v):
    v = v[np.isfinite(v)]
    b = np.median(v[rng.integers(0, len(v), size=(500, len(v)))], axis=1)
    return float(np.median(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
with open("corr_panel_b.tsv", "w") as fh:
    fh.write("group\tmeasure\tn\tmedian\tci_lo\tci_hi\tq25\tq75\tfrac_site_within_500\n")
    print("\n===== panel b: PolyA_DB usage =====")
    for nm, m in GRP:
        near = float(np.mean(pdb_d[m] <= 500))
        for meas, v in [("RPM", pdb_rpm), ("PSE", pdb_pse)]:
            x = v[m]; md, lo, hi = med_ci(x)
            q1, q3 = np.percentile(x[np.isfinite(x)], [25, 75])
            print(f"  {nm:<26} {meas:<4} n {int(m.sum()):>6} median {md:8.3f} [{lo:.3f},{hi:.3f}] "
                  f"IQR [{q1:.3f},{q3:.3f}] site within 500 nt {near:.3f}")
            fh.write(f"{nm}\t{meas}\t{int(m.sum())}\t{md:.5f}\t{lo:.5f}\t{hi:.5f}\t{q1:.5f}\t{q3:.5f}\t{near:.5f}\n")
SUB = rng.choice(np.where(isc | (comp & APA) | (ROW & comp))[0], size=12000, replace=False)
with open("corr_panel_b_points.tsv", "w") as fh:
    fh.write("group\tmeasure\tvalue\n")
    for i in SUB:
        g = "canonical" if isc[i] else ("removed competitor (APA)" if APA[i] else
                                        ("retained competitor" if ROW[i] else None))
        if g is None: continue
        fh.write(f"{g}\tRPM\t{pdb_rpm[i]:.5f}\n{g}\tPSE\t{pdb_pse[i]:.5f}\n")

# panels c, d: curves
rows_all = np.where(ROW)[0]
def curve(sv, tag, fh):
    o = rows_all[np.argsort(-sv[rows_all], kind="mergesort")]; yy = y[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy); P = yy.sum(); N = len(yy) - P
    for i in np.unique(np.linspace(0, len(yy) - 1, 400).astype(int)):
        fh.write(f"{tag}\t{fp[i]/N:.6f}\t{tp[i]/P:.6f}\t{tp[i]/(i+1):.6f}\n")
with open("corr_curves.tsv", "w") as fh:
    fh.write("model\tfpr\ttpr\tprecision\n")
    curve(pw_all, "reranker", fh); curve(p3_all, "three features", fh); curve(rf, "baseline model", fh)

hits, rrs, ns, gl = per_gene(pw_all, ALL)
gch = np.array([chrom[ix[0]] for _, ix in gl])
with open("corr_bychrom.tsv", "w") as fh:
    fh.write("chrom\tn_genes\tp1\tci_lo\tci_hi\tchance\tlift\n")
    for c in CHROMS:
        m = gch == c
        if m.sum() < 20: continue
        v, lo, hi = bci(hits[m]); ch = float(np.mean(1.0 / ns[m]))
        fh.write(f"{c}\t{int(m.sum())}\t{v:.5f}\t{lo:.5f}\t{hi:.5f}\t{ch:.5f}\t{v-ch:.5f}\n")

def coefs(X, names, idx):
    dev = np.where(ROW & ~is_hold)[0]
    A, _ = prep(X[dev], X[dev]); beta = fit_ridge(A, y[dev])[1:]
    devg = [ix[ROW[ix]] for g, ix in gi.items() if not is_hold[ix[0]] and ROW[ix].sum() >= 2]
    BB = np.empty((200, len(idx)))
    for t in range(200):
        ii = np.concatenate([devg[j] for j in rng.integers(0, len(devg), len(devg))])
        Ab, _ = prep(X[ii], X[ii]); BB[t] = fit_ridge(Ab, y[ii])[1:][idx]
    out = []
    for a, j in enumerate(idx):
        lo, hi = np.percentile(BB[:, a], [2.5, 97.5])
        out.append((names[a], float(beta[j]), float(lo), float(hi),
                    float(np.mean(np.sign(BB[:, a]) != np.sign(beta[j])))))
    return out
with open("corr_coefs.tsv", "w") as fh:
    fh.write("fit\tfeature\tbeta_std\tci_lo\tci_hi\tsign_flip_frac\n")
    print("\n===== panel f: coefficients of the three named features =====")
    for fitnm, X, idx in [("three features", X3, [0, 1, 2]), ("full model", XW, [0, 1, 2])]:
        for nm, b, lo, hi, fl in coefs(X, K3, idx):
            print(f"  {fitnm:<16} {nm:<24} {b:>+8.4f} [{lo:>+8.4f},{hi:>+8.4f}] flip {fl:.3f}")
            fh.write(f"{fitnm}\t{nm}\t{b:.5f}\t{lo:.5f}\t{hi:.5f}\t{fl:.4f}\n")

_, _, _, gl3 = per_gene(p3_all, ALL)
rows3 = np.concatenate([ix for _, ix in gl3])
def pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b); a = a[m] - a[m].mean(); b = b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan
def rankv(v):
    o = np.argsort(v, kind="mergesort"); sv = v[o]; n = len(v); rk = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    r = np.empty(n); r[o] = rk; return r
def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b); return pearson(rankv(a[m]), rankv(b[m]))
with open("corr_gc_points.tsv", "w") as fh:
    fh.write("gene\tclass\tgc\tdiscriminant\n")
    for i in rows3:
        fh.write(f"{gene[i]}\t{'canonical' if y[i]==1 else 'competitor'}\t{GC[i]:.5f}\t{p3_all[i]:.5f}\n")

h3, _, n3, _ = per_gene(p3_all, ALL)
hw, _, nw, _ = per_gene(pw_all, ALL)
hg, _, ng_, _ = per_gene(pg_all, ALL)
gc_all = np.where(np.isfinite(GC), GC, np.nan)
hgc, _, ngc, _ = per_gene(gc_all, ALL)
mfin = np.isfinite(p3_all[rows3]) & np.isfinite(GC[rows3])
A1 = np.column_stack([np.ones(mfin.sum()), GC[rows3][mfin]])
bb = np.linalg.lstsq(A1, p3_all[rows3][mfin], rcond=None)[0]
res = np.full(len(y), np.nan); res[rows3[mfin]] = p3_all[rows3][mfin] - A1 @ bb
hr, _, nr, glr = per_gene(res, ALL)
rowsr = np.concatenate([ix for _, ix in glr])
ROWSW = np.concatenate([ix for _, ix in gl])
rowsg = np.concatenate([ix for _, ix in per_gene(pg_all, ALL)[3]])
ENT = [("GC, single 500 nt window", hgc, ngc, winrate(gc_all), auroc(y[rows3], GC[rows3])),
       ("GC, 25 binned profile", hg, ng_, winrate(pg_all), auroc(y[rowsg], pg_all[rowsg])),
       ("three features", h3, n3, winrate(p3_all), auroc(y[rows3], p3_all[rows3])),
       ("three features, GC removed", hr, nr, winrate(res), auroc(y[rowsr], res[rowsr])),
       ("full model", hw, nw, winrate(pw_all), auroc(y[ROWSW], pw_all[ROWSW]))]
with open("corr_panel_h.tsv", "w") as fh:
    fh.write("predictor\tp1\tci_lo\tci_hi\tchance\twinrate\twinrate_lo\twinrate_hi\tauroc\n")
    print("\n===== panel h: composition as a competing explanation =====")
    for nm, h, n_, w, au in ENT:
        p, lo, hi = bci(h); wm, wlo, whi = bci(w)
        print(f"  {nm:<28} P@1 {p:.4f} [{lo:.4f},{hi:.4f}] chance {np.mean(1.0/n_):.4f}  "
              f"win-rate {wm:.4f} [{wlo:.4f},{whi:.4f}]  AUROC {au:.4f}")
        fh.write(f"{nm}\t{p:.5f}\t{lo:.5f}\t{hi:.5f}\t{np.mean(1.0/n_):.5f}\t{wm:.5f}\t{wlo:.5f}\t{whi:.5f}\t{au:.5f}\n")
with open("corr_gc_stats.tsv", "w") as fh:
    fh.write("quantity\tpearson\tspearman\n")
    yv3 = y[rows3]
    for lab, sub in [("all peaks", np.ones(len(rows3), bool)), ("canonical", yv3 == 1), ("competitor", yv3 == 0)]:
        fh.write(f"discriminant vs GC, {lab}\t{pearson(p3_all[rows3][sub], GC[rows3][sub]):.5f}\t"
                 f"{spearman(p3_all[rows3][sub], GC[rows3][sub]):.5f}\n")

def report(tag, pred, mask):
    h, rr, n_, g_ = per_gene(pred, mask)
    rws = np.concatenate([ix for _, ix in g_])
    Hn = np.array([np.sum(1.0 / np.arange(1, int(v) + 1)) for v in n_])
    p, plo, phi = bci(h); m_, mlo, mhi = bci(rr)
    e, elo, ehi = bci(h - 1.0 / n_)
    print(f"{tag:<18} genes {len(g_):>5} P@1 {p:.4f}[{plo:.4f},{phi:.4f}] chance {np.mean(1.0/n_):.4f} "
          f"excess {e:+.4f}[{elo:+.4f},{ehi:+.4f}] MRR {m_:.4f} chance {np.mean(Hn/n_):.4f} "
          f"AUROC {auroc(y[rws], pred[rws]):.4f} AUPRC {auprc(y[rws], pred[rws]):.4f} base {y[rws].mean():.4f}")
    return dict(model=tag, n_genes=len(g_), p1=p, p1_lo=plo, p1_hi=phi, p1_chance=float(np.mean(1.0/n_)),
                p1_exc=e, p1_exc_lo=elo, p1_exc_hi=ehi, mrr=m_, mrr_lo=mlo, mrr_hi=mhi,
                mrr_chance=float(np.mean(Hn/n_)), auroc=auroc(y[rws], pred[rws]),
                auprc=auprc(y[rws], pred[rws]), base_rate=float(y[rws].mean()))
print("\n===== headline metrics, corrected set =====")
RES = [report("full_hold", pw_hold, is_hold), report("full_loco_all", pw_all, ALL),
       report("k3_loco_all", p3_all, ALL), report("baseline_rf", rf, ALL)]
ks = [k for k in RES[0] if k != "model"]
with open("corr_metrics.tsv", "w") as fh:
    fh.write("model\t" + "\t".join(ks) + "\n")
    for r in RES:
        fh.write(r["model"] + "\t" + "\t".join(
            f"{r[k]:.5f}" if isinstance(r[k], float) else str(r[k]) for k in ks) + "\n")

out = pd.DataFrame(dict(
    gene=gene, chrom=chrom, center=D.center.to_numpy(), strand=D.strand.to_numpy(),
    is_canonical=(y == 1), apa_annotated=APA, in_reference_terminal_exon=TERMEX,
    in_analysis_set=ROW, gc500=GC,
    discriminant_full=pw_all, discriminant_k3=p3_all,
    polyadb_dist=pdb_d, polyadb_rpm=pdb_rpm, polyadb_pse=pdb_pse,
    atlas_dist=ATL["dist"][ridx], atlas_frac=ATL["frac"][ridx]))
for nm in K3: out[nm] = D[nm].to_numpy(float)
for j in GCB: out[str(BN[j])] = BX[:, j]
out.to_parquet("rerank_corr_peaks.parquet", engine="fastparquet", compression="snappy")
print(f"\nwrote rerank_corr_peaks.parquet: {len(out)} peaks, {int(out.in_analysis_set.sum())} in the analysis set, "
      f"{out.shape[1]} columns")
log("DONE_CORR_PANELS")
