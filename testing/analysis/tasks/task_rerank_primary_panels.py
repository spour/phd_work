#!/usr/bin/env python3
# panel data for the primary reranker: 3 features, unannotated competitors only
import csv, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, HOLD_FRAC, RIDGE, NBOOT, NPERM = 42, 0.20, 1.0, 400, 200
FINAL = ["spl_upstream_max_me3", "spl_pas_prox_u1", "spl_u1_suppression_up"]
rng = np.random.default_rng(SEED)

D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
rf = D.rf_prob.to_numpy(float)
X = np.column_stack([D[n].to_numpy(float) for n in FINAL])
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
At = np.load("atlas_peak_cache.npz")["dist"][ridx]
Pd = np.load("polyadb_peak_cache.npz")["dist"][ridx]
UNANNOT = (At > 500) & (Pd > 500)
fc_all = np.load("fine_compartment.npy")
Lb = list(csv.DictReader(open("cpa_peaks_labeled.tsv"), delimiter="\t"))
fmap = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): int(fc_all[i]) for i, r in enumerate(Lb)}
fine = np.array([fmap[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
TERM = fine == 2; INTRON = np.isin(fine, [3, 4, 5])

gi = defaultdict(list)
for i in range(len(gene)): gi[gene[i]].append(i)
gi = {g: np.array(v) for g, v in gi.items()}
def gene_filter(mask):
    keep = np.zeros(len(y), bool)
    for g, ix in gi.items():
        sub = ix[mask[ix]]
        if (y[sub] == 1).sum() == 1 and (y[sub] == 0).sum() >= 1: keep[sub] = True
    return keep
ROW = gene_filter((y == 1) | ((y == 0) & UNANNOT))
allg = np.array(sorted(set(gene)))
hold_g = set(rng.choice(allg, size=int(round(HOLD_FRAC * len(allg))), replace=False))
is_hold = np.array([g in hold_g for g in gene])
CHROMS = sorted(set(chrom))
log(f"primary rows {int(ROW.sum())}, competitors {int((ROW&(y==0)).sum())}, "
    f"genes {len({gene[i] for i in np.where(ROW&(y==0))[0]})}")

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
def loco_pred(rowmask, train_mask, test_mask, yv=None):
    yy = y if yv is None else yv
    pred = np.full(len(yy), np.nan)
    for c in CHROMS:
        te = rowmask & test_mask & (chrom == c); tr = rowmask & train_mask & (chrom != c)
        if te.sum() == 0 or tr.sum() == 0: continue
        A, B = prep(X[tr], X[te]); pred[te] = B @ fit_ridge(A, yy[tr])
    return pred
def per_gene(pred, rowmask, mask, yv=None):
    yy = y if yv is None else yv
    hits, rrs, ns, gl = [], [], [], []
    for g, ix in gi.items():
        if not mask[ix[0]]: continue
        ix = ix[rowmask[ix]]
        if len(ix) < 2 or (yy[ix] == 1).sum() != 1: continue
        s = pred[ix]
        if not np.isfinite(s).all(): continue
        n = len(ix); j = np.argsort(-(s + rng.random(n) * 1e-12), kind="mergesort")
        srt = s[j]; rk = np.empty(n); i = 0
        while i < n:
            k = i
            while k + 1 < n and srt[k + 1] == srt[i]: k += 1
            rk[i:k + 1] = (i + k) / 2.0 + 1.0; i = k + 1
        r = np.empty(n); r[j] = rk
        c = int(np.argmax(yy[ix]))
        hits.append(1.0 if j[0] == c else 0.0); rrs.append(1.0 / r[c]); ns.append(n); gl.append((g, ix))
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
def bci(v):
    b = v[rng.integers(0, len(v), size=(NBOOT, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
def metrics(tag, pred, rowmask, mask):
    hits, rrs, ns, gl = per_gene(pred, rowmask, mask)
    rows = np.concatenate([ix for _, ix in gl]); yv = y[rows]; sv = pred[rows]
    Hn = np.array([np.sum(1.0 / np.arange(1, int(n) + 1)) for n in ns])
    p1, p1lo, p1hi = bci(hits); mr, mrlo, mrhi = bci(rrs)
    e1, e1lo, e1hi = bci(hits - 1.0 / ns); em, emlo, emhi = bci(rrs - Hn / ns)
    gidx = [ix for _, ix in gl]; ab = np.empty(200); pb = np.empty(200)
    for t in range(200):
        ii = np.concatenate([gidx[j] for j in rng.integers(0, len(gidx), len(gidx))])
        ab[t] = auroc(y[ii], pred[ii]); pb[t] = auprc(y[ii], pred[ii])
    return dict(model=tag, n_genes=len(gl), n_rows=len(rows), p1=p1, p1lo=p1lo, p1hi=p1hi,
                p1_chance=float(np.mean(1.0 / ns)), p1_exc=e1, p1_exc_lo=e1lo, p1_exc_hi=e1hi,
                mrr=mr, mrr_lo=mrlo, mrr_hi=mrhi, mrr_chance=float(np.mean(Hn / ns)),
                mrr_exc=em, mrr_exc_lo=emlo, mrr_exc_hi=emhi,
                auroc=auroc(yv, sv), auroc_lo=float(np.percentile(ab, 2.5)), auroc_hi=float(np.percentile(ab, 97.5)),
                auprc=auprc(yv, sv), auprc_lo=float(np.percentile(pb, 2.5)), auprc_hi=float(np.percentile(pb, 97.5)),
                base_rate=float(yv.mean()))
def coefs(rowmask):
    dev = np.where(rowmask & ~is_hold)[0]
    A, _ = prep(X[dev], X[dev]); beta = fit_ridge(A, y[dev])[1:]
    devg = [ix[rowmask[ix]] for g, ix in gi.items() if not is_hold[ix[0]] and rowmask[ix].sum() >= 2]
    BB = np.empty((300, len(FINAL)))
    for t in range(300):
        ii = np.concatenate([devg[j] for j in rng.integers(0, len(devg), len(devg))])
        Ab, _ = prep(X[ii], X[ii]); BB[t] = fit_ridge(Ab, y[ii])[1:]
    out = []
    for a, nm in enumerate(FINAL):
        lo, hi = np.percentile(BB[:, a], [2.5, 97.5])
        out.append((nm, float(beta[a]), float(lo), float(hi),
                    float(np.mean(np.sign(BB[:, a]) != np.sign(beta[a])))))
    return out

ALL = np.ones(len(y), bool)
RES = [metrics("primary_hold", loco_pred(ROW, ~is_hold, is_hold), ROW, is_hold),
       metrics("primary_dev", loco_pred(ROW, ~is_hold, ~is_hold), ROW, ~is_hold),
       metrics("primary_loco_all", loco_pred(ROW, ALL, ALL), ROW, ALL)]
pred_all = loco_pred(ROW, ALL, ALL)
RES.append(metrics("rf_baseline", rf, ROW, ALL))
ks = ["n_genes","n_rows","p1","p1lo","p1hi","p1_chance","p1_exc","p1_exc_lo","p1_exc_hi","mrr","mrr_lo","mrr_hi",
      "mrr_chance","mrr_exc","mrr_exc_lo","mrr_exc_hi","auroc","auroc_lo","auroc_hi","auprc","auprc_lo","auprc_hi","base_rate"]
with open("rerank_pri_metrics.tsv", "w") as fh:
    fh.write("model\tn_feat\t" + "\t".join(ks) + "\n")
    for r in RES:
        fh.write(f"{r['model']}\t{len(FINAL)}\t" + "\t".join(f"{r[k]:.5f}" if isinstance(r[k], float) else str(r[k]) for k in ks) + "\n")
for r in RES:
    print(f"{r['model']:<18} genes {r['n_genes']:>5} P@1 {r['p1']:.4f}[{r['p1lo']:.4f},{r['p1hi']:.4f}] chance {r['p1_chance']:.4f} "
          f"exc {r['p1_exc']:+.4f} MRR {r['mrr']:.4f} chance {r['mrr_chance']:.4f} AUROC {r['auroc']:.4f} AUPRC {r['auprc']:.4f} base {r['base_rate']:.4f}")

with open("rerank_pri_coefs.tsv", "w") as fh:
    fh.write("feature\tbeta_std\tci_lo\tci_hi\tsign_flip_frac\n")
    for nm, b, lo, hi, fl in coefs(ROW): fh.write(f"{nm}\t{b:.5f}\t{lo:.5f}\t{hi:.5f}\t{fl:.4f}\n")

# term-term probe on this feature set
TT_PRI = gene_filter(ROW & TERM)
TT_ALL = gene_filter(TERM)
print("\n===== TERM-TERM probe, 3-feature model =====")
with open("rerank_pri_termterm.tsv", "w") as fh:
    fh.write("stratum\tfeature\tbeta_std\tci_lo\tci_hi\tsign_flip_frac\n")
    for nm, m in [("primary (unrestricted compartment)", ROW),
                  ("TERM-TERM, unannotated competitors", TT_PRI),
                  ("TERM-TERM, all competitors", TT_ALL)]:
        r = metrics(nm, loco_pred(m, ~is_hold, is_hold), m, is_hold)
        co = coefs(m)
        print(f"  {nm}: competitors {int((m&(y==0)).sum())}, genes {len({gene[i] for i in np.where(m&(y==0))[0]})}, "
              f"holdout genes {r['n_genes']}, P@1 {r['p1']:.4f}[{r['p1lo']:.4f},{r['p1hi']:.4f}] chance {r['p1_chance']:.4f} "
              f"excess {r['p1_exc']:+.4f}[{r['p1_exc_lo']:+.4f},{r['p1_exc_hi']:+.4f}]")
        for f, b, lo, hi, fl in co:
            print(f"      {f:<24} {b:>+8.4f} [{lo:>+8.4f},{hi:>+8.4f}] flip {fl:.3f}")
            fh.write(f"{nm}\t{f}\t{b:.5f}\t{lo:.5f}\t{hi:.5f}\t{fl:.4f}\n")
        fh.write(f"#{nm}\tP@1\t{r['p1']:.5f}\t{r['p1lo']:.5f}\t{r['p1hi']:.5f}\t{r['p1_chance']:.5f}\t{r['p1_exc']:.5f}\t{r['p1_exc_lo']:.5f}\t{r['p1_exc_hi']:.5f}\t{r['n_genes']}\n")

# curves, per-chromosome, contest, permutation, override, margin, misses
def curves(sv, tag, fh):
    rows = np.where(ROW)[0]; o = rows[np.argsort(-sv[rows], kind="mergesort")]; yy = y[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy); P = yy.sum(); N = len(yy) - P
    for i in np.unique(np.linspace(0, len(yy) - 1, 400).astype(int)):
        fh.write(f"{tag}\t{fp[i]/N:.6f}\t{tp[i]/P:.6f}\t{tp[i]/(i+1):.6f}\n")
with open("rerank_pri_curves.tsv", "w") as fh:
    fh.write("model\tfpr\ttpr\tprecision\n"); curves(pred_all, "reranker", fh); curves(rf, "RF baseline", fh)

hits, rrs, ns, gl = per_gene(pred_all, ROW, ALL)
gch = np.array([chrom[ix[0]] for _, ix in gl])
with open("rerank_pri_bychrom.tsv", "w") as fh:
    fh.write("chrom\tn_genes\tp1\tci_lo\tci_hi\tchance\tlift\n")
    for c in CHROMS:
        m = gch == c
        if m.sum() < 20: continue
        v, lo, hi = bci(hits[m]); ch = float(np.mean(1.0 / ns[m]))
        fh.write(f"{c}\t{int(m.sum())}\t{v:.5f}\t{lo:.5f}\t{hi:.5f}\t{ch:.5f}\t{v-ch:.5f}\n")
nc = ns - 1
with open("rerank_pri_bycontest.tsv", "w") as fh:
    fh.write("bucket\tn_genes\tp1\tci_lo\tci_hi\tchance\tlift\n")
    print("\n===== contest difficulty (primary set) =====")
    for nm, m in [("1", nc == 1), ("2", nc == 2), ("3-4", (nc >= 3) & (nc <= 4)),
                  ("5-8", (nc >= 5) & (nc <= 8)), ("9+", nc >= 9)]:
        v, lo, hi = bci(hits[m]); ch = float(np.mean(1.0 / ns[m]))
        print(f"  {nm:>4} n {int(m.sum()):>5} P@1 {v:.4f} chance {ch:.4f} lift {v-ch:+.4f}")
        fh.write(f"{nm}\t{int(m.sum())}\t{v:.5f}\t{lo:.5f}\t{hi:.5f}\t{ch:.5f}\t{v-ch:.5f}\n")

log(f"permutation null: {NPERM} LOCO refits")
obs = hits.mean(); nul = np.empty(NPERM)
for t in range(NPERM):
    yp = y.copy()
    for g, ix in gi.items():
        sub = ix[ROW[ix]]
        if len(sub) < 2: continue
        yp[sub] = 0.0; yp[sub[rng.integers(0, len(sub))]] = 1.0
    pp = loco_pred(ROW, ALL, ALL, yv=yp)
    h, _, _, _ = per_gene(pp, ROW, ALL, yv=yp); nul[t] = h.mean()
pv = (np.sum(nul >= obs) + 1) / (NPERM + 1)
with open("rerank_pri_permnull.tsv", "w") as fh:
    fh.write("perm\tp1\n")
    for t in range(NPERM): fh.write(f"{t+1}\t{nul[t]:.5f}\n")
    fh.write(f"#observed\t{obs:.5f}\n#p\t{pv:.5f}\n#chance\t{float(np.mean(1.0/ns)):.5f}\n")
print(f"\npermutation null: mean {nul.mean():.4f} [{np.percentile(nul,2.5):.4f},{np.percentile(nul,97.5):.4f}] "
      f"vs observed {obs:.4f}, p = {pv:.4f}")

ov = {"model correct, RF wrong": 0, "RF correct, model wrong": 0, "both correct": 0, "both wrong": 0}
GRP = {"correct pick": [], "marginal miss": [], "confident miss": []}
with open("rerank_pri_margin.tsv", "w") as fh:
    fh.write("gene\thit\tn_candidates\tmargin_frac\n")
    for g, ix in gl:
        s = pred_all[ix]; b = rf[ix]; c = int(np.argmax(y[ix]))
        mp = int(np.argmax(s + rng.random(len(ix)) * 1e-12)); bp = int(np.argmax(b + rng.random(len(ix)) * 1e-12))
        mh, bh = mp == c, bp == c
        ov["both correct" if (mh and bh) else "model correct, RF wrong" if mh else
           "RF correct, model wrong" if bh else "both wrong"] += 1
        span = s.max() - s.min(); frac = 0.0 if span <= 0 else float((s[mp] - s[c]) / span)
        fh.write(f"{g}\t{1 if mh else 0}\t{len(ix)}\t{frac:.5f}\n")
        GRP["correct pick" if mh else ("confident miss" if frac > 0.5 else "marginal miss")].append(ix[mp])
with open("rerank_pri_override.tsv", "w") as fh:
    fh.write("outcome\tn_genes\tfrac\n"); tot = sum(ov.values())
    print("\n===== override outcome (primary set) =====")
    for k2, v2 in ov.items():
        print(f"  {k2:<26} {v2:>5}  {v2/tot:.4f}"); fh.write(f"{k2}\t{v2}\t{v2/tot:.5f}\n")

CAT = lambda idx: np.array([float((TERM[idx]).mean()), float((INTRON[idx]).mean()),
                            float((~TERM[idx] & ~INTRON[idx]).mean())])
with open("rerank_pri_misses.tsv", "w") as fh:
    fh.write("group\tn\tcategory\tfrac\n")
    print("\n===== where the picks sit (3 categories) =====")
    for nm, idx in [("correct pick", np.array(GRP["correct pick"])), ("marginal miss", np.array(GRP["marginal miss"])),
                    ("confident miss", np.array(GRP["confident miss"])), ("competitor pool", np.where(ROW & (y == 0))[0])]:
        c3 = CAT(idx)
        print(f"  {nm:<16} n {len(idx):>6}  terminal exon {c3[0]:.3f}  intron {c3[1]:.3f}  other {c3[2]:.3f}")
        for lab, v in zip(["terminal exon", "intron", "other"], c3):
            fh.write(f"{nm}\t{len(idx)}\t{lab}\t{v:.5f}\n")
mg = np.array([float(l.split("\t")[3]) for l in open("rerank_pri_margin.tsv").readlines()[1:] if l.split("\t")[1] == "0"])
print(f"\nmisses: median gap {np.median(mg):.3f} of the score range, {100*np.mean(mg<=0.10):.1f}% within 10% of the top")
log("DONE_PRIMARY_PANELS")
