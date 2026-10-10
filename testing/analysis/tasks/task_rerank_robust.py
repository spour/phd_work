#!/usr/bin/env python3
# two robustness runs on the final 4-feature reranker, plus the anchor-annotation check
import csv, re, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)

SEED, HOLD_FRAC, RIDGE, NBOOT = 42, 0.20, 1.0, 400
FINAL = ["scan_n3ss_dn5k", "spl_upstream_max_me3", "spl_pas_prox_u1", "spl_n_5ss_1k"]
rng = np.random.default_rng(SEED)

D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
S = np.load("splice_feats_cache.npz")
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
col = {"scan_n3ss_dn5k": S["C_n3ss_dn5k"][ridx]}
for n in FINAL:
    if n not in col: col[n] = D[n].to_numpy(float)
X = np.column_stack([col[n] for n in FINAL])

At = np.load("atlas_peak_cache.npz")["dist"][ridx]
Pd = np.load("polyadb_peak_cache.npz")["dist"][ridx]
UNANNOT = (At > 500) & (Pd > 500)
fc_all = np.load("fine_compartment.npy")
Lb = list(csv.DictReader(open("cpa_peaks_labeled.tsv"), delimiter="\t"))
fmap = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): int(fc_all[i]) for i, r in enumerate(Lb)}
fine = np.array([fmap[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
TERM = fine == 2

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
def loco_pred(rowmask, train_mask, test_mask):
    pred = np.full(len(y), np.nan)
    for c in CHROMS:
        te = rowmask & test_mask & (chrom == c); tr = rowmask & train_mask & (chrom != c)
        if te.sum() == 0 or tr.sum() == 0: continue
        A, B = prep(X[tr], X[te]); pred[te] = B @ fit_ridge(A, y[tr])
    return pred
def per_gene(pred, rowmask, mask):
    hits, rrs, ns, gl = [], [], [], []
    for g, ix in gi.items():
        if not mask[ix[0]]: continue
        ix = ix[rowmask[ix]]
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
def run(tag, rowmask):
    pred = loco_pred(rowmask, ~is_hold, is_hold)
    hits, rrs, ns, gl = per_gene(pred, rowmask, is_hold)
    rows = np.concatenate([ix for _, ix in gl]); yv = y[rows]; sv = pred[rows]
    Hn = np.array([np.sum(1.0 / np.arange(1, int(n) + 1)) for n in ns])
    p1, p1lo, p1hi = bci(hits); mr, mrlo, mrhi = bci(rrs)
    e1, e1lo, e1hi = bci(hits - 1.0 / ns); em, emlo, emhi = bci(rrs - Hn / ns)
    gidx = [ix for _, ix in gl]; ab = np.empty(200); pb = np.empty(200)
    for t in range(200):
        ii = np.concatenate([gidx[j] for j in rng.integers(0, len(gidx), len(gidx))])
        ab[t] = auroc(y[ii], pred[ii]); pb[t] = auprc(y[ii], pred[ii])
    r = dict(model=tag, n_genes_hold=len(gl), n_rows_hold=len(rows),
             n_peaks_all=int((rowmask & (y == 0)).sum()),
             n_genes_all=len({gene[i] for i in np.where(rowmask & (y == 0))[0]}),
             p1=p1, p1lo=p1lo, p1hi=p1hi, p1_chance=float(np.mean(1.0 / ns)),
             p1_exc=e1, p1_exc_lo=e1lo, p1_exc_hi=e1hi,
             mrr=mr, mrr_lo=mrlo, mrr_hi=mrhi, mrr_chance=float(np.mean(Hn / ns)),
             mrr_exc=em, mrr_exc_lo=emlo, mrr_exc_hi=emhi,
             auroc=auroc(yv, sv), auroc_lo=float(np.percentile(ab, 2.5)), auroc_hi=float(np.percentile(ab, 97.5)),
             auprc=auprc(yv, sv), auprc_lo=float(np.percentile(pb, 2.5)), auprc_hi=float(np.percentile(pb, 97.5)),
             base_rate=float(yv.mean()))
    dev = np.where(rowmask & ~is_hold)[0]
    A, _ = prep(X[dev], X[dev]); beta = fit_ridge(A, y[dev])[1:]
    devg = [ix[rowmask[ix]] for g, ix in gi.items() if not is_hold[ix[0]] and rowmask[ix].sum() >= 2]
    BB = np.empty((300, len(FINAL)))
    for t in range(300):
        ii = np.concatenate([devg[j] for j in rng.integers(0, len(devg), len(devg))])
        Ab, _ = prep(X[ii], X[ii]); BB[t] = fit_ridge(Ab, y[ii])[1:]
    co = []
    for a, nm in enumerate(FINAL):
        lo, hi = np.percentile(BB[:, a], [2.5, 97.5])
        co.append((nm, float(beta[a]), float(lo), float(hi), float(np.mean(np.sign(BB[:, a]) != np.sign(beta[a])))))
    return r, co

BASE = np.ones(len(y), bool)
def gene_filter(mask):
    keep = np.zeros(len(y), bool)
    for g, ix in gi.items():
        sub = ix[mask[ix]]
        if (y[sub] == 1).sum() == 1 and (y[sub] == 0).sum() >= 1: keep[sub] = True
    return keep
M_UN = gene_filter((y == 1) | ((y == 0) & UNANNOT))
gterm = {g for g, ix in gi.items() if TERM[ix[y[ix] == 1]].all() and (TERM[ix] & (y[ix] == 0)).sum() >= 1}
M_TT = gene_filter(np.array([TERM[i] and gene[i] in gterm for i in range(len(y))]))

RUNS = [("unrestricted", BASE), ("unannotated competitors", M_UN), ("TERM-TERM", M_TT)]
OUT = []; COEFS = {}
for tag, m in RUNS:
    r, co = run(tag, m); OUT.append(r); COEFS[tag] = co
    log(f"  {tag} done")

ks = ["n_peaks_all","n_genes_all","n_genes_hold","n_rows_hold","p1","p1lo","p1hi","p1_chance","p1_exc","p1_exc_lo","p1_exc_hi",
      "mrr","mrr_lo","mrr_hi","mrr_chance","mrr_exc","mrr_exc_lo","mrr_exc_hi","auroc","auroc_lo","auroc_hi",
      "auprc","auprc_lo","auprc_hi","base_rate"]
with open("rerank_robust_metrics.tsv", "w") as fh:
    fh.write("model\t" + "\t".join(ks) + "\n")
    for r in OUT:
        fh.write(r["model"] + "\t" + "\t".join(f"{r[k]:.5f}" if isinstance(r[k], float) else str(r[k]) for k in ks) + "\n")
with open("rerank_robust_coefs.tsv", "w") as fh:
    fh.write("model\tfeature\tbeta_std\tci_lo\tci_hi\tsign_flip_frac\n")
    for tag, co in COEFS.items():
        for nm, b, lo, hi, fl in co: fh.write(f"{tag}\t{nm}\t{b:.5f}\t{lo:.5f}\t{hi:.5f}\t{fl:.4f}\n")

print("\n=====  ROBUSTNESS: holdout (untouched 20% of genes), 4-feature model  =====")
print(f"{'run':<26} {'comp peaks':>10} {'comp genes':>10} {'hold genes':>10} | {'P@1 [CI]':>22} {'chance':>7} {'excess':>8} | "
      f"{'MRR':>7} {'chance':>7} | {'AUROC [CI]':>22} | {'AUPRC [CI]':>22} {'base':>6}")
for r in OUT:
    print(f"{r['model']:<26} {r['n_peaks_all']:>10} {r['n_genes_all']:>10} {r['n_genes_hold']:>10} | "
          f"{r['p1']:>6.4f}[{r['p1lo']:.4f},{r['p1hi']:.4f}] {r['p1_chance']:>7.4f} {r['p1_exc']:>+8.4f} | "
          f"{r['mrr']:>7.4f} {r['mrr_chance']:>7.4f} | {r['auroc']:>6.4f}[{r['auroc_lo']:.4f},{r['auroc_hi']:.4f}] | "
          f"{r['auprc']:>6.4f}[{r['auprc_lo']:.4f},{r['auprc_hi']:.4f}] {r['base_rate']:>6.4f}")
print("\n=====  STANDARDIZED COEFFICIENTS per run  =====")
for tag, co in COEFS.items():
    print(f"  {tag}")
    for nm, b, lo, hi, fl in co:
        print(f"    {nm:<24} {b:>+8.4f} [{lo:>+8.4f},{hi:>+8.4f}]  flip {fl:.3f}")

GTF = "../train/data/annotation/gencode.v49.annotation.gtf"
acc = defaultdict(list)
TID = re.compile(r'transcript_id "([^"]+)"')
tx = defaultdict(list)
with open(GTF) as fh:
    for line in fh:
        if line[0] == "#": continue
        f = line.split("\t")
        if f[2] != "exon": continue
        t = TID.search(f[8])
        if not t: continue
        tx[(f[0], f[6], t.group(1))].append((int(f[3]), int(f[4])))
for (ch, st, _), ex in tx.items():
    ex.sort()
    if len(ex) < 2: continue
    if st == "+":
        for s, e in ex[1:]: acc[(ch, st)].append(s)
    else:
        for s, e in ex[:-1]: acc[(ch, st)].append(e)
ACC = {k: np.unique(np.array(v)) for k, v in acc.items()}
log(f"GENCODE acceptors: {sum(len(v) for v in ACC.values())} unique positions")

anchor = np.array([float(r["anchor_3ss_pos"]) if r["anchor_3ss_pos"] not in ("", "nan") else np.nan for r in M])
mchrom = np.array([r["chrom"] for r in M]); mstr = np.array([r["strand"] for r in M])
misc = np.array([int(float(r["is_canonical"])) for r in M])
d = np.full(len(M), np.nan)
for key in set(zip(mchrom, mstr)):
    a = ACC.get(key)
    if a is None: continue
    sel = np.where((mchrom == key[0]) & (mstr == key[1]) & np.isfinite(anchor))[0]
    if not len(sel): continue
    p = anchor[sel]
    j = np.searchsorted(a, p)
    lo = a[np.clip(j - 1, 0, len(a) - 1)]; hi = a[np.clip(j, 0, len(a) - 1)]
    d[sel] = np.minimum(np.abs(p - lo), np.abs(p - hi))
off = d[np.isfinite(d)]
print("\n=====  ANCHOR vs GENCODE ANNOTATED ACCEPTOR  =====")
print(f"anchors with a finite distance: {int(np.isfinite(d).sum())} of {len(M)}")
with open("rerank_anchor_annot.tsv", "w") as fh:
    fh.write("group\tn\ttol\tfrac\n")
    for gname, m in [("canonical", misc == 1), ("competitor", misc == 0)]:
        mm = m & np.isfinite(d); n = int(mm.sum())
        line = f"  {gname:<12} n {n:>6}  "
        for tol in (0, 2, 10, 25, 100):
            fr = float((d[mm] <= tol).mean()); line += f"<= {tol:>3} nt {fr:.3f}   "
            fh.write(f"{gname}\t{n}\t{tol}\t{fr:.5f}\n")
        print(line)
        print(f"{'':<14} median distance to the nearest annotated acceptor {np.median(d[mm]):.0f} nt")
log("DONE_ROBUST")
