#!/usr/bin/env python3
# two contained follow-ups to task_composition_test.py
import sys, csv, os
import numpy as np
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)

FASTA = "../train/data/hg38.fa"; POS = "fig3_new_positional_features.tsv"; U1NPZ = "u1_window_cache.npz"
FCACHE1 = "hypothesis_test_features.npy"; FCACHE2 = "hypothesis_test_features2.npy"
PAD = 12; NFOLD = 5; RIDGE = 1.0; BASE = 0.09308
PAS_SET = ["AATAAA","ATTAAA","TATAAA","AGTAAA","AATACA","CATAAA",
           "GATAAA","AATATA","AATGAA","TTTAAA","ACTAAA","AATAGA"]
PAS_STRONG = ("AATAAA", "ATTAAA")
PAS_RANK = {h: len(PAS_SET) - i for i, h in enumerate(PAS_SET)}
FINE_LO, FINE_HI, FINE_BIN = -1000, 250, 50
NBIN = (FINE_HI - FINE_LO) // FINE_BIN
CANON_LO, CANON_HI = -40, -10

# peaks
R = list(csv.DictReader(open(POS), delimiter="\t"))
NP = len(R)
gene = np.array([r["gene"] for r in R]); isc = np.array([int(float(r["is_canonical"])) for r in R])
chrom = [r["chrom"] for r in R]; center = [int(r["center"]) for r in R]; strand = [r["strand"] for r in R]
u1 = np.load(U1NPZ, allow_pickle=True)["dens_5.25"].astype(float)

F1 = np.load(FCACHE1)
COMP_COARSE = list(range(0, 60)); PAS_DEF = [60, 61, 62, 63]; SPLICE = [66, 67, 68, 69, 70]

class Faidx:
    _T = str.maketrans("ACGTNacgtn", "TGCANTGCAN")
    def __init__(s, fa):
        s.fh = open(fa, "rb"); s.idx = {}
        for L in open(fa + ".fai"):
            p = L.split(); s.idx[p[0]] = (int(p[1]), int(p[2]), int(p[3]), int(p[4]))
    def fetch(s, c, a, b, st="+"):
        c = c if c in s.idx else c.lstrip("chr")
        if c not in s.idx: return None
        Ln, off, lb, lw = s.idx[c]; a = max(0, a); b = min(b, Ln)
        if b <= a: return None
        bp = lambda p: off + (p // lb) * lw + (p % lb)
        s.fh.seek(bp(a)); raw = s.fh.read(bp(b) - bp(a))
        q = raw.replace(b"\n", b"").replace(b"\r", b"")[:b - a].decode("ascii").upper()
        return q.translate(s._T)[::-1] if st == "-" else q

MONO = {nt: i for i, nt in enumerate("ACGT")}
def best_pas(seq, lo_idx):
    best_sc, best_pos, cnt, strong = 0, None, 0, 0
    for k in range(len(seq) - 5):
        h = seq[k:k + 6]; rk = PAS_RANK.get(h)
        if rk is None: continue
        cnt += 1
        if h in PAS_STRONG: strong = 1
        if rk > best_sc: best_sc = rk; best_pos = lo_idx + k
    dist = float(-best_pos) if best_pos is not None else np.nan
    return strong, cnt, float(best_sc), dist

# new features: [canonical-window pas (4)] + [comp_fine (100)]
NF2 = 4 + NBIN * 4
if os.path.exists(FCACHE2):
    F2 = np.load(FCACHE2); log(f"loaded cached F2 {F2.shape}")
else:
    F2 = np.full((NP, NF2), np.nan)
    G = Faidx(FASTA)
    for i in range(NP):
        ce = center[i]; st = strand[i]; ch = chrom[i]
        if st == "+": a, b = ce + FINE_LO - PAD, ce + FINE_HI + PAD
        else:         a, b = ce - FINE_HI - PAD, ce - FINE_LO + PAD
        seq = G.fetch(ch, a, b, st)
        if not seq: continue
        o = -FINE_LO + PAD
        def sl(p0, p1): return seq[max(0, o + p0):max(0, o + p1)]
        ps, pc, psc, pd = best_pas(sl(CANON_LO, CANON_HI), CANON_LO)
        F2[i, 0] = ps; F2[i, 1] = pc; F2[i, 2] = psc; F2[i, 3] = pd
        for bi in range(NBIN):
            lo = FINE_LO + bi * FINE_BIN
            w = sl(lo, lo + FINE_BIN); n = len(w)
            if n:
                cnt = np.zeros(4)
                for ch2 in w:
                    j = MONO.get(ch2)
                    if j is not None: cnt[j] += 1
                F2[i, 4 + bi * 4: 4 + bi * 4 + 4] = cnt / n
        if (i + 1) % 10000 == 0: log(f"  F2 {i+1}/{NP}")
    np.save(FCACHE2, F2); log("F2 done, cached")

CANON_PAS = [0, 1, 2, 3]
FINE = list(range(4, 4 + NBIN * 4))

# cv infrastructure (identical to run 1)
y = isc.astype(float)
genes = sorted(set(gene)); rs = np.random.default_rng(7)
perm = rs.permutation(len(genes)); fold_of = {g: perm[k] % NFOLD for k, g in enumerate(genes)}
fold = np.array([fold_of[g] for g in gene])

def fit_ridge(X, yv, lam):
    P = np.eye(X.shape[1]) * lam; P[0, 0] = 0.0; beta = np.zeros(X.shape[1])
    for _ in range(100):
        eta = np.clip(X @ beta, -30, 30); mu = 1 / (1 + np.exp(-eta))
        w = np.maximum(mu * (1 - mu), 1e-9); z = eta + (yv - mu) / w; XtW = X.T * w
        try: new = np.linalg.solve(XtW @ X + P, XtW @ z + P @ beta)
        except np.linalg.LinAlgError: new = np.linalg.lstsq(XtW @ X + P, XtW @ z + P @ beta, rcond=None)[0]
        if np.max(np.abs(new - beta)) < 1e-9: beta = new; break
        beta = new
    return beta

def cv_scores(mats_cols):
    pred = np.full(NP, np.nan)
    for f in range(NFOLD):
        tr = fold != f; te = fold == f
        Xtr = np.column_stack([M[np.ix_(tr, c)] for M, c in mats_cols]).astype(float)
        Xte = np.column_stack([M[np.ix_(te, c)] for M, c in mats_cols]).astype(float)
        mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0); sd[sd == 0] = 1.0
        Xtr = np.where(np.isfinite(Xtr), Xtr, mu); Xte = np.where(np.isfinite(Xte), Xte, mu)
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        Xtr = np.column_stack([np.ones(Xtr.shape[0]), Xtr]); Xte = np.column_stack([np.ones(Xte.shape[0]), Xte])
        beta = fit_ridge(Xtr, y[tr], RIDGE); pred[te] = Xte @ beta
    return pred

def auroc(yv, s):
    order = np.argsort(s, kind="mergesort"); sv = s[order]; n = len(s)
    ranks = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]: j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    rr = np.empty(n); rr[order] = ranks; npos = yv.sum(); nneg = n - npos
    if npos == 0 or nneg == 0: return np.nan
    return (rr[yv == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)

def auprc(yv, s):
    order = np.argsort(-s, kind="mergesort"); ys = yv[order]; ss = s[order]
    P = ys.sum()
    if P == 0: return np.nan
    ctp = np.cumsum(ys); cfp = np.cumsum(1 - ys); n = len(ys)
    ap = 0.0; prev_rec = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ss[j + 1] == ss[i]: j += 1
        rec = ctp[j] / P; prec = ctp[j] / (ctp[j] + cfp[j])
        ap += prec * (rec - prev_rec); prev_rec = rec
        i = j + 1
    return ap

gp = defaultdict(list)
for i in range(NP): gp[gene[i]].append(i)
GEN = [np.array(v) for v in gp.values()]; NG = len(GEN)
def fair_rr(sc, yv):
    pos = sc[yv == 1]
    if len(pos) == 0: return 0.0
    best = np.nanmax(pos)
    if not np.isfinite(best): return 0.0
    greater = int((sc > best).sum()); ties = int((sc == best).sum())
    return 1.0 / (greater + (ties + 1) / 2.0)
def mrr(s):
    return float(np.mean([fair_rr(s[idx], y[idx]) for idx in GEN]))

NBOOT = 500; rsb = np.random.default_rng(11); BS = [rsb.integers(0, NG, NG) for _ in range(NBOOT)]
def pooled(sample, s, metric):
    peaks = np.concatenate([GEN[g] for g in sample])
    return auroc(y[peaks], s[peaks]) if metric == "auroc" else auprc(y[peaks], s[peaks])
def mrr_genes(sample, s):
    return float(np.mean([fair_rr(s[GEN[g]], y[GEN[g]]) for g in sample]))
def ci(s, metric):
    if metric == "mrr": v = np.array([mrr_genes(bs, s) for bs in BS])
    else: v = np.array([pooled(bs, s, metric) for bs in BS])
    return np.percentile(v, 2.5), np.percentile(v, 97.5)

def metrics(s, with_ci=True):
    ar, ap, mr = auroc(y, s), auprc(y, s), mrr(s)
    if not with_ci: return ar, ap, mr, None, None, None, None, None, None
    al, ah = ci(s, "auroc"); pl, ph = ci(s, "auprc"); ml, mh = ci(s, "mrr")
    return ar, ap, mr, al, ah, pl, ph, ml, mh

PAS_FEATS = [
    ("presence",              F1, [60]), ("count",             F1, [61]),
    ("best_score",            F1, [62]), ("dist_to_pA",        F1, [63]),
    ("presence@[-40,-10]",    F2, [0]),  ("count@[-40,-10]",   F2, [1]),
    ("best_score@[-40,-10]",  F2, [2]),  ("dist@[-40,-10]",    F2, [3]),
]
log("PAS decomposition (RAW, sign-preserving) ...")
def raw_metrics(v):
    ok = np.isfinite(v)
    ar = auroc(y[ok], v[ok]); ap = auprc(y[ok], v[ok])
    mr = float(np.mean([fair_rr(v[idx], y[idx]) for idx in GEN]))
    n_miss = int((~ok).sum())
    return ar, ap, mr, n_miss
pas_rows = []
for nm, M, cols in PAS_FEATS:
    v = M[:, cols[0]].astype(float)
    ar, ap, mr, nm_miss = raw_metrics(v)
    pas_rows.append((nm, ar, ap, mr, nm_miss))
    log(f"  {nm:22s} AUROC {ar:.4f}  AUPRC {ap:.4f}  MRR {mr:.4f}  (missing {nm_miss})")
with open("pas_decomposition.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["pas_feature", "auroc_raw", "auprc_raw", "mrr_raw", "n_missing", "direction"])
    for nm, ar, ap, mr, nm_miss in pas_rows:
        note = "predicts REAL" if ar > 0.5 else ("predicts COMPETITOR" if ar < 0.5 else "null")
        w.writerow([nm, f"{ar:.5f}", f"{ap:.5f}", f"{mr:.5f}", nm_miss, note])

log("COMP_fine ...")
sFine = cv_scores([(F2, FINE)])
sFineSpl = cv_scores([(F2, FINE), (F1, SPLICE)])
sFinePas = cv_scores([(F2, FINE), (F1, PAS_DEF)])
mF = metrics(sFine); mFS = metrics(sFineSpl)
log(f"  COMP_fine        AUROC {mF[0]:.4f} AUPRC {mF[1]:.4f} MRR {mF[2]:.4f}")
log(f"  COMP_fine+SPLICE AUROC {mFS[0]:.4f} AUPRC {mFS[1]:.4f} MRR {mFS[2]:.4f}")

chance_mrr = float(np.mean([np.sum(1.0/np.arange(1,len(idx)+1))/len(idx) for idx in GEN]))
NEWB = {"COMP_fine", "COMP_fine+SPLICE"}
existing = list(csv.reader(open("hypothesis_test_results.tsv"), delimiter="\t"))
existing = [r for r in existing if r and r[0] not in NEWB]
with open("hypothesis_test_results.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    for r in existing: w.writerow(r)
    for nm, m in [("COMP_fine", mF), ("COMP_fine+SPLICE", mFS)]:
        ar, ap, mr, al, ah, pl, ph, ml, mh = m
        w.writerow([nm, f"{ar:.5f}", f"{al:.5f}", f"{ah:.5f}", f"{ap:.5f}", f"{pl:.5f}", f"{ph:.5f}",
                    f"{mr:.5f}", f"{ml:.5f}", f"{mh:.5f}", "0.5", f"{BASE:.5f}", f"{chance_mrr:.5f}"])

def paired(sa, sb, metric):
    if metric == "mrr":
        d = mrr(sa) - mrr(sb); v = np.array([mrr_genes(bs, sa) - mrr_genes(bs, sb) for bs in BS])
    else:
        fa = auroc if metric == "auroc" else auprc
        d = fa(y, sa) - fa(y, sb); v = np.array([pooled(bs, sa, metric) - pooled(bs, sb, metric) for bs in BS])
    return d, np.percentile(v, 2.5), np.percentile(v, 97.5)
log("increment PAS over COMP_fine ...")
inc_ex = list(csv.reader(open("hypothesis_test_increments.tsv"), delimiter="\t"))
inc_ex = [r for r in inc_ex if r and r[0] != "PAS over COMP_fine"]
with open("hypothesis_test_increments.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    for r in inc_ex: w.writerow(r)
    for m in ("auroc", "auprc", "mrr"):
        d, lo, hi = paired(sFinePas, sFine, m)
        w.writerow(["PAS over COMP_fine", "COMP_fine+PAS - COMP_fine",
                    "does the hexamer add beyond positional composition?", m, f"{d:+.5f}", f"{lo:+.5f}", f"{hi:+.5f}"])
        log(f"  PAS over COMP_fine {m}: {d:+.4f} [{lo:+.4f},{hi:+.4f}]")

X = F2[:, FINE].astype(float); mu = np.nanmean(X, 0); sd = np.nanstd(X, 0); sd[sd == 0] = 1.0
X = (np.where(np.isfinite(X), X, mu) - mu) / sd
X = np.column_stack([np.ones(NP), X]); beta = fit_ridge(X, y, RIDGE)[1:]
with open("comp_fine_coefs.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(["bin_center", "nt", "coef"])
    for bi in range(NBIN):
        cen = FINE_LO + bi * FINE_BIN + FINE_BIN / 2
        for j, nt in enumerate("ACGT"):
            w.writerow([int(cen), nt, f"{beta[bi*4+j]:.5f}"])
coef = beta.reshape(NBIN, 4)
absmax = np.abs(coef).max(1); pk = int(np.argmax(absmax)); pk_nt = "ACGT"[int(np.argmax(np.abs(coef[pk])))]
pk_cen = FINE_LO + pk * FINE_BIN + FINE_BIN / 2
Acoef = coef[:, 0]; Apk = int(np.argmax(np.abs(Acoef))); A_cen = FINE_LO + Apk * FINE_BIN + FINE_BIN / 2

sBlock = cv_scores([(F1, PAS_DEF)]); block_ar = auroc(y, sBlock)
print("\n==== 1. PAS PER-FEATURE (RAW, sign-preserving; canonical positive) ====")
print("strong-hexamer presence by class: real 0.809 vs competitor 0.927")
print("windows: PAS block search [-60,0]; earlier decoy result [-50,0] (PAS_W=50); canonical positional [-40,-10].")
print(f"{'feature':24s} {'AUROC':>8s} {'AUPRC':>8s} {'MRR':>8s}  direction")
for nm, ar, ap, mr, _m in pas_rows:
    d = "REAL" if ar > 0.5 else ("COMPETITOR" if ar < 0.5 else "null")
    print(f"{nm:24s} {ar:8.4f} {ap:8.4f} {mr:8.4f}  {d}")
print(f"fitted PAS block (4 default feats, joint, oriented) AUROC {block_ar:.4f}  [run 1 reported 0.599]")
print("NOTE: a fitted single-feature logistic always orients AUROC>=0.5; raw scores above keep the sign.")
print("\n==== 2. POSITIONAL COMPOSITION ====")
print(f"{'block':18s} {'AUROC':>8s} {'AUPRC':>8s} {'MRR':>8s}")
print(f"{'COMP (coarse)':18s} {'0.7185':>8s} {'0.2148':>8s} {'0.6302':>8s}   (run 1)")
print(f"{'COMP_fine':18s} {mF[0]:8.4f} {mF[1]:8.4f} {mF[2]:8.4f}")
print(f"{'COMP_fine+SPLICE':18s} {mFS[0]:8.4f} {mFS[1]:8.4f} {mFS[2]:8.4f}")
print(f"{'ALL':18s} {'0.7614':>8s} {'0.2652':>8s} {'0.6989':>8s}   (run 1)")
print("\nincrement PAS over COMP_fine (paired bootstrap):")
for m in ("auroc", "auprc", "mrr"):
    d, lo, hi = paired(sFinePas, sFine, m); print(f"  {m}: {d:+.4f} [{lo:+.4f},{hi:+.4f}]")
print(f"\npositional coefficient peak: |coef| max at bin center {pk_cen:+.0f} nt ({pk_nt})")
print(f"A coefficient peak at {A_cen:+.0f} nt ; A |coef| profile min {np.abs(Acoef).min():.2f} max {np.abs(Acoef).max():.2f} (dispersed if many bins comparable)")
nbig = int((np.abs(Acoef) > 0.5 * np.abs(Acoef).max()).sum())
print(f"A: {nbig}/{NBIN} bins carry >50% of the peak |coef|  -> {'dispersed' if nbig>=5 else 'concentrated'} A-content")
print("\n[out] appended hypothesis_test_results.tsv, hypothesis_test_increments.tsv; pas_decomposition.tsv, comp_fine_coefs.tsv")
