#!/usr/bin/env python3
# four contained checks appended to the composition test
import sys, csv, os
import numpy as np
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)

FASTA = "../train/data/hg38.fa"; POS = "fig3_new_positional_features.tsv"
FCACHE1 = "hypothesis_test_features.npy"; FCACHE3 = "hypothesis_test_features3.npy"
UP, DOWN, PAD = 2000, 500, 12; NFOLD = 5; RIDGE = 1.0; BASE = 0.09308
BINW = 50; NBIN = (UP + DOWN) // BINW
MONO = {nt: i for i, nt in enumerate("ACGT")}

R = list(csv.DictReader(open(POS), delimiter="\t"))
NP = len(R)
gene = np.array([r["gene"] for r in R]); isc = np.array([int(float(r["is_canonical"])) for r in R])
chrom = [r["chrom"] for r in R]; center = [int(r["center"]) for r in R]; strand = [r["strand"] for r in R]
y = isc.astype(float)
F1 = np.load(FCACHE1)

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

# f3: [nbin*4 mono counts] + [down_afrac, has6run, has12of20]
NF3 = NBIN * 4 + 3
if os.path.exists(FCACHE3):
    F3 = np.load(FCACHE3); log(f"loaded cached F3 {F3.shape}")
else:
    F3 = np.zeros((NP, NF3)); G = Faidx(FASTA)
    for i in range(NP):
        ce = center[i]; st = strand[i]; ch = chrom[i]
        if st == "+": a, b = ce - UP - PAD, ce + DOWN + PAD
        else:         a, b = ce - DOWN - PAD, ce + UP + PAD
        seq = G.fetch(ch, a, b, st)
        if not seq: F3[i, :] = np.nan; continue
        o = UP + PAD
        def sl(p0, p1): return seq[max(0, o + p0):max(0, o + p1)]
        for bi in range(NBIN):
            lo = -UP + bi * BINW; w = sl(lo, lo + BINW)
            for c2 in w:
                j = MONO.get(c2)
                if j is not None: F3[i, bi * 4 + j] += 1
        dn = sl(0, 50)
        F3[i, NBIN*4 + 0] = dn.count("A") / len(dn) if dn else np.nan
        run = 0; has6 = 0
        for c2 in dn:
            run = run + 1 if c2 == "A" else 0
            if run >= 6: has6 = 1; break
        F3[i, NBIN*4 + 1] = has6
        has12 = 0
        if len(dn) >= 20:
            aflag = np.frombuffer(dn.encode("ascii"), np.uint8) == ord("A")
            cs = np.concatenate([[0], np.cumsum(aflag)])
            for s0 in range(len(dn) - 19):
                if cs[s0 + 20] - cs[s0] >= 12: has12 = 1; break
        F3[i, NBIN*4 + 2] = has12
        if (i + 1) % 10000 == 0: log(f"  F3 {i+1}/{NP}")
    np.save(FCACHE3, F3); log("F3 done, cached")

CNT = F3[:, :NBIN*4].reshape(NP, NBIN, 4)
dnA = F3[:, NBIN*4 + 0]; has6 = F3[:, NBIN*4 + 1]; has12 = F3[:, NBIN*4 + 2]

# cv infra (identical folds
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
def cv_scores(X):
    pred = np.full(NP, np.nan)
    for f in range(NFOLD):
        tr = fold != f; te = fold == f
        Xtr = X[tr].astype(float).copy(); Xte = X[te].astype(float).copy()
        mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0); sd[sd == 0] = 1.0
        Xtr = np.where(np.isfinite(Xtr), Xtr, mu); Xte = np.where(np.isfinite(Xte), Xte, mu)
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        Xtr = np.column_stack([np.ones(Xtr.shape[0]), Xtr]); Xte = np.column_stack([np.ones(Xte.shape[0]), Xte])
        beta = fit_ridge(Xtr, y[tr], RIDGE); pred[te] = Xte @ beta
    return pred
def auroc(yv, s):
    n = len(s); sorter = np.argsort(s, kind="mergesort"); ss = s[sorter]
    obs = np.r_[True, ss[1:] != ss[:-1]]; dense = np.cumsum(obs)
    cnt = np.r_[np.nonzero(obs)[0], n]; avg = 0.5 * (cnt[:-1] + cnt[1:] + 1)
    ranks = np.empty(n); ranks[sorter] = avg[dense - 1]
    npos = yv.sum(); nneg = n - npos
    if npos == 0 or nneg == 0: return np.nan
    return (ranks[yv == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
def auprc(yv, s):
    order = np.argsort(-s, kind="mergesort"); ys = yv[order]; ss = s[order]; P = ys.sum()
    if P == 0: return np.nan
    ctp = np.cumsum(ys); n = len(ys)
    last = np.r_[np.nonzero(ss[1:] != ss[:-1])[0], n - 1]
    rec = ctp[last] / P; prec = ctp[last] / (last + 1)
    drec = np.diff(np.r_[0.0, rec]); return float((prec * drec).sum())
gp = defaultdict(list)
for i in range(NP): gp[gene[i]].append(i)
GEN = [np.array(v) for v in gp.values()]; NG = len(GEN)
def fair_rr(sc, yv):
    pos = sc[yv == 1]
    if len(pos) == 0: return 0.0
    best = np.nanmax(pos)
    if not np.isfinite(best): return 0.0
    return 1.0 / (int((sc > best).sum()) + (int((sc == best).sum()) + 1) / 2.0)
def mrr(s): return float(np.mean([fair_rr(s[idx], y[idx]) for idx in GEN]))
NBOOT = 500; rsb = np.random.default_rng(11); BS = [rsb.integers(0, NG, NG) for _ in range(NBOOT)]
def pooled(sample, s, m):
    pk = np.concatenate([GEN[g] for g in sample]); return auroc(y[pk], s[pk]) if m == "auroc" else auprc(y[pk], s[pk])
def mrr_genes(sample, s): return float(np.mean([fair_rr(s[GEN[g]], y[GEN[g]]) for g in sample]))
def full_metrics(s):
    ar, ap, mr = auroc(y, s), auprc(y, s), mrr(s)
    va = np.array([pooled(bs, s, "auroc") for bs in BS]); vp = np.array([pooled(bs, s, "auprc") for bs in BS])
    vm = np.array([mrr_genes(bs, s) for bs in BS])
    return (ar, np.percentile(va,2.5), np.percentile(va,97.5), ap, np.percentile(vp,2.5), np.percentile(vp,97.5),
            mr, np.percentile(vm,2.5), np.percentile(vm,97.5))

# build resolution feature matrices from bin counts
def merge_freq(nmerge):
    g = NBIN // nmerge
    m = CNT[:, :g*nmerge, :].reshape(NP, g, nmerge, 4).sum(2)
    return (m / (nmerge * BINW)).reshape(NP, g * 4)
X_50 = merge_freq(1); X_25 = merge_freq(2); X_10 = merge_freq(5)
w1 = CNT[:, 0:30, :].sum(1) / 1500.0; w2 = CNT[:, 30:40, :].sum(1) / 500.0; w3 = CNT[:, 40:50, :].sum(1) / 500.0
X_3 = np.column_stack([w1, w2, w3])

log("resolution ...")
RES = [("COMP_mono_3win", X_3), ("COMP_mono_10bin", X_10), ("COMP_mono_25bin", X_25), ("COMP_mono_50bin", X_50)]
res_metrics = {}
for nm, X in RES:
    s = cv_scores(X); m = full_metrics(s); res_metrics[nm] = m
    log(f"  {nm:16s} AUROC {m[0]:.4f} AUPRC {m[3]:.4f} MRR {m[6]:.4f}")
s_monodi = cv_scores(F1[:, 0:60]); m_monodi = full_metrics(s_monodi)
log(f"  COMP(mono+di,3win) AUROC {m_monodi[0]:.4f}  [isolates di vs mono_3win]")

log("PAS 2-feature ...")
X2 = F1[:, [60, 62]]
s2 = cv_scores(X2); m2 = full_metrics(s2)
Xf = X2.astype(float).copy(); mu = np.nanmean(Xf,0); sd = np.nanstd(Xf,0); sd[sd==0]=1
Xf = (np.where(np.isfinite(Xf),Xf,mu)-mu)/sd; Xf = np.column_stack([np.ones(NP),Xf])
b2 = fit_ridge(Xf, y, RIDGE); sign_pres = "+" if b2[1] > 0 else "-"; sign_best = "+" if b2[2] > 0 else "-"

NEWB = {nm for nm, _ in RES} | {"PAS_pres+bestscore"}
chance_mrr = float(np.mean([np.sum(1.0/np.arange(1,len(idx)+1))/len(idx) for idx in GEN]))
ex = [r for r in csv.reader(open("hypothesis_test_results.tsv"), delimiter="\t") if r and r[0] not in NEWB]
with open("hypothesis_test_results.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    for r in ex: w.writerow(r)
    for nm, m in list(res_metrics.items()) + [("PAS_pres+bestscore", m2)]:
        w.writerow([nm] + [f"{x:.5f}" for x in m] + ["0.5", f"{BASE:.5f}", f"{chance_mrr:.5f}"])

log("per-window ...")
WINS = []
b = 20
while b <= 44:
    if b + 1 <= 43: WINS.append((b, b + 2)); b += 2
    else: WINS.append((b, b + 1)); b += 1
win_rows = []
for lo_b, hi_b in WINS:
    cen = (-UP + lo_b * BINW + (-UP + hi_b * BINW)) / 2.0
    cnts = CNT[:, lo_b:hi_b, :].sum(1); span = (hi_b - lo_b) * BINW
    Xw = cnts / span
    s = cv_scores(Xw); ar = auroc(y, s)
    win_rows.append((cen, lo_b, hi_b, ar)); log(f"  win [{-UP+lo_b*BINW:+5.0f},{-UP+hi_b*BINW:+5.0f}] AUROC {ar:.4f}")
with open("window_auroc.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(["win_center", "start", "end", "auroc"])
    for cen, lb, hb, ar in win_rows:
        w.writerow([f"{cen:.0f}", f"{-UP+lb*BINW:.0f}", f"{-UP+hb*BINW:.0f}", f"{ar:.5f}"])
top3 = sorted(win_rows, key=lambda r: -r[3])[:3]

def cls_stats(mask):
    return np.nanmean(dnA[mask]), np.nanmean(has6[mask]), np.nanmean(has12[mask])
can = isc == 1; comp = isc == 0
ca = cls_stats(can); co = cls_stats(comp)

# report
print("\n==== 1. RESOLUTION (mono-only, span [-2000,+500]) ====")
print(f"{'model':18s} {'AUROC':>8s} {'AUPRC':>8s} {'MRR':>8s}")
for nm, m in res_metrics.items():
    print(f"{nm:18s} {m[0]:8.4f} {m[3]:8.4f} {m[6]:8.4f}")
d = res_metrics["COMP_mono_50bin"][0] - res_metrics["COMP_mono_3win"][0]
print(f"AUROC change 3win -> 50bin: {d:+.4f}  ({'position carries info' if d>0.005 else 'flat/falling: resolution does not help'})")
print(f"di contribution: COMP mono+di 3win {m_monodi[0]:.4f} vs mono-only 3win {res_metrics['COMP_mono_3win'][0]:.4f} = +{m_monodi[0]-res_metrics['COMP_mono_3win'][0]:.4f}")
print("\n==== 2. LOCALISE (per-100-nt-window held-out AUROC) ====")
print("top 3 windows:")
for cen, lb, hb, ar in top3:
    print(f"  [{-UP+lb*BINW:+5.0f},{-UP+hb*BINW:+5.0f}]  AUROC {ar:.4f}")
print("\n==== 3. PAS SUPPRESSION (2-feature: presence + best_score) ====")
print(f"  AUROC {m2[0]:.4f} [{m2[1]:.4f},{m2[2]:.4f}]   coef signs: presence {sign_pres}, best_score {sign_best}")
print(f"  PAS block (4-feature) was 0.599; 2-feature {'reaches it' if m2[0]>=0.595 else 'does NOT fully reach it'}")
print("\n==== 4. DOWNSTREAM-A internal priming ([0,+50]) ====")
print(f"  mean A-content:   canonical {ca[0]:.4f}  competitor {co[0]:.4f}")
print(f"  frac >=6 consec A: canonical {ca[1]:.4f}  competitor {co[1]:.4f}")
print(f"  frac >=12/20 A:    canonical {ca[2]:.4f}  competitor {co[2]:.4f}")
flag = ca[0] > co[0] + 0.01 or ca[1] > co[1] + 0.01 or ca[2] > co[2] + 0.01
print(f"  -> {'FLAG: canonical enriched for downstream-A (possible internal-priming artifact)' if flag else 'canonical NOT enriched vs competitor: internal-priming artifact unlikely'}")
print("\n[out] appended hypothesis_test_results.tsv; window_auroc.tsv")
