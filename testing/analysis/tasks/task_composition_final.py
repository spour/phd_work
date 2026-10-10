#!/usr/bin/env python3
# final contained analysis (freeze after this)
import sys, csv, os
import numpy as np
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)

FASTA = "../train/data/hg38.fa"
POS = "fig3_new_positional_features.tsv"
UP, DOWN, PAD, CAP, NFOLD, RIDGE = 2000, 500, 12, 8000, 5, 1.0
MONO = {nt: i for i, nt in enumerate("ACGT")}
DI = {a1 + a2: 4 + 4 * i + j for i, a1 in enumerate("ACGT") for j, a2 in enumerate("ACGT")}

# peaks
R = list(csv.DictReader(open(POS), delimiter="\t"))
NP = len(R)
gene = np.array([r["gene"] for r in R])
isc = np.array([int(float(r["is_canonical"])) for r in R])
chrom = [r["chrom"] for r in R]; center = [int(r["center"]) for r in R]; strand = [r["strand"] for r in R]
def col(name): return np.array([float(r[name]) if r[name] not in ("", "nan", "NA") else np.nan for r in R])
dpa = col("d_anchor_to_pa")
F1 = np.load("hypothesis_test_features.npy")

# part-2 per-window mono+di features (fetch fasta once, cache)
NWIN = 25
WINS = [(-2000 + 100 * k, -2000 + 100 * (k + 1)) for k in range(NWIN)]
WCACHE = "hypothesis_test_win2060.npy"

def comp_window(seq):
    out = np.zeros(20); n = len(seq)
    if n == 0: return out
    for ch in seq:
        j = MONO.get(ch)
        if j is not None: out[j] += 1
    out[:4] /= n
    if n >= 2:
        for k in range(n - 1):
            j = DI.get(seq[k:k + 2])
            if j is not None: out[j] += 1
        out[4:] /= (n - 1)
    return out

if os.path.exists(WCACHE):
    WIN = np.load(WCACHE); log(f"loaded cached window features {WIN.shape}")
else:
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
    G = Faidx(FASTA)
    WIN = np.full((NP, NWIN * 20), np.nan)
    for i in range(NP):
        ce, st, ch = center[i], strand[i], chrom[i]
        up_i = UP
        if np.isfinite(dpa[i]) and dpa[i] > UP: up_i = int(min(dpa[i], CAP))
        if st == "+": a, b = ce - up_i - PAD, ce + DOWN + PAD
        else:         a, b = ce - DOWN - PAD, ce + up_i + PAD
        seq = G.fetch(ch, a, b, st)
        if not seq: continue
        o = up_i + PAD
        for w, (p0, p1) in enumerate(WINS):
            WIN[i, w * 20:(w + 1) * 20] = comp_window(seq[max(0, o + p0):max(0, o + p1)])
        if (i + 1) % 10000 == 0: log(f"  win features {i+1}/{NP}")
    np.save(WCACHE, WIN); log("window features done, cached")

# kept peaks (composition present), same convention as task_composition_test.py
have = np.isfinite(F1[:, 0:60]).all(1)
keep = np.where(have)[0]
gene_k = gene[keep]; y = isc[keep].astype(float)
Fk = F1[keep]; WINk = WIN[keep]
NK = len(keep); log(f"kept peaks {NK}/{NP}")

genes = sorted(set(gene_k)); rs = np.random.default_rng(7)
perm = rs.permutation(len(genes)); fold_of = {g: perm[i] % NFOLD for i, g in enumerate(genes)}
fold = np.array([fold_of[g] for g in gene_k])

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
    pred = np.full(NK, np.nan)
    for f in range(NFOLD):
        tr = fold != f; te = fold == f
        Xtr = X[tr].astype(float).copy(); Xte = X[te].astype(float).copy()
        mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0); sd[sd == 0] = 1.0
        Xtr = np.where(np.isfinite(Xtr), Xtr, mu); Xte = np.where(np.isfinite(Xte), Xte, mu)
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        Xtr = np.column_stack([np.ones(Xtr.shape[0]), Xtr]); Xte = np.column_stack([np.ones(Xte.shape[0]), Xte])
        beta = fit_ridge(Xtr, y[tr], RIDGE); pred[te] = Xte @ beta
    return pred

# metrics
def auroc(yv, s):
    order = np.argsort(s, kind="mergesort"); sv = s[order]; n = len(s)
    ranks = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]: j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    rr = np.empty(n); rr[order] = ranks
    npos = yv.sum(); nneg = n - npos
    if npos == 0 or nneg == 0: return np.nan
    return (rr[yv == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)

def auprc(yv, s):
    order = np.argsort(-s, kind="mergesort"); ys = yv[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); P = ys.sum()
    if P == 0: return np.nan
    prec = tp / (tp + fp); rec = tp / P
    ap = 0.0; prev = 0.0
    for k in range(len(ys)):
        if ys[k] == 1: ap += prec[k] * (rec[k] - prev); prev = rec[k]
    return ap

gp = defaultdict(list)
for i in range(NK): gp[gene_k[i]].append(i)
GEN = [np.array(v) for v in gp.values()]
NG = len(GEN)
GENP = [idx for idx in GEN if len(idx) >= 2 and (isc[keep][idx] == 1).any()]
NGP = len(GENP)
CHANCE_P1 = float(np.mean([1.0 / len(idx) for idx in GENP]))
def hn(n): return float(np.sum(1.0 / np.arange(1, n + 1)))
CHANCE_MRR = float(np.mean([hn(len(idx)) / len(idx) for idx in GEN]))

def mrr(s):
    tot = 0.0
    for idx in GEN:
        sc = s[idx]; yv = y[idx]
        order = np.argsort(-sc, kind="mergesort"); ranks = np.empty(len(sc)); ranks[order] = np.arange(1, len(sc) + 1)
        pos = ranks[yv == 1]
        if len(pos): tot += 1.0 / pos.min()
    return tot / NG

def p_at1_credit(s):
    out = np.empty(NGP)
    for k, idx in enumerate(GENP):
        sc = s[idx]; mx = np.nanmax(sc)
        if not np.isfinite(mx): out[k] = 0.0; continue
        atmax = sc == mx
        out[k] = float((isc[keep][idx][atmax] == 1).sum()) / atmax.sum()
    return out

NBOOT = 500
rsb = np.random.default_rng(11); BOOT = [rsb.integers(0, NG, NG) for _ in range(NBOOT)]
rsbp = np.random.default_rng(11); BOOTP = [rsbp.integers(0, NGP, NGP) for _ in range(NBOOT)]

def pooled(sample, s, metric):
    peaks = np.concatenate([GEN[g] for g in sample])
    return auroc(y[peaks], s[peaks]) if metric == "auroc" else auprc(y[peaks], s[peaks])
def mrr_boot(sample, s):
    tot = 0.0
    for g in sample:
        idx = GEN[g]; sc = s[idx]; yv = y[idx]
        order = np.argsort(-sc, kind="mergesort"); ranks = np.empty(len(sc)); ranks[order] = np.arange(1, len(sc) + 1)
        pos = ranks[yv == 1]
        if len(pos): tot += 1.0 / pos.min()
    return tot / len(sample)

def metrics(s):
    ar, ap, mr = auroc(y, s), auprc(y, s), mrr(s)
    arv = np.array([pooled(bs, s, "auroc") for bs in BOOT]); apv = np.array([pooled(bs, s, "auprc") for bs in BOOT])
    mrv = np.array([mrr_boot(bs, s) for bs in BOOT])
    cred = p_at1_credit(s); p1 = cred.mean(); p1v = np.array([cred[bs].mean() for bs in BOOTP])
    return dict(auroc=ar, auroc_lo=np.percentile(arv, 2.5), auroc_hi=np.percentile(arv, 97.5),
                auprc=ap, auprc_lo=np.percentile(apv, 2.5), auprc_hi=np.percentile(apv, 97.5),
                mrr=mr, mrr_lo=np.percentile(mrv, 2.5), mrr_hi=np.percentile(mrv, 97.5),
                p_at1=p1, p_at1_lo=np.percentile(p1v, 2.5), p_at1_hi=np.percentile(p1v, 97.5), cred=cred)

def paired_inc(sa, sb, metric):
    if metric == "auroc": d = auroc(y, sa) - auroc(y, sb); v = np.array([pooled(bs, sa, "auroc") - pooled(bs, sb, "auroc") for bs in BOOT])
    elif metric == "auprc": d = auprc(y, sa) - auprc(y, sb); v = np.array([pooled(bs, sa, "auprc") - pooled(bs, sb, "auprc") for bs in BOOT])
    elif metric == "mrr": d = mrr(sa) - mrr(sb); v = np.array([mrr_boot(bs, sa) - mrr_boot(bs, sb) for bs in BOOT])
    else:
        ca, cb = p_at1_credit(sa), p_at1_credit(sb); d = ca.mean() - cb.mean()
        v = np.array([ca[bs].mean() - cb[bs].mean() for bs in BOOTP])
    return d, np.percentile(v, 2.5), np.percentile(v, 97.5)

COMP_far = Fk[:, 0:20]
COMP_near = Fk[:, 20:60]
SPLICE = Fk[:, 66:71]
ALL_cols = list(range(0, 71)); NOPAS_cols = list(range(0, 60)) + list(range(64, 71))
def hs(*m): return np.column_stack(m)

log("PART 1: ALL vs ALL_noPAS ...")
s_ALL = cv_scores(Fk[:, ALL_cols]); m_ALL = metrics(s_ALL)
s_NOPAS = cv_scores(Fk[:, NOPAS_cols]); m_NOPAS = metrics(s_NOPAS)
inc1 = {m: paired_inc(s_NOPAS, s_ALL, m) for m in ("auroc", "auprc", "mrr", "p_at1")}

log("PART 2: per-window mono+di localiser ...")
win_auroc = []
win_pred = []
for w in range(NWIN):
    s = cv_scores(WINk[:, w * 20:(w + 1) * 20]); win_pred.append(s)
    a = auroc(y, s); win_auroc.append(a)
    log(f"  win [{WINS[w][0]:+5d},{WINS[w][1]:+5d}] AUROC {a:.4f}")
win_auroc = np.array(win_auroc)
top3 = np.argsort(-win_auroc)[:3]

log("PART 2: greedy-forward cumulative ...")
selected = []; remaining = list(range(NWIN)); cum_auroc = []
DEPTH = 20
for step in range(DEPTH):
    best_a, best_w = -1.0, None
    for w in remaining:
        cols = []
        for s in selected + [w]: cols.extend(range(s * 20, (s + 1) * 20))
        a = auroc(y, cv_scores(WINk[:, cols]))
        if a > best_a: best_a, best_w = a, w
    selected.append(best_w); remaining.remove(best_w); cum_auroc.append(best_a)
    log(f"  +{step+1} win [{WINS[best_w][0]:+5d},{WINS[best_w][1]:+5d}]  cum AUROC {best_a:.4f}")
CUM_AT = [1, 2, 5, 10, 20]
cum_report = [(k, cum_auroc[k - 1], selected[k - 1]) for k in CUM_AT]

log("PART 3: complementarity ...")
COMPL = {
    "COMP_far": COMP_far, "COMP_near": COMP_near, "SPLICE": SPLICE,
    "COMP_far+SPLICE": hs(COMP_far, SPLICE), "COMP_near+SPLICE": hs(COMP_near, SPLICE),
    "COMP_far+near+SPLICE": hs(COMP_far, COMP_near, SPLICE),
}
cpred = {nm: cv_scores(X) for nm, X in COMPL.items()}
cmet = {nm: metrics(cpred[nm]) for nm in COMPL}
inc3 = {}
for nm in ("COMP_far+SPLICE", "COMP_near+SPLICE", "COMP_far+near+SPLICE"):
    inc3[nm] = {m: paired_inc(cpred[nm], cpred["SPLICE"], m) for m in ("auroc", "p_at1")}

NEW_ROWS = {
    "ALL_noPAS": m_NOPAS,
    "COMP_far": cmet["COMP_far"], "COMP_near": cmet["COMP_near"],
    "COMP_far+SPLICE": cmet["COMP_far+SPLICE"], "COMP_near+SPLICE": cmet["COMP_near+SPLICE"],
}
rows = list(csv.reader(open("hypothesis_test_results.tsv"), delimiter="\t"))
hdr = rows[0]
body = [r for r in rows[1:] if r[0] not in NEW_ROWS]
BASE = float(body[0][11]) if body else 0.09308
for blk, m in NEW_ROWS.items():
    body.append([blk,
        f"{m['auroc']:.5f}", f"{m['auroc_lo']:.5f}", f"{m['auroc_hi']:.5f}",
        f"{m['auprc']:.5f}", f"{m['auprc_lo']:.5f}", f"{m['auprc_hi']:.5f}",
        f"{m['mrr']:.5f}", f"{m['mrr_lo']:.5f}", f"{m['mrr_hi']:.5f}",
        "0.5", f"{BASE:.5f}", f"{CHANCE_MRR:.5f}",
        f"{m['p_at1']:.5f}", f"{m['p_at1_lo']:.5f}", f"{m['p_at1_hi']:.5f}", f"{CHANCE_P1:.5f}"])
with open("hypothesis_test_results.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(hdr)
    for r in body: w.writerow(r)

irows = list(csv.reader(open("hypothesis_test_increments.tsv"), delimiter="\t"))
ihdr = irows[0]
NEW_INC_LABELS = {"ALL_noPAS vs ALL", "COMP_far+SPLICE over SPLICE",
                  "COMP_near+SPLICE over SPLICE", "COMP_far+near+SPLICE over SPLICE"}
ibody = [r for r in irows[1:] if r[0] not in NEW_INC_LABELS]
for m in ("auroc", "auprc", "mrr", "p_at1"):
    d, lo, hi = inc1[m]
    ibody.append(["ALL_noPAS vs ALL", "ALL_noPAS - ALL", "does dropping PAS cost the full model?", m, f"{d:+.5f}", f"{lo:+.5f}", f"{hi:+.5f}"])
for nm in ("COMP_far+SPLICE", "COMP_near+SPLICE", "COMP_far+near+SPLICE"):
    lab = f"{nm} over SPLICE"
    for m in ("auroc", "p_at1"):
        d, lo, hi = inc3[nm][m]
        ibody.append([lab, f"{nm} - SPLICE", "which composition compartment adds over splice features?", m, f"{d:+.5f}", f"{lo:+.5f}", f"{hi:+.5f}"])
with open("hypothesis_test_increments.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(ihdr)
    for r in ibody: w.writerow(r)

with open("window_auroc_monodi.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(["win_lo", "win_hi", "win_center", "auroc"])
    for wi in range(NWIN): w.writerow([WINS[wi][0], WINS[wi][1], (WINS[wi][0] + WINS[wi][1]) // 2, f"{win_auroc[wi]:.5f}"])
with open("cumulative_windows.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(["step", "added_win_lo", "added_win_hi", "cum_auroc"])
    for st in range(DEPTH): w.writerow([st + 1, WINS[selected[st]][0], WINS[selected[st]][1], f"{cum_auroc[st]:.5f}"])

def fmt(m, k): return f"{m[k]:.4f} [{m[k+'_lo']:.4f},{m[k+'_hi']:.4f}]"
print("\n================ 1. ALL WITHOUT PAS ================")
print("competitor set is atlas-detectable CPA usage (needs a functional hexamer) -> hexamer strength")
print("is partly a set-construction artifact. Descriptive fact kept as text: competitors carry the")
print("stronger/more canonical hexamer (best_score raw AUROC 0.401, presence 0.441; canonical direction<0.5).")
print(f"{'model':10s} {'AUROC':>22s} {'AUPRC':>22s} {'MRR':>22s} {'P@1':>22s}")
for nm, m in (("ALL", m_ALL), ("ALL_noPAS", m_NOPAS)):
    print(f"{nm:10s} {fmt(m,'auroc'):>22s} {fmt(m,'auprc'):>22s} {fmt(m,'mrr'):>22s} {fmt(m,'p_at1'):>22s}")
print("change (ALL_noPAS - ALL), paired bootstrap over genes:")
for m in ("auroc", "auprc", "mrr", "p_at1"):
    d, lo, hi = inc1[m]; print(f"  {m:6s} {d:+.4f} [{lo:+.4f},{hi:+.4f}]")

print("\n================ 2. FAIR LOCALISER (mono+di, 100-nt windows across [-2000,+500]) ================")
print(f"{'window':>16s}  AUROC")
for wi in range(NWIN):
    star = "  <-- top3" if wi in top3 else ""
    print(f"  [{WINS[wi][0]:+5d},{WINS[wi][1]:+5d}]  {win_auroc[wi]:.4f}{star}")
print("top 3 windows:")
for wi in top3: print(f"  [{WINS[wi][0]:+5d},{WINS[wi][1]:+5d}]  AUROC {win_auroc[wi]:.4f}")
print("cumulative (greedy forward, gene-split CV) -- one region or distributed?")
for k, a, w in cum_report:
    print(f"  best {k:2d} window(s): AUROC {a:.4f}   (last added [{WINS[w][0]:+5d},{WINS[w][1]:+5d}])")
print(f"  single best window {win_auroc[top3[0]]:.4f} -> best 20 {cum_auroc[19]:.4f}: "
      f"gain from adding windows = {cum_auroc[19]-win_auroc[top3[0]]:+.4f}")

print("\n================ 3. COMPLEMENTARITY (COMP_far=[-2000,-500], COMP_near=[-500,+500]) ================")
print(f"{'model':22s} {'AUROC':>22s} {'P@1':>22s}")
for nm in ("COMP_far", "COMP_near", "SPLICE", "COMP_far+SPLICE", "COMP_near+SPLICE", "COMP_far+near+SPLICE"):
    m = cmet[nm]; print(f"{nm:22s} {fmt(m,'auroc'):>22s} {fmt(m,'p_at1'):>22s}")
print("increment over SPLICE (paired bootstrap over genes):")
for nm in ("COMP_far+SPLICE", "COMP_near+SPLICE", "COMP_far+near+SPLICE"):
    da, la, ha = inc3[nm]["auroc"]; dp, lp, hp = inc3[nm]["p_at1"]
    print(f"  {nm+' - SPLICE':30s} AUROC {da:+.4f} [{la:+.4f},{ha:+.4f}]   P@1 {dp:+.4f} [{lp:+.4f},{hp:+.4f}]")

print("\n[out] hypothesis_test_results.tsv (+rows), hypothesis_test_increments.tsv (+rows),")
print("      window_auroc_monodi.tsv, cumulative_windows.tsv")
