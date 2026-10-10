#!/usr/bin/env python3
# one contained test of one hypothesis
import sys, csv, math
import numpy as np
from collections import defaultdict
sys.path.insert(0, "scripts")
from extract_splicing_features import load_me5_lookup
def log(*a): print(*a, file=sys.stderr, flush=True)

FASTA = "../train/data/hg38.fa"; ME5NPY = "../train/splice_train/data/me5_lookup.npy"
POS = "fig3_new_positional_features.tsv"; U1NPZ = "u1_window_cache.npz"
UP = 2000; DOWN = 500; PAD = 12; CAP = 8000; BASE = 0.09308; NFOLD = 5; RIDGE = 1.0
STRONG5 = 8.46
PAS_STRONG = ("AATAAA", "ATTAAA")
PAS_SET = ["AATAAA","ATTAAA","TATAAA","AGTAAA","AATACA","CATAAA",
           "GATAAA","AATATA","AATGAA","TTTAAA","ACTAAA","AATAGA"]
PAS_RANK = {h: len(PAS_SET) - i for i, h in enumerate(PAS_SET)}
PAS_LO, PAS_HI = -60, 0

# me5 dense table (== score5, byte-identical per reproducibility)
Bm = np.full(256, -1, np.int64)
for ch, v in zip(b"ACGT", (0, 1, 2, 3)): Bm[ch] = v
POW = (4 ** np.arange(8, -1, -1)).astype(np.int64)
me5d = load_me5_lookup(ME5NPY)
ks = np.frombuffer("".join(me5d.keys()).encode("ascii"), np.uint8).reshape(-1, 9)
ME5 = np.full(4 ** 9, -99.0); ME5[Bm[ks] @ POW] = np.fromiter(me5d.values(), np.float64, len(me5d))

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

# read peaks
R = list(csv.DictReader(open(POS), delimiter="\t"))
NP = len(R)
gene = np.array([r["gene"] for r in R])
isc = np.array([int(float(r["is_canonical"])) for r in R])
chrom = [r["chrom"] for r in R]; center = [int(r["center"]) for r in R]; strand = [r["strand"] for r in R]
def col(name):
    return np.array([float(r[name]) if r[name] not in ("", "nan", "NA") else np.nan for r in R])
acc = col("acc"); apos = col("anchor_3ss_pos"); dpa = col("d_anchor_to_pa")
u1 = np.load(U1NPZ, allow_pickle=True)["dens_5.25"].astype(float)
assert (np.load(U1NPZ, allow_pickle=True)["isc"] == isc).all(), "U1 cache row order mismatch"

# master feature matrix
COMP_N = 60
names = []
for w in ("up", "mid", "down"):
    for nt in "ACGT": names.append(f"comp_{w}_{nt}")
    for a1 in "ACGT":
        for a2 in "ACGT": names.append(f"comp_{w}_{a1}{a2}")
names += ["pas_present", "pas_count", "pas_bestscore", "pas_dist"]
names += ["arch_dse_gu", "arch_use_u"]
names += ["spl_acc", "spl_dpa", "spl_strong5", "spl_max5", "spl_u1"]
NF = len(names)
F = np.full((NP, NF), np.nan)
DI = {a1 + a2: 4 + 4 * i + j for i, a1 in enumerate("ACGT") for j, a2 in enumerate("ACGT")}
MONO = {nt: i for i, nt in enumerate("ACGT")}

import os
FCACHE = "hypothesis_test_features.npy"
if os.path.exists(FCACHE):
    F = np.load(FCACHE); log(f"loaded cached features {F.shape}")
    _SKIP_FEATURES = True
else:
    _SKIP_FEATURES = False

def comp_window(seq):
    out = np.zeros(20)
    n = len(seq)
    if n == 0: return out
    for ch in seq:
        j = MONO.get(ch)
        if j is not None: out[j] += 1
    out[:4] /= n
    if n >= 2:
        for k in range(n - 1):
            d = seq[k:k + 2]; j = DI.get(d)
            if j is not None: out[j] += 1
        out[4:] /= (n - 1)
    return out

def best_pas(seq, lo_idx):
    best_sc, best_pos, cnt = 0, None, 0
    strong = 0
    L = len(seq)
    for k in range(L - 5):
        h = seq[k:k + 6]
        rk = PAS_RANK.get(h)
        if rk is None: continue
        cnt += 1
        if h in PAS_STRONG: strong = 1
        if rk > best_sc:
            best_sc = rk; best_pos = lo_idx + k
    dist = float(-best_pos) if best_pos is not None else np.nan
    return strong, cnt, float(best_sc), dist

for i in ([] if _SKIP_FEATURES else range(NP)):
    ce = center[i]; st = strand[i]; ch = chrom[i]
    up_i = UP
    if np.isfinite(dpa[i]) and dpa[i] > UP: up_i = int(min(dpa[i], CAP))
    if st == "+": a, b = ce - up_i - PAD, ce + DOWN + PAD
    else:         a, b = ce - DOWN - PAD, ce + up_i + PAD
    seq = G.fetch(ch, a, b, st)
    if not seq: continue
    o = up_i + PAD
    def sl(p0, p1): return seq[max(0, o + p0):max(0, o + p1)]
    win_up = sl(-2000, -500); win_mid = sl(-500, 0); win_dn = sl(0, 500)
    F[i, 0:20] = comp_window(win_up); F[i, 20:40] = comp_window(win_mid); F[i, 40:60] = comp_window(win_dn)
    ps, pc, pscore, pdist = best_pas(sl(PAS_LO, PAS_HI), PAS_LO)
    F[i, 60] = ps; F[i, 61] = pc; F[i, 62] = pscore; F[i, 63] = pdist
    dse = sl(0, 50); use = sl(-40, -10)
    F[i, 64] = (sum(dse[k:k+2] in ("GT", "TG") for k in range(len(dse)-1)) / max(len(dse)-1, 1)) if dse else 0.0
    F[i, 65] = (use.count("T") / len(use)) if use else 0.0
    F[i, 66] = acc[i]; F[i, 67] = dpa[i]
    mx5 = np.nan
    if np.isfinite(dpa[i]) and dpa[i] >= 9:
        d = int(min(dpa[i], CAP)); tok = Bm[np.frombuffer(seq.encode("ascii"), np.uint8)]
        sw = np.lib.stride_tricks.sliding_window_view(tok, 9); good = (sw >= 0).all(1)
        if good.any():
            sc = ME5[sw[good] @ POW]; idx = np.where(good)[0] + 3 - o; m = (idx >= -d) & (idx < 0)
            mx5 = float(sc[m].max()) if m.any() else np.nan
    F[i, 68] = 1.0 if (np.isfinite(mx5) and mx5 >= STRONG5) else (0.0 if np.isfinite(mx5) else np.nan)
    F[i, 69] = mx5; F[i, 70] = u1[i]
    if (i + 1) % 10000 == 0: log(f"  features {i+1}/{NP}")
if not _SKIP_FEATURES:
    np.save(FCACHE, F); log("features done, cached")

BLOCKS = {
    "COMP": list(range(0, 60)),
    "PAS": [60, 61, 62, 63],
    "ARCH": [64, 65],
    "SPLICE": [66, 67, 68, 69, 70],
    "COMP+PAS": list(range(0, 60)) + [60, 61, 62, 63],
    "COMP+SPLICE": list(range(0, 60)) + [66, 67, 68, 69, 70],
    "COMP+PAS+ARCH": list(range(0, 60)) + [60, 61, 62, 63, 64, 65],
    "ALL": list(range(0, NF)),
}

have = np.isfinite(F[:, 0:60]).all(1)
log(f"peaks with composition: {have.sum()}/{NP}")
keep = np.where(have)[0]
gene_k = gene[keep]; y = isc[keep].astype(float); Fk = F[keep]
NK = len(keep)

genes = sorted(set(gene_k)); rs = np.random.default_rng(7)
perm = rs.permutation(len(genes)); fold_of = {g: perm[i] % NFOLD for i, g in enumerate(genes)}
fold = np.array([fold_of[g] for g in gene_k])

def fit_ridge(X, yv, lam):
    P = np.eye(X.shape[1]) * lam; P[0, 0] = 0.0
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        eta = np.clip(X @ beta, -30, 30); mu = 1 / (1 + np.exp(-eta))
        w = np.maximum(mu * (1 - mu), 1e-9); z = eta + (yv - mu) / w
        XtW = X.T * w
        try: new = np.linalg.solve(XtW @ X + P, XtW @ z + P @ beta)
        except np.linalg.LinAlgError: new = np.linalg.lstsq(XtW @ X + P, XtW @ z + P @ beta, rcond=None)[0]
        if np.max(np.abs(new - beta)) < 1e-9: beta = new; break
        beta = new
    return beta

def cv_scores(cols):
    pred = np.full(NK, np.nan)
    for f in range(NFOLD):
        tr = fold != f; te = fold == f
        Xtr = Fk[np.ix_(tr, cols)].copy(); Xte = Fk[np.ix_(te, cols)].copy()
        mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0); sd[sd == 0] = 1.0
        Xtr = np.where(np.isfinite(Xtr), Xtr, mu); Xte = np.where(np.isfinite(Xte), Xte, mu)
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        Xtr = np.column_stack([np.ones(Xtr.shape[0]), Xtr]); Xte = np.column_stack([np.ones(Xte.shape[0]), Xte])
        beta = fit_ridge(Xtr, y[tr], RIDGE)
        pred[te] = Xte @ beta
    return pred

# metrics
def auroc(yv, s):
    order = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); sv = s[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sv[j + 1] == sv[i]: j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    rr = np.empty(len(s)); rr[order] = ranks
    npos = yv.sum(); nneg = len(yv) - npos
    if npos == 0 or nneg == 0: return np.nan
    return (rr[yv == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)

def auprc(yv, s):
    order = np.argsort(-s, kind="mergesort"); ys = yv[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); P = ys.sum()
    if P == 0: return np.nan
    prec = tp / (tp + fp); rec = tp / P
    ap = 0.0; prev = 0.0
    for k in range(len(ys)):
        if ys[k] == 1:
            ap += prec[k] * (rec[k] - prev); prev = rec[k]
    return ap

gp = defaultdict(list)
for i in range(NK): gp[gene_k[i]].append(i)
GEN = [np.array(v) for v in gp.values()]
NG = len(GEN)

def mrr(s):
    tot = 0.0
    for idx in GEN:
        sc = s[idx]; yv = y[idx]
        order = np.argsort(-sc, kind="mergesort")
        ranks = np.empty(len(sc)); ranks[order] = np.arange(1, len(sc) + 1)
        pos = ranks[yv == 1]
        if len(pos): tot += 1.0 / pos.min()
    return tot / NG

def hn(n): return float(np.sum(1.0 / np.arange(1, n + 1)))
chance_mrr = float(np.mean([hn(len(idx)) / len(idx) for idx in GEN]))

NBOOT = 500
gidx_arr = [idx for idx in GEN]
rsb = np.random.default_rng(11)
BOOT_SAMPLES = [rsb.integers(0, NG, NG) for _ in range(NBOOT)]

def boot_ci(fn_gene_metric):
    vals = np.array([fn_gene_metric(bs) for bs in BOOT_SAMPLES])
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)

def pooled_from_genes(sample, s, metric):
    peaks = np.concatenate([GEN[g] for g in sample])
    if metric == "auroc": return auroc(y[peaks], s[peaks])
    if metric == "auprc": return auprc(y[peaks], s[peaks])

def mrr_from_genes(sample, s):
    tot = 0.0
    for g in sample:
        idx = GEN[g]; sc = s[idx]; yv = y[idx]
        order = np.argsort(-sc, kind="mergesort")
        ranks = np.empty(len(sc)); ranks[order] = np.arange(1, len(sc) + 1)
        pos = ranks[yv == 1]
        if len(pos): tot += 1.0 / pos.min()
    return tot / len(sample)

PRED = {}; rows = []
for blk, cols in BLOCKS.items():
    s = cv_scores(cols); PRED[blk] = s
    ar = auroc(y, s); ap = auprc(y, s); mr = mrr(s)
    arlo, arhi = boot_ci(lambda bs, s=s: pooled_from_genes(bs, s, "auroc"))
    aplo, aphi = boot_ci(lambda bs, s=s: pooled_from_genes(bs, s, "auprc"))
    mrlo, mrhi = boot_ci(lambda bs, s=s: mrr_from_genes(bs, s))
    rows.append([blk, ar, arlo, arhi, ap, aplo, aphi, mr, mrlo, mrhi])
    log(f"  {blk:14s} AUROC {ar:.4f} [{arlo:.4f},{arhi:.4f}]  AUPRC {ap:.4f}  MRR {mr:.4f}")

with open("hypothesis_test_results.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["block", "auroc", "auroc_lo", "auroc_hi", "auprc", "auprc_lo", "auprc_hi",
                "mrr", "mrr_lo", "mrr_hi", "auroc_baseline", "auprc_baseline", "mrr_chance"])
    for r in rows: w.writerow([r[0]] + [f"{x:.5f}" for x in r[1:]] + ["0.5", f"{BASE:.5f}", f"{chance_mrr:.5f}"])

def paired_inc(a, b, metric):
    sa, sb = PRED[a], PRED[b]
    if metric == "mrr":
        d = mrr(sa) - mrr(sb)
        vals = np.array([mrr_from_genes(bs, sa) - mrr_from_genes(bs, sb) for bs in BOOT_SAMPLES])
    else:
        fa = {"auroc": auroc, "auprc": auprc}[metric]
        d = fa(y, sa) - fa(y, sb)
        vals = np.array([pooled_from_genes(bs, sa, metric) - pooled_from_genes(bs, sb, metric) for bs in BOOT_SAMPLES])
    return d, np.percentile(vals, 2.5), np.percentile(vals, 97.5)

INC = [("COMP+PAS", "COMP", "PAS over COMP", "does the hexamer add beyond composition?"),
       ("COMP+SPLICE", "COMP", "SPLICE over COMP", "do splice features add beyond composition?"),
       ("COMP", "SPLICE", "COMP over SPLICE", "is composition the better single block?")]
with open("hypothesis_test_increments.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["comparison", "models", "question", "metric", "delta", "ci_lo", "ci_hi"])
    for a, b, lab, q in INC:
        for m in ("auroc", "auprc", "mrr"):
            d, lo, hi = paired_inc(a, b, m)
            w.writerow([lab, f"{a} - {b}", q, m, f"{d:+.5f}", f"{lo:+.5f}", f"{hi:+.5f}"])
            log(f"  {lab} ({a}-{b}) {m}: {d:+.4f} [{lo:+.4f},{hi:+.4f}]")

sC = PRED["COMP"]
def pear(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    return float(np.corrcoef(a, b)[0, 1])
acc_k = Fk[:, 66]; u1_k = Fk[:, 70]
r_acc = pear(sC, acc_k); r_u1 = pear(sC, u1_k)

rsp = np.random.default_rng(3); perm_mrr = []
for _ in range(200):
    tot = 0.0
    for idx in GEN:
        n = len(idx); pos = rsp.integers(0, n) + 1; tot += 1.0 / pos
    perm_mrr.append(tot / NG)
perm_mrr = np.array(perm_mrr)

cols = BLOCKS["COMP"]; X = Fk[:, cols].copy()
mu = np.nanmean(X, 0); sd = np.nanstd(X, 0); sd[sd == 0] = 1.0
X = (np.where(np.isfinite(X), X, mu) - mu) / sd
X = np.column_stack([np.ones(NK), X]); beta = fit_ridge(X, y, RIDGE)
coef = beta[1:]; ordr = np.argsort(-np.abs(coef))[:10]
top = [(names[cols[j]], coef[j]) for j in ordr]

print("\n==== RESULTS (held-out, pooled over 5 gene-split folds) ====")
print(f"peaks used {NK}  genes {NG}  base rate {y.mean():.4f}")
print(f"{'block':14s} {'AUROC':>20s} {'AUPRC':>10s} {'MRR':>18s}")
for r in rows:
    print(f"{r[0]:14s} {r[1]:.4f} [{r[2]:.4f},{r[3]:.4f}] {r[4]:8.4f} {r[7]:.4f} [{r[8]:.4f},{r[9]:.4f}]")
print(f"{'baseline':14s} {'0.5':>20s} {BASE:8.4f} {chance_mrr:.4f} (analytic H_n/n)")
print(f"MRR chance: analytic {chance_mrr:.4f} vs permutation {perm_mrr.mean():.4f} [{np.percentile(perm_mrr,2.5):.4f},{np.percentile(perm_mrr,97.5):.4f}]")
print("\n==== INCREMENTS (paired bootstrap over genes, nested models) ====")
for a, b, lab, q in INC:
    ds = []
    for m in ("auroc", "auprc", "mrr"):
        d, lo, hi = paired_inc(a, b, m); ds.append(f"{m} {d:+.4f}[{lo:+.4f},{hi:+.4f}]")
    print(f"  {lab:16s} ({a} - {b}): " + "  ".join(ds) + f"   [{q}]")
print(f"\ncorr(COMP score, anchor acceptor MaxEnt) = {r_acc:+.3f}")
print(f"corr(COMP score, U1@5.25 density)         = {r_u1:+.3f}")
print("\ntop 10 COMP features by |standardised coef|:")
for nm, c in top: print(f"  {c:+.3f}  {nm}")
print("\n[out] hypothesis_test_results.tsv, hypothesis_test_increments.tsv")

np.savez("hypothesis_test_pred.npz",
         chance_mrr=chance_mrr, base=BASE, perm_mrr=perm_mrr,
         r_acc=r_acc, r_u1=r_u1)
