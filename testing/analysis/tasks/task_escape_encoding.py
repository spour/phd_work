#!/usr/bin/env python3
# items 2 & 3 (freeze extension)
import sys, csv, os
import numpy as np
from collections import defaultdict
sys.path.insert(0, "scripts")
from extract_splicing_features import load_me5_lookup, init_maxentscan
def log(*a): print(*a, file=sys.stderr, flush=True)

FASTA = "../train/data/hg38.fa"; ME5NPY = "../train/splice_train/data/me5_lookup.npy"
GTF = "../train/data/annotation/gencode.v49.annotation.gtf"
POS = "fig3_new_positional_features.tsv"
PAD = 25; NFOLD = 5; RIDGE = 1.0
STRONG5 = 8.46; STRONG3 = 6.5
ESC_DON_WIN = 2000; ESC_ACC_WIN = 5000; ANC_DON_WIN = 5000
ECACHE = "hypothesis_test_escape.npy"; GCACHE = "gencode_acceptors.npz"

# peaks
R = list(csv.DictReader(open(POS), delimiter="\t"))
NP = len(R)
gene = np.array([r["gene"] for r in R]); isc = np.array([int(float(r["is_canonical"])) for r in R])
chrom = [r["chrom"] for r in R]; center = [int(r["center"]) for r in R]; strand = [r["strand"] for r in R]
def col(n): return np.array([float(r[n]) if r[n] not in ("", "nan", "NA") else np.nan for r in R])
accsc = col("acc"); dpa = col("d_anchor_to_pa"); anchor_pos_abs = col("anchor_3ss_pos")
F1 = np.load("hypothesis_test_features.npy")

# me5 dense table
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

# escape / exon-definition features
ENAMES = ["esc_donor", "esc_acc", "esc_present", "esc_len", "anc_donor", "anc_present", "anc_len"]
if os.path.exists(ECACHE):
    ESC = np.load(ECACHE); log(f"loaded cached escape features {ESC.shape}")
else:
    G = Faidx(FASTA); _, score3, m5, m3 = init_maxentscan()
    from maxentpy import maxent
    ESC = np.full((NP, 7), np.nan)
    for i in range(NP):
        ce, st, ch = center[i], strand[i], chrom[i]
        d = int(dpa[i]) if np.isfinite(dpa[i]) else 0
        up_ext = d + ANC_DON_WIN + 30
        dn_ext = ESC_DON_WIN + ESC_ACC_WIN + 30
        if st == "+": a, b = ce - up_ext - PAD, ce + dn_ext + PAD
        else:         a, b = ce - dn_ext - PAD, ce + up_ext + PAD
        seq = G.fetch(ch, a, b, st)
        if not seq: continue
        o = up_ext + PAD
        tok = Bm[np.frombuffer(seq.encode("ascii"), np.uint8)]
        n5 = len(tok) - 9 + 1
        if n5 <= 0: continue
        sw = np.lib.stride_tricks.sliding_window_view(tok, 9)
        good = (sw >= 0).all(1); gt = (sw[:, 3] == 2) & (sw[:, 4] == 3)
        valid5 = good & gt
        don_score = np.full(n5, -99.0); don_score[valid5] = ME5[sw[valid5] @ POW]
        don_pos = (np.arange(n5) + 3) - o

        # escape donor: best 5'ss in (0, 2000] downstream
        mdon = valid5 & (don_pos > 0) & (don_pos <= ESC_DON_WIN)
        if mdon.any():
            jbest = np.where(mdon)[0][np.argmax(don_score[mdon])]
            dscore = float(don_score[jbest]); dpos = int(don_pos[jbest])
            ESC[i, 0] = dscore
            lo = o + dpos + 1; hi = min(o + dpos + ESC_ACC_WIN, len(seq) - 23)
            best_a, best_apos = -99.0, None
            for k in range(max(lo, 0), hi + 1):
                if tok[k + 18] == 0 and tok[k + 19] == 2:
                    kmer = seq[k:k + 23]
                    if "N" in kmer: continue
                    try: v = maxent.score3(kmer, matrix=m3)
                    except Exception: continue
                    if v > best_a: best_a, best_apos = float(v), (k + 20) - o
            if best_apos is not None:
                ESC[i, 1] = best_a; ESC[i, 3] = float(best_apos - dpos)
                ESC[i, 2] = 1.0 if (dscore >= STRONG5 and best_a >= STRONG3) else 0.0
            else:
                ESC[i, 2] = 0.0
        else:
            ESC[i, 2] = 0.0

        apos = -d
        manc = valid5 & (don_pos >= apos - ANC_DON_WIN) & (don_pos < apos)
        if manc.any():
            jb = np.where(manc)[0][np.argmax(don_score[manc])]
            ascore = float(don_score[jb]); apos_don = int(don_pos[jb])
            ESC[i, 4] = ascore; ESC[i, 6] = float(apos - apos_don)
            ESC[i, 5] = 1.0 if (ascore >= STRONG5 and np.isfinite(accsc[i]) and accsc[i] >= STRONG3) else 0.0
        else:
            ESC[i, 5] = 0.0
        if (i + 1) % 5000 == 0: log(f"  escape {i+1}/{NP}")
    np.save(ECACHE, ESC); log("escape features done, cached")

def build_acceptors():
    if os.path.exists(GCACHE):
        z = np.load(GCACHE, allow_pickle=True); log("loaded cached GENCODE acceptors")
        return {k: z[k] for k in z.files}
    log("parsing GENCODE GTF for annotated 3'ss acceptors ...")
    tx = defaultdict(list); txstrand = {}
    with open(GTF) as fh:
        for L in fh:
            if L[0] == "#": continue
            f = L.split("\t")
            if len(f) < 9 or f[2] != "exon": continue
            ch, st, s0, e0 = f[0], f[6], int(f[3]), int(f[4])
            attr = f[8]
            j = attr.find('transcript_id "');
            if j < 0: continue
            j += 15; tid = attr[j:attr.find('"', j)]
            tx[tid].append((s0, e0)); txstrand[tid] = (ch, st)
    acc = defaultdict(list)
    for tid, exons in tx.items():
        ch, st = txstrand[tid]; exons.sort()
        if len(exons) < 2: continue
        if st == "+":
            for (s0, e0) in exons[1:]: acc[(ch, "+")].append(s0)
        else:
            for (s0, e0) in exons[:-1]: acc[(ch, "-")].append(e0)
    out = {}
    for (ch, st), v in acc.items():
        out[f"{ch}|{st}"] = np.array(sorted(set(v)), dtype=np.int64)
    np.savez(GCACHE, **out); log(f"cached {len(out)} chrom/strand acceptor sets")
    return out

ACC = build_acceptors()
def coincide(tol):
    hit = 0; tot = 0
    for i in range(NP):
        if not np.isfinite(anchor_pos_abs[i]): continue
        tot += 1; key = f"{chrom[i]}|{strand[i]}"
        arr = ACC.get(key)
        if arr is None or len(arr) == 0: continue
        p = int(round(anchor_pos_abs[i])); j = np.searchsorted(arr, p)
        near = False
        for jj in (j - 1, j):
            if 0 <= jj < len(arr) and abs(int(arr[jj]) - p) <= tol: near = True; break
        if near: hit += 1
    return hit, tot
h0, tot = coincide(0); h2, _ = coincide(2); h5, _ = coincide(5)
log(f"anchor coincidence: exact {h0}/{tot}={h0/tot:.3f}  +-2 {h2/tot:.3f}  +-5 {h5/tot:.3f}")

have = np.isfinite(F1[:, 0:60]).all(1)
keep = np.where(have)[0]
gene_k = gene[keep]; y = isc[keep].astype(float)
COMP = F1[keep, 0:60]; SPL = F1[keep, 66:71]; ESCk = ESC[keep]
NK = len(keep); log(f"kept peaks {NK}/{NP}")
BASEX = np.column_stack([COMP, SPL, ESCk])
NFEAT = BASEX.shape[1]

genes = sorted(set(gene_k)); rs = np.random.default_rng(7)
perm = rs.permutation(len(genes)); fold_of = {g: perm[i] % NFOLD for i, g in enumerate(genes)}
fold = np.array([fold_of[g] for g in gene_k])
gp = defaultdict(list)
for i in range(NK): gp[gene_k[i]].append(i)
GEN = [np.array(v) for v in gp.values()]; NG = len(GEN)
GENP = [idx for idx in GEN if len(idx) >= 2 and (isc[keep][idx] == 1).any()]; NGP = len(GENP)
CHANCE_P1 = float(np.mean([1.0 / len(idx) for idx in GENP]))
def hn(n): return float(np.sum(1.0 / np.arange(1, n + 1)))
CHANCE_MRR = float(np.mean([hn(len(idx)) / len(idx) for idx in GEN]))

def fit_ridge(X, yv, lam, pen_intercept=False):
    P = np.eye(X.shape[1]) * lam
    if not pen_intercept: P[0, 0] = 0.0
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        eta = np.clip(X @ beta, -30, 30); mu = 1 / (1 + np.exp(-eta))
        w = np.maximum(mu * (1 - mu), 1e-9); z = eta + (yv - mu) / w; XtW = X.T * w
        try: new = np.linalg.solve(XtW @ X + P, XtW @ z + P @ beta)
        except np.linalg.LinAlgError: new = np.linalg.lstsq(XtW @ X + P, XtW @ z + P @ beta, rcond=None)[0]
        if np.max(np.abs(new - beta)) < 1e-9: beta = new; break
        beta = new
    return beta

def standardize_fit(Xtr, Xte):
    mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0); sd[sd == 0] = 1.0
    Xtr = np.where(np.isfinite(Xtr), Xtr, mu); Xte = np.where(np.isfinite(Xte), Xte, mu)
    return (Xtr - mu) / sd, (Xte - mu) / sd

def cv_pointwise(X):
    pred = np.full(NK, np.nan)
    for f in range(NFOLD):
        tr = fold != f; te = fold == f
        Xtr, Xte = standardize_fit(X[tr].astype(float).copy(), X[te].astype(float).copy())
        Xtr = np.column_stack([np.ones(Xtr.shape[0]), Xtr]); Xte = np.column_stack([np.ones(Xte.shape[0]), Xte])
        beta = fit_ridge(Xtr, y[tr], RIDGE); pred[te] = Xte @ beta
    return pred

def within_gene_z(X):
    Z = X.astype(float).copy()
    for idx in GEN:
        sub = Z[idx]; mu = np.nanmean(sub, 0); sd = np.nanstd(sub, 0); sd[sd == 0] = 1.0
        Z[idx] = (sub - mu) / sd
    return Z
def within_gene_rank(X):
    Z = X.astype(float).copy(); Rr = np.full_like(Z, np.nan)
    for idx in GEN:
        sub = Z[idx]; m = len(idx)
        for c in range(sub.shape[1]):
            col_ = sub[:, c]; fin = np.isfinite(col_)
            r = np.full(m, np.nan)
            if fin.sum() > 0:
                order = np.argsort(col_[fin], kind="mergesort")
                rr = np.empty(order.shape[0]); rr[order] = np.arange(order.shape[0])
                denom = max(order.shape[0] - 1, 1)
                r[np.where(fin)[0]] = rr / denom
            Rr[idx[np.arange(m)], c] = r
    return Rr

def cv_pairwise(X):
    pred = np.full(NK, np.nan)
    for f in range(NFOLD):
        tr = fold != f; te = fold == f
        Xtr, Xte = standardize_fit(X[tr].astype(float).copy(), X[te].astype(float).copy())
        diffs = []; labs = []
        tr_idx = np.where(tr)[0]; pos_in_tr = {gi: k for k, gi in enumerate(tr_idx)}
        gtr = defaultdict(list)
        for gi in tr_idx: gtr[gene_k[gi]].append(gi)
        for g, members in gtr.items():
            can = [gi for gi in members if isc[keep][gi] == 1]
            comp = [gi for gi in members if isc[keep][gi] == 0]
            if not can or not comp: continue
            c0 = can[0]
            xc = Xtr[pos_in_tr[c0]]
            for gj in comp:
                dv = xc - Xtr[pos_in_tr[gj]]
                diffs.append(dv); labs.append(1.0)
                diffs.append(-dv); labs.append(0.0)
        D = np.array(diffs); lab = np.array(labs)
        D = np.column_stack([np.ones(D.shape[0]), D])
        beta = fit_ridge(D, lab, RIDGE, pen_intercept=True)
        w = beta[1:]
        pred[te] = Xte @ w
    return pred

def auroc(yv, s):
    order = np.argsort(s, kind="mergesort"); sv = s[order]; n = len(s); ranks = np.empty(n); i = 0
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
    prec = tp / (tp + fp); rec = tp / P; ap = 0.0; prev = 0.0
    for k in range(len(ys)):
        if ys[k] == 1: ap += prec[k] * (rec[k] - prev); prev = rec[k]
    return ap
def mrr(s):
    tot = 0.0
    for idx in GEN:
        sc = s[idx]; yv = y[idx]; order = np.argsort(-sc, kind="mergesort")
        ranks = np.empty(len(sc)); ranks[order] = np.arange(1, len(sc) + 1); pos = ranks[yv == 1]
        if len(pos): tot += 1.0 / pos.min()
    return tot / NG
def p1_credit(s):
    out = np.empty(NGP)
    for k, idx in enumerate(GENP):
        sc = s[idx]; mx = np.nanmax(sc)
        if not np.isfinite(mx): out[k] = 0.0; continue
        atmax = sc == mx; out[k] = float((isc[keep][idx][atmax] == 1).sum()) / atmax.sum()
    return out

NBOOT = 500; rsb = np.random.default_rng(11); BOOT = [rsb.integers(0, NG, NG) for _ in range(NBOOT)]
rsbp = np.random.default_rng(11); BOOTP = [rsbp.integers(0, NGP, NGP) for _ in range(NBOOT)]
def pooled(sample, s, metric):
    peaks = np.concatenate([GEN[g] for g in sample])
    return auroc(y[peaks], s[peaks]) if metric == "auroc" else auprc(y[peaks], s[peaks])
def mrr_boot(sample, s):
    tot = 0.0
    for g in sample:
        idx = GEN[g]; sc = s[idx]; yv = y[idx]; order = np.argsort(-sc, kind="mergesort")
        ranks = np.empty(len(sc)); ranks[order] = np.arange(1, len(sc) + 1); pos = ranks[yv == 1]
        if len(pos): tot += 1.0 / pos.min()
    return tot / len(sample)
def metrics(s):
    ar, ap, mr = auroc(y, s), auprc(y, s), mrr(s)
    arv = np.array([pooled(bs, s, "auroc") for bs in BOOT]); apv = np.array([pooled(bs, s, "auprc") for bs in BOOT])
    mrv = np.array([mrr_boot(bs, s) for bs in BOOT]); cred = p1_credit(s); p1 = cred.mean()
    p1v = np.array([cred[bs].mean() for bs in BOOTP])
    return dict(auroc=ar, auroc_lo=np.percentile(arv, 2.5), auroc_hi=np.percentile(arv, 97.5),
                auprc=ap, auprc_lo=np.percentile(apv, 2.5), auprc_hi=np.percentile(apv, 97.5),
                mrr=mr, mrr_lo=np.percentile(mrv, 2.5), mrr_hi=np.percentile(mrv, 97.5),
                p_at1=p1, p_at1_lo=np.percentile(p1v, 2.5), p_at1_hi=np.percentile(p1v, 97.5), cred=cred)
def paired_inc(sa, sb, metric):
    if metric == "auroc": d = auroc(y, sa) - auroc(y, sb); v = np.array([pooled(bs, sa, "auroc") - pooled(bs, sb, "auroc") for bs in BOOT])
    elif metric == "p_at1":
        ca, cb = p1_credit(sa), p1_credit(sb); d = ca.mean() - cb.mean(); v = np.array([ca[bs].mean() - cb[bs].mean() for bs in BOOTP])
    else: d = mrr(sa) - mrr(sb); v = np.array([mrr_boot(bs, sa) - mrr_boot(bs, sb) for bs in BOOT])
    return d, np.percentile(v, 2.5), np.percentile(v, 97.5)

log("fitting: escape-only, COMP+SPLICE, and 4 encodings ...")
s_esc = cv_pointwise(ESCk); m_esc = metrics(s_esc)
s_cs = cv_pointwise(np.column_stack([COMP, SPL])); m_cs = metrics(s_cs)
s_a = cv_pointwise(BASEX); m_a = metrics(s_a)
s_b = cv_pointwise(within_gene_z(BASEX)); m_b = metrics(s_b)
s_c = cv_pointwise(within_gene_rank(BASEX)); m_c = metrics(s_c)
s_d = cv_pairwise(BASEX); m_d = metrics(s_d)

inc_esc_over_cs = {mm: paired_inc(s_a, s_cs, mm) for mm in ("auroc", "p_at1", "mrr")}

NEW = {"ESCAPE": m_esc, "COMP+SPLICE+ESC(raw)": m_a,
       "CSE_within_gene_z": m_b, "CSE_within_gene_rank": m_c, "CSE_pairwise": m_d}
rows = list(csv.reader(open("hypothesis_test_results.tsv"), delimiter="\t")); hdr = rows[0]
body = [r for r in rows[1:] if r[0] not in NEW]
BASE = float(body[0][11]) if body else 0.09308
for blk, m in NEW.items():
    body.append([blk,
        f"{m['auroc']:.5f}", f"{m['auroc_lo']:.5f}", f"{m['auroc_hi']:.5f}",
        f"{m['auprc']:.5f}", f"{m['auprc_lo']:.5f}", f"{m['auprc_hi']:.5f}",
        f"{m['mrr']:.5f}", f"{m['mrr_lo']:.5f}", f"{m['mrr_hi']:.5f}",
        "0.5", f"{BASE:.5f}", f"{CHANCE_MRR:.5f}",
        f"{m['p_at1']:.5f}", f"{m['p_at1_lo']:.5f}", f"{m['p_at1_hi']:.5f}", f"{CHANCE_P1:.5f}"])
with open("hypothesis_test_results.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(hdr)
    for r in body: w.writerow(r)

def fmt(m, k): return f"{m[k]:.4f} [{m[k+'_lo']:.4f},{m[k+'_hi']:.4f}]"
esc_present_rate = float(np.nanmean(ESCk[:, 2])); anc_present_rate = float(np.nanmean(ESCk[:, 5]))
print("\n================ 2. ESCAPE / EXON-DEFINITION FEATURES (MaxEnt scan, no annotation) ================")
print(f"complete escape intron present (donor>=8.46 & acc>=6.5): canonical {np.nanmean(ESCk[isc[keep]==1,2]):.3f}"
      f" vs competitor {np.nanmean(ESCk[isc[keep]==0,2]):.3f}")
print(f"complete anchor intron present:                          canonical {np.nanmean(ESCk[isc[keep]==1,5]):.3f}"
      f" vs competitor {np.nanmean(ESCk[isc[keep]==0,5]):.3f}")
print(f"escape block alone:  AUROC {fmt(m_esc,'auroc')}  P@1 {fmt(m_esc,'p_at1')}")
print("escape-feature increment over COMP+SPLICE (paired bootstrap):")
for mm in ("auroc", "p_at1", "mrr"):
    d, lo, hi = inc_esc_over_cs[mm]; print(f"   {mm:6s} {d:+.4f} [{lo:+.4f},{hi:+.4f}]")
print(f"COMP+SPLICE      AUROC {m_cs['auroc']:.4f}  P@1 {m_cs['p_at1']:.4f}")
print(f"COMP+SPLICE+ESC  AUROC {m_a['auroc']:.4f}  P@1 {m_a['p_at1']:.4f}")
print("\n[data-quality, NOT a feature] fraction of anchors coinciding with a GENCODE-annotated 3'ss (any acceptor):")
print(f"   exact {h0/tot:.3f}   within +-2 nt {h2/tot:.3f}   within +-5 nt {h5/tot:.3f}   (n={tot})")
print("   -> anchors are overwhelmingly annotated acceptors; using annotation as a FEATURE would be")
print("      partly circular (the competitor/anchor set is built from splice-site geometry). Kept as text.")

print("\n================ 3. WITHIN-GENE ENCODING + RANKING OBJECTIVE (COMP+SPLICE+escape) ================")
print("reference (standalone classifier, prior task): ALL_noPAS AUROC 0.7466  P@1 0.4787")
print(f"{'encoding':28s} {'AUROC':>22s} {'P@1':>22s} {'MRR':>22s}  kind")
print(f"{'a. raw features':28s} {fmt(m_a,'auroc'):>22s} {fmt(m_a,'p_at1'):>22s} {fmt(m_a,'mrr'):>22s}  standalone site classifier")
print(f"{'b. within-gene z-score':28s} {fmt(m_b,'auroc'):>22s} {fmt(m_b,'p_at1'):>22s} {fmt(m_b,'mrr'):>22s}  RANKING (needs full gene set)")
print(f"{'c. within-gene rank':28s} {fmt(m_c,'auroc'):>22s} {fmt(m_c,'p_at1'):>22s} {fmt(m_c,'mrr'):>22s}  RANKING (needs full gene set)")
print(f"{'d. pairwise logistic':28s} {fmt(m_d,'auroc'):>22s} {fmt(m_d,'p_at1'):>22s} {fmt(m_d,'mrr'):>22s}  RANKING (needs full gene set)")
best = max([("b. within-gene z", m_b), ("c. within-gene rank", m_c), ("d. pairwise", m_d)], key=lambda t: t[1]["p_at1"])
bestA = max([("b. within-gene z", m_b), ("c. within-gene rank", m_c), ("d. pairwise", m_d)], key=lambda t: t[1]["auroc"])
print("\nCEILING (best within-gene ranking encoding; interpretation of the missing item-1 'ceiling'):")
print(f"   best P@1  : {best[0]:20s} P@1 {best[1]['p_at1']:.4f} [{best[1]['p_at1_lo']:.4f},{best[1]['p_at1_hi']:.4f}]"
      f"  (raw baseline P@1 {m_a['p_at1']:.4f})")
print(f"   best AUROC: {bestA[0]:20s} AUROC {bestA[1]['auroc']:.4f} [{bestA[1]['auroc_lo']:.4f},{bestA[1]['auroc_hi']:.4f}]"
      f"  (raw baseline AUROC {m_a['auroc']:.4f})")
print("   (b)-(d) require the gene's full peak set at inference: they are RANKING models, not standalone")
print("   site classifiers, so they are not comparable to a per-site deployment of encoding (a).")
print("\n[out] hypothesis_test_results.tsv (+5 rows: ESCAPE, COMP+SPLICE+ESC(raw), CSE_within_gene_z/rank/pairwise)")
