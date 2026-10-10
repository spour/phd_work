#!/usr/bin/env python3
# build cpa_peaks_final.parquet: the shareable per-peak table for the final analysis
import csv, os, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, HOLD_FRAC, RIDGE = 42, 0.20, 1.0
rng = np.random.default_rng(SEED)
FASTA = "../train/data/hg38.fa"
UP, DN, ACC_END, SPLIT = 2000, 523, 1874, 437
OUT = "cpa_peaks_final.parquet"
OLD_FILE = "rerank_corr_peaks.parquet"
OLD_MARKED = "rerank_corr_peaks.SUPERSEDED.parquet"

M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); strand = D.strand.to_numpy()
cen = D.center.to_numpy().astype(np.int64)
y = D.is_canonical.to_numpy().astype(bool)
N = len(D)
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
Zu = np.load("upstream_rowmask.npz"); ROW = Zu["ROW"]; UPFLAG = Zu["UP"]
Zc = np.load("corr_rowmask.npz"); APA = Zc["APA"]; TERMEX = Zc["TERMEX"]; OLDROW = Zc["ROW"]
me3 = D["spl_upstream_max_me3"].to_numpy(float)

# flags
tpos = np.where(strand == "+", cen, -cen)
gi = defaultdict(list)
for i in range(N): gi[gene[i]].append(i)
gi = {g: np.array(v) for g, v in gi.items()}
DOWN = np.zeros(N, bool)
for g, ix in gi.items():
    c = ix[y[ix]]
    if len(c) != 1: continue
    DOWN[ix[(~y[ix]) & (tpos[ix] >= tpos[c[0]])]] = True

# gc_contrast for every peak
if not os.path.exists("gc_contrast_all.npz"):
    class Faidx:
        _T = str.maketrans("ACGTNacgtn", "TGCANTGCAN")
        def __init__(s, fa):
            s.fh = open(fa, "rb"); s.idx = {}
            for L in open(fa + ".fai"):
                p = L.split(); s.idx[p[0]] = (int(p[1]), int(p[2]), int(p[3]), int(p[4]))
        def fetch(s, c, a, b, st="+"):
            c = c if c in s.idx else c.lstrip("chr")
            if c not in s.idx: return None
            Ln, off, lb, lw = s.idx[c]
            if a < 0 or b > Ln: return None
            bp = lambda p: off + (p // lb) * lw + (p % lb)
            s.fh.seek(bp(a)); raw = s.fh.read(bp(b) - bp(a))
            q = raw.replace(b"\n", b"").replace(b"\r", b"")[:b - a].decode("ascii").upper()
            return q.translate(s._T)[::-1] if st == "-" else q
    _MAP = np.full(256, -1, np.int64)
    for i, ch in enumerate("ACGT"): _MAP[ord(ch)] = i
    G = Faidx(FASTA)
    dsl = slice(0, UP - SPLIT); psl = slice(UP - SPLIT, ACC_END + 1)
    gd = np.full(N, np.nan); gp = np.full(N, np.nan)
    for i in range(N):
        a, b = (cen[i] - UP, cen[i] + DN) if strand[i] == "+" else (cen[i] - DN, cen[i] + UP)
        q = G.fetch(chrom[i], int(a), int(b), strand[i])
        if q is None or len(q) != UP + DN: continue
        c = _MAP[np.frombuffer(q.encode("ascii"), np.uint8)]
        isgc = (c == 1) | (c == 2); isn = c < 0
        for sl, arr in ((dsl, gd), (psl, gp)):
            d_ = int((~isn[sl]).sum())
            if d_: arr[i] = isgc[sl].sum() / d_
        if (i + 1) % 20000 == 0: log(f"  gc scan {i+1:,}/{N:,}")
    np.savez_compressed("gc_contrast_all.npz", gc_distal=gd, gc_proximal=gp)
Zg = np.load("gc_contrast_all.npz")
gcc = Zg["gc_distal"] - Zg["gc_proximal"]
CS = np.load("control_contrast_scan.npz", allow_pickle=True); CSN = list(CS["names"])
ref = CS["X"][:, CSN.index("gc_distal")] - CS["X"][:, CSN.index("gc_proximal")]
m = ROW & np.isfinite(ref) & np.isfinite(gcc)
print(f"[check] gc_contrast rescan vs the analysis cache: n {int(m.sum()):,}, "
      f"max abs diff {np.max(np.abs(gcc[m] - ref[m])):.10f}")
assert np.max(np.abs(gcc[m] - ref[m])) == 0.0

# holdout split and the discriminant
allg = np.array(sorted(set(gene)))
hold_g = set(rng.choice(allg, size=int(round(HOLD_FRAC * len(allg))), replace=False))
is_hold = np.array([g in hold_g for g in gene])
CHROMS = sorted(set(chrom))
X2 = np.column_stack([me3, gcc])
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
    return (np.column_stack([np.ones(len(Xtr)), (np.where(np.isfinite(Xtr), Xtr, mu) - mu) / sd]),
            np.column_stack([np.ones(len(Xte)), (np.where(np.isfinite(Xte), Xte, mu) - mu) / sd]))
disc = np.full(N, np.nan)
yf = y.astype(float)
for c in CHROMS:
    te = ROW & (chrom == c); tr = ROW & ~is_hold & (chrom != c)
    if te.sum() == 0 or tr.sum() == 0: continue
    A, B = prep(X2[tr], X2[te]); disc[te] = B @ fit_ridge(A, yf[tr])

# verification
def ng(mask): return len({gene[i] for i in np.where(mask)[0]})
comp = ~y
print("\n===== VERIFICATION =====")
print(f"rows {N:,} (all peaks kept)   genes {len(allg):,}")
print(f"in_analysis_set {int(ROW.sum()):,} in {ng(ROW):,} genes = "
      f"{int((ROW & y).sum()):,} canonical + {int((ROW & comp).sum()):,} competitor")
ok = [(int(ROW.sum()) == 48394), (ng(ROW) == 4847), (int((ROW & y).sum()) == 4847),
      (int((ROW & comp).sum()) == 43547)]
print("\nfilter counts on the competitor pool, as the figure builds them:")
f1 = comp & APA
f2 = comp & TERMEX & ~APA
f3 = OLDROW & comp & ~UPFLAG
for lab, v, want in [("APA-annotated", f1, 18119),
                     ("reference terminal exon, beyond the APA set", f2, 1076),
                     ("downstream of the canonical, beyond the first two", f3, 1398)]:
    print(f"  {lab:<50} {int(v.sum()):>7,}  expected {want:>7,}  {'OK' if int(v.sum())==want else 'MISMATCH'}")
    ok.append(int(v.sum()) == want)
print(f"  kept: upstream, unannotated, intronic              {int((ROW & comp).sum()):>7,}  "
      f"expected  43,547  {'OK' if int((ROW&comp).sum())==43547 else 'MISMATCH'}")
print("\nraw flag counts over all 70,723 peaks (a peak may carry more than one):")
for lab, v in [("flag_apa_annotated", APA), ("flag_reference_terminal_exon", TERMEX),
               ("flag_downstream_of_canonical", DOWN)]:
    print(f"  {lab:<32} {int(v.sum()):>7,}   of which canonical {int((v & y).sum()):>6,}")
print(f"\nin_gene_holdout: {ng(is_hold):,} genes, {int(is_hold.sum()):,} peaks, "
      f"{int((ROW & is_hold).sum()):,} in the analysis set")
print(f"discriminant present for {int(np.isfinite(disc).sum()):,} peaks "
      f"(analysis set {int(ROW.sum()):,}); NA elsewhere: "
      f"{'OK' if np.isfinite(disc).sum() == ROW.sum() else 'MISMATCH'}")
ok.append(int(np.isfinite(disc).sum()) == int(ROW.sum()))

def midrank(s):
    o = np.argsort(s, kind="mergesort"); sv = s[o]; n = len(s); rk = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    rr = np.empty(n); rr[o] = rk
    return rr
def auroc(yv, s):
    mm = np.isfinite(s); yv = yv[mm]; rr = midrank(s[mm]); P = yv.sum(); n = len(yv)
    return float((rr[yv == 1].sum() - P * (P + 1) / 2) / (P * (n - P)))
GEN = [(g, ix[ROW[ix]]) for g, ix in gi.items() if ROW[ix].sum() >= 2 and (y[ix[ROW[ix]]]).sum() == 1]
HG = [(g, ix) for g, ix in GEN if is_hold[ix[0]]]
HR = np.concatenate([ix for _, ix in HG])
wr = []
for g, ix in HG:
    s = disc[ix]; k = int(np.argmax(y[ix])); d = np.delete(s, k); d = d[np.isfinite(d)]
    if np.isfinite(s[k]) and len(d): wr.append(np.mean((s[k] > d) + 0.5 * (s[k] == d)))
a_h = auroc(yf[HR], disc[HR]); w_h = float(np.mean(wr))
print(f"\nreproduces the headline on the {len(HG):,} holdout genes: "
      f"AUROC {a_h:.4f} (figure 0.6486), win-rate {w_h:.4f} (figure 0.6450)")
ok.append(abs(a_h - 0.6486) < 5e-4 and abs(w_h - 0.6450) < 5e-4)
print(f"\nALL CHECKS {'PASS' if all(ok) else 'FAIL'}")
assert all(ok), "verification failed; nothing written"

T = pd.DataFrame({
    "gene": gene, "chrom": chrom, "center": cen, "strand": strand,
    "is_canonical": y,
    "flag_apa_annotated": APA.astype(bool),
    "flag_reference_terminal_exon": TERMEX.astype(bool),
    "flag_downstream_of_canonical": DOWN,
    "in_analysis_set": ROW.astype(bool),
    "spl_upstream_max_me3": me3,
    "gc_contrast": gcc,
    "discriminant": disc,
    "loco_chromosome": np.where(ROW, chrom, None),
    "in_gene_holdout": is_hold,
})
T.columns = [str(c) for c in T.columns]
T.to_parquet(OUT, engine="fastparquet", compression="snappy", index=False)
print(f"\nwrote {OUT}: {T.shape[0]:,} rows x {T.shape[1]} columns")
if os.path.exists(OLD_FILE):
    os.rename(OLD_FILE, OLD_MARKED)
    print(f"renamed {OLD_FILE} -> {OLD_MARKED}")
log("DONE_BUILD_FINAL_PARQUET")
