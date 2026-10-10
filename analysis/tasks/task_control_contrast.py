#!/usr/bin/env python3
# the control the protection panel failed, applied to the model's own u1 feature
import csv, os, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, HOLD_FRAC, RIDGE, NBOOT, NPERM = 42, 0.20, 1.0, 500, 1000
rng = np.random.default_rng(SEED)
FASTA = "../train/data/hg38.fa"
U1_LU = "../train/splice_train/data/u1_lookup.npy"
ME5_LU = "../train/splice_train/data/me5_lookup.npy"
UP, DN = 2000, 523
ACC_END = 1874
SPLIT = 437
NSEED = 3
CACHE = os.environ.get("CACHE", "control_contrast_scan.npz")

M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
ROW = np.load("upstream_rowmask.npz")["ROW"]
me3 = D["spl_upstream_max_me3"].to_numpy(float)
uprox = D["spl_pas_prox_u1"].to_numpy(float)
u1c = (6 * D["spl_upstream_mean_u1"].to_numpy(float) - uprox) / 5.0 - uprox
N = len(D)

SCORERS = ["u1"] + [f"perm{k}" for k in range(NSEED)] + ["rc", "me5", "gc"]
NAMES = [f"{s}_{w}" for s in SCORERS for w in ("distal", "proximal")]

if not os.path.exists(CACHE):
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
    def code(seq): return _MAP[np.frombuffer(seq.encode("ascii"), np.uint8)]
    def lookup_array(path, neg):
        d = np.load(path, allow_pickle=True); d = d.item() if d.dtype == object else dict(d)
        A = np.full(4 ** 9, np.nan); p = 4 ** np.arange(8, -1, -1)
        for k, v in d.items():
            cc = code(k)
            if (cc < 0).any(): continue
            A[int(cc @ p)] = (-float(v)) if neg else float(v)
        return A
    U1 = lookup_array(U1_LU, True); ME5 = lookup_array(ME5_LU, False)
    PERM = []
    for k in range(NSEED):
        r = np.random.default_rng(k); A = U1.copy()
        f = np.where(np.isfinite(A))[0]
        A[f] = A[f][r.permutation(len(f))]
        PERM.append(A)
    pw = 4 ** np.arange(8, -1, -1)
    dig = ((np.arange(4 ** 9)[:, None] // pw[None, :]) % 4)
    RCIDX = ((3 - dig)[:, ::-1] * pw[None, :]).sum(1)
    U1RC = U1[RCIDX]
    log(f"tables: u1 {np.isfinite(U1).sum():,}, me5 {np.isfinite(ME5).sum():,}")

    G = Faidx(FASTA)
    OUT = np.full((N, len(NAMES)), np.nan)
    rows = np.where(ROW)[0]
    if os.environ.get("NROWS"): rows = rows[:int(os.environ["NROWS"])]
    dsl = slice(0, UP - SPLIT); psl = slice(UP - SPLIT, ACC_END + 1)
    for n, i in enumerate(rows):
        cen = int(D.center.iloc[i]); st = D.strand.iloc[i]
        a, b = (cen - UP, cen + DN) if st == "+" else (cen - DN, cen + UP)
        q = G.fetch(D.chrom.iloc[i], a, b, st)
        if q is None or len(q) != UP + DN: continue
        c = code(q); nn = ACC_END + 1
        w = np.lib.stride_tricks.sliding_window_view(c, 9)[:nn]
        h = w @ pw; bad = (w < 0).any(1); h[bad] = -1
        ok = h >= 0
        gt = np.zeros(nn, bool); gt[ok] = (c[3:3 + nn][ok] == 2) & (c[4:4 + nn][ok] == 3)
        j = 0
        for tab, mask in ([(U1, ok)] + [(P, ok) for P in PERM] + [(U1RC, ok), (ME5, ok & gt)]):
            v = np.full(nn, np.nan); v[mask] = tab[h[mask]]
            for sl in (dsl, psl):
                seg = v[sl]; f = np.isfinite(seg)
                OUT[i, j] = seg[f].mean() if f.any() else np.nan; j += 1
        isgc = (c == 1) | (c == 2); isn = c < 0
        for sl in (dsl, psl):
            d_ = (~isn[sl]).sum()
            OUT[i, j] = isgc[sl].sum() / d_ if d_ else np.nan; j += 1
        if (n + 1) % 10000 == 0: log(f"  {n+1:,}/{len(rows):,}")
    np.savez_compressed(CACHE, X=OUT.astype(np.float64), names=np.array(NAMES))
    log("scan written")

Z = np.load(CACHE, allow_pickle=True)
S = Z["X"]; SN = list(Z["names"])
cc = lambda s: S[:, SN.index(f"{s}_distal")] - S[:, SN.index(f"{s}_proximal")]
CONTRASTS = [("u1_contrast (real U1 table)", cc("u1"))] + \
            [(f"permuted U1 table, seed {k}", cc(f"perm{k}")) for k in range(NSEED)] + \
            [("U1 table on reverse complement", cc("rc")),
             ("MaxEnt 5'ss over the same windows", cc("me5")),
             ("GC fraction, same windows", cc("gc"))]

gi = defaultdict(list)
for i in range(N): gi[gene[i]].append(i)
gi = {g: np.array(v) for g, v in gi.items()}
allg = np.array(sorted(set(gene)))
hold_g = set(rng.choice(allg, size=int(round(HOLD_FRAC * len(allg))), replace=False))
is_hold = np.array([g in hold_g for g in gene])
CHROMS = sorted(set(chrom))
GEN = [(g, ix[ROW[ix]]) for g, ix in gi.items() if ROW[ix].sum() >= 2 and (y[ix[ROW[ix]]] == 1).sum() == 1]
HG = [(g, ix) for g, ix in GEN if is_hold[ix[0]]]
IDX = np.where(ROW)[0]; HROWS = np.concatenate([ix for _, ix in HG])

def midrank(s):
    o = np.argsort(s, kind="mergesort"); sv = s[o]; n = len(s); rk = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    rr = np.empty(n); rr[o] = rk
    return rr
def auroc(yv, s):
    m = np.isfinite(s); yv = yv[m]; rr = midrank(s[m]); P = yv.sum(); n = len(yv)
    return float((rr[yv == 1].sum() - P * (P + 1) / 2) / (P * (n - P)))
def null_band(score, G):
    gl = []
    for g, ix in G:
        f = ix[np.isfinite(score[ix])]
        if len(f) >= 2: gl.append(f)
    rows = np.concatenate(gl); rr = midrank(score[rows])
    Gn = len(gl); n = len(rows)
    ng = np.array([len(ix) for ix in gl]); off = np.r_[0, np.cumsum(ng)[:-1]]
    pick = off[None, :] + (rng.random((NPERM, Gn)) * ng[None, :]).astype(int)
    perm = (rr[pick].sum(1) - Gn * (Gn + 1) / 2) / (Gn * (n - Gn))
    mr = np.array([rr[off[k]:off[k] + ng[k]].mean() for k in range(Gn)])
    return (float((mr.mean() - (Gn + 1) / 2) / (n - Gn)),
            float(np.percentile(perm, 2.5)), float(np.percentile(perm, 97.5)))
def winrate(v, G):
    out = np.full(len(G), np.nan)
    for t, (g, ix) in enumerate(G):
        s = v[ix]; c = int(np.argmax(y[ix])); o = np.delete(np.arange(len(ix)), c)
        if not np.isfinite(s[c]): continue
        d = s[o][np.isfinite(s[o])]
        if len(d): out[t] = float(np.mean((s[c] > d) + 0.5 * (s[c] == d)))
    return out
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
def loco_hold(X):
    pred = np.full(N, np.nan)
    for c in CHROMS:
        te = ROW & is_hold & (chrom == c); tr = ROW & ~is_hold & (chrom != c)
        if te.sum() == 0 or tr.sum() == 0: continue
        A, B = prep(X[tr], X[te]); pred[te] = B @ fit_ridge(A, y[tr])
    return pred

print("\n===== VALIDATION =====")
m = ROW & np.isfinite(cc("u1")) & np.isfinite(u1c)
print(f"rescanned u1_contrast vs the deployed feature: r = {np.corrcoef(cc('u1')[m], u1c[m])[0,1]:.6f}, "
      f"max abs diff {np.max(np.abs(cc('u1')[m]-u1c[m])):.6f}")

print(f"\n===== MARGINAL, all {int(ROW.sum()):,} sites, {len(GEN):,} genes ({NPERM} permutations) =====")
print(f"{'contrast':<38} {'AUROC':>7} {'null':>7} {'excess':>8} {'win-rate':>9} {'r vs u1':>8}")
fh = open("control_contrast.tsv", "w")
fh.write("kind\tname\tauroc\tnull\tnull_lo\tnull_hi\texcess\twinrate\twin_lo\twin_hi\tr_vs_u1\n")
gl = [ix for _, ix in GEN]
for nm, v in CONTRASTS:
    a = auroc(y[IDX], v[IDX]); nu, lo, hi = null_band(v, GEN)
    w = winrate(v, GEN); wm = float(np.nanmean(w))
    wb = np.array([np.nanmean(w[rng.integers(0, len(w), len(w))]) for _ in range(300)])
    mm = np.isfinite(v[IDX]) & np.isfinite(u1c[IDX])
    r = float(np.corrcoef(v[IDX][mm], u1c[IDX][mm])[0, 1])
    print(f"{nm:<38} {a:>7.4f} {nu:>7.4f} {a-nu:>+8.4f} {wm:>9.4f} {r:>+8.3f}")
    fh.write(f"marginal\t{nm}\t{a:.5f}\t{nu:.5f}\t{lo:.5f}\t{hi:.5f}\t{a-nu:.5f}\t{wm:.5f}\t"
             f"{np.percentile(wb,2.5):.5f}\t{np.percentile(wb,97.5):.5f}\t{r:.4f}\n")

print(f"\n===== MODEL: acceptor + each contrast, holdout {len(HG):,} genes =====")
print(f"{'model':<44} {'AUROC':>7} {'null':>7} {'excess':>8} {'win-rate':>9} {'95% CI':>17}")
gidx = [ix for _, ix in HG]
for nm, v in [("acceptor alone", None)] + CONTRASTS:
    X = me3[:, None] if v is None else np.column_stack([me3, v])
    p = loco_hold(X)
    a = auroc(y[HROWS], p[HROWS]); nu, lo, hi = null_band(p, HG)
    w = winrate(p, HG); wm = float(np.nanmean(w))
    wb = np.array([np.nanmean(w[rng.integers(0, len(gidx), len(gidx))]) for _ in range(NBOOT)])
    lab = "acceptor alone" if v is None else f"acceptor + {nm.split(' (')[0]}"
    print(f"{lab:<44} {a:>7.4f} {nu:>7.4f} {a-nu:>+8.4f} {wm:>9.4f} "
          f"[{np.percentile(wb,2.5):.4f},{np.percentile(wb,97.5):.4f}]")
    fh.write(f"model\t{lab}\t{a:.5f}\t{nu:.5f}\t{lo:.5f}\t{hi:.5f}\t{a-nu:.5f}\t{wm:.5f}\t"
             f"{np.percentile(wb,2.5):.5f}\t{np.percentile(wb,97.5):.5f}\t\n")

print("\n===== PARTITION: each contrast with the other removed =====")
def resid(a, b):
    out = np.full(N, np.nan)
    m = ROW & np.isfinite(a) & np.isfinite(b)
    A = np.column_stack([np.ones(int(m.sum())), b[m]])
    out[m] = a[m] - A @ np.linalg.lstsq(A, a[m], rcond=None)[0]
    return out
U1V = cc("u1"); GCV = cc("gc")
mm = ROW & np.isfinite(U1V) & np.isfinite(GCV)
print(f"r(u1_contrast, gc_contrast) = {np.corrcoef(U1V[mm], GCV[mm])[0,1]:+.4f}")
PART = [("u1_contrast, gc_contrast removed", resid(U1V, GCV)),
        ("gc_contrast, u1_contrast removed", resid(GCV, U1V))]
print(f"{'contrast':<38} {'AUROC':>7} {'null':>7} {'excess':>8} {'win-rate':>9}")
with open("control_contrast.tsv", "a") as fh:
    for nm, v in PART:
        a = auroc(y[IDX], v[IDX]); nu, lo, hi = null_band(v, GEN)
        w = winrate(v, GEN); wm = float(np.nanmean(w))
        wb = np.array([np.nanmean(w[rng.integers(0, len(w), len(w))]) for _ in range(300)])
        print(f"{nm:<38} {a:>7.4f} {nu:>7.4f} {a-nu:>+8.4f} {wm:>9.4f}")
        fh.write(f"partition\t{nm}\t{a:.5f}\t{nu:.5f}\t{lo:.5f}\t{hi:.5f}\t{a-nu:.5f}\t{wm:.5f}\t"
                 f"{np.percentile(wb,2.5):.5f}\t{np.percentile(wb,97.5):.5f}\t\n")
    print(f"\n{'model':<44} {'AUROC':>7} {'null':>7} {'excess':>8} {'win-rate':>9} {'95% CI':>17}")
    MODS = [("acceptor + u1_contrast, GC gradient removed", np.column_stack([me3, PART[0][1]])),
            ("acceptor + gc_contrast, u1 removed", np.column_stack([me3, PART[1][1]])),
            ("acceptor + both contrasts", np.column_stack([me3, U1V, GCV]))]
    for nm, X in MODS:
        p = loco_hold(X)
        a = auroc(y[HROWS], p[HROWS]); nu, lo, hi = null_band(p, HG)
        w = winrate(p, HG); wm = float(np.nanmean(w))
        wb = np.array([np.nanmean(w[rng.integers(0, len(gidx), len(gidx))]) for _ in range(NBOOT)])
        print(f"{nm:<44} {a:>7.4f} {nu:>7.4f} {a-nu:>+8.4f} {wm:>9.4f} "
              f"[{np.percentile(wb,2.5):.4f},{np.percentile(wb,97.5):.4f}]")
        fh.write(f"model\t{nm}\t{a:.5f}\t{nu:.5f}\t{lo:.5f}\t{hi:.5f}\t{a-nu:.5f}\t{wm:.5f}\t"
                 f"{np.percentile(wb,2.5):.5f}\t{np.percentile(wb,97.5):.5f}\t\n")
log("DONE_CONTROL_CONTRAST")
