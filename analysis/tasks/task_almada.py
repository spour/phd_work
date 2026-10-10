#!/usr/bin/env python3
# almada 2013 telescripting: does a strong upstream u1/5'ss protect a pas?
import csv, os, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, HOLD_FRAC, RIDGE, NBOOT, NPERM = 42, 0.20, 1.0, 400, 500
rng = np.random.default_rng(SEED)
FASTA = "../train/data/hg38.fa"; ME5_LU = "../train/splice_train/data/me5_lookup.npy"
SPAN, PAD = 10000, 30
THR = {"8.46 (this project)": 8.46, "8.77 (Almada median donor)": 8.77}
WS = [100, 200, 500, 1000, 2000]

M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
ROW = np.load("upstream_rowmask.npz")["ROW"]
uprox = D["spl_pas_prox_u1"].to_numpy(float)
udist = (6 * D["spl_upstream_mean_u1"].to_numpy(float) - uprox) / 5.0
u1c = udist - uprox
me3 = D["spl_upstream_max_me3"].to_numpy(float)

# scan
if not os.path.exists("almada_scan.npz"):
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
    d = np.load(ME5_LU, allow_pickle=True); d = d.item() if d.dtype == object else dict(d)
    ME5 = np.full(4 ** 9, np.nan); pw = 4 ** np.arange(8, -1, -1)
    for k, v in d.items():
        cc = code(k)
        if (cc < 0).any(): continue
        ME5[int(cc @ pw)] = float(v)
    G = Faidx(FASTA); L = SPAN + PAD
    KEYS = list(THR)
    OUT = np.full((len(D), 2 * len(KEYS)), np.nan)
    idx_rows = np.where(ROW)[0]
    for n, i in enumerate(idx_rows):
        cen = int(D.center.iloc[i]); st = D.strand.iloc[i]
        a, b = (cen - L, cen + L) if st == "+" else (cen - L + 1, cen + L + 1)
        q = G.fetch(D.chrom.iloc[i], a, b, st)
        if q is None or len(q) != 2 * L: continue
        c = code(q); S = L
        nn = len(c) - 8
        w = np.lib.stride_tricks.sliding_window_view(c, 9)[:nn]
        h = w @ (4 ** np.arange(8, -1, -1)); h[(w < 0).any(1)] = -1
        gt = (c[3:3 + nn] == 2) & (c[4:4 + nn] == 3) & (h >= 0)
        pos = np.where(gt)[0]
        if pos.size == 0: continue
        sc = ME5[h[pos]]
        dpos = pos + 3
        for k, key in enumerate(KEYS):
            ok = np.isfinite(sc) & (sc >= THR[key])
            p = dpos[ok]
            up = p[p < S]; dn = p[p > S]
            if up.size: OUT[i, 2 * k] = S - up[-1]
            if dn.size: OUT[i, 2 * k + 1] = dn[0] - S
        if (n + 1) % 10000 == 0: log(f"  site {n+1:,}/{len(idx_rows):,}")
    np.savez_compressed("almada_scan.npz", X=OUT.astype(np.float32), keys=np.array(list(THR)))
    log("scan cached")
Z = np.load("almada_scan.npz"); SC = Z["X"].astype(float); KEYS = [str(s) for s in Z["keys"]]

gi = defaultdict(list)
for i in range(len(gene)): gi[gene[i]].append(i)
gi = {g: np.array(v) for g, v in gi.items()}
GEN = [(g, ix[ROW[ix]]) for g, ix in gi.items() if ROW[ix].sum() >= 2 and (y[ix[ROW[ix]]] == 1).sum() == 1]
IDX = np.where(ROW)[0]
allg = np.array(sorted(set(gene)))
hold_g = set(rng.choice(allg, size=int(round(HOLD_FRAC * len(allg))), replace=False))
is_hold = np.array([g in hold_g for g in gene])
CHROMS = sorted(set(chrom))
HG = [(g, ix) for g, ix in GEN if is_hold[ix[0]]]

def auroc(yv, s):
    m = np.isfinite(s); yv = yv[m]; s = s[m]
    o = np.argsort(s, kind="mergesort"); sv = s[o]; n = len(s); rk = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    rr = np.empty(n); rr[o] = rk; npos = yv.sum(); nneg = n - npos
    return float((rr[yv == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def winrates(v):
    out = np.full(len(GEN), np.nan)
    for t, (g, ix) in enumerate(GEN):
        s = v[ix]; c = int(np.argmax(y[ix])); o = np.delete(np.arange(len(ix)), c)
        if not np.isfinite(s[c]): continue
        dd = s[o][np.isfinite(s[o])]
        if len(dd): out[t] = float(np.mean((s[c] > dd) + 0.5 * (s[c] == dd)))
    return out
def perm_win(v, nperm=NPERM):
    vals = [v[ix][np.isfinite(v[ix])] for g, ix in GEN]
    vals = [s for s in vals if len(s) >= 2]
    out = np.empty(nperm)
    for t in range(nperm):
        w = []
        for s in vals:
            c = rng.integers(0, len(s)); o = np.delete(np.arange(len(s)), c)
            w.append(np.mean((s[c] > s[o]) + 0.5 * (s[c] == s[o])))
        out[t] = np.mean(w)
    return float(out.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
BOOTG = rng.integers(0, len(GEN), size=(NBOOT, len(GEN)))
def wstats(v):
    w = winrates(v); wm = float(np.nanmean(w))
    b = np.nanmean(w[BOOTG], axis=1)
    return wm, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

print(f"\n===== ALMADA PROTECTION WINDOW, current set: {len(GEN)} genes, "
      f"{int((ROW&(y==0)).sum())} competitors, {int((ROW&(y==1)).sum())} canonical =====")
for k, key in enumerate(KEYS):
    du = SC[:, 2 * k]; dd = SC[:, 2 * k + 1]
    print(f"\n--- strong donor cutoff {key} ---")
    print(f"  censored (no strong donor within {SPAN} nt): upstream "
          f"{np.mean(~np.isfinite(du[IDX])):.4f}, downstream {np.mean(~np.isfinite(dd[IDX])):.4f}")
    print(f"  median distance: upstream {np.nanmedian(du[IDX]):.0f} nt, downstream {np.nanmedian(dd[IDX]):.0f} nt")
with open("almada_sweep.tsv", "w") as fh:
    fh.write("cutoff\tdirection\tfeature\twinrate\twin_lo\twin_hi\tperm\tauroc\tfrac_canonical\tfrac_competitor\n")
    for k, key in enumerate(KEYS):
        for dircol, dirname in [(2 * k, "upstream"), (2 * k + 1, "downstream")]:
            dv = SC[:, dircol]
            print(f"\n===== {dirname} strong donor, cutoff {key} =====")
            print(f"{'feature':<34} {'win-rate':>9} {'95% CI':>17} {'perm':>7} {'AUROC':>7} "
                  f"{'canon':>7} {'comp':>7}")
            for W in WS:
                v = np.where(np.isfinite(dv), (dv <= W).astype(float), 0.0)
                wm, lo, hi = wstats(v); pm = perm_win(v, 200)[0]; au = auroc(y[IDX], v[IDX])
                fc_ = float(v[IDX][y[IDX] == 1].mean()); fk = float(v[IDX][y[IDX] == 0].mean())
                print(f"{'protected, W = ' + str(W) + ' nt':<34} {wm:>9.4f} [{lo:.4f},{hi:.4f}] "
                      f"{pm:>7.4f} {au:>7.4f} {fc_:>7.3f} {fk:>7.3f}")
                fh.write(f"{key}\t{dirname}\tprotected_W{W}\t{wm:.5f}\t{lo:.5f}\t{hi:.5f}\t{pm:.5f}\t"
                         f"{au:.5f}\t{fc_:.5f}\t{fk:.5f}\n")
            vc = np.where(np.isfinite(dv), dv, 1e5)
            wm, lo, hi = wstats(vc); pm = perm_win(vc, 200)[0]; au = auroc(y[IDX], vc[IDX])
            print(f"{'distance to nearest (nt)':<34} {wm:>9.4f} [{lo:.4f},{hi:.4f}] {pm:>7.4f} {au:>7.4f}")
            fh.write(f"{key}\t{dirname}\tdistance\t{wm:.5f}\t{lo:.5f}\t{hi:.5f}\t{pm:.5f}\t{au:.5f}\t\t\n")
    wm, lo, hi = wstats(u1c); au = auroc(y[IDX], u1c[IDX])
    print(f"\n===== reference on the same axis =====")
    print(f"{'u1_contrast':<34} {wm:>9.4f} [{lo:.4f},{hi:.4f}] {perm_win(u1c,200)[0]:>7.4f} {au:>7.4f}")
    fh.write(f"-\t-\tu1_contrast\t{wm:.5f}\t{lo:.5f}\t{hi:.5f}\t\t{au:.5f}\t\t\n")
    wm, lo, hi = wstats(me3); au = auroc(y[IDX], me3[IDX])
    print(f"{'spl_upstream_max_me3':<34} {wm:>9.4f} [{lo:.4f},{hi:.4f}] {'':>7} {au:>7.4f}")
    fh.write(f"-\t-\tspl_upstream_max_me3\t{wm:.5f}\t{lo:.5f}\t{hi:.5f}\t\t{au:.5f}\t\t\n")

# does either add to the two-feature model
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
    pred = np.full(len(y), np.nan)
    for c in CHROMS:
        te = ROW & is_hold & (chrom == c); tr = ROW & ~is_hold & (chrom != c)
        if te.sum() == 0 or tr.sum() == 0: continue
        A, B = prep(X[tr], X[te]); pred[te] = B @ fit_ridge(A, y[tr])
    return pred
rowsH = np.concatenate([ix for _, ix in HG]); gidxH = [ix for _, ix in HG]
BH = rng.integers(0, len(gidxH), size=(300, len(gidxH)))
def hold_auroc(pred):
    a = auroc(y[rowsH], pred[rowsH])
    b = np.array([auroc(y[np.concatenate([gidxH[j] for j in BH[t]])],
                        pred[np.concatenate([gidxH[j] for j in BH[t]])]) for t in range(300)])
    return a, b
k0 = 0
du = SC[:, 0]; dd = SC[:, 1]
BASE = np.column_stack([me3, u1c])
ADD = {"+ protected upstream, W = 500": np.where(np.isfinite(du), (du <= 500).astype(float), 0.0),
       "+ protected upstream, W = 2000": np.where(np.isfinite(du), (du <= 2000).astype(float), 0.0),
       "+ distance to upstream donor": np.where(np.isfinite(du), du, 1e5),
       "+ protected downstream, W = 500": np.where(np.isfinite(dd), (dd <= 500).astype(float), 0.0),
       "+ distance to downstream donor": np.where(np.isfinite(dd), dd, 1e5)}
a0, b0 = hold_auroc(loco_hold(BASE))
print(f"\n===== added to the two-feature model (holdout {len(HG)} genes, cutoff 8.46) =====")
print(f"{'model':<36} {'AUROC':>7} {'dAUROC':>8} {'95% CI':>19}")
print(f"{'two-feature model':<36} {a0:>7.4f}")
with open("almada_sweep.tsv", "a") as fh:
    for nm, v in ADD.items():
        a1, b1 = hold_auroc(loco_hold(np.column_stack([BASE, v])))
        d = b1 - b0
        print(f"{nm:<36} {a1:>7.4f} {a1-a0:>+8.4f} [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]")
        fh.write(f"#delta\t{nm}\t{a1:.5f}\t{a1-a0:.5f}\t{np.percentile(d,2.5):.5f}\t{np.percentile(d,97.5):.5f}\n")
for nm, X in [("acceptor + protected upstream W=500", np.column_stack([me3, ADD["+ protected upstream, W = 500"]])),
              ("acceptor + distance to upstream donor", np.column_stack([me3, ADD["+ distance to upstream donor"]])),
              ("acceptor + u1_contrast (current model)", BASE)]:
    a, _ = hold_auroc(loco_hold(X))
    print(f"{nm:<44} holdout AUROC {a:.4f}")
log("DONE_ALMADA")
