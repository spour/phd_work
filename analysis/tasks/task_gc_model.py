#!/usr/bin/env python3
# feature 2 becomes gc_contrast
import csv, os, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, HOLD_FRAC, RIDGE, NBOOT, NPERM = 42, 0.20, 1.0, 500, 1000
rng = np.random.default_rng(SEED)
FASTA = "../train/data/hg38.fa"
PAS_STRONG = ("AATAAA", "ATTAAA"); PAS_LO, PAS_HI = -60, 0

M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
rf = D.rf_prob.to_numpy(float)
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
ROW = np.load("upstream_rowmask.npz")["ROW"]
N = len(D)
me3 = D["spl_upstream_max_me3"].to_numpy(float)
uprox = D["spl_pas_prox_u1"].to_numpy(float)
u1c = (6 * D["spl_upstream_mean_u1"].to_numpy(float) - uprox) / 5.0 - uprox
CS = np.load("control_contrast_scan.npz", allow_pickle=True)
CSN = list(CS["names"])
gcc = CS["X"][:, CSN.index("gc_distal")] - CS["X"][:, CSN.index("gc_proximal")]
X2 = np.column_stack([me3, gcc])
NAMES = ["spl_upstream_max_me3", "gc_contrast"]
PB = np.load("binned_features178.npz"); BN = [str(s) for s in PB["names"]]; BX = PB["X"][ridx].astype(float)
GCB = [BN.index(f"bin{b}_gc") for b in range(1, 26)]
SPL = [i for i, n in enumerate(BN) if n.startswith("bin") and not n.endswith("_gc")]

# hexamer
if not os.path.exists("hexamer_cache.npz"):
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
    G = Faidx(FASTA); W = PAS_HI - PAS_LO
    hexa = np.full(N, np.nan)
    for i in np.where(ROW)[0]:
        cen = int(D.center.iloc[i]); st = D.strand.iloc[i]
        a, b = (cen + PAS_LO, cen + PAS_HI) if st == "+" else (cen - PAS_HI, cen - PAS_LO)
        q = G.fetch(D.chrom.iloc[i], a, b, st)
        if q is None or len(q) != W: continue
        hexa[i] = float(any(h in q for h in PAS_STRONG))
    np.savez_compressed("hexamer_cache.npz", hexa=hexa)
    log("hexamer scan written")
hexa = np.load("hexamer_cache.npz")["hexa"]

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
def auprc(yv, s):
    o = np.argsort(-s, kind="mergesort"); yy = yv[o]; sv = s[o]
    tp = np.cumsum(yy); k = np.arange(1, len(yy) + 1)
    idx = np.where(np.r_[sv[1:] != sv[:-1], True])[0]
    prec = (tp[idx] / k[idx])[np.searchsorted(idx, np.arange(len(yy)))]
    return float((prec * yy).sum() / yy.sum())
def null_band(score, G):
    gl = []
    for g, ix in G:
        f = ix[np.isfinite(score[ix])]
        if len(f) >= 2 and (y[f] == 1).sum() == 1: gl.append(f)
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
def loco(X, test_mask=None, train_mask=None):
    tm = np.ones(N, bool) if test_mask is None else test_mask
    rm = np.ones(N, bool) if train_mask is None else train_mask
    pred = np.full(N, np.nan)
    for c in CHROMS:
        te = ROW & tm & (chrom == c); tr = ROW & rm & (chrom != c)
        if te.sum() == 0 or tr.sum() == 0: continue
        A, B = prep(X[tr], X[te]); pred[te] = B @ fit_ridge(A, y[tr])
    return pred

print("\n===== FEATURE 2 = gc_contrast =====")
m = ROW & np.isfinite(gcc) & np.isfinite(u1c)
print(f"r(gc_contrast, u1_contrast) = {np.corrcoef(gcc[m], u1c[m])[0,1]:+.4f}   "
      f"mean gc_contrast: canonical {np.nanmean(gcc[IDX][y[IDX]==1]):+.4f}, "
      f"competitor {np.nanmean(gcc[IDX][y[IDX]==0]):+.4f}")

pred2 = loco(X2); pred2h = loco(X2, is_hold, ~is_hold)
rows_all = IDX; yv = y[rows_all]

# coefficients
dev = np.where(ROW & ~is_hold)[0]
A, _ = prep(X2[dev], X2[dev]); beta = fit_ridge(A, y[dev])[1:]
devg = [ix for g, ix in GEN if not is_hold[ix[0]]]
BB = np.empty((300, 2))
for t in range(300):
    ii = np.concatenate([devg[j] for j in rng.integers(0, len(devg), len(devg))])
    Ab, _ = prep(X2[ii], X2[ii]); BB[t] = fit_ridge(Ab, y[ii])[1:]
with open("gc_coefs.tsv", "w") as fh:
    fh.write("feature\tbeta_std\tci_lo\tci_hi\tsign_flip_frac\tauroc\tnull\texcess\twinrate\n")
    print(f"\n{'feature':<24} {'beta':>8} {'95% CI':>19} {'flip':>5} {'AUROC':>7} {'null':>7} "
          f"{'excess':>8} {'win-rate':>9}")
    for a, nm in enumerate(NAMES):
        lo, hi = np.percentile(BB[:, a], [2.5, 97.5])
        fl = float(np.mean(np.sign(BB[:, a]) != np.sign(beta[a])))
        v = X2[:, a]; au = auroc(y[rows_all], v[rows_all]); nu, _, _ = null_band(v, GEN)
        w = float(np.nanmean(winrate(v, GEN)))
        print(f"{nm:<24} {beta[a]:>+8.4f} [{lo:>+8.4f},{hi:>+8.4f}] {fl:>5.2f} {au:>7.4f} "
              f"{nu:>7.4f} {au-nu:>+8.4f} {w:>9.4f}")
        fh.write(f"{nm}\t{beta[a]:.5f}\t{lo:.5f}\t{hi:.5f}\t{fl:.4f}\t{au:.5f}\t{nu:.5f}\t"
                 f"{au-nu:.5f}\t{w:.5f}\n")

# curves
def curve(sv, tag, fh):
    o = rows_all[np.argsort(-sv[rows_all], kind="mergesort")]; yy = y[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy); P = yy.sum(); Nn = len(yy) - P
    for i in np.unique(np.linspace(0, len(yy) - 1, 400).astype(int)):
        fh.write(f"{tag}\t{fp[i]/Nn:.6f}\t{tp[i]/P:.6f}\t{tp[i]/(i+1):.6f}\n")
nu_all, nlo_all, nhi_all = null_band(pred2, GEN)
w_all = winrate(pred2, GEN)
with open("gc_curves.tsv", "w") as fh:
    fh.write("model\tfpr\ttpr\tprecision\n")
    curve(pred2, "two-feature model", fh)
    for k, v in [("base_rate", float(yv.mean())), ("auroc_model", auroc(yv, pred2[rows_all])),
                 ("auprc_model", auprc(yv, pred2[rows_all])), ("null_model", nu_all),
                 ("null_lo", nlo_all), ("null_hi", nhi_all),
                 ("excess_model", auroc(yv, pred2[rows_all]) - nu_all),
                 ("winrate_model", float(np.nanmean(w_all))),
                 ("auroc_rf", auroc(yv, rf[rows_all])), ("auprc_rf", auprc(yv, rf[rows_all])),
                 ("null_rf", null_band(rf, GEN)[0])]:
        fh.write(f"#{k}\t{v:.5f}\t\t\n")
print(f"\nall rows: AUROC {auroc(yv, pred2[rows_all]):.4f} null {nu_all:.4f} "
      f"excess {auroc(yv, pred2[rows_all])-nu_all:+.4f}  win-rate {np.nanmean(w_all):.4f}")

# per chromosome
with open("gc_bychrom.tsv", "w") as fh:
    fh.write("chrom\tn_genes\tauroc\tci_lo\tci_hi\tnull\tnull_lo\tnull_hi\texcess\n")
    for c in CHROMS:
        G = [(g, ix) for g, ix in GEN if chrom[ix[0]] == c]
        if len(G) < 20: continue
        gl = [ix for _, ix in G]; rws = np.concatenate(gl)
        a = auroc(y[rws], pred2[rws]); nu, lo, hi = null_band(pred2, G)
        b = np.empty(200)
        for t in range(200):
            ii = np.concatenate([gl[j] for j in rng.integers(0, len(gl), len(gl))])
            b[t] = auroc(y[ii], pred2[ii])
        fh.write(f"{c}\t{len(G)}\t{a:.5f}\t{np.percentile(b,2.5):.5f}\t{np.percentile(b,97.5):.5f}\t"
                 f"{nu:.5f}\t{lo:.5f}\t{hi:.5f}\t{a-nu:.5f}\n")

BLOCKS = [("acceptor alone (1)", me3[:, None]),
          ("acceptor + gc_contrast (2)", X2),
          ("GC-binned (25)", BX[:, GCB]),
          ("splice-binned (150)", BX[:, SPL]),
          ("union (177)", np.column_stack([X2, BX[:, SPL], BX[:, GCB]]))]
gidxH = [ix for _, ix in HG]
print(f"\n===== PARTITION, holdout {len(HG)} genes, base rate {float(y[HROWS].mean()):.4f} =====")
print(f"{'block':<28} {'AUROC':>7} {'95% CI':>17} {'null':>7} {'excess':>8} {'win-rate':>9} {'95% CI':>17}")
with open("gc_blocks.tsv", "w") as fh:
    fh.write("block\tn_feat\tauroc\tauroc_lo\tauroc_hi\tnull\tnull_lo\tnull_hi\texcess\tauprc\t"
             "base_rate\twinrate\twin_lo\twin_hi\n")
    for nm, X in BLOCKS:
        p = loco(X, is_hold, ~is_hold)
        a = auroc(y[HROWS], p[HROWS]); ap = auprc(y[HROWS], p[HROWS])
        nu, lo, hi = null_band(p, HG)
        w = winrate(p, HG); wm = float(np.nanmean(w))
        ab = np.empty(NBOOT); wb = np.empty(NBOOT)
        for t in range(NBOOT):
            sel = rng.integers(0, len(gidxH), len(gidxH))
            ii = np.concatenate([gidxH[j] for j in sel])
            ab[t] = auroc(y[ii], p[ii]); wb[t] = np.nanmean(w[sel])
        print(f"{nm:<28} {a:>7.4f} [{np.percentile(ab,2.5):.4f},{np.percentile(ab,97.5):.4f}] "
              f"{nu:>7.4f} {a-nu:>+8.4f} {wm:>9.4f} [{np.percentile(wb,2.5):.4f},{np.percentile(wb,97.5):.4f}]")
        fh.write(f"{nm}\t{X.shape[1]}\t{a:.5f}\t{np.percentile(ab,2.5):.5f}\t{np.percentile(ab,97.5):.5f}\t"
                 f"{nu:.5f}\t{lo:.5f}\t{hi:.5f}\t{a-nu:.5f}\t{ap:.5f}\t{float(y[HROWS].mean()):.5f}\t"
                 f"{wm:.5f}\t{np.percentile(wb,2.5):.5f}\t{np.percentile(wb,97.5):.5f}\n")

print("\n===== GC PROFILE =====")
gl_all = [ix for _, ix in GEN]
BOOTG = rng.integers(0, len(gl_all), size=(200, len(gl_all)))
with open("gc_profile.tsv", "w") as fh:
    fh.write("class\tposition\tvalue\tci_lo\tci_hi\n")
    for cl, msk in [("canonical", y == 1), ("competitor", y == 0)]:
        for b in range(25):
            v = BX[:, GCB[b]]
            sel = ROW & msk & np.isfinite(v)
            pos = -2000 + 100 * b + 50
            bs = np.empty(200)
            for t in range(200):
                ii = np.concatenate([gl_all[j] for j in BOOTG[t]])
                ii = ii[msk[ii] & np.isfinite(v[ii])]
                bs[t] = v[ii].mean() if len(ii) else np.nan
            fh.write(f"{cl}\t{pos}\t{v[sel].mean():.5f}\t{np.nanpercentile(bs,2.5):.5f}\t"
                     f"{np.nanpercentile(bs,97.5):.5f}\n")

Q = np.load("decoy_features.npz"); QX = Q["X"].astype(float); QN = [str(s) for s in Q["names"]]
DEC = list(csv.DictReader(open("decoy_positions.tsv"), delimiter="\t"))
dgene = np.array([r["gene"] for r in DEC])
DGCB = [QN.index(f"bin{b}_gc") for b in range(1, 26)]
PGCB = [BN.index(f"bin{b}_gc") for b in range(1, 26)]
def approx_gcc(Xb, cols):
    return np.nanmean(Xb[:, [cols[b] for b in range(0, 16)]], 1) - \
           np.nanmean(Xb[:, [cols[b] for b in range(16, 19)]], 1)
gcc_ap = approx_gcc(BX, PGCB); gcc_dec = approx_gcc(QX, DGCB)
me3_dec = QX[:, QN.index("spl_upstream_max_me3")]
mm = ROW & np.isfinite(gcc_ap) & np.isfinite(gcc)
print(f"\n===== DECOY TRANSFER =====")
print(f"100 nt bin approximation of gc_contrast vs exact: r = {np.corrcoef(gcc_ap[mm], gcc[mm])[0,1]:.4f}")
gidD = defaultdict(lambda: ([], []))
for i in IDX: gidD[gene[i]][0].append(i)
for j in range(len(DEC)): gidD[dgene[j]][1].append(j)
GK = [g for g in gidD if len(gidD[g][0]) and len(gidD[g][1])]
PIX = {g: np.array(gidD[g][0]) for g in GK}; DIX = {g: np.array(gidD[g][1]) for g in GK}
dchrom = np.array([r["chrom"] for r in DEC])
BOOTD = rng.integers(0, len(GK), size=(300, len(GK)))
with open("gc_decoy.tsv", "w") as fh:
    fh.write("block\tn_feat\tcontrast\tgap\tci_lo\tci_hi\n")
    for nm, Xp, Xd in [("acceptor + gc_contrast (2)",
                        np.column_stack([me3, gcc_ap]), np.column_stack([me3_dec, gcc_dec])),
                       ("GC-binned (25)", BX[:, PGCB], QX[:, DGCB])]:
        sp = np.full(N, np.nan); sd = np.full(len(DEC), np.nan)
        for c in CHROMS:
            tr = ROW & (chrom != c)
            if tr.sum() == 0: continue
            A0, _ = prep(Xp[tr], Xp[tr]); b0 = fit_ridge(A0, y[tr])
            mu = np.nanmean(Xp[tr], 0); s0 = np.nanstd(Xp[tr], 0); s0[s0 == 0] = 1.0
            mu = np.where(np.isfinite(mu), mu, 0.0)
            tp_ = ROW & (chrom == c)
            if tp_.sum():
                Z_ = (np.where(np.isfinite(Xp[tp_]), Xp[tp_], mu) - mu) / s0
                sp[tp_] = np.column_stack([np.ones(int(tp_.sum())), Z_]) @ b0
            td = dchrom == c
            if td.sum():
                Z_ = (np.where(np.isfinite(Xd[td]), Xd[td], mu) - mu) / s0
                sd[td] = np.column_stack([np.ones(int(td.sum())), Z_]) @ b0
        def gaps(sel):
            can = []; com = []; dec = []
            for k in sel:
                g = GK[k]; pi = PIX[g]; di = DIX[g]
                cc_ = pi[y[pi] == 1]; kk = pi[y[pi] == 0]
                if len(cc_): can.append(np.nanmean(sp[cc_]))
                if len(kk): com.append(np.nanmean(sp[kk]))
                if len(di): dec.append(np.nanmean(sd[di]))
            return np.nanmean(can) - np.nanmean(com), np.nanmean(com) - np.nanmean(dec)
        g1, g2 = gaps(range(len(GK)))
        B1 = np.empty(300); B2 = np.empty(300)
        for t in range(300): B1[t], B2[t] = gaps(BOOTD[t])
        for lab, gv, bv in [("canonical - competitor", g1, B1), ("competitor - intronic decoy", g2, B2)]:
            print(f"  {nm:<28} {lab:<28} {gv:+.4f} [{np.percentile(bv,2.5):+.4f},{np.percentile(bv,97.5):+.4f}]")
            fh.write(f"{nm}\t{Xp.shape[1]}\t{lab}\t{gv:.5f}\t{np.percentile(bv,2.5):.5f}\t"
                     f"{np.percentile(bv,97.5):.5f}\n")

print("\n===== HEXAMER TEST =====")
hf = hexa[IDX]
print(f"AATAAA/ATTAAA in [pA-60,pA): canonical {np.nanmean(hf[yv==1]):.4f}, "
      f"competitor {np.nanmean(hf[yv==0]):.4f}  (n = {int(np.isfinite(hf).sum()):,})")
with open("gc_hexamer.tsv", "w") as fh:
    fh.write("stratum\tn_sites\tn_genes\tauroc\tnull\texcess\twinrate\twin_lo\twin_hi\t"
             "mean_can\tmean_comp\thex_can\thex_comp\n")
    for lab, keep in [("all peaks", np.ones(N, bool)),
                      ("hexamer present", hexa == 1), ("hexamer absent", hexa == 0)]:
        sub = ROW & keep & np.isfinite(gcc)
        G = []
        for g, ix in GEN:
            f = ix[sub[ix]]
            if len(f) >= 2 and (y[f] == 1).sum() == 1: G.append((g, f))
        if len(G) < 20: continue
        rws = np.concatenate([ix for _, ix in G])
        a = auroc(y[rws], gcc[rws]); nu, lo, hi = null_band(gcc, G)
        w = winrate(gcc, G); wm = float(np.nanmean(w))
        wb = np.array([np.nanmean(w[rng.integers(0, len(w), len(w))]) for _ in range(300)])
        mc = float(np.nanmean(gcc[rws][y[rws] == 1])); mk = float(np.nanmean(gcc[rws][y[rws] == 0]))
        print(f"{lab:<18} {len(G):>5} genes {len(rws):>7,} sites  AUROC {a:.4f} null {nu:.4f} "
              f"excess {a-nu:+.4f}  win-rate {wm:.4f} [{np.percentile(wb,2.5):.4f},"
              f"{np.percentile(wb,97.5):.4f}]  mean {mc:+.4f} vs {mk:+.4f}")
        fh.write(f"{lab}\t{len(rws)}\t{len(G)}\t{a:.5f}\t{nu:.5f}\t{a-nu:.5f}\t{wm:.5f}\t"
                 f"{np.percentile(wb,2.5):.5f}\t{np.percentile(wb,97.5):.5f}\t{mc:.5f}\t{mk:.5f}\t"
                 f"{float(np.nanmean(hf[yv==1])):.5f}\t{float(np.nanmean(hf[yv==0])):.5f}\n")
    print(f"\n{'group':<34} {'n':>8} {'mean gc_contrast':>18}")
    for cl, cv in [("canonical", 1.0), ("competitor", 0.0)]:
        for hl, hv in [("hexamer present", 1.0), ("hexamer absent", 0.0)]:
            s = ROW & (y == cv) & (hexa == hv) & np.isfinite(gcc)
            print(f"{cl + ', ' + hl:<34} {int(s.sum()):>8,} {float(gcc[s].mean()):>+18.4f}")
            fh.write(f"cell: {cl}, {hl}\t{int(s.sum())}\t\t\t\t\t\t\t\t{float(gcc[s].mean()):.5f}\t\t\t\n")
log("DONE_GC_MODEL")
