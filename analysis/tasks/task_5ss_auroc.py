#!/usr/bin/env python3
# how much predictive power is in panel a of figure_5ss_terminal_exon
import sys, math
import numpy as np
import u1_5ss_landscape_data as L

log = L.log
rng = np.random.default_rng(42)
NPERM = 400

S = L.build_sites(); NS = len(S["cls"])
Z = np.load("u1_5ss_sitescan.npz", allow_pickle=True); F = Z["F"].astype(np.float64)
assert (Z["cls"] == S["cls"]).all() and (Z["center"] == S["center"]).all()
C = np.load("upstream_count_scan.npz")["F"].astype(np.float64)
assert len(C) == NS
WINDOWS = (100, 200, 300, 500, 1000, 2000); NW = len(WINDOWS)
PROF = F[:, L.O_D5_100:L.O_D5_100 + L.N100]
cls = S["cls"]; gene = S["gene"]
log(f"sites {NS}: " + "  ".join(f"{c} {(cls==c).sum()}" for c in L.CLASSES))

# metrics
def auroc(pos, neg):
    a = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); rank = ((cs - cnt) + cs + 1) / 2.0
    r = rank[inv][:len(pos)].sum()
    return (r - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))

def auprc(pos, neg):
    s = np.concatenate([pos, neg]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    o = np.argsort(-s, kind="stable"); y = y[o]; s = s[o]
    end = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    tp = np.cumsum(y)[end]; k = (end + 1).astype(float)
    prec = tp / k; rec = tp / len(pos)
    drec = np.diff(np.r_[0.0, rec])
    return float(np.sum(prec * drec)), float(rec[-1])

def mwu_p(a, n1, n0, ties_arr):
    U = a * n1 * n0; n = n1 + n0
    _, cnt = np.unique(ties_arr, return_counts=True)
    tie = np.sum(cnt ** 3 - cnt)
    var = n1 * n0 / 12.0 * ((n + 1) - tie / (n * (n - 1.0)))
    if var <= 0: return np.nan, np.nan
    z = (U - n1 * n0 / 2.0) / math.sqrt(var)
    return z, math.erfc(abs(z) / math.sqrt(2))

def perm_null(v, ypos, gene_codes, nperm=NPERM):
    idx_by_gene = {}
    for i, g in enumerate(gene_codes): idx_by_gene.setdefault(g, []).append(i)
    groups = [np.array(x) for x in idx_by_gene.values()]
    npos = {id(g): int(ypos[g].sum()) for g in groups}
    out = np.empty(nperm)
    for t in range(nperm):
        yl = np.zeros(len(v), bool)
        for g in groups:
            k = npos[id(g)]
            if k: yl[rng.choice(g, k, replace=False)] = True
        out[t] = auroc(v[yl], v[~yl])
    return out

def report(name, v, m_pos, m_neg, higher_is_real=False):
    s = v if higher_is_real else -v
    pos = s[m_pos]; neg = s[m_neg]
    ok = np.isfinite(pos); pos = pos[ok]; neg = neg[np.isfinite(neg)]
    a = auroc(pos, neg); ap, _ = auprc(pos, neg)
    base = len(pos) / (len(pos) + len(neg))
    z, p = mwu_p(a, len(pos), len(neg), np.concatenate([pos, neg]))
    return dict(name=name, auroc=a, auprc=ap, base=base, lift=ap / base, z=z, p=p,
                n1=len(pos), n0=len(neg))

def fmt(r, null=None):
    s = (f"  {r['name']:<34s} AUROC {r['auroc']:.4f}   AUPRC {r['auprc']:.4f} "
         f"(baseline {r['base']:.4f}, lift {r['lift']:.2f}x)   MWU z {r['z']:+.1f} "
         f"p {r['p']:.2e}")
    if null is not None:
        mu, sd, pe = null
        s += f"\n{'':38s}within-gene null {mu:.4f} +/- {sd:.4f}  ->  excess {r['auroc']-mu:+.4f}, z {(r['auroc']-mu)/sd:+.1f}, perm p {pe:.3g}"
    return s

# scalars
def col_count(w, thr8=True): return C[:, (0 if thr8 else NW) + WINDOWS.index(w)]
def col_max(w): return C[:, 2 * NW + WINDOWS.index(w)]
def dens_win(lo, hi):
    sel = (L.BC100 >= lo) & (L.BC100 < hi)
    return np.nanmean(PROF[:, sel], 1)

SCALARS = [
    ("density [-500,0) (panel A dip)", dens_win(-500, 0), False),
    ("density [-300,0)",               dens_win(-300, 0), False),
    ("density [-100,0) (pA bin)",      dens_win(-100, 0), False),
    ("density [0,+500) (downstream)",  dens_win(0, 500),  False),
    ("count MaxEnt>=8 in [-500,0)",    col_count(500),    False),
    ("count MaxEnt>=8 in [-2000,0)",   col_count(2000),   False),
    ("max MaxEnt5 in [-500,0)",        col_max(500),      False),
    ("ACCEPTOR max me3 [-2000,-126)",  F[:, L.K["me3_max_up"]], True),
]

for other in ("competitor", "decoy"):
    m_pos = cls == "real"; m_neg = cls == other
    sub = m_pos | m_neg
    gsub = gene[sub]; ypos = (cls[sub] == "real")
    print(f"\n=============== real 3' end vs {other} "
          f"({m_pos.sum()} vs {m_neg.sum()}) ===============")
    for nm, v, hi in SCALARS:
        r = report(nm, v, m_pos, m_neg, hi)
        nul = None
        if other == "competitor":
            s = (v if hi else -v)[sub]
            good = np.isfinite(s)
            nd = perm_null(s[good], ypos[good], gsub[good])
            pe = (np.sum(np.abs(nd - nd.mean()) >= abs(r["auroc"] - nd.mean())) + 1) / (NPERM + 1)
            nul = (nd.mean(), nd.std(ddof=1), pe)
        print(fmt(r, nul))

# the whole 40-bin pattern, gene-grouped cv
def fit_ridge(X, y, lam=1.0, iters=60):
    b = np.zeros(X.shape[1]); P = np.eye(X.shape[1]) * lam; P[0, 0] = 0
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30); mu = 1 / (1 + np.exp(-eta))
        w = np.maximum(mu * (1 - mu), 1e-9); z = eta + (y - mu) / w
        XtW = X.T * w
        nb = np.linalg.solve(XtW @ X + P, XtW @ z + P @ b)
        if np.max(np.abs(nb - b)) < 1e-9: b = nb; break
        b = nb
    return b

def cv_profile(m_pos, m_neg, label, extra=None):
    sub = m_pos | m_neg
    X0 = PROF[sub]
    if extra is not None: X0 = np.column_stack([X0, extra[sub]])
    y = (cls[sub] == "real").astype(float); g = gene[sub]
    gg = np.unique(g); fold = {k: i % 5 for i, k in enumerate(rng.permutation(gg))}
    f = np.array([fold[x] for x in g])
    sc = np.full(len(y), np.nan)
    for k in range(5):
        tr, te = f != k, f == k
        mu = X0[tr].mean(0); sd = X0[tr].std(0); sd[sd == 0] = 1
        A = np.column_stack([np.ones(tr.sum()), (X0[tr] - mu) / sd])
        B = np.column_stack([np.ones(te.sum()), (X0[te] - mu) / sd])
        sc[te] = B @ fit_ridge(A, y[tr])
    a = auroc(sc[y == 1], sc[y == 0]); ap, _ = auprc(sc[y == 1], sc[y == 0])
    base = y.mean()
    print(f"  {label:<34s} AUROC {a:.4f}   AUPRC {ap:.4f} (baseline {base:.4f}, lift {ap/base:.2f}x)")
    return sc, y, g

print("\n=============== whole 40-bin profile, gene-grouped 5-fold CV ===============")
cv_profile(cls == "real", cls == "competitor", "profile alone (40 bins), vs comp")
cv_profile(cls == "real", cls == "competitor", "profile + acceptor, vs comp",
           extra=F[:, [L.K["me3_max_up"]]])
cv_profile(cls == "real", cls == "competitor", "acceptor alone, vs comp",
           extra=None) if False else None
sub = (cls == "real") | (cls == "competitor")
acc = F[:, L.K["me3_max_up"]]
print(f"  {'acceptor alone (reference)':<34s} AUROC "
      f"{auroc(acc[cls=='real'], acc[cls=='competitor']):.4f}   AUPRC "
      f"{auprc(acc[cls=='real'], acc[cls=='competitor'])[0]:.4f} "
      f"(baseline {(cls=='real').sum()/sub.sum():.4f})")
cv_profile(cls == "real", cls == "decoy", "profile alone (40 bins), vs decoy")

# within-gene equivalent
P = L.gene_pairs(gene, cls)
print("\n=============== within-gene (confound-free) equivalent ===============")
for nm, v, hi in SCALARS:
    wr = L.winrate(v, P)
    wg = wr[0] if not hi else 1 - wr[0]
    print(f"  {nm:<34s} P(comp>real) {wr[0]:.4f} [{wr[1]:.4f},{wr[2]:.4f}]  "
          f"-> oriented within-gene AUROC {wg:.4f}")
