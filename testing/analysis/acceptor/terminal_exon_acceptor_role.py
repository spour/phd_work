#!/usr/bin/env python3
# validate the role of the 3' splice site in defining the terminal exon, from the data
import sys, csv, math, argparse
import numpy as np
from multiprocessing import Pool

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--procs", type=int, default=6)
A = P.parse_args()
UP, WIN = 2000, 2500
from cnn_interp_lib import NT

D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]; gene = D["gene"][ok]
chrom = D["chrom"][ok]; cen = D["center"][ok]
log(f"[data] {len(X):,} analysis-set peaks, {int((y==1).sum()):,} real 3' ends, "
    f"{len(set(gene)):,} genes")

_M3 = None
def _init():
    global _M3
    from maxentpy.maxent import load_matrix3
    _M3 = load_matrix3()
def best_acceptor(row):
    from maxentpy import maxent
    w = "".join(NT[b] if b < 4 else "N" for b in row)
    best = (-99.0, -1)
    for p in range(0, UP - 22):
        if w[p+18] == "A" and w[p+19] == "G":
            try: v = float(maxent.score3(w[p:p+23], matrix=_M3))
            except Exception: continue
            if v > best[0]: best = (v, p)
    return best[0], (UP - (best[1] + 18)) if best[1] >= 0 else -1
with Pool(A.procs, initializer=_init) as pool:
    res = pool.map(best_acceptor, list(X), chunksize=64)
ACC = np.array([r[0] for r in res]); DIST = np.array([r[1] for r in res], float)
DIST[DIST < 0] = np.nan
log(f"[acceptor] recomputed from the model's own windows: median score3 {np.median(ACC):.2f}, "
    f"median distance upstream {np.nanmedian(DIST):,.0f} nt")

Umeta = {(r["gene"], r["chrom"], int(r["center"])): r
         for r in csv.DictReader(open("fig3_ucube_meta.tsv"), delimiter="\t")}
def g(k, c):
    r = Umeta.get(k)
    if r is None: return np.nan
    try:
        v = float(r[c]); return v if np.isfinite(v) else np.nan
    except Exception: return np.nan
K = [(a, b, int(v)) for a, b, v in zip(gene, chrom, cen)]
NDN = np.array([g(k, "nearest_dn") for k in K])
U1U = np.array([g(k, "u1_dw_up") for k in K])
PAS = np.array([g(k, "pas") for k in K])
GC = np.load("analysis_columns.npz", allow_pickle=True)["gc_contrast"]
Y = (y == 1).astype(float)
TERM = ~np.isfinite(NDN)
INTL = np.isfinite(NDN) & (NDN <= 300)
log(f"[architecture] terminal-like (no downstream donor) {TERM.sum():,} "
    f"({100*TERM.mean():.0f}%); internal-like (donor within 300 nt) {INTL.sum():,} "
    f"({100*INTL.mean():.0f}%)")

def auroc(s, yy):
    m = np.isfinite(s); s, yy = s[m], yy[m]
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s), float); r[o] = np.arange(1, len(s)+1)
    ss = s[o]; i = 0
    while i < len(ss):
        j = i
        while j+1 < len(ss) and ss[j+1] == ss[i]: j += 1
        if j > i: r[o[i:j+1]] = (i+1+j+1)/2
        i = j+1
    n1 = int(yy.sum()); n0 = len(yy)-n1
    return (r[yy == 1].sum()-n1*(n1+1)/2)/(n1*n0) if n1 and n0 else np.nan
def wingene(s, yy, gg):
    m = np.isfinite(s); s, yy, gg = s[m], yy[m], gg[m]; by = {}
    for k in range(len(s)):
        r, c = by.setdefault(gg[k], ([], []))
        (r if yy[k] == 1 else c).append(s[k])
    v = []
    for r, c in by.values():
        if not r or not c: continue
        r = np.array(r); c = np.array(c)
        v.append(((r[:,None] > c[None,:]).sum()+0.5*(r[:,None] == c[None,:]).sum())/(len(r)*len(c)))
    return float(np.mean(v)), len(v)
def irls(Xd, yy, it=80):
    Xd = np.column_stack([np.ones(len(Xd)), Xd]); b = np.zeros(Xd.shape[1])
    for _ in range(it):
        e = Xd@b; p = 1/(1+np.exp(-np.clip(e,-30,30))); w = np.clip(p*(1-p),1e-9,None)
        z = e+(yy-p)/w
        nb = np.linalg.solve(Xd.T@(Xd*w[:,None])+1e-9*np.eye(Xd.shape[1]), Xd.T@(w*z))
        if np.max(np.abs(nb-b)) < 1e-9: b = nb; break
        b = nb
    p = 1/(1+np.exp(-np.clip(Xd@b,-30,30))); w = np.clip(p*(1-p),1e-9,None)
    return b, np.sqrt(np.diag(np.linalg.inv(Xd.T@(Xd*w[:,None])+1e-9*np.eye(Xd.shape[1]))))
def z(v):
    m = np.isfinite(v); o = np.zeros_like(v); o[m] = (v[m]-v[m].mean())/v[m].std(); return o

rows = []
log(f"\n1  DOES THE ACCEPTOR MARK THE REAL 3' END AT ALL?")
w, ng = wingene(ACC, Y, gene)
log(f"   acceptor alone: pooled AUROC {auroc(ACC, Y):.4f}   within-gene {w:.4f} ({ng:,} genes)")
log(f"   median score3: real 3' ends {np.median(ACC[Y==1]):.2f}, "
    f"competitors {np.median(ACC[Y==0]):.2f}")
rows.append(("overall", "acceptor", auroc(ACC, Y), w, ng, np.nan, np.nan))

log(f"\n2  DOES IT WORK AT TERMINAL-EXON DISTANCES?")
log(f"   {'distance upstream':>20} {'n':>7} {'AUROC':>8} {'within-gene':>12} {'%real':>7}")
qs = [0, 200, 500, 1000, 1500, 2000]
for i in range(len(qs)-1):
    m = np.isfinite(DIST) & (DIST >= qs[i]) & (DIST < qs[i+1])
    if m.sum() < 200: continue
    wg2, n2 = wingene(ACC[m], Y[m], gene[m])
    log(f"   {f'{qs[i]}-{qs[i+1]} nt':>20} {m.sum():>7,} {auroc(ACC[m], Y[m]):>8.4f} "
        f"{wg2:>12.4f} {100*Y[m].mean():>6.1f}%")
    rows.append((f"dist_{qs[i]}_{qs[i+1]}", "acceptor", auroc(ACC[m], Y[m]), wg2, n2,
                 int(m.sum()), float(Y[m].mean())))

log(f"\n3  IS IT SPECIFIC TO THE TERMINAL-EXON CONFIGURATION?")
log(f"   {'architecture':>34} {'n':>7} {'AUROC':>8} {'within-gene':>12}")
for nm, m in (("terminal-like: no downstream donor", TERM),
              ("internal-like: donor within 300 nt", INTL)):
    wg2, n2 = wingene(ACC[m], Y[m], gene[m])
    log(f"   {nm:>34} {m.sum():>7,} {auroc(ACC[m], Y[m]):>8.4f} {wg2:>12.4f}")
    rows.append((nm, "acceptor", auroc(ACC[m], Y[m]), wg2, n2, int(m.sum()), float(Y[m].mean())))

log(f"\n4  THE INTERACTION, AND THE ACCEPTOR WITH EVERYTHING ELSE HELD")
za, zt = z(ACC), TERM.astype(float)
b, se = irls(np.column_stack([za, zt, za*zt]), Y)
log(f"   y ~ acceptor + terminal + acceptor:terminal")
for i, n2 in enumerate(("acceptor", "terminal-like", "acceptor x terminal")):
    log(f"     {n2:<22} {b[i+1]:+.4f} +/- {se[i+1]:.4f}   z = {b[i+1]/se[i+1]:+.1f}")
    rows.append(("interaction", n2, b[i+1], se[i+1], b[i+1]/se[i+1], np.nan, np.nan))
cov = np.column_stack([za, z(GC), z(np.nan_to_num(PAS, nan=np.nanmedian(PAS))),
                       z(np.nan_to_num(U1U, nan=np.nanmedian(U1U)))])
b2, se2 = irls(cov, Y)
log(f"   y ~ acceptor + GC contrast + PAS + upstream U1")
for i, n2 in enumerate(("acceptor", "GC contrast", "PAS", "upstream U1")):
    log(f"     {n2:<22} {b2[i+1]:+.4f} +/- {se2[i+1]:.4f}   z = {b2[i+1]/se2[i+1]:+.1f}")
    rows.append(("adjusted", n2, b2[i+1], se2[i+1], b2[i+1]/se2[i+1], np.nan, np.nan))
with open("terminal_exon_acceptor_role.tsv", "w") as fh:
    fh.write("scope\titem\tv1\tv2\tv3\tn\tfrac_real\n")
    for r in rows: fh.write("\t".join(str(x) for x in r) + "\n")
with open("terminal_exon_acceptor_bins.tsv", "w") as fh:
    fh.write("acceptor\tdistance\tis_real\tterminal_like\tgene\n")
    for i in range(len(ACC)):
        fh.write(f"{ACC[i]:.4f}\t{'' if not np.isfinite(DIST[i]) else int(DIST[i])}\t"
                 f"{int(Y[i])}\t{int(TERM[i])}\t{gene[i]}\n")
log("\n[out] terminal_exon_acceptor_role.tsv terminal_exon_acceptor_bins.tsv")
