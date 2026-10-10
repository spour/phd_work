#!/usr/bin/env python3
# test 1 and test 4
import sys, math
import numpy as np

def log(*a): print(*a, file=sys.stderr, flush=True)
import pyarrow.parquet as pq
P = pq.read_table("cpa_peaks_final.parquet").to_pandas()
Z = np.load("cnn_wg_scores.npz", allow_pickle=True)
key = {(g, c, int(x)): i for i, (g, c, x) in enumerate(zip(Z["gene"], Z["chrom"], Z["center"]))}
idx = np.array([key.get((g, c, int(x)), -1) for g, c, x in zip(P["gene"], P["chrom"], P["center"])])
ok = (idx >= 0) & np.isfinite(P["spl_upstream_max_me3"].to_numpy())
L = np.asarray(Z["logit"])[idx[ok]]
A = P["spl_upstream_max_me3"].to_numpy()[ok]
C = P["is_canonical"].to_numpy()[ok].astype(bool)
G = P["gene"].to_numpy()[ok]
log(f"[join] {ok.sum():,} peaks carry both an out-of-fold CNN logit and a real upstream acceptor "
    f"score ({C.sum():,} real 3' ends, {(~C).sum():,} competitors, {len(set(G)):,} genes)")

def pear(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x); r = float(np.corrcoef(x, y)[0, 1])
    if n < 5: return r, np.nan, np.nan
    z = 0.5 * math.log((1 + r) / (1 - r)); se = 1 / math.sqrt(n - 3)
    lo, hi = (math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se))
    return r, lo, hi
def rank(v):
    o = np.argsort(v, kind="mergesort"); r = np.empty(len(v), float); r[o] = np.arange(len(v))
    s = v[o]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j+1] == s[i]: j += 1
        if j > i: r[o[i:j+1]] = (i + j) / 2
        i = j + 1
    return r
def spear(x, y): return pear(rank(np.asarray(x, float)), rank(np.asarray(y, float)))

log("\nTEST 1  does the CNN's score track REAL upstream acceptor strength?")
log(f"  {'scope':<28} {'n':>7} {'Pearson r':>10} {'95% CI':>18} {'Spearman':>9}")
rows = []
for nm, m in (("all peaks", np.ones(len(L), bool)),
              ("real 3' ends only", C), ("competitors only", ~C)):
    r, lo, hi = pear(A[m], L[m]); s, _, _ = spear(A[m], L[m])
    log(f"  {nm:<28} {m.sum():>7,} {r:>+10.4f} {f'[{lo:+.4f}, {hi:+.4f}]':>18} {s:>+9.4f}")
    rows.append((nm, int(m.sum()), r, lo, hi, s))
uq, inv = np.unique(G, return_inverse=True)
cnt = np.bincount(inv)
mA = np.bincount(inv, A) / cnt; mL = np.bincount(inv, L) / cnt
keep = cnt[inv] >= 2
rA = A[keep] - mA[inv[keep]]; rL = L[keep] - mL[inv[keep]]
r, lo, hi = pear(rA, rL); s, _, _ = spear(rA, rL)
log(f"  {'within gene (gene demeaned)':<28} {keep.sum():>7,} {r:>+10.4f} "
    f"{f'[{lo:+.4f}, {hi:+.4f}]':>18} {s:>+9.4f}")
rows.append(("within gene", int(keep.sum()), r, lo, hi, s))
kc = keep & C[keep] if False else None
for nm, cls in (("within gene, real ends", True), ("within gene, competitors", False)):
    m = keep & (C == cls)
    if m.sum() < 50: continue
    g2 = inv[m]; u2, i2 = np.unique(g2, return_inverse=True); c2 = np.bincount(i2)
    k2 = c2[i2] >= 2
    a2 = A[m][k2] - (np.bincount(i2, A[m]) / c2)[i2[k2]]
    l2 = L[m][k2] - (np.bincount(i2, L[m]) / c2)[i2[k2]]
    r, lo, hi = pear(a2, l2); s, _, _ = spear(a2, l2)
    log(f"  {nm:<28} {k2.sum():>7,} {r:>+10.4f} {f'[{lo:+.4f}, {hi:+.4f}]':>18} {s:>+9.4f}")
    rows.append((nm, int(k2.sum()), r, lo, hi, s))
with open("acceptor_real_test.tsv", "w") as f:
    f.write("scope\tn\tpearson_r\tci_lo\tci_hi\tspearman\n")
    for r_ in rows: f.write("\t".join(str(x) for x in r_) + "\n")

log("\nTEST 1  mean CNN logit by decile of the real upstream acceptor score")
q = np.quantile(A, np.linspace(0, 1, 11))
log(f"  {'decile':>6} {'acceptor range':>16} {'n':>7} {'all':>9} {'real ends':>11} {'competitors':>12}")
with open("acceptor_real_binned.tsv", "w") as f:
    f.write("decile\tacc_lo\tacc_hi\tn\tmean_logit\tse\tmean_logit_real\tmean_logit_comp\tfrac_real\n")
    for i in range(10):
        m = (A >= q[i]) & (A <= q[i+1]) if i == 9 else (A >= q[i]) & (A < q[i+1])
        if m.sum() < 5: continue
        a = L[m]; ar = L[m & C]; ac = L[m & ~C]
        log(f"  {i+1:>6} {f'{q[i]:.2f}-{q[i+1]:.2f}':>16} {m.sum():>7,} {a.mean():>+9.3f} "
            f"{(ar.mean() if len(ar) else np.nan):>+11.3f} {(ac.mean() if len(ac) else np.nan):>+12.3f}")
        f.write(f"{i+1}\t{q[i]:.4f}\t{q[i+1]:.4f}\t{m.sum()}\t{a.mean():.5f}\t"
                f"{a.std(ddof=1)/math.sqrt(len(a)):.5f}\t"
                f"{(ar.mean() if len(ar) else float('nan')):.5f}\t"
                f"{(ac.mean() if len(ac) else float('nan')):.5f}\t{C[m].mean():.5f}\n")

# test 4
log("\nTEST 4  is the implanted acceptor an acceptor AS THE MODEL SEES IT?")
sys.path.insert(0, "scripts")
from cnn_interp_lib import NT
from maxentpy import maxent
from maxentpy.maxent import load_matrix3
M3 = load_matrix3()
def s3(s):
    try: return float(maxent.score3(s, matrix=M3))
    except Exception: return -99.0
D = np.load("cnn_data.npz", allow_pickle=True); ok2 = D["ok"]
X = D["X"][ok2]; y = D["y"][ok2]
te = np.flatnonzero((D["split"][ok2] == "test") & (X.max(1) <= 3))
UP, WIN = 2000, 2500
ACCSEQ = "TTTCCTCTCTCTCCCCGCAGGTG"
log(f"  insert standing alone: {ACCSEQ}  score3 {s3(ACCSEQ):+.2f}")
m = np.array([NT.index(c) for c in ACCSEQ], np.uint8)
rng = np.random.default_rng(0)
for pos in (-800, -400, -200, 0):
    b = X[te[rng.integers(len(te))]].copy()
    s0 = pos + UP - len(m) // 2
    b[s0:s0+len(m)] = m
    w = "".join(NT[i] for i in b)
    got = w[s0:s0+23]
    allsc = np.array([s3(w[i:i+23]) for i in range(0, WIN - 22)])
    best = int(np.argmax(allsc))
    log(f"  implanted at {pos:>+6}: window[{s0}:{s0+23}] = {got}  "
        f"{'MATCHES' if got == ACCSEQ else 'DOES NOT MATCH'} the insert, score3 {s3(got):+.2f}")
    log(f"  {'':16} best acceptor anywhere in the 2,500 nt window is at {best} "
        f"(insert at {s0}), score3 {allsc[best]:+.2f}"
        f"{'  <- the insert' if best == s0 else '  <- NOT the insert'}")
log("[out] acceptor_real_test.tsv acceptor_real_binned.tsv")
