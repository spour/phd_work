#!/usr/bin/env python3
# positional splice features for every analysis-set candidate
import os, sys, numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
os.chdir(PUB); sys.path.insert(0, PUB); sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as L
from maxentpy import maxent
from maxentpy.maxent import load_matrix3
U1 = np.asarray(L.U1, float); ME5 = np.asarray(L.ME5, float); POW = np.array([int(L.POW[p]) for p in range(9)], np.int64)
D = np.load("cnn_data.npz", allow_pickle=True); X = D["X"][D["ok"]]; N, W = X.shape
A = np.load("analysis_columns.npz", allow_pickle=True)
assert np.array_equal(np.array([str(x) for x in D["gene"][D["ok"]]]), np.array([str(x) for x in A["gene"]]))
B = "ACGT"
def work(rng):
    lo, hi = rng; M3 = load_matrix3(); Xi = X[lo:hi].astype(np.int64); bad = Xi > 3; Xi[bad] = 0; n = hi - lo; K = W - 8
    code = np.zeros((n, K), np.int64)
    for p in range(9): code += Xi[:, p:p + K] * POW[p]
    inv = np.zeros((n, K), bool)
    for p in range(9): inv |= bad[:, p:p + K]
    u = U1[code]; d5 = ME5[code]; u[inv] = np.nan; d5[inv] = np.nan
    gt = (Xi[:, 3:3 + K] == 2) & (Xi[:, 4:4 + K] == 3) & ~inv
    b100 = np.arange(K) // 100; b100 = np.minimum(b100, 24); b50 = np.minimum(np.arange(K) // 50, 49)
    out = {k: np.zeros((n, 25), np.float32) for k in ("u1_4", "u1_7", "u1_mean", "d5_8", "d5_max", "a3_max", "a3_65")}
    out["u1_4_50"] = np.zeros((n, 50), np.float32); out["a3_max"][:] = -20; out["d5_max"][:] = -20
    for j in range(25):
        m = b100 == j
        out["u1_4"][:, j] = np.nansum(u[:, m] >= 4, 1) * 10; out["u1_7"][:, j] = np.nansum(u[:, m] >= 7, 1) * 10
        out["u1_mean"][:, j] = np.nanmean(u[:, m], 1)
        dd = np.where(gt[:, m], d5[:, m], -20.0); out["d5_8"][:, j] = (dd >= 8).sum(1) * 10; out["d5_max"][:, j] = dd.max(1)
    for j in range(50): out["u1_4_50"][:, j] = np.nansum(u[:, b50 == j] >= 4, 1) * 20
    for i in range(n):
        s = "".join(B[c] if c < 4 else "N" for c in X[lo + i])
        for p in range(0, W - 23):
            if s[p + 18] == "A" and s[p + 19] == "G":
                q = s[p:p + 23]
                if "N" in q: continue
                v = maxent.score3(q, matrix=M3); j = min(p // 100, 24)
                if v > out["a3_max"][i, j]: out["a3_max"][i, j] = v
                if v >= 6.5: out["a3_65"][i, j] += 10
    return lo, out
if __name__ == "__main__":
    chunks = [(a, min(a + 500, N)) for a in range(0, N, 500)]
    with Pool(6) as pool: res = pool.map(work, chunks)
    res.sort(key=lambda r: r[0]); F = {k: np.concatenate([r[1][k] for r in res]) for k in res[0][1]}
    F["acc"] = A["spl_upstream_max_me3"].astype(np.float32)
    np.savez_compressed(os.path.join(ROOT, "data/splice_positional.npz"), **F)
    print({k: v.shape for k, v in F.items()}); print("[out] data/splice_positional.npz")
