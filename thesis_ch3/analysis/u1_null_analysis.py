#!/usr/bin/env python3
# fraction of u1 sites expected from dinucleotide composition, on the analysis set
import os, sys, collections, numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
os.chdir(PUB); sys.path.insert(0, PUB); sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as L
U1 = np.asarray(L.U1, float); POW = np.array([int(L.POW[p]) for p in range(9)], np.int64)
D = np.load("cnn_data.npz", allow_pickle=True); X = D["X"][D["ok"]]; N, W = X.shape; THR = np.arange(1, 11)
def dishuffle(v, r):
    v = v.astype(int); last = v[-1]; edges = collections.defaultdict(list)
    for a, b in zip(v[:-1], v[1:]): edges[a].append(b)
    while True:
        le = {a: e[r.integers(len(e))] for a, e in edges.items() if a != last}; okk = True
        for a in le:
            seen = set(); c = a
            while c != last and c in le and c not in seen: seen.add(c); c = le[c]
            if c != last: okk = False; break
        if okk: break
    out = [v[0]]; pools = {}
    for a, e in edges.items():
        e = list(e)
        if a in le: e.remove(le[a])
        e = list(r.permutation(e))
        if a in le: e.append(le[a])
        pools[a] = e[::-1]
    c = v[0]
    while pools.get(c): c = pools[c].pop(); out.append(c)
    return np.array(out, np.int64)
def counts(Xi):
    K = Xi.shape[1] - 8; code = np.zeros((len(Xi), K), np.int64)
    for p in range(9): code += Xi[:, p:p + K] * POW[p]
    u = U1[code]; return np.array([(u >= t).sum() for t in THR]), u.size
def work(rng_):
    lo, hi = rng_; r = np.random.default_rng(lo); Xi = X[lo:hi].astype(np.int64); Xi[Xi > 3] = 0
    a, n = counts(Xi); b, _ = counts(np.stack([dishuffle(x, r) for x in Xi])); return a, b, n
if __name__ == "__main__":
    with Pool(6) as pool: R = pool.map(work, [(a, min(a + 1000, N)) for a in range(0, N, 1000)])
    a = sum(r[0] for r in R); b = sum(r[1] for r in R); n = sum(r[2] for r in R)
    with open(os.path.join(ROOT, "data/u1_null_analysis.tsv"), "w") as fh:
        fh.write("threshold\tfraction_real\tfraction_shuffled\tshuffled_over_real\tpositions\n")
        for t, x, y in zip(THR, a, b): fh.write(f"{t}\t{x/n:.5f}\t{y/n:.5f}\t{y/x:.3f}\t{n}\n"); print(t, round(x / n, 4), round(y / n, 4), round(y / x, 3))
