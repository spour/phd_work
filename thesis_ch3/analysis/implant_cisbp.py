#!/usr/bin/env python3
# response of the network to every rna-binding protein motif of cisbp-rna
import sys, os, time, collections, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, pysam
def log(*a): print(*a, file=sys.stderr, flush=True)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
dev = "cuda" if torch.cuda.is_available() else "cpu"; log("device", dev)
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); X = D["X"]; y = D["y"].astype(int); s = D["split"]
rng = np.random.default_rng(0)
ci_ = np.flatnonzero((s == "test") & (y == 0)); ci_ = np.sort(rng.choice(ci_, 3000, replace=False)); ri_ = np.flatnonzero((s == "test") & (y == 1))
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
        e = list(r.permutation(e)); 
        if a in le: e.append(le[a])
        pools[a] = e[::-1]
    c = v[0]
    while pools.get(c): c = pools[c].pop(); out.append(c)
    assert len(out) == len(v); return np.array(out, np.uint8)
BG = {"competitor windows": X[ci_].copy()}
for k, v in BG.items(): log(k, v.shape)
TBL = torch.zeros(256, 4); TBL[255] = 0.25
for k in range(4): TBL[k, k] = 1.0
TBL = TBL.to(dev)
class V(nn.Module):
    def __init__(s):
        super().__init__()
        s.c1 = nn.Conv1d(4, 32, 19, padding=9, bias=False); s.b1 = nn.BatchNorm1d(32)
        s.c2 = nn.Conv1d(32, 64, 7, padding=3, bias=False); s.b2 = nn.BatchNorm1d(64)
        s.c3 = nn.Conv1d(64, 96, 5, padding=2, bias=False); s.b3 = nn.BatchNorm1d(96)
        s.do = nn.Dropout(0.4); s.fc1 = nn.Linear(96 * 25, 64); s.fc2 = nn.Linear(64, 1)
    def forward(s, x):
        h = F.max_pool1d(torch.exp(torch.clamp(s.b1(s.c1(x)), max=8.0)), 10)
        h = F.max_pool1d(F.relu(s.b2(s.c2(h))), 5)
        h = F.max_pool1d(F.relu(s.b3(s.c3(h))), 2)
        return s.fc2(s.do(F.relu(s.fc1(s.do(h.flatten(1)))))).squeeze(1)
nets = []
for seed in (42, 43, 44, 45, 46):
    n = V().to(dev); n.load_state_dict(torch.load(os.path.join(PUB, f"acc_arch_base_s{seed}.pt"), map_location=dev)["state"]); n.eval(); nets.append(n)
@torch.no_grad()
def score(Xa, bs=2048):
    out = np.zeros(len(Xa), np.float32)
    for k in range(0, len(Xa), bs):
        xb = TBL[torch.from_numpy(Xa[k:k+bs].astype(np.int64)).to(dev)].permute(0, 2, 1).contiguous()
        out[k:k+bs] = np.mean([n(xb).float().cpu().numpy() for n in nets], 0)
    return out
CODE = {c: i for i, c in enumerate("ACGT")}; enc = lambda q: np.array([CODE[c] for c in q], np.uint8)
MOT = []
with open(os.path.join(PUB, "../cisbp_rna.meme")) as fh:
    name = None; rows = []
    for line in fh:
        if line.startswith("MOTIF"):
            if name and rows: MOT.append((name, np.array(rows)))
            name = line.split()[1]; rows = []
        elif name and line.strip() and line.strip()[0].isdigit():
            v = [float(x) for x in line.split()]
            if len(v) == 4: rows.append(v)
    if name and rows: MOT.append((name, np.array(rows)))
def trim(P):
    P = P / P.sum(1, keepdims=True); ic = 2 + (P * np.log2(np.clip(P, 1e-9, 1))).sum(1); k = np.flatnonzero(ic >= 0.25)
    return (P[k[0]:k[-1] + 1], ic[k[0]:k[-1] + 1].sum()) if len(k) else (P[:0], 0.0)
OFFS = [-1000, -300, -200, -100, 50]
out = open(os.path.join(ROOT, "data/implant_cisbp.tsv"), "w")
out.write("motif\tprotein\twidth\tconsensus\tinformation_bits\toffset\tdiff\tdiff_lo\tdiff_hi\n"); t0 = time.time()
def stat(d): b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(500)]; return d.mean(), *np.percentile(b, [2.5, 97.5])
Xb = BG["competitor windows"]; base = score(Xb); n = len(Xb)
for mi, (name, P0) in enumerate(MOT):
    P, ic = trim(P0); L = len(P)
    if L < 4: continue
    cdf = np.cumsum(P, 1); u = rng.random((n, L)); ins = (u[:, :, None] > cdf[None, :, :]).sum(2).clip(0, 3).astype(np.uint8)
    ctl = [np.stack([rng.permutation(q) for q in ins]) for _ in range(2)]
    cons = "".join("ACGU"[j] for j in P.argmax(1)); prot = name.split("_")[-1]
    for off in OFFS:
        a = off + 2000; A = Xb.copy(); A[:, a:a+L] = ins; dc = score(A) - base; dk = np.zeros(n, np.float32)
        for cset in ctl:
            B = Xb.copy(); B[:, a:a+L] = cset; dk += (score(B) - base) / 2
        m, lo, hi = stat(dc - dk); out.write(f"{name}\t{prot}\t{L}\t{cons}\t{ic:.2f}\t{off}\t{m:.4f}\t{lo:.4f}\t{hi:.4f}\n")
    out.flush()
    if mi % 50 == 0: log(f"  {mi}/{len(MOT)}  {time.time()-t0:.0f}s")
out.close(); log("[out] data/implant_cisbp.tsv")
