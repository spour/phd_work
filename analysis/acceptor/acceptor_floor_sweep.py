#!/usr/bin/env python3
# does the model still want it once it has to look like a real acceptor?
import sys, argparse
import numpy as np, torch
def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser(); P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--nbg", type=int, default=256); P.add_argument("--restarts", type=int, default=3)
P.add_argument("--iters", type=int, default=40); P.add_argument("--pos", type=int, default=-230)
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
NT = "ACGT"; L = 23; AGI = 18; UP = 2000; TRACT = 18
dev = "cuda" if torch.cuda.is_available() else "cpu"
from maxentpy import maxent
from maxentpy.maxent import load_matrix3
M3 = load_matrix3()
def s3(s):
    try: return float(maxent.score3(s, matrix=M3))
    except Exception: return -99.0
def pyr(s): return (s[:TRACT].count("C")+s[:TRACT].count("T"))/TRACT
def gfrac(s): return s[:TRACT].count("G")/TRACT
ARMS = {"floor10": lambda s: s3(s) >= 10, "floor12": lambda s: s3(s) >= 12,
        "floor14": lambda s: s3(s) >= 14, "floor16": lambda s: s3(s) >= 16,
        "pyr80":   lambda s: s3(s) >= 10 and pyr(s) >= 0.80,
        "pyr80_noG": lambda s: s3(s) >= 10 and pyr(s) >= 0.80 and "G" not in s[:TRACT]}
from cnn_model_def import Net
ck = torch.load(A.model, map_location=dev, weights_only=False)
net = Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]
te = np.flatnonzero((D["split"][ok] == "test") & (X.max(1) <= 3))
BA = X[te[y[te] == 1]]
rng = np.random.default_rng(A.seed)
B = BA[rng.choice(len(BA), min(A.nbg, len(BA)), replace=False)]
TBL = np.zeros((256, 4), np.float32); TBL[255] = .25
for i in range(4): TBL[i, i] = 1.
TBLt = torch.from_numpy(TBL).to(dev)
def oh(a): return TBLt[torch.from_numpy(np.ascontiguousarray(a).astype(np.int64)).to(dev)].permute(0,2,1).contiguous()
def fwd(a, bs=8192):
    o = np.empty(len(a), np.float64)
    with torch.inference_mode():
        for k in range(0, len(a), bs): o[k:k+bs] = net(oh(a[k:k+bs])).float().cpu().numpy()
    return o
base = fwd(B)
def tok(s): return np.array([NT.index(c) for c in s], np.uint8)
def deltas(seqs, chunk=64):
    out = np.empty(len(seqs))
    for i in range(0, len(seqs), chunk):
        cs = seqs[i:i+chunk]
        q = np.empty((len(cs)*len(B), B.shape[1]), B.dtype)
        for j, s in enumerate(cs):
            m = tok(s); s0 = A.pos+UP-len(m)//2
            blk = B.copy(); blk[:, s0:s0+len(m)] = m
            q[j*len(B):(j+1)*len(B)] = blk
        out[i:i+chunk] = (fwd(q).reshape(len(cs), len(B))-base[None, :]).mean(1)
    return out
def seed_for(ok_fn, r):
    for _ in range(200000):
        s = "".join(r.choice(list(NT), L)); s = s[:AGI]+"AG"+s[AGI+2:]
        if ok_fn(s): return s
    for cand in ("TTTCCTCTCTCTCCCCGCAGGTG", "TTTTTTTTTGTGTGTTTTAGGTT"):
        if ok_fn(cand): return cand
    return None
rows = []
for arm, ok_fn in ARMS.items():
    best = None
    for rs in range(A.restarts):
        r = np.random.default_rng(A.seed+977*rs+len(arm))
        s = seed_for(ok_fn, r)
        if s is None: log(f"[{arm}] no feasible seed found"); break
        v = float(deltas([s])[0])
        for it in range(A.iters):
            cand = [c for c in (s[:i]+b+s[i+1:] for i in range(L) if i not in (AGI, AGI+1)
                                for b in NT if b != s[i]) if ok_fn(c)]
            if not cand: break
            vals = deltas(cand); j = int(np.argmax(vals))
            if vals[j] <= v+1e-6: break
            s, v = cand[j], float(vals[j])
        log(f"[{arm}] restart {rs}: {s}  delta {v:+.3f}  score3 {s3(s):+.2f}  "
            f"pyr {pyr(s):.2f}  G {gfrac(s):.2f}")
        if best is None or v > best[1]: best = (s, v)
    if best is None: continue
    s, v = best
    rows.append((arm, s, v, s3(s), pyr(s), gfrac(s)))
    log(f"[{arm}] WINNER {s}  delta {v:+.3f}  score3 {s3(s):+.2f}  pyr {pyr(s):.2f}  G {gfrac(s):.2f}")
with open("acceptor_floor_sweep.tsv", "w") as f:
    f.write("arm\tsequence\tdelta_logit\tscore3\ttract_pyrimidine\ttract_G\tn_bg\n")
    for r in rows: f.write("\t".join(str(x) for x in r)+f"\t{len(B)}\n")
log(f"\n{'arm':<12}{'sequence':<26}{'delta':>9}{'score3':>9}{'pyr':>7}{'G':>7}")
for r in rows: log(f"{r[0]:<12}{r[1]:<26}{r[2]:>+9.3f}{r[3]:>+9.2f}{r[4]:>7.2f}{r[5]:>7.2f}")
log("[out] acceptor_floor_sweep.tsv")
