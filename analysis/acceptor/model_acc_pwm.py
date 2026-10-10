#!/usr/bin/env python3
# the acceptor ensemble by the model's measure, same recipe as me3_pwm.py
import sys, argparse
import numpy as np, torch
def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model.pt"); P.add_argument("--nbg", type=int, default=192)
P.add_argument("--chains", type=int, default=128); P.add_argument("--steps", type=int, default=900)
P.add_argument("--burn", type=int, default=200); P.add_argument("--temp", type=float, default=0.08)
P.add_argument("--auto_temp", action="store_true")
P.add_argument("--thresh", type=float, default=0.6); P.add_argument("--floor", type=float, default=10.0)
P.add_argument("--pos", type=int, default=-230); P.add_argument("--out", default="model_acc_pwm.tsv")
P.add_argument("--pyr", type=float, default=0.0)
P.add_argument("--no_g", action="store_true")
P.add_argument("--seed_seq", default="ACGGTGTACTTTTGTTGCAGGTG")
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
NT = "ACGT"; L = 23; AGI = 18; UP = 2000
SEED_SEQ = A.seed_seq; TRACT = 18
dev = "cuda" if torch.cuda.is_available() else "cpu"
from maxentpy import maxent
from maxentpy.maxent import load_matrix3
M3 = load_matrix3()
def s3(s):
    try: return float(maxent.score3(s, matrix=M3))
    except Exception: return -99.0
from cnn_model_def import Net
ck = torch.load(A.model, map_location=dev, weights_only=False)
net = Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]
te = np.flatnonzero((D["split"][ok] == "test") & (X.max(1) <= 3))
rng = np.random.default_rng(A.seed)
BA = X[te[y[te] == 1]]
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
def pyrf(s): return (s[:TRACT].count("C")+s[:TRACT].count("T"))/TRACT
def feasible(s):
    if s[AGI:AGI+2] != "AG" or s3(s) < A.floor: return False
    if A.pyr > 0 and pyrf(s) < A.pyr: return False
    if A.no_g and "G" in s[:TRACT]: return False
    return True
cur = [SEED_SEQ]*A.chains
curv = deltas(cur)
if A.auto_temp:
    A.temp = max(0.02, float(curv.mean())/23.0); A.thresh = max(0.15, float(curv.mean())/3.0)
    log(f"[auto] seed delta {curv.mean():+.3f} -> temp {A.temp:.3f}, thresh {A.thresh:.3f}")
assert feasible(SEED_SEQ), "seed is not a feasible acceptor"
log(f"[mh] {A.chains} chains x {A.steps} steps, {len(B)} backgrounds, temp {A.temp}, "
    f"score3 floor {A.floor}, AG pinned"
    + (f", tract pyrimidine >= {A.pyr:.2f}" if A.pyr > 0 else "")
    + (", no G in tract" if A.no_g else ""))
kept, keptv = [], []
best = float(curv.max())
for t in range(A.steps):
    prop = []
    for s in cur:
        for _ in range(60):
            i = int(rng.integers(L))
            if i in (AGI, AGI+1): continue
            b = NT[int(rng.integers(4))]
            if b == s[i]: continue
            c = s[:i]+b+s[i+1:]
            if feasible(c): prop.append(c); break
        else: prop.append(s)
    pv = deltas(prop)
    acc = rng.random(A.chains) < np.exp(np.clip((pv-curv)/A.temp, -60, 0))
    for j in range(A.chains):
        if acc[j]: cur[j], curv[j] = prop[j], pv[j]
    best = max(best, float(curv.max()))
    if t >= A.burn:
        kept.extend(cur); keptv.extend(curv.tolist())
    if t % 100 == 0: log(f"  step {t:>4}  best {best:+.3f}  mean {curv.mean():+.3f}  "
                         f"accept {acc.mean():.2f}  kept {len(kept):,}")
kept = np.array(kept); keptv = np.array(keptv)
sel = keptv >= best-A.thresh
U, idx = np.unique(kept[sel], return_index=True)
w = keptv[sel][idx]
w = w-(best-A.thresh)
log(f"[keep] {len(U):,} distinct sequences within {A.thresh} of the best delta {best:+.3f}")
M = np.zeros((L, 4))
for s, wt in zip(U, w):
    for i, c in enumerate(s): M[i, NT.index(c)] += wt
M /= M.sum(1, keepdims=True)
ic = np.array([2.0+np.sum(np.where(r > 0, r*np.log2(np.maximum(r, 1e-12)), 0)) for r in M])
consensus = "".join(NT[j] for j in M.argmax(1))
log(f"[pwm] consensus {consensus}   AG at 19-20: {consensus[18:20]}   mean IC {ic.mean():.2f} bits")
meta = A.out.replace(".tsv", "_meta.tsv")
with open(meta, "w") as f:
    f.write("n_sequences\tthresh\tbest_delta\tfloor_score3\tmin_pyrimidine\tno_G\tconsensus\t"
            "mean_ic\tchains\tsteps\ttemp\tn_bg\n")
    f.write(f"{len(U)}\t{A.thresh:.4f}\t{best:.4f}\t{A.floor}\t{A.pyr}\t{int(A.no_g)}\t"
            f"{consensus}\t{ic.mean():.4f}\t{A.chains}\t{A.steps}\t{A.temp:.5f}\t{len(B)}\n")
log(f"[out] {meta}")
with open(A.out, "w") as f:
    f.write("offset\tA\tC\tG\tT\tic\n")
    for i in range(L):
        f.write(f"{i-AGI}\t"+"\t".join(f"{v:.4f}" for v in M[i])+f"\t{ic[i]:.4f}\n")
log(f"[out] {A.out}  ({len(U):,} sequences)")
