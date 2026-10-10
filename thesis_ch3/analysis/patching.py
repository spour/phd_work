#!/usr/bin/env python3
# where in the window does the network find the difference between a real 3' end
import sys, os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
def log(*a): print(*a, file=sys.stderr, flush=True)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
dev = "cuda" if torch.cuda.is_available() else "cpu"; log("device", dev)
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True)
X = D["X"]; y = D["y"].astype(int); g = D["gene"]; s = D["split"]
ti = np.flatnonzero(s == "test")
real_of = {g[i]: i for i in ti if y[i] == 1}
comp = np.array([i for i in ti if y[i] == 0]); rr = np.array([real_of[g[i]] for i in comp])
log(f"pairs {len(comp):,} in {len(set(g[comp])):,} genes")
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
    out = np.zeros((len(nets), len(Xa)), np.float32)
    for k in range(0, len(Xa), bs):
        xb = TBL[torch.from_numpy(Xa[k:k+bs].astype(np.int64)).to(dev)].permute(0, 2, 1).contiguous()
        for j, n in enumerate(nets): out[j, k:k+bs] = n(xb).float().cpu().numpy()
    return out
t0 = time.time()
Lc = score(X[comp]); Lr = score(X[rr])
BINS = [(100 * b, 100 * (b + 1)) for b in range(25)]
REG = {"upstream": (0, 1750), "core": (1750, 2250), "downstream": (2250, 2500)}
gain = np.zeros((len(comp), 25), np.float32); loss = np.zeros((len(comp), 25), np.float32)
rg = {k: np.zeros(len(comp), np.float32) for k in REG}; rl = {k: np.zeros(len(comp), np.float32) for k in REG}
for b, (lo, hi) in enumerate(BINS):
    A = X[comp].copy(); A[:, lo:hi] = X[rr][:, lo:hi]; gain[:, b] = (score(A) - Lc).mean(0)
    B = X[rr].copy(); B[:, lo:hi] = X[comp][:, lo:hi]; loss[:, b] = (Lr - score(B)).mean(0)
    log(f"  bin {b:2d} [{lo-2000:+5d},{hi-2000:+5d})  gain {gain[:,b].mean():+.3f}  loss {loss[:,b].mean():+.3f}  {time.time()-t0:.0f}s")
for k, (lo, hi) in REG.items():
    A = X[comp].copy(); A[:, lo:hi] = X[rr][:, lo:hi]; rg[k] = (score(A) - Lc).mean(0)
    B = X[rr].copy(); B[:, lo:hi] = X[comp][:, lo:hi]; rl[k] = (Lr - score(B)).mean(0)
    log(f"  region {k:10s} gain {rg[k].mean():+.3f}  loss {rl[k].mean():+.3f}")
gap = (Lr - Lc).mean(0)
log(f"  mean logit gap real - competitor {gap.mean():+.3f}")
np.savez_compressed(os.path.join(ROOT, "data/patching.npz"), comp=comp, real=rr, gene=g[comp], gap=gap,
                    logit_comp=Lc.mean(0), logit_real=Lr.mean(0), gain=gain, loss=loss,
                    **{f"gain_{k}": v for k, v in rg.items()}, **{f"loss_{k}": v for k, v in rl.items()})
arms = ["acc", "don", "both", "ctrl_acc", "ctrl_don", "ctrl_both"]; out = {"test_idx": ti, "intact": score(X[ti])}
for tag in ("scramble", "resample"):
    W = np.load(os.path.join(ROOT, f"data/splice_{tag}_windows.npz")); assert (W["test_idx"] == ti).all()
    for a in arms: out[f"{tag}_{a}"] = score(W[a])
np.savez_compressed(os.path.join(ROOT, "data/scramble_logits.npz"), **out)
log(f"[out] data/patching.npz data/scramble_logits.npz  {time.time()-t0:.0f}s")
