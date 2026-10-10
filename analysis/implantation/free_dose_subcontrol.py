#!/usr/bin/env python3
# dose table for the unconstrained maximum, with a substitution control
import sys, argparse
import numpy as np, torch
def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser(); P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--nshuf", type=int, default=20); P.add_argument("--maxn", type=int, default=7)
P.add_argument("--pos", type=int, default=-230); P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
NT = "ACGT"; UP = 2000
UNIT = "GCCGACGGAGGACCGACCGATTC"
dev = "cuda" if torch.cuda.is_available() else "cpu"
from maxentpy import maxent
from maxentpy.maxent import load_matrix3
M3 = load_matrix3()
from cnn_model_def import Net
ck = torch.load(A.model, map_location=dev, weights_only=False)
net = Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]
te = np.flatnonzero((D["split"][ok] == "test") & (X.max(1) <= 3))
B = X[te[y[te] == 1]]
TBL = np.zeros((256, 4), np.float32); TBL[255] = .25
for i in range(4): TBL[i, i] = 1.
TBLt = torch.from_numpy(TBL).to(dev)
def oh(a): return TBLt[torch.from_numpy(np.ascontiguousarray(a).astype(np.int64)).to(dev)].permute(0,2,1).contiguous()
def fwd(a, bs=4096):
    o = np.empty(len(a), np.float64)
    with torch.inference_mode():
        for k in range(0, len(a), bs): o[k:k+bs] = net(oh(a[k:k+bs])).float().cpu().numpy()
    return o
base = fwd(B)
def tok(s): return np.array([NT.index(c) for c in s], np.uint8)
def deltas(seqs, pos, chunk=32):
    out = np.empty(len(seqs))
    for i in range(0, len(seqs), chunk):
        cs = seqs[i:i+chunk]
        q = np.empty((len(cs)*len(B), B.shape[1]), B.dtype)
        for j, s in enumerate(cs):
            m = tok(s); s0 = pos+UP-len(m)//2
            blk = B.copy(); blk[:, s0:s0+len(m)] = m
            q[j*len(B):(j+1)*len(B)] = blk
        out[i:i+chunk] = (fwd(q).reshape(len(cs), len(B))-base[None, :]).mean(1)
    return out
GRID = list(range(-1900, 500, 40))
log(f"[bg] {len(B):,} real 3' end windows; fixed position {A.pos}; unit {UNIT} "
    f"(score3 {maxent.score3(UNIT, matrix=M3):+.2f})")
rows = []
for n in range(1, A.maxn+1):
    ins = UNIT*n
    raw = float(deltas([ins], A.pos)[0])
    W = B.shape[1]
    grid = [p for p in GRID if 0 <= p+UP-len(ins)//2 and p+UP-len(ins)//2+len(ins) <= W]
    best = float(np.max([deltas([ins], p)[0] for p in grid]))
    su = [(UNIT[:i]+b+UNIT[i+1:])*n for i in range(len(UNIT)) for b in NT if b != UNIT[i]]
    vu = deltas(su, A.pos)
    si = [ins[:i]+b+ins[i+1:] for i in range(len(ins)) for b in NT if b != ins[i]]
    vi = deltas(si, A.pos)
    r = np.random.default_rng(A.seed+n)
    vs = deltas(["".join(r.permutation(list(ins))) for _ in range(A.nshuf)], A.pos)
    rows.append((n, len(ins), raw, best, float(vu.mean()), float(vu.std(ddof=1)/np.sqrt(len(vu))),
                 float(vi.mean()), float(vs.mean()), raw-float(vu.mean()), raw-float(vi.mean()),
                 raw-float(vs.mean()), len(vu), len(vi)))
    log(f"  n={n} len={len(ins):3d} grid={len(grid):2d}  raw {raw:+8.4f}  sub_unit {vu.mean():+8.4f} (n={len(vu)})  "
        f"sub_insert {vi.mean():+8.4f} (n={len(vi)})  shuffle {vs.mean():+8.4f}")
with open("free_dose_subcontrol.tsv", "w") as f:
    f.write("n_copies\tinsert_len\traw\traw_best_of_grid\tsub_unit_mean\tsub_unit_se\t"
            "sub_insert_mean\tshuffle_mean\tminus_sub_unit\tminus_sub_insert\tminus_shuffle\t"
            "n_sub_unit\tn_sub_insert\tn_bg\n")
    for r_ in rows: f.write("\t".join(str(x) for x in r_)+f"\t{len(B)}\n")
log(f"\nDOSE TABLE   {UNIT}   real 3' end background, {len(B):,} windows, fixed position {A.pos}")
log(f"  {'n':>2} {'nt':>4} {'raw':>9} {'- sub_unit':>12} {'- sub_insert':>14} {'- shuffle':>11}"
    f" {'best of grid':>14}")
for r_ in rows:
    log(f"  {r_[0]:>2} {r_[1]:>4} {r_[2]:>+9.4f} {r_[8]:>+12.4f} {r_[9]:>+14.4f} {r_[10]:>+11.4f}"
        f" {r_[3]:>+14.4f}")
log("[out] free_dose_subcontrol.tsv")
