#!/usr/bin/env python3
# the substitution null for gccgacggaggaccgaccgattc, across the dose series
import sys, argparse, itertools
import numpy as np, torch
def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser(); P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--maxn", type=int, default=7); P.add_argument("--nshuf", type=int, default=50)
P.add_argument("--pos", type=int, default=-230); P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
NT = "ACGT"; UP = 2000; UNIT = "GCCGACGGAGGACCGACCGATTC"
dev = "cuda" if torch.cuda.is_available() else "cpu"
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
def deltas(seqs, chunk=32):
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
SHIFT = [{a: NT[(NT.index(a)+k) % 4] for a in NT} for k in (1, 2, 3)]
DERANGE = [dict(zip(NT, p)) for p in itertools.permutations(NT) if all(a != b for a, b in zip(NT, p))]
def amap(s, m): return "".join(m[c] for c in s)
log(f"[bg] {len(B):,} real 3' end windows; fixed position {A.pos}; unit {UNIT}")
rows = []
for n in range(1, A.maxn+1):
    ins = UNIT*n
    raw = float(deltas([ins])[0])
    sh = deltas([amap(ins, m) for m in SHIFT])
    dg = deltas([amap(ins, m) for m in DERANGE])
    r = np.random.default_rng(A.seed+n)
    sf = deltas(["".join(r.permutation(list(ins))) for _ in range(A.nshuf)])
    rows.append((n, len(ins), raw, float(sh.mean()), float(dg.mean()), float(dg.std(ddof=1)),
                 float(dg.max()), float(sf.mean()), raw-float(sh.mean()), raw-float(dg.mean()),
                 raw-float(sf.mean()), (raw-float(dg.mean()))/float(dg.std(ddof=1))))
    log(f"  n={n} len={len(ins):3d}  raw {raw:+8.4f}  3 shifts {sh.mean():+8.4f}  "
        f"9 derang {dg.mean():+8.4f} (max {dg.max():+7.4f})  {A.nshuf} shuffles {sf.mean():+8.4f}")
with open("free_dose_derange.tsv", "w") as f:
    f.write("n_copies\tinsert_len\traw\tshift_mean\tderange_mean\tderange_sd\tderange_max\t"
            "shuffle_mean\tminus_shift\tminus_derange\tminus_shuffle\tz_vs_derange\tn_bg\n")
    for r_ in rows: f.write("\t".join(str(x) for x in r_)+f"\t{len(B)}\n")
log(f"\nDOSE TABLE   {UNIT}   real 3' end background, {len(B):,} windows, fixed position {A.pos}")
log(f"  {'n':>2} {'nt':>4} {'raw':>9} {'- 3 shifts':>12} {'- 9 derang':>12} {'- shuffle':>11} {'z':>8}")
for r_ in rows:
    log(f"  {r_[0]:>2} {r_[1]:>4} {r_[2]:>+9.4f} {r_[8]:>+12.4f} {r_[9]:>+12.4f} {r_[10]:>+11.4f}"
        f" {r_[11]:>+8.1f}")
log("[out] free_dose_derange.tsv")
